import io
import os
import time
import asyncio
import threading
from typing import Optional, List, Dict
from collections import OrderedDict

import ray
from ray import serve
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from faster_whisper import WhisperModel

# Language code mapping for common language names
LANGUAGE_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "italian": "it",
    "portuguese": "pt",
    "chinese": "zh",
    "japanese": "ja",
    "korean": "ko",
    "russian": "ru",
    "arabic": "ar",
    "hindi": "hi",
}

def normalize_language(language: Optional[str]) -> Optional[str]:
    """Convert language name to ISO 639-1 code if needed."""
    if not language:
        return None
    language_lower = language.lower().strip()
    # If it's already a valid 2-letter code, return as-is
    if len(language_lower) == 2:
        return language_lower
    # Otherwise, try to map it
    return LANGUAGE_MAP.get(language_lower, language_lower)

# ----- FastAPI app that Serve will mount -----
api = FastAPI(title="Whisper Inference", version="1.0")

class UrlJob(BaseModel):
    url: str
    task: Optional[str] = "transcribe"  # or "translate"
    beam_size: Optional[int] = 5
    language: Optional[str] = None
    model: Optional[str] = None  # Optional: specify model to use

class PathJob(BaseModel):
    file_path: str  # Path relative to /data or absolute path
    task: Optional[str] = "transcribe"
    beam_size: Optional[int] = 5
    language: Optional[str] = None
    model: Optional[str] = None  # Optional: specify model to use (e.g., "large-v3", "base", "tiny")

class BatchPathJob(BaseModel):
    files: List[PathJob]  # Array of file paths to transcribe
    max_parallel: Optional[int] = None  # Max parallel processing within batch (None = process all in parallel)

# ----- Ray Serve deployment -----
# FastAPI provides /docs automatically, so we keep it
# The routing issue is Ray Serve configuration, not FastAPI
@serve.deployment(
    num_replicas=int(os.environ.get("NUM_REPLICAS", "1")),
    ray_actor_options={
        # 1 GPU per replica; change to 0 if you want CPU-only
        "num_gpus": float(os.environ.get("NUM_GPUS_PER_REPLICA", "1")),
        # CPU-bound workload (audio decoding, preprocessing) - try 2-4 CPUs per replica
        # More CPUs might help with parallel processing, though Python GIL limits true parallelism
        "num_cpus": int(os.environ.get("NUM_CPUS_PER_REPLICA", "2")),
    },
    # Allow multiple concurrent requests per replica to increase throughput
    # With I/O-bound workloads (file reads), we need high concurrency to overlap:
    # - While request A waits on file I/O, request B can use GPU
    # - While request B waits on file I/O, request C can use GPU
    # - This maximizes GPU utilization despite I/O overhead
    # Default is 30 to allow good I/O/GPU overlap
    max_ongoing_requests=int(os.environ.get("MAX_ONGOING_REQUESTS", "30")),
    # Try autoscaling config even with fixed replicas - might force better routing
    # Setting min=max keeps replicas fixed but enables autoscaler logic
    autoscaling_config={
        "min_replicas": int(os.environ.get("NUM_REPLICAS", "1")),
        "max_replicas": int(os.environ.get("NUM_REPLICAS", "1")),
        "target_ongoing_requests": 5,  # Target 5 requests per replica
    } if os.environ.get("ENABLE_AUTOSCALING", "false").lower() == "true" else None,
)
@serve.ingress(api)
class WhisperService:
    # Class-level tracking (shared across all replicas, but initialized lazily)
    _request_tracking = None
    _request_tracking_lock = None
    
    @classmethod
    def _get_tracking(cls):
        """Lazy initialization of request tracking to avoid serialization issues."""
        if cls._request_tracking is None:
            cls._request_tracking = {}
            cls._request_tracking_lock = threading.Lock()
        return cls._request_tracking, cls._request_tracking_lock
    def __init__(self):
        # Default model name from environment
        self.default_model_name = os.environ.get("WHISPER_MODEL", "large-v3")
        # compute_type: "float16" for NVIDIA GPUs, "int8_float16" for memory savings
        self.compute_type = os.environ.get("COMPUTE_TYPE", "float16")
        # Check if we have GPU access (even with fractional GPUs, num_gpus > 0 means GPU)
        num_gpus_per_replica = float(os.environ.get("NUM_GPUS_PER_REPLICA", "1"))
        
        # For fractional GPUs, we need to check actual CUDA availability
        # Ray with fractional GPUs might not set up CUDA context immediately
        import torch
        if num_gpus_per_replica > 0 and torch.cuda.is_available():
            self.device = "cuda"
            # Ensure CUDA is initialized
            torch.cuda.init()
        else:
            self.device = "cpu"
        
        # Primary model (always loaded) - this is the fast path for the common case
        self.model = self._load_model(self.default_model_name)
        
        # Model cache: LRU cache with configurable max size (only used when switching models)
        # Format: {model_name: WhisperModel}
        self.model_cache: OrderedDict[str, WhisperModel] = OrderedDict()
        self.model_cache_lock = threading.Lock()
        # Max cache size: default 2, can be overridden via env var
        # With many replicas, use 2 to save memory. With fewer replicas, can use 3.
        self.max_cache_size = int(os.environ.get("MODEL_CACHE_SIZE", "2"))  # Keep up to 2 models in memory by default
        
        # Store default model in cache for consistency
        self.model_cache[self.default_model_name] = self.model
        
        # Get replica ID for tracking (Ray actor ID)
        try:
            self.replica_id = ray.get_runtime_context().get_actor_id()
        except:
            self.replica_id = f"unknown_{id(self)}"
        
        # Get which GPU this replica is using
        import torch
        if torch.cuda.is_available():
            self.gpu_id = torch.cuda.current_device()
        else:
            self.gpu_id = None
        
        # Signal that this replica is ready
        # This helps Ray Serve know when replicas are actually ready to handle requests
        print(f"Replica {self.replica_id} initialized and ready (GPU {self.gpu_id})")
    
    def _load_model(self, model_name: str) -> WhisperModel:
        """Load a model, handling errors and fallbacks."""
        try:
            model = WhisperModel(model_name, device=self.device, compute_type=self.compute_type)
            return model
        except (ValueError, RuntimeError) as e:
            error_str = str(e)
            if "float16" in error_str and self.compute_type == "float16":
                # Fall back to float32 if float16 isn't supported
                print(f"Warning: float16 not supported for {model_name}, falling back to float32. Error: {e}")
                model = WhisperModel(model_name, device=self.device, compute_type="float32")
                return model
            else:
                raise
    
    def _get_model(self, model_name: Optional[str] = None) -> WhisperModel:
        """Get a model from cache or load it. Thread-safe LRU cache."""
        if model_name is None:
            # Fast path: return primary model directly (no lock, no lookup)
            return self.model
        
        # If requesting a different model, check cache or load it
        if model_name == self.default_model_name:
            # Requesting default model - return primary model directly
            return self.model
        
        # Need to check cache or load model - acquire lock
        with self.model_cache_lock:
            # Check if model is in cache
            if model_name in self.model_cache:
                model = self.model_cache[model_name]
                # Only update LRU order if cache is full (optimization: don't update on every access)
                if len(self.model_cache) >= self.max_cache_size:
                    self.model_cache.move_to_end(model_name)
                return model
            
            # Model not in cache, need to load it
            print(f"Loading model '{model_name}' (cache has {len(self.model_cache)} models)...")
            model = self._load_model(model_name)
            
            # Add to cache
            self.model_cache[model_name] = model
            self.model_cache.move_to_end(model_name)
            
            # Evict least recently used if cache is full
            if len(self.model_cache) > self.max_cache_size:
                lru_model_name, lru_model = self.model_cache.popitem(last=False)
                # Don't evict the primary model
                if lru_model_name != self.default_model_name:
                    # Explicitly delete the model to free GPU memory
                    del lru_model
                    import gc
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    print(f"Evicted model '{lru_model_name}' from cache to make room for '{model_name}'")
            
            return model

    def _run(self, audio_bytes: bytes, task="transcribe", beam_size=5, language=None, model_name=None):
        # Fast path: if no model specified, use primary model directly (zero overhead)
        if model_name is None:
            model = self.model
        else:
            model = self._get_model(model_name)
        
        # Normalize language code
        language = normalize_language(language)
        
        # Track inference time
        start_time = time.time()
        segments, info = model.transcribe(
            audio_bytes,
            task=task,
            beam_size=beam_size,
            language=language,
            vad_filter=True,  # helps on long audio
        )
        text = "".join(seg.text for seg in segments)
        inference_time = time.time() - start_time
        
        # Convert segments to list of dicts with all Whisper data
        segments_data = []
        for seg in segments:
            segments_data.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "tokens": seg.tokens if hasattr(seg, 'tokens') else None,
                "avg_logprob": seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                "no_speech_prob": seg.no_speech_prob if hasattr(seg, 'no_speech_prob') else None,
                "compression_ratio": seg.compression_ratio if hasattr(seg, 'compression_ratio') else None,
                "seek": seg.seek if hasattr(seg, 'seek') else None,
                "temperature": seg.temperature if hasattr(seg, 'temperature') else None,
            })
        
        return {
            "model": model_name or self.default_model_name,
            "language": info.language,
            "language_probability": info.language_probability if hasattr(info, 'language_probability') and info.language_probability is not None else None,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') and info.duration_after_vad is not None else None,
            "all_language_probs": dict(info.all_language_probs) if hasattr(info, 'all_language_probs') and info.all_language_probs is not None else None,
            "inference_time_seconds": round(inference_time, 3),
            "text": text,
            "segments": segments_data,
        }
    
    def _run_from_path(self, file_path: str, task="transcribe", beam_size=5, language=None, model_name=None):
        """Transcribe from a file path - faster_whisper can handle file paths directly."""
        # Track which replica handled this request
        tracking, lock = self._get_tracking()
        with lock:
            tracking[self.replica_id] = tracking.get(self.replica_id, 0) + 1
        
        # Fast path: if no model specified, use primary model directly (zero overhead)
        if model_name is None:
            model = self.model
        else:
            model = self._get_model(model_name)
        
        # Normalize language code
        language = normalize_language(language)
        
        # Track inference time
        start_time = time.time()
        segments, info = model.transcribe(
            file_path,
            task=task,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments)
        inference_time = time.time() - start_time
        
        # Convert segments to list of dicts with all Whisper data
        segments_data = []
        for seg in segments:
            segments_data.append({
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "tokens": seg.tokens if hasattr(seg, 'tokens') else None,
                "avg_logprob": seg.avg_logprob if hasattr(seg, 'avg_logprob') else None,
                "no_speech_prob": seg.no_speech_prob if hasattr(seg, 'no_speech_prob') else None,
                "compression_ratio": seg.compression_ratio if hasattr(seg, 'compression_ratio') else None,
                "seek": seg.seek if hasattr(seg, 'seek') else None,
                "temperature": seg.temperature if hasattr(seg, 'temperature') else None,
            })
        
        return {
            "model": model_name or self.default_model_name,
            "language": info.language,
            "language_probability": info.language_probability if hasattr(info, 'language_probability') and info.language_probability is not None else None,
            "duration": info.duration,
            "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') and info.duration_after_vad is not None else None,
            "all_language_probs": dict(info.all_language_probs) if hasattr(info, 'all_language_probs') and info.all_language_probs is not None else None,
            "inference_time_seconds": round(inference_time, 3),
            "text": text,
            "segments": segments_data,
        }

    @api.post("/transcribe/file")
    async def transcribe_file(
        self,
        file: UploadFile = File(...),
        task: str = Form("transcribe"),
        beam_size: int = Form(5),
        language: Optional[str] = Form(None),
        model: Optional[str] = Form(None),
    ):
        if file.content_type and not file.content_type.startswith("audio"):
            raise HTTPException(400, "Upload must be an audio file")
        audio_bytes = await file.read()
        return self._run(audio_bytes, task=task, beam_size=beam_size, language=language, model_name=model)

    @api.post("/transcribe/url")
    async def transcribe_url(self, job: UrlJob):
        # Simple fetch (no internet in some deployments—replace with signed URLs on your side if needed)
        import requests
        r = requests.get(job.url, timeout=120)
        if r.status_code != 200:
            raise HTTPException(400, f"Could not fetch audio: {r.status_code}")
        return self._run(r.content, task=job.task, beam_size=job.beam_size, language=job.language, model_name=getattr(job, 'model', None))
    
    @api.post("/transcribe/path")
    async def transcribe_path(self, job: PathJob):
        """Transcribe a WAV file from the mounted /data directory or absolute path."""
        # If path doesn't start with /, assume it's relative to /data
        if not job.file_path.startswith("/"):
            file_path = os.path.join("/data", job.file_path)
        else:
            file_path = job.file_path
        
        # Fast file existence check (only if file doesn't exist, this is fast)
        # We do this to return a proper 404 instead of letting faster_whisper raise a generic error
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_path}")
        
        try:
            return self._run_from_path(file_path, task=job.task, beam_size=job.beam_size, language=job.language, model_name=job.model)
        except FileNotFoundError:
            raise HTTPException(404, f"File not found: {file_path}")
        except Exception as e:
            # Re-raise other exceptions (faster_whisper will raise appropriate errors)
            raise
    
    @api.post("/transcribe/batch")
    async def transcribe_batch(self, job: BatchPathJob):
        """Transcribe multiple files in a single request. Files are processed in parallel."""
        if not job.files:
            raise HTTPException(400, "No files provided in batch")
        
        if len(job.files) > 100:  # Reasonable limit
            raise HTTPException(400, f"Batch size too large: {len(job.files)}. Maximum is 100 files.")
        
        # Process files in parallel using asyncio
        max_parallel = job.max_parallel or len(job.files)
        
        async def process_file(path_job: PathJob):
            """Process a single file in the batch."""
            # Resolve file path
            if not path_job.file_path.startswith("/"):
                file_path = os.path.join("/data", path_job.file_path)
            else:
                file_path = path_job.file_path
            
            if not os.path.exists(file_path):
                return {
                    "file_path": path_job.file_path,
                    "status": "error",
                    "error": f"File not found: {file_path}"
                }
            
            if not os.path.isfile(file_path):
                return {
                    "file_path": path_job.file_path,
                    "status": "error",
                    "error": f"Path is not a file: {file_path}"
                }
            
            try:
                result = self._run_from_path(
                    file_path,
                    task=path_job.task,
                    beam_size=path_job.beam_size,
                    language=path_job.language,
                    model_name=path_job.model
                )
                result["file_path"] = path_job.file_path
                result["status"] = "success"
                return result
            except Exception as e:
                return {
                    "file_path": path_job.file_path,
                    "status": "error",
                    "error": str(e)
                }
        
        # Process files with concurrency limit
        semaphore = asyncio.Semaphore(max_parallel)
        
        async def process_with_semaphore(path_job: PathJob):
            async with semaphore:
                return await process_file(path_job)
        
        # Run all tasks
        tasks = [process_with_semaphore(path_job) for path_job in job.files]
        results = await asyncio.gather(*tasks)
        
        # Calculate batch statistics
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]
        
        total_inference_time = sum(r.get("inference_time_seconds", 0) for r in successful)
        total_audio_duration = sum(r.get("duration", 0) for r in successful)
        
        return {
            "batch_summary": {
                "total_files": len(job.files),
                "successful": len(successful),
                "failed": len(failed),
                "total_inference_time_seconds": round(total_inference_time, 3),
                "total_audio_duration_seconds": round(total_audio_duration, 3),
                "average_real_time_factor": round(total_inference_time / total_audio_duration, 3) if total_audio_duration > 0 else 0,
            },
            "results": results
        }
    
    @api.get("/config")
    async def get_config(self):
        """Get server configuration including Ray Serve deployment settings."""
        import torch
        num_replicas = int(os.environ.get("NUM_REPLICAS", "1"))
        num_gpus_per_replica = float(os.environ.get("NUM_GPUS_PER_REPLICA", "1"))
        compute_type = os.environ.get("COMPUTE_TYPE", "float16")
        
        # Get GPU info if available
        gpu_info = None
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_info = []
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024**3), 2),
                    "major": props.major,
                    "minor": props.minor,
                })
        
        # Get cached models
        with self.model_cache_lock:
            cached_models = list(self.model_cache.keys())
        
        # Get actual running replica count from Ray Serve
        actual_replicas = None
        replica_details = None
        try:
            from ray import serve
            status = serve.status()
            if status and hasattr(status, 'deployments'):
                deployments = status.deployments
                for name, deployment in deployments.items():
                    if name == "WhisperService" or "WhisperService" in name:
                        # Count actual running replicas
                        if hasattr(deployment, 'replicas'):
                            running = [r for r in deployment.replicas if r.state == "RUNNING"]
                            actual_replicas = len(running)
                            replica_details = {
                                "total": len(deployment.replicas),
                                "running": actual_replicas,
                                "states": {r.state: sum(1 for rep in deployment.replicas if rep.state == r.state) for r in deployment.replicas}
                            }
                        elif hasattr(deployment, 'num_replicas'):
                            actual_replicas = deployment.num_replicas
        except Exception as e:
            # If we can't query Ray Serve, that's okay - just show configured value
            pass
        
        # Get request distribution across replicas
        tracking, lock = self._get_tracking()
        with lock:
            request_distribution = dict(tracking)
            total_requests = sum(request_distribution.values())
        
        return {
            "default_model": self.default_model_name,
            "cached_models": cached_models,
            "max_cache_size": self.max_cache_size,
            "compute_type": compute_type,
            "num_replicas_configured": num_replicas,
            "num_replicas_running": actual_replicas if actual_replicas is not None else num_replicas,
            "replica_details": replica_details,
            "num_gpus_per_replica": num_gpus_per_replica,
            "total_gpus_allocated": num_replicas * num_gpus_per_replica,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_info": gpu_info,
            "request_distribution": {
                "total_requests": total_requests,
                "requests_per_replica": request_distribution,
                "num_replicas_handled_requests": len(request_distribution),
            },
            "this_replica": {
                "replica_id": self.replica_id,
                "gpu_id": self.gpu_id,
                "requests_handled": request_distribution.get(self.replica_id, 0),
            },
            "ray_serve": {
                "deployment_name": "WhisperService",
                "application_name": "default",
            }
        }
    
    @api.get("/stats")
    async def get_stats(self):
        """Get request distribution statistics to see how load is balanced."""
        # Get request distribution across replicas
        tracking, lock = self._get_tracking()
        with lock:
            request_distribution = dict(tracking)
            total_requests = sum(request_distribution.values())
        
        # Calculate statistics
        if request_distribution:
            requests_list = list(request_distribution.values())
            import statistics
            avg_requests = statistics.mean(requests_list)
            median_requests = statistics.median(requests_list)
            min_requests = min(requests_list)
            max_requests = max(requests_list)
            std_dev = statistics.stdev(requests_list) if len(requests_list) > 1 else 0
        else:
            avg_requests = median_requests = min_requests = max_requests = std_dev = 0
        
        # Get replica info
        num_replicas_configured = int(os.environ.get("NUM_REPLICAS", "1"))
        num_replicas_with_requests = len(request_distribution)
        
        return {
            "summary": {
                "total_requests": total_requests,
                "num_replicas_configured": num_replicas_configured,
                "num_replicas_handled_requests": num_replicas_with_requests,
                "requests_per_replica_avg": round(avg_requests, 1),
                "requests_per_replica_median": round(median_requests, 1),
                "requests_per_replica_min": min_requests,
                "requests_per_replica_max": max_requests,
                "requests_per_replica_std_dev": round(std_dev, 1),
                "load_balance_ratio": round(max_requests / avg_requests, 2) if avg_requests > 0 else 0,
            },
            "distribution": {
                replica_id: count for replica_id, count in sorted(request_distribution.items(), key=lambda x: x[1], reverse=True)
            },
            "this_replica": {
                "replica_id": self.replica_id,
                "gpu_id": self.gpu_id,
                "requests_handled": request_distribution.get(self.replica_id, 0),
            },
            "interpretation": {
                "load_balance_ratio_explanation": "Ratio of max/avg requests. Closer to 1.0 = better balance. >2.0 = uneven distribution.",
                "if_all_replicas_used": num_replicas_with_requests == num_replicas_configured,
            }
        }

# Entry point for `serve run`
app = WhisperService.bind()
