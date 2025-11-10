import io
import os
import time
from typing import Optional

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

class PathJob(BaseModel):
    file_path: str  # Path relative to /data or absolute path
    task: Optional[str] = "transcribe"
    beam_size: Optional[int] = 5
    language: Optional[str] = None

# ----- Ray Serve deployment -----
@serve.deployment(
    num_replicas=int(os.environ.get("NUM_REPLICAS", "1")),
    ray_actor_options={
        # 1 GPU per replica; change to 0 if you want CPU-only
        "num_gpus": float(os.environ.get("NUM_GPUS_PER_REPLICA", "1")),
    },
)
@serve.ingress(api)
class WhisperService:
    def __init__(self):
        # Model name can be: tiny, base, small, medium, large-v3, etc.
        self.model_name = os.environ.get("WHISPER_MODEL", "large-v3")
        # compute_type: "float16" for NVIDIA GPUs, "int8_float16" for memory savings
        compute_type = os.environ.get("COMPUTE_TYPE", "float16")
        device = "cuda" if float(os.environ.get("NUM_GPUS_PER_REPLICA", "1")) >= 1 else "cpu"
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)

    def _run(self, audio_bytes: bytes, task="transcribe", beam_size=5, language=None):
        # Normalize language code
        language = normalize_language(language)
        
        # Track inference time
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_bytes,
            task=task,
            beam_size=beam_size,
            language=language,
            vad_filter=True,  # helps on long audio
        )
        text = "".join(seg.text for seg in segments)
        inference_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "language": info.language,
            "duration": info.duration,
            "inference_time_seconds": round(inference_time, 3),
            "text": text,
        }
    
    def _run_from_path(self, file_path: str, task="transcribe", beam_size=5, language=None):
        """Transcribe from a file path - faster_whisper can handle file paths directly."""
        # Normalize language code
        language = normalize_language(language)
        
        # Track inference time
        start_time = time.time()
        segments, info = self.model.transcribe(
            file_path,
            task=task,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
        )
        text = "".join(seg.text for seg in segments)
        inference_time = time.time() - start_time
        
        return {
            "model": self.model_name,
            "language": info.language,
            "duration": info.duration,
            "inference_time_seconds": round(inference_time, 3),
            "text": text,
        }

    @api.post("/transcribe/file")
    async def transcribe_file(
        self,
        file: UploadFile = File(...),
        task: str = Form("transcribe"),
        beam_size: int = Form(5),
        language: Optional[str] = Form(None),
    ):
        if file.content_type and not file.content_type.startswith("audio"):
            raise HTTPException(400, "Upload must be an audio file")
        audio_bytes = await file.read()
        return self._run(audio_bytes, task=task, beam_size=beam_size, language=language)

    @api.post("/transcribe/url")
    async def transcribe_url(self, job: UrlJob):
        # Simple fetch (no internet in some deployments—replace with signed URLs on your side if needed)
        import requests
        r = requests.get(job.url, timeout=120)
        if r.status_code != 200:
            raise HTTPException(400, f"Could not fetch audio: {r.status_code}")
        return self._run(r.content, task=job.task, beam_size=job.beam_size, language=job.language)
    
    @api.post("/transcribe/path")
    async def transcribe_path(self, job: PathJob):
        """Transcribe a WAV file from the mounted /data directory or absolute path."""
        # If path doesn't start with /, assume it's relative to /data
        if not job.file_path.startswith("/"):
            file_path = os.path.join("/data", job.file_path)
        else:
            file_path = job.file_path
        
        if not os.path.exists(file_path):
            raise HTTPException(404, f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise HTTPException(400, f"Path is not a file: {file_path}")
        
        return self._run_from_path(file_path, task=job.task, beam_size=job.beam_size, language=job.language)

# Entry point for `serve run`
app = WhisperService.bind()
