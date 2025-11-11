#!/usr/bin/env python3
"""
Stress test script to process all WAV files in /data directory.
Uses Ray to parallelize across multiple GPUs.
"""
import os
import time
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import statistics

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Default to the host path that maps to container's /data
# Container mounts: /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats -> /data
DATA_DIR = os.environ.get("DATA_DIR", "/scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # Number of parallel requests
MODEL = os.environ.get("MODEL", None)  # Optional: model to use for all files (e.g., "large-v3", "base", "tiny")

def get_wav_files(data_dir: str) -> List[str]:
    """Get all WAV files from the data directory."""
    data_path = Path(data_dir)
    wav_files = list(data_path.glob("*.wav"))
    return sorted([str(f.name) for f in wav_files])

def get_gpu_snapshot() -> List[Dict]:
    """Get current GPU memory and utilization snapshot."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 4:
                        gpus.append({
                            "index": int(parts[0]),
                            "memory_used_mb": int(parts[1]),
                            "memory_total_mb": int(parts[2]),
                            "gpu_utilization_percent": int(parts[3]),
                            "memory_used_percent": round((int(parts[1]) / int(parts[2])) * 100, 1) if int(parts[2]) > 0 else 0
                        })
            return gpus
    except Exception as e:
        print(f"Warning: Could not get GPU snapshot: {e}")
    return None

def transcribe_file(file_name: str, api_url: str, model: Optional[str] = None) -> Dict:
    """Transcribe a single file and return results with timing."""
    # Use container path /data since the API runs inside the container
    # The file_name is just the filename, and we reference it as /data/{filename} in the container
    file_path = f"/data/{file_name}"
    
    start_time = time.time()
    try:
        request_data = {
            "file_path": file_path,
            "task": "transcribe",
            "beam_size": 5,
            "language": "en"
        }
        if model:
            request_data["model"] = model
        
        # Use a new session for each request to avoid connection pooling
        # This forces Ray Serve to route to different replicas
        session = requests.Session()
        # Disable connection pooling to force new connections
        adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=0)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        try:
            response = session.post(
                f"{api_url}/transcribe/path",
                json=request_data,
                timeout=300  # 5 minute timeout per file
            )
            response.raise_for_status()
            result = response.json()
        finally:
            session.close()
        
        total_time = time.time() - start_time
        
        return {
            "file": file_name,
            "status": "success",
            "total_time_seconds": round(total_time, 3),
            "inference_time_seconds": result.get("inference_time_seconds", 0),
            "audio_duration": result.get("duration", 0),
            "model": result.get("model", "unknown"),
            "language": result.get("language", "unknown"),
            "text_length": len(result.get("text", "")),
            "error": None
        }
    except Exception as e:
        total_time = time.time() - start_time
        return {
            "file": file_name,
            "status": "error",
            "total_time_seconds": round(total_time, 3),
            "inference_time_seconds": 0,
            "audio_duration": 0,
            "model": None,
            "language": None,
            "text_length": 0,
            "error": str(e)
        }

def run_stress_test(data_dir: str, api_url: str, max_workers: int = 4, model: Optional[str] = None):
    """Run stress test on all WAV files."""
    print(f"=== Stress Test Configuration ===")
    print(f"Data directory: {data_dir}")
    print(f"API URL: {api_url}")
    print(f"Max parallel workers: {max_workers}")
    if model:
        print(f"Model: {model}")
    else:
        print(f"Model: (using server default)")
    print()
    
    # Get all WAV files
    wav_files = get_wav_files(data_dir)
    total_files = len(wav_files)
    
    if total_files == 0:
        print(f"No WAV files found in {data_dir}")
        return
    
    print(f"Found {total_files} WAV files to process")
    print(f"Starting stress test with {max_workers} parallel workers...")
    print(f"Submitting all {total_files} requests in parallel...")
    print()
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    # Get GPU memory snapshot at start
    gpu_snapshot_start = get_gpu_snapshot()
    
    # Track periodic snapshots (every 500 files)
    gpu_snapshots_periodic = []
    snapshot_interval = 500
    
    # Submit ALL tasks immediately (not one at a time)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once - this creates all the futures immediately
        # Note: We scan files from host path (DATA_DIR), but API requests use /data (container path)
        print("Submitting requests...")
        submit_start = time.time()
        future_to_file = {
            executor.submit(transcribe_file, file_name, api_url, model): file_name
            for file_name in wav_files
        }
        submit_time = time.time() - submit_start
        print(f"✓ All {total_files} requests submitted in {submit_time:.3f}s")
        print(f"Waiting for responses (processing in parallel with {max_workers} workers)...")
        print()
        
        # Process results as they complete (they'll come back in parallel)
        completed = 0
        in_flight = len(future_to_file)
        throughput_report_interval = 100  # Report throughput every N files
        
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            in_flight -= 1
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                # Calculate and report throughput every N files
                if completed % throughput_report_interval == 0:
                    elapsed = time.time() - start_time
                    throughput = completed / elapsed if elapsed > 0 else 0
                    remaining = total_files - completed
                    eta_seconds = remaining / throughput if throughput > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"\n{'='*80}")
                    print(f"📊 Throughput Report (at {completed}/{total_files} files)")
                    print(f"{'='*80}")
                    print(f"  Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
                    print(f"  Throughput: {throughput:.2f} files/second")
                    print(f"  Remaining: {remaining} files")
                    print(f"  Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)")
                    print(f"{'='*80}\n")
                
                # Take GPU snapshot every 500 files
                if completed % snapshot_interval == 0:
                    snapshot = get_gpu_snapshot()
                    if snapshot:
                        gpu_snapshots_periodic.append({
                            "completed_files": completed,
                            "timestamp": datetime.now().isoformat(),
                            "gpus": snapshot
                        })
                        print(f"  [GPU Snapshot at {completed} files] ", end="")
                        for gpu in snapshot:
                            print(f"GPU{gpu['index']}: {gpu['memory_used_mb']}MB ({gpu['memory_used_percent']:.1f}%) ", end="")
                        print()
                
                # Only print errors, not every successful file (too verbose and slows down output)
                if result["status"] != "success":
                    print(f"[{completed}/{total_files}] ✗ ERROR: {file_name} - {result.get('error', 'Unknown error')}")
            except Exception as e:
                completed += 1
                in_flight -= 1
                print(f"[{completed}/{total_files}] ✗ {file_name[:50]:<50} Exception: {e}")
                results.append({
                    "file": file_name,
                    "status": "error",
                    "error": str(e)
                })
    
    # Get GPU snapshot at end
    gpu_snapshot_end = get_gpu_snapshot()
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    if successful:
        total_times = [r["total_time_seconds"] for r in successful]
        inference_times = [r["inference_time_seconds"] for r in successful]
        audio_durations = [r["audio_duration"] for r in successful]
        
        # Track which models were used
        models_used = {}
        for r in successful:
            model_name = r.get("model", "unknown")
            models_used[model_name] = models_used.get(model_name, 0) + 1
        
        # Calculate real-time factor (RTF) = inference_time / audio_duration
        rtfs = [
            inf / dur if dur > 0 else 0
            for inf, dur in zip(inference_times, audio_durations)
        ]
        
        print()
        print("=== Stress Test Results ===")
        print(f"Total files processed: {total_files}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total wall-clock time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print()
        print("=== Timing Statistics (Successful) ===")
        print(f"Total time (request + inference):")
        print(f"  Mean: {statistics.mean(total_times):.3f}s")
        print(f"  Median: {statistics.median(total_times):.3f}s")
        print(f"  Min: {min(total_times):.3f}s")
        print(f"  Max: {max(total_times):.3f}s")
        print()
        print(f"Inference time:")
        print(f"  Mean: {statistics.mean(inference_times):.3f}s")
        print(f"  Median: {statistics.median(inference_times):.3f}s")
        print(f"  Min: {min(inference_times):.3f}s")
        print(f"  Max: {max(inference_times):.3f}s")
        print()
        print(f"Real-time factor (RTF = inference_time / audio_duration):")
        print(f"  Mean: {statistics.mean(rtfs):.3f}x")
        print(f"  Median: {statistics.median(rtfs):.3f}x")
        print(f"  Min: {min(rtfs):.3f}x")
        print(f"  Max: {max(rtfs):.3f}x")
        print(f"  (RTF < 1.0 means faster than real-time)")
        print()
        print(f"Models used:")
        for model_name, count in sorted(models_used.items()):
            print(f"  {model_name}: {count} files ({count/len(successful)*100:.1f}%)")
        print()
        print(f"Total audio duration: {sum(audio_durations):.1f} seconds ({sum(audio_durations)/60:.1f} minutes)")
        print(f"Total inference time: {sum(inference_times):.2f} seconds ({sum(inference_times)/60:.2f} minutes)")
        print(f"Throughput: {len(successful) / total_time:.2f} files/second")
        print(f"Average speedup: {sum(audio_durations) / total_time:.2f}x real-time")
    
    if failed:
        print()
        print("=== Failed Files ===")
        for result in failed:
            print(f"  {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Get server configuration
    server_config = None
    try:
        config_response = requests.get(f"{api_url}/config", timeout=10)
        if config_response.status_code == 200:
            server_config = config_response.json()
    except Exception as e:
        print(f"Warning: Could not fetch server config: {e}")
    
    # Save detailed results to JSON with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"stress_test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump({
            "test_configuration": {
                "api_url": api_url,
                "data_dir": data_dir,
                "max_workers": max_workers,
                "model": model,
                "timestamp": datetime.now().isoformat(),
            },
            "server_configuration": server_config,
            "gpu_snapshots": {
                "start": gpu_snapshot_start,
                "end": gpu_snapshot_end,
                "periodic": gpu_snapshots_periodic,  # Snapshots every 500 files
            },
            "summary": {
                "total_files": total_files,
                "successful": len(successful),
                "failed": len(failed),
                "total_wall_clock_time": total_time,
            },
            "results": results
        }, f, indent=2)
    
    print()
    print(f"Detailed results saved to: {output_file}")
    if gpu_snapshot_start or gpu_snapshot_end:
        print("GPU memory snapshots (start/end) included in results file")

if __name__ == "__main__":
    run_stress_test(DATA_DIR, API_URL, MAX_WORKERS, MODEL)

