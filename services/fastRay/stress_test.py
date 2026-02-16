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

# Try to import psutil for CPU metrics (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU metrics will be limited.")

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Default to the host path that maps to container's /data
# Container mounts: /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats -> /data
DATA_DIR = os.environ.get("DATA_DIR", "/scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats")
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # Number of parallel requests
MODEL = os.environ.get("MODEL", None)  # Optional: model to use for all files (e.g., "large-v3", "base", "tiny")
MAX_FILES = int(os.environ.get("MAX_FILES", "0"))  # Limit number of files (0 = process all)

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

def get_cpu_snapshot() -> Optional[Dict]:
    """Get current CPU utilization snapshot."""
    if not PSUTIL_AVAILABLE:
        return None
    
    try:
        # Get overall CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Get per-CPU usage
        cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
        
        # Get load average (Linux)
        try:
            load_avg = os.getloadavg()
        except:
            load_avg = (0, 0, 0)
        
        return {
            "cpu_percent": round(cpu_percent, 1),
            "cpu_count": cpu_count,
            "cpu_per_core": [round(c, 1) for c in cpu_per_core],
            "load_avg_1min": round(load_avg[0], 2) if len(load_avg) > 0 else 0,
            "load_avg_5min": round(load_avg[1], 2) if len(load_avg) > 1 else 0,
            "load_avg_15min": round(load_avg[2], 2) if len(load_avg) > 2 else 0,
        }
    except Exception as e:
        print(f"Warning: Could not get CPU snapshot: {e}")
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
        
        # Get transcription text - save full text for analysis
        transcription_text = result.get("text", "")
        
        return {
            "file": file_name,
            "status": "success",
            "total_time_seconds": round(total_time, 3),
            "inference_time_seconds": result.get("inference_time_seconds", 0),
            "audio_duration": result.get("duration", 0),
            "model": result.get("model", "unknown"),
            "language": result.get("language", "unknown"),
            "text": transcription_text,  # Full transcription text
            "text_length": len(transcription_text),
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
    
    # Limit files if MAX_FILES is set
    if MAX_FILES > 0 and total_files > MAX_FILES:
        wav_files = wav_files[:MAX_FILES]
        print(f"Found {total_files} WAV files, limiting to first {MAX_FILES} for testing")
        total_files = MAX_FILES
    else:
        print(f"Found {total_files} WAV files to process")
    print(f"Starting stress test with {max_workers} parallel workers...")
    print(f"Submitting all {total_files} requests in parallel...")
    print()
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    # Get GPU and CPU snapshots at start
    gpu_snapshot_start = get_gpu_snapshot()
    cpu_snapshot_start = get_cpu_snapshot()
    
    # Track periodic snapshots (every 500 files)
    gpu_snapshots_periodic = []
    snapshot_interval = 500
    
    # Track completion times for time-windowed throughput analysis
    completion_times = []  # List of (completion_time, audio_duration) tuples
    window_size = 10.0  # 10-second windows for throughput calculation
    throughput_windows = []  # List of (window_start, window_end, count, audio_seconds) tuples
    last_window_end = start_time
    current_window_count = 0
    current_window_audio = 0.0
    
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
            completion_time = time.time()
            try:
                result = future.result()
                result["completion_time"] = completion_time  # Track when it completed
                results.append(result)
                completed += 1
                
                # Track completion for time-windowed throughput
                audio_dur = result.get("audio_duration", 0) if result.get("status") == "success" else 0
                completion_times.append((completion_time, audio_dur))
                
                # Update current window
                while completion_time >= last_window_end + window_size:
                    # Save current window
                    if current_window_count > 0:
                        throughput_windows.append((
                            last_window_end,
                            last_window_end + window_size,
                            current_window_count,
                            current_window_audio
                        ))
                    # Start new window
                    last_window_end += window_size
                    current_window_count = 0
                    current_window_audio = 0.0
                
                # Add to current window
                if result.get("status") == "success":
                    current_window_count += 1
                    current_window_audio += audio_dur
                
                # Calculate and report throughput every N files
                if completed % throughput_report_interval == 0:
                    elapsed = time.time() - start_time
                    cumulative_throughput = completed / elapsed if elapsed > 0 else 0
                    
                    # Calculate recent throughput (last window if available, or last 10 seconds)
                    recent_throughput = 0.0
                    recent_audio_throughput = 0.0
                    if throughput_windows:
                        last_window = throughput_windows[-1]
                        recent_throughput = last_window[2] / window_size
                        recent_audio_throughput = last_window[3] / window_size
                    elif elapsed >= window_size:
                        # Use last window_size seconds
                        recent_completions = [ct for ct in completion_times if ct[0] >= completion_time - window_size]
                        recent_throughput = len(recent_completions) / window_size
                        recent_audio_throughput = sum(ct[1] for ct in recent_completions) / window_size
                    
                    remaining = total_files - completed
                    eta_seconds = remaining / cumulative_throughput if cumulative_throughput > 0 else 0
                    eta_minutes = eta_seconds / 60
                    
                    print(f"\n{'='*80}")
                    print(f"📊 Throughput Report (at {completed}/{total_files} files)")
                    print(f"{'='*80}")
                    print(f"  Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
                    print(f"  Cumulative throughput: {cumulative_throughput:.2f} messages/second")
                    if recent_throughput > 0:
                        print(f"  Recent throughput (last {window_size}s): {recent_throughput:.2f} messages/second")
                        if recent_audio_throughput > 0:
                            print(f"  Recent audio throughput: {recent_audio_throughput:.2f} audio-seconds/second")
                    print(f"  Remaining: {remaining} files")
                    print(f"  Estimated time remaining: {eta_minutes:.1f} minutes ({eta_seconds:.0f} seconds)")
                    print(f"{'='*80}\n")
                
                # Take GPU and CPU snapshots every 500 files
                if completed % snapshot_interval == 0:
                    gpu_snapshot = get_gpu_snapshot()
                    cpu_snapshot = get_cpu_snapshot()
                    
                    snapshot_data = {
                        "completed_files": completed,
                        "timestamp": datetime.now().isoformat(),
                    }
                    if gpu_snapshot:
                        snapshot_data["gpus"] = gpu_snapshot
                    if cpu_snapshot:
                        snapshot_data["cpu"] = cpu_snapshot
                    
                    if gpu_snapshot or cpu_snapshot:
                        gpu_snapshots_periodic.append(snapshot_data)
                        # Metrics are saved to JSON, but not printed to console to reduce verbosity
                
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
    
    # Get GPU and CPU snapshots at end
    gpu_snapshot_end = get_gpu_snapshot()
    cpu_snapshot_end = get_cpu_snapshot()
    
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
        print()
        print("=== Throughput Statistics ===")
        overall_throughput = len(successful) / total_time if total_time > 0 else 0
        audio_throughput = sum(audio_durations) / total_time if total_time > 0 else 0
        print(f"Overall throughput: {overall_throughput:.2f} messages/second")
        print(f"Audio throughput: {audio_throughput:.2f} audio-seconds/second")
        print(f"Average speedup: {sum(audio_durations) / total_time:.2f}x real-time")
        
        # Calculate peak and time-windowed throughput
        if throughput_windows:
            window_throughputs = [w[2] / window_size for w in throughput_windows]
            window_audio_throughputs = [w[3] / window_size for w in throughput_windows]
            
            peak_throughput = max(window_throughputs) if window_throughputs else 0
            peak_audio_throughput = max(window_audio_throughputs) if window_audio_throughputs else 0
            avg_window_throughput = statistics.mean(window_throughputs) if window_throughputs else 0
            avg_window_audio_throughput = statistics.mean(window_audio_throughputs) if window_audio_throughputs else 0
            
            print()
            print(f"Peak throughput ({window_size}s window): {peak_throughput:.2f} messages/second")
            if peak_audio_throughput > 0:
                print(f"Peak audio throughput ({window_size}s window): {peak_audio_throughput:.2f} audio-seconds/second")
            print(f"Average window throughput: {avg_window_throughput:.2f} messages/second")
            if avg_window_audio_throughput > 0:
                print(f"Average window audio throughput: {avg_window_audio_throughput:.2f} audio-seconds/second")
            
            # Calculate throughput excluding warmup (first 30 seconds)
            warmup_period = 30.0
            steady_state_windows = [
                w for w in throughput_windows 
                if w[0] >= start_time + warmup_period
            ]
            if steady_state_windows:
                steady_throughputs = [w[2] / window_size for w in steady_state_windows]
                steady_audio_throughputs = [w[3] / window_size for w in steady_state_windows]
                steady_avg = statistics.mean(steady_throughputs) if steady_throughputs else 0
                steady_audio_avg = statistics.mean(steady_audio_throughputs) if steady_audio_throughputs else 0
                print()
                print(f"Steady-state throughput (excluding first {warmup_period}s): {steady_avg:.2f} messages/second")
                if steady_audio_avg > 0:
                    print(f"Steady-state audio throughput: {steady_audio_avg:.2f} audio-seconds/second")
    
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
    
    # Calculate final window if test ended mid-window
    if completion_times:
        final_window_end = time.time()
        if final_window_end > last_window_end:
            if current_window_count > 0:
                throughput_windows.append((
                    last_window_end,
                    final_window_end,
                    current_window_count,
                    current_window_audio
                ))
    
    # Calculate throughput statistics for JSON output
    throughput_stats = {}
    if throughput_windows:
        window_throughputs = [w[2] / window_size for w in throughput_windows]
        window_audio_throughputs = [w[3] / window_size for w in throughput_windows if w[1] > w[0]]
        
        throughput_stats = {
            "window_size_seconds": window_size,
            "num_windows": len(throughput_windows),
            "peak_throughput_messages_per_second": max(window_throughputs) if window_throughputs else 0,
            "average_window_throughput_messages_per_second": statistics.mean(window_throughputs) if window_throughputs else 0,
            "min_window_throughput_messages_per_second": min(window_throughputs) if window_throughputs else 0,
            "throughput_windows": [
                {
                    "window_start": w[0],
                    "window_end": w[1],
                    "messages": w[2],
                    "audio_seconds": w[3],
                    "throughput_messages_per_second": w[2] / window_size,
                    "audio_throughput_per_second": w[3] / (w[1] - w[0]) if w[1] > w[0] else 0
                }
                for w in throughput_windows
            ]
        }
        
        if window_audio_throughputs:
            throughput_stats["peak_audio_throughput_per_second"] = max(window_audio_throughputs)
            throughput_stats["average_audio_throughput_per_second"] = statistics.mean(window_audio_throughputs)
    
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
            "cpu_snapshots": {
                "start": cpu_snapshot_start,
                "end": cpu_snapshot_end,
                "periodic": [s.get("cpu") for s in gpu_snapshots_periodic if "cpu" in s],  # CPU data from periodic snapshots
            },
            "summary": {
                "total_files": total_files,
                "successful": len(successful),
                "failed": len(failed),
                "total_wall_clock_time": total_time,
                "overall_throughput_messages_per_second": len(successful) / total_time if total_time > 0 else 0,
            },
            "throughput_analysis": throughput_stats,
            "results": results
        }, f, indent=2)
    
    print()
    print(f"Detailed results saved to: {output_file}")
    if gpu_snapshot_start or gpu_snapshot_end:
        print("GPU snapshots (start/end) included in results file")
    if cpu_snapshot_start or cpu_snapshot_end:
        print("CPU snapshots (start/end) included in results file")

if __name__ == "__main__":
    run_stress_test(DATA_DIR, API_URL, MAX_WORKERS, MODEL)

