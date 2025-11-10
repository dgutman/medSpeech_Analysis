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
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

# Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8000")
DATA_DIR = "/data"
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", "4"))  # Number of parallel requests

def get_wav_files(data_dir: str) -> List[str]:
    """Get all WAV files from the data directory."""
    data_path = Path(data_dir)
    wav_files = list(data_path.glob("*.wav"))
    return sorted([str(f.name) for f in wav_files])

def transcribe_file(file_name: str, api_url: str) -> Dict:
    """Transcribe a single file and return results with timing."""
    file_path = f"/data/{file_name}"
    
    start_time = time.time()
    try:
        response = requests.post(
            f"{api_url}/transcribe/path",
            json={
                "file_path": file_path,
                "task": "transcribe",
                "beam_size": 5,
                "language": "en"
            },
            timeout=300  # 5 minute timeout per file
        )
        response.raise_for_status()
        result = response.json()
        
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

def run_stress_test(data_dir: str, api_url: str, max_workers: int = 4):
    """Run stress test on all WAV files."""
    print(f"=== Stress Test Configuration ===")
    print(f"Data directory: {data_dir}")
    print(f"API URL: {api_url}")
    print(f"Max parallel workers: {max_workers}")
    print()
    
    # Get all WAV files
    wav_files = get_wav_files(data_dir)
    total_files = len(wav_files)
    
    if total_files == 0:
        print(f"No WAV files found in {data_dir}")
        return
    
    print(f"Found {total_files} WAV files to process")
    print(f"Starting stress test with {max_workers} parallel workers...")
    print()
    
    # Process files in parallel
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(transcribe_file, file_name, api_url): file_name
            for file_name in wav_files
        }
        
        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            file_name = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                
                status_icon = "✓" if result["status"] == "success" else "✗"
                print(f"[{completed}/{total_files}] {status_icon} {file_name[:50]:<50} "
                      f"Total: {result['total_time_seconds']:.2f}s | "
                      f"Inference: {result['inference_time_seconds']:.2f}s | "
                      f"Audio: {result['audio_duration']:.1f}s")
                
                if result["status"] == "error":
                    print(f"    Error: {result['error']}")
            except Exception as e:
                completed += 1
                print(f"[{completed}/{total_files}] ✗ {file_name[:50]:<50} Exception: {e}")
                results.append({
                    "file": file_name,
                    "status": "error",
                    "error": str(e)
                })
    
    total_time = time.time() - start_time
    
    # Calculate statistics
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    
    if successful:
        total_times = [r["total_time_seconds"] for r in successful]
        inference_times = [r["inference_time_seconds"] for r in successful]
        audio_durations = [r["audio_duration"] for r in successful]
        
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
        print(f"Total audio duration: {sum(audio_durations):.1f} seconds ({sum(audio_durations)/60:.1f} minutes)")
        print(f"Total inference time: {sum(inference_times):.2f} seconds ({sum(inference_times)/60:.2f} minutes)")
        print(f"Throughput: {len(successful) / total_time:.2f} files/second")
        print(f"Average speedup: {sum(audio_durations) / total_time:.2f}x real-time")
    
    if failed:
        print()
        print("=== Failed Files ===")
        for result in failed:
            print(f"  {result['file']}: {result.get('error', 'Unknown error')}")
    
    # Save detailed results to JSON
    output_file = "stress_test_results.json"
    with open(output_file, "w") as f:
        json.dump({
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

if __name__ == "__main__":
    run_stress_test(DATA_DIR, API_URL, MAX_WORKERS)

