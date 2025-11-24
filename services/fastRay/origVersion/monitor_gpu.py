#!/usr/bin/env python3
"""
Monitor GPU memory usage during inference.
Can be run alongside stress tests to track GPU utilization.
"""
import time
import subprocess
import json
from datetime import datetime
from typing import List, Dict

def get_gpu_memory() -> List[Dict]:
    """Get GPU memory usage using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mb": int(parts[2]),
                    "memory_total_mb": int(parts[3]),
                    "gpu_utilization_percent": int(parts[4]),
                    "memory_used_percent": round((int(parts[2]) / int(parts[3])) * 100, 1) if int(parts[3]) > 0 else 0
                })
        return gpus
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return []

def monitor_gpu(interval: float = 1.0, duration: float = None, output_file: str = None):
    """Monitor GPU memory usage continuously."""
    print("Starting GPU monitoring...")
    print("Press Ctrl+C to stop")
    print()
    
    samples = []
    start_time = time.time()
    
    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break
            
            timestamp = datetime.now().isoformat()
            gpus = get_gpu_memory()
            
            if gpus:
                sample = {
                    "timestamp": timestamp,
                    "elapsed_seconds": round(time.time() - start_time, 2),
                    "gpus": gpus
                }
                samples.append(sample)
                
                # Print current status
                print(f"[{sample['elapsed_seconds']:.1f}s] ", end="")
                for gpu in gpus:
                    print(f"GPU{gpu['index']}: {gpu['memory_used_mb']}/{gpu['memory_total_mb']}MB "
                          f"({gpu['memory_used_percent']:.1f}%) "
                          f"Util: {gpu['gpu_utilization_percent']}%  ", end="")
                print()
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\nStopping GPU monitoring...")
    
    # Calculate statistics
    if samples:
        print()
        print("=== GPU Memory Statistics ===")
        for gpu_idx in range(len(samples[0]["gpus"])):
            gpu_samples = [s["gpus"][gpu_idx] for s in samples]
            memory_used = [g["memory_used_mb"] for g in gpu_samples]
            memory_percent = [g["memory_used_percent"] for g in gpu_samples]
            utilization = [g["gpu_utilization_percent"] for g in gpu_samples]
            
            gpu_name = gpu_samples[0]["name"]
            print(f"\nGPU {gpu_idx} ({gpu_name}):")
            print(f"  Memory used: {min(memory_used)}-{max(memory_used)} MB "
                  f"(avg: {sum(memory_used)/len(memory_used):.0f} MB)")
            print(f"  Memory percent: {min(memory_percent):.1f}-{max(memory_percent):.1f}% "
                  f"(avg: {sum(memory_percent)/len(memory_percent):.1f}%)")
            print(f"  GPU utilization: {min(utilization)}-{max(utilization)}% "
                  f"(avg: {sum(utilization)/len(utilization):.0f}%)")
        
        # Save to file if requested
        if output_file:
            with open(output_file, "w") as f:
                json.dump({
                    "monitoring_duration_seconds": time.time() - start_time,
                    "samples": samples
                }, f, indent=2)
            print(f"\nDetailed monitoring data saved to: {output_file}")

if __name__ == "__main__":
    import sys
    
    interval = 1.0
    duration = None
    output_file = "gpu_monitoring.json"
    
    if len(sys.argv) > 1:
        interval = float(sys.argv[1])
    if len(sys.argv) > 2:
        duration = float(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = sys.argv[3]
    
    monitor_gpu(interval=interval, duration=duration, output_file=output_file)

