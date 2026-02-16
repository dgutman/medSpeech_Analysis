#!/usr/bin/env python3
"""
Analyze GPU utilization from Prometheus/dcgm-exporter metrics.
Queries Prometheus for GPU utilization during benchmark tests.
"""

import requests
import json
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional

PROMETHEUS_URL = "http://localhost:9090"
DCGM_EXPORTER_URL = "http://localhost:9400"

def query_prometheus(query: str, time_range: Optional[tuple] = None) -> Dict:
    """Query Prometheus API."""
    url = f"{PROMETHEUS_URL}/api/v1/query"
    if time_range:
        # Range query
        url = f"{PROMETHEUS_URL}/api/v1/query_range"
        params = {
            "query": query,
            "start": time_range[0],
            "end": time_range[1],
            "step": "5s"  # 5 second intervals
        }
    else:
        # Instant query
        params = {"query": query}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return {}

def get_gpu_utilization(time_range: Optional[tuple] = None) -> Dict:
    """Get GPU utilization metrics."""
    query = "DCGM_FI_DEV_GPU_UTIL"
    
    if time_range:
        # Range query for historical data
        result = query_prometheus(query, time_range)
    else:
        # Current utilization
        result = query_prometheus(query)
    
    return result

def analyze_utilization_during_test(test_start_time: str, test_end_time: str):
    """Analyze GPU utilization during a specific test period."""
    print("="*80)
    print("GPU UTILIZATION ANALYSIS FROM PROMETHEUS")
    print("="*80)
    print()
    
    # Convert to Unix timestamps
    try:
        start_dt = datetime.fromisoformat(test_start_time.replace('Z', '+00:00'))
        end_dt = datetime.fromisoformat(test_end_time.replace('Z', '+00:00'))
    except:
        print("Error: Invalid time format. Use ISO format: 2025-12-15T18:00:00Z")
        return
    
    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())
    
    print(f"Test period: {test_start_time} to {test_end_time}")
    print(f"Duration: {(end_ts - start_ts) / 60:.1f} minutes")
    print()
    
    # Query GPU utilization
    result = get_gpu_utilization((start_ts, end_ts))
    
    if 'data' in result and 'result' in result['data']:
        print("GPU Utilization by GPU:")
        print("-" * 80)
        
        gpu_stats = {}
        
        for series in result['data']['result']:
            gpu_id = series['metric'].get('gpu', 'unknown')
            values = series.get('values', [])
            
            if values:
                utilizations = [float(v[1]) for v in values]
                avg_util = sum(utilizations) / len(utilizations)
                min_util = min(utilizations)
                max_util = max(utilizations)
                
                # Count samples at 100%
                at_100 = sum(1 for u in utilizations if u >= 99)
                percent_at_100 = (at_100 / len(utilizations)) * 100
                
                gpu_stats[gpu_id] = {
                    'avg': avg_util,
                    'min': min_util,
                    'max': max_util,
                    'samples': len(utilizations),
                    'at_100_percent': percent_at_100
                }
                
                status = "✓" if avg_util >= 95 else "⚠️" if avg_util >= 80 else "✗"
                print(f"GPU {gpu_id}: {status}")
                print(f"  Average: {avg_util:.1f}%")
                print(f"  Range: {min_util:.1f}% - {max_util:.1f}%")
                print(f"  Time at 100%: {percent_at_100:.1f}% of samples")
                print()
        
        # Overall summary
        if gpu_stats:
            overall_avg = sum(s['avg'] for s in gpu_stats.values()) / len(gpu_stats)
            overall_at_100 = sum(s['at_100_percent'] for s in gpu_stats.values()) / len(gpu_stats)
            
            print("="*80)
            print("OVERALL SUMMARY:")
            print(f"  Average utilization across all GPUs: {overall_avg:.1f}%")
            print(f"  Average time at 100%: {overall_at_100:.1f}%")
            print()
            
            if overall_avg >= 95:
                print("✓ Excellent GPU utilization!")
            elif overall_avg >= 80:
                print("⚠️  Good utilization, but room for improvement")
            else:
                print("✗ Low utilization - consider increasing MAX_ONGOING_REQUESTS or replicas")
            print("="*80)
    else:
        print("No data found. Check:")
        print("  1. Prometheus is running on port 9090")
        print("  2. dcgm-exporter is running on port 9400")
        print("  3. Prometheus is scraping dcgm-exporter")
        print("  4. Test time range is correct")

def get_current_utilization():
    """Get current GPU utilization."""
    result = get_gpu_utilization()
    
    if 'data' in result and 'result' in result['data']:
        print("Current GPU Utilization:")
        print("-" * 40)
        for series in result['data']['result']:
            gpu_id = series['metric'].get('gpu', 'unknown')
            value = float(series['value'][1])
            status = "✓" if value >= 95 else "⚠️" if value >= 80 else "✗"
            print(f"GPU {gpu_id}: {value:.1f}% {status}")
    else:
        print("No current data available")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "current":
            get_current_utilization()
        elif len(sys.argv) >= 3:
            # Analyze time range
            analyze_utilization_during_test(sys.argv[1], sys.argv[2])
        else:
            print("Usage:")
            print("  python3 analyze_gpu_utilization.py current")
            print("  python3 analyze_gpu_utilization.py <start_time> <end_time>")
            print()
            print("Example:")
            print("  python3 analyze_gpu_utilization.py 2025-12-15T18:00:00Z 2025-12-15T18:05:00Z")
    else:
        get_current_utilization()





