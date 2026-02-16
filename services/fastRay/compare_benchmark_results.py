#!/usr/bin/env python3
"""
Compare benchmark results from multiple configuration tests.
Usage: python3 compare_benchmark_results.py <results_directory>
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd

def extract_config_from_filename(filename: str) -> Dict:
    """Extract configuration from filename like 'stress_test_16replicas_0.25gpu_2cpu_30req.json'"""
    # Remove prefix and suffix
    name = filename.replace('stress_test_', '').replace('.json', '')
    
    # Parse components
    parts = name.split('_')
    config = {}
    
    for part in parts:
        if 'replicas' in part:
            config['replicas'] = int(part.replace('replicas', ''))
        elif 'gpu' in part:
            config['gpus_per_replica'] = float(part.replace('gpu', ''))
        elif 'cpu' in part:
            config['cpus_per_replica'] = int(part.replace('cpu', ''))
        elif 'req' in part:
            config['max_ongoing_requests'] = int(part.replace('req', ''))
    
    return config

def load_results(results_dir: str) -> List[Dict]:
    """Load all result files and extract key metrics."""
    results = []
    results_path = Path(results_dir)
    
    for result_file in results_path.glob("stress_test_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            # Extract configuration from filename
            config = extract_config_from_filename(result_file.name)
            
            # Extract metrics
            summary = data.get('summary', {})
            throughput = data.get('throughput_analysis', {})
            
            result = {
                'config_name': result_file.stem.replace('stress_test_', ''),
                **config,
                'successful': summary.get('successful', 0),
                'failed': summary.get('failed', 0),
                'total_time': summary.get('total_wall_clock_time', 0),
                'overall_throughput': summary.get('overall_throughput_messages_per_second', 0),
                'peak_throughput': throughput.get('peak_throughput_messages_per_second', 0),
                'avg_window_throughput': throughput.get('average_window_throughput_messages_per_second', 0),
                'peak_audio_throughput': throughput.get('peak_audio_throughput_per_second', 0),
                'file': str(result_file)
            }
            
            results.append(result)
        except Exception as e:
            print(f"Error loading {result_file}: {e}", file=sys.stderr)
    
    return results

def print_comparison_table(results: List[Dict]):
    """Print a formatted comparison table."""
    if not results:
        print("No results to compare")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by overall throughput (descending)
    df = df.sort_values('overall_throughput', ascending=False)
    
    print("\n" + "="*120)
    print("BENCHMARK RESULTS COMPARISON")
    print("="*120)
    print()
    
    # Print table
    print(f"{'Config':<30} {'Replicas':<8} {'GPU/Rep':<8} {'CPU/Rep':<8} {'MaxReq':<8} {'Throughput':<12} {'Peak':<12} {'Time(s)':<10}")
    print("-" * 120)
    
    for _, row in df.iterrows():
        print(f"{row['config_name']:<30} "
              f"{int(row.get('replicas', 0)):<8} "
              f"{row.get('gpus_per_replica', 0):<8.3f} "
              f"{int(row.get('cpus_per_replica', 0)):<8} "
              f"{int(row.get('max_ongoing_requests', 0)):<8} "
              f"{row['overall_throughput']:<12.2f} "
              f"{row['peak_throughput']:<12.2f} "
              f"{row['total_time']:<10.1f}")
    
    print()
    print("="*120)
    print(f"Best configuration: {df.iloc[0]['config_name']}")
    print(f"  Throughput: {df.iloc[0]['overall_throughput']:.2f} messages/second")
    print(f"  Peak: {df.iloc[0]['peak_throughput']:.2f} messages/second")
    print("="*120)
    
    # Save to CSV
    csv_file = Path(results[0]['file']).parent / 'benchmark_comparison.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed comparison saved to: {csv_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 compare_benchmark_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    results = load_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        sys.exit(1)
    
    print_comparison_table(results)






