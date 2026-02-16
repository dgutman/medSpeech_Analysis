#!/bin/bash
# Automated benchmarking script to test different Ray configuration permutations
# Tests each configuration with a specified number of messages and collects results
#
# Usage:
#   bash benchmark_configurations.sh
#   
#   Or customize:
#   FILES_PER_TRIAL=500 MAX_WORKERS=300 bash benchmark_configurations.sh
#
# Note: Each file = 1 transcription request = 1 message
#       So FILES_PER_TRIAL=1000 means transcribing 1000 audio files
#
# The script will:
#   1. Test each configuration permutation
#   2. Stop and restart Ray between each test
#   3. Run stress test with specified number of files
#   4. Save results to benchmark_results_<timestamp>/
#   5. Generate a summary file
#
# After completion, compare results:
#   python3 compare_benchmark_results.py benchmark_results_<timestamp>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
CONTAINER_NAME="fastray-container"
# Number of files to transcribe per trial (each file = 1 transcription request = 1 message)
FILES_PER_TRIAL="${FILES_PER_TRIAL:-${MESSAGES_PER_TRIAL:-2000}}"  # Support both names for clarity (default 2000)
MAX_WORKERS="${MAX_WORKERS:-200}"  # Stress test workers (should be high to saturate)
WAIT_FOR_READY="${WAIT_FOR_READY:-60}"  # Seconds to wait for Ray to be ready after restart (increased for reliability)
MODEL="${MODEL:-}"  # Optional: model to use

# Results directory
RESULTS_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Ray Configuration Benchmarking Script"
echo "=========================================="
echo "Files to transcribe per trial: $FILES_PER_TRIAL (each file = 1 transcription request)"
echo "Max workers (stress test): $MAX_WORKERS"
echo "Results directory: $RESULTS_DIR"
echo ""

# Function to wait for Ray API to be ready
wait_for_ray_ready() {
    local max_wait=$1
    local waited=0
    echo "Waiting for Ray API to be ready..."
    while [ $waited -lt $max_wait ]; do
        # Check if docs endpoint is accessible
        if curl -s "$API_URL/docs" > /dev/null 2>&1; then
            # Check if /config endpoint works (Ray Serve is up)
            if curl -s "$API_URL/config" > /dev/null 2>&1; then
                # Also verify the /transcribe/path endpoint is registered
                # Check OpenAPI spec to ensure endpoint exists
                if curl -s "$API_URL/openapi.json" | grep -q "/transcribe/path" 2>/dev/null; then
                    echo "✓ Ray API is ready (waited ${waited}s)"
                    # Give it a couple more seconds for full initialization
                    sleep 3
                    return 0
                fi
            fi
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo ""
    echo "⚠️  Warning: Ray API may not be fully ready after ${max_wait}s"
    return 1
}

# Function to stop Ray container
stop_ray() {
    echo "Stopping Ray container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || echo "  (container not running)"
    docker rm "$CONTAINER_NAME" 2>/dev/null || echo "  (container not found)"
    sleep 2
}

# Function to start Ray with specific configuration
start_ray() {
    local num_replicas=$1
    local num_gpus_per_replica=$2
    local num_cpus_per_replica=$3
    local max_ongoing_requests=$4
    
    echo "Starting Ray with configuration:"
    echo "  NUM_REPLICAS=$num_replicas"
    echo "  NUM_GPUS_PER_REPLICA=$num_gpus_per_replica"
    echo "  NUM_CPUS_PER_REPLICA=$num_cpus_per_replica"
    echo "  MAX_ONGOING_REQUESTS=$max_ongoing_requests"
    
    # Export configuration
    export NUM_REPLICAS=$num_replicas
    export NUM_GPUS_PER_REPLICA=$num_gpus_per_replica
    export NUM_CPUS_PER_REPLICA=$num_cpus_per_replica
    export MAX_ONGOING_REQUESTS=$max_ongoing_requests
    export ENABLE_AUTOSCALING=false
    
    # Start Ray (suppress output, we'll check status separately)
    bash run_fastray.sh > /dev/null 2>&1
    
    # Wait for Ray to be ready
    wait_for_ray_ready $WAIT_FOR_READY
}

# Function to run stress test and save results
run_stress_test() {
    local config_name=$1
    local output_file="$RESULTS_DIR/stress_test_${config_name}.json"
    
    echo "Running stress test with $FILES_PER_TRIAL files (transcription requests)..."
    
    # Export environment variables for stress test
    export API_URL
    export MAX_WORKERS
    export MODEL
    export MAX_FILES=$FILES_PER_TRIAL
    
    # Run stress test and capture output
    python3 stress_test.py 2>&1 | tee "$RESULTS_DIR/stress_test_${config_name}.log"
    
    # Find the most recent results file and rename it
    latest_result=$(ls -t stress_test_results_*.json 2>/dev/null | head -1)
    if [ -n "$latest_result" ]; then
        mv "$latest_result" "$output_file"
        echo "✓ Results saved to: $output_file"
    else
        echo "⚠️  Warning: No results file found"
    fi
}

# Define configuration permutations to test
# Format: "config_name:num_replicas:num_gpus_per_replica:num_cpus_per_replica:max_ongoing_requests"
# IMPORTANT: NUM_REPLICAS × NUM_GPUS_PER_REPLICA must equal 4.0 to use all 4 GPUs
# Ray will distribute replicas evenly across all 4 GPUs when total = 4.0
declare -a CONFIGURATIONS=(
    # Conservative configurations (all use 4 GPUs)
    "4replicas_1gpu_2cpu_30req:4:1:2:30"          # 4.0 GPUs ✓
    "8replicas_0.5gpu_2cpu_30req:8:0.5:2:30"      # 4.0 GPUs ✓
    
    # Balanced configurations (all use 4 GPUs)
    "16replicas_0.25gpu_2cpu_30req:16:0.25:2:30"  # 4.0 GPUs ✓
    "16replicas_0.25gpu_3cpu_30req:16:0.25:3:30"  # 4.0 GPUs ✓
    "16replicas_0.25gpu_2cpu_50req:16:0.25:2:50"  # 4.0 GPUs ✓
    
    # High throughput configurations (all use 4 GPUs)
    "20replicas_0.2gpu_2cpu_50req:20:0.2:2:50"    # 4.0 GPUs ✓
    "20replicas_0.2gpu_3cpu_50req:20:0.2:3:50"    # 4.0 GPUs ✓
    "40replicas_0.1gpu_2cpu_50req:40:0.1:2:50"    # 4.0 GPUs ✓
    "40replicas_0.1gpu_3cpu_50req:40:0.1:3:50"    # 4.0 GPUs ✓
    "40replicas_0.1gpu_3cpu_100req:40:0.1:3:100"  # 4.0 GPUs ✓
)

# Summary file
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary.txt"
echo "Benchmark Summary" > "$SUMMARY_FILE"
echo "=================" >> "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "Files per trial: $FILES_PER_TRIAL (each file = 1 transcription request)" >> "$SUMMARY_FILE"
echo "Max workers: $MAX_WORKERS" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Run benchmarks
total_configs=${#CONFIGURATIONS[@]}
config_num=0

for config in "${CONFIGURATIONS[@]}"; do
    config_num=$((config_num + 1))
    IFS=':' read -r config_name num_replicas num_gpus num_cpus max_ongoing <<< "$config"
    
    echo ""
    echo "=========================================="
    echo "Configuration $config_num/$total_configs: $config_name"
    echo "=========================================="
    echo ""
    
    # Stop Ray
    stop_ray
    
    # Start Ray with new configuration
    if ! start_ray "$num_replicas" "$num_gpus" "$num_cpus" "$max_ongoing"; then
        echo "⚠️  Failed to start Ray with configuration $config_name, skipping..."
        echo "SKIPPED: $config_name - Failed to start" >> "$SUMMARY_FILE"
        continue
    fi
    
    # Run stress test
    run_stress_test "$config_name"
    
    # Extract key metrics from results (if available)
    result_file="$RESULTS_DIR/stress_test_${config_name}.json"
    if [ -f "$result_file" ]; then
        # Use Python to extract metrics - pass variables as arguments to avoid bash parsing issues
        python3 - "$result_file" "$config_name" "$num_replicas" "$num_gpus" "$num_cpus" "$max_ongoing" << 'PYTHON_SCRIPT' >> "$SUMMARY_FILE"
import json
import sys

result_file = sys.argv[1]
config_name = sys.argv[2]
num_replicas = sys.argv[3]
num_gpus = sys.argv[4]
num_cpus = sys.argv[5]
max_ongoing = sys.argv[6]

try:
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    summary = data.get('summary', {})
    throughput = data.get('throughput_analysis', {})
    
    print(f"\n{config_name}:")
    print(f"  Replicas: {num_replicas}, GPUs/replica: {num_gpus}, CPUs/replica: {num_cpus}, Max ongoing: {max_ongoing}")
    print(f"  Successful: {summary.get('successful', 'N/A')}")
    total_time = summary.get('total_wall_clock_time', 'N/A')
    if isinstance(total_time, (int, float)):
        print(f"  Total time: {total_time:.2f}s")
    else:
        print(f"  Total time: {total_time}")
    overall = summary.get('overall_throughput_messages_per_second', 'N/A')
    if isinstance(overall, (int, float)):
        print(f"  Overall throughput: {overall:.2f} msg/s")
    else:
        print(f"  Overall throughput: {overall} msg/s")
    
    if throughput:
        peak = throughput.get('peak_throughput_messages_per_second', 'N/A')
        if isinstance(peak, (int, float)):
            print(f"  Peak throughput: {peak:.2f} msg/s")
        else:
            print(f"  Peak throughput: {peak} msg/s")
        avg = throughput.get('average_window_throughput_messages_per_second', 'N/A')
        if isinstance(avg, (int, float)):
            print(f"  Avg window throughput: {avg:.2f} msg/s")
        else:
            print(f"  Avg window throughput: {avg} msg/s")
except Exception as e:
    print(f"  Error extracting metrics: {e}")
PYTHON_SCRIPT
    else
        echo "$config_name: No results file found" >> "$SUMMARY_FILE"
    fi
    
    echo ""
    echo "✓ Completed configuration $config_name"
    echo ""
done

echo ""
echo "=========================================="
echo "Benchmarking Complete!"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "To view summary:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "To compare results:"
echo "  python3 compare_benchmark_results.py $RESULTS_DIR"
echo ""
echo "Or view individual results:"
echo "  ls -lh $RESULTS_DIR/stress_test_*.json"
echo ""

