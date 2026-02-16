#!/bin/bash
# Test different MAX_ONGOING_REQUESTS values with 32 replicas

# Suppress bash state management errors (harmless but noisy)
exec 2> >(grep -v "dump_bash_state" >&2)

set -e

API_URL="${API_URL:-http://localhost:8000}"
FILES_PER_TRIAL="${FILES_PER_TRIAL:-2000}"
MAX_WORKERS="${MAX_WORKERS:-200}"

# Fixed configuration
NUM_REPLICAS=32
NUM_GPUS_PER_REPLICA=0.125
NUM_CPUS_PER_REPLICA=2

# MAX_ONGOING_REQUESTS values to test
# Testing both low values (to force distribution) and high values (to test saturation)
MAX_REQ_VALUES=(5 10 15 20 30 50 100)

# Results directory
RESULTS_DIR="max_ongoing_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "MAX_ONGOING_REQUESTS Test Results" > "$SUMMARY_FILE"
echo "=================================" >> "$SUMMARY_FILE"
echo "Configuration: $NUM_REPLICAS replicas × $NUM_GPUS_PER_REPLICA GPU × $NUM_CPUS_PER_REPLICA CPU" >> "$SUMMARY_FILE"
echo "Files per test: $FILES_PER_TRIAL" >> "$SUMMARY_FILE"
echo "Started: $(date)" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

# Function to stop Ray
stop_ray() {
    echo "Stopping Ray container..."
    docker stop fastray-container 2>/dev/null || echo "  (container not running)"
    docker rm fastray-container 2>/dev/null || echo "  (container not found)"
    sleep 2
}

# Function to start Ray
start_ray() {
    local max_ongoing=$1
    echo "Starting Ray with MAX_ONGOING_REQUESTS=$max_ongoing..."
    export NUM_REPLICAS=$NUM_REPLICAS
    export NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA
    export NUM_CPUS_PER_REPLICA=$NUM_CPUS_PER_REPLICA
    export MAX_ONGOING_REQUESTS=$max_ongoing
    
    bash run_fastray.sh > /dev/null 2>&1
    
    # Wait for Ray to be ready
    echo "Waiting for Ray API..."
    local waited=0
    while [ $waited -lt 60 ]; do
        if curl -s "$API_URL/docs" > /dev/null 2>&1 && \
           curl -s "$API_URL/config" > /dev/null 2>&1 && \
           curl -s "$API_URL/openapi.json" | grep -q "/transcribe/path" 2>/dev/null; then
            echo "✓ Ray API is ready"
            sleep 3
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
        echo -n "."
    done
    echo "⚠️  Ray may not be fully ready"
    return 1
}

# Function to run stress test
run_test() {
    local max_ongoing=$1
    local output_file="$RESULTS_DIR/stress_test_32replicas_0.125gpu_2cpu_${max_ongoing}req.json"
    
    echo ""
    echo "Running stress test with MAX_ONGOING_REQUESTS=$max_ongoing..."
    export MAX_FILES=$FILES_PER_TRIAL
    export API_URL=$API_URL
    export MAX_WORKERS=$MAX_WORKERS  # Ensure MAX_WORKERS is exported for stress test
    
    python3 stress_test.py > /tmp/stress_test_output.log 2>&1
    
    # Find the results file
    local result_file=$(ls -t stress_test_results_*.json 2>/dev/null | head -1)
    if [ -n "$result_file" ] && [ -f "$result_file" ]; then
        mv "$result_file" "$output_file"
        echo "✓ Results saved to: $output_file"
        
        # Extract key metrics
        python3 << PYEOF
import json
with open('$output_file', 'r') as f:
    data = json.load(f)
summary = data.get('summary', {})
throughput = data.get('throughput_analysis', {})
print(f"  Overall: {summary.get('overall_throughput_messages_per_second', 0):.2f} msg/s")
print(f"  Peak: {throughput.get('peak_throughput_messages_per_second', 0):.2f} msg/s")
print(f"  Time: {summary.get('total_wall_clock_time', 0):.1f}s")
PYEOF
        
        # Append to summary
        python3 << PYEOF >> "$SUMMARY_FILE"
import json
max_ongoing = $max_ongoing
with open('$output_file', 'r') as f:
    data = json.load(f)
summary = data.get('summary', {})
throughput = data.get('throughput_analysis', {})
print(f"MAX_ONGOING_REQUESTS={max_ongoing}:")
print(f"  Overall: {summary.get('overall_throughput_messages_per_second', 0):.2f} msg/s")
print(f"  Peak: {throughput.get('peak_throughput_messages_per_second', 0):.2f} msg/s")
print(f"  Steady-state: {throughput.get('steady_state_throughput_messages_per_second', 0):.2f} msg/s")
print(f"  Time: {summary.get('total_wall_clock_time', 0):.1f}s")
print("")
PYEOF
    else
        echo "⚠️  Warning: No results file found"
        echo "MAX_ONGOING_REQUESTS=${max_ongoing}: FAILED" >> "$SUMMARY_FILE"
    fi
}

# Run tests
total_tests=${#MAX_REQ_VALUES[@]}
test_num=0

for max_req in "${MAX_REQ_VALUES[@]}"; do
    test_num=$((test_num + 1))
    echo ""
    echo "=========================================="
    echo "Test $test_num/$total_tests: MAX_ONGOING_REQUESTS=$max_req"
    echo "=========================================="
    
    # Stop Ray
    stop_ray
    
    # Start Ray with new MAX_ONGOING_REQUESTS
    if ! start_ray "$max_req"; then
        echo "⚠️  Failed to start Ray, skipping..."
        continue
    fi
    
    # Run stress test
    run_test "$max_req"
done

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo "Results directory: $RESULTS_DIR"
echo "Summary: $SUMMARY_FILE"
echo ""
echo "Generating comparison..."
python3 << PYEOF
import json
from pathlib import Path

results_dir = Path("$RESULTS_DIR")
results = []

for result_file in sorted(results_dir.glob("stress_test_*.json")):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract MAX_ONGOING_REQUESTS from filename
        name = result_file.stem
        max_req = int(name.split('_')[-1].replace('req', ''))
        
        summary = data.get('summary', {})
        throughput = data.get('throughput_analysis', {})
        
        results.append({
            'max_ongoing': max_req,
            'overall': summary.get('overall_throughput_messages_per_second', 0),
            'peak': throughput.get('peak_throughput_messages_per_second', 0),
            'steady': throughput.get('steady_state_throughput_messages_per_second', 0),
            'time': summary.get('total_wall_clock_time', 0),
        })
    except Exception as e:
        print(f"Error loading {result_file}: {e}")

results.sort(key=lambda x: x['max_ongoing'])

print("\n" + "="*80)
print("MAX_ONGOING_REQUESTS COMPARISON (32 replicas)")
print("="*80)
print()
print(f"{'MAX_REQ':<10} {'Overall':<12} {'Peak':<12} {'Steady':<12} {'Time(s)':<10}")
print("-" * 80)
for r in results:
    print(f"{r['max_ongoing']:<10} {r['overall']:<12.2f} {r['peak']:<12.2f} {r['steady']:<12.2f} {r['time']:<10.1f}")

if results:
    best = max(results, key=lambda x: x['overall'])
    print()
    print("="*80)
    print(f"🏆 BEST: MAX_ONGOING_REQUESTS={best['max_ongoing']}")
    print(f"   Throughput: {best['overall']:.2f} msg/s")
    print("="*80)
PYEOF

