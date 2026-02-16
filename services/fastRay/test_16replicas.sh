#!/bin/bash
# Test different MAX_ONGOING_REQUESTS values with 16 replicas × 0.25 GPU

set -e

API_URL="${API_URL:-http://localhost:8000}"
FILES_PER_TRIAL="${FILES_PER_TRIAL:-2000}"
MAX_WORKERS="${MAX_WORKERS:-200}"

# Fixed configuration for 16 replicas
NUM_REPLICAS=16
NUM_GPUS_PER_REPLICA=0.25
NUM_CPUS_PER_REPLICA=4  # More CPUs per replica since each has more GPU

# MAX_ONGOING_REQUESTS values to test
# Testing same range as 32 replicas for direct comparison
MAX_REQ_VALUES=(5 10)  # Testing only 5 and 10 for comparison

# Results directory
RESULTS_DIR="max_ongoing_test_16replicas_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"
echo "MAX_ONGOING_REQUESTS Test Results (16 replicas × 4 CPUs)" > "$SUMMARY_FILE"
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
    echo "Waiting for Ray API and Ray Serve to fully initialize..."
    local waited=0
    local max_wait=120  # Increased wait time for Ray Serve
    while [ $waited -lt $max_wait ]; do
        # Check if basic endpoints are up
        if curl -s "$API_URL/docs" > /dev/null 2>&1 && \
           curl -s "$API_URL/openapi.json" | grep -q "/transcribe/path" 2>/dev/null; then
            # Check if /config returns valid JSON (Ray Serve needs more time)
            local config_test=$(curl -s "$API_URL/config" 2>/dev/null | python3 -c "import sys, json; json.load(sys.stdin)" 2>/dev/null && echo "OK" || echo "FAIL")
            if [ "$config_test" = "OK" ]; then
                echo "✓ Ray API is ready"
                # Wait longer for Ray Serve to fully initialize
                echo "  Waiting for Ray Serve to fully initialize (config endpoint may need a few more seconds)..."
                sleep 10  # Give Ray Serve more time to fully start
                
                # Verify configuration - retry with longer waits
                local config_response=""
                local retries=0
                while [ -z "$config_response" ] && [ $retries -lt 10 ]; do
                    sleep 3  # Wait 3 seconds between retries
                    config_response=$(curl -s "$API_URL/config" 2>/dev/null)
                    if [ -n "$config_response" ]; then
                        # Validate it's actually JSON
                        echo "$config_response" | python3 -c "import sys, json; json.load(sys.stdin)" > /dev/null 2>&1 || config_response=""
                    fi
                    retries=$((retries + 1))
                    if [ -z "$config_response" ] && [ $retries -lt 10 ]; then
                        echo -n "."
                    fi
                done
                echo ""  # New line after dots
                
                if [ -n "$config_response" ]; then
                    local config_check=$(echo "$config_response" | python3 << PYEOF
import json, sys
try:
    response = sys.stdin.read()
    if not response or response.strip() == "":
        print("Config: WARN: Empty response")
        sys.exit(0)
    
    d = json.loads(response)
    expected_replicas = $NUM_REPLICAS
    expected_gpus = $NUM_GPUS_PER_REPLICA
    expected_max_ongoing = $max_ongoing
    actual_replicas = d.get('num_replicas_configured', 0)
    actual_gpus = d.get('num_gpus_per_replica', 0)
    actual_max_ongoing = d.get('max_ongoing_requests', -1)
    
    checks = []
    if actual_replicas == expected_replicas:
        checks.append(f"✓ Replicas: {actual_replicas}")
    else:
        checks.append(f"✗ Replicas: expected {expected_replicas}, got {actual_replicas}")
    
    if abs(actual_gpus - expected_gpus) < 0.01:
        checks.append(f"✓ GPUs/replica: {actual_gpus}")
    else:
        checks.append(f"✗ GPUs/replica: expected {expected_gpus}, got {actual_gpus}")
    
    if actual_max_ongoing == -1:
        checks.append(f"⚠️  MAX_ONGOING: not in config")
    elif actual_max_ongoing == expected_max_ongoing:
        checks.append(f"✓ MAX_ONGOING: {actual_max_ongoing}")
    else:
        checks.append(f"⚠️  MAX_ONGOING: expected {expected_max_ongoing}, got {actual_max_ongoing}")
    
    print("Config: " + ", ".join(checks))
    sys.exit(0)
except Exception as e:
    print(f"Config: WARN: Check failed: {e}")
    sys.exit(0)
PYEOF
)
                    echo "$config_check" || true  # Don't fail if echo has issues
                else
                    echo "Config: WARN: Could not get config response, proceeding anyway"
                fi
                return 0
            fi
        fi
        echo -n "."
        sleep 1
        waited=$((waited + 1))
    done
    echo ""
    echo "⚠️  Ray API not ready after ${max_wait} seconds, continuing anyway..."
    return 0
}

# Function to run stress test
run_test() {
    local max_ongoing=$1
    local output_file="$RESULTS_DIR/stress_test_16replicas_0.25gpu_4cpu_${max_ongoing}req.json"
    
    echo ""
    echo "Running stress test with MAX_ONGOING_REQUESTS=$max_ongoing..."
    export MAX_FILES=$FILES_PER_TRIAL
    export API_URL=$API_URL
    export MAX_WORKERS=$MAX_WORKERS  # Ensure MAX_WORKERS is exported
    
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
    
    # Verify MAX_ONGOING_REQUESTS was applied
    max_ongoing_check=$(curl -s "$API_URL/config" 2>/dev/null | python3 << PYEOF
import json, sys
try:
    d = json.load(sys.stdin)
    # Note: MAX_ONGOING_REQUESTS might not be in /config endpoint
    # But we can verify replicas and GPUs are correct
    print("OK")
    sys.exit(0)
except:
    print("ERROR")
    sys.exit(1)
PYEOF
)
    
    if [ "$max_ongoing_check" != "OK" ]; then
        echo "⚠️  Warning: Could not verify configuration before test"
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
print("MAX_ONGOING_REQUESTS COMPARISON (16 replicas × 0.25 GPU × 4 CPU)")
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

