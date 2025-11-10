#!/bin/bash
# Script to run stress test with GPU monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
MAX_WORKERS="${MAX_WORKERS:-4}"  # Number of parallel requests (should match NUM_REPLICAS)
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1.0}"  # GPU monitoring interval in seconds

echo "=== Stress Test Configuration ==="
echo "API URL: $API_URL"
echo "Max Workers: $MAX_WORKERS"
echo "GPU Monitor Interval: ${MONITOR_INTERVAL}s"
echo ""

# Check if API is accessible
echo "Checking if API is accessible..."
if ! curl -s "$API_URL/docs" > /dev/null 2>&1; then
    echo "ERROR: API is not accessible at $API_URL"
    echo "Make sure the container is running: docker ps | grep fastray"
    exit 1
fi
echo "✓ API is accessible"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
python3 monitor_gpu.py "$MONITOR_INTERVAL" > gpu_monitor.log 2>&1 &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID"
echo ""

# Run stress test
echo "Starting stress test..."
echo ""

python3 stress_test.py

# Stop GPU monitoring
echo ""
echo "Stopping GPU monitoring..."
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

echo ""
echo "=== Stress Test Complete ==="
echo "Results saved to: stress_test_results.json"
echo "GPU monitoring log: gpu_monitor.log"
echo "GPU monitoring data: gpu_monitoring.json"

