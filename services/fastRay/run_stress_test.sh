#!/bin/bash
# Script to run stress test with GPU monitoring

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
# Default to 200 workers to fully saturate 40 replicas
# With 40 replicas × 1 concurrent = 40 concurrent capacity
# Requests complete in ~0.5s, so we need many workers to keep queue full
# More workers = more concurrent HTTP connections = better chance of hitting all replicas
MAX_WORKERS="${MAX_WORKERS:-200}"  # Number of parallel HTTP connections (5x replicas to saturate)
MONITOR_INTERVAL="${MONITOR_INTERVAL:-1.0}"  # GPU monitoring interval in seconds
MODEL="${MODEL:-}"  # Optional: model to use (e.g., "large-v3", "base", "tiny"). Empty = use server default

echo "=== Stress Test Configuration ==="
echo "API URL: $API_URL"
echo "Max Workers: $MAX_WORKERS"
echo "GPU Monitor Interval: ${MONITOR_INTERVAL}s"
if [ -n "$MODEL" ]; then
    echo "Model: $MODEL"
else
    echo "Model: (using server default)"
fi
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

# Function to cleanup GPU monitoring on exit
cleanup_monitor() {
    if [ -n "$MONITOR_PID" ]; then
        echo ""
        echo "Stopping GPU monitoring (PID: $MONITOR_PID)..."
        kill $MONITOR_PID 2>/dev/null || true
        wait $MONITOR_PID 2>/dev/null || true
    fi
}

# Set trap to cleanup on script exit (including Ctrl+C)
trap cleanup_monitor EXIT INT TERM

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
# Monitor GPU and save to log file only (not shown in terminal to reduce verbosity)
python3 monitor_gpu.py "$MONITOR_INTERVAL" > gpu_monitor.log 2>&1 &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID"
echo "GPU monitoring data saved to gpu_monitor.log (not shown in terminal)"
echo ""

# Run stress test
echo "Starting stress test..."
echo ""

# Export environment variables so stress_test.py can use them
export MAX_WORKERS
export MODEL

python3 stress_test.py

# Stop GPU monitoring (cleanup function will also handle this, but explicit is good)
cleanup_monitor

echo ""
echo "=== Stress Test Complete ==="
echo "Results saved to: stress_test_results.json"
echo "GPU monitoring log: gpu_monitor.log"
echo "GPU monitoring data: gpu_monitoring.json"

