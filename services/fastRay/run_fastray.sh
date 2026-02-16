#!/bin/bash
set -e

# Configuration
IMAGE_NAME="fastray"
CONTAINER_NAME="fastray-container"
DOCKERFILE_DIR="/scr/dagutman/devel/medSpeech_Analysis/services/fastRay"

cd "$DOCKERFILE_DIR"

# Save environment variables if they're already set (they take precedence over .env)
# This allows the benchmark script to override .env file values
_SAVED_NUM_REPLICAS="${NUM_REPLICAS:-}"
_SAVED_NUM_GPUS_PER_REPLICA="${NUM_GPUS_PER_REPLICA:-}"
_SAVED_NUM_CPUS_PER_REPLICA="${NUM_CPUS_PER_REPLICA:-}"
_SAVED_MAX_ONGOING_REQUESTS="${MAX_ONGOING_REQUESTS:-}"
_SAVED_ENABLE_AUTOSCALING="${ENABLE_AUTOSCALING:-}"

# Load .env file if it exists (allows configuration via .env file)
# NOTE: Environment variables override .env file values
if [ -f .env ]; then
    echo "Loading configuration from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
fi

# Restore environment variables if they were set (they override .env)
# This ensures environment variables from benchmark script take precedence
if [ -n "$_SAVED_NUM_REPLICAS" ]; then
    NUM_REPLICAS="$_SAVED_NUM_REPLICAS"
fi
if [ -n "$_SAVED_NUM_GPUS_PER_REPLICA" ]; then
    NUM_GPUS_PER_REPLICA="$_SAVED_NUM_GPUS_PER_REPLICA"
fi
if [ -n "$_SAVED_NUM_CPUS_PER_REPLICA" ]; then
    NUM_CPUS_PER_REPLICA="$_SAVED_NUM_CPUS_PER_REPLICA"
fi
if [ -n "$_SAVED_MAX_ONGOING_REQUESTS" ]; then
    MAX_ONGOING_REQUESTS="$_SAVED_MAX_ONGOING_REQUESTS"
fi
if [ -n "$_SAVED_ENABLE_AUTOSCALING" ]; then
    ENABLE_AUTOSCALING="$_SAVED_ENABLE_AUTOSCALING"
fi

# Display final configuration
echo "Final configuration:"
echo "  NUM_REPLICAS=$NUM_REPLICAS"
echo "  NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA"
echo "  NUM_CPUS_PER_REPLICA=${NUM_CPUS_PER_REPLICA:-2}"
echo "  MAX_ONGOING_REQUESTS=${MAX_ONGOING_REQUESTS:-30}"
echo ""

echo "=== Stopping existing container ==="
docker stop "$CONTAINER_NAME" 2>/dev/null || echo "No container to stop"
docker rm "$CONTAINER_NAME" 2>/dev/null || echo "No container to remove"

echo ""
echo "=== Building Docker image ==="
# Enable BuildKit for cache mount support
DOCKER_BUILDKIT=1 docker build -t "$IMAGE_NAME:latest" .

echo ""
echo "=== Starting container ==="
# Create model cache directory if it doesn't exist
MODEL_CACHE_DIR="/scr/dagutman/devel/medSpeech_Analysis/services/fastRay/model_cache"
mkdir -p "$MODEL_CACHE_DIR" 2>/dev/null || true

# Set defaults if not already set
NUM_REPLICAS=${NUM_REPLICAS:-1}
NUM_GPUS_PER_REPLICA=${NUM_GPUS_PER_REPLICA:-1}
NUM_CPUS_PER_REPLICA=${NUM_CPUS_PER_REPLICA:-2}
MAX_ONGOING_REQUESTS=${MAX_ONGOING_REQUESTS:-30}
ENABLE_AUTOSCALING=${ENABLE_AUTOSCALING:-false}

if [ "$NUM_REPLICAS" = "1" ] && [ "$NUM_GPUS_PER_REPLICA" = "1" ]; then
    echo "⚠️  WARNING: Using default configuration (1 replica, 1 GPU per replica)"
    echo "   This is likely not optimal. Create a .env file or set environment variables."
    echo "   See .env.example for an example configuration."
    echo ""
fi

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=20g \
    -p 8000:8000 \
    -p 8265:8265 \
    -p 8266:8266 \
    -v /scr/dagutman/devel/medSpeech_Analysis:/data \
    -v /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats:/data/eleven_octo_cats \
    -v /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray:/data_medspeech \
    -v "$MODEL_CACHE_DIR:/root/.cache/huggingface" \
    -e RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -e NUM_REPLICAS="$NUM_REPLICAS" \
    -e NUM_GPUS_PER_REPLICA="$NUM_GPUS_PER_REPLICA" \
    -e NUM_CPUS_PER_REPLICA="$NUM_CPUS_PER_REPLICA" \
    -e MAX_ONGOING_REQUESTS="$MAX_ONGOING_REQUESTS" \
    -e ENABLE_AUTOSCALING="$ENABLE_AUTOSCALING" \
    "$IMAGE_NAME:latest"

echo ""
echo "=== Container started ==="
echo "API available at: http://localhost:8000"
echo "Ray Dashboard at: http://localhost:8265 (or http://localhost:8266 if 8265 is in use)"
echo "Data directories mounted:"
echo "  - /data (host: /scr/dagutman/devel/medSpeech_Analysis - includes train_data/ and test_data/)"
echo "  - /data/eleven_octo_cats (host: /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats)"
echo "  - /data_medspeech (host: /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray)"
echo "Model cache mounted at: $MODEL_CACHE_DIR (persists across restarts)"
echo ""
echo "GPU/CPU Configuration:"
echo "  - All 4 L40S GPUs are available to the container (~46GB each)"
echo "  - Current settings:"
echo "    NUM_REPLICAS=$NUM_REPLICAS"
echo "    NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA"
echo "    NUM_CPUS_PER_REPLICA=$NUM_CPUS_PER_REPLICA"
echo "    MAX_ONGOING_REQUESTS=$MAX_ONGOING_REQUESTS"
echo "    ENABLE_AUTOSCALING=$ENABLE_AUTOSCALING"
echo ""
echo "=== Performance Tuning Guide ==="
echo ""
echo "1. REPLICA CONFIGURATION (each replica uses ~3-4GB GPU memory):"
echo "   - 1 replica per GPU (4 total):  NUM_REPLICAS=4  NUM_GPUS_PER_REPLICA=1"
echo "   - 2 replicas per GPU (8 total): NUM_REPLICAS=8  NUM_GPUS_PER_REPLICA=0.5"
echo "   - 4 replicas per GPU (16 total): NUM_REPLICAS=16 NUM_GPUS_PER_REPLICA=0.25"
echo "   - 6 replicas per GPU (24 total): NUM_REPLICAS=24 NUM_GPUS_PER_REPLICA=0.166"
echo ""
echo "2. CPU PER REPLICA (for audio decoding/preprocessing):"
echo "   - Default: NUM_CPUS_PER_REPLICA=2 (good for most cases)"
echo "   - For heavy I/O: NUM_CPUS_PER_REPLICA=4 (more parallel file reads)"
echo "   - For CPU-bound preprocessing: NUM_CPUS_PER_REPLICA=3-4"
echo ""
echo "3. MAX_ONGOING_REQUESTS (concurrent requests per replica):"
echo "   - Default: MAX_ONGOING_REQUESTS=30 (good I/O/GPU overlap)"
echo "   - For high throughput: MAX_ONGOING_REQUESTS=50-100"
echo "   - For low latency: MAX_ONGOING_REQUESTS=10-20"
echo "   - Higher = more throughput but more GPU memory per replica"
echo ""
echo "4. RECOMMENDED STARTING POINT (4 GPUs, balanced):"
echo "   NUM_REPLICAS=16 NUM_GPUS_PER_REPLICA=0.25 NUM_CPUS_PER_REPLICA=2 MAX_ONGOING_REQUESTS=30"
echo ""
echo "5. FOR MAXIMUM THROUGHPUT:"
echo "   NUM_REPLICAS=24 NUM_GPUS_PER_REPLICA=0.166 NUM_CPUS_PER_REPLICA=3 MAX_ONGOING_REQUESTS=50"
echo ""
echo "Example usage:"
echo "  NUM_REPLICAS=16 NUM_GPUS_PER_REPLICA=0.25 NUM_CPUS_PER_REPLICA=2 MAX_ONGOING_REQUESTS=30 bash run_fastray.sh"
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"

