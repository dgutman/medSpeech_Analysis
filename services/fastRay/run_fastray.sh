#!/bin/bash
set -e

# Configuration
IMAGE_NAME="fastray"
CONTAINER_NAME="fastray-container"
DOCKERFILE_DIR="/scr/dagutman/devel/medSpeech_Analysis/services/fastRay"

cd "$DOCKERFILE_DIR"

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

# Allow NUM_REPLICAS to be set via environment variable (default: 1)
NUM_REPLICAS=${NUM_REPLICAS:-1}
NUM_GPUS_PER_REPLICA=${NUM_GPUS_PER_REPLICA:-1}

docker run -d \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size=20g \
    -p 8000:8000 \
    -p 8265:8265 \
    -v /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats:/data \
    -v "$MODEL_CACHE_DIR:/root/.cache/huggingface" \
    -e RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -e NUM_REPLICAS="$NUM_REPLICAS" \
    -e NUM_GPUS_PER_REPLICA="$NUM_GPUS_PER_REPLICA" \
    "$IMAGE_NAME:latest"

echo ""
echo "=== Container started ==="
echo "API available at: http://localhost:8000"
echo "Ray Dashboard at: http://localhost:8265"
echo "Data directory mounted at: /data (host: /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats)"
echo "Model cache mounted at: $MODEL_CACHE_DIR (persists across restarts)"
echo ""
echo "GPU Configuration:"
echo "  - All 4 L40S GPUs are available to the container"
echo "  - Current: NUM_REPLICAS=$NUM_REPLICAS, NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA"
echo "  - To use all 4 GPUs, run: NUM_REPLICAS=4 NUM_GPUS_PER_REPLICA=1 bash run_fastray.sh"
echo "  - Or use 2 replicas with 2 GPUs each: NUM_REPLICAS=2 NUM_GPUS_PER_REPLICA=2 bash run_fastray.sh"
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"

