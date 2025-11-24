#!/bin/bash
set -e

# Configuration
IMAGE_NAME="fastray"
CONTAINER_NAME="fastray-container"
DOCKERFILE_DIR="/scr/dagutman/devel/medSpeech_Analysis/services/fastRay"

cd "$DOCKERFILE_DIR"

# Load .env file if it exists (allows configuration via .env file)
if [ -f .env ]; then
    echo "Loading configuration from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a  # stop automatically exporting
    echo "  NUM_REPLICAS=$NUM_REPLICAS"
    echo "  NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA"
    echo "  NUM_CPUS_PER_REPLICA=${NUM_CPUS_PER_REPLICA:-2}"
    echo ""
fi

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

# NUM_REPLICAS, NUM_GPUS_PER_REPLICA, and NUM_CPUS_PER_REPLICA can be set via:
# 1. .env file (recommended - copy .env.example to .env)
# 2. Environment variables (overrides .env)
# 3. Defaults to 1 replica if neither is set
NUM_REPLICAS=${NUM_REPLICAS:-1}
NUM_GPUS_PER_REPLICA=${NUM_GPUS_PER_REPLICA:-1}
NUM_CPUS_PER_REPLICA=${NUM_CPUS_PER_REPLICA:-2}

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
    -v /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats:/data \
    -v /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray:/data_medspeech \
    -v "$MODEL_CACHE_DIR:/root/.cache/huggingface" \
    -e RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1 \
    -e HF_HOME=/root/.cache/huggingface \
    -e NUM_REPLICAS="$NUM_REPLICAS" \
    -e NUM_GPUS_PER_REPLICA="$NUM_GPUS_PER_REPLICA" \
    -e NUM_CPUS_PER_REPLICA="$NUM_CPUS_PER_REPLICA" \
    "$IMAGE_NAME:latest"

echo ""
echo "=== Container started ==="
echo "API available at: http://localhost:8000"
echo "Ray Dashboard at: http://localhost:8265 (or http://localhost:8266 if 8265 is in use)"
echo "Data directories mounted:"
echo "  - /data (host: /scr/dagutman/devel/medSpeech_Analysis/eleven_octo_cats)"
echo "  - /data_medspeech (host: /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray)"
echo "Model cache mounted at: $MODEL_CACHE_DIR (persists across restarts)"
echo ""
echo "GPU/CPU Configuration:"
echo "  - All 4 L40S GPUs are available to the container (~46GB each)"
echo "  - Current: NUM_REPLICAS=$NUM_REPLICAS, NUM_GPUS_PER_REPLICA=$NUM_GPUS_PER_REPLICA, NUM_CPUS_PER_REPLICA=$NUM_CPUS_PER_REPLICA"
echo ""
echo "Recommended configurations (each replica uses ~3-4GB GPU memory):"
echo "  - 1 replica per GPU (4 total):  NUM_REPLICAS=4  NUM_GPUS_PER_REPLICA=1"
echo "  - 2 replicas per GPU (8 total): NUM_REPLICAS=8  NUM_GPUS_PER_REPLICA=0.5"
echo "  - 4 replicas per GPU (16 total): NUM_REPLICAS=16 NUM_GPUS_PER_REPLICA=0.25"
echo "  - 6 replicas per GPU (24 total): NUM_REPLICAS=24 NUM_GPUS_PER_REPLICA=0.166"
echo ""
echo "Example: NUM_REPLICAS=16 NUM_GPUS_PER_REPLICA=0.25 bash run_fastray.sh"
echo ""
echo "To view logs: docker logs -f $CONTAINER_NAME"
echo "To stop: docker stop $CONTAINER_NAME"

