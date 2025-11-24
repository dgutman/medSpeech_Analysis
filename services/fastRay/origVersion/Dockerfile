FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-dev ffmpeg curl ca-certificates \
    build-essential \
    pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libswscale-dev libswresample-dev libavfilter-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install uv and use it in the same RUN to ensure PATH is available
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    export PATH="/root/.local/bin:$PATH" && \
    uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 && \
    uv pip install --system -r requirements.txt

ENV PATH="/root/.local/bin:$PATH"

COPY app.py start.sh start_serve.py preload_model.py .
RUN chmod +x start.sh start_serve.py preload_model.py

# Pre-download the Whisper model during build to avoid downloads at runtime
# Use BuildKit cache mount to persist the HuggingFace cache between builds
# This way the model only downloads once and is cached for subsequent builds
RUN --mount=type=cache,target=/root/.cache/huggingface \
    python3 preload_model.py || echo "Model preload failed, will download at runtime"

# Ray Serve listens on 8000, Ray dashboard on 8265 (or 8266 if 8265 is in use)
EXPOSE 8000 8265 8266

# Environment knobs
# L40S GPUs: Optimized for Ada Lovelace architecture (compute capability 8.9)
# NUM_REPLICAS and NUM_GPUS_PER_REPLICA should be set at runtime via docker run -e
# Examples:
#   - 1 replica per GPU (4 total):  NUM_REPLICAS=4  NUM_GPUS_PER_REPLICA=1
#   - 10 replicas per GPU (40 total): NUM_REPLICAS=40 NUM_GPUS_PER_REPLICA=0.1
ENV WHISPER_MODEL=large-v3 \
    COMPUTE_TYPE=float16 \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH} \
    CUDA_HOME=/usr/local/cuda \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_ARCH_LIST="8.9"

# Start Ray head locally and run Serve
CMD ["./start.sh"]
