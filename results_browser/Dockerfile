FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY results_browser/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY results_browser/ .

# Copy fine-tuning results data (pre-prepared JSON files only)
# The fine_tuning_data/ directory contains only the necessary files (baseline_wer.txt and trainer_state.json)
COPY results_browser/fine_tuning_data/ /app/fine_tuning_data/

# Create cache directory
RUN mkdir -p /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV CACHE_DIR=/tmp
ENV DATA_DIR=/tmp

# Accept build arguments for API key and user IDs
ARG PIXELTABLE_API_KEY
ARG PIXELTABLE_DATASET_URL=pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare
ARG UID=1002
ARG GID=1002

# Set environment variables for build
ENV PIXELTABLE_API_KEY=${PIXELTABLE_API_KEY}
ENV PIXELTABLE_DATASET_URL=${PIXELTABLE_DATASET_URL}

# Create a non-root user with the same UID as the host user to avoid permission issues
RUN groupadd -r -g ${GID} appuser && useradd -r -u ${UID} -g appuser -m appuser

# Create cache and data directories inside container (not bind mounted)
RUN mkdir -p /var/lib/app/cache /var/lib/app/data && \
    chown -R appuser:appuser /var/lib/app

# Create cache and data directories in /app for backwards compatibility
# These need to be writable by appuser even when bind-mounted in dev mode
RUN mkdir -p /app/cache /app/data && \
    chown -R appuser:appuser /app/cache /app/data && \
    chmod -R 755 /app/cache /app/data && \
    chown -R appuser:appuser /app && \
    chmod -R u+r /app/fine_tuning_data 2>/dev/null || true

# Ensure appuser home directory exists (Pixeltable will use ~/.pixeltable by default)
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

# Note: .env file is NOT copied during build
# - In dev mode: docker-compose mounts ../.env:/app/.env at runtime
# - Build args (PIXELTABLE_API_KEY, PIXELTABLE_DATASET_URL) are passed from docker-compose
# - Runtime: env_file and volume mount ensure .env is available

# Switch to appuser before data loading
USER appuser

# Preload data as appuser (so pixeltable data is owned by appuser)
# This pulls data into the container during build so it's available on every start
# If preload fails, the build will fail (this is critical for the container to work)
RUN python preload_data.py

# Precompute analytics tab (WER + violin + summary table) into viz cache so the tab loads instantly
RUN CACHE_DIR=/var/lib/app/cache python warm_analytics_cache.py

# Make startup script executable
RUN chmod +x startup.sh

# Run the application
CMD ["./startup.sh"]
