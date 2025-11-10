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

# Create cache directory
RUN mkdir -p /app/cache

# Set environment variables
ENV PYTHONPATH=/app
ENV CACHE_DIR=/app/cache

# Accept build arguments for API key and user IDs
ARG PIXELTABLE_API_KEY
ARG PIXELTABLE_DATASET_URL=pxt://speech-to-text-analytics:main/hani89_asr_dataset
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

# Create cache directory in /app for backwards compatibility (though we use /var/lib/app)
RUN mkdir -p /app/cache && chown -R appuser:appuser /app

# Ensure appuser home directory exists (Pixeltable will use ~/.pixeltable by default)
RUN mkdir -p /home/appuser && chown -R appuser:appuser /home/appuser

# Copy .env file for data preparation
COPY .env /app/.env

# Switch to appuser before data loading
USER appuser

# Preload data as appuser (so pixeltable data is owned by appuser)
# This pulls data into the container during build so it's available on every start
RUN python preload_data.py || echo "Warning: Data preload failed, will retry at runtime"

# Make startup script executable
RUN chmod +x startup.sh

# Run the application
CMD ["./startup.sh"]