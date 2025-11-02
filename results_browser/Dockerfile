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

# Accept build arguments for API key
ARG PIXELTABLE_API_KEY
ARG PIXELTABLE_DATASET_URL=pxt://speech-to-text-analytics:main/hani89_asr_dataset

# Set environment variables for build
ENV PIXELTABLE_API_KEY=${PIXELTABLE_API_KEY}
ENV PIXELTABLE_DATASET_URL=${PIXELTABLE_DATASET_URL}

# Create a non-root user with the same UID as the host user to avoid permission issues
RUN groupadd -r -g 1002 appuser && useradd -r -u 1002 -g appuser -m appuser

# Create cache directory and set permissions
RUN mkdir -p /app/cache && chown -R appuser:appuser /app

# Copy .env file for data preparation
COPY .env /app/.env

# Switch to appuser before data loading
USER appuser

# Preload data as appuser (so pixeltable data is owned by appuser)
RUN python preload_data.py

# Make startup script executable
RUN chmod +x startup.sh

# Run the application
CMD ["./startup.sh"]