#!/bin/bash

# Startup script for the Medical Speech Analysis Results Browser
echo "🚀 Starting Medical Speech Analysis Results Browser..."

# Load environment variables
if [ -f "../.env" ]; then
    echo "📋 Loading environment variables from ../.env"
    export $(cat ../.env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found, using defaults"
fi

# Set default values (will be overridden by environment variables from docker-compose)
export CACHE_DIR=${CACHE_DIR:-/var/lib/app/cache}
export DATA_DIR=${DATA_DIR:-/var/lib/app/data}
export PIXELTABLE_DATASET_URL=${PIXELTABLE_DATASET_URL:-pxt://speech-to-text-analytics:main/hani89_asr_dataset}

# Create directories (these are inside the container, not bind mounted)
mkdir -p $CACHE_DIR $DATA_DIR

# Ensure home directory exists (Pixeltable uses ~/.pixeltable by default)
mkdir -p ~/.pixeltable

echo "📁 Cache directory: $CACHE_DIR"
echo "📁 Data directory: $DATA_DIR"
echo "📁 Pixeltable data: ~/.pixeltable (default)"
echo "🔗 Dataset URL: $PIXELTABLE_DATASET_URL"

# Data is preloaded during Docker build
echo "✅ Using Pixeltable data from container image"

# Wait for Pixeltable PostgreSQL to be ready
echo "⏳ Waiting for Pixeltable database to be ready..."
max_attempts=30
attempt=0
while [ $attempt -lt $max_attempts ]; do
    if python -c "import pixeltable as pxt; pxt.get_table('local_hani89'); print('Database ready')" 2>/dev/null; then
        echo "✅ Pixeltable database is ready"
        break
    fi
    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - waiting for database..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "⚠️  Warning: Database may not be fully ready, but proceeding anyway"
fi

# Start the application
echo "🌐 Starting web application on port 8050..."

# Check if we're in development mode (bind mount exists)
if [ -f "app.py" ] && [ -w "app.py" ]; then
    echo "🔄 Development mode detected - using Dash dev server with hot reload"
    exec python app.py
else
    echo "🏭 Production mode - using Gunicorn"
    # Use fewer workers and preload app to reduce concurrent database connections
    exec gunicorn -b 0.0.0.0:8050 --workers 1 --timeout 120 --preload app:server
fi
