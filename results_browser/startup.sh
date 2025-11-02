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

# Set default values
export CACHE_DIR=${CACHE_DIR:-./cache}
export DATA_DIR=${DATA_DIR:-./data}
export PIXELTABLE_DATASET_URL=${PIXELTABLE_DATASET_URL:-pxt://speech-to-text-analytics:main/hani89_asr_dataset}

# Create directories
mkdir -p $CACHE_DIR $DATA_DIR

echo "📁 Cache directory: $CACHE_DIR"
echo "📁 Data directory: $DATA_DIR"
echo "🔗 Dataset URL: $PIXELTABLE_DATASET_URL"

# Data is preloaded in a separate data preparation step
echo "✅ Using preloaded dataset from persistent volume"

# Pixeltable will handle database connection automatically
echo "📊 Pixeltable will connect to local database automatically"

# Start the application
echo "🌐 Starting web application on port 8050..."

# Check if we're in development mode (bind mount exists)
if [ -f "app.py" ] && [ -w "app.py" ]; then
    echo "🔄 Development mode detected - using Dash dev server with hot reload"
    exec python app.py
else
    echo "🏭 Production mode - using Gunicorn"
    exec gunicorn -b 0.0.0.0:8050 app:server
fi
