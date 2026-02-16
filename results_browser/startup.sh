#!/bin/bash

# Startup script for the Medical Speech Analysis Results Browser
echo "🚀 Starting Medical Speech Analysis Results Browser..."

# Load environment variables
# Docker-compose loads env_file into environment, but we also try to load from file if needed
# Check if variables are already set (from docker-compose env_file)
if [ -n "$PIXELTABLE_API_KEY" ] && [ -n "$PIXELTABLE_DATASET_URL" ]; then
    echo "📋 Using environment variables from docker-compose (env_file)"
elif [ -f "/app/.env" ]; then
    echo "📋 Loading environment variables from /app/.env"
    export $(cat /app/.env | grep -v '^#' | xargs)
elif [ -f "../.env" ]; then
    echo "📋 Loading environment variables from ../.env"
    export $(cat ../.env | grep -v '^#' | xargs)
elif [ -f ".env" ]; then
    echo "📋 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found and no env vars set, using defaults"
fi

# Set default values (will be overridden by environment variables from docker-compose)
export CACHE_DIR=${CACHE_DIR:-/var/lib/app/cache}
export DATA_DIR=${DATA_DIR:-/var/lib/app/data}
export PIXELTABLE_DATASET_URL=${PIXELTABLE_DATASET_URL:-pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare}

# Create directories (these are inside the container, not bind mounted)
# In dev mode, these might be bind-mounted, so ensure they exist and are writable
mkdir -p $CACHE_DIR $DATA_DIR 2>/dev/null || true
# Try to fix permissions if we can (won't work if bind-mounted, but won't hurt)
chmod 755 $CACHE_DIR $DATA_DIR 2>/dev/null || true

# Ensure home directory exists (Pixeltable uses ~/.pixeltable by default)
mkdir -p ~/.pixeltable

echo "📁 Cache directory: $CACHE_DIR"
echo "📁 Data directory: $DATA_DIR"
echo "📁 Pixeltable data: ~/.pixeltable (default)"
echo "🔗 Dataset URL: $PIXELTABLE_DATASET_URL"

# Data is preloaded during Docker build
echo "✅ Pixeltable will use local pgdata at: ${PIXELTABLE_PGDATA:-~/.pixeltable/pgdata}"

# Quick check for Pixeltable database readiness
echo "⏳ Verifying Pixeltable database..."
# Derive expected table name from dataset URL
TABLE_NAME=$(python -c "
import os
dataset_url = os.environ.get('PIXELTABLE_DATASET_URL', 'pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare')
if '/' in dataset_url:
    table_name = dataset_url.split('/')[-1]
    if ':' in table_name:
        table_name = table_name.split(':')[0]
    print(f'local_{table_name}')
else:
    print('local_transcribe_compare')
" 2>/dev/null || echo "local_transcribe_compare")

echo "   Expected table: $TABLE_NAME"

# Check if table exists (robust), if not, run preload
if timeout 30 python -c "
import os, json
# PIXELTABLE_PGDATA should already be set by docker-compose environment; do not override after import.
import pixeltable as pxt

expected = '$TABLE_NAME'
cache_dir = os.environ.get('CACHE_DIR', '/var/lib/app/cache')
meta_path = os.path.join(cache_dir, 'local_table_metadata.json')

try:
    tables = pxt.list_tables()
except Exception as e:
    print('ERROR', e)
    raise

# Prefer actual table name from metadata if present
if os.path.exists(meta_path):
    try:
        meta = json.load(open(meta_path))
        actual = meta.get('actual_table_name') or meta.get('local_table_name')
        if actual and actual in tables:
            print('EXISTS')
            raise SystemExit(0)
    except Exception:
        pass

# Exact expected name
if expected in tables:
    print('EXISTS')
    raise SystemExit(0)

# Heuristic: any local_* that looks like our dataset
candidates = [t for t in tables if t.startswith('local_')]
for t in candidates:
    low = t.lower()
    if 'transcribe' in low or 'compare' in low:
        print('EXISTS')
        raise SystemExit(0)

print('MISSING')
" 2>/dev/null | grep -q "EXISTS"; then
    echo "✅ Pixeltable local table exists"
else
    echo "⚠️  Pixeltable local table not found, running preload..."
    python preload_data.py
fi

# Start the application
echo "🌐 Starting web application on port 8050..."

# Check which app to run (simple or full). Fall back to full app if app_simple.py is missing.
APP_MODULE=${APP_MODULE:-app:server}
if [ "$USE_SIMPLE_APP" = "1" ] && [ -f "app_simple.py" ]; then
    APP_MODULE="app_simple:server"
    echo "📋 Using simplified app (fast table only)"
else
    echo "📋 Using full app (app:server)"
fi

# Check if FORCE_PRODUCTION_MODE is set, or if we're in development mode (bind mount exists)
if [ "$FORCE_PRODUCTION_MODE" = "1" ] || [ "$USE_GUNICORN" = "1" ]; then
    echo "🏭 Production mode (forced) - using Gunicorn"
    # Pixeltable: multiple PROCESSES are OK, multiple THREADS in the same process are NOT.
    # So we use --workers 1 (one process). Do not increase workers or use threaded workers.
    # Preload warms the app (and Pixeltable connection) once in the single process.
    exec gunicorn -b 0.0.0.0:8050 --workers 1 --timeout 120 --preload $APP_MODULE
elif [ -f "app.py" ] && [ -w "app.py" ]; then
    echo "🔄 Development mode detected - using Dash dev server"
    # Pixeltable: multiple threads in the same process are NOT supported. Keep threading off.
    export DASH_USE_RELOADER=${DASH_USE_RELOADER:-0}
    export DASH_THREADED=${DASH_THREADED:-0}
    exec python app.py
else
    echo "🏭 Production mode - using Gunicorn"
    # Pixeltable: use --workers 1 only; no threaded workers (multiple threads per process not supported).
    exec gunicorn -b 0.0.0.0:8050 --workers 1 --timeout 120 --preload $APP_MODULE
fi
