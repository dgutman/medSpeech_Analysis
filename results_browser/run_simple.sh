#!/bin/bash
# Run the simplified version of the results browser

echo "🚀 Starting Simplified Results Browser..."
echo "   Using Gunicorn for best performance"

# Use Gunicorn with single worker and preload for global variable caching
gunicorn -b 0.0.0.0:8050 --workers 1 --timeout 120 --preload app_simple:server
