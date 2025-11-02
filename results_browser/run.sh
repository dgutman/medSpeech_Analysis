#!/bin/bash

# Script to manage the results browser
# Auto-detects current user's UID/GID for container user matching

set -e

# Auto-detect current user's UID/GID (unless already set)
if [ -z "$UID" ]; then
    export UID=$(id -u)
fi
if [ -z "$GID" ]; then
    export GID=$(id -g)
fi

echo "👤 Using UID=$UID, GID=$GID for container user"
echo ""

case "$1" in
    "up")
        echo "🚀 Starting results browser (data prep happens automatically)..."
        docker compose up results-browser
        ;;
    "dev")
        echo "🚀 Starting results browser in development mode (with bind mounts)..."
        docker compose --profile dev up results-browser-dev
        ;;
    "build")
        echo "🔨 Building application container (includes data prep)..."
        docker compose build results-browser
        ;;
    "rebuild")
        echo "🔨 Rebuilding application container (no cache)..."
        docker compose build --no-cache results-browser
        ;;
    "down")
        echo "🛑 Stopping results browser..."
        docker compose down
        ;;
    "clean")
        echo "🧹 Cleaning up containers and volumes..."
        docker compose down -v
        docker system prune -f
        ;;
    *)
        echo "Usage: $0 {up|dev|build|rebuild|down|clean}"
        echo ""
        echo "Commands:"
        echo "  up         - Start the application (data prep happens automatically)"
        echo "  dev        - Start in development mode (bind mounts, no rebuild needed)"
        echo "  build      - Build the application container (includes data prep)"
        echo "  rebuild    - Rebuild the application container (no cache)"
        echo "  down       - Stop the application"
        echo "  clean      - Clean up everything"
        echo ""
        echo "Typical workflow:"
        echo "  1. Development: ./run.sh dev (changes are live, no rebuild needed)"
        echo "  2. Production:  ./run.sh up"
        exit 1
        ;;
esac
