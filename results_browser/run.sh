#!/bin/bash

# Script to manage the results browser
# Auto-detects current user's UID/GID for container user matching

set -e

# Load environment variables for BOTH runtime and build args.
# docker-compose `env_file:` only affects runtime environment; build args come from the host env.
# We keep the project-wide secrets in ../.env, so source it here to ensure builds can preload Pixeltable.
if [ -f "../.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "../.env"
    set +a
fi

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
        shift  # Remove "up" from arguments
        docker compose up results-browser "$@"
        ;;
    "dev")
        echo "🚀 Starting results browser in development mode (with bind mounts)..."
        shift  # Remove "dev" from arguments
        docker compose --profile dev up results-browser-dev "$@"
        ;;
    "build")
        echo "🔨 Building application container (includes data prep)..."
        # Build both targets: production and dev use the same Dockerfile but can produce distinct images.
        docker compose build results-browser results-browser-dev
        ;;
    "rebuild")
        echo "🔨 Rebuilding application container (no cache)..."
        # Rebuild both so `./run.sh dev` doesn't keep running an old image (e.g., pixeltable==0.5.0)
        docker compose build --no-cache results-browser results-browser-dev
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
    "bash")
        echo "🐚 Starting container in bash mode for debugging..."
        shift  # Remove "bash" from arguments
        # Check if user wants dev mode (bash-dev)
        if [ "$1" = "dev" ]; then
            shift
            docker compose --profile dev run --rm --entrypoint /bin/bash results-browser-dev "$@"
        else
            docker compose run --rm --entrypoint /bin/bash results-browser "$@"
        fi
        ;;
    *)
        echo "Usage: $0 {up|dev|build|rebuild|down|clean|bash}"
        echo ""
        echo "Commands:"
        echo "  up         - Start the application (data prep happens automatically)"
        echo "  dev        - Start in development mode (bind mounts, no rebuild needed)"
        echo "  build      - Build the application container (includes data prep)"
        echo "  rebuild    - Rebuild the application container (no cache)"
        echo "  down       - Stop the application"
        echo "  clean      - Clean up everything"
        echo "  bash       - Start container in bash mode for debugging"
        echo "  bash dev   - Start dev container in bash mode for debugging"
        echo ""
        echo "Typical workflow:"
        echo "  1. Development: ./run.sh dev (changes are live, no rebuild needed)"
        echo "     Detached:    ./run.sh dev -d"
        echo "  2. Production:  ./run.sh up"
        echo "     Detached:    ./run.sh up -d"
        exit 1
        ;;
esac
