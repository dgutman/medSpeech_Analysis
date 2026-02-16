#!/bin/bash
# Script to prepare fine-tuning data staging directory for Docker build
# Run this before building the Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STAGING_DIR="$PROJECT_ROOT/fine_tuning_data_staging"

echo "📦 Preparing fine-tuning data staging directory..."

# Remove old staging directory
rm -rf "$STAGING_DIR"

# Run the Python script to copy only necessary files
python3 "$SCRIPT_DIR/copy_fine_tuning_data.py" \
    "$PROJECT_ROOT/Fine-Tuning-Whisper-on-Custom-Dataset" \
    "$STAGING_DIR"

# Show size
if [ -d "$STAGING_DIR" ]; then
    SIZE=$(du -sh "$STAGING_DIR" | cut -f1)
    echo "✅ Staging directory created: $STAGING_DIR ($SIZE)"
    echo "   Ready for Docker build!"
else
    echo "❌ Failed to create staging directory"
    exit 1
fi



