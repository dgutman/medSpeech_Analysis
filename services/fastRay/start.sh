#!/bin/bash
set -e

# Start Ray head in the background with dashboard on 0.0.0.0
# Ray will automatically detect all available GPUs (all 4 L40S GPUs)
# Configure object store with reasonable memory limit (20GB) and temp directory
ray start --head \
    --dashboard-host=0.0.0.0 \
    --object-store-memory=20000000000 \
    --temp-dir=/dev/shm/ray \
    --block &

# Wait for Ray to be fully ready
sleep 3

# Start Ray Serve using Python script in the background
# This will deploy the application and keep Serve running
python3 start_serve.py &

# Keep the container alive forever
# Use tail -f /dev/null which will never exit
tail -f /dev/null

