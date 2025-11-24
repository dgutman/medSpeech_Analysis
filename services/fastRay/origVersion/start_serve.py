#!/usr/bin/env python3
"""Start Ray Serve with HTTP host bound to 0.0.0.0."""
import os
import time
import signal
import sys
from ray import serve
from app import app

# Set HTTP host via environment variable before importing serve
os.environ["RAY_SERVE_HTTP_HOST"] = "0.0.0.0"
os.environ["RAY_SERVE_HTTP_PORT"] = "8000"

# Signal handler for graceful shutdown
def signal_handler(sig, frame):
    print("\nReceived shutdown signal, shutting down Serve...")
    try:
        serve.shutdown()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

print("Starting Ray Serve and deploying application...")
print("Ray Serve will be available at http://0.0.0.0:8000 once ready")

# Connect to existing Ray cluster (started by start.sh)
# IMPORTANT: We must connect to the existing cluster, not start a new one
import ray

# Wait for Ray cluster to be fully ready
time.sleep(3)

# Connect to existing Ray cluster
# Use "auto" to find the cluster started by start.sh
try:
    if ray.is_initialized():
        print("Ray already initialized")
    else:
        print("Connecting to existing Ray cluster...")
        ray.init(address="auto", ignore_reinit_error=True)
        print("✓ Connected to Ray cluster")
except Exception as e:
    print(f"Error connecting to Ray cluster: {e}")
    import traceback
    traceback.print_exc()
    raise

# Start Serve with HTTP options first
# Then deploy using serve.run()
print("Starting Serve with HTTP host 0.0.0.0...")
if serve.context._global_client is None:
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
        }
    )
    time.sleep(1)
else:
    print("Serve already running")

# Deploy the application using serve.run()
# serve.run() deploys the app and blocks
print("Deploying application with serve.run()...")
try:
    # serve.run() deploys the app - route_prefix must be passed here
    serve.run(app, route_prefix="/")
    
    # serve.run() should block, but if it returns, keep alive
    print("serve.run() returned, keeping process alive...")
    while True:
        time.sleep(1)
except Exception as e:
    print(f"Error in serve.run(): {e}")
    import traceback
    traceback.print_exc()
    print("Keeping process alive despite error...")
    while True:
        time.sleep(1)
