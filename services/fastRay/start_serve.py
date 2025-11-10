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

# Start Serve with HTTP host explicitly set to 0.0.0.0
# Check if Serve is already running
if serve.context._global_client is None:
    print("Starting Serve with HTTP host 0.0.0.0...")
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    time.sleep(1)
else:
    print("Serve is already running, reconfiguring HTTP host...")
    # Try to reconfigure - if it fails, that's okay, Serve is already running
    try:
        serve.shutdown()
        time.sleep(1)
        serve.start(http_options={"host": "0.0.0.0", "port": 8000})
        time.sleep(1)
    except:
        pass

# Deploy the application using serve.run()
# serve.run() should block and keep the process alive
print("Deploying application...")

# Import ray to ensure we're connected to the cluster
import ray
if not ray.is_initialized():
    ray.init(address="auto", ignore_reinit_error=True)

# Ensure we have a Serve client
if serve.context._global_client is None:
    print("Warning: Serve client is None, starting Serve again...")
    serve.start(http_options={"host": "0.0.0.0", "port": 8000})
    time.sleep(1)

# Now deploy - serve.run() should deploy and block
# If Serve is already started, serve.run() will just deploy and might return
# So we'll use it and then keep alive
try:
    serve.run(app)
    # If serve.run() returns (which it might if Serve is already started),
    # we need to keep the process alive
    print("serve.run() completed, keeping process alive...")
    while True:
        time.sleep(1)
except Exception as e:
    print(f"Error in serve.run(): {e}")
    import traceback
    traceback.print_exc()
    # If serve.run() exits, keep the process alive anyway
    print("serve.run() exited, but keeping process alive...")
    while True:
        time.sleep(1)

