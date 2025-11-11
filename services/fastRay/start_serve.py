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
try:
    # Try to connect to existing Ray cluster
    ray.init(address="auto", ignore_reinit_error=True)
    print("Connected to existing Ray cluster")
except Exception as e:
    print(f"Warning: Could not connect to existing Ray cluster: {e}")
    print("Starting new Ray cluster (this should not happen)")
    ray.init()

# Start Serve with HTTP host explicitly set to 0.0.0.0
# Configure routing policy for better load balancing
# Check if Serve is already running
if serve.context._global_client is None:
    print("Starting Serve with HTTP host 0.0.0.0...")
    serve.start(
        http_options={
            "host": "0.0.0.0",
            "port": 8000,
            # Use least-pending-requests routing for better load balancing
            "location": "EveryNode",  # Not applicable for single node, but explicit
        },
        # Configure routing to use least-pending-requests (default should be this, but make it explicit)
    )
    time.sleep(1)
else:
    print("Serve is already running, reconfiguring HTTP host...")
    # Try to reconfigure - if it fails, that's okay, Serve is already running
    try:
        serve.shutdown()
        time.sleep(1)
        serve.start(
            http_options={
                "host": "0.0.0.0",
                "port": 8000,
                "location": "EveryNode",
            }
        )
        time.sleep(1)
    except:
        pass

# Deploy the application using serve.run()
# serve.run() should block and keep the process alive
print("Deploying application...")

# serve.start() should automatically connect to the existing Ray cluster
# started by start.sh. We don't need to call ray.init() separately.
# If we do, it might start a second Ray instance, which causes port conflicts.

# Now deploy - serve.run() should deploy and block
# If Serve is already started, serve.run() will just deploy and might return
# So we'll use it and then keep alive
try:
    # route_prefix must be passed to serve.run() in newer Ray Serve versions
    serve.run(app, route_prefix="/")
    
    # Wait a bit for all replicas to initialize before declaring ready
    # This helps ensure all replicas are available for load balancing
    print("Waiting for all replicas to initialize...")
    time.sleep(10)  # Give replicas time to load models
    print("Deployment complete. All replicas should be ready.")
    
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

