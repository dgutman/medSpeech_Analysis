#!/usr/bin/env python3
"""Pre-download Whisper model to cache."""
import os
from faster_whisper import WhisperModel

# Get model name from environment or use default
model_name = os.environ.get("WHISPER_MODEL", "large-v3")
compute_type = os.environ.get("COMPUTE_TYPE", "float16")

print(f"Pre-downloading Whisper model: {model_name}")
print("This may take a few minutes on first run...")

# Download and cache the model (using CPU for download, faster)
# The model will be cached in ~/.cache/huggingface/hub/
model = WhisperModel(model_name, device="cpu", compute_type=compute_type)
print(f"✓ Model {model_name} downloaded and cached successfully!")


