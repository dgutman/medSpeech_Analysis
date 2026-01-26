#!/bin/bash

# Distributed training launch script for 4 L40S GPUs - Whisper Medium
# Usage: ./run_distributed_training_medium.sh [--clear-cache] [--epochs N] [--batch-size N] [--learning-rate LR] [--output-dir DIR]

echo "Starting distributed Whisper Medium fine-tuning on 4 L40S GPUs..."

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Pass all command line arguments to the Python script
echo "Arguments passed: $@"

# Launch distributed training with all arguments
torchrun \
    --nproc_per_node=4 \
    --master_port=29503 \
    fine_tuner_medium.py "$@"

echo "Distributed training completed!"
