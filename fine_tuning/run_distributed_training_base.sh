#!/bin/bash

# Distributed training launch script for 4 L40S GPUs - Whisper Base
# Usage: ./run_distributed_training_base.sh [--clear-cache] [--epochs N] [--batch-size N] [--learning-rate LR] [--output-dir DIR]

echo "Starting distributed Whisper Base fine-tuning on 4 L40S GPUs..."

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
    --master_port=29502 \
    fine_tuner_base.py "$@"

echo "Distributed training completed!"



