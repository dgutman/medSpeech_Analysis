# Fine-Tuning Whisper Models on Medical Speech Data

This directory contains scripts and tools for fine-tuning Whisper models on the Hani89 medical speech dataset.

## 📁 Directory Structure

```
fine_tuning/
├── README.md                                  # This file
├── requirements.txt                           # Python dependencies
├── fine_tuner_*.py                           # Training scripts for each model size
├── run_distributed_training_*.sh             # Launch scripts for distributed training
├── benchmark_inference_optimizations.py       # Performance benchmarking
├── whisper-{size}_hani89/                    # Trained model checkpoints
└── wandb/                                    # Weights & Biases logs
```

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have the base environment set up:
```bash
# From project root
cd /scr/dgutman/devel/medSpeech_Analysis
source .venv/bin/activate  # or your virtual environment
```

### 2. Install Fine-Tuning Dependencies

```bash
cd fine_tuning
pip install -r requirements.txt
```

### 3. Configure Weights & Biases

Create a `.env` file in this directory:
```bash
# Weights & Biases API Key
WANDB_API_KEY=your_api_key_here

# Optional: Set wandb project name
WANDB_PROJECT=whisper-finetuning-hani89
```

## 📊 Available Models

Training scripts are available for all Whisper model sizes:

| Model | Script | Parameters | Best Use Case |
|-------|--------|------------|---------------|
| Tiny | `fine_tuner_orig_tiny.py` | 39M | Fast inference, low resource |
| Base | `fine_tuner_base.py` | 74M | Balanced speed/accuracy |
| Small | `fine_tune_whisper_small_medData.py` | 244M | Good accuracy, reasonable speed |
| Medium | `fine_tuner_medium.py` | 769M | High accuracy |
| Large | `fine_tuner_large.py` | 1.5B | Highest accuracy |
| Large-v3 | `fine_tuner_orig_large_v3.py` | 1.5B | Latest version, best quality |

## 🎯 Distributed Training (Recommended)

For multi-GPU training on 4× L40S GPUs:

### Basic Training
```bash
# Train Whisper-Small (recommended starting point)
./run_distributed_training.sh

# Train Whisper-Base
./run_distributed_training_base.sh

# Train Whisper-Medium
./run_distributed_training_medium.sh

# Train Whisper-Large
./run_distributed_training_large.sh

# Train Whisper-Large-v3
./run_distributed_training_large_v3.sh
```

### Advanced Options

```bash
# Clear cache and retrain from scratch
./run_distributed_training.sh --clear-cache

# Custom epochs and batch size
./run_distributed_training.sh --epochs 20 --batch-size 32

# Different learning rate
./run_distributed_training.sh --learning-rate 0.00002

# Custom output directory
./run_distributed_training.sh --output-dir my_experiment

# Combine multiple options
./run_distributed_training.sh \
    --clear-cache \
    --epochs 15 \
    --batch-size 16 \
    --learning-rate 0.00002 \
    --output-dir experiment_1
```

## 📈 Training Monitoring

### Weights & Biases Dashboard

Access your training metrics at: https://wandb.ai

Tracked metrics include:
- Training loss (every 100 steps)
- Word Error Rate / WER (every 500 steps)
- Learning rate schedule
- Gradient norms
- Sample predictions
- GPU utilization
- Memory usage

### Local Logs

Training logs are saved to:
- `{model}_training.log` - Main training log
- `wandb/` - Local W&B cache

## 💾 Trained Models

Fine-tuned models are saved to:
```
fine_tuning/
├── whisper-tiny_hani89/
├── whisper-base_hani89/
├── whisper-small_hani89/
├── whisper-medium_hani89/
├── whisper-large_hani89/
└── whisper-large-v3_hani89/
```

Each directory contains:
- `best_model/` - Best checkpoint based on WER
- `checkpoint-*/` - Intermediate checkpoints
- `cache_*.arrow` - Preprocessed data cache

## 🔧 Configuration

### Caching System

The training scripts use intelligent caching:
- **First run**: ~2-3 minutes preprocessing
- **Subsequent runs**: ~10-30 seconds (loads from cache)
- Cache files: `cache_train_*.arrow`, `cache_valid_*.arrow`

### GPU Optimization

Optimized for 4× L40S GPUs:
- Distributed Data Parallel (DDP)
- Mixed Precision (bfloat16)
- Gradient Checkpointing
- Large effective batch size (24 per GPU)

### Hyperparameters

Default settings (can be overridden via CLI):
```python
epochs = 10
batch_size = 24 (per GPU)
learning_rate = 1e-5
warmup_steps = 500
```

## 🧪 Inference & Evaluation

### Benchmark Inference

```bash
python benchmark_inference_optimizations.py
```

This compares:
- Original Whisper models
- Fine-tuned models
- Different optimization levels
- Speed vs accuracy tradeoffs

### Evaluate WER

```bash
python debug_wer.py
```

## 🔄 Integration with Main Pipeline

After training, integrate fine-tuned models with the main transcription pipeline:

1. **Update FastRay service** to load fine-tuned models:
   ```python
   # In services/fastRay/app.py
   model = WhisperModel("fine_tuning/whisper-base_hani89/best_model")
   ```

2. **Update config** to use fine-tuned models:
   ```bash
   # In .env
   FINE_TUNED_MODEL_PATH=fine_tuning/whisper-base_hani89/best_model
   ```

3. **Compare performance** with baseline models using `run_transcriptions.py`

## 📝 Training Data

The scripts expect data in the format used by the main pipeline:
- `train_data/train_metadata.csv`
- `test_data/test_metadata.csv`
- Audio files in corresponding directories

The training scripts will automatically:
- Load data from these locations
- Preprocess and cache
- Split into train/validation sets

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   ```bash
   # Reduce batch size
   ./run_distributed_training.sh --batch-size 16
   ```

2. **Cache Issues**
   ```bash
   # Clear and rebuild cache
   ./run_distributed_training.sh --clear-cache
   ```

3. **NCCL/GPU Communication Errors**
   ```bash
   # Check GPU availability
   nvidia-smi
   
   # The scripts automatically set these:
   export NCCL_IB_DISABLE=1
   export NCCL_P2P_DISABLE=1
   ```

4. **Port Conflicts**
   ```bash
   # Edit the shell script to change port
   # Change: --master_port=29500
   # To:     --master_port=29501
   ```

### Debug Mode

```bash
# Quick test with minimal data
./run_distributed_training.sh --epochs 1 --batch-size 8
```

## 📚 Additional Resources

- Original tutorial: [LearnOpenCV - Fine-Tuning Whisper](https://learnopencv.com/fine-tuning-whisper-on-custom-dataset/)
- Whisper model: [OpenAI Whisper](https://github.com/openai/whisper)
- Hugging Face Transformers: [Docs](https://huggingface.co/docs/transformers/)

## 🔗 Related Files

- `../config.py` - Main project configuration
- `../run_transcriptions.py` - Batch transcription pipeline
- `../services/fastRay/` - FastRay inference service
- `../.env` - Project environment variables

## ✅ Best Practices

1. **Start with smaller models** (Base or Small) for faster iteration
2. **Use distributed training** for faster convergence
3. **Monitor W&B dashboard** during training
4. **Keep cache files** to speed up subsequent runs
5. **Compare WER** between baseline and fine-tuned models
6. **Document experiments** with clear output directory names

## 📊 Expected Results

Based on previous training runs:

| Model | Baseline WER | Fine-tuned WER | Training Time |
|-------|--------------|----------------|---------------|
| Tiny | ~15% | ~12% | ~1 hour |
| Base | ~12% | ~9% | ~2 hours |
| Small | ~10% | ~7% | ~4 hours |
| Medium | ~8% | ~5% | ~8 hours |
| Large | ~6% | ~4% | ~16 hours |

*Note: Results vary based on dataset and hyperparameters*

## 🎯 Next Steps

1. Train a baseline model (start with Base or Small)
2. Monitor training via W&B
3. Evaluate WER on test set
4. Compare with baseline Whisper
5. Integrate best model into main pipeline
6. Benchmark inference performance
