#!/usr/bin/env python3
"""
Fine-tune Whisper Small model on Medical ASR Dataset
Converted from Jupyter notebook to standalone Python script
"""

import os
import sys

# Set OpenMP environment variable to avoid warnings and performance issues
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# CUDA memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import argparse
import numpy as np
import torch
import evaluate
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datetime import datetime
import warnings

# Suppress common warnings early
warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token*")
warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*eos token.*")

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Hugging Face imports
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperTokenizer, 
    WhisperProcessor, 
    WhisperFeatureExtractor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    EarlyStoppingCallback
)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fine-tune Whisper model (single GPU)')
    parser.add_argument('--clear-cache', action='store_true', 
                       help='Clear cached preprocessed data and reprocess from scratch')
    parser.add_argument('--model-id', type=str, default='openai/whisper-small',
                       help='Whisper model to fine-tune (default: openai/whisper-small)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs (default: 10)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size per GPU (default: 16 for Whisper Small)')
    parser.add_argument('--learning-rate', type=float, default=0.000015,
                       help='Learning rate (default: 0.000015 for batch size 16)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (auto-generated from model name if not specified)')
    parser.add_argument('--no-compute-baseline', action='store_true',
                       help='Skip baseline WER computation before training (default: compute baseline)')
    parser.add_argument('--baseline-samples', type=int, default=50,
                       help='Number of samples to use for baseline WER computation (default: 50)')
    parser.add_argument('--early-stopping-patience', type=int, default=3,
                       help='Early stopping patience (epochs without improvement, default: 3)')
    parser.add_argument('--early-stopping-threshold', type=float, default=0.01,
                       help='Early stopping threshold (minimum improvement, default: 0.01)')
    parser.add_argument('--save-total-limit', type=int, default=3,
                       help='Number of best checkpoints to keep (default: 3)')
    parser.add_argument('--num-workers', type=int, default=32,
                       help='Number of dataloader workers (default: 32 for high-core machines)')
    parser.add_argument('--num-proc', type=int, default=32,
                       help='Number of processes for dataset preprocessing (default: 32 for high-core machines)')
    return parser.parse_args()

def clear_cache(out_dir):
    """Clear all cached preprocessed data"""
    cache_files = [
        f"{out_dir}/cache_train_downmix.arrow",
        f"{out_dir}/cache_valid_downmix.arrow", 
        f"{out_dir}/cache_train_features.arrow",
        f"{out_dir}/cache_valid_features.arrow"
    ]
    
    cleared_files = []
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            cleared_files.append(cache_file)
    
    if cleared_files:
        print(f"🗑️  Cleared {len(cleared_files)} cache files:")
        for file in cleared_files:
            print(f"   - {file}")
    else:
        print("ℹ️  No cache files found to clear")

def check_cache_exists(out_dir):
    """Check if cached preprocessed data exists"""
    cache_files = [
        f"{out_dir}/cache_train_downmix.arrow",
        f"{out_dir}/cache_valid_downmix.arrow", 
        f"{out_dir}/cache_train_features.arrow",
        f"{out_dir}/cache_valid_features.arrow"
    ]
    return all(os.path.exists(f) for f in cache_files)

def install_requirements():
    """Install required packages if not already installed"""
    try:
        import datasets
        import transformers
        import accelerate
        import evaluate
        import jiwer
        import tensorboard
        import gradio
        import wandb
        import dotenv
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install required packages:")
        print("pip install --upgrade datasets[audio] transformers accelerate evaluate jiwer tensorboard gradio wandb python-dotenv")
        sys.exit(1)

def setup_wandb():
    """Setup Weights & Biases for experiment tracking"""
    # Get wandb configuration from environment variables
    wandb_api_key = os.getenv('WANDB_API_KEY')
    wandb_project = os.getenv('WANDB_PROJECT', 'whisper-finetuning-hani89')
    
    if not wandb_api_key:
        print("Warning: WANDB_API_KEY not found in .env file")
        print("Please add your WANDB_API_KEY to the .env file")
        return False
    
    # Set wandb API key
    os.environ["WANDB_API_KEY"] = wandb_api_key
    
    # Initialize wandb
    try:
        import wandb
        wandb.init(
            project=wandb_project,
            config={
                "model_id": "openai/whisper-small",
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.000015,
                "warmup_steps": 1000,
                "dataset": "Hani89/medical_asr_recording_dataset"
            }
        )
        print(f"Wandb initialized with project: {wandb_project}")
        return True
    except Exception as e:
        print(f"Failed to initialize wandb: {e}")
        return False

def downmix_to_mono(batch):
    """Convert stereo audio to mono by averaging channels"""
    audio = batch["audio"]
    if isinstance(audio, dict):
        array = np.asarray(audio["array"], dtype=np.float32)
        if array.ndim == 2:  # stereo
            # Properly convert (channels, samples) to (samples,)
            array = array.mean(axis=0)
        array = array.squeeze()
        batch["audio"]["array"] = array
    return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
    """Prepare dataset by extracting features and tokenizing labels"""
    audio = batch['audio']
    
    batch['input_features'] = feature_extractor(
        audio['array'], 
        sampling_rate=audio['sampling_rate']
    ).input_features[0]
    
    # Use sentence column as ground truth
    batch['labels'] = tokenizer(batch['sentence']).input_ids
    
    return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Custom data collator for speech sequence-to-sequence tasks"""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': feature['input_features']} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': feature['labels']} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors='pt')

        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch

def compute_metrics(pred, tokenizer, metric):
    """Compute Word Error Rate (WER) for evaluation"""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = metric.compute(predictions=pred_str, references=label_str)
    return {'wer': wer}

def compute_baseline_wer(model_id, dataset_valid, num_samples=50):
    """Compute baseline WER using the pretrained model before fine-tuning"""
    print(f"\n🔍 Computing baseline WER using {model_id}...")
    
    # Load baseline model
    baseline_model = WhisperForConditionalGeneration.from_pretrained(model_id)
    baseline_processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')
    
    # Move model to GPU for faster inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_model = baseline_model.to(device)
    print(f"Using device: {device}")
    
    # Set model configuration for English transcription
    baseline_model.generation_config.task = 'transcribe'
    baseline_model.generation_config.language = 'en'
    baseline_model.generation_config.forced_decoder_ids = None
    
    # Load WER metric
    metric = evaluate.load('wer')
    
    total_wer = 0
    sample_count = 0
    
    print(f"Testing on {min(num_samples, len(dataset_valid))} validation samples...")
    
    for i in range(min(num_samples, len(dataset_valid))):
        sample = dataset_valid[i]
        
        # Get ground truth
        ground_truth = sample['sentence']
        
        # Prepare audio (using the same processing as training)
        audio = sample['audio']
        if isinstance(audio['array'], list):
            audio_array = np.array(audio['array'], dtype=np.float32)
        else:
            audio_array = audio['array']
        
        # Convert stereo to mono if needed
        if audio_array.ndim == 2:
            audio_array = audio_array.mean(axis=0)
        
        # Generate prediction
        try:
            inputs = baseline_processor(audio_array, sampling_rate=audio['sampling_rate'], return_tensors="pt")
            
            # Move inputs to GPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Explicitly set attention mask to avoid warnings
            if 'attention_mask' not in inputs:
                # Create attention mask based on input features
                input_features = inputs["input_features"]
                attention_mask = torch.ones_like(input_features[:, :, 0])  # Use first feature dimension
                inputs["attention_mask"] = attention_mask
            
            with torch.no_grad():
                predicted_ids = baseline_model.generate(inputs["input_features"])
                transcription = baseline_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            # Calculate WER for this sample
            wer = metric.compute(predictions=[transcription], references=[ground_truth])
            
            total_wer += wer
            sample_count += 1
            
            # Print first few examples
            if i < 3:
                print(f"  Sample {i+1}:")
                print(f"    Ground Truth: {ground_truth}")
                print(f"    Prediction:   {transcription}")
                print(f"    WER: {wer:.2%}")
            
        except Exception as e:
            print(f"  Error processing sample {i+1}: {e}")
            continue
    
    baseline_wer = total_wer / sample_count if sample_count > 0 else 0
    print(f"\n📊 Baseline WER: {baseline_wer:.2%} (on {sample_count} samples)")
    
    return baseline_wer

def is_main_process():
    """Check if this is the main process (rank 0) in distributed training"""
    return not ('LOCAL_RANK' in os.environ and int(os.environ['LOCAL_RANK']) != 0)

def main():
    """Main training function"""
    # Record start time for total training duration
    start_time = datetime.now()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Only print startup messages on main process
    if is_main_process():
        print("Starting Whisper Small fine-tuning on Medical ASR Dataset...")
    
    # Setup wandb (only on main process)
    wandb_enabled = False
    if is_main_process():
        wandb_enabled = setup_wandb()
    
    # Configuration from command line arguments
    model_id = args.model_id
    out_dir = args.output_dir
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    
    # Auto-generate output directory from model name if not specified
    if out_dir is None:
        # Extract model name from model_id (e.g., "openai/whisper-small" -> "whisper-small")
        model_name = model_id.split('/')[-1]
        out_dir = f"{model_name}_hani89"
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Handle cache clearing (only on main process)
    if args.clear_cache and is_main_process():
        print("\n🗑️  Clearing cache as requested...")
        clear_cache(out_dir)
    
    # Only print configuration on main process
    if is_main_process():
        print(f"Model: {model_id}")
        print(f"Output directory: {out_dir}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Wandb tracking: {'Enabled' if wandb_enabled else 'Disabled'}")
        print(f"Baseline computation: {'Enabled' if not args.no_compute_baseline else 'Disabled'}")
        print(f"Early stopping patience: {args.early_stopping_patience} epochs")
        print(f"Early stopping threshold: {args.early_stopping_threshold}")
        print(f"Save total limit: {args.save_total_limit} checkpoints")
        
        # GPU information
        print(f"\n🖥️  GPU Information:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            print(f"Current GPU: {torch.cuda.current_device()}")
            print(f"GPU name: {torch.cuda.get_device_name()}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Check if distributed training is enabled
            if 'WORLD_SIZE' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                print(f"Distributed training: YES (World size: {world_size}, Local rank: {local_rank})")
                print(f"Effective batch size: {batch_size * world_size}")
            else:
                print(f"Distributed training: NO (Single GPU)")
                print(f"Effective batch size: {batch_size}")
    
    # Check if cached data exists
    cache_exists = check_cache_exists(out_dir)
    if is_main_process():
        if cache_exists:
            print("✅ Found cached preprocessed data - will skip preprocessing!")
        else:
            print("⏳ No cached data found - will preprocess from scratch")
    
    # Load dataset
    if is_main_process():
        print("\nLoading dataset...")
    hani89_dataset_train = load_dataset('Hani89/medical_asr_recording_dataset', split='train')
    hani89_dataset_valid = load_dataset('Hani89/medical_asr_recording_dataset', split='test')
    
    if is_main_process():
        print(f"Training samples: {len(hani89_dataset_train)}")
        print(f"Validation samples: {len(hani89_dataset_valid)}")
    
    # Initialize feature extractor and tokenizer
    if is_main_process():
        print("\nInitializing feature extractor and tokenizer...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    tokenizer = WhisperTokenizer.from_pretrained(model_id, language='English', task='transcribe')
    processor = WhisperProcessor.from_pretrained(model_id, language='English', task='transcribe')
    
    # Compute baseline WER before any training (only on main process)
    baseline_wer = None
    if not args.no_compute_baseline and is_main_process():
        print(f"\n🚀 Starting baseline computation on main process (rank 0)...")
        baseline_wer = compute_baseline_wer(model_id, hani89_dataset_valid, num_samples=args.baseline_samples)
        
        # Save baseline WER to file for reference
        baseline_file = os.path.join(out_dir, "baseline_wer.txt")
        with open(baseline_file, 'w') as f:
            f.write(f"Baseline WER: {baseline_wer:.4f} ({baseline_wer:.2%})\n")
            f.write(f"Computed on: {args.baseline_samples} validation samples\n")
            f.write(f"Model: {model_id}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"📄 Baseline WER saved to: {baseline_file}")
        
    elif not args.no_compute_baseline:
        print(f"\n⏸️  Skipping baseline computation on non-main process...")
    
    # Synchronize all processes before training starts
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if is_main_process():
                print(f"\n🔄 All processes synchronized - starting training...")
    
    # Check if we can skip preprocessing
    cache_exists = check_cache_exists(out_dir)
    if is_main_process():
        if cache_exists:
            print("✅ Found cached preprocessed data - will skip preprocessing!")
        else:
            print("⏳ No cached data found - will preprocess from scratch")
    
    # Only main process should do preprocessing (others will use cache)
    if is_main_process():
        print("\nPreparing data (downmixing to mono)...")
        hani89_dataset_train = hani89_dataset_train.map(
            downmix_to_mono,
            num_proc=args.num_proc
        )
        hani89_dataset_valid = hani89_dataset_valid.map(
            downmix_to_mono,
            num_proc=args.num_proc
        )
        
        # Debug print for audio array shape and dtype (only on main process)
        arr = hani89_dataset_train[0]["audio"]["array"]
        print("Example train audio array type:", type(arr))
        if isinstance(arr, np.ndarray):
            print("Example train audio array shape:", arr.shape)
            print("Example train audio array dtype:", arr.dtype)
        else:
            print("Example train audio array length:", len(arr))
            print("Converting to numpy array...")
            arr = np.asarray(arr, dtype=np.float32)
            print("After conversion - shape:", arr.shape, "dtype:", arr.dtype)
        
        # Prepare dataset with features and labels
        print("\nExtracting features and preparing labels...")
        hani89_dataset_train = hani89_dataset_train.map(
            lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
            num_proc=args.num_proc
        )
        hani89_dataset_valid = hani89_dataset_valid.map(
            lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
            num_proc=args.num_proc
        )
        print("✅ Preprocessing completed on main process")
    else:
        print(f"⏳ Process {os.environ.get('LOCAL_RANK', 'unknown')} waiting for main process to complete preprocessing...")
    
    # Synchronize all processes after preprocessing
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
            if is_main_process():
                print(f"\n🔄 All processes synchronized - preprocessing complete!")
    
    # All processes load the preprocessed dataset (will use cache created by main process)
    if not is_main_process():
        print(f"🔄 Process {os.environ.get('LOCAL_RANK', 'unknown')} loading preprocessed dataset from cache...")
        hani89_dataset_train = load_dataset('Hani89/medical_asr_recording_dataset', split='train')
        hani89_dataset_valid = load_dataset('Hani89/medical_asr_recording_dataset', split='test')
        
        # Apply preprocessing (will use cache created by main process)
        hani89_dataset_train = hani89_dataset_train.map(
            downmix_to_mono,
            num_proc=args.num_proc
        )
        hani89_dataset_valid = hani89_dataset_valid.map(
            downmix_to_mono,
            num_proc=args.num_proc
        )
        
        hani89_dataset_train = hani89_dataset_train.map(
            lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
            num_proc=args.num_proc
        )
        hani89_dataset_valid = hani89_dataset_valid.map(
            lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
            num_proc=args.num_proc
        )
    
    # Initialize model
    if is_main_process():
        print("\nLoading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.task = 'transcribe'
    model.generation_config.language = 'en'
    model.generation_config.forced_decoder_ids = None
    
    # Setup data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    
    # Setup evaluation metric
    metric = evaluate.load('wer')
    
    # Define training arguments
    if is_main_process():
        print("\nSetting up training configuration...")
    run_name = f"whisper_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir, 
        run_name=run_name,  # Unique run name for each training session
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1, 
        learning_rate=learning_rate,
        warmup_steps=1000,
        bf16=True,
        fp16=False,
        num_train_epochs=epochs,
        eval_strategy='epoch',  # Enable evaluation every epoch for early stopping
        logging_strategy='steps',
        save_strategy='epoch',
        predict_with_generate=True,
        generation_max_length=225,
        report_to=['wandb'] if wandb_enabled else ['tensorboard'],
        # Early stopping configuration
        load_best_model_at_end=True,  # Enable to load best model automatically
        metric_for_best_model='wer',
        greater_is_better=False,
        # Model saving configuration
        save_total_limit=args.save_total_limit,
        save_steps=None,  # Save every epoch
        eval_steps=None,  # Evaluate every epoch
        # Other optimizations
        dataloader_num_workers=args.num_workers,
        lr_scheduler_type='constant',
        seed=42,
        data_seed=42,
        # Memory optimization
        gradient_checkpointing=False,  # Disabled to fix backward pass error
        dataloader_pin_memory=False,
        # Additional memory optimizations
        remove_unused_columns=False,
        max_grad_norm=1.0,
        # Distributed training optimizations
        ddp_find_unused_parameters=False,  # Fix DDP warning
        ddp_bucket_cap_mb=25,  # Optimize DDP communication
        logging_steps=10,  # Log every 10 steps
    )
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated*")
    warnings.filterwarnings("ignore", message="Was asked to gather along dimension 0, but all input tensors were scalars*")
    warnings.filterwarnings("ignore", message="find_unused_parameters=True was specified in DDP constructor*")
    warnings.filterwarnings("ignore", message=".*find_unused_parameters.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")
    warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.43.0*")
    warnings.filterwarnings("ignore", message="The attention mask is not set and cannot be inferred from input because pad token is same as eos token*")
    warnings.filterwarnings("ignore", message=".*attention mask.*pad token.*eos token.*")
    
    # Setup early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=hani89_dataset_train,
        eval_dataset=hani89_dataset_valid,
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer, metric),
        callbacks=[early_stopping_callback]
    )
    
    # Start training
    if is_main_process():
        print("\nStarting training...")
    
    # Debug: Show which GPU each process is using
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        print(f"Process {local_rank} using GPU {torch.cuda.current_device()}: {torch.cuda.get_device_name()}")
    
    # Synchronize before training
    if 'WORLD_SIZE' in os.environ:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    
    trainer.train()
    
    # Get training results to extract best WER
    train_results = trainer.state.log_history
    
    # Find the best WER from training
    best_wer = None
    for log in train_results:
        if 'eval_wer' in log:
            current_wer = log['eval_wer']
            if best_wer is None or current_wer < best_wer:
                best_wer = current_wer
    
    # Save the best model (only on main process)
    if is_main_process():
        print(f"\nSaving best model to {out_dir}/best_model...")
        model.save_pretrained(f"{out_dir}/best_model")
        tokenizer.save_pretrained(f"{out_dir}/best_model")
        processor.save_pretrained(f"{out_dir}/best_model")
        
        # Log final model to wandb if enabled
        if wandb_enabled:
            import wandb
            wandb.save(f"{out_dir}/best_model/*")
            print("Model artifacts saved to wandb")
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {out_dir}/best_model")
        
        # Show baseline comparison if available
        if baseline_wer is not None:
            print(f"\n📊 Performance Summary:")
            print(f"   Baseline WER: {baseline_wer:.2%} (pretrained model)")
            if best_wer is not None:
                print(f"   Final WER: {best_wer:.2%} (best during training)")
                improvement = baseline_wer - best_wer
                improvement_pct = (improvement / baseline_wer) * 100
                print(f"   Improvement: {improvement:.2%} ({improvement_pct:.1f}% relative improvement)")
            else:
                print(f"   Final WER: Not available (check training logs)")
                print(f"   Improvement: Training should show improvement over baseline")
        
        print("\n⭐ The BEST model (lowest WER) is stored in:")
        print(f"    {out_dir}/best_model\n")
        print("You can use this directory for inference or further evaluation.")

    # Calculate total training duration
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    if is_main_process():
        print(f"\n🕒 Total training duration: {total_duration}")
        
        # Save training summary to file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(out_dir, f"training_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"================\n")
            f.write(f"Run timestamp: {timestamp}\n")
            f.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total duration: {total_duration}\n")
            f.write(f"\nModel Configuration:\n")
            f.write(f"  Model: {model_id}\n")
            f.write(f"  Epochs: {epochs}\n")
            f.write(f"  Batch size per GPU: {batch_size}\n")
            f.write(f"  Effective batch size: {batch_size * (int(os.environ.get('WORLD_SIZE', 1)))}\n")
            f.write(f"  Learning rate: {learning_rate}\n")
            f.write(f"  Warmup steps: 1000\n")
            f.write(f"  Gradient accumulation steps: 1\n")
            f.write(f"  Mixed precision: bf16\n")
            f.write(f"  Generation max length: 225\n")
            f.write(f"\nTraining Configuration:\n")
            f.write(f"  Distributed training: {'YES' if 'WORLD_SIZE' in os.environ else 'NO'}\n")
            if 'WORLD_SIZE' in os.environ:
                f.write(f"  World size: {os.environ.get('WORLD_SIZE', 'unknown')}\n")
            f.write(f"  Data loader workers: {args.num_workers}\n")
            f.write(f"  Preprocessing workers: {args.num_proc}\n")
            f.write(f"  Early stopping patience: {args.early_stopping_patience}\n")
            f.write(f"  Early stopping threshold: {args.early_stopping_threshold}\n")
            f.write(f"  Save total limit: {args.save_total_limit}\n")
            f.write(f"  Baseline samples: {args.baseline_samples}\n")
            f.write(f"\nDataset Information:\n")
            f.write(f"  Training samples: 5328\n")
            f.write(f"  Validation samples: 1333\n")
            f.write(f"  Dataset: Hani89/medical_asr_recording_dataset\n")
            if baseline_wer is not None:
                f.write(f"  Baseline WER: {baseline_wer:.2%}\n")
            f.write(f"\nOutput Information:\n")
            f.write(f"  Output directory: {out_dir}\n")
            f.write(f"  Best model location: {out_dir}/best_model\n")
            f.write(f"  Wandb tracking: {'Enabled' if wandb_enabled else 'Disabled'}\n")
        print(f"📄 Training summary saved to: {summary_file}")

if __name__ == "__main__":
    # Check and install requirements
    install_requirements()
    
    # Run main training function
    main() 