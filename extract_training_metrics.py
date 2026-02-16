#!/usr/bin/env python3
"""
Extract training metrics from fine-tuned Whisper models.
Shows training duration, WER, accuracy improvements, etc.
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime

def parse_training_summary(summary_file):
    """Parse training summary file to extract duration and config"""
    if not os.path.exists(summary_file):
        return None
    
    result = {}
    with open(summary_file, 'r') as f:
        content = f.read()
        
        # Extract duration
        duration_match = re.search(r'Total duration:\s*(.+)', content)
        if duration_match:
            result['duration'] = duration_match.group(1).strip()
        
        # Extract start/end times
        start_match = re.search(r'Start time:\s*(.+)', content)
        if start_match:
            result['start_time'] = start_match.group(1).strip()
        
        end_match = re.search(r'End time:\s*(.+)', content)
        if end_match:
            result['end_time'] = end_match.group(1).strip()
        
        # Extract model config
        model_match = re.search(r'Model:\s*(.+)', content)
        if model_match:
            result['model'] = model_match.group(1).strip()
        
        epochs_match = re.search(r'Epochs:\s*(\d+)', content)
        if epochs_match:
            result['epochs'] = int(epochs_match.group(1))
        
        batch_match = re.search(r'Batch size per GPU:\s*(\d+)', content)
        if batch_match:
            result['batch_size'] = int(batch_match.group(1))
        
        lr_match = re.search(r'Learning rate:\s*([\d.e-]+)', content)
        if lr_match:
            result['learning_rate'] = float(lr_match.group(1))
    
    return result

def parse_trainer_state(trainer_state_file):
    """Parse trainer_state.json to extract WER and training metrics"""
    if not os.path.exists(trainer_state_file):
        return None
    
    with open(trainer_state_file, 'r') as f:
        data = json.load(f)
    
    return {
        'best_wer': data.get('best_metric'),
        'best_step': data.get('best_global_step'),
        'total_epochs': data.get('epoch'),
        'total_steps': data.get('global_step'),
    }

def parse_baseline_wer(baseline_file):
    """Parse baseline WER from baseline_wer.txt"""
    if not os.path.exists(baseline_file):
        return None
    
    with open(baseline_file, 'r') as f:
        content = f.read()
        match = re.search(r'Baseline WER:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))
    return None

def get_model_metrics(model_dir):
    """Get all metrics for a model directory"""
    model_name = os.path.basename(model_dir)
    
    metrics = {
        'model_name': model_name,
        'base_model': model_name.replace('whisper-', '').replace('_hani89', '')
    }
    
    # Find best checkpoint
    best_model_dir = os.path.join(model_dir, 'best_model')
    if os.path.exists(best_model_dir):
        # Look for trainer_state.json in parent or checkpoints
        checkpoint_dirs = [d for d in os.listdir(model_dir) 
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            best_checkpoint = os.path.join(model_dir, checkpoint_dirs[-1])
            trainer_state_file = os.path.join(best_checkpoint, 'trainer_state.json')
            if os.path.exists(trainer_state_file):
                trainer_stats = parse_trainer_state(trainer_state_file)
                metrics.update(trainer_stats)
    
    # Parse baseline WER
    baseline_file = os.path.join(model_dir, 'baseline_wer.txt')
    baseline_wer = parse_baseline_wer(baseline_file)
    if baseline_wer is not None:
        metrics['baseline_wer'] = baseline_wer
        if metrics.get('best_wer') is not None:
            metrics['wer_improvement'] = baseline_wer - metrics['best_wer']
            metrics['wer_improvement_pct'] = ((baseline_wer - metrics['best_wer']) / baseline_wer) * 100
    
    # Find most recent training summary
    summary_files = [f for f in os.listdir(model_dir) if f.startswith('training_summary_')]
    if summary_files:
        summary_files.sort(reverse=True)  # Most recent first
        summary_file = os.path.join(model_dir, summary_files[0])
        summary_data = parse_training_summary(summary_file)
        if summary_data:
            metrics.update(summary_data)
    
    return metrics

def main():
    base_dir = 'Fine-Tuning-Whisper-on-Custom-Dataset'
    
    # Find all model directories
    model_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, d)) 
                  and d.startswith('whisper-') and '_hani89' in d]
    
    # Filter to only tiny, base, small (the ones that are configured)
    configured_models = ['whisper-tiny_hani89', 'whisper-base_hani89', 'whisper-small_hani89']
    model_dirs = [d for d in model_dirs if os.path.basename(d) in configured_models]
    
    print("="*80)
    print("Fine-Tuning Metrics Summary")
    print("="*80)
    print()
    
    all_metrics = []
    for model_dir in sorted(model_dirs):
        metrics = get_model_metrics(model_dir)
        all_metrics.append(metrics)
    
    # Print formatted table
    print(f"{'Model':<15} {'Baseline WER':<15} {'Best WER':<15} {'Improvement':<15} {'Duration':<20} {'Epochs':<10}")
    print("-"*80)
    
    for m in all_metrics:
        base = m.get('base_model', 'unknown')
        baseline = f"{m.get('baseline_wer', 0):.4f}" if m.get('baseline_wer') else "N/A"
        best = f"{m.get('best_wer', 0):.4f}" if m.get('best_wer') else "N/A"
        improvement = f"{m.get('wer_improvement_pct', 0):.2f}%" if m.get('wer_improvement_pct') else "N/A"
        duration = m.get('duration', 'N/A')
        epochs = f"{m.get('total_epochs', 0):.1f}" if m.get('total_epochs') else "N/A"
        
        print(f"{base:<15} {baseline:<15} {best:<15} {improvement:<15} {duration:<20} {epochs:<10}")
    
    print()
    print("="*80)
    print("Detailed Information:")
    print("="*80)
    
    for m in all_metrics:
        print(f"\n📊 {m.get('base_model', 'unknown').upper()} Model")
        print(f"   Model: {m.get('model', 'N/A')}")
        if m.get('baseline_wer') is not None:
            print(f"   Baseline WER: {m['baseline_wer']:.4f} ({m['baseline_wer']*100:.2f}%)")
        if m.get('best_wer') is not None:
            print(f"   Best WER: {m['best_wer']:.4f} ({m['best_wer']*100:.2f}%)")
        if m.get('wer_improvement') is not None:
            print(f"   WER Improvement: {m['wer_improvement']:.4f} ({m['wer_improvement_pct']:.2f}% reduction)")
        if m.get('duration'):
            print(f"   Training Duration: {m['duration']}")
        if m.get('start_time'):
            print(f"   Start Time: {m['start_time']}")
        if m.get('end_time'):
            print(f"   End Time: {m['end_time']}")
        if m.get('total_epochs'):
            print(f"   Total Epochs: {m['total_epochs']:.1f}")
        if m.get('total_steps'):
            print(f"   Total Steps: {m['total_steps']}")
        if m.get('batch_size'):
            print(f"   Batch Size: {m['batch_size']}")
        if m.get('learning_rate'):
            print(f"   Learning Rate: {m['learning_rate']}")

if __name__ == '__main__':
    main()
