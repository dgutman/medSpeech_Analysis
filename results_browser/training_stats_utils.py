"""
Training statistics utilities for parsing fine-tuning results.

This module provides functions for extracting and analyzing training statistics
from fine-tuned Whisper models stored in the Fine-Tuning-Whisper-on-Custom-Dataset directory.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_baseline_wer(baseline_file: str) -> Optional[float]:
    """
    Parse baseline WER from a baseline_wer.txt file.
    
    Args:
        baseline_file: Path to baseline_wer.txt file
        
    Returns:
        Baseline WER as float, or None if file doesn't exist or can't be parsed
    """
    if not os.path.exists(baseline_file):
        return None
    
    try:
        with open(baseline_file, 'r') as f:
            content = f.read()
            # Look for pattern like "Baseline WER: 0.1992 (19.92%)"
            match = re.search(r'Baseline WER:\s*([\d.]+)', content)
            if match:
                return float(match.group(1))
    except Exception as e:
        logger.warning(f"Error parsing baseline WER from {baseline_file}: {e}")
    
    return None


def parse_trainer_state(trainer_state_file: str) -> Optional[Dict]:
    """
    Parse training statistics from a trainer_state.json file.
    
    Args:
        trainer_state_file: Path to trainer_state.json file
        
    Returns:
        Dictionary with training statistics, or None if file doesn't exist
    """
    if not os.path.exists(trainer_state_file):
        return None
    
    try:
        with open(trainer_state_file, 'r') as f:
            data = json.load(f)
        
        # Extract key metrics
        result = {
            'best_metric': data.get('best_metric'),  # This is typically the WER
            'best_global_step': data.get('best_global_step'),
            'best_model_checkpoint': data.get('best_model_checkpoint'),
            'epoch': data.get('epoch'),
            'global_step': data.get('global_step'),
            'log_history': data.get('log_history', [])
        }
        
        # Extract training curves data
        training_steps = []
        training_losses = []
        eval_steps = []
        eval_losses = []
        eval_wers = []
        
        for entry in result['log_history']:
            if 'step' in entry:
                if 'loss' in entry:
                    training_steps.append(entry['step'])
                    training_losses.append(entry['loss'])
                if 'eval_wer' in entry:
                    eval_steps.append(entry['step'])
                    eval_wers.append(entry['eval_wer'])
                    if 'eval_loss' in entry:
                        eval_losses.append(entry['eval_loss'])
        
        result['training_curve'] = {
            'steps': training_steps,
            'losses': training_losses,
            'eval_steps': eval_steps,
            'eval_losses': eval_losses,
            'eval_wers': eval_wers
        }
        
        return result
    except Exception as e:
        logger.warning(f"Error parsing trainer state from {trainer_state_file}: {e}")
        return None


def find_best_checkpoint(model_dir: str) -> Optional[str]:
    """
    Find the best checkpoint directory for a model.
    
    Args:
        model_dir: Path to model directory (e.g., whisper-small_hani89)
        
    Returns:
        Path to best checkpoint directory, or None if not found
    """
    # Check for best_model directory first
    best_model_dir = os.path.join(model_dir, 'best_model')
    if os.path.exists(best_model_dir):
        # Look for trainer_state.json in parent directory or checkpoints
        parent_dir = os.path.dirname(model_dir)
        # Try to find the checkpoint that matches best_model
        checkpoint_dirs = [d for d in os.listdir(model_dir) 
                          if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
        if checkpoint_dirs:
            # Sort by checkpoint number and get the highest
            checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
            return os.path.join(model_dir, checkpoint_dirs[-1])
    
    # Otherwise, find the checkpoint with the best metric
    checkpoint_dirs = [d for d in os.listdir(model_dir) 
                      if d.startswith('checkpoint-') and os.path.isdir(os.path.join(model_dir, d))]
    
    if not checkpoint_dirs:
        return None
    
    best_checkpoint = None
    best_metric = float('inf')
    
    for checkpoint_dir in checkpoint_dirs:
        trainer_state_file = os.path.join(model_dir, checkpoint_dir, 'trainer_state.json')
        if os.path.exists(trainer_state_file):
            stats = parse_trainer_state(trainer_state_file)
            if stats and stats['best_metric'] is not None:
                if stats['best_metric'] < best_metric:
                    best_metric = stats['best_metric']
                    best_checkpoint = checkpoint_dir
    
    if best_checkpoint:
        return os.path.join(model_dir, best_checkpoint)
    
    # Fallback: return the highest numbered checkpoint
    checkpoint_dirs.sort(key=lambda x: int(x.split('-')[1]))
    return os.path.join(model_dir, checkpoint_dirs[-1])


def parse_training_summary(summary_file: str) -> Optional[Dict]:
    """
    Parse training summary file to extract duration and configuration.
    
    Args:
        summary_file: Path to training_summary_*.txt file
        
    Returns:
        Dictionary with training duration and config, or None if file doesn't exist
    """
    if not os.path.exists(summary_file):
        return None
    
    try:
        with open(summary_file, 'r') as f:
            content = f.read()
        
        result = {}
        
        # Parse duration (format: "0:24:11.120954" or "1:23:20.353217")
        duration_match = re.search(r'Total duration:\s*([\d:\.]+)', content)
        if duration_match:
            duration_str = duration_match.group(1).strip()
            # Parse duration string to seconds
            try:
                parts = duration_str.split(':')
                if len(parts) == 3:
                    hours, minutes, seconds = parts
                    total_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                    result['training_duration_seconds'] = total_seconds
                    result['training_duration_str'] = duration_str
                else:
                    # Just seconds
                    result['training_duration_seconds'] = float(duration_str)
                    result['training_duration_str'] = duration_str
            except:
                result['training_duration_str'] = duration_str
        
        # Parse start and end times
        start_match = re.search(r'Start time:\s*(.+)', content)
        if start_match:
            result['start_time'] = start_match.group(1).strip()
        
        end_match = re.search(r'End time:\s*(.+)', content)
        if end_match:
            result['end_time'] = end_match.group(1).strip()
        
        # Parse batch size
        batch_match = re.search(r'Batch size per GPU:\s*(\d+)', content)
        if batch_match:
            result['batch_size'] = int(batch_match.group(1))
        
        # Parse effective batch size
        eff_batch_match = re.search(r'Effective batch size:\s*(\d+)', content)
        if eff_batch_match:
            result['effective_batch_size'] = int(eff_batch_match.group(1))
        
        # Parse learning rate
        lr_match = re.search(r'Learning rate:\s*([\d.e-]+)', content)
        if lr_match:
            result['learning_rate'] = float(lr_match.group(1))
        
        return result
    except Exception as e:
        logger.warning(f"Error parsing training summary from {summary_file}: {e}")
        return None


def get_model_training_stats(base_dir: str) -> Dict[str, Dict]:
    """
    Get training statistics for all fine-tuned models.
    
    Args:
        base_dir: Base directory containing fine-tuning results
                  (e.g., Fine-Tuning-Whisper-on-Custom-Dataset)
        
    Returns:
        Dictionary mapping model names to their training statistics
    """
    stats = {}
    
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory does not exist: {base_dir}")
        return stats
    
    # Find all model directories (whisper-*_hani89)
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith('whisper-') and '_hani89' in item:
            model_name = item
            
            # Extract base model name (e.g., 'small' from 'whisper-small_hani89')
            base_model = model_name.replace('whisper-', '').replace('_hani89', '')
            
            # Parse baseline WER
            baseline_file = os.path.join(item_path, 'baseline_wer.txt')
            baseline_wer = parse_baseline_wer(baseline_file)
            
            # Find best checkpoint
            best_checkpoint = find_best_checkpoint(item_path)
            
            # Parse training statistics
            training_stats = None
            if best_checkpoint:
                trainer_state_file = os.path.join(best_checkpoint, 'trainer_state.json')
                training_stats = parse_trainer_state(trainer_state_file)
            
            # Parse training summary for duration
            training_summary_data = None
            summary_files = [f for f in os.listdir(item_path) if f.startswith('training_summary_')]
            if summary_files:
                # Get most recent summary file
                summary_files.sort(reverse=True)
                summary_file = os.path.join(item_path, summary_files[0])
                training_summary_data = parse_training_summary(summary_file)
            
            # Compile statistics
            model_stats = {
                'model_name': model_name,
                'base_model': base_model,
                'baseline_wer': baseline_wer,
                'best_checkpoint': best_checkpoint,
                'training_stats': training_stats,
                'training_summary': training_summary_data
            }
            
            # Calculate best WER and improvement
            if training_stats and training_stats['best_metric'] is not None:
                model_stats['best_wer'] = training_stats['best_metric']
                if baseline_wer is not None:
                    model_stats['wer_improvement'] = baseline_wer - training_stats['best_metric']
                    model_stats['wer_improvement_pct'] = ((baseline_wer - training_stats['best_metric']) / baseline_wer) * 100
                else:
                    model_stats['wer_improvement'] = None
                    model_stats['wer_improvement_pct'] = None
            else:
                model_stats['best_wer'] = None
                model_stats['wer_improvement'] = None
                model_stats['wer_improvement_pct'] = None
            
            # Add training duration info if available
            if training_stats:
                model_stats['total_epochs'] = training_stats.get('epoch')
                model_stats['total_steps'] = training_stats.get('global_step')
            
            # Add training duration from summary
            if training_summary_data:
                model_stats['training_duration_seconds'] = training_summary_data.get('training_duration_seconds')
                model_stats['training_duration_str'] = training_summary_data.get('training_duration_str')
                model_stats['batch_size'] = training_summary_data.get('batch_size')
                model_stats['effective_batch_size'] = training_summary_data.get('effective_batch_size')
                model_stats['learning_rate'] = training_summary_data.get('learning_rate')
            
            stats[base_model] = model_stats
    
    return stats


def get_training_stats_summary(base_dir: str) -> List[Dict]:
    """
    Get a summary of training statistics for all models as a list of dictionaries
    suitable for creating a DataFrame.
    
    Args:
        base_dir: Base directory containing fine-tuning results
        
    Returns:
        List of dictionaries with model statistics
    """
    all_stats = get_model_training_stats(base_dir)
    
    summary = []
    for base_model, stats in all_stats.items():
        # Format training duration
        duration_str = "N/A"
        duration_seconds = None
        if stats.get('training_duration_str'):
            duration_str = stats['training_duration_str']
            duration_seconds = stats.get('training_duration_seconds')
        elif stats.get('training_duration_seconds'):
            duration_seconds = stats['training_duration_seconds']
            # Format seconds to HH:MM:SS
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)
            duration_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        
        summary.append({
            'Model': base_model.title(),
            'Baseline WER': f"{stats['baseline_wer']:.4f}" if stats['baseline_wer'] is not None else "N/A",
            'Best WER': f"{stats['best_wer']:.4f}" if stats['best_wer'] is not None else "N/A",
            'WER Improvement': f"{stats['wer_improvement']:.4f}" if stats['wer_improvement'] is not None else "N/A",
            'Improvement %': f"{stats['wer_improvement_pct']:.2f}%" if stats['wer_improvement_pct'] is not None else "N/A",
            'Training Duration': duration_str,
            'Total Epochs': f"{stats['total_epochs']:.1f}" if stats.get('total_epochs') is not None else "N/A",
            'Total Steps': f"{stats['total_steps']}" if stats.get('total_steps') is not None else "N/A",
            'Batch Size': f"{stats.get('batch_size', 'N/A')}" if stats.get('batch_size') is not None else "N/A",
            'Effective Batch Size': f"{stats.get('effective_batch_size', 'N/A')}" if stats.get('effective_batch_size') is not None else "N/A",
            # For sorting/comparison
            '_baseline_wer': stats['baseline_wer'] if stats['baseline_wer'] is not None else float('inf'),
            '_best_wer': stats['best_wer'] if stats['best_wer'] is not None else float('inf'),
            '_improvement': stats['wer_improvement'] if stats['wer_improvement'] is not None else -float('inf'),
            '_duration_seconds': duration_seconds if duration_seconds is not None else float('inf'),
        })
    
    return summary



