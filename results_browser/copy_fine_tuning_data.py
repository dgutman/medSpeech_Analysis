#!/usr/bin/env python3
"""
Script to copy only necessary fine-tuning statistics files.
This is used during Docker build to avoid copying huge model checkpoints.
"""

import os
import shutil
from pathlib import Path

def copy_fine_tuning_stats(source_dir, target_dir):
    """Copy only baseline_wer.txt, trainer_state.json, and training_summary files"""
    source = Path(source_dir)
    target = Path(target_dir)
    
    target.mkdir(parents=True, exist_ok=True)
    
    # Find all whisper-*_hani89 directories
    for model_dir in source.glob("whisper-*_hani89"):
        if not model_dir.is_dir():
            continue
            
        # Get relative path
        rel_path = model_dir.relative_to(source)
        target_model_dir = target / rel_path
        target_model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy baseline_wer.txt if it exists
        baseline_file = model_dir / "baseline_wer.txt"
        if baseline_file.exists():
            shutil.copy2(baseline_file, target_model_dir / "baseline_wer.txt")
            print(f"Copied: {baseline_file} -> {target_model_dir / 'baseline_wer.txt'}")
        
        # Copy training_summary_*.txt files (for training duration)
        for summary_file in model_dir.glob("training_summary_*.txt"):
            if summary_file.is_file():
                shutil.copy2(summary_file, target_model_dir / summary_file.name)
                print(f"Copied: {summary_file} -> {target_model_dir / summary_file.name}")
        
        # Copy trainer_state.json from all checkpoints
        for checkpoint_dir in model_dir.glob("checkpoint-*"):
            if not checkpoint_dir.is_dir():
                continue
                
            checkpoint_rel = checkpoint_dir.relative_to(source)
            target_checkpoint_dir = target / checkpoint_rel
            target_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            trainer_state_file = checkpoint_dir / "trainer_state.json"
            if trainer_state_file.exists():
                shutil.copy2(trainer_state_file, target_checkpoint_dir / "trainer_state.json")
                print(f"Copied: {trainer_state_file} -> {target_checkpoint_dir / 'trainer_state.json'}")

if __name__ == "__main__":
    import sys
    source = sys.argv[1] if len(sys.argv) > 1 else "Fine-Tuning-Whisper-on-Custom-Dataset"
    target = sys.argv[2] if len(sys.argv) > 2 else "/app/fine_tuning_data"
    copy_fine_tuning_stats(source, target)



