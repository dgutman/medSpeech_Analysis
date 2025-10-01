#!/usr/bin/env python3
"""
Audio Data Extractor for Medical ASR Dataset

This script extracts audio files and metadata from the Hani89/medical_asr_recording_dataset
and saves them as WAV files with corresponding CSV metadata files.
Supports resuming interrupted extractions.
"""

import os
import csv
import glob
import numpy as np
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional


def extract_audio_data(
    dataset_split: str,
    output_dir: str,
    transcription_key: str = "sentence",
    resume: bool = True,
    file_prefix: str = ""
) -> None:
    """
    Extract audio data from a dataset split and save as WAV files with metadata.
    
    Args:
        dataset_split: The dataset split to extract ('train', 'test', 'validation')
        output_dir: Directory to save the extracted audio files and metadata
        transcription_key: Key name for the transcription field in the dataset
        resume: If True, skip existing files and resume from where it left off
        file_prefix: Prefix to add to filenames (e.g., 'train_', 'test_')
    """
    # Load the dataset
    print(f"Loading {dataset_split} split...")
    dataset = load_dataset("Hani89/medical_asr_recording_dataset", split=dataset_split)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_split}_metadata.csv")
    
    # Check for existing files if resuming
    existing_files = set()
    if resume:
        print("Checking for existing audio files in directory...")
        pattern = os.path.join(output_dir, f"{file_prefix}sample_*.wav")
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            existing_files.add(filename)
        print(f"Found {len(existing_files)} existing files. Will skip these.")
    
    # Write metadata CSV (append mode if resuming and files exist)
    mode = 'a' if resume and len(existing_files) > 0 else 'w'
    with open(csv_path, mode=mode, newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        
        # Write header only if creating new file
        if mode == 'w':
            writer.writerow(["filename", "transcription"])
        
        processed_count = 0
        skipped_count = 0
        
        print(f"Processing {len(dataset)} audio files...")
        for i, sample in enumerate(tqdm(dataset)):
            filename = f"{file_prefix}sample_{i}.wav"
            
            # Skip if file already exists and we're resuming
            if resume and filename in existing_files:
                skipped_count += 1
                continue
            
            audio = sample["audio"]
            transcription = sample[transcription_key]
            file_path = os.path.join(output_dir, filename)
            
            # Process audio data
            array = np.array(audio["array"])
            sr = int(audio["sampling_rate"])
            
            # Handle stereo audio - transpose if needed
            if array.shape[0] == 2 and array.ndim == 2:
                array = array.T
            
            # Normalize and convert to int16
            array = np.clip(array, -1.0, 1.0)
            int16_audio = (array * 32767).astype(np.int16)
            
            try:
                wav_write(file_path, sr, int16_audio)
                writer.writerow([filename, transcription])
                processed_count += 1
            except Exception as e:
                print(f"Error saving {filename}: {e}")
    
    print(f"Successfully processed {processed_count} new files")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} existing files")
    print(f"Total files in {output_dir}: {processed_count + skipped_count}")


def main():
    """Main function to extract both train and test splits."""
    
    # Extract training data
    print("=== Extracting Training Data ===")
    extract_audio_data(
        dataset_split="train",
        output_dir="train_data",
        resume=True,
        file_prefix="train_"
    )
    
    print("\n" + "="*50 + "\n")
    
    # Extract test data
    print("=== Extracting Test Data ===")
    extract_audio_data(
        dataset_split="test", 
        output_dir="test_data",
        resume=True,
        file_prefix="test_"
    )
    
    print("\n=== Extraction Complete ===")
    print("Training data saved to: train_data/ (files prefixed with 'train_')")
    print("Training metadata: train_data/train_metadata.csv")
    print("Test data saved to: test_data/ (files prefixed with 'test_')")
    print("Test metadata: test_data/test_metadata.csv")


if __name__ == "__main__":
    main() 