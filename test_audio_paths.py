#!/usr/bin/env python3
"""Test script to examine how pixeltable returns Audio file paths"""

import pixeltable as pxt
import os

# Get the table
try:
    t = pxt.get_table('medSpeechAnalysis_hf_ray.hani89_asr_dataset')
    print("✅ Table found")
except Exception as e:
    print(f"❌ Error getting table: {e}")
    exit(1)

# Get a sample row
print("\n=== Examining Audio paths ===")
sample = t.select(t.audio, t.id, t.filePath).limit(3).collect()

if len(sample) == 0:
    print("⚠️  No data in table")
    exit(1)

# Convert to pandas DataFrame for easier iteration
df = sample.to_pandas()

print(f"\nFound {len(df)} sample rows\n")

for idx, row in df.iterrows():
    audio_obj = row['audio']
    audio_id = row.get('id', 'N/A')
    file_path = row.get('filePath', 'N/A')
    
    print(f"Row {idx}:")
    print(f"  ID: {audio_id}")
    print(f"  filePath column: {file_path}")
    print(f"  Audio object type: {type(audio_obj)}")
    print(f"  Audio object value: {audio_obj}")
    print(f"  str(audio_obj): {str(audio_obj)}")
    print(f"  repr(audio_obj): {repr(audio_obj)}")
    
    # Check if it's a path-like object
    audio_path = str(audio_obj)
    print(f"  Audio path (as string): {audio_path}")
    print(f"  Path exists: {os.path.exists(audio_path)}")
    
    if os.path.exists(audio_path):
        print(f"  Absolute path: {os.path.abspath(audio_path)}")
        print(f"  Real path: {os.path.realpath(audio_path)}")
        print(f"  Directory: {os.path.dirname(audio_path)}")
        print(f"  Filename: {os.path.basename(audio_path)}")
        
        # Check if it contains medSpeechAnalysis_hf_ray
        if 'medSpeechAnalysis_hf_ray' in audio_path:
            rel_part = audio_path.split('medSpeechAnalysis_hf_ray', 1)[1].lstrip('/')
            print(f"  Relative part after 'medSpeechAnalysis_hf_ray': {rel_part}")
            print(f"  Container path (/data_medspeech/{rel_part}): /data_medspeech/{rel_part}")
    
    print()

print("\n=== Summary ===")
print("Audio objects appear to be file paths that can be converted with str()")
print("We need to ensure these paths match what's mounted in the FastRay container")

