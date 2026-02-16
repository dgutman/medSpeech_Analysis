#!/usr/bin/env python3
"""
Quick script to check which Whisper models were actually used for transcriptions.
This queries the stored JSON in Pixeltable to see the 'model' field.
"""
import os
from dotenv import load_dotenv

# Load .env before importing pixeltable
load_dotenv()

if 'PIXELTABLE_PGDATA' not in os.environ:
    os.environ['PIXELTABLE_PGDATA'] = '/scr/dagutman/devel/medSpeech_Analysis/.pxtData'

import pixeltable as pxt
from db_helpers import TABLE_NAME, MODEL_COLUMNS

# Get table
t = pxt.get_table(TABLE_NAME)

print(f"Checking models used in table: {TABLE_NAME}")
print("="*80)

# For each model column, check what models were actually used
for model_column_name in MODEL_COLUMNS:
    if not hasattr(t, model_column_name):
        print(f"\n⚠️  Column '{model_column_name}' doesn't exist, skipping...")
        continue
    
    model_column = getattr(t, model_column_name)
    
    # Get rows with transcriptions
    rows = t.select(t.id, model_column).where(model_column != None).limit(100).collect()
    
    if len(rows) == 0:
        print(f"\n{model_column_name}: No transcriptions found")
        continue
    
    # Count models used
    models_used = {}
    for row in rows:
        transcription_data = row.get(model_column_name) if hasattr(row, 'get') else row[model_column_name]
        if isinstance(transcription_data, dict):
            model_name = transcription_data.get("model", "unknown")
            models_used[model_name] = models_used.get(model_name, 0) + 1
    
    print(f"\n{model_column_name}:")
    if models_used:
        for model, count in sorted(models_used.items()):
            print(f"  - {model}: {count} transcriptions")
    else:
        print(f"  ⚠️  No 'model' field found in transcription data")
    
    # Check expected vs actual
    from db_helpers import get_whisper_model_name
    expected_model = get_whisper_model_name(model_column_name)
    if models_used:
        actual_models = list(models_used.keys())
        if len(actual_models) == 1 and actual_models[0] == expected_model:
            print(f"  ✓ All using expected model: {expected_model}")
        else:
            print(f"  ⚠️  Expected: {expected_model}, but found: {actual_models}")

print("\n" + "="*80)
print("Done!")





