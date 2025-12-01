import asyncio
import os

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Set PIXELTABLE_PGDATA to absolute path before importing pixeltable
# This prevents PostgreSQL from using wrong username in paths and ensures
# we're using the same database as load_hf_hani89_to_pxt.py
# Load from .env first, then fall back to default
if 'PIXELTABLE_PGDATA' not in os.environ:
    os.environ['PIXELTABLE_PGDATA'] = '/scr/dagutman/devel/medSpeech_Analysis/.pxtData'

# Ensure we're using absolute paths to avoid username-based path resolution issues
# The workspace is at /scr/dagutman/ even if user is dgutman
os.chdir('/scr/dagutman/devel/medSpeech_Analysis')

from service_utils import (
    check_service_status,
    print_service_status
)
from db_helpers import (
    initialize_table,
    MODEL_COLUMNS
)
import pixeltable as pxt
from batch_processing_utils import process_batch


# Initialize table
t = initialize_table()

# ============================================================================
# TO ADD MORE MODELS:
# ============================================================================
# Add models to MODEL_CONFIG in db_helpers.py:
#   MODEL_CONFIG = {
#       "whisper_base": "base",
#       "whisper_tiny": "tiny",    # Add when ready
#       "whisper_small": "small",   # Add when ready
#       # etc.
#   }
# 
# Then change model_column below to process different models
# ============================================================================

# Process a batch of rows
if __name__ == "__main__":
    api_url = "http://localhost:8000"  # Use localhost to avoid network routing overhead
    
    # Check service status first to get recommendations
    print("Checking FastRay service status...")
    status_info = asyncio.run(check_service_status(api_url))
    print_service_status(status_info)
    
    # Simple approach: match stress test exactly
    # Stress test uses MAX_WORKERS=200, no complex calculations
    # Just use 200 concurrent requests - simple and straightforward
    recommended = 200  # Match stress test: MAX_WORKERS=200
    print(f"Using 200 concurrent requests (matching stress test MAX_WORKERS)")
    print(f"Simple approach - no complex capacity calculations")
    
    # Manual override: Set this to override the multiplier-based calculation
    # With 40 replicas, typical range: 400-600 (10-15x replicas)
    # The ConcurrencyMonitor will track performance and warn if you're overloading
    # Set this to override the automatic recommendation
    MANUAL_OVERRIDE = None  # Set to a number like 400, 500, 600, etc. to override recommendation
    if MANUAL_OVERRIDE is not None:
        recommended = MANUAL_OVERRIDE
        print(f"⚠️  Using manual override: {recommended} concurrent requests")
    
    # Automatically iterate through all models and process each until complete
    # Each model will be processed until all rows have transcriptions (no more None values)
    print(f"\n{'='*60}")
    print(f"🔄 Auto-processing all models in MODEL_CONFIG")
    print(f"   Models to process: {MODEL_COLUMNS}")
    print(f"{'='*60}\n")
    
    all_updates = {}
    for model_column in MODEL_COLUMNS:
        print(f"\n{'='*80}")
        print(f"📝 Processing Model: {model_column}")
        print(f"{'='*80}")
        
        # Check if there are any rows that need transcription for this model
        model_column_ref = getattr(t, model_column, None)
        if model_column_ref is None:
            print(f"⚠️  Column '{model_column}' doesn't exist, skipping...")
            continue
        
        # Count rows that need transcription
        # The column should store a dict, but we check for None, empty dicts, and empty lists (legacy)
        # Collect rows and filter in Python
        all_rows = t.select(t.id, model_column_ref).collect()
        initial_rows_needing = 0
        for row in all_rows:
            # Try multiple ways to access the column value
            col_value = None
            if hasattr(row, 'get'):
                col_value = row.get(model_column)
            elif isinstance(row, dict):
                col_value = row.get(model_column)
            else:
                col_value = getattr(row, model_column, None)
            
            # Check if None, empty dict, or empty list
            if col_value is None or (isinstance(col_value, dict) and len(col_value) == 0) or (isinstance(col_value, list) and len(col_value) == 0):
                initial_rows_needing += 1
        
        if initial_rows_needing == 0:
            print(f"✅ {model_column}: All rows already have transcriptions, skipping...")
            continue
        
        print(f"📊 Found {initial_rows_needing:,} rows needing transcription for {model_column}")
        
        # Process in batches until all rows are complete
        batch_size = 2500
        batch_count = 0
        
        while True:
            batch_count += 1
            
            # Only check remaining rows every 3 batches to avoid expensive queries
            # This significantly speeds up processing
            if batch_count == 1 or batch_count % 3 == 0:
                # Check how many rows still need transcription (always check from database)
                # The column should store a dict, but we check for None, empty dicts, and empty lists (legacy)
                print(f"   Checking remaining rows...")
                all_rows = t.select(t.id, model_column_ref).collect()
                rows_needing_transcription = 0
                for row in all_rows:
                    # Try multiple ways to access the column value
                    col_value = None
                    if hasattr(row, 'get'):
                        col_value = row.get(model_column)
                    elif isinstance(row, dict):
                        col_value = row.get(model_column)
                    else:
                        col_value = getattr(row, model_column, None)
                    
                    # Check if None, empty dict, or empty list
                    if col_value is None or (isinstance(col_value, dict) and len(col_value) == 0) or (isinstance(col_value, list) and len(col_value) == 0):
                        rows_needing_transcription += 1
                
                if rows_needing_transcription == 0:
                    total_processed = initial_rows_needing
                    print(f"\n✅ {model_column}: All {total_processed:,} rows completed!")
                    break
                
                print(f"\n🔄 Processing batch {batch_count} for {model_column} ({rows_needing_transcription:,} remaining)...")
            else:
                # Don't check remaining rows, just process another batch
                print(f"\n🔄 Processing batch {batch_count} for {model_column}...")
            
            updates = process_batch(
                table=t,
                model_columns=MODEL_COLUMNS,
                batch_size=batch_size, 
                max_concurrent_requests=recommended,
                check_service=False,  # Already checked above
                api_url=api_url,
                model_column=model_column
            )
            
            # Only re-check remaining rows every 3 batches (skip expensive query)
            if batch_count % 3 == 0:
                # Re-check remaining rows from database (more accurate than counting updates)
                # Since the column stores a list, filter in Python for None or empty lists
                all_rows = t.select(t.id, model_column_ref).collect()
                rows_after = 0
                for row in all_rows:
                    col_value = row.get(model_column) if hasattr(row, 'get') else row[model_column] if isinstance(row, dict) else getattr(row, model_column, None)
                    if col_value is None or (isinstance(col_value, list) and len(col_value) == 0):
                        rows_after += 1
                
                rows_processed_this_batch = rows_needing_transcription - rows_after
                
                if rows_after == 0:
                    total_processed = initial_rows_needing
                    print(f"\n✅ {model_column}: All {total_processed:,} rows completed!")
                    break
                else:
                    total_processed_so_far = initial_rows_needing - rows_after
                    print(f"   Progress: {total_processed_so_far:,} completed ({rows_processed_this_batch:,} this batch), {rows_after:,} remaining")
                    rows_needing_transcription = rows_after  # Update for next iteration
        
        all_updates[model_column] = total_processed
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"🎉 All Models Processing Complete!")
    print(f"{'='*80}")
    for model, count in all_updates.items():
        print(f"   {model}: {count:,} rows processed")
    print(f"{'='*80}\n")




## #Whisper Tiny English Model (local whisper)
#t.add_computed_column( whisper_tinyEn_transcription  = whisper.transcribe(audio= t.audio, model='tiny.en'), if_exists='ignore')  

# # #Whisper English Small Model
# table2.add_computed_column( whisper_smallEn_transcription = whisper.transcribe( audio = table2.audio, model = "small.en"), if_exists='ignore')

# # #Whisper English medium Model
# table2.add_computed_column( whisper_mediumEn_transcription = whisper.transcribe( audio = table2.audio, model = "medium.en"), if_exists='ignore')

# # #Whisper Large Model
# table2.add_computed_column( whisper_large_transcription = whisper.transcribe( audio = table2.audio, model = "large"), if_exists='ignore')

# # # #Whisper Base English model
# table2.add_computed_column( whisper_baseEn_transcription = whisper.transcribe( audio = table2.audio, model = "base.en"), if_exists='ignore')

# # # #Turbo
# table2.add_computed_column( whisper_turbo_transcription = whisper.transcribe( audio = table2.audio, model = "turbo"), if_exists='ignore')