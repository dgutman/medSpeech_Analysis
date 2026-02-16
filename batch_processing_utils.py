"""
Batch processing utilities for transcribing audio files.
"""
import asyncio
import sys
import pandas as pd
from io import StringIO
from tqdm import tqdm

from transcribe_helpers import (
    transcribe_audio,
    get_transcription_stats,
    print_transcription_stats,
    get_all_model_stats,
    print_all_model_stats
)
from service_utils import (
    check_service_status,
    print_service_status,
    ConcurrencyMonitor,
    get_realtime_stats,
    print_realtime_stats
)
from db_helpers import get_whisper_model_name


def process_batch(table, model_columns, batch_size=100, max_concurrent_requests=20, 
                  check_service=True, api_url="http://localhost:8000", 
                  model_column="whisper_base"):
    """
    Process a batch of rows, transcribing and updating them
    
    Args:
        table: Pixeltable table object
        model_columns: List of model column names to track
        batch_size: Number of rows to process in this batch
        max_concurrent_requests: Maximum number of concurrent API requests (default: 20)
                                FastRay service can handle ~30 concurrent requests per replica
        check_service: Whether to check service status before processing (default: True)
        api_url: FastRay API URL
        model_column: Name of the column to update (e.g., "whisper_base", "whisper_tiny", etc.)
    
    Returns:
        List of completed transcription results
    """
    # Show stats ONLY for the model being processed (not all models - that's confusing!)
    print(f"\n{'='*60}")
    print(f"🎯 Processing Model: {model_column}")
    print(f"{'='*60}")
    stats = get_transcription_stats(table, model_column)
    print_transcription_stats(stats)
    
    if not stats.get("column_exists"):
        print(f"Error: Column '{model_column}' does not exist. Available columns: {[c for c in dir(table) if not c.startswith('_')]}")
        return []
    
    # Check service status if requested
    if check_service:
        print("Checking FastRay service status...")
        status_info = asyncio.run(check_service_status(api_url))
        print_service_status(status_info)
    
    # Verify ID column exists (needed for batch updates)
    if not hasattr(table, 'id'):
        print("⚠️  Warning: ID column not found. Batch updates require an 'id' column.")
        print("   Attempting to create 'id' column from audio filename...")
        from db_helpers import ensure_id_column
        ensure_id_column(table)
        if not hasattr(table, 'id'):
            print("❌ Error: Could not create 'id' column. Batch processing cannot proceed.")
            print("   Please add an 'id' column manually or use a table with a primary key.")
            return []
    
    # Get the model column reference
    model_column_ref = getattr(table, model_column)
    
    # Get rows where model_column is None/NULL or empty list
    # Since the column stores a list, we need to check for both None and empty lists
    # First, try to get rows where column is None/NULL
    # Build select columns - only include columns that exist
    select_cols = [table.audio, model_column_ref]
    if hasattr(table, 'id'):
        select_cols.insert(0, table.id)
    if hasattr(table, 'filePath'):
        select_cols.append(table.filePath)
    
    # Query for rows that need transcription
    # IMPORTANT: We query WITHOUT a limit first to ensure we get all rows that need transcription.
    # The previous code used .limit(batch_size * 2) which could miss rows if more than 5000 needed processing.
    # We filter in Python and then apply batch_size limit only after filtering.
    try:
        # Try direct == None first - NO LIMIT, we want all rows that need transcription
        # Only apply limit after we've filtered to rows that actually need transcription
        result = table.select(*select_cols).where(model_column_ref == None).collect()
        if len(result) == 0:
            # If that returns nothing, try .str == None (Json-specific)
            result = table.select(*select_cols).where(model_column_ref.str == None).collect()
    except Exception as e:
        # If both fail, collect all and filter in Python
        # NO LIMIT here either - we need to see all rows to filter properly
        print(f"⚠️  Note: SQL NULL check failed, filtering in Python: {e}")
        result = table.select(*select_cols).collect()
    
    # Filter in Python to handle both None and empty lists
    # A row needs transcription if: column is None, or column is an empty list []
    filtered_result = []
    for row in result:
        col_value = row.get(model_column) if hasattr(row, 'get') else row[model_column] if isinstance(row, dict) else getattr(row, model_column, None)
        # Check if None or empty list
        if col_value is None or (isinstance(col_value, list) and len(col_value) == 0):
            row_dict = {
                'audio': row.get('audio') if hasattr(row, 'get') else row['audio'] if isinstance(row, dict) else getattr(row, 'audio', None)
            }
            # Only add filePath if it exists in the table and row
            if hasattr(table, 'filePath'):
                row_dict['filePath'] = row.get('filePath') if hasattr(row, 'get') else row.get('filePath', None) if isinstance(row, dict) else getattr(row, 'filePath', None)
            # Only add id if it exists
            if hasattr(table, 'id'):
                row_dict['id'] = row.get('id') if hasattr(row, 'get') else row['id'] if isinstance(row, dict) else getattr(row, 'id', None)
            filtered_result.append(row_dict)
    
    # Limit to batch_size AFTER filtering - this ensures we process rows that actually need transcription
    result = filtered_result[:batch_size]
    
    if len(result) == 0:
        print(f"No rows found with {model_column} = None or empty list")
        return []
    
    # Convert result to DataFrame if it's not already
    if not hasattr(result, 'iterrows'):
        # result is a dict-like object, convert to DataFrame
        result_df = pd.DataFrame(result)
    else:
        result_df = result
    
    print(f"\nFound {len(result_df)} rows with {model_column} = None")
    print(f"Sample IDs: {result_df['id'].head().tolist() if len(result_df) > 0 else []}")
    print(f"Submitting all {len(result_df)} requests immediately (like stress test)")
    print(f"Ray Serve will queue them (queue depth per replica: max_ongoing_requests)")
    
    # Create semaphore to limit concurrent requests (but submit all tasks immediately)
    # This allows all tasks to be created and submitted, but limits how many run at once
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    
    # Process and update in batches as results come in
    # This saves progress incrementally and provides better feedback
    update_batch_size = 100  # Update database every N completed transcriptions
    max_retries = 3  # Retry failed transcriptions up to 3 times
    
    # Get Whisper model name from column name
    whisper_model = get_whisper_model_name(model_column)
    print(f"📌 Using Whisper model: '{whisper_model}' for column '{model_column}'")
    
    # Initialize performance monitor
    monitor = ConcurrencyMonitor(api_url=api_url)
    monitor.start()
    
    async def process_row_with_retry(idx, row, retry_count=0):
        """Process a row with retry logic for resilience"""
        import time
        start_time = time.time()
        success = False
        
        try:
            import os
            # Prefer filePath if it exists (should be container path)
            if 'filePath' in row and row.get('filePath'):
                audio_path = row['filePath']
            else:
                # Extract audio path - handle dict, Audio object, or string
                audio_val = row['audio']
                if isinstance(audio_val, dict):
                    # If it's a dict, try to get the 'path' key
                    audio_path = audio_val.get('path', str(audio_val))
                elif hasattr(audio_val, 'path'):
                    # If it's an Audio object with a path attribute
                    audio_path = audio_val.path
                else:
                    # Otherwise, convert to string
                    audio_path = str(audio_val)
                
                # Map host path to container path if needed
                # Container: all files are in /data/
                if audio_path.startswith('./'):
                    # Relative path - extract just the filename
                    audio_path = os.path.basename(audio_path.lstrip('./'))
                elif os.path.isabs(audio_path):
                    # Absolute path - extract just the filename
                    audio_path = os.path.basename(audio_path)
                
                # Map to container path: /data/<filename>
                if not audio_path.startswith('/'):
                    audio_path = f"/data/{audio_path}"
            
            row_id = row.get('id') if hasattr(row, 'get') else row['id'] if isinstance(row, dict) and 'id' in row else getattr(row, 'id', None)
            if row_id is None:
                raise Exception("Row ID is required for updates but not found in row data")
            
            # Submit request immediately (like stress test)
            # Semaphore limits concurrent execution, but all tasks are created
            async with semaphore:  # Limit concurrent requests
                # Debug: Verify we're using the correct model
                transcription_result = await transcribe_audio(audio_path, model=whisper_model, api_url=api_url)
            
            # Check if transcription failed
            if isinstance(transcription_result, dict) and "error" in transcription_result:
                raise Exception(transcription_result["error"])
            
            # Verify the result has the expected model name
            if isinstance(transcription_result, dict):
                result_model = transcription_result.get("model", "unknown")
                if result_model != whisper_model:
                    print(f"⚠️  Warning: Expected model '{whisper_model}' but got '{result_model}' for row {row_id}")
            
            success = True
            response_time = time.time() - start_time
            # Extract inference time from the response if available
            inference_time = None
            if isinstance(transcription_result, dict) and "inference_time_seconds" in transcription_result:
                inference_time = transcription_result["inference_time_seconds"]
            monitor.record_response(response_time, success=True, inference_time=inference_time)
            return row_id, transcription_result
        except Exception as e:
            response_time = time.time() - start_time
            if retry_count < max_retries:
                row_id_str = row.get('id', 'unknown') if hasattr(row, 'get') else row.get('id', 'unknown') if isinstance(row, dict) else getattr(row, 'id', 'unknown')
                print(f"⚠️  Retry {retry_count + 1}/{max_retries} for row {row_id_str}: {str(e)[:100]}")
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await process_row_with_retry(idx, row, retry_count + 1)
            else:
                row_id_str = row.get('id', 'unknown') if hasattr(row, 'get') else row.get('id', 'unknown') if isinstance(row, dict) else getattr(row, 'id', 'unknown')
                print(f"❌ Failed after {max_retries} retries for row {row_id_str}: {str(e)[:100]}")
                monitor.record_response(response_time, success=False)
                # Return error result so we can track it
                row_id_for_error = row.get('id') if hasattr(row, 'get') else row['id'] if isinstance(row, dict) and 'id' in row else getattr(row, 'id', None)
                return row_id_for_error, {"error": str(e), "retries_exhausted": True}
    
    def update_batch(batch_results):
        """Update a batch of rows using batch_update() - ensures correct matching"""
        if not batch_results:
            return 0
        
        # Prepare data: ensure all results are dicts, not lists
        validated_results = []
        
        for row_id, transcription_result in batch_results:
            # Ensure we're storing a dict, not a list
            if isinstance(transcription_result, list):
                # If it's a list, take the first element if it's a dict
                if len(transcription_result) > 0 and isinstance(transcription_result[0], dict):
                    print(f"⚠️  Warning: Row {row_id} got a list instead of dict, using first element")
                    transcription_result = transcription_result[0]
                else:
                    transcription_result = {"error": "Invalid transcription result format (list)"}
            elif not isinstance(transcription_result, dict):
                # If it's not a dict or list, create an error dict
                transcription_result = {"error": f"Invalid transcription result type: {type(transcription_result)}"}
            
            validated_results.append((row_id, transcription_result))
        
        # Use batch_update() - this properly matches each result to its row
        # Format: [{'id': row_id1, 'column': value1}, {'id': row_id2, 'column': value2}, ...]
        update_dicts = []
        for row_id, transcription_result in validated_results:
            update_dict = {
                'id': row_id,
                model_column: transcription_result
            }
            update_dicts.append(update_dict)
        
        # Suppress verbose output
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            table.batch_update(update_dicts)
            sys.stdout = old_stdout
            return len(validated_results)
        except Exception as e:
            sys.stdout = old_stdout
            print(f"⚠️  Batch update failed: {e}")
            print(f"   Error type: {type(e).__name__}")
            print(f"   Falling back to individual updates...")
            
            # Fallback: update each row individually
            saved_count = 0
            errors = []
            for row_id, transcription_result in validated_results:
                try:
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    table.update(
                        {model_column: transcription_result}, 
                        where=table.id == row_id
                    )
                    sys.stdout = old_stdout
                    saved_count += 1
                except Exception as e2:
                    sys.stdout = old_stdout
                    error_msg = f"Row {row_id}: {str(e2)[:100]}"
                    errors.append(error_msg)
                    if len(errors) <= 5:
                        print(f"❌ Error updating row {row_id}: {e2}")
            
            if errors and len(errors) > 5:
                print(f"❌ ... and {len(errors) - 5} more update errors")
            
            if saved_count < len(validated_results):
                print(f"⚠️  Only {saved_count}/{len(validated_results)} rows updated successfully")
            
            return saved_count
    
    async def process_and_update_in_batches():
        """Process transcriptions and update database in batches as results arrive"""
        # Create all tasks immediately (like stress test)
        # Wrap in create_task() to schedule them
        tasks = [asyncio.create_task(process_row_with_retry(idx, row)) for idx, row in result_df.iterrows()]
        total_tasks = len(tasks)
        completed_updates = []
        failed_updates = []
        batch_results = []
        last_save_count = 0
        last_status_check_time = 0
        status_check_interval = 30  # Check service status every 30 seconds
        
        # Process tasks and collect results as they complete
        import time
        start_time = time.time()
        with tqdm(total=total_tasks, desc="Transcribing") as pbar:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    row_id, transcription_result = result
                    
                    # Check if this was a failed transcription
                    if isinstance(transcription_result, dict) and transcription_result.get("error"):
                        failed_updates.append((row_id, transcription_result))
                        pbar.set_postfix({"failed": len(failed_updates), "saved": last_save_count})
                    else:
                        batch_results.append(result)
                        completed_updates.append(result)
                        pbar.set_postfix({"saved": last_save_count, "pending": len(batch_results)})
                    
                    # Update database when we reach batch size (using batch_update() for correct matching)
                    if len(batch_results) >= update_batch_size:
                        saved_count = update_batch(batch_results)
                        last_save_count += saved_count
                        batch_results = []  # Clear batch
                        pbar.set_postfix({"saved": last_save_count, "failed": len(failed_updates)})
                    
                    # Periodically check service status during processing
                    current_time = time.time()
                    if current_time - last_status_check_time >= status_check_interval:
                        status = await monitor.check_service_status()
                        if "metrics" in status:
                            util = status["metrics"].get("utilization_percent", 0)
                            pbar.set_postfix({
                                "saved": last_save_count, 
                                "failed": len(failed_updates),
                                "util": f"{util:.0f}%"
                            })
                        last_status_check_time = current_time
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"\n❌ Unexpected error processing task: {e}")
                    pbar.update(1)
        
        # Update any remaining results
        if batch_results:
            print(f"\nUpdating final batch of {len(batch_results)} rows...")
            saved_count = update_batch(batch_results)
            last_save_count += saved_count
        
        return completed_updates, failed_updates, last_save_count
    
    # Run async processing with batched updates (efficient database updates)
    print(f"\nProcessing {len(result_df)} transcriptions with batched database updates (every {update_batch_size} results)...")
    print(f"Retry logic: up to {max_retries} retries per failed transcription")
    print(f"Concurrency: {max_concurrent_requests} concurrent requests (monitoring enabled)")
    
    # Run the async processing
    async def run_with_monitoring():
        completed, failed, total_saved = await process_and_update_in_batches()
        
        # Print performance monitoring summary
        print(f"\n{'='*60}")
        print(f"📈 Performance Monitoring Summary")
        print(f"{'='*60}")
        monitor.print_summary()
        
        # Check service status one more time to see final load
        print(f"\n📊 Final Service Status:")
        final_status = await monitor.check_service_status()
        print_realtime_stats(final_status)
        
        return completed, failed, total_saved
    
    completed, failed, total_saved = asyncio.run(run_with_monitoring())
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Processing Complete")
    print(f"{'='*60}")
    print(f"Successfully transcribed and saved: {total_saved} rows")
    print(f"Failed after retries: {len(failed)} rows")
    if failed:
        print(f"\n⚠️  Failed rows (IDs): {[row_id for row_id, _ in failed[:10]]}")
        if len(failed) > 10:
            print(f"   ... and {len(failed) - 10} more")
    
    # Show updated statistics for the model we just processed
    print(f"\n📊 Updated Status for {model_column}:")
    updated_stats = get_transcription_stats(table, model_column, refresh_table=True)
    print_transcription_stats(updated_stats)
    
    return completed

