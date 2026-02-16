"""
Helper functions for transcribing audio via FastRay service.
"""
import os
import httpx
import asyncio


async def transcribe_audio(audio_path: str, model: str = "base", api_url: str = "http://oppenheimer.neurology.emory.edu:8000", container_path: str = "/data_medspeech") -> dict:
    """
    Transcribe audio using FastRay Whisper service via file path.
    
    Args:
        audio_path: Path to audio file on host
        model: Whisper model to use (e.g., "base", "tiny", "small", "medium", "large", "turbo")
        api_url: Base URL for FastRay API
        container_path: Path prefix in container where files are mounted
    
    Returns:
        Full JSON response with text, model, language, duration, inference_time_seconds
    """
    try:
        # Check if path is already a container path
        if audio_path.startswith('/data'):
            # Already a container path, use it directly
            container_file_path = audio_path
        else:
            # Host path - check if it exists, then map to container
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}
            
            # Map host path to container path
            # If filePath is provided (e.g., "train_data/train_sample_0.wav"), preserve subdirectory structure
            # Otherwise, extract just the filename
            if '/' in audio_path and not os.path.isabs(audio_path):
                # Relative path with subdirectory (e.g., "train_data/train_sample_0.wav")
                # Use it directly as /data/train_data/train_sample_0.wav
                container_file_path = f"/data/{audio_path}"
            else:
                # Absolute path or just filename - extract filename only
                filename = os.path.basename(audio_path)
                container_file_path = f"/data/{filename}"
        
        # Make request to FastRay API using path endpoint
        request_data = {
            "file_path": container_file_path,
            "task": "transcribe",
            "beam_size": 5,
            "language": "en",
            "model": model  # Model parameter is passed here - API will cache/switch models as needed
        }
        
        # Make async request to FastRay API
        # Create new client for each request (like stress test) to force routing to different replicas
        # This helps with load balancing and prevents connection clustering
        async with httpx.AsyncClient(
            timeout=300.0,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0)  # Disable connection pooling
        ) as client:
            response = await client.post(
                f"{api_url}/transcribe/path",
                json=request_data
            )
            # Get more details on errors
            if response.status_code >= 400:
                error_detail = f"HTTP {response.status_code}"
                try:
                    error_body = response.json()
                    if isinstance(error_body, dict) and "detail" in error_body:
                        error_detail += f": {error_body['detail']}"
                    elif isinstance(error_body, dict):
                        error_detail += f": {error_body}"
                    else:
                        error_detail += f": {response.text[:200]}"
                except:
                    error_detail += f": {response.text[:200] if hasattr(response, 'text') else 'No error details'}"
                return {"error": f"API request failed: {error_detail}", "status_code": response.status_code, "file_path": container_file_path}
            
            response.raise_for_status()
            result = response.json()
            return result
        
    except httpx.HTTPStatusError as e:
        # Get response body for more details
        error_detail = f"HTTP {e.response.status_code}"
        try:
            error_body = e.response.json()
            if isinstance(error_body, dict) and "detail" in error_body:
                error_detail += f": {error_body['detail']}"
        except:
            error_detail += f": {str(e)}"
        return {"error": f"API request failed: {error_detail}", "status_code": e.response.status_code, "file_path": container_file_path if 'container_file_path' in locals() else audio_path}
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {str(e)}", "file_path": container_file_path if 'container_file_path' in locals() else audio_path}
    except Exception as e:
        return {"error": str(e), "file_path": container_file_path if 'container_file_path' in locals() else audio_path}


def get_transcription_stats(table, model_column_name="whisper_base", refresh_table=False):
    """
    Get statistics about transcription completion for a given model column.
    
    Args:
        table: Pixeltable table object
        model_column_name: Name of the column to check (e.g., "whisper_base", "whisper_tiny", etc.)
        refresh_table: If True, get a fresh table reference to ensure we see latest data
    
    Returns:
        dict with statistics about completed vs missing transcriptions
    """
    # Get a fresh table reference if requested (to see latest data after inserts)
    if refresh_table:
        import pixeltable as pxt
        try:
            # Try to get table name and refresh
            if hasattr(table, 'name'):
                table = pxt.get_table(table.name)
            elif hasattr(table, '_name'):
                table = pxt.get_table(table._name)
            # If table has a path/identifier, try to use it
            elif hasattr(table, 'path'):
                table = pxt.get_table(table.path)
        except Exception:
            pass  # If refresh fails, use original table
    
    if not hasattr(table, model_column_name):
        return {
            "column_exists": False,
            "error": f"Column '{model_column_name}' does not exist in table"
        }
    
    # Get the column reference
    model_column = getattr(table, model_column_name)
    
    # Count total rows (fresh query to ensure we see latest data)
    total_rows = table.select().count()
    
    # Count rows with None or empty dict/list (missing transcriptions)
    # The column should store a dict, but we check for None, empty dicts, and empty lists (legacy)
    # We don't need the ID column for counting, so just select the model column
    all_rows = table.select(model_column).collect()
    rows_with_none = 0
    for row in all_rows:
        # Try multiple ways to access the column value
        col_value = None
        if hasattr(row, 'get'):
            col_value = row.get(model_column_name)
        elif isinstance(row, dict):
            col_value = row.get(model_column_name)
        else:
            col_value = getattr(row, model_column_name, None)
        
        # Check if None, empty dict, or empty list
        if col_value is None or (isinstance(col_value, dict) and len(col_value) == 0) or (isinstance(col_value, list) and len(col_value) == 0):
            rows_with_none += 1
    
    # Count rows with transcriptions (not None)
    rows_with_transcription = total_rows - rows_with_none
    
    # Calculate percentage
    completion_pct = (rows_with_transcription / total_rows * 100) if total_rows > 0 else 0
    
    return {
        "column_exists": True,
        "model_column": model_column_name,
        "total_rows": total_rows,
        "rows_with_transcription": rows_with_transcription,
        "rows_without_transcription": rows_with_none,
        "completion_percentage": round(completion_pct, 2)
    }


def print_transcription_stats(stats):
    """Print transcription statistics in a readable format"""
    if not stats.get("column_exists"):
        print(f"⚠️  {stats.get('error', 'Unknown error')}")
        return
    
    print("\n" + "="*60)
    print(f"Transcription Statistics: {stats['model_column']}")
    print("="*60)
    print(f"Total rows: {stats['total_rows']:,}")
    print(f"Rows with transcription: {stats['rows_with_transcription']:,} ({stats['completion_percentage']}%)")
    print(f"Rows without transcription: {stats['rows_without_transcription']:,} ({100 - stats['completion_percentage']:.2f}%)")
    print("="*60 + "\n")


def get_all_model_stats(table, model_columns, refresh_table=False):
    """
    Get statistics for all model columns.
    
    Args:
        table: Pixeltable table object
        model_columns: List of model column names to check
        refresh_table: If True, get a fresh table reference to ensure we see latest data
    
    Returns:
        dict with total_rows and stats for each model
    """
    # Get a fresh table reference if requested (to see latest data after inserts)
    if refresh_table:
        import pixeltable as pxt
        try:
            # Try to get table name and refresh
            if hasattr(table, 'name'):
                table = pxt.get_table(table.name)
            elif hasattr(table, '_name'):
                table = pxt.get_table(table._name)
            # If table has a path/identifier, try to use it
            elif hasattr(table, 'path'):
                table = pxt.get_table(table.path)
        except Exception:
            pass  # If refresh fails, use original table
    
    # Get total rows (only need to count once) - fresh query
    total_rows = table.select().count()
    
    # Get stats for each model
    model_stats = {}
    for model_column_name in model_columns:
        stats = get_transcription_stats(table, model_column_name, refresh_table=False)  # Already refreshed above
        if stats.get("column_exists"):
            model_stats[model_column_name] = {
                "rows_with_transcription": stats["rows_with_transcription"],
                "rows_without_transcription": stats["rows_without_transcription"],
                "completion_percentage": stats["completion_percentage"]
            }
        else:
            model_stats[model_column_name] = {
                "error": stats.get("error", "Column not found")
            }
    
    return {
        "total_rows": total_rows,
        "models": model_stats
    }


def print_all_model_stats(all_stats, include_inference_times=True, table=None):
    """Print statistics for all models in a compact format"""
    print("\n" + "="*60)
    print("Transcription Status Summary")
    print("="*60)
    print(f"Total rows: {all_stats['total_rows']:,}")
    print("\nModel Transcription Counts:")
    
    for model_name, stats in all_stats['models'].items():
        if "error" in stats:
            print(f"  {model_name:20s}: ⚠️  {stats['error']}")
        else:
            completed = stats['rows_with_transcription']
            pct = stats['completion_percentage']
            print(f"  {model_name:20s}: {completed:6,} transcriptions ({pct:5.2f}%)")
            
            # Add inference time stats if requested and table is provided
            if include_inference_times and table is not None and completed > 0:
                inference_stats = get_inference_time_stats(table, model_name)
                if inference_stats['count'] > 0:
                    print(f"    {'':20s}  Mean inference: {inference_stats['mean']}s, Total: {inference_stats['total']}s")
    
    print("="*60 + "\n")


# Note: Model configuration has been moved to db_helpers.py
# Import get_whisper_model_name from there instead


def extract_inference_times(table, model_column_name="whisper_base"):
    """
    Extract inference times from transcription results.
    
    Args:
        table: Pixeltable table object
        model_column_name: Name of the model column to extract from
    
    Returns:
        List of inference times (in seconds) for rows with transcriptions
    """
    if not hasattr(table, model_column_name):
        return []
    
    model_column = getattr(table, model_column_name)
    
    # Get rows with transcriptions (not None)
    # We don't need the ID column for this, just the model column
    rows_with_transcription = table.select(model_column).where(model_column != None).collect()
    
    inference_times = []
    for row in rows_with_transcription:
        transcription_data = row[model_column_name]
        if isinstance(transcription_data, dict) and "inference_time_seconds" in transcription_data:
            inference_times.append(transcription_data["inference_time_seconds"])
    
    return inference_times


def get_inference_time_stats(table, model_column_name="whisper_base"):
    """
    Get statistics about inference times for a model.
    
    Args:
        table: Pixeltable table object
        model_column_name: Name of the model column
    
    Returns:
        dict with inference time statistics
    """
    inference_times = extract_inference_times(table, model_column_name)
    
    if not inference_times:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "total": None
        }
    
    import statistics
    
    return {
        "count": len(inference_times),
        "mean": round(statistics.mean(inference_times), 3),
        "median": round(statistics.median(inference_times), 3),
        "min": round(min(inference_times), 3),
        "max": round(max(inference_times), 3),
        "total": round(sum(inference_times), 3)
    }


def print_inference_time_stats(stats):
    """Print inference time statistics"""
    if stats["count"] == 0:
        print("  No inference time data available")
        return
    
    print(f"  Inference Time Statistics:")
    print(f"    Count: {stats['count']:,}")
    print(f"    Mean: {stats['mean']}s")
    print(f"    Median: {stats['median']}s")
    print(f"    Min: {stats['min']}s")
    print(f"    Max: {stats['max']}s")
    print(f"    Total: {stats['total']}s")

