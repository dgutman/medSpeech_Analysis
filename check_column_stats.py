#!/usr/bin/env python3
"""
Standalone script to check column statistics in the Pixeltable.
Shows non-null counts, types, and percentages for all columns.
"""
import os
import sys

# Try to load from .env file if it exists (like other scripts)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Set PIXELTABLE_PGDATA to absolute path before importing pixeltable
# This matches the config used in db_helpers.py, transcribe_via_raycontainer.py, etc.
if 'PIXELTABLE_PGDATA' not in os.environ:
    os.environ['PIXELTABLE_PGDATA'] = '/scr/dagutman/devel/medSpeech_Analysis/.pxtData'

# Ensure we're using absolute paths
os.chdir('/scr/dagutman/devel/medSpeech_Analysis')

import pixeltable as pxt
import statistics

# Table configuration
TABLE_NAME = 'medSpeechAnalysis_hf_ray.hani89_asr_dataset'

# Model columns that might have inference_time_seconds
MODEL_COLUMNS = [
    "whisper_tiny", "whisper_base", "whisper_small", 
    "whisper_medium", "whisper_large", "whisper_turbo",
    # Replica columns for stability testing
    "tiny_rep1", "tiny_rep2", "tiny_rep3", "tiny_rep4", "tiny_rep5"
]


def get_column_stats(table_name: str = None):
    """
    Get statistics about all columns in the table, including non-null counts.
    
    Args:
        table_name: Name of the table (defaults to TABLE_NAME)
    
    Returns:
        dict with column statistics
    """
    if table_name is None:
        table_name = TABLE_NAME
    
    table = pxt.get_table(table_name)
    
    # Get total row count
    total_rows = table.select().count()
    
    # Get all column names from a sample row (most reliable method)
    sample = table.select().limit(1).collect()
    if sample and len(sample) > 0:
        # Get column names from the first row's keys
        if isinstance(sample[0], dict):
            column_names = list(sample[0].keys())
        else:
            # Try to get keys another way
            column_names = list(sample[0]._asdict().keys()) if hasattr(sample[0], '_asdict') else []
    else:
        # If no rows, try to get column names from table schema
        # This is a fallback - we'll get an empty list but that's okay
        column_names = []
    
    stats = {
        "table_name": table_name,
        "total_rows": total_rows,
        "columns": {}
    }
    
    # For each column, count non-null values
    for col_name in column_names:
        try:
            col_ref = getattr(table, col_name)
            
            # Collect all rows for this column
            all_rows = table.select(col_ref).collect()
            
            non_null_count = 0
            null_count = 0
            empty_dict_count = 0
            empty_list_count = 0
            dict_count = 0
            list_count = 0
            
            for row in all_rows:
                # Try multiple ways to access the column value
                col_value = None
                if hasattr(row, 'get'):
                    col_value = row.get(col_name)
                elif isinstance(row, dict):
                    col_value = row.get(col_name)
                else:
                    col_value = getattr(row, col_name, None)
                
                if col_value is None:
                    null_count += 1
                elif isinstance(col_value, dict):
                    dict_count += 1
                    if len(col_value) == 0:
                        empty_dict_count += 1
                    else:
                        non_null_count += 1
                elif isinstance(col_value, list):
                    list_count += 1
                    if len(col_value) == 0:
                        empty_list_count += 1
                    else:
                        non_null_count += 1
                else:
                    # Other types (string, int, etc.)
                    non_null_count += 1
            
            col_stats = {
                "non_null": non_null_count,
                "null": null_count,
                "empty_dict": empty_dict_count,
                "empty_list": empty_list_count,
                "dict": dict_count,
                "list": list_count,
                "non_null_pct": round((non_null_count / total_rows * 100) if total_rows > 0 else 0, 2)
            }
            
            # If this is a model column, extract inference time statistics
            if col_name in MODEL_COLUMNS and non_null_count > 0:
                inference_times = []
                for row in all_rows:
                    col_value = None
                    if hasattr(row, 'get'):
                        col_value = row.get(col_name)
                    elif isinstance(row, dict):
                        col_value = row.get(col_name)
                    else:
                        col_value = getattr(row, col_name, None)
                    
                    # Extract inference_time_seconds from dict
                    if isinstance(col_value, dict) and "inference_time_seconds" in col_value:
                        inference_times.append(col_value["inference_time_seconds"])
                    # Handle list of dicts (legacy format)
                    elif isinstance(col_value, list):
                        for item in col_value:
                            if isinstance(item, dict) and "inference_time_seconds" in item:
                                inference_times.append(item["inference_time_seconds"])
                
                if inference_times:
                    col_stats["inference_time"] = {
                        "count": len(inference_times),
                        "mean": round(statistics.mean(inference_times), 3),
                        "median": round(statistics.median(inference_times), 3),
                        "min": round(min(inference_times), 3),
                        "max": round(max(inference_times), 3),
                        "total": round(sum(inference_times), 3)
                    }
                else:
                    col_stats["inference_time"] = None
            
            stats["columns"][col_name] = col_stats
        except Exception as e:
            stats["columns"][col_name] = {
                "error": str(e)
            }
    
    return stats


def print_column_stats(table_name: str = None):
    """
    Print statistics about all columns in the table.
    
    Args:
        table_name: Name of the table (defaults to TABLE_NAME)
    """
    stats = get_column_stats(table_name)
    
    print("\n" + "="*80)
    print(f"Column Statistics: {stats['table_name']}")
    print("="*80)
    print(f"Total rows: {stats['total_rows']:,}")
    print("\nColumn Details:")
    print("-"*80)
    
    for col_name, col_stats in stats["columns"].items():
        if "error" in col_stats:
            print(f"  {col_name:30s}: ⚠️  Error: {col_stats['error']}")
        else:
            print(f"  {col_name:30s}: {col_stats['non_null']:6,} non-null ({col_stats['non_null_pct']:5.2f}%)")
            if col_stats['null'] > 0:
                print(f"    {'':30s}   {col_stats['null']:6,} null")
            if col_stats['empty_dict'] > 0:
                print(f"    {'':30s}   {col_stats['empty_dict']:6,} empty dicts")
            if col_stats['empty_list'] > 0:
                print(f"    {'':30s}   {col_stats['empty_list']:6,} empty lists")
            if col_stats['dict'] > 0:
                print(f"    {'':30s}   {col_stats['dict']:6,} dicts (total)")
            if col_stats['list'] > 0:
                print(f"    {'':30s}   {col_stats['list']:6,} lists (total)")
            
            # Show inference time stats if available
            if "inference_time" in col_stats and col_stats["inference_time"] is not None:
                inf_stats = col_stats["inference_time"]
                print(f"    {'':30s}   Inference time: mean={inf_stats['mean']}s, median={inf_stats['median']}s")
                print(f"    {'':30s}     (min={inf_stats['min']}s, max={inf_stats['max']}s, n={inf_stats['count']:,})")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Allow table name to be passed as command line argument
    table_name = sys.argv[1] if len(sys.argv) > 1 else None
    print_column_stats(table_name)

