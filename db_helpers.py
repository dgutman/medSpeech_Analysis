"""
Database helper functions for pixeltable operations.

Integrates with the refactored config system for consistent table naming.
Model configuration is loaded from models.json for easy modification.
"""
import os
import json
import pixeltable as pxt
from config import get_config
from pathlib import Path

# Get configuration
config = get_config()

# Full table name from config
TABLE_NAME = config.get_table_name()

# Load model configuration from JSON file
def load_model_config():
    """Load model configuration from models.json file."""
    models_file = Path(__file__).parent / "models.json"
    
    if not models_file.exists():
        # Fallback to hardcoded config if file doesn't exist
        print(f"⚠️  Warning: {models_file} not found, using default configuration")
        return {
            "whisper_tiny": "tiny",
            "whisper_base": "base",
            "whisper_small": "small",
            "whisper_medium": "medium",
            "whisper_large": "large",
            "whisper_turbo": "large-v3",
        }, {}
    
    try:
        with open(models_file, 'r') as f:
            config_data = json.load(f)
        
        models = config_data.get('models', {})
        replicas = config_data.get('replicas', {})
        
        # Remove comment keys (those starting with _)
        replicas = {k: v for k, v in replicas.items() if not k.startswith('_')}
        
        return models, replicas
    
    except Exception as e:
        print(f"⚠️  Warning: Error loading {models_file}: {e}")
        print("   Using default configuration")
        return {
            "whisper_tiny": "tiny",
            "whisper_base": "base",
            "whisper_small": "small",
            "whisper_medium": "medium",
            "whisper_large": "large",
            "whisper_turbo": "large-v3",
        }, {}

# Load model and replica configurations
MODEL_CONFIG, REPLICA_CONFIG = load_model_config()

# Combined model configuration (includes both standard models and replicas)
# Set INCLUDE_REPLICAS=True in .env to enable replica columns
INCLUDE_REPLICAS = os.getenv('INCLUDE_REPLICAS', 'false').lower() == 'true'

if INCLUDE_REPLICAS:
    MODEL_CONFIG_FULL = {**MODEL_CONFIG, **REPLICA_CONFIG}
    print(f"ℹ️  Replica columns enabled: {list(REPLICA_CONFIG.keys())}")
else:
    MODEL_CONFIG_FULL = MODEL_CONFIG.copy()

# List of model columns (derived from MODEL_CONFIG_FULL keys)
MODEL_COLUMNS = list(MODEL_CONFIG_FULL.keys())


def get_table(table_name: str = None):
    """
    Get the pixeltable table.
    
    Args:
        table_name: Name of the table (defaults to TABLE_NAME)
    
    Returns:
        Pixeltable table object
    """
    if table_name is None:
        table_name = TABLE_NAME
    return pxt.get_table(table_name)


def ensure_column(table, column_name: str, column_type=pxt.Json, if_exists='ignore'):
    """
    Ensure a column exists in the table.
    
    Args:
        table: Pixeltable table object
        column_name: Name of the column to add
        column_type: Type of the column (default: pxt.Json)
        if_exists: What to do if column exists ('ignore', 'error', etc.)
    
    Returns:
        The table object
    """
    table.add_column(**{column_name: column_type}, if_exists=if_exists)
    return table


def ensure_model_columns(table, model_columns=None):
    """
    Ensure all model columns exist in the table.
    
    Args:
        table: Pixeltable table object
        model_columns: List of model column names (defaults to MODEL_COLUMNS)
    
    Returns:
        The table object
    """
    if model_columns is None:
        model_columns = MODEL_COLUMNS
    
    for column_name in model_columns:
        ensure_column(table, column_name, pxt.Json, if_exists='ignore')
    
    return table


def get_whisper_model_name(model_column_name: str) -> str:
    """
    Get Whisper model name from column name.
    
    Args:
        model_column_name: Name of the model column (e.g., "whisper_base")
    
    Returns:
        Whisper model name (e.g., "base") or "base" as default
    """
    return MODEL_CONFIG_FULL.get(model_column_name, "base")


def initialize_table(table_name: str = None, ensure_models: bool = True):
    """
    Initialize the table and ensure required columns exist.
    
    Args:
        table_name: Name of the table (defaults to TABLE_NAME)
        ensure_models: Whether to ensure model columns exist
    
    Returns:
        Pixeltable table object
    """
    table = get_table(table_name)
    
    if ensure_models:
        ensure_model_columns(table)
    
    return table


def get_column_stats(table_name: str = None):
    """
    Get statistics about all columns in the table, including non-null counts.
    
    Args:
        table_name: Name of the table (defaults to TABLE_NAME)
    
    Returns:
        dict with column statistics
    """
    table = get_table(table_name)
    
    # Get total row count
    total_rows = table.select().count()
    
    # Get all column names
    column_names = [col.name for col in table.columns()]
    
    stats = {
        "table_name": table_name or TABLE_NAME,
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
            
            stats["columns"][col_name] = {
                "non_null": non_null_count,
                "null": null_count,
                "empty_dict": empty_dict_count,
                "empty_list": empty_list_count,
                "dict": dict_count,
                "list": list_count,
                "non_null_pct": round((non_null_count / total_rows * 100) if total_rows > 0 else 0, 2)
            }
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
    
    print("="*80 + "\n")

