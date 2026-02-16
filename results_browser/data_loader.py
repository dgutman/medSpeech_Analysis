"""
Data loading utilities for the results browser.

This module handles all data loading from Pixeltable, including pagination,
caching, and connection management.
"""

import os

# Set PIXELTABLE_PGDATA before importing pixeltable (container uses fixed path)
# In container, use default location; on host, this will be overridden by .env
if 'PIXELTABLE_PGDATA' not in os.environ or not os.environ.get('PIXELTABLE_PGDATA'):
    # Container default: use appuser's home directory
    container_pgdata = '/home/appuser/.pixeltable/pgdata'
    if os.path.exists('/home/appuser'):
        os.environ['PIXELTABLE_PGDATA'] = container_pgdata

import operator
import pandas as pd
import pixeltable as pxt
from datetime import datetime
from functools import reduce
import logging
import time

logger = logging.getLogger(__name__)

# Global variables for caching
cached_data = None
cache_timestamp = None
_cached_env_vars = None


def load_env_vars():
    """Load environment variables from .env file (cached to avoid circular initialization)"""
    global _cached_env_vars
    
    # Return cached version if available
    if _cached_env_vars is not None:
        return _cached_env_vars
    
    env_vars = {}

    # Prefer /app/.env inside container (docker-compose mounts it there)
    candidate_paths = [
        "/app/.env",
        os.path.join(os.path.dirname(__file__), ".env"),
        os.path.join(os.path.dirname(__file__), "..", ".env"),
    ]
    env_file = next((p for p in candidate_paths if os.path.exists(p)), None)

    if env_file is not None:
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key] = value
                    # Only set env var if it's not already set
                    if key not in os.environ:
                        os.environ[key] = value
    
    # Cache the result
    _cached_env_vars = env_vars
    return env_vars


def get_local_table_name():
    """Derive local table name from dataset URL in .env"""
    env_vars = load_env_vars()
    dataset_url = env_vars.get('PIXELTABLE_DATASET_URL', 'pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare')
    
    # Extract table name from URL: pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare
    # Use the last part of the path as the table name, with 'local_' prefix
    if '/' in dataset_url:
        # Get the last part after the last /
        table_name = dataset_url.split('/')[-1]
        # Remove any version info (e.g., :main)
        if ':' in table_name:
            table_name = table_name.split(':')[0]
        return f'local_{table_name}'
    else:
        # Fallback: use a default name
        return 'local_transcribe_compare'


def load_cached_data():
    """Load data from local cache"""
    cache_dir = os.environ.get('CACHE_DIR', './cache')
    cache_file = os.path.join(cache_dir, 'dataset_cache.pkl')
    
    logger.info(f"Looking for cache file at: {cache_file}")
    logger.info(f"File exists: {os.path.exists(cache_file)}")
    
    if not os.path.exists(cache_file):
        logger.info("No cached data found")
        return None
    
    try:
        df = pd.read_pickle(cache_file)
        record_count = len(df) if hasattr(df, '__len__') else df.shape[0] if hasattr(df, 'shape') else 'unknown'
        logger.info(f"Loaded {record_count} records from cache")
        return df
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}")
        return None


def initialize_connection():
    """Initialize Pixeltable connection at startup (e.g. warm cache, validate table exists)."""
    get_pixeltable_table()
    logger.info("Pixeltable connection initialized.")


def get_pixeltable_table():
    """Get the pixeltable table connection with retry logic"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Load environment variables (cached, won't cause circular initialization)
            env_vars = load_env_vars()
            # Only set API key if not already set (avoid circular initialization)
            if 'PIXELTABLE_API_KEY' in env_vars and 'PIXELTABLE_API_KEY' not in os.environ:
                os.environ['PIXELTABLE_API_KEY'] = env_vars['PIXELTABLE_API_KEY']
            
            if attempt > 0:
                logger.info(f"Retrying connection to local Pixeltable replica (attempt {attempt + 1}/{max_retries})...")
            else:
                logger.info("Connecting to local Pixeltable replica...")
            
            table_name = get_local_table_name()
            
            # Try to get the table with the expected name
            try:
                local_table = pxt.get_table(table_name)
                return local_table
            except Exception as e:
                # If that fails, check metadata to see what the actual table name is
                cache_dir = os.environ.get('CACHE_DIR', './cache')
                metadata_file = os.path.join(cache_dir, 'local_table_metadata.json')
                if os.path.exists(metadata_file):
                    try:
                        import json
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        actual_table_name = metadata.get('actual_table_name', table_name)
                        if actual_table_name != table_name:
                            logger.info(f"Table name mismatch: expected '{table_name}', found '{actual_table_name}' in metadata")
                            local_table = pxt.get_table(actual_table_name)
                            return local_table
                    except Exception as e2:
                        logger.warning(f"Could not read metadata file: {e2}")
                
                # If metadata doesn't help, try to locate the correct local replica.
                # We should NOT silently fall back to an unrelated local_* table (that can load the wrong dataset).
                try:
                    all_tables = pxt.list_tables()
                    logger.info(f"Available tables: {all_tables}")
                    # Look for any table starting with 'local_'
                    local_tables = [t for t in all_tables if t.startswith('local_')]
                    if local_tables:
                        logger.info(f"Found local tables: {local_tables}")
                        # Prefer tables that look like our dataset name
                        for local_table_name in local_tables:
                            if 'transcribe' in local_table_name.lower() or 'compare' in local_table_name.lower():
                                logger.info(f"Trying table name: {local_table_name}")
                                local_table = pxt.get_table(local_table_name)
                                logger.info(f"✅ Found table: {local_table_name}")
                                return local_table
                except Exception as e2:
                    logger.warning(f"Could not list tables: {e2}")
                
                # Fallback: try common alternative names
                alternative_names = [
                    'local_transcribe_compare',  # Expected name
                    'local_hani89',  # Based on directory name
                    'local_hani89_asr_data_reload',  # Full directory name
                ]
                for alt_name in alternative_names:
                    try:
                        logger.info(f"Trying alternative table name: {alt_name}")
                        local_table = pxt.get_table(alt_name)
                        logger.info(f"✅ Found table with alternative name: {alt_name}")
                        return local_table
                    except:
                        continue
                
                # Re-raise the original error if nothing worked
                raise
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error connecting to local Pixeltable replica (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Error connecting to local Pixeltable replica after {max_retries} attempts: {e}")
                raise
    
    return None


def get_total_count(filters=None):
    """Get total count of records in the table, optionally with filters"""
    try:
        local_table = get_pixeltable_table()
        query = local_table.select()
        if filters and 'split' in filters and filters['split']:
            query = query.where(local_table.split == filters['split'])
        count = query.count()
        logger.debug(f"Total records in table (with filters {filters}): {count}")
        return count
    except Exception as e:
        logger.error(f"Error getting total count: {e}")
        return 0


def get_split_counts():
    """Get count of records by split using efficient database-level queries"""
    try:
        local_table = get_pixeltable_table()
        
        # Get distinct split values (fast database query)
        df_splits = local_table.select(local_table.split).distinct().collect().to_pandas()
        
        if df_splits is None or df_splits.empty or 'split' not in df_splits.columns:
            logger.warning("No split column found or empty result")
            return {}
        
        split_values = df_splits['split'].unique().tolist()
        logger.info(f"Found split values: {split_values}")
        split_counts = {}
        
        # Count each split value using database queries (fast with WHERE clause)
        for split_val in split_values:
            try:
                count = local_table.select().where(local_table.split == split_val).count()
                split_counts[split_val] = count
                logger.info(f"Split '{split_val}': {count} records (from database query)")
            except Exception as e:
                # Fallback: if where() doesn't work, use pandas filtering on split column only
                logger.warning(f"Could not use database-level count for {split_val}, using fallback: {e}")
                # Load ALL split values (no limit) to count accurately
                df_all = local_table.select(local_table.split).collect().to_pandas()
                if df_all is not None and 'split' in df_all.columns:
                    count = len(df_all[df_all['split'] == split_val])
                    split_counts[split_val] = count
                    logger.info(f"Split '{split_val}': {count} records (from pandas fallback, loaded {len(df_all)} total rows)")
                else:
                    split_counts[split_val] = 0
                    logger.warning(f"Could not count split '{split_val}', defaulting to 0")
        
        logger.info(f"Final split counts: {split_counts}")
        return split_counts
    except Exception as e:
        logger.error(f"Error getting split counts: {e}", exc_info=True)
        return {}


def load_pixeltable_data_paginated(limit=None, offset=0, filters=None):
    """
    Load paginated data from local pixeltable replica
    
    Args:
        limit: Number of records to retrieve (None for all)
        offset: Number of records to skip
        filters: Dictionary of filters to apply (e.g., {'split': 'train'})
    
    Returns:
        pandas DataFrame with the requested page of data
    """
    try:
        local_table = get_pixeltable_table()
        
        # Build query
        query = local_table.select()

        # Always use a stable ordering for pagination.
        # Prefer row_idx if present (added during preload), otherwise fall back to filename/id.
        try:
            if hasattr(local_table, 'row_idx'):
                query = query.order_by(local_table.row_idx)
            elif hasattr(local_table, 'filename'):
                query = query.order_by(local_table.filename)
            elif hasattr(local_table, 'id'):
                query = query.order_by(local_table.id)
        except Exception:
            # If ordering fails for any reason, proceed without it (pagination may be non-deterministic).
            pass
        
        # Apply filters at database level if provided
        if filters:
            if 'split' in filters and filters['split']:
                query = query.where(local_table.split == filters['split'])
            if 'search_term' in filters and filters['search_term']:
                search_term = filters['search_term']
                query = query.where(local_table.transcription.ilike(f'%{search_term}%'))
        
        # Apply pagination: offset first (skip rows), then limit (take rows). Order matters.
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)
        
        # Execute query and convert to pandas
        result = query.collect().to_pandas()
        if offset >= 50:
            logger.info(
                f"Paginated load: offset={offset} limit={limit} -> {len(result)} rows (filters={filters})"
            )
        else:
            logger.debug(f"Loaded {len(result)} records (offset={offset}, limit={limit}, filters={filters})")
        return result

    except Exception as e:
        logger.error(f"Error loading paginated data: {e}")
        return pd.DataFrame()


def load_paginated_index(filters=None):
    """
    Load the full ordered list of row IDs (and optionally row_idx) for the current filter.
    Used for "local paging": we have no offset in Pixeltable, so we fetch the full index
    and slice it client-side to know which IDs to fetch for each page.

    Returns:
        list: Ordered list of row IDs (e.g. strings or ints) matching filters.
    """
    try:
        local_table = get_pixeltable_table()
        # Select only id (and row_idx for ordering if present)
        if hasattr(local_table, "row_idx"):
            query = local_table.select(local_table.id, local_table.row_idx).order_by(local_table.row_idx)
        elif hasattr(local_table, "id"):
            query = local_table.select(local_table.id).order_by(local_table.id)
        else:
            logger.warning("Table has no id column for paginated index")
            return []

        if filters:
            if "split" in filters and filters["split"]:
                query = query.where(local_table.split == filters["split"])
            if "search_term" in filters and filters["search_term"]:
                search_term = filters["search_term"]
                query = query.where(local_table.transcription.ilike(f"%{search_term}%"))

        df = query.collect().to_pandas()
        if df is None or df.empty or "id" not in df.columns:
            return []
        ids = df["id"].tolist()
        logger.info(f"Loaded paginated index: {len(ids)} row IDs (filters={filters})")
        return ids
    except Exception as e:
        logger.error(f"Error loading paginated index: {e}", exc_info=True)
        return []


def load_rows_by_ids(ids):
    """
    Load full row data for the given list of row IDs.
    Pixeltable has no offset; we use this after slicing the full index for the current page.
    """
    if not ids:
        return pd.DataFrame()
    try:
        local_table = get_pixeltable_table()
        id_col = local_table.id
        # Build (id == id1) | (id == id2) | ... (Pixeltable may not have isin())
        condition = reduce(operator.or_, (id_col == i for i in ids))
        query = local_table.select().where(condition)
        result = query.collect().to_pandas()
        if result is None or result.empty:
            return pd.DataFrame()
        # Preserve order of requested ids (collect() may return in arbitrary order)
        if "id" in result.columns and len(ids) > 1:
            id_to_order = {vid: i for i, vid in enumerate(ids)}
            result["_order"] = result["id"].map(id_to_order)
            result = result.sort_values("_order").drop(columns=["_order"])
        logger.debug(f"Loaded {len(result)} rows by ids (requested {len(ids)})")
        return result
    except Exception as e:
        logger.error(f"Error loading rows by ids: {e}", exc_info=True)
        return pd.DataFrame()


def load_pixeltable_data():
    """Load ALL data from local pixeltable replica (for backward compatibility)"""
    # This function is kept for backward compatibility but should be avoided for large datasets
    logger.warning("load_pixeltable_data() loads all data - consider using pagination for large datasets")
    
    global cached_data, cache_timestamp
    
    try:
        local_table = get_pixeltable_table()
        logger.info("Loading all data from local table (consider using pagination)...")
        df = local_table.collect().to_pandas()
        
        # Remove duplicate entries based on 'id' column (keep first occurrence)
        if 'id' in df.columns:
            initial_count = len(df)
            df = df.drop_duplicates(subset=['id'], keep='first')
            final_count = len(df)
            if initial_count != final_count:
                logger.warning(f"Removed {initial_count - final_count} duplicate entries (based on 'id' column)")
        
        cached_data = df
        cache_timestamp = datetime.now()
        
        record_count = len(df) if hasattr(df, '__len__') else df.shape[0] if hasattr(df, 'shape') else 'unknown'
        logger.info(f"Loaded {record_count} records from local Pixeltable replica")
        return df
        
    except Exception as e:
        logger.error(f"Error loading from local Pixeltable replica: {e}")
        logger.info("Falling back to sample data")
        return create_sample_data()


def create_sample_data():
    """Create sample data for demonstration"""
    return pd.DataFrame({
        'id': [f'sample_{i}' for i in range(10)],
        'transcription': [f'Sample transcription {i}' for i in range(10)],
        'split': ['train'] * 5 + ['test'] * 5,
        'filePath': [f'/path/to/audio_{i}.wav' for i in range(10)],
        'whisper_tinyEn_transcription': [f'Whisper result {i}' for i in range(10)]
    })


def get_data():
    """Get data with caching"""
    global cached_data, cache_timestamp
    
    if cached_data is None or cache_timestamp is None:
        return load_pixeltable_data()
    
    # Check if cache is older than 5 minutes
    if (datetime.now() - cache_timestamp).seconds > 300:
        return load_pixeltable_data()
    
    return cached_data



