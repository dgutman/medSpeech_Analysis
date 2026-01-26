"""
Visualization cache utility for immutable data.

Since the data is immutable once the container starts, we can cache
generated visualizations to avoid recomputing them on every tab switch.
"""

import os
import json
import pickle
import hashlib
import logging
from typing import Any, Optional, Dict
import pandas as pd

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = os.environ.get('CACHE_DIR', './cache')
VIZ_CACHE_DIR = os.path.join(CACHE_DIR, 'viz_cache')
try:
    os.makedirs(VIZ_CACHE_DIR, exist_ok=True)
except (PermissionError, OSError) as e:
    # Fallback to /tmp if we can't create in CACHE_DIR
    VIZ_CACHE_DIR = os.path.join('/tmp', 'viz_cache')
    os.makedirs(VIZ_CACHE_DIR, exist_ok=True)
    logger.warning(f"Using /tmp for visualization cache due to permission error: {e}")


def get_data_hash(df: pd.DataFrame) -> str:
    """Generate a hash of the dataframe to use as cache key"""
    # Use shape, columns, and a sample of data for hashing
    # This is fast and should be unique enough for immutable data
    data_str = f"{df.shape}_{list(df.columns)}_{df.head(10).to_string()}"
    return hashlib.md5(data_str.encode()).hexdigest()


def get_cache_key(tab_name: str, df: pd.DataFrame, **kwargs) -> str:
    """Generate a cache key for a visualization"""
    # Include tab name, data hash, and any additional parameters
    params_str = json.dumps(kwargs, sort_keys=True)
    data_hash = get_data_hash(df)
    key_str = f"{tab_name}_{data_hash}_{params_str}"
    return hashlib.md5(key_str.encode()).hexdigest()


def get_cache_path(cache_key: str) -> str:
    """Get the file path for a cache entry"""
    return os.path.join(VIZ_CACHE_DIR, f"{cache_key}.pkl")


def cache_visualization(tab_name: str, df: pd.DataFrame, visualization: Any, **kwargs) -> None:
    """Cache a visualization result"""
    try:
        cache_key = get_cache_key(tab_name, df, **kwargs)
        cache_path = get_cache_path(cache_key)
        
        # Store the visualization (Plotly figures can be serialized)
        with open(cache_path, 'wb') as f:
            pickle.dump(visualization, f)
        
        logger.info(f"Cached visualization for {tab_name} (key: {cache_key[:8]}...)")
    except Exception as e:
        logger.warning(f"Failed to cache visualization for {tab_name}: {e}")


def get_cached_visualization(tab_name: str, df: pd.DataFrame, **kwargs) -> Optional[Any]:
    """Retrieve a cached visualization if it exists"""
    try:
        cache_key = get_cache_key(tab_name, df, **kwargs)
        cache_path = get_cache_path(cache_key)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                visualization = pickle.load(f)
            logger.info(f"Retrieved cached visualization for {tab_name} (key: {cache_key[:8]}...)")
            return visualization
        else:
            logger.debug(f"No cache found for {tab_name} (key: {cache_key[:8]}...)")
            return None
    except Exception as e:
        logger.warning(f"Failed to retrieve cached visualization for {tab_name}: {e}")
        return None


def cache_function(tab_name: str, **cache_kwargs):
    """
    Decorator to cache function results based on dataframe hash.
    
    Usage:
        @cache_function('analytics-tab')
        def create_analytics_tab(df):
            # ... generate visualization ...
            return visualization
    """
    def decorator(func):
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            # Merge cache_kwargs with function kwargs for cache key
            all_kwargs = {**cache_kwargs, **kwargs}
            
            # Try to get from cache
            cached = get_cached_visualization(tab_name, df, **all_kwargs)
            if cached is not None:
                return cached
            
            # Generate visualization
            result = func(df, *args, **kwargs)
            
            # Cache the result
            cache_visualization(tab_name, df, result, **all_kwargs)
            
            return result
        return wrapper
    return decorator


def clear_viz_cache():
    """Clear all cached visualizations"""
    try:
        for filename in os.listdir(VIZ_CACHE_DIR):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(VIZ_CACHE_DIR, filename))
        logger.info("Cleared visualization cache")
    except Exception as e:
        logger.warning(f"Failed to clear visualization cache: {e}")
