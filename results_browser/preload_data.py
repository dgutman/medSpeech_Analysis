#!/usr/bin/env python3
"""
Data preloader script for the Medical Speech Analysis Results Browser.
This script downloads and caches the pixeltable dataset locally.
"""

import os
import json
import pandas as pd
from datetime import datetime
import logging
from typing import Optional, Dict, Any
import pickle
from dotenv import load_dotenv
import math

# Load environment variables BEFORE importing pixeltable
# Try both locations for .env file
load_dotenv('/app/.env')
load_dotenv()  # Also try default location

# Set PIXELTABLE_PGDATA before importing pixeltable (container uses fixed path)
# In container, use default location; on host, this will be overridden by .env
if 'PIXELTABLE_PGDATA' not in os.environ or not os.environ.get('PIXELTABLE_PGDATA'):
    # Container default: use appuser's home directory
    container_pgdata = '/home/appuser/.pixeltable/pgdata'
    if os.path.exists('/home/appuser'):
        os.environ['PIXELTABLE_PGDATA'] = container_pgdata

import pixeltable as pxt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_env_vars():
    """Load environment variables from .env file using python-dotenv"""
    # First check if we're in a build context with environment variables already set
    if 'PIXELTABLE_API_KEY' in os.environ and os.environ['PIXELTABLE_API_KEY']:
        logger.info("Using environment variables from build context")
        return dict(os.environ)
    
    # Try to load from .env file - check both possible locations
    env_files = [
        '/app/.env',  # Container location
        os.path.join(os.path.dirname(__file__), '..', '.env')  # Relative path
    ]
    
    for env_file in env_files:
        if os.path.exists(env_file):
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
            break
    else:
        logger.warning(f".env file not found in any of these locations: {env_files}")
    
    # Return environment variables as dict
    return dict(os.environ)


def get_local_table_name(dataset_url: str = None):
    """Derive local table name from dataset URL"""
    if dataset_url is None:
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

def setup_pixeltable():
    """Setup pixeltable with API key"""
    env_vars = load_env_vars()
    
    if 'PIXELTABLE_API_KEY' in env_vars:
        os.environ['PIXELTABLE_API_KEY'] = env_vars['PIXELTABLE_API_KEY']
        logger.info("PIXELTABLE_API_KEY loaded from .env file")
        logger.info(f"API Key value: {env_vars['PIXELTABLE_API_KEY'][:10]}...")  # Show first 10 chars
    elif 'PIXELTABLE_API_KEY' in os.environ and os.environ.get('PIXELTABLE_API_KEY'):
        # Build args / runtime env can provide this without being present in .env
        logger.info("PIXELTABLE_API_KEY found in environment (not from .env file)")
        logger.info(f"API Key value: {os.environ['PIXELTABLE_API_KEY'][:10]}...")
    else:
        logger.warning("PIXELTABLE_API_KEY not found in .env file or environment")
    
    # Verify the environment variable is set
    if 'PIXELTABLE_API_KEY' in os.environ:
        logger.info(f"Environment variable PIXELTABLE_API_KEY is set: {os.environ['PIXELTABLE_API_KEY'][:10]}...")
    else:
        logger.error("Environment variable PIXELTABLE_API_KEY is NOT set")
    
    # Pixeltable doesn't need explicit initialization
    logger.info("Pixeltable ready for use")
    return True

def download_dataset(dataset_url: str, cache_dir: str) -> Optional[pd.DataFrame]:
    """Create local replica of remote pixeltable dataset"""
    try:
        logger.info(f"Creating replica of remote table: {dataset_url}")
        logger.info(f"Pixeltable version: {pxt.__version__}")
        
        # Derive local table name from dataset URL
        local_table_name = get_local_table_name(dataset_url)
        logger.info(f"Using local table name: {local_table_name}")
        
        # Create a local replica of the remote table
        local_table = pxt.replicate(
            remote_uri=dataset_url,
            local_path=local_table_name
        )
        
        logger.info("Pulling latest data from remote table...")
        local_table.pull()
        
        # Optional: add a stable integer index for paging (local replica only)
        # This is useful for deterministic pagination + UI row numbering.
        try:
            # Prefer filename ordering if present, otherwise fall back to id.
            has_id = hasattr(local_table, 'id')
            has_filename = hasattr(local_table, 'filename')
            has_row_idx = hasattr(local_table, 'row_idx')

            if not has_row_idx:
                logger.info("Adding local paging index column: row_idx (Int)")
                local_table.add_column(row_idx=pxt.Int)

            if has_id:
                if has_filename:
                    ids_df = local_table.select(local_table.id, local_table.filename).order_by(local_table.filename).collect().to_pandas()
                    id_series = ids_df['id'].astype(str)
                else:
                    ids_df = local_table.select(local_table.id).order_by(local_table.id).collect().to_pandas()
                    id_series = ids_df['id'].astype(str)

                updates = [{'id': rid, 'row_idx': i} for i, rid in enumerate(id_series.tolist())]
                # Batch updates in chunks to keep memory bounded
                chunk_size = 2000
                for chunk_i in range(0, len(updates), chunk_size):
                    chunk = updates[chunk_i:chunk_i + chunk_size]
                    local_table.batch_update(chunk)
                logger.info(f"✅ Populated row_idx for {len(updates)} rows")
            else:
                logger.warning("Could not populate row_idx (no 'id' column found on table)")
        except Exception as e:
            logger.warning(f"Failed to add/populate row_idx column (continuing without it): {e}")

        # Verify the local table works by getting a sample
        logger.info("Verifying local table...")
        sample_df = local_table.select().limit(5).collect().to_pandas()
        logger.info(f"Local table created successfully with {len(sample_df)} sample records")
        logger.info(f"Columns: {list(sample_df.columns)}")
        
        # Save metadata about the local table
        metadata_file = os.path.join(cache_dir, 'local_table_metadata.json')
        actual_table_name = getattr(local_table, 'name', None) or getattr(local_table, '_name', None) or local_table_name
        metadata = {
            'dataset_url': dataset_url,
            'local_table_name': local_table_name,
            'actual_table_name': actual_table_name,
            'replication_timestamp': datetime.now().isoformat(),
            'pixeltable_version': pxt.__version__,
            'status': 'replicated_successfully'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Local table metadata saved to: {metadata_file}")
        logger.info("✅ Local Pixeltable replica created successfully")
        
        return sample_df  # Return sample for verification
        
    except Exception as e:
        logger.error(f"Failed to create local replica: {e}")
        return None

def load_cached_data(cache_dir: str) -> Optional[pd.DataFrame]:
    """Load data from local cache"""
    cache_file = os.path.join(cache_dir, 'dataset_cache.pkl')
    metadata_file = os.path.join(cache_dir, 'dataset_metadata.json')
    
    if not os.path.exists(cache_file):
        logger.info("No cached data found")
        return None
    
    try:
        # Load metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Loading cached data from {metadata['download_timestamp']} ({metadata['num_records']} records)")
        
        # Load DataFrame
        df = pd.read_pickle(cache_file)
        logger.info(f"Loaded {len(df)} records from cache")
        return df
        
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}")
        return None

def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration"""
    logger.info("Creating sample data for demonstration")
    
    sample_data = {
        'id': [f'sample_{i:03d}' for i in range(50)],
        'transcription': [
            f'This is sample medical transcription number {i} for demonstration purposes.'
            for i in range(50)
        ],
        'split': ['train'] * 30 + ['test'] * 15 + ['validation'] * 5,
        'filePath': [f'/path/to/audio_{i:03d}.wav' for i in range(50)],
        'whisper_tinyEn_transcription': [
            f'Whisper tiny model transcription for sample {i}'
            for i in range(50)
        ],
        'whisper_smallEn_transcription': [
            f'Whisper small model transcription for sample {i}'
            for i in range(50)
        ]
    }
    
    return pd.DataFrame(sample_data)

def main():
    """Main function to create local pixeltable replica"""
    # Load environment variables
    env_vars = load_env_vars()
    
    # Get configuration
    dataset_url = env_vars.get('PIXELTABLE_DATASET_URL', 'pxt://speech-to-text-analytics:main/hani89_asr_data_reload/transcribe_compare')
    cache_dir = env_vars.get('CACHE_DIR', './cache')
    data_dir = env_vars.get('DATA_DIR', './data')
    
    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    logger.info(f"Cache directory: {cache_dir}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Dataset URL: {dataset_url}")
    
    # Check if local replica already exists and matches current dataset URL
    try:
        if setup_pixeltable():
            # Derive local table name from dataset URL
            local_table_name = get_local_table_name(dataset_url)
            logger.info(f"Using local table name: {local_table_name}")
            
            # Try to connect to existing local table
            try:
                local_table = pxt.get_table(local_table_name)
                
                # Check if metadata exists and matches current dataset URL
                metadata_file = os.path.join(cache_dir, 'local_table_metadata.json')
                needs_replication = True
                
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            if metadata.get('dataset_url') == dataset_url:
                                logger.info("✅ Local Pixeltable replica exists and matches current dataset URL")
                                logger.info(f"   Dataset URL: {dataset_url}")
                                logger.info("   Skipping replication (use --force-reload to force re-replication)")
                                return None  # Data already exists and matches
                            else:
                                logger.info(f"⚠️  Local replica exists but dataset URL changed")
                                logger.info(f"   Old URL: {metadata.get('dataset_url')}")
                                logger.info(f"   New URL: {dataset_url}")
                                logger.info("   Will delete old replica and create new one...")
                                # Delete the old local table
                                try:
                                    pxt.drop_table(local_table_name)
                                    logger.info(f"   Deleted old local replica: {local_table_name}")
                                except Exception as e:
                                    logger.warning(f"   Could not delete old replica: {e}")
                    except Exception as e:
                        logger.warning(f"Could not read metadata file: {e}, will re-replicate")
                else:
                    logger.info("⚠️  Local replica exists but no metadata found, will re-replicate to ensure correctness")
                    # Delete existing table to force fresh replication
                    try:
                        pxt.drop_table(local_table_name)
                        logger.info(f"   Deleted existing replica: {local_table_name} (no metadata)")
                    except Exception as e:
                        logger.warning(f"   Could not delete existing replica: {e}")
                
                logger.info("Proceeding with replication...")
            except Exception as e:
                logger.info(f"Local table doesn't exist yet: {e}")
                logger.info("Proceeding with replication...")
    except Exception as e:
        logger.warning(f"Could not check for existing table: {e}")
    
    # Create the local replica (only if it doesn't exist)
    if setup_pixeltable():
        df = download_dataset(dataset_url, cache_dir)
        if df is not None:
            logger.info("✅ Successfully created local Pixeltable replica")
            return df
    
    # Replication failed: default to failing fast so we don't silently run with fake data.
    # If you *really* want sample data for offline UI work, set ALLOW_SAMPLE_DATA=1.
    if os.environ.get('ALLOW_SAMPLE_DATA') == '1':
        logger.warning("Replication failed, ALLOW_SAMPLE_DATA=1 set -> falling back to sample data")
        df = create_sample_data()
        cache_file = os.path.join(cache_dir, 'dataset_cache.pkl')
        df.to_pickle(cache_file)
        logger.info(f"Sample data cached to: {cache_file}")
        return df

    logger.error("Replication failed and ALLOW_SAMPLE_DATA is not set; exiting with failure")
    return None

if __name__ == "__main__":
    import sys
    df = main()
    if df is not None:
        print(f"✅ Data loaded successfully: {len(df)} records")
        print(f"Columns: {list(df.columns)}")
        print(f"Sample data:\n{df.head()}")
        sys.exit(0)  # Success
    else:
        print("❌ Failed to load data")
        sys.exit(1)  # Fail the build/script