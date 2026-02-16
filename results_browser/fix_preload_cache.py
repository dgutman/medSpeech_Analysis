#!/usr/bin/env python3
"""Script to fix preload_data.py cache directory handling"""

# Read the file
with open('preload_data.py', 'r') as f:
    content = f.read()

# Fix 1: Make cache directory creation handle errors
old_cache_create = """    # Create directories
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)"""

new_cache_create = """    # Create directories (handle permission errors gracefully)
    # During Docker build, cache_dir might not be writable, but that's OK
    # The actual table data goes into pixeltable's database at PIXELTABLE_PGDATA, not the cache
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not create cache directory {cache_dir}: {e}")
        logger.warning("Using /tmp for cache (this is OK - actual data goes into pixeltable database)")
        cache_dir = '/tmp'
        os.makedirs(cache_dir, exist_ok=True)
    
    try:
        os.makedirs(data_dir, exist_ok=True)
    except (PermissionError, OSError) as e:
        logger.warning(f"Could not create data directory {data_dir}: {e}")
        data_dir = '/tmp'
        os.makedirs(data_dir, exist_ok=True)"""

if old_cache_create in content:
    content = content.replace(old_cache_create, new_cache_create)
    print("Fixed cache directory creation")
else:
    print("Cache directory creation already fixed or pattern not found")

# Fix 2: Make metadata saving handle errors and capture actual table name
old_metadata = """        logger.info("Pulling latest data from remote table...")
        local_table.pull()
        
        # Verify the local table works by getting a sample
        logger.info("Verifying local table...")
        sample_df = local_table.select().limit(5).collect().to_pandas()
        logger.info(f"Local table created successfully with {len(sample_df)} sample records")
        logger.info(f"Columns: {list(sample_df.columns)}")
        
        # Save metadata about the local table
        metadata_file = os.path.join(cache_dir, 'local_table_metadata.json')
        metadata = {
            'dataset_url': dataset_url,
            'local_table_name': local_table_name,
            'replication_timestamp': datetime.now().isoformat(),
            'pixeltable_version': pxt.__version__,
            'status': 'replicated_successfully'
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Local table metadata saved to: {metadata_file}")"""

new_metadata = """        # Get the actual table name/path that was created (may differ from local_table_name)
        actual_table_name = str(local_table.path) if hasattr(local_table, 'path') else local_table_name
        logger.info(f"Replicated table - requested: {local_table_name}, actual: {actual_table_name}")
        
        logger.info("Pulling latest data from remote table...")
        local_table.pull()
        
        # Verify the local table works by getting a sample and row count
        logger.info("Verifying local table...")
        sample_df = local_table.select().limit(5).collect().to_pandas()
        total_count = local_table.select().count()
        logger.info(f"Local table created successfully:")
        logger.info(f"   Table name/path: {actual_table_name}")
        logger.info(f"   Total rows: {total_count:,}")
        logger.info(f"   Sample records: {len(sample_df)}")
        logger.info(f"   Columns: {list(sample_df.columns)}")
        
        # Save metadata about the local table (if cache_dir is writable)
        # During Docker build, this might fail, but that's OK - the table is replicated
        try:
            metadata_file = os.path.join(cache_dir, 'local_table_metadata.json')
            metadata = {
                'dataset_url': dataset_url,
                'local_table_name': local_table_name,
                'actual_table_name': actual_table_name,
                'replication_timestamp': datetime.now().isoformat(),
                'pixeltable_version': pxt.__version__,
                'status': 'replicated_successfully',
                'row_count': total_count
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Local table metadata saved to: {metadata_file}")
        except (PermissionError, OSError) as e:
            logger.warning(f"Could not save metadata file (this is OK during Docker build): {e}")"""

if old_metadata in content:
    content = content.replace(old_metadata, new_metadata)
    print("Fixed metadata saving")
else:
    print("Metadata saving already fixed or pattern not found")

# Write back
with open('preload_data.py', 'w') as f:
    f.write(content)

print("Done!")
