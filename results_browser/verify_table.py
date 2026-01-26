#!/usr/bin/env python3
"""Script to verify the imported table and show row count"""
import os
import pixeltable as pxt

# Set PIXELTABLE_PGDATA
os.environ['PIXELTABLE_PGDATA'] = '/home/appuser/.pixeltable/pgdata'

print("Checking imported tables...\n")

# Try to list all tables
try:
    tables = pxt.list_tables()
    print(f"Found {len(tables)} table(s): {tables}\n")
except Exception as e:
    print(f"Could not list tables: {e}\n")

# Try each possible table name
table_names = ['local_hani89', 'local_transcribe_compare', 'local_hani89_asr_data_reload']

for table_name in table_names:
    try:
        table = pxt.get_table(table_name)
        count = table.select().count()
        
        print(f"✅ Table: {table_name}")
        print(f"   Row count: {count:,}")
        
        # Get a sample row to show structure
        sample = table.select().limit(1).collect()
        if len(sample) > 0:
            row = sample[0]
            print(f"   Columns: {list(row.keys())[:10]}...")  # Show first 10 columns
            if 'id' in row:
                print(f"   Sample ID: {row['id']}")
            if 'split' in row:
                print(f"   Sample split: {row['split']}")
        print()
        
    except Exception as e:
        print(f"❌ Table '{table_name}': {e}\n")
