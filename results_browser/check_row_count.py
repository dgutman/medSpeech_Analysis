#!/usr/bin/env python3
"""Simple script to check row count of replicated table"""
import os
import pixeltable as pxt

os.environ['PIXELTABLE_PGDATA'] = '/home/appuser/.pixeltable/pgdata'

# Try all possible table names
for name in ['local_hani89', 'local_transcribe_compare', 'local_hani89_asr_data_reload']:
    try:
        table = pxt.get_table(name)
        count = table.select().count()
        print(f"✅ Table '{name}': {count:,} rows")
        
        # Show sample to verify
        sample = table.select().limit(1).collect()
        if sample:
            print(f"   Sample columns: {list(sample[0].keys())[:5]}")
        break
    except Exception as e:
        continue
else:
    print("❌ No table found")
