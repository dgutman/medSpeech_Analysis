from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import os

# Try to load from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required

# Set PIXELTABLE_PGDATA to absolute path before importing pixeltable
# This prevents PostgreSQL from using wrong username in paths
# Load from .env first, then fall back to default
if 'PIXELTABLE_PGDATA' not in os.environ:
    os.environ['PIXELTABLE_PGDATA'] = '/scr/dagutman/devel/medSpeech_Analysis/.pxtData'


import pixeltable as pxt

import time


pxtDir = "medSpeechAnalysis"
rawTable = "hani89_asr_dataset_raw_v1"

pxt.create_dir(pxtDir, if_exists='ignore')

# Get or create table
try:
    t = pxt.get_table(f'{pxtDir}.{rawTable}')
    print("✅ Table already exists, checking for missing columns...")
except Exception as e: 
    print("📦 Creating new table from HuggingFace dataset...")
    dataset = load_dataset("Hani89/medical_asr_recording_dataset")

    startTime = time.time()

    t = pxt.io.import_huggingface_dataset(
        f'{pxtDir}.{rawTable}',
        dataset,
        column_name_for_split='split',
    )
    endTime = time.time()
    print(f"Time taken to load {len(dataset)} samples: {endTime - startTime}")

import pixeltable.functions.audio as a

t.add_computed_column( audio_file = a.encode_audio ( t.audio.array.astype(pxt.Array[pxt.Float]) ,input_sample_rate= t.audio.sampling_rate, format='wav' ), if_exists='ignore')


### I want to create a primary key which is the filename of the audio file.
# UDFs for processing
@pxt.udf
def name_files_from_json(audio: pxt.Json, split: pxt.String) -> pxt.Required[pxt.String]:
    """Extract filename from Json audio object"""
    filename = os.path.basename(audio["path"])
    return split + '_' + filename


t.add_computed_column( filename_key = name_files_from_json(t.audio, t.split), if_exists='replace')

print(t.count(),"rows in table")
print(t.columns)
print(t.head(5))

### This creates the primary table, which I will then insert into a new table so I can set the primary key..
t_with_pk = pxt.create_table(
    f'{pxtDir}.hani89_asr_dataset',
    source=t,
    
    primary_key='filename_key' , if_exists='ignore')