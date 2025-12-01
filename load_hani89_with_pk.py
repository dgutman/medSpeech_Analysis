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

pxt.create_dir('medSpeechAnalysis', if_exists='ignore')



# Get or create table
try:
    t = pxt.get_table('medSpeechAnalysis.hani89_asr_dataset_raw')
    print("✅ Table already exists, checking for missing columns...")
except Exception as e: 
    print("📦 Creating new table from HuggingFace dataset...")
    dataset = load_dataset("Hani89/medical_asr_recording_dataset")
    t = pxt.io.import_huggingface_dataset(
        'medSpeechAnalysis.hani89_asr_dataset_raw',
        dataset,
        column_name_for_split='split',
    )

import pixeltable.functions.audio as a

t.add_computed_column( audio_file = a.encode_audio ( t.audio.array.astype(pxt.Array[pxt.Float]) ,input_sample_rate= t.audio.sampling_rate, format='wav' ), if_exists='ignore')


print(t.count(),"rows in table")
print(t.columns)
#print(t.head())
