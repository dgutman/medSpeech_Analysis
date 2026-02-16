from datasets.search import BatchedSearchResults
import numpy as np
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pixeltable as pxt
import os
from pixeltable.functions import whisper


# Load dataset
dataset = load_dataset("Hani89/medical_asr_recording_dataset")

# Create directory
pxt.create_dir('medSpeechAnalysis_hf', if_exists='ignore')

# Get or create table
try:
    t = pxt.get_table('medSpeechAnalysis_hf.hani89_asr_dataset')
    print("✅ Table already exists, checking for missing columns...")
except: 
    print("📦 Creating new table from HuggingFace dataset...")
    t = pxt.io.import_huggingface_dataset(
        'medSpeechAnalysis_hf.hani89_asr_dataset',
        dataset,
        column_name_for_split='split',
    )


# UDFs for processing
@pxt.udf
def name_files_from_json(audio: pxt.Json, split: pxt.String) -> pxt.String:
    """Extract filename from Json audio object"""
    filename = audio["path"][13:]
    return split + '_' + filename

@pxt.udf
def name_files_from_audio(audio: pxt.Audio, split: pxt.String) -> pxt.String:
    """Extract filename from Audio path"""
    # Audio path format: medSpeechAnalysis_hf_ray/train_xxx.wav or similar
    filename = os.path.basename(str(audio))
    return filename

@pxt.udf
def to_auds(audio: pxt.Json, output_dir: pxt.String) -> pxt.Audio:
    """Convert Json audio to Audio file"""
    os.makedirs(output_dir, exist_ok=True)
    x = audio["array"]
    sr = int(audio["sampling_rate"])
    filename = audio["path"][13:]
    if x.shape[0] == 2 and x.ndim == 2:
        x = x.T
    x = np.clip(x, -1.0, 1.0)
    int16_audio = (x * 32767).astype(np.int16)
    file_path = os.path.join(output_dir, filename)
    wav_write(file_path, sr, int16_audio)
    return file_path

@pxt.udf
def name_paths_from_json(audio: pxt.Json, output_dir: pxt.String) -> pxt.String:
    """Get file path from Json audio object"""
    return output_dir + '/' + audio["path"][13:]

@pxt.udf
def name_paths_from_audio(audio: pxt.Audio, output_dir: pxt.String) -> pxt.String:
    """Get file path from Audio object"""
    return str(audio)


# Check current state of table and process idempotently
# In pixeltable, use hasattr() to check if a column exists
has_id = hasattr(t, 'id')
has_transcription = hasattr(t, 'transcription')
has_filePath = hasattr(t, 'filePath')
has_json_junk = hasattr(t, 'json_junk')
has_audio = hasattr(t, 'audio')
has_sentence = hasattr(t, 'sentence')

# Determine audio column type:
# - If json_junk exists, audio has been converted to Audio type
# - If json_junk doesn't exist, audio is still Json type (needs conversion)
audio_is_json = has_audio and not has_json_junk
audio_is_audio_type = has_audio and has_json_junk

print(f"Table state: id={has_id}, transcription={has_transcription}, filePath={has_filePath}, "
      f"json_junk={has_json_junk}, audio_is_json={audio_is_json}, audio_is_audio_type={audio_is_audio_type}")

# Step 1: Add id column (if missing)
if not has_id:
    print("Adding 'id' column...")
    if audio_is_json:
        t.add_computed_column(id=name_files_from_json(t.audio, t.split), if_exists='ignore')
    elif audio_is_audio_type:
        t.add_computed_column(id=name_files_from_audio(t.audio, t.split), if_exists='ignore')
    else:
        # Try Json first, fallback to Audio
        try:
            t.add_computed_column(id=name_files_from_json(t.audio, t.split), if_exists='ignore')
        except:
            t.add_computed_column(id=name_files_from_audio(t.audio, t.split), if_exists='ignore')
else:
    print("✅ 'id' column already exists")

# Step 2: Rename sentence to transcription (if needed)
if not has_transcription and has_sentence:
    print("Renaming 'sentence' to 'transcription'...")
    try:
        t.rename_column('sentence', 'transcription')
        has_transcription = True  # Update state after rename
    except Exception as e:
        print(f"⚠️  Could not rename 'sentence' to 'transcription': {e}")
elif has_transcription:
    print("✅ 'transcription' column already exists")
else:
    print("⚠️  Neither 'sentence' nor 'transcription' column found")

# Step 3: Add filePath column (if missing)
if not has_filePath:
    print("Adding 'filePath' column...")
    if audio_is_json:
        t.add_computed_column(filePath=name_paths_from_json(t.audio, t.split), if_exists='ignore')
    elif audio_is_audio_type:
        t.add_computed_column(filePath=name_paths_from_audio(t.audio, t.split), if_exists='ignore')
    else:
        # Try Json first, fallback to Audio
        try:
            t.add_computed_column(filePath=name_paths_from_json(t.audio, t.split), if_exists='ignore')
        except:
            t.add_computed_column(filePath=name_paths_from_audio(t.audio, t.split), if_exists='ignore')
else:
    print("✅ 'filePath' column already exists")

# Step 4: Convert audio from Json to Audio (if needed)
if audio_is_json:
    print("Converting 'audio' from Json to Audio...")
    try:
        # Create temporary column for Audio conversion
        t.add_computed_column(soundsss=to_auds(t.audio, output_dir='medSpeechAnalysis_hf'), if_exists='ignore')
        # Rename original audio to json_junk
        if not has_json_junk:
            t.rename_column('audio', 'json_junk')
        # Rename soundsss to audio
        if hasattr(t, 'soundsss'):
            t.rename_column('soundsss', 'audio')
        print("✅ Audio conversion complete")
    except Exception as e:
        print(f"⚠️  Error during audio conversion: {e}")
        print("   You may need to manually fix the table state")
elif audio_is_audio_type:
    print("✅ 'audio' column already converted to Audio type")
else:
    if not has_audio:
        print("⚠️  No 'audio' column found - table may be in unexpected state")
    else:
        print("⚠️  Audio column state unclear - skipping conversion")

print("\n✅ Table processing complete! All columns are ready.")



## Whisper Transcription using Pixeltable's built-in whisper.transcribe function
## These use local Whisper models (no FastRay/API calls needed)
print(t.columns())
# Whisper Tiny English Model
t.add_computed_column(whisper_tinyEn_transcription=whisper.transcribe(audio=t.audio, model='tiny.en' ), if_exists='ignore')

# Whisper Base English Model
t.add_computed_column(whisper_baseEn_transcription=whisper.transcribe(audio=t.audio, model='base.en'), if_exists='ignore')

# Whisper Small English Model
t.add_computed_column(whisper_smallEn_transcription=whisper.transcribe(audio=t.audio, model='small.en'), if_exists='ignore')

# Whisper Medium English Model
t.add_computed_column(whisper_mediumEn_transcription=whisper.transcribe(audio=t.audio, model='medium.en'), if_exists='ignore')

# Whisper Large Model (multilingual)
t.add_computed_column(whisper_large_transcription=whisper.transcribe(audio=t.audio, model='large'), if_exists='ignore')

# Whisper Turbo Model
#t.add_computed_column(whisper_turbo_transcription=whisper.transcribe(audio=t.audio, model='turbo'), if_exists='ignore')
