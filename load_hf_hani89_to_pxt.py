import numpy as np
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pixeltable as pxt
import os
from pixeltable.functions import whisper
import httpx
import json
import asyncio

# Load dataset
dataset = load_dataset("Hani89/medical_asr_recording_dataset")

# Create directory
pxt.create_dir('medSpeechAnalysis_hf_ray', if_exists='ignore')

# Get or create table
try:
    t = pxt.get_table('medSpeechAnalysis_hf_ray.hani89_asr_dataset')
    print("✅ Table already exists, checking for missing columns...")
except: 
    print("📦 Creating new table from HuggingFace dataset...")
    t = pxt.io.import_huggingface_dataset(
        'medSpeechAnalysis_hf_ray.hani89_asr_dataset',
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

@pxt.udf(resource_pool='request-rate:my-ray-cluster')
async def fastray_transcribe_tiny(audio: pxt.Audio, api_url: pxt.String = "http://atlas.neuro.emory.edu:8000") -> pxt.String:
    """
    Transcribe audio using FastRay Whisper service with tiny model.
    
    Args:
        audio: Audio file to transcribe
        api_url: Base URL for FastRay API (default: http://atlas.neuro.emory.edu:8000)
    
    Returns:
        Transcription text as string
    """
    try:
        # Read the audio file
        audio_path = str(audio)
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found: {audio_path}"
        
        # Read file content
        with open(audio_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare multipart form data
        files = {'file': (os.path.basename(audio_path), file_content, 'audio/wav')}
        data = {
            'task': 'transcribe',
            'beam_size': '5',
            'language': 'en',
            'model': 'tiny'
        }
        
        # Make async request to FastRay API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{api_url}/transcribe/file",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
            # Return just the transcription text
            return result.get('text', '')
            
    except httpx.HTTPError as e:
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

@pxt.udf
async def fastray_transcribe_tiny_full(audio: pxt.Audio, api_url: pxt.String = "http://atlas.neuro.emory.edu:8000") -> pxt.Json:
    """
    Transcribe audio using FastRay Whisper service with tiny model, returning full response.
    
    Args:
        audio: Audio file to transcribe
        api_url: Base URL for FastRay API (default: http://atlas.neuro.emory.edu:8000)
    
    Returns:
        Full JSON response with text, model, language, duration, inference_time_seconds
    """
    try:
        # Read the audio file
        audio_path = str(audio)
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Read file content
        with open(audio_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare multipart form data
        files = {'file': (os.path.basename(audio_path), file_content, 'audio/wav')}
        data = {
            'task': 'transcribe',
            'beam_size': '5',
            'language': 'en',
            'model': 'tiny'
        }
        
        # Make async request to FastRay API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{api_url}/transcribe/file",
                files=files,
                data=data
            )
            response.raise_for_status()
            result = response.json()
            
            # Return full result as JSON
            return result
            
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

@pxt.udf(resource_pool='request-rate:my-ray-cluster')
async def fastray_transcribe_tiny_by_path(audio: pxt.Audio, api_url: pxt.String = "http://atlas.neuro.emory.edu:8000", container_path: pxt.String = "/data_medspeech") -> pxt.String:
    """
    Transcribe audio using FastRay Whisper service with tiny model via file path.
    This is more efficient than uploading files, but requires the audio files to be
    accessible from the FastRay container (e.g., mounted volume).
    
    Args:
        audio: Audio file to transcribe
        api_url: Base URL for FastRay API (default: http://atlas.neuro.emory.edu:8000)
        container_path: Path prefix in container where files are mounted (default: /data_medspeech)
    
    Returns:
        Transcription text as string
    """
    try:
        # Get the audio file path
        audio_path = str(audio)
        if not os.path.exists(audio_path):
            return f"Error: Audio file not found: {audio_path}"
        
        # Map host path to container path
        # Pixeltable returns audio paths as absolute strings:
        # /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray/{filename}.wav
        # The medSpeechAnalysis_hf_ray directory is mounted at /data_medspeech in FastRay container
        # Files are stored directly in medSpeechAnalysis_hf_ray/ (no subdirectories)
        # So we just need the filename: /data_medspeech/{filename}.wav
        filename = os.path.basename(audio_path)
        container_file_path = f"{container_path}/{filename}"
        
        # Make request to FastRay API using path endpoint
        request_data = {
            "file_path": container_file_path,
            "task": "transcribe",
            "beam_size": 5,
            "language": "en",
            "model": "tiny"
        }
        
        # Make async request to FastRay API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{api_url}/transcribe/path",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            # Return just the transcription text
            return result.get('text', '')
        
    except httpx.HTTPError as e:
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

@pxt.udf
async def fastray_transcribe_tiny_by_path_full(audio: pxt.Audio, api_url: pxt.String = "http://atlas.neuro.emory.edu:8000", container_path: pxt.String = "/data_medspeech") -> pxt.Json:
    """
    Transcribe audio using FastRay Whisper service with tiny model via file path, returning full response.
    This is more efficient than uploading files, but requires the audio files to be
    accessible from the FastRay container (e.g., mounted volume).
    
    Args:
        audio: Audio file to transcribe
        api_url: Base URL for FastRay API (default: http://atlas.neuro.emory.edu:8000)
        container_path: Path prefix in container where files are mounted (default: /data_medspeech)
    
    Returns:
        Full JSON response with text, model, language, duration, inference_time_seconds
    """
    try:
        # Get the audio file path
        audio_path = str(audio)
        if not os.path.exists(audio_path):
            return {"error": f"Audio file not found: {audio_path}"}
        
        # Map host path to container path
        # Pixeltable returns audio paths as absolute strings:
        # /scr/dagutman/devel/medSpeech_Analysis/medSpeechAnalysis_hf_ray/{filename}.wav
        # The medSpeechAnalysis_hf_ray directory is mounted at /data_medspeech in FastRay container
        # Files are stored directly in medSpeechAnalysis_hf_ray/ (no subdirectories)
        # So we just need the filename: /data_medspeech/{filename}.wav
        filename = os.path.basename(audio_path)
        container_file_path = f"{container_path}/{filename}"
        
        # Make request to FastRay API using path endpoint
        request_data = {
            "file_path": container_file_path,
            "task": "transcribe",
            "beam_size": 5,
            "language": "en",
            "model": "tiny"
        }
        
        # Make async request to FastRay API
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{api_url}/transcribe/path",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            
            # Return full result as JSON (includes inference_time_seconds, duration, model, language, text)
            return result
        
    except httpx.HTTPError as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

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
        t.add_computed_column(soundsss=to_auds(t.audio, output_dir='medSpeechAnalysis_hf_ray'), if_exists='ignore')
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



## #Whisper Tiny English Model (using FastRay service - file upload)
#t.add_computed_column(whisper_tinyEn_transcription_fastray=fastray_transcribe_tiny(t.audio), if_exists='ignore')

# #Whisper Tiny English Model (using FastRay service - by path, more efficient if files are mounted)
# Returns just text:
t.add_computed_column(whisper_tinyEn_transcription_fastray_path=fastray_transcribe_tiny_by_path(t.audio), if_exists='ignore')

# #Whisper Tiny English Model (using FastRay service - by path, returns full JSON with timing info)
# Returns full response with text, inference_time_seconds, duration, model, language:
#t.add_computed_column(whisper_tinyEn_transcription_fastray_path_full=fastray_transcribe_tiny_by_path_full(t.audio), if_exists='ignore')

# Note: Individual async calls will be parallelized by Pixeltable and FastAPI/Ray automatically

## #Whisper Tiny English Model (local whisper)
#t.add_computed_column( whisper_tinyEn_transcription  = whisper.transcribe(audio= t.audio, model='tiny.en'), if_exists='ignore')  

# # #Whisper English Small Model
# table2.add_computed_column( whisper_smallEn_transcription = whisper.transcribe( audio = table2.audio, model = "small.en"), if_exists='ignore')

# # #Whisper English medium Model
# table2.add_computed_column( whisper_mediumEn_transcription = whisper.transcribe( audio = table2.audio, model = "medium.en"), if_exists='ignore')

# # #Whisper Large Model
# table2.add_computed_column( whisper_large_transcription = whisper.transcribe( audio = table2.audio, model = "large"), if_exists='ignore')

# # # #Whisper Base English model
# table2.add_computed_column( whisper_baseEn_transcription = whisper.transcribe( audio = table2.audio, model = "base.en"), if_exists='ignore')

# # # #Turbo
# table2.add_computed_column( whisper_turbo_transcription = whisper.transcribe( audio = table2.audio, model = "turbo"), if_exists='ignore')