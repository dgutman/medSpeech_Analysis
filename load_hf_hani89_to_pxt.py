import numpy as np
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional
import pixeltable as pxt
import os
from pixeltable.functions import whisper


dataset = load_dataset("Hani89/medical_asr_recording_dataset")

pxt.create_dir('medSpeechAnalysis_hf',if_exists='ignore')


try:
    t = pxt.get_table('medSpeechAnalysis_hf.hani89_asr_dataset')
except: 
    ### Ignore logic doesn't work here since it's an import ... so just catch the error and create the table
    t=   pxt.io.import_huggingface_dataset(
    'medSpeechAnalysis_hf.hani89_asr_dataset',
    dataset,
    column_name_for_split='split',
    
    )



@pxt.udf
def name_files(audio: pxt.Json, split: pxt.String) -> pxt.String:
  filename = audio["path"][13:]
  return split+'_'+filename

@pxt.udf
def to_auds(audio: pxt.Json, output_dir: pxt.String) -> pxt.Audio:
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
def name_paths(audio: pxt.Json, output_dir: pxt.String) -> pxt.String:
  return output_dir+'/'+audio["path"][13:]
  




t.add_computed_column(id = name_files(t.audio, t.split), if_exists='ignore')
t.rename_column('sentence', 'transcription')
t.add_computed_column(filePath = name_paths(t.audio, t.split), if_exists='ignore')
t.add_computed_column(soundsss = to_auds(t.audio, output_dir = 'medSpeechAnalysis_hf'), if_exists='ignore')
t.rename_column('audio', 'json_junk')
t.rename_column('soundsss', 'audio')



# #Whisper Tiny English Model
t.add_computed_column( whisper_tinyEn_transcription  = whisper.transcribe(audio= t.audio, model='tiny.en'), if_exists='ignore')  

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