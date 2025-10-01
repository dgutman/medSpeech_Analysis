import os
from dotenv import load_dotenv
import pandas as pd
from pixeltable.functions.audio import get_metadata
from pixeltable.functions import whisper


## I set the PIXELTABLE_PGDATA environment variable in the .env file so that pixeltable stores the data on my NVME Drive
## You need to import this before starting pixeltable.
load_dotenv()
import pixeltable as pxt

### Load the trainind and testing data
train_df = pd.read_csv('train_data/train_metadata.csv')
test_df = pd.read_csv('test_data/test_metadata.csv')
print("There are ", len(train_df), " training samples and ", len(test_df), " test samples")



## Will load the two data sets independently, but want to track the training and test splits
## Also add the file path to the data frame for the directory where the audio files are stored
train_df['split'] = 'train'
test_df['split'] = 'test'
train_df['filePath'] = "train_data/" + train_df['filename']
test_df['filePath'] = "test_data/" + test_df['filename']


pxt.create_dir('hani89_asr_data',if_exists='ignore')

## Create a table to store audio files
pxt.create_table('hani89_asr_data.transcribe_compare',
                  {'id': pxt.String,
                   'audio': pxt.Audio,
                   'filename': pxt.String,
                   'transcription': pxt.String,
                   'split': pxt.String,
                   'filePath': pxt.String}
                  ,if_exists='ignore'
                  )

## Now get a pointer to the table
t = pxt.get_table('hani89_asr_data.transcribe_compare')

## Load the training data
## DEBUG: Only load the first 100 samples for testing



## Using filename as the unique id and only inserting the first 100 samples
#t.insert({'id': row['filename'], 'audio': row['filePath'], 'filename': row['filename'], 'transcription': row['transcription'], 'split': row['split'], 'filePath': row['filePath']} for row in train_df.to_dict(orient='records')[:100])

## TO DO IS HAVE IT DETERMINE WHAT SAMPLES ARE ALREADY LOADED AND ONLY LOAD THE ONES THAT ARE NOT


### CREATE COMPUTED COLUMNS FOR TRANSCRIPTIONS OF AUDIO FILES

#Whisper Tiny English Model
t.add_computed_column( whisper_tinyEn_transcription  = whisper.transcribe(audio=t.audio, model='tiny.en'), if_exists='ignore')  ### HOW COULD I OVERRIDE COLUMN IF I CHANGE MY FUNCTION

#Whisper English Small Model
t.add_computed_column( whisper_smallEn_transcription = whisper.transcribe( audio = t.audio, model = "small.en"), if_exists='ignore')

#Whisper English medium Model
t.add_computed_column( whisper_mediumEn_transcription = whisper.transcribe( audio = t.audio, model = "medium.en"), if_exists='ignore')

#Whisper Large Model
t.add_computed_column( whisper_large_transcription = whisper.transcribe( audio = t.audio, model = "large"), if_exists='ignore')

# #Whisper Base English model
t.add_computed_column( whisper_baseEn_transcription = whisper.transcribe( audio = t.audio, model = "base.en"), if_exists='ignore')

# #Turbo
t.add_computed_column( whisper_turbo_transcription = whisper.transcribe( audio = t.audio, model = "turbo"), if_exists='ignore')
