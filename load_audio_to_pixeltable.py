import os
from dotenv import load_dotenv
import pandas as pd
from pixeltable.functions.audio import get_metadata
from pixeltable.functions import whisper
from datetime import datetime

## I set the PIXELTABLE_PGDATA environment variable in the .env file so that pixeltable stores the data on my NVME Drive
## You need to import this before starting pixeltable.
load_dotenv()
import pixeltable as pxt

### Load the training and testing data
train_df = pd.read_csv('train_data/train_metadata.csv')
test_df = pd.read_csv('test_data/test_metadata.csv')
print("There are ", len(train_df), " training samples and ", len(test_df), " test samples")

## Not using data set loader because of primary key issues

## Combine train and test dataframes into a single dataframe
## Add split and filePath columns to track which dataset each row came from
train_df['split'] = 'train'
test_df['split'] = 'test'
train_df['filePath'] = "train_data/" + train_df['filename']
test_df['filePath'] = "test_data/" + test_df['filename']

# Combine into single dataframe
df = pd.concat([train_df, test_df], ignore_index=True)
print(f"Combined dataframe has {len(df)} total samples ({len(train_df)} train, {len(test_df)} test)")


# Get table configuration from .env with defaults
pxt_dir = os.getenv('PIXELTABLE_DIR', 'hani89_asr_data_reload')
table_name = os.getenv('PIXELTABLE_TABLE', 'transcribe_compare')
full_table_name = f'{pxt_dir}.{table_name}'

pxt.create_dir(pxt_dir, if_exists='ignore')

## Create a table to store audio files
pxt.create_table(full_table_name,
                  {'id': pxt.Required[pxt.String],
                   'audio': pxt.Audio,
                   'filename': pxt.String,
                   'transcription': pxt.String,
                   'split': pxt.String,
                   'filePath': pxt.String}
                  ,if_exists='ignore',
                  primary_key='id'
                  )

## Now get a pointer to the table
t = pxt.get_table(full_table_name)

## Load the training data
## DEBUG: Only load the first 100 samples for testing

print(t.count(), "rows currently in table")

# ### TO DO: See if there is more elegant syntax for this
alreadyLoadedFiles = set(t.select(t.filePath).collect()['filePath'])

recordsToLoad = 3000

print(f"Already loaded {len(alreadyLoadedFiles)} files")
# startTime = datetime.now()

recs = []
for i, row in df.iterrows():
    # print(row['filename'])
    if row['filePath'] not in alreadyLoadedFiles:
        # print(row['filename'], "not in alreadyLoadedFiles")
        recs.append({'id': row['filename'], 'audio': row['filePath'], 'filename': row['filename'], 'transcription': row['transcription'], 'split': row['split'], 'filePath': row['filePath']})
        if len(recs) >= recordsToLoad:
            startTime = datetime.now()
            t.insert(recs)
            endTime = datetime.now()
            print(f"Time taken to load {recordsToLoad} samples: {endTime - startTime}")
            recs = []
            break

# Insert any remaining records
if recs:
    startTime = datetime.now()
    t.insert(recs)
    endTime = datetime.now()
    print(f"Time taken to load {len(recs)} remaining samples: {endTime - startTime}")

## Using filename as the unique id and only inserting the first 100 samples
#t.insert({'id': row['filename'], 'audio': row['filePath'], 'filename': row['filename'], 'transcription': row['transcription'], 'split': row['split'], 'filePath': row['filePath']} for row in train_df.to_dict(orient='records')[:100])

## TO DO IS HAVE IT DETERMINE WHAT SAMPLES ARE ALREADY LOADED AND ONLY LOAD THE ONES THAT ARE NOT


### CREATE COMPUTED COLUMNS FOR TRANSCRIPTIONS OF AUDIO FILES

# #Whisper Tiny English Model
# t.add_computed_column( whisper_tinyEn_transcription  = whisper.transcribe(audio=t.audio, model='tiny.en'), if_exists='ignore')  ### HOW COULD I OVERRIDE COLUMN IF I CHANGE MY FUNCTION

# #Whisper English Small Model
# t.add_computed_column( whisper_smallEn_transcription = whisper.transcribe( audio = t.audio, model = "small.en"), if_exists='ignore')

# #Whisper English medium Model
# t.add_computed_column( whisper_mediumEn_transcription = whisper.transcribe( audio = t.audio, model = "medium.en"), if_exists='ignore')

# #Whisper Large Model
# t.add_computed_column( whisper_large_transcription = whisper.transcribe( audio = t.audio, model = "large"), if_exists='ignore')

# # #Whisper Base English model
# t.add_computed_column( whisper_baseEn_transcription = whisper.transcribe( audio = t.audio, model = "base.en"), if_exists='ignore')

# # #Turbo
# t.add_computed_column( whisper_turbo_transcription = whisper.transcribe( audio = t.audio, model = "turbo"), if_exists='ignore')
