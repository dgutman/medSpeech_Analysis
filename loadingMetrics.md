Record loading time, with embeddings

This is to process 100 records for the 6 whisper flavors I have currently added as computed columns

Inserting rows into `transcribe_compare`: 0 rows [00:00, ? rows/s]
Inserting rows into `transcribe_compare`: 100 rows [01:39,  1.01 rows/s]
Inserted 100 rows with 0 errors.
Time taken to load 100 samples: 0:05:34.111209




Inserting rows into `transcribe_compare`: 1000 rows [00:00, 3444.75 rows/s]
Inserted 1000 rows with 0 errors.
Time taken to load 1000 samples: 0:46:14.377067


There are  5328  training samples and  1333  test samples
Connected to Pixeltable database at: postgresql+psycopg://postgres:@/pixeltable?host=/scr/dagutman/devel/medSpeech_Analysis/.pxtDataV2
Inserting rows into `transcribe_compare`: 2000 rows [00:00, 3543.55 rows/s] 
Inserted 2000 rows with 0 errors.
Time taken to load 2000 samples: 1:31:54.445872