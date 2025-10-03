# medSpeech_Analysis
Will be using open source data, pixel table, and open source text to speech models to compare overall transcript accuracy across various open source speech to text models


I will be using pixeltable to load the data sets, and also to run the models across the data set.


## Data set
I am currently leveraging a data set I found on hugging faces called Hani89 that has short medically related text.   Notably, these are fairly short (one or two sentences), which has made it easier to develop the pipeline.



### Data Loading

The pixelTable loader script will build the basic table structure, and then import messages from the Hani89 dataset.  Currently, I could not import the data set directly, so I wrote a script that will pull the data set, and split it into local WAV files that I can then upload to my local pixeltable database.





## Step 1 
    python extract_audio_data.py

    This will download the hani89 dataset, and split it into WAV files.  I can then use this as input for pixeltable.
image.png

## TO DO:
    Allow direct ingestion of hugging face dataset into pixeltable


