# compare_model_transcripts.py

# This script compares the transcripts of different models and saves the results to a CSV file.

# Import the necessary libraries
import pandas as pd
import jiwer
import os
from dotenv import load_dotenv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

import pixeltable as pxt

t =  pxt.get_table('hani89_asr_data.transcribe_compare')

# Get the data
df = t.select(t.whisper_tinyEn_transcription.text, t.whisper_baseEn_transcription.text, t.whisper_smallEn_transcription.text,
t.whisper_mediumEn_transcription.text, t.whisper_turbo_transcription.text, t.whisper_large_transcription.text, t.transcription, t.filename    ).show(500).to_pandas()

# make the filename the id column
df['id'] = df['filename']


# Save the data to a CSV file
df.to_csv('analysisResults/hani89_asr_data.csv', index=False)

## this function calculates the WER for a given hypothesis and reference
def calculate_wer(hypothesis, reference):
    # Check for invalid inputs
    if pd.isna(hypothesis) or pd.isna(reference):
        return np.nan
    
    # Convert to string and check for empty strings
    hypothesis_str = str(hypothesis).strip()
    reference_str = str(reference).strip()
    
    if hypothesis_str == '' or reference_str == '' or hypothesis_str == 'nan' or reference_str == 'nan':
        return np.nan
    
    try:
        # Simple WER calculation without complex transformations
        return jiwer.wer(reference_str, hypothesis_str)
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error processing:\nReference: {reference_str}\nHypothesis: {hypothesis_str}\nError: {e}")
        return np.nan

# Define the model output columns used for graphs/ analysis
model_cols = [
    'whispertinyEntranscription_text',
    'whispersmallEntranscription_text',
    'whispermediumEntranscription_text',
    'whisperlargetranscription_text',
    'whisperturbotranscription_text',
]

# Compute WERs
for col in model_cols:
    wer_col = col.replace('transcription_text', 'wer')
    print(f"Calculating WER for {col}...")
    df[wer_col] = df.apply(lambda row: calculate_wer(row[col], row['transcription']), axis=1)
    # Convert to percentage for better interpretability
    df[wer_col] = df[wer_col] * 100

# Optional: ensure the model columns are in a consistent order
model_order = [
    'whispertinyEnwer',
    'whispersmallEnwer',
    'whispermediumEnwer',
    'whisperlargewer',
    'whisperturbower',
 
]

# Melt the DataFrame
df_long = df.melt(id_vars='id', value_vars=model_order,
                  var_name='Model', value_name='WER')

# Clean labels
label_map = {
    'whispertinyEnwer': 'Whisper Tiny',
    'whispersmallEnwer': 'Whisper Small',
    'whispermediumEnwer': 'Whisper Medium',
    'whisperlargewer': 'Whisper Large',
    'whisperturbower': 'Whisper Turbo',
  
}
df_long['Model'] = df_long['Model'].map(label_map)

# Sort models in logical order
df_long['Model'] = pd.Categorical(df_long['Model'],
                                  categories=['Whisper Tiny', 'Whisper Small', 'Whisper Medium', 'Whisper Large', 'Whisper Turbo',],
                                  ordered=True)

# Create matplotlib boxplot
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.boxplot(data=df_long, x='Model', y='WER', hue='Model')
sns.stripplot(data=df_long, x='Model', y='WER', color='black', alpha=0.3, size=3)

plt.title("WER Comparison Across Whisper Models", fontsize=20, pad=20)
plt.xlabel("Whisper Model", fontsize=14)
plt.ylabel("Word Error Rate (%)", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, min(120, df_long['WER'].max() + 10))
plt.legend().remove()  # Remove legend since colors match x-axis labels

plt.tight_layout()

# Save as PNG
plt.savefig('analysisResults/hani89_asr_data_wer_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the figure to free memory

# Also save the processed data with WER scores
df.to_csv('analysisResults/hani89_asr_data_with_wer.csv', index=False)

print("Analysis complete!")
print("Files saved:")
print("- analysisResults/hani89_asr_data.csv (raw data)")
print("- analysisResults/hani89_asr_data_with_wer.csv (data with WER scores)")
print("- analysisResults/hani89_asr_data_wer_boxplot.png (static image)")


