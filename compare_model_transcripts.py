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

# Create multiple visualizations for different scales
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Create a figure with multiple subplots
fig = plt.figure(figsize=(20, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Boxplot (good for current scale, ~200 points)
ax1 = fig.add_subplot(gs[0, 0])
sns.set_style("whitegrid")
sns.boxplot(data=df_long, x='Model', y='WER', hue='Model', ax=ax1)
sns.stripplot(data=df_long, x='Model', y='WER', color='black', alpha=0.3, size=2, ax=ax1)
ax1.set_title("WER Comparison - Boxplot", fontsize=14, pad=15)
ax1.set_xlabel("Whisper Model", fontsize=12)
ax1.set_ylabel("Word Error Rate (%)", fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylim(0, min(120, df_long['WER'].max() + 10))
ax1.legend().remove()

# 2. Violin plot (better for distribution shape)
ax2 = fig.add_subplot(gs[0, 1])
sns.violinplot(data=df_long, x='Model', y='WER', hue='Model', ax=ax2)
ax2.set_title("WER Distribution - Violin Plot", fontsize=14, pad=15)
ax2.set_xlabel("Whisper Model", fontsize=12)
ax2.set_ylabel("Word Error Rate (%)", fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(0, min(120, df_long['WER'].max() + 10))
ax2.legend().remove()

# 3. Kernel Density Plot (much better for overlapping distributions)
ax3 = fig.add_subplot(gs[0, 2])
# Define colors to match seaborn's default palette
colors = sns.color_palette("husl", len(df_long['Model'].unique()))
for i, model in enumerate(df_long['Model'].unique()):
    model_data = df_long[df_long['Model'] == model]['WER']
    model_data = model_data.dropna()  # Remove NaN values
    if len(model_data) > 0:  # Only plot if we have data
        ax3.hist(model_data, alpha=0.3, label=model, bins=15, density=True, 
                histtype='step', linewidth=2, edgecolor=colors[i], color=colors[i])
ax3.set_title("WER Distribution - Density Plot", fontsize=14, pad=15)
ax3.set_xlabel("Word Error Rate (%)", fontsize=12)
ax3.set_ylabel("Density", fontsize=12)
ax3.legend()

# 4. Summary statistics table
ax4 = fig.add_subplot(gs[1, :])
ax4.axis('off')

# Calculate summary statistics
summary_stats = df_long.groupby('Model')['WER'].agg(['count', 'mean', 'std', 'min', 'max', 'median']).round(2)
summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']

# Create table
table_data = []
for model in summary_stats.index:
    row = [model] + [f"{summary_stats.loc[model, col]:.2f}" for col in summary_stats.columns]
    table_data.append(row)

table = ax4.table(cellText=table_data,
                 colLabels=['Model'] + list(summary_stats.columns),
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
ax4.set_title("Summary Statistics", fontsize=16, pad=20)

plt.suptitle("Comprehensive WER Analysis Across Whisper Models", fontsize=20, y=0.98)
plt.tight_layout()

# Save the comprehensive plot
plt.savefig('analysisResults/hani89_asr_data_wer_comprehensive.png', dpi=300, bbox_inches='tight')
plt.close()

# Also create a simple boxplot for quick reference
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
sns.boxplot(data=df_long, x='Model', y='WER', hue='Model')
sns.stripplot(data=df_long, x='Model', y='WER', color='black', alpha=0.3, size=2)

plt.title("WER Comparison Across Whisper Models", fontsize=20, pad=20)
plt.xlabel("Whisper Model", fontsize=14)
plt.ylabel("Word Error Rate (%)", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, min(120, df_long['WER'].max() + 10))
plt.legend().remove()

plt.tight_layout()
plt.savefig('analysisResults/hani89_asr_data_wer_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# Also save the processed data with WER scores
df.to_csv('analysisResults/hani89_asr_data_with_wer.csv', index=False)

print("Analysis complete!")
print("Files saved:")
print("- analysisResults/hani89_asr_data.csv (raw data)")
print("- analysisResults/hani89_asr_data_with_wer.csv (data with WER scores)")
print("- analysisResults/hani89_asr_data_wer_boxplot.png (simple boxplot)")
print("- analysisResults/hani89_asr_data_wer_comprehensive.png (multi-panel analysis)")


