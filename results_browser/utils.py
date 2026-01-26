"""
Utility functions for the results browser.
"""

import pandas as pd


def extract_text_from_transcription(transcription_data):
    """Extract text from transcription JSON objects"""
    if pd.isna(transcription_data) or transcription_data is None:
        return ""
    
    # If it's already a string, return as is
    if isinstance(transcription_data, str):
        return transcription_data
    
    # If it's a dict, extract the 'text' property
    if isinstance(transcription_data, dict) and 'text' in transcription_data:
        return transcription_data['text']
    
    # If it's a list, extract text from each item
    if isinstance(transcription_data, list):
        texts = []
        for item in transcription_data:
            if isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
            elif isinstance(item, str):
                texts.append(item)
        return ' '.join(texts)
    
    return str(transcription_data)



