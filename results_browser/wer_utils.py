"""
Word Error Rate (WER) computation utilities.

This module provides functions for calculating Word Error Rate between
reference and hypothesis transcriptions using various methods.
"""

import pandas as pd
import numpy as np
import jiwer
import logging

logger = logging.getLogger(__name__)


def calculate_wer(hypothesis, reference, method='basic'):
    """
    Calculate Word Error Rate between hypothesis and reference text
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        method: WER computation method ('basic' or 'ignore_punctuation')
    
    Returns:
        WER as a float between 0 and 1
    """
    # Check for invalid inputs
    if pd.isna(hypothesis) or pd.isna(reference):
        return np.nan
    
    # Convert to string and check for empty strings
    hypothesis_str = str(hypothesis).strip()
    reference_str = str(reference).strip()
    
    if hypothesis_str == '' or reference_str == '' or hypothesis_str == 'nan' or reference_str == 'nan':
        return np.nan
    
    try:
        if method == 'ignore_punctuation':
            # Remove punctuation before calculating WER
            import string
            hypothesis_str = hypothesis_str.translate(str.maketrans('', '', string.punctuation))
            reference_str = reference_str.translate(str.maketrans('', '', string.punctuation))
            # Remove extra whitespace
            hypothesis_str = ' '.join(hypothesis_str.split())
            reference_str = ' '.join(reference_str.split())
        
        # Calculate WER using jiwer
        return jiwer.wer(reference_str, hypothesis_str)
    except (ValueError, TypeError, AttributeError) as e:
        logger.warning(f"Error processing:\nReference: {reference_str}\nHypothesis: {hypothesis_str}\nError: {e}")
        return np.nan


def calculate_wer_detailed(hypothesis, reference, method='basic'):
    """
    Calculate detailed WER metrics including substitutions, deletions, insertions
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        method: WER computation method ('basic' or 'ignore_punctuation')
    
    Returns:
        Dictionary with 'wer', 'substitutions', 'deletions', 'insertions'
    """
    if pd.isna(hypothesis) or pd.isna(reference) or hypothesis == '' or reference == '':
        return {'wer': np.nan, 'substitutions': np.nan, 'deletions': np.nan, 'insertions': np.nan}
    
    try:
        # Use the same simple approach as the basic WER function
        return {
            'wer': calculate_wer(hypothesis, reference, method=method),
            'substitutions': np.nan,  # Not calculating detailed metrics for now
            'deletions': np.nan,
            'insertions': np.nan
        }
    except Exception as e:
        logger.warning(f"Error calculating detailed WER: {e}")
        return {'wer': np.nan, 'substitutions': np.nan, 'deletions': np.nan, 'insertions': np.nan}


def get_wer_methods():
    """
    Get available WER computation methods.
    
    Returns:
        Dictionary mapping method keys to human-readable labels
    """
    return {
        "basic": "Basic (Standard WER)",
        "ignore_punctuation": "Ignore Punctuation"
    }


def detect_repetition(hypothesis, min_repeat_length=3, min_repeats=2):
    """
    Detect repetition/hallucination in transcription by looking for repeated phrases.
    
    Args:
        hypothesis: Model output text
        min_repeat_length: Minimum number of words in a repeated phrase (default: 3)
        min_repeats: Minimum number of times phrase must repeat (default: 2)
    
    Returns:
        Dictionary with 'has_repetition' (bool) and 'repetition_info' (description)
    """
    if pd.isna(hypothesis) or hypothesis == '':
        return {'has_repetition': False, 'repetition_info': ''}
    
    try:
        hypothesis_str = str(hypothesis).strip()
        if hypothesis_str == '':
            return {'has_repetition': False, 'repetition_info': ''}
        
        words = hypothesis_str.split()
        if len(words) < min_repeat_length * min_repeats:
            return {'has_repetition': False, 'repetition_info': ''}
        
        # Check for repeated phrases of length min_repeat_length
        for phrase_len in range(min_repeat_length, len(words) // min_repeats + 1):
            for i in range(len(words) - phrase_len * min_repeats + 1):
                phrase = words[i:i+phrase_len]
                phrase_str = ' '.join(phrase)
                
                # Check if this phrase repeats
                count = 0
                j = i
                while j <= len(words) - phrase_len:
                    if words[j:j+phrase_len] == phrase:
                        count += 1
                        j += phrase_len
                    else:
                        break
                
                if count >= min_repeats:
                    return {
                        'has_repetition': True,
                        'repetition_info': f"Repeated phrase '{phrase_str}' {count} times"
                    }
        
        return {'has_repetition': False, 'repetition_info': ''}
    except Exception as e:
        logger.warning(f"Error detecting repetition: {e}")
        return {'has_repetition': False, 'repetition_info': ''}


def detect_length_anomaly(hypothesis, reference, min_ratio=0.5, max_ratio=2.0):
    """
    Detect length-based hallucinations where output is significantly longer or shorter than reference.
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        min_ratio: Minimum word count ratio (hypothesis/reference) before flagging as too short (default: 0.5)
        max_ratio: Maximum word count ratio (hypothesis/reference) before flagging as too long (default: 2.0)
    
    Returns:
        Dictionary with 'has_anomaly' (bool), 'anomaly_type' ('too_short', 'too_long', or None), 
        and 'length_info' (description)
    """
    if pd.isna(hypothesis) or pd.isna(reference) or hypothesis == '' or reference == '':
        return {'has_anomaly': False, 'anomaly_type': None, 'length_info': ''}
    
    try:
        hypothesis_str = str(hypothesis).strip()
        reference_str = str(reference).strip()
        
        if hypothesis_str == '' or reference_str == '':
            return {'has_anomaly': False, 'anomaly_type': None, 'length_info': ''}
        
        hyp_words = len(hypothesis_str.split())
        ref_words = len(reference_str.split())
        
        if ref_words == 0:
            return {'has_anomaly': False, 'anomaly_type': None, 'length_info': ''}
        
        ratio = hyp_words / ref_words
        
        if ratio < min_ratio:
            return {
                'has_anomaly': True,
                'anomaly_type': 'too_short',
                'length_info': f"Too short: {hyp_words} words vs {ref_words} words (ratio: {ratio:.2f})"
            }
        elif ratio > max_ratio:
            return {
                'has_anomaly': True,
                'anomaly_type': 'too_long',
                'length_info': f"Too long: {hyp_words} words vs {ref_words} words (ratio: {ratio:.2f})"
            }
        
        return {'has_anomaly': False, 'anomaly_type': None, 'length_info': ''}
    except Exception as e:
        logger.warning(f"Error detecting length anomaly: {e}")
        return {'has_anomaly': False, 'anomaly_type': None, 'length_info': ''}


def detect_hallucinations(hypothesis, reference, check_repetition=True, check_length=True, 
                          min_repeat_length=3, min_repeats=2, min_ratio=0.5, max_ratio=2.0):
    """
    Comprehensive hallucination detection combining multiple methods.
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        check_repetition: Whether to check for repetitions (default: True)
        check_length: Whether to check for length anomalies (default: True)
        min_repeat_length: Minimum phrase length for repetition detection (default: 3)
        min_repeats: Minimum number of repeats (default: 2)
        min_ratio: Minimum length ratio (default: 0.5)
        max_ratio: Maximum length ratio (default: 2.0)
    
    Returns:
        Dictionary with flags and details for all detected hallucinations
    """
    result = {
        'has_hallucination': False,
        'repetition': {'detected': False, 'info': ''},
        'length_anomaly': {'detected': False, 'type': None, 'info': ''}
    }
    
    if check_repetition:
        rep_result = detect_repetition(hypothesis, min_repeat_length, min_repeats)
        result['repetition'] = {
            'detected': rep_result['has_repetition'],
            'info': rep_result['repetition_info']
        }
        if rep_result['has_repetition']:
            result['has_hallucination'] = True
    
    if check_length:
        len_result = detect_length_anomaly(hypothesis, reference, min_ratio, max_ratio)
        result['length_anomaly'] = {
            'detected': len_result['has_anomaly'],
            'type': len_result['anomaly_type'],
            'info': len_result['length_info']
        }
        if len_result['has_anomaly']:
            result['has_hallucination'] = True
    
    return result

