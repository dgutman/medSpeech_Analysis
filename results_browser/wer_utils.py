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


def detect_character_repetition(hypothesis, min_char_repeat=5):
    """
    Detect character-level repetition (e.g., "the the the", character stuttering).
    
    Args:
        hypothesis: Model output text
        min_char_repeat: Minimum number of characters that repeat (default: 5)
    
    Returns:
        Dictionary with 'has_repetition' and 'info'
    """
    if pd.isna(hypothesis) or hypothesis == '':
        return {'has_repetition': False, 'info': ''}
    
    try:
        hypothesis_str = str(hypothesis).strip()
        if len(hypothesis_str) < min_char_repeat * 2:
            return {'has_repetition': False, 'info': ''}
        
        # Check for repeated character sequences
        for i in range(len(hypothesis_str) - min_char_repeat * 2):
            seq = hypothesis_str[i:i+min_char_repeat]
            # Check if this sequence repeats immediately
            if i + min_char_repeat * 2 <= len(hypothesis_str):
                next_seq = hypothesis_str[i+min_char_repeat:i+min_char_repeat*2]
                if seq == next_seq:
                    return {
                        'has_repetition': True,
                        'info': f"Character repetition: '{seq}' repeated"
                    }
        
        return {'has_repetition': False, 'info': ''}
    except Exception as e:
        logger.warning(f"Error detecting character repetition: {e}")
        return {'has_repetition': False, 'info': ''}


def detect_unlikely_insertions(hypothesis, reference, wer_threshold=0.8):
    """
    Detect when hypothesis has very high insertion rate (potential hallucination).
    Uses WER and word count comparison to identify excessive insertions.
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        wer_threshold: WER threshold above which to flag (default: 0.8)
    
    Returns:
        Dictionary with 'has_insertion' and 'info'
    """
    if pd.isna(hypothesis) or pd.isna(reference) or hypothesis == '' or reference == '':
        return {'has_insertion': False, 'info': ''}
    
    try:
        wer = calculate_wer(hypothesis, reference)
        if pd.isna(wer):
            return {'has_insertion': False, 'info': ''}
        
        hyp_words = len(str(hypothesis).split())
        ref_words = len(str(reference).split())
        
        if ref_words == 0:
            return {'has_insertion': False, 'info': ''}
        
        # High WER combined with significantly more words suggests insertions
        word_ratio = hyp_words / ref_words if ref_words > 0 else 0
        if wer > wer_threshold and word_ratio > 1.5:
            return {
                'has_insertion': True,
                'info': f"High WER ({wer:.2f}) with {word_ratio:.1f}x word count suggests insertions"
            }
        
        return {'has_insertion': False, 'info': ''}
    except Exception as e:
        logger.warning(f"Error detecting insertions: {e}")
        return {'has_insertion': False, 'info': ''}


def detect_word_stuttering(hypothesis, min_stutter=3):
    """
    Detect word-level stuttering (same word repeated many times).
    
    Args:
        hypothesis: Model output text
        min_stutter: Minimum number of times a word must repeat (default: 3)
    
    Returns:
        Dictionary with 'has_stuttering' and 'info'
    """
    if pd.isna(hypothesis) or hypothesis == '':
        return {'has_stuttering': False, 'info': ''}
    
    try:
        words = str(hypothesis).strip().split()
        if len(words) < min_stutter:
            return {'has_stuttering': False, 'info': ''}
        
        # Check for consecutive repeated words
        current_word = None
        count = 1
        for word in words:
            if word == current_word:
                count += 1
                if count >= min_stutter:
                    return {
                        'has_stuttering': True,
                        'info': f"Word stuttering: '{word}' repeated {count} times"
                    }
            else:
                current_word = word
                count = 1
        
        return {'has_stuttering': False, 'info': ''}
    except Exception as e:
        logger.warning(f"Error detecting word stuttering: {e}")
        return {'has_stuttering': False, 'info': ''}


def detect_hallucinations(hypothesis, reference, check_repetition=True, check_length=True, 
                          check_char_repetition=True, check_insertions=True, check_stuttering=True,
                          min_repeat_length=3, min_repeats=2, min_ratio=0.5, max_ratio=2.0,
                          wer_threshold=0.8):
    """
    Comprehensive hallucination detection combining multiple methods.
    
    Args:
        hypothesis: Model output text
        reference: Ground truth text
        check_repetition: Whether to check for phrase repetitions (default: True)
        check_length: Whether to check for length anomalies (default: True)
        check_char_repetition: Whether to check for character-level repetition (default: True)
        check_insertions: Whether to check for unlikely insertions (default: True)
        check_stuttering: Whether to check for word stuttering (default: True)
        min_repeat_length: Minimum phrase length for repetition detection (default: 3)
        min_repeats: Minimum number of repeats (default: 2)
        min_ratio: Minimum length ratio (default: 0.5)
        max_ratio: Maximum length ratio (default: 2.0)
        wer_threshold: WER threshold for insertion detection (default: 0.8)
    
    Returns:
        Dictionary with flags and details for all detected hallucinations
    """
    result = {
        'has_hallucination': False,
        'repetition': {'detected': False, 'info': ''},
        'length_anomaly': {'detected': False, 'type': None, 'info': ''},
        'char_repetition': {'detected': False, 'info': ''},
        'insertions': {'detected': False, 'info': ''},
        'stuttering': {'detected': False, 'info': ''}
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
    
    if check_char_repetition:
        char_result = detect_character_repetition(hypothesis)
        result['char_repetition'] = {
            'detected': char_result['has_repetition'],
            'info': char_result['info']
        }
        if char_result['has_repetition']:
            result['has_hallucination'] = True
    
    if check_insertions:
        ins_result = detect_unlikely_insertions(hypothesis, reference, wer_threshold)
        result['insertions'] = {
            'detected': ins_result['has_insertion'],
            'info': ins_result['info']
        }
        if ins_result['has_insertion']:
            result['has_hallucination'] = True
    
    if check_stuttering:
        stut_result = detect_word_stuttering(hypothesis)
        result['stuttering'] = {
            'detected': stut_result['has_stuttering'],
            'info': stut_result['info']
        }
        if stut_result['has_stuttering']:
            result['has_hallucination'] = True
    
    return result

