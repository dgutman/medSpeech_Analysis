"""
Compare tab component for transcription comparison.
"""

from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import difflib
import logging

from utils import extract_text_from_transcription
from wer_utils import calculate_wer, get_wer_methods

logger = logging.getLogger(__name__)


def highlight_differences(source: str, target: str, method='basic') -> html.Div:
    """
    Compare two strings and return a Dash component with highlighted differences.

    Args:
        source: Original string (reference)
        target: Modified string (hypothesis)
        method: WER computation method ('basic' or 'ignore_punctuation')

    Returns:
        html.Div with highlighted text
    """
    if pd.isna(source) or pd.isna(target):
        return html.Div(str(source) if not pd.isna(source) else str(target))
    
    # Store original for display
    original_source = str(source)
    original_target = str(target)
    original_source_words = original_source.split()
    original_target_words = original_target.split()
    
    # Normalize strings based on method for comparison
    source_str = str(source).strip()
    target_str = str(target).strip()
    
    if method == 'ignore_punctuation':
        # Remove punctuation before comparison
        import string
        source_str = source_str.translate(str.maketrans('', '', string.punctuation))
        target_str = target_str.translate(str.maketrans('', '', string.punctuation))
        # Normalize whitespace
        source_str = ' '.join(source_str.split())
        target_str = ' '.join(target_str.split())
    
    # Split strings into words for comparison
    source_words = source_str.split()
    target_words = target_str.split()
    
    # For display, use original words
    display_source_words = original_source_words
    display_target_words = original_target_words

    # Use difflib to get the differences
    matcher = difflib.SequenceMatcher(None, source_words, target_words)

    children = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            # Words are the same - highlight in green, show target words
            words = display_target_words[j1:j2] if j2 <= len(display_target_words) else display_target_words[j1:]
            children.append(
                html.Span(
                    " ".join(words) + " ",
                    style={
                        "backgroundColor": "#d4edda",
                        "color": "#155724",
                        "padding": "1px 2px",
                        "borderRadius": "2px",
                        "fontWeight": "bold",
                    },
                )
            )
        elif tag == "replace":
            # Words are different - highlight source in red (what's missing), target in orange (what's different)
            source_diff = display_source_words[i1:i2] if i2 <= len(display_source_words) else display_source_words[i1:]
            target_diff = display_target_words[j1:j2] if j2 <= len(display_target_words) else display_target_words[j1:]

            if source_diff:
                children.append(
                    html.Span(
                        " ".join(source_diff) + " ",
                        style={
                            "backgroundColor": "#f8d7da",
                            "color": "#721c24",
                            "padding": "1px 2px",
                            "borderRadius": "2px",
                            "textDecoration": "line-through",
                        },
                    )
                )

            if target_diff:
                children.append(
                    html.Span(
                        " ".join(target_diff) + " ",
                        style={
                            "backgroundColor": "#fff3cd",
                            "color": "#856404",
                            "padding": "1px 2px",
                            "borderRadius": "2px",
                            "fontWeight": "bold",
                        },
                    )
                )
        elif tag == "delete":
            # Words deleted from source - highlight in red
            words = display_source_words[i1:i2] if i2 <= len(display_source_words) else display_source_words[i1:]
            children.append(
                html.Span(
                    " ".join(words) + " ",
                    style={
                        "backgroundColor": "#f8d7da",
                        "color": "#721c24",
                        "padding": "1px 2px",
                        "borderRadius": "2px",
                        "textDecoration": "line-through",
                    },
                )
            )
        elif tag == "insert":
            # Words added to target - highlight in orange
            # Use display words (original) for target
            words = display_target_words[j1:j2] if j2 <= len(display_target_words) else display_target_words[j1:]
            children.append(
                html.Span(
                    " ".join(words) + " ",
                    style={
                        "backgroundColor": "#fff3cd",
                        "color": "#856404",
                        "padding": "1px 2px",
                        "borderRadius": "2px",
                        "fontWeight": "bold",
                    },
                )
            )

    return html.Div(
        children,
        style={
            "fontFamily": "monospace",
            "lineHeight": "1.5",
            "whiteSpace": "pre-wrap",
        },
    )


def create_compare_tab(df, wer_method='basic', sample_id=None):
    """Create the transcription comparison tab with compact table layout"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return html.Div("No data available")
    
    try:
        # Process the dataframe to extract text from transcription columns
        df_processed = df.copy()
        
        # Process whisper columns to extract text
        whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
        for col in whisper_cols:
            df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
        
        # Select sample - by ID if provided, otherwise random
        if sample_id is not None and 'id' in df_processed.columns:
            # Try to find the sample by ID
            matching_rows = df_processed[df_processed['id'] == sample_id]
            if matching_rows.shape[0] > 0:
                sample_row = matching_rows.iloc[0]
            else:
                # ID not found, use random
                sample_row = df_processed.sample(n=1).iloc[0]
        else:
            # Random selection
            sample_row = df_processed.sample(n=1).iloc[0]
        
        # Get reference transcription
        reference_text = sample_row['transcription']
        current_sample_id = sample_row.get('id', 'Unknown')
        
        # Get all WER methods from utility module
        wer_methods = get_wer_methods()
        
        # Calculate WER for all methods for all models
        wer_results = {}  # {model_name: {method: wer_value}}
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            hypothesis_text = sample_row[col]
            wer_results[model_name] = {}
            for method_key, method_label in wer_methods.items():
                wer = calculate_wer(hypothesis_text, reference_text, method=method_key)
                wer_results[model_name][method_key] = wer
        
        # WER method pill buttons
        wer_method_pills = []
        for method_key, method_label in wer_methods.items():
            is_active = method_key == wer_method
            pill = dbc.Button(
                method_label,
                id={"type": "wer-method-btn", "index": method_key},
                color="primary" if is_active else "outline-primary",
                className="rounded-pill me-2",
                size="sm",
                outline=not is_active
            )
            wer_method_pills.append(pill)
        
        # Create table rows for detailed comparison
        table_rows = []
        
        # Reference row
        table_rows.append(
            html.Tr([
                html.Td("Reference", style={"fontWeight": "bold", "width": "150px", "verticalAlign": "top", "padding": "8px"}),
                html.Td([
                    html.Span(reference_text, style={"fontFamily": "monospace", "lineHeight": "1.5"})
                ], style={"padding": "8px"})
            ], style={"backgroundColor": "#f8f9fa"})
        )
        
        # Model rows with all WER methods inline
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            hypothesis_text = sample_row[col]
            
            # Create WER display for all methods
            wer_badges = []
            for method_key, method_label in wer_methods.items():
                wer_value = wer_results[model_name][method_key]
                wer_display = f"{wer_value * 100:.1f}%" if not pd.isna(wer_value) else "N/A"
                # Highlight the selected method
                badge_style = {
                    "fontWeight": "bold" if method_key == wer_method else "normal",
                    "color": "#0066cc" if method_key == wer_method else "#666",
                    "marginRight": "8px",
                    "fontSize": "13px",
                    "padding": "2px 6px",
                    "backgroundColor": "#e7f3ff" if method_key == wer_method else "transparent",
                    "borderRadius": "3px"
                }
                # Create compact badge text (e.g., "Basic: 9.1%")
                method_short = method_label.split('(')[0].strip()
                badge_text = f"{method_short}: {wer_display}"
                wer_badges.append(html.Span(badge_text, style=badge_style))
            
            # Create highlighted comparison using the selected method
            highlighted_comparison = highlight_differences(reference_text, hypothesis_text, method=wer_method)
            
            # Combine all WER badges with highlighted text
            combined_content = html.Div([
                html.Div(wer_badges, style={"marginBottom": "8px"}),
                highlighted_comparison
            ])
            
            table_rows.append(
                html.Tr([
                    html.Td(model_name, style={"fontWeight": "bold", "width": "150px", "verticalAlign": "top", "padding": "8px"}),
                    html.Td(combined_content, style={"padding": "8px"})
                ])
            )
        
        # Legend
        legend = html.Div([
            html.Small([
                html.Span(" ", style={"backgroundColor": "#d4edda", "padding": "2px 4px", "marginRight": "8px"}),
                "Correct ",
                html.Span(" ", style={"backgroundColor": "#f8d7da", "padding": "2px 4px", "marginRight": "8px"}),
                "Missing ",
                html.Span(" ", style={"backgroundColor": "#fff3cd", "padding": "2px 4px", "marginRight": "8px"}),
                "Different"
            ], className="text-muted")
        ], style={"marginBottom": "10px"})
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H4("🔍 Transcription Comparison", className="d-inline me-3 mb-0"),
                                html.Span(f"Sample ID: {current_sample_id}", className="text-muted")
                            ])
                        ], width="auto"),
                        dbc.Col([
                            html.Div(wer_method_pills, className="d-flex justify-content-end")
                        ], width="auto", className="ms-auto")
                    ], className="mb-2 align-items-center"),
                    # Sample selection controls
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Button(
                                    "🎲 New Random Sample",
                                    id="new-random-sample-btn",
                                    color="primary",
                                    className="me-2"
                                )
                            ], width="auto"),
                            dbc.Col([
                                html.Div([
                                    html.Label("Or specify Sample ID: ", style={"marginRight": "10px", "fontWeight": "bold"}),
                                    dbc.Input(
                                        id="sample-id-input",
                                        type="text",
                                        placeholder="Enter sample ID...",
                                        value=current_sample_id if sample_id is not None else "",
                                        style={"width": "250px", "display": "inline-block"}
                                    )
                                ], style={"display": "inline-block"})
                            ], width="auto")
                        ], className="mb-3")
                    ]),
                    legend,
                    html.Table([
                        html.Tbody(table_rows)
                    ], className="table table-bordered", style={"width": "100%", "fontSize": "14px"})
                ])
            ])
        ], fluid=True)
        
    except Exception as e:
        logger.error(f"Error creating compare tab: {e}")
        return html.Div(f"Error creating comparison: {str(e)}")



