import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional
import base64
import io
import pixeltable as pxt
from datetime import datetime
import logging
import pickle
import jiwer
import re
import difflib
from wer_utils import calculate_wer, calculate_wer_detailed, get_wer_methods, detect_hallucinations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Medical Speech Analysis Results Browser"

# Global variables for caching
cached_data = None
cache_timestamp = None

def load_env_vars():
    """Load environment variables from .env file"""
    env_vars = {}
    env_file = os.path.join(os.path.dirname(__file__), '..', '.env')
    
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value
                    os.environ[key] = value
    
    return env_vars

def load_cached_data():
    """Load data from local cache"""
    cache_dir = os.environ.get('CACHE_DIR', './cache')
    cache_file = os.path.join(cache_dir, 'dataset_cache.pkl')
    
    logger.info(f"Looking for cache file at: {cache_file}")
    logger.info(f"File exists: {os.path.exists(cache_file)}")
    
    if not os.path.exists(cache_file):
        logger.info("No cached data found")
        return None
    
    try:
        df = pd.read_pickle(cache_file)
        record_count = len(df) if hasattr(df, '__len__') else df.shape[0] if hasattr(df, 'shape') else 'unknown'
        logger.info(f"Loaded {record_count} records from cache")
        return df
    except Exception as e:
        logger.error(f"Failed to load cached data: {e}")
        return None

def load_pixeltable_data():
    """Load data from local pixeltable replica"""
    global cached_data, cache_timestamp
    
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Load environment variables
            env_vars = load_env_vars()
            if 'PIXELTABLE_API_KEY' in env_vars:
                os.environ['PIXELTABLE_API_KEY'] = env_vars['PIXELTABLE_API_KEY']
            
            # Connect to the local replicated table
            if attempt > 0:
                logger.info(f"Retrying connection to local Pixeltable replica (attempt {attempt + 1}/{max_retries})...")
            else:
                logger.info("Connecting to local Pixeltable replica...")
            
            local_table = pxt.get_table('local_hani89')
            
            # Get data from local table - collect and convert to pandas DataFrame
            logger.info("Loading data from local table...")
            df = local_table.collect().to_pandas()
            
            cached_data = df
            cache_timestamp = datetime.now()
            
            # Get record count safely
            record_count = len(df) if hasattr(df, '__len__') else df.shape[0] if hasattr(df, 'shape') else 'unknown'
            logger.info(f"Loaded {record_count} records from local Pixeltable replica")
            return df
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Error loading from local Pixeltable replica (attempt {attempt + 1}/{max_retries}): {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                import time
                time.sleep(retry_delay)
            else:
                logger.error(f"Error loading from local Pixeltable replica after {max_retries} attempts: {e}")
                logger.info("Falling back to sample data")
                # Return sample data if local table fails
                return create_sample_data()
    
    # Should not reach here, but just in case
    return create_sample_data()

def create_sample_data():
    """Create sample data for demonstration"""
    return pd.DataFrame({
        'id': [f'sample_{i}' for i in range(10)],
        'transcription': [f'Sample transcription {i}' for i in range(10)],
        'split': ['train'] * 5 + ['test'] * 5,
        'filePath': [f'/path/to/audio_{i}.wav' for i in range(10)],
        'whisper_tinyEn_transcription': [f'Whisper result {i}' for i in range(10)]
    })

def get_data():
    """Get data with caching"""
    global cached_data, cache_timestamp
    
    if cached_data is None or cache_timestamp is None:
        return load_pixeltable_data()
    
    # Check if cache is older than 5 minutes
    if (datetime.now() - cache_timestamp).seconds > 300:
        return load_pixeltable_data()
    
    return cached_data

# Add custom CSS for hallucination table styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .hallucination-yes {
                background-color: #ffcccc !important;
                text-align: center;
            }
            .hallucination-no {
                background-color: #ccffcc !important;
                text-align: center;
            }
            .flag-repetition { background-color: #ffeb3b !important; text-align: center; opacity: 0.7; }
            .flag-length-anomaly { background-color: #ff9800 !important; text-align: center; opacity: 0.7; }
            .flag-char-repetition { background-color: #f44336 !important; text-align: center; opacity: 0.7; }
            .flag-insertions { background-color: #9c27b0 !important; text-align: center; opacity: 0.7; }
            .flag-stuttering { background-color: #e91e63 !important; text-align: center; opacity: 0.7; }
            .has-details { background-color: #fff3cd !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Layout
app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("🎤 Medical Speech Analysis Results Browser", 
                   className="text-center mb-2 text-primary")
        ])
    ]),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Dataset Overview", className="py-2"),
                dbc.CardBody([
                    html.Div(id="dataset-stats", className="py-1")
                ], className="py-2")
            ], className="mb-1")
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Filters", className="py-2"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id="split-filter",
                                placeholder="Filter by split...",
                                clearable=True
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Input(
                                id="search-input",
                                placeholder="Search transcriptions...",
                                type="text"
                            )
                        ], width=6)
                    ], className="g-2")
                ], className="py-2")
            ], className="mb-1")
        ], width=6)
    ], className="mb-1"),
    
    # Main Content Tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="📋 Data Table", tab_id="table-tab"),
                dbc.Tab(label="📈 Analytics", tab_id="analytics-tab"),
                dbc.Tab(label="🔍 Compare", tab_id="compare-tab"),
                dbc.Tab(label="⚠️ Hallucinations", tab_id="hallucinations-tab"),
                dbc.Tab(label="🎵 Audio Player", tab_id="audio-tab"),
                dbc.Tab(label="🤖 Models", tab_id="models-tab")
            ], id="main-tabs", active_tab="table-tab")
        ])
    ]),
    
    # Tab Content
    dbc.Row([
        dbc.Col([
            html.Div(id="tab-content")
        ])
    ], className="mt-4"),
    
    # Hidden stores for compare tab state
    dcc.Store(id="wer-method-store", data="basic"),
    dcc.Store(id="compare-sample-id-store", data=None),
    
    # Lightweight store for hallucination flags (just row IDs and flags, not full data)
    dcc.Store(id="hallucinations-flags-store", data=None),
    
    # Footer
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Medical Speech Analysis Results Browser", 
                   className="text-center text-muted")
        ])
    ], className="mt-5")
], fluid=True)

# Callbacks
@app.callback(
    [Output("dataset-stats", "children"),
     Output("split-filter", "options")],
    [Input("main-tabs", "active_tab")]
)
def update_dataset_stats(active_tab):
    """Update dataset statistics and filter options"""
    try:
        df = get_data()
        
        # Calculate stats
        total_records = df.shape[0] if hasattr(df, 'shape') else len(df) if hasattr(df, '__len__') else 0
        splits = df['split'].value_counts().to_dict() if 'split' in df.columns else {}
        
        stats_children = [
            html.H5(
                f"Total Records: {total_records} | Train: {splits.get('train', 0)} | Test: {splits.get('test', 0)} | Validation: {splits.get('validation', 0)}",
                className="mb-0 mt-0"
            )
        ]
        
        # Filter options
        split_options = [{"label": split, "value": split} for split in splits.keys()]
        
        return stats_children, split_options
        
    except Exception as e:
        logger.error(f"Error updating stats: {e}")
        return [html.P("Error loading data")], []

@app.callback(
    [Output("tab-content", "children"),
     Output("hallucinations-flags-store", "data")],
    [Input("main-tabs", "active_tab"),
     Input("split-filter", "value"),
     Input("search-input", "value"),
     Input("wer-method-store", "data"),
     Input("compare-sample-id-store", "data")]
)
def update_tab_content(active_tab, split_filter, search_term, wer_method, sample_id):
    """Update content based on active tab and filters"""
    try:
        df = get_data()
        hallucination_flags = None
        
        # Apply filters
        if split_filter:
            df = df[df['split'] == split_filter]
        
        if search_term:
            df = df[df['transcription'].str.contains(search_term, case=False, na=False)]
        
        # Default WER method if not specified
        if wer_method is None:
            wer_method = 'basic'
        
        if active_tab == "table-tab":
            content = create_data_table(df)
        elif active_tab == "analytics-tab":
            content = create_analytics_tab(df)
        elif active_tab == "compare-tab":
            content = create_compare_tab(df, wer_method, sample_id)
        elif active_tab == "hallucinations-tab":
            content, hallucination_flags = create_hallucinations_tab(df)
        elif active_tab == "audio-tab":
            content = create_audio_tab(df)
        elif active_tab == "models-tab":
            content = create_models_tab()
        else:
            content = html.Div("Select a tab")
        
        return content, hallucination_flags
            
    except Exception as e:
        logger.error(f"Error updating tab content: {e}")
        return html.Div(f"Error: {str(e)}"), None

@app.callback(
    Output("wer-method-store", "data"),
    [Input({"type": "wer-method-btn", "index": dash.dependencies.ALL}, "n_clicks")],
    prevent_initial_call=True
)
def update_wer_method_store(n_clicks_list):
    """Update the WER method store when pill button is clicked"""
    ctx = callback_context
    if not ctx.triggered:
        raise PreventUpdate
    
    # Find which button was clicked
    triggered_id = ctx.triggered[0]['prop_id']
    if 'wer-method-btn' in triggered_id:
        # Extract the method key from the button ID
        button_prop = json.loads(triggered_id.split('.')[0])
        return button_prop['index']
    
    raise PreventUpdate

@app.callback(
    Output("compare-sample-id-store", "data"),
    [Input("main-tabs", "active_tab")],
    [State("compare-sample-id-store", "data")],
    prevent_initial_call=False
)
def handle_tab_change_for_sample_id(active_tab, current_id):
    """Handle sample ID when switching to compare tab"""
    # When switching to compare tab, if no ID is set, generate one
    if active_tab == "compare-tab" and current_id is None:
        try:
            df = get_data()
            if df is not None and not (hasattr(df, 'empty') and df.empty):
                if 'id' in df.columns:
                    random_sample = df.sample(n=1).iloc[0]
                    return random_sample.get('id', None)
        except Exception as e:
            logger.error(f"Error generating random sample ID: {e}")
    if current_id is None:
        raise PreventUpdate
    return current_id

@app.callback(
    Output("compare-sample-id-store", "data", allow_duplicate=True),
    [Input("new-random-sample-btn", "n_clicks")],
    [State("compare-sample-id-store", "data")],
    prevent_initial_call=True
)
def handle_random_sample_click(n_clicks, current_id):
    """Handle random sample button click"""
    if not n_clicks:
        raise PreventUpdate
    try:
        df = get_data()
        if df is not None and not (hasattr(df, 'empty') and df.empty):
            if 'id' in df.columns:
                random_sample = df.sample(n=1).iloc[0]
                return random_sample.get('id', None)
    except Exception as e:
        logger.error(f"Error generating random sample ID: {e}")
    raise PreventUpdate

@app.callback(
    Output("compare-sample-id-store", "data", allow_duplicate=True),
    [Input("sample-id-input", "n_submit")],
    [State("sample-id-input", "value"),
     State("compare-sample-id-store", "data")],
    prevent_initial_call=True
)
def handle_sample_id_input(n_submit, sample_id_input, current_id):
    """Handle sample ID input submission"""
    if not n_submit:
        raise PreventUpdate
    # Use the entered ID when Enter is pressed (or None if empty)
    return sample_id_input if sample_id_input else None

@app.callback(
    [Output("hallucinations-grid", "rowData"),
     Output("hallucinations-grid", "columnDefs")],
    [Input("hallucinations-filter-checkbox", "value"),
     Input("hallucinations-model-selector", "value"),
     Input("main-tabs", "active_tab"),
     Input("split-filter", "value"),
     Input("search-input", "value"),
     Input("hallucinations-flags-store", "data")],
    prevent_initial_call=False
)
def update_hallucinations_table(filter_hallucinations, selected_models, active_tab, split_filter, search_term, hallucination_flags):
    """Filter hallucinations table based on checkbox and model selection
    
    Uses lightweight hallucination_flags store to efficiently filter rows before processing.
    """
    if active_tab != "hallucinations-tab":
        raise PreventUpdate
    
    if hallucination_flags is None:
        return [], []
    
    try:
        # Get fresh data
        df = get_data()
        
        # Apply global filters first
        if split_filter:
            df = df[df['split'] == split_filter]
        
        if search_term:
            df = df[df['transcription'].str.contains(search_term, case=False, na=False)]
        
        # Filter rows using lightweight flags store BEFORE processing
        # Only rows with hallucinations are in the flags store
        if filter_hallucinations and "filter" in filter_hallucinations:
            # Only keep rows that have hallucinations in selected models
            selected_models_list = selected_models or []
            if selected_models_list:
                row_ids_with_hallucinations = set()
                # Only iterate through rows that have hallucinations (stored in flags)
                for row_id, model_flags in hallucination_flags.items():
                    # Check if any selected model has a hallucination for this row
                    for model_name in selected_models_list:
                        if model_name in model_flags:  # Model has hallucination if it's in the flags
                            row_ids_with_hallucinations.add(row_id)
                            break
                # Filter dataframe to only rows with hallucinations
                df = df[df['id'].astype(str).isin(row_ids_with_hallucinations)]
        
        # Now process only the filtered rows
        df_processed = df.copy()
        whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
        for col in whisper_cols:
            df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
        
        # Build model name to column mapping
        model_name_to_col = {}
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            model_name_to_col[model_name] = col
        
        # Calculate full data only for filtered rows (we already have flags, but need full records)
        hallucination_data = []
        
        for idx, row in df_processed.iterrows():
            row_id = str(row.get('id', ''))
            record = {
                'id': row_id,
                'split': row.get('split', ''),
                'transcription': row.get('transcription', '')
            }
            
            reference = row.get('transcription', '')
            
            # Get flags from store (if row has hallucinations, it will be in the store)
            row_flags = hallucination_flags.get(row_id, {})
            
            # Check each whisper model for hallucinations
            for col in whisper_cols:
                model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
                hypothesis = row.get(col, '')
                
                # Use flags from store if available, otherwise no hallucinations
                if row_id in hallucination_flags and model_name in row_flags:
                    # Use stored flags (this model has hallucinations)
                    flags = row_flags[model_name]
                    record[f'{model_name}_has_hallucination'] = True  # Always true if in store
                    record[f'{model_name}_repetition'] = flags.get('repetition', False)
                    record[f'{model_name}_length_anomaly'] = flags.get('length_anomaly', False)
                    record[f'{model_name}_char_repetition'] = flags.get('char_repetition', False)
                    record[f'{model_name}_insertions'] = flags.get('insertions', False)
                    record[f'{model_name}_stuttering'] = flags.get('stuttering', False)
                    
                    # Build info string from flags
                    all_issues = []
                    if flags.get('repetition', False):
                        all_issues.append("Repetition detected")
                    if flags.get('length_anomaly', False):
                        all_issues.append("Length anomaly detected")
                    if flags.get('char_repetition', False):
                        all_issues.append("Char repetition detected")
                    if flags.get('insertions', False):
                        all_issues.append("Insertions detected")
                    if flags.get('stuttering', False):
                        all_issues.append("Stuttering detected")
                    record[f'{model_name}_info'] = '; '.join(all_issues) if all_issues else ''
                else:
                    # Row not in store or model not in row flags = no hallucinations
                    record[f'{model_name}_has_hallucination'] = False
                    record[f'{model_name}_repetition'] = False
                    record[f'{model_name}_length_anomaly'] = False
                    record[f'{model_name}_char_repetition'] = False
                    record[f'{model_name}_insertions'] = False
                    record[f'{model_name}_stuttering'] = False
                    record[f'{model_name}_info'] = ''
                
                record[col] = hypothesis  # Keep original column name for reference
            
            hallucination_data.append(record)
        
        df_hall = pd.DataFrame(hallucination_data)
        
        # Build column definitions for selected models
        columnDefs = [
            {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
            {"field": "split", "headerName": "Split", "width": 100},
            {"field": "transcription", "headerName": "Reference", "width": 300, "wrapText": True},
        ]
        
        # Only add columns for selected models
        selected_models_list = selected_models or []
        for model_name in selected_models_list:
            col = model_name_to_col.get(model_name)
            if not col:
                continue
            
            # Main hallucination flag
            columnDefs.append({
                "field": f"{model_name}_has_hallucination",
                "headerName": f"{model_name} ⚠️",
                "width": 80,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellRendererParams": {"disabled": True},
                "cellClassRules": {
                    "hallucination-yes": "params.value === true",
                    "hallucination-no": "params.value === false"
                }
            })
            
            # Individual detection flags
            detection_flags = [
                ("repetition", "Rep", "#ffeb3b"),
                ("length_anomaly", "Len", "#ff9800"),
                ("char_repetition", "Char", "#f44336"),
                ("insertions", "Ins", "#9c27b0"),
                ("stuttering", "Stut", "#e91e63")
            ]
            
            for flag_key, label, color in detection_flags:
                # Create unique class name for this flag type
                flag_class = f"flag-{flag_key.lower().replace('_', '-')}"
                columnDefs.append({
                    "field": f"{model_name}_{flag_key}",
                    "headerName": f"{model_name} {label}",
                    "width": 70,
                    "cellRenderer": "agCheckboxCellRenderer",
                    "cellRendererParams": {"disabled": True},
                    "cellClassRules": {
                        flag_class: "params.value === true"
                    }
                })
            
            # Details column
            columnDefs.append({
                "field": f"{model_name}_info",
                "headerName": f"{model_name} Details",
                "width": 400,
                "wrapText": True,
                "cellClassRules": {
                    "has-details": "params.value && params.value.length > 0"
                }
            })
            
            # Transcription column
            columnDefs.append({
                "field": col,
                "headerName": f"{model_name} Text",
                "width": 350,
                "wrapText": True
            })
        
        return df_hall.to_dict('records'), columnDefs
        
    except Exception as e:
        logger.error(f"Error filtering hallucinations table: {e}")
        return [], []

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

# WER computation functions have been moved to wer_utils.py

def create_data_table(df):
    """Create the data table tab"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return dbc.Spinner(
            html.Div(id="data-table-loading", style={"height": "600px"}),
            fullscreen=False,
            spinner_style={"width": "3rem", "height": "3rem"}
        )
    
    # Process the dataframe to extract text from transcription columns
    df_processed = df.copy()
    
    # Process whisper columns to extract text
    whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
    for col in whisper_cols:
        df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
    
    # Prepare columns for AG Grid
    columnDefs = [
        {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
        {"field": "split", "headerName": "Split", "width": 100},
        {"field": "transcription", "headerName": "Transcription", "width": 300, "wrapText": True},
    ]
    
    # Add whisper columns if they exist
    for col in whisper_cols:
        columnDefs.append({
            "field": col, 
            "headerName": col.replace('_', ' ').title(), 
            "width": 300, 
            "wrapText": True
        })
    
    return dcc.Loading(
        id="data-table-loading",
        type="default",
        children=dag.AgGrid(
            id="data-grid",
            columnDefs=columnDefs,
            rowData=df_processed.to_dict('records'),
            defaultColDef={
                "resizable": True,
                "sortable": True,
                "filter": True,
                "floatingFilter": False
            },
            dashGridOptions={
                "pagination": True,
                "paginationPageSize": 20,
                "suppressRowClickSelection": False,
                "rowSelection": "single"
            },
            style={"height": "600px", "width": "100%"}
        )
    )

def create_hallucinations_tab(df):
    """Create the hallucination detection tab with filters and model selection
    
    Returns:
        tuple: (html_content, hallucination_flags_dict)
        - html_content: The tab UI layout
        - hallucination_flags_dict: Lightweight dict mapping row_id -> {model: {flags}}
    """
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return dbc.Spinner(
            html.Div(id="hallucinations-table-loading", style={"height": "600px"}),
            fullscreen=False,
            spinner_style={"width": "3rem", "height": "3rem"}
        ), None
    
    try:
        # Process the dataframe to extract text from transcription columns
        df_processed = df.copy()
        
        # Process whisper columns to extract text
        whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
        for col in whisper_cols:
            df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
        
        # Calculate hallucination flags for each model
        hallucination_data = []
        # Only store rows that have hallucinations: {row_id: {model: {flags}}}
        # If a row_id is not in this dict, it means no hallucinations
        hallucination_flags = {}
        
        for idx, row in df_processed.iterrows():
            row_id = str(row.get('id', ''))
            record = {
                'id': row_id,
                'split': row.get('split', ''),
                'transcription': row.get('transcription', '')
            }
            
            reference = row.get('transcription', '')
            
            # Track if this row has any hallucinations (across any model)
            row_has_hallucinations = False
            row_model_flags = {}  # Only store models with hallucinations for this row
            
            # Check each whisper model for hallucinations
            for col in whisper_cols:
                model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
                hypothesis = row.get(col, '')
                
                # Detect hallucinations with all methods
                hall_result = detect_hallucinations(hypothesis, reference)
                
                # Only store flags if this model has hallucinations
                if hall_result['has_hallucination']:
                    row_has_hallucinations = True
                    row_model_flags[model_name] = {
                        'has_hallucination': True,
                        'repetition': hall_result['repetition']['detected'],
                        'length_anomaly': hall_result['length_anomaly']['detected'],
                        'char_repetition': hall_result['char_repetition']['detected'],
                        'insertions': hall_result['insertions']['detected'],
                        'stuttering': hall_result['stuttering']['detected']
                    }
                
                # Add flags to record for DataFrame (always, for table display)
                record[f'{model_name}_has_hallucination'] = hall_result['has_hallucination']
                record[f'{model_name}_repetition'] = hall_result['repetition']['detected']
                record[f'{model_name}_length_anomaly'] = hall_result['length_anomaly']['detected']
                record[f'{model_name}_length_type'] = hall_result['length_anomaly']['type'] or ''
                record[f'{model_name}_char_repetition'] = hall_result['char_repetition']['detected']
                record[f'{model_name}_insertions'] = hall_result['insertions']['detected']
                record[f'{model_name}_stuttering'] = hall_result['stuttering']['detected']
                
                # Collect all detected issues for info column
                all_issues = []
                if hall_result['repetition']['detected']:
                    all_issues.append(f"Repetition: {hall_result['repetition']['info']}")
                if hall_result['length_anomaly']['detected']:
                    all_issues.append(f"Length: {hall_result['length_anomaly']['info']}")
                if hall_result['char_repetition']['detected']:
                    all_issues.append(f"Char rep: {hall_result['char_repetition']['info']}")
                if hall_result['insertions']['detected']:
                    all_issues.append(f"Insertions: {hall_result['insertions']['info']}")
                if hall_result['stuttering']['detected']:
                    all_issues.append(f"Stuttering: {hall_result['stuttering']['info']}")
                
                record[f'{model_name}_info'] = '; '.join(all_issues) if all_issues else ''
                record[col] = hypothesis  # Keep original column name for reference
            
            # Only add this row to flags store if it has hallucinations
            if row_has_hallucinations:
                hallucination_flags[row_id] = row_model_flags
            
            hallucination_data.append(record)
        
        # Create DataFrame from hallucination results
        df_hall = pd.DataFrame(hallucination_data)
        
        # Get model names for selection dropdown
        model_options = []
        model_name_to_col = {}
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            model_options.append({"label": model_name, "value": model_name})
            model_name_to_col[model_name] = col
        
        # Calculate hallucination statistics for each model
        hallucination_stats = []
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            
            total = len(df_hall)
            has_hallucination = df_hall[f'{model_name}_has_hallucination'].sum() if f'{model_name}_has_hallucination' in df_hall.columns else 0
            hallucination_rate = (has_hallucination / total * 100) if total > 0 else 0
            
            # Breakdown by type
            repetition_count = df_hall[f'{model_name}_repetition'].sum() if f'{model_name}_repetition' in df_hall.columns else 0
            length_count = df_hall[f'{model_name}_length_anomaly'].sum() if f'{model_name}_length_anomaly' in df_hall.columns else 0
            char_rep_count = df_hall[f'{model_name}_char_repetition'].sum() if f'{model_name}_char_repetition' in df_hall.columns else 0
            insertions_count = df_hall[f'{model_name}_insertions'].sum() if f'{model_name}_insertions' in df_hall.columns else 0
            stuttering_count = df_hall[f'{model_name}_stuttering'].sum() if f'{model_name}_stuttering' in df_hall.columns else 0
            
            hallucination_stats.append({
                'Model': model_name,
                'Total Samples': total,
                'With Hallucinations': int(has_hallucination),
                'Hallucination Rate (%)': f"{hallucination_rate:.1f}",
                'Repetition': int(repetition_count),
                'Length Anomaly': int(length_count),
                'Char Repetition': int(char_rep_count),
                'Insertions': int(insertions_count),
                'Stuttering': int(stuttering_count)
            })
        
        stats_df = pd.DataFrame(hallucination_stats)
        # Sort by hallucination rate (highest first)
        stats_df['_sort_key'] = stats_df['Hallucination Rate (%)'].str.replace('%', '').astype(float)
        stats_df = stats_df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
        
        def get_rate_color(rate):
            """Return color based on hallucination rate"""
            if rate < 10:
                return '#d4edda'  # Light green - excellent
            elif rate < 20:
                return '#fff3cd'  # Light yellow - good
            elif rate < 30:
                return '#ffeaa7'  # Yellow - moderate
            else:
                return '#f8d7da'  # Light red - poor
        
        # Create summary statistics table with color coding
        rate_colors = [get_rate_color(float(rate.replace('%', ''))) for rate in stats_df['Hallucination Rate (%)']]
        
        stats_table = go.Figure(data=[go.Table(
            header=dict(
                values=list(stats_df.columns),
                fill_color='#2c3e50',
                align='center',
                font=dict(size=12, color='white', family='Arial'),
                height=35
            ),
            cells=dict(
                values=[stats_df[col] for col in stats_df.columns],
                fill_color=[
                    ['white'] * len(stats_df),  # Model
                    ['#f8f9fa'] * len(stats_df),  # Total Samples
                    ['#f8f9fa'] * len(stats_df),  # With Hallucinations
                    rate_colors,  # Rate - color coded
                    ['white'] * len(stats_df),  # Repetition
                    ['white'] * len(stats_df),  # Length
                    ['white'] * len(stats_df),  # Char Rep
                    ['white'] * len(stats_df),  # Insertions
                    ['white'] * len(stats_df)   # Stuttering
                ],
                align='center',
                font=dict(size=11, color='black', family='Arial'),
                height=30
            )
        )])
        stats_table.update_layout(
            title=dict(
                text="📊 Hallucination Statistics by Model",
                font=dict(size=14, family='Arial', color='#2c3e50'),
                x=0.5
            ),
            height=200,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Control panel with filters
        controls = dbc.Card([
            dbc.CardHeader("🎯 Hallucination Filters", className="py-2"),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Show only rows with hallucinations:", className="fw-bold mb-2"),
                        dbc.Checklist(
                            id="hallucinations-filter-checkbox",
                            options=[{"label": " Filter hallucinations only", "value": "filter"}],
                            value=[],
                            inline=True,
                            switch=True
                        )
                    ], width=6),
                    dbc.Col([
                        dbc.Label("Select models to display:", className="fw-bold mb-2"),
                        dcc.Dropdown(
                            id="hallucinations-model-selector",
                            options=model_options,
                            value=[opt["value"] for opt in model_options],  # All models selected by default
                            multi=True,
                            placeholder="Select models...",
                            clearable=False
                        )
                    ], width=6)
                ], className="g-3")
            ], className="py-3")
        ], className="mb-3")
        
        # Prepare columns for AG Grid with better organization
        columnDefs = [
            {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
            {"field": "split", "headerName": "Split", "width": 100},
            {"field": "transcription", "headerName": "Reference", "width": 300, "wrapText": True},
        ]
        
        # Add hallucination detection columns for each model
        for col in whisper_cols:
            model_name = col.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            
            # Main hallucination flag with color coding
            columnDefs.append({
                "field": f"{model_name}_has_hallucination",
                "headerName": f"{model_name} ⚠️",
                "width": 80,
                "cellRenderer": "agCheckboxCellRenderer",
                "cellRendererParams": {"disabled": True},
                "cellClassRules": {
                    "hallucination-yes": "params.value === true",
                    "hallucination-no": "params.value === false"
                }
            })
            
            # Individual detection flags
            detection_flags = [
                ("repetition", "Rep", "#ffeb3b"),
                ("length_anomaly", "Len", "#ff9800"),
                ("char_repetition", "Char", "#f44336"),
                ("insertions", "Ins", "#9c27b0"),
                ("stuttering", "Stut", "#e91e63")
            ]
            
            for flag_key, label, color in detection_flags:
                # Create unique class name for this flag type
                flag_class = f"flag-{flag_key.lower().replace('_', '-')}"
                columnDefs.append({
                    "field": f"{model_name}_{flag_key}",
                    "headerName": f"{model_name} {label}",
                    "width": 70,
                    "cellRenderer": "agCheckboxCellRenderer",
                    "cellRendererParams": {"disabled": True},
                    "cellClassRules": {
                        flag_class: "params.value === true"
                    }
                })
            
            # Details column
            columnDefs.append({
                "field": f"{model_name}_info",
                "headerName": f"{model_name} Details",
                "width": 400,
                "wrapText": True,
                "cellClassRules": {
                    "has-details": "params.value && params.value.length > 0"
                }
            })
            
            # Transcription column
            columnDefs.append({
                "field": col,
                "headerName": f"{model_name} Text",
                "width": 350,
                "wrapText": True
            })
        
        # Initial table data (will be updated by callback)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=stats_table, config={'displayModeBar': False})
                ], width=12)
            ], className="mb-3"),
            controls,
            dcc.Loading(
                id="hallucinations-table-loading",
                type="default",
                children=dag.AgGrid(
                    id="hallucinations-grid",
                    columnDefs=columnDefs,
                    rowData=[],  # Empty initially, populated by callback
                    defaultColDef={
                        "resizable": True,
                        "sortable": True,
                        "filter": True,
                        "floatingFilter": False
                    },
                    dashGridOptions={
                        "pagination": True,
                        "paginationPageSize": 20,
                        "suppressRowClickSelection": False,
                        "rowSelection": "single"
                    },
                    style={"height": "600px", "width": "100%"}
                )
            )
        ]), hallucination_flags
        
    except Exception as e:
        logger.error(f"Error creating hallucinations tab: {e}")
        return html.Div(f"Error creating hallucinations table: {str(e)}"), None

def create_models_tab():
    """Create the models information tab with details about Whisper models"""
    
    # Whisper model specifications
    models_data = [
        {
            "Model": "Tiny (English)",
            "Full Name": "Whisper Tiny English",
            "Parameters": "39M",
            "Size": "~75 MB",
            "Speed": "Fastest (real-time factor ~0.01-0.1x)",
            "Multi-language": False,
            "Accuracy": "Basic",
            "Use Case": "Quick prototyping, low-latency applications",
            "Notes": "English-only variant, optimized for speed"
        },
        {
            "Model": "Base (English)",
            "Full Name": "Whisper Base English",
            "Parameters": "74M",
            "Size": "~140 MB",
            "Speed": "Very Fast (real-time factor ~0.1-0.3x)",
            "Multi-language": False,
            "Accuracy": "Good",
            "Use Case": "Production applications requiring speed",
            "Notes": "English-only variant, good speed/accuracy balance"
        },
        {
            "Model": "Small (English)",
            "Full Name": "Whisper Small English",
            "Parameters": "244M",
            "Size": "~460 MB",
            "Speed": "Fast (real-time factor ~0.3-0.8x)",
            "Multi-language": False,
            "Accuracy": "Very Good",
            "Use Case": "High-accuracy English transcription",
            "Notes": "English-only variant, strong accuracy"
        },
        {
            "Model": "Medium (English)",
            "Full Name": "Whisper Medium English",
            "Parameters": "769M",
            "Size": "~1.5 GB",
            "Speed": "Moderate (real-time factor ~0.8-2x)",
            "Multi-language": False,
            "Accuracy": "Excellent",
            "Use Case": "High-quality English transcription",
            "Notes": "English-only variant, excellent accuracy"
        },
        {
            "Model": "Large",
            "Full Name": "Whisper Large",
            "Parameters": "1550M",
            "Size": "~3 GB",
            "Speed": "Slow (real-time factor ~2-5x)",
            "Multi-language": True,
            "Accuracy": "Best",
            "Use Case": "Highest quality transcription, multi-language support",
            "Notes": "Full multilingual model, best accuracy, slower inference"
        },
        {
            "Model": "Turbo",
            "Full Name": "Whisper Turbo",
            "Parameters": "1550M",
            "Size": "~3 GB",
            "Speed": "Moderate-Fast (real-time factor ~0.5-1.5x)",
            "Multi-language": True,
            "Accuracy": "Excellent",
            "Use Case": "High-quality transcription with faster inference",
            "Notes": "Optimized version of Large model, 2-3x faster while maintaining accuracy"
        }
    ]
    
    # Create DataFrame for easier manipulation
    models_df = pd.DataFrame(models_data)
    
    # Create a styled table using Plotly
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "Parameters", "Size", "Inference Speed", "Multi-language", "Accuracy", "Use Case"],
            fill_color='#2c3e50',
            align='left',
            font=dict(size=12, color='white', family='Arial', weight='bold'),
            height=40
        ),
        cells=dict(
            values=[
                models_df['Model'],
                models_df['Parameters'],
                models_df['Size'],
                models_df['Speed'],
                models_df['Multi-language'].apply(lambda x: "✅ Yes" if x else "❌ No"),
                models_df['Accuracy'],
                models_df['Use Case']
            ],
            fill_color=[
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df)
            ],
            align='left',
            font=dict(size=11, color='black', family='Arial'),
            height=35
        )
    )])
    
    table_fig.update_layout(
        title=dict(
            text="🤖 Whisper Model Specifications",
            font=dict(size=18, family='Arial', color='#2c3e50'),
            x=0.5
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Create detailed cards for each model
    model_cards = []
    for _, model in models_df.iterrows():
        # Color code based on model size/accuracy
        if "Tiny" in model['Model'] or "Base" in model['Model']:
            card_color = "light"
            header_color = "secondary"
        elif "Small" in model['Model'] or "Medium" in model['Model']:
            card_color = "info"
            header_color = "info"
        else:  # Large or Turbo
            card_color = "primary"
            header_color = "primary"
        
        lang_badge = dbc.Badge(
            "🌍 Multilingual" if model['Multi-language'] else "🇺🇸 English Only",
            color="success" if model['Multi-language'] else "warning",
            className="me-2"
        )
        
        card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    model['Model'],
                    " ",
                    lang_badge
                ], className="mb-0")
            ], className=f"bg-{header_color} text-white"),
            dbc.CardBody([
                html.P([
                    html.Strong("Full Name: "),
                    model['Full Name']
                ], className="mb-2"),
                html.P([
                    html.Strong("Parameters: "),
                    model['Parameters']
                ], className="mb-2"),
                html.P([
                    html.Strong("Model Size: "),
                    model['Size']
                ], className="mb-2"),
                html.P([
                    html.Strong("Inference Speed: "),
                    model['Speed']
                ], className="mb-2"),
                html.P([
                    html.Strong("Accuracy: "),
                    dbc.Badge(model['Accuracy'], color="success", className="ms-1")
                ], className="mb-2"),
                html.P([
                    html.Strong("Use Case: "),
                    model['Use Case']
                ], className="mb-2"),
                html.P([
                    html.Strong("Notes: "),
                    html.Em(model['Notes'], style={"color": "#6c757d"})
                ], className="mb-0")
            ])
        ], className="mb-3", color=card_color, outline=True)
        
        model_cards.append(dbc.Col(card, width=12, md=6, lg=4))
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("Model Overview", className="mb-3"),
                html.P([
                    "This dataset uses OpenAI's Whisper models for automatic speech recognition (ASR). ",
                    "Whisper is a family of transformer-based models trained on multilingual and multitask ",
                    "supervised data. The models vary in size and capabilities, with trade-offs between ",
                    "accuracy, speed, and computational requirements."
                ], className="mb-4"),
                dcc.Graph(figure=table_fig, config={'displayModeBar': False})
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H4("Detailed Model Information", className="mb-3")
            ], width=12)
        ]),
        dbc.Row(model_cards),
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    html.Strong("References: "),
                    html.A("Whisper Paper", href="https://arxiv.org/abs/2212.04356", target="_blank", className="me-3"),
                    html.A("OpenAI Whisper", href="https://github.com/openai/whisper", target="_blank", className="me-3"),
                    html.A("Model Cards", href="https://huggingface.co/models?search=whisper", target="_blank")
                ], className="text-muted small")
            ], width=12)
        ])
    ])

def create_analytics_tab(df):
    """Create the analytics tab with comprehensive WER analysis"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return html.Div("No data available")
    
    try:
        # Process the dataframe to extract text from transcription columns
        df_processed = df.copy()
        
        # Process whisper columns to extract text
        whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
        for col in whisper_cols:
            df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
        
        # Calculate WER for each model
        model_wer_cols = {}
        for col in whisper_cols:
            wer_col = col.replace('_transcription', '_wer')
            df_processed[wer_col] = df_processed.apply(
                lambda row: calculate_wer(row[col], row['transcription']), axis=1
            )
            model_wer_cols[col] = wer_col
        
        # Create model comparison data
        model_order = [
            'whisper_tinyEn_transcription',
            'whisper_smallEn_transcription', 
            'whisper_mediumEn_transcription',
            'whisper_large_transcription',
            'whisper_turbo_transcription'
        ]
        
        # Filter to only include models that exist in the data
        available_models = [col for col in model_order if col in df_processed.columns]
        
        if not available_models:
            return html.Div("No Whisper model data available for analysis")
        
        # Prepare data for visualization
        wer_data = []
        for model in available_models:
            wer_col = model_wer_cols[model]
            model_name = model.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            
            for _, row in df_processed.iterrows():
                if not pd.isna(row[wer_col]):
                    wer_data.append({
                        'Model': model_name,
                        'WER': row[wer_col] * 100,  # Convert to percentage
                        'ID': row.get('id', 'unknown')
                    })
        
        wer_df = pd.DataFrame(wer_data)
        
        if wer_df.empty:
            return html.Div("No valid WER data available for analysis")
        
        # Calculate mean WER for each model (for annotations)
        mean_wer_by_model = wer_df.groupby('Model')['WER'].mean()
        y_max = min(120, wer_df['WER'].max() + 10)
        annotation_y_position = y_max * 0.95  # Position annotations near the top of the plot
        
        # Create visualizations
        # 1. Box plot comparison
        fig_box = px.box(
            wer_df,
            x='Model',
            y='WER',
            title="Word Error Rate Comparison Across Whisper Models",
            color='Model',
            labels={"WER": "Word Error Rate (%)", "Model": "Whisper Model"},
            template="plotly_white"
        )
        fig_box.update_layout(
            title_font_size=16,
            yaxis=dict(range=[0, y_max]),
            xaxis_title="",
            yaxis_title="Word Error Rate (%)",
            font=dict(size=12),
            showlegend=False
        )
        
        # Add mean WER annotations to box plot
        annotations_box = []
        for model in mean_wer_by_model.index:
            mean_value = mean_wer_by_model[model]
            annotations_box.append(
                dict(
                    x=model,
                    y=annotation_y_position,
                    text=f"μ={mean_value:.1f}%",
                    showarrow=False,
                    font=dict(size=11, color='#2c3e50', family='Arial Black'),
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='#2c3e50',
                    borderwidth=1.5,
                    borderpad=4
                )
            )
        fig_box.update_layout(annotations=annotations_box)
        
        # 2. Violin plot for distribution shape
        fig_violin = px.violin(
            wer_df,
            x='Model',
            y='WER',
            title="WER Distribution Across Whisper Models",
            color='Model',
            labels={"WER": "Word Error Rate (%)", "Model": "Whisper Model"},
            template="plotly_white",
            box=True,
            points="outliers"
        )
        fig_violin.update_layout(
            title_font_size=16,
            yaxis=dict(range=[0, y_max]),
            xaxis_title="",
            yaxis_title="Word Error Rate (%)",
            font=dict(size=12),
            showlegend=False
        )
        
        # Add mean WER annotations to violin plot
        annotations_violin = []
        for model in mean_wer_by_model.index:
            mean_value = mean_wer_by_model[model]
            annotations_violin.append(
                dict(
                    x=model,
                    y=annotation_y_position,
                    text=f"μ={mean_value:.1f}%",
                    showarrow=False,
                    font=dict(size=11, color='#2c3e50', family='Arial Black'),
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='#2c3e50',
                    borderwidth=1.5,
                    borderpad=4
                )
            )
        fig_violin.update_layout(annotations=annotations_violin)
        
        # 3. Summary statistics table with improved formatting
        summary_stats = wer_df.groupby('Model')['WER'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ])
        summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        
        # Sort by Mean WER (best to worst)
        summary_stats = summary_stats.sort_values('Mean')
        
        # Format values appropriately
        formatted_data = {
            'Model': summary_stats.index.tolist(),
            'Count': [f"{int(count)}" for count in summary_stats['Count']],
            'Mean WER (%)': [f"{mean:.1f}" for mean in summary_stats['Mean']],
            'Std Dev': [f"{std:.1f}" for std in summary_stats['Std Dev']],
            'Min': [f"{min_val:.1f}" for min_val in summary_stats['Min']],
            'Max': [f"{max_val:.1f}" for max_val in summary_stats['Max']],
            'Median': [f"{median:.1f}" for median in summary_stats['Median']]
        }
        
        # Color code cells based on Mean WER (green=good, yellow=medium, red=bad)
        def get_wer_color(wer_value):
            """Return color based on WER value"""
            if wer_value < 15:
                return '#d4edda'  # Light green - excellent
            elif wer_value < 25:
                return '#fff3cd'  # Light yellow - good
            elif wer_value < 35:
                return '#ffeaa7'  # Yellow - moderate
            else:
                return '#f8d7da'  # Light red - poor
        
        # Prepare cell colors for Mean WER column
        mean_colors = [get_wer_color(mean) for mean in summary_stats['Mean']]
        
        table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Model', 'Count', 'Mean WER (%)', 'Std Dev', 'Min', 'Max', 'Median'],
                fill_color='#2c3e50',
                align='left',
                font=dict(size=13, color='white', family='Arial'),
                height=40
            ),
            cells=dict(
                values=[
                    formatted_data['Model'],
                    formatted_data['Count'],
                    formatted_data['Mean WER (%)'],
                    formatted_data['Std Dev'],
                    formatted_data['Min'],
                    formatted_data['Max'],
                    formatted_data['Median']
                ],
                fill_color=[
                    ['white'] * len(summary_stats),  # Model column
                    ['#f8f9fa'] * len(summary_stats),  # Count column
                    mean_colors,  # Mean WER - color coded
                    ['white'] * len(summary_stats),  # Std Dev
                    ['#f8f9fa'] * len(summary_stats),  # Min
                    ['white'] * len(summary_stats),  # Max
                    ['#f8f9fa'] * len(summary_stats)  # Median
                ],
                align=['left', 'center', 'center', 'center', 'center', 'center', 'center'],
                font=dict(size=12, color='black', family='Arial'),
                height=35
            )
        )])
        table_fig.update_layout(
            title=dict(
                text="📊 WER Summary Statistics (sorted by Mean WER)",
                font=dict(size=16, family='Arial', color='#2c3e50'),
                x=0.5
            ),
            height=350,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # 4. Heatmap of WER by sample and model
        pivot_df = wer_df.pivot(index='ID', columns='Model', values='WER')
        fig_heatmap = px.imshow(
            pivot_df,
            labels=dict(x="Model", y="Sample ID", color="WER"),
            color_continuous_scale='Reds',
            aspect='auto',
            title="WER Heatmap by Sample and Model"
        )
        fig_heatmap.update_layout(
            yaxis=dict(title='Sample ID'),
            xaxis=dict(title='Whisper Model'),
            coloraxis_colorbar=dict(title="Word Error Rate (%)"),
            font=dict(size=10),
            title_font_size=16,
            height=400
        )
        
        return dbc.Container([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_box)
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=fig_violin)
                ], width=6)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=table_fig)
                ], width=6),
                dbc.Col([
                    dcc.Graph(figure=fig_heatmap)
                ], width=6)
            ])
        ])
        
    except Exception as e:
        logger.error(f"Error creating analytics tab: {e}")
        return html.Div(f"Error creating analytics: {str(e)}")

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

def create_audio_tab(df):
    """Create the audio player tab"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return html.Div("No data available")
    
    return dbc.Card([
        dbc.CardHeader("🎵 Audio Player"),
        dbc.CardBody([
            html.P("Audio playback functionality would be implemented here."),
            html.P("This would include audio controls and waveform visualization."),
            html.P("For now, showing sample data:"),
            html.Pre(json.dumps(df.head(3).to_dict('records'), indent=2))
        ])
    ])

# Health check endpoint
@app.server.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Define server for Gunicorn
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
