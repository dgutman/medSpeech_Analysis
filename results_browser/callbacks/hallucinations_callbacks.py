"""
Callbacks for the hallucinations tab.
"""

from dash import Input, Output, State
from dash.exceptions import PreventUpdate
import logging

from data_loader import get_data
from utils import extract_text_from_transcription
from config import app

logger = logging.getLogger(__name__)


def register_hallucinations_callbacks():
    """Register all hallucinations tab callbacks"""
    
    @app.callback(
        [Output("hallucinations-grid", "rowData"),
         Output("hallucinations-grid", "columnDefs")],
        [Input("main-tabs", "active_tab"),
         Input("hallucinations-flags-store", "data")],
        [State("split-filter", "value"),
         State("search-input", "value")],
        prevent_initial_call=False
    )
    def update_hallucinations_table(active_tab, hallucination_flags, split_filter, search_term):
        """Update hallucinations table with flattened structure
        
        Creates one row per hallucination instance (message + model combination).
        Only messages with hallucinations are shown. If a message has hallucinations
        in multiple models, each model's hallucination appears as a separate row.
        This makes filtering and analysis much easier.
        
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
            # Note: We always filter to only rows with hallucinations (that's the point of this tab)
            # The model selection filter is applied later when creating flattened rows
            
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
            
            # Flatten data: create one row per hallucination (message + model combination)
            # Only include rows that have hallucinations
            hallucination_data = []
            
            for idx, row in df_processed.iterrows():
                row_id = str(row.get('id', ''))
                
                # Only process rows that have hallucinations
                if row_id not in hallucination_flags:
                    continue
                
                row_flags = hallucination_flags[row_id]
                
                # Create one row per model that has a hallucination
                for model_name, flags in row_flags.items():
                    # Create a flattened record for this hallucination
                    record = {
                        'id': row_id,
                        'split': row.get('split', ''),
                        'transcription': row.get('transcription', ''),
                        'model': model_name,
                        'model_transcription': flags.get('hypothesis', ''),
                        'has_hallucination': True,  # Always true since we only show hallucinations
                        'repetition': flags.get('repetition', False),
                        'length_anomaly': flags.get('length_anomaly', False),
                        'char_repetition': flags.get('char_repetition', False),
                        'insertions': flags.get('insertions', False),
                        'stuttering': flags.get('stuttering', False),
                        'info': flags.get('info', '')
                    }
                    
                    hallucination_data.append(record)
            
            import pandas as pd
            df_hall = pd.DataFrame(hallucination_data)
            
            # Return empty if no hallucinations found
            if df_hall.empty:
                return [], []
            
            # Build column definitions for flattened structure
            columnDefs = [
                {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
                {"field": "split", "headerName": "Split", "width": 100},
                {
                    "field": "model", 
                    "headerName": "Model", 
                    "width": 150, 
                    "filter": True,
                    "floatingFilter": True,
                    "filterParams": {
                        "filterOptions": ["contains", "equals", "startsWith", "endsWith"],
                        "defaultOption": "contains"
                    }
                },
                {"field": "transcription", "headerName": "Reference", "width": 300, "wrapText": True},
                {"field": "model_transcription", "headerName": "Model Output", "width": 350, "wrapText": True},
                # Details column - contains all flag information
                {
                    "field": "info",
                    "headerName": "Details",
                    "width": 500,
                    "wrapText": True,
                    "cellClassRules": {
                        "has-details": "params.value && params.value.length > 0"
                    }
                }
            ]
            
            return df_hall.to_dict('records'), columnDefs
            
        except Exception as e:
            logger.error(f"Error filtering hallucinations table: {e}")
            return [], []



