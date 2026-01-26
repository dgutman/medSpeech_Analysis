"""
Simple callbacks for fast data table loading
Minimal dependencies, no cascading updates
"""

from dash import Input, Output, State, html, callback_context
from dash.exceptions import PreventUpdate
import dash.dependencies
import logging
import time
import json

from data_loader import (
    load_pixeltable_data_paginated,
    get_total_count,
    get_pixeltable_table
)

logger = logging.getLogger(__name__)

# Global caches for static tabs (computed once, served forever)
_analytics_cache = None
_models_cache = None
_audio_cache = None
_compare_data_cache = None  # Cache full dataset for compare tab
_hallucinations_data_cache = None  # Cache hallucination detection results
_hallucinations_flags_cache = None  # Cache hallucination flags for grid updates


def render_analytics_tab():
    """Render analytics tab with cached visualizations"""
    global _analytics_cache
    
    # Return cached version if available
    if _analytics_cache is not None:
        logger.info("📊 Serving analytics from cache (instant)")
        return _analytics_cache
    
    logger.info("📊 Computing analytics visualizations (first time only)...")
    start = time.time()
    
    try:
        # Load all data once
        df = load_pixeltable_data_paginated(limit=None, offset=0)
        
        if df is None or df.empty:
            return html.Div("No data available for analytics")
        
        # Import the analytics tab creator from the full app
        try:
            from components.tabs.analytics import create_analytics_tab
            content = create_analytics_tab(df)
        except ImportError as e:
            logger.warning(f"Could not import full analytics tab: {e}")
            # Fallback to simple summary
            import dash_bootstrap_components as dbc
            content = dbc.Card([
                dbc.CardHeader("📊 Analytics Summary"),
                dbc.CardBody([
                    html.H5(f"Total Records: {len(df)}"),
                    html.P(f"Columns: {', '.join(df.columns[:10])}..."),
                    html.P("Full analytics tab coming soon!")
                ])
            ])
        
        elapsed = time.time() - start
        logger.info(f"✅ Analytics computed in {elapsed:.3f}s (cached for future use)")
        
        # Cache the result
        _analytics_cache = content
        return content
        
    except Exception as e:
        logger.error(f"Error creating analytics tab: {e}", exc_info=True)
        return html.Div(f"Error loading analytics: {str(e)}")


def render_models_tab():
    """Render models tab with cached content"""
    global _models_cache
    
    # Return cached version if available
    if _models_cache is not None:
        logger.info("🤖 Serving models tab from cache (instant)")
        return _models_cache
    
    logger.info("🤖 Computing models tab content (first time only)...")
    start = time.time()
    
    try:
        # Import the models tab creator
        from components.tabs.models import create_models_tab
        content = create_models_tab()
        
        elapsed = time.time() - start
        logger.info(f"✅ Models tab computed in {elapsed:.3f}s (cached for future use)")
        
        # Cache the result
        _models_cache = content
        return content
        
    except Exception as e:
        logger.error(f"Error creating models tab: {e}", exc_info=True)
        return html.Div(f"Error loading models tab: {str(e)}")


def render_audio_tab():
    """Render audio tab with cached content"""
    global _audio_cache
    
    # Return cached version if available
    if _audio_cache is not None:
        logger.info("🎵 Serving audio tab from cache (instant)")
        return _audio_cache
    
    logger.info("🎵 Computing audio tab content (first time only)...")
    start = time.time()
    
    try:
        # Load sample data
        df = load_pixeltable_data_paginated(limit=100, offset=0)
        
        if df is None or df.empty:
            return html.Div("No data available for audio playback")
        
        # Import the audio tab creator
        from components.tabs.audio import create_audio_tab
        content = create_audio_tab(df)
        
        elapsed = time.time() - start
        logger.info(f"✅ Audio tab computed in {elapsed:.3f}s (cached for future use)")
        
        # Cache the result
        _audio_cache = content
        return content
        
    except Exception as e:
        logger.error(f"Error creating audio tab: {e}", exc_info=True)
        return html.Div(f"Error loading audio tab: {str(e)}")


def get_compare_data():
    """Load and cache full dataset for compare tab"""
    global _compare_data_cache
    
    if _compare_data_cache is not None:
        logger.info("🔍 Using cached data for compare tab")
        return _compare_data_cache
    
    logger.info("🔍 Loading data for compare tab (first time only)...")
    start = time.time()
    
    df = load_pixeltable_data_paginated(limit=None, offset=0)
    
    if df is not None and not df.empty:
        _compare_data_cache = df
        elapsed = time.time() - start
        logger.info(f"✅ Compare data loaded in {elapsed:.3f}s (cached for future use)")
    
    return df


def render_compare_tab(sample_id=None, wer_method='basic'):
    """Render compare tab with sample comparison"""
    try:
        df = get_compare_data()
        
        if df is None or df.empty:
            return html.Div("No data available for comparison")
        
        # Import the compare tab creator
        from components.tabs.compare import create_compare_tab
        content = create_compare_tab(df, sample_id=sample_id, wer_method=wer_method)
        
        return content
        
    except Exception as e:
        logger.error(f"Error creating compare tab: {e}", exc_info=True)
        return html.Div(f"Error loading compare tab: {str(e)}")


def get_hallucinations_data():
    """Load and cache hallucination detection results"""
    global _hallucinations_data_cache
    
    if _hallucinations_data_cache is not None:
        logger.info("⚠️ Using cached hallucinations data")
        return _hallucinations_data_cache
    
    logger.info("⚠️ Computing hallucinations (first time only)...")
    start = time.time()
    
    df = load_pixeltable_data_paginated(limit=None, offset=0)
    
    if df is not None and not df.empty:
        _hallucinations_data_cache = df
        elapsed = time.time() - start
        logger.info(f"✅ Hallucinations data loaded in {elapsed:.3f}s (cached for future use)")
    
    return df


def render_hallucinations_tab():
    """Render hallucinations tab and cache flags"""
    global _hallucinations_flags_cache
    
    try:
        df = get_hallucinations_data()
        
        if df is None or df.empty:
            return html.Div("No data available for hallucination detection")
        
        # Import the hallucinations tab creator
        from components.tabs.hallucinations import create_hallucinations_tab
        content, flags = create_hallucinations_tab(df)
        
        # Cache the flags for later use
        if flags is not None:
            _hallucinations_flags_cache = flags
        
        return content
        
    except Exception as e:
        logger.error(f"Error creating hallucinations tab: {e}", exc_info=True)
        return html.Div(f"Error loading hallucinations tab: {str(e)}")


def register_simple_callbacks(app):
    """Register minimal callbacks for data table"""
    
    @app.callback(
        Output("tab-content-simple", "children"),
        [Input("main-tabs-simple", "active_tab"),
         Input("wer-method-store", "data"),
         Input("compare-sample-id-store", "data")],
        prevent_initial_call=False
    )
    def render_tab_content(active_tab, wer_method, sample_id):
        """Render content based on active tab"""
        if active_tab == "table-tab":
            # Import here to avoid circular dependency
            from app_simple import create_table_tab_content
            return create_table_tab_content()
        elif active_tab == "analytics-tab":
            return render_analytics_tab()
        elif active_tab == "compare-tab":
            return render_compare_tab(sample_id=sample_id, wer_method=wer_method or 'basic')
        elif active_tab == "hallucinations-tab":
            return render_hallucinations_tab()
        elif active_tab == "models-tab":
            return render_models_tab()
        elif active_tab == "audio-tab":
            return render_audio_tab()
        else:
            return html.Div("Select a tab")
    
    # Compare tab interactive callbacks
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
        [Input("main-tabs-simple", "active_tab")],
        [State("compare-sample-id-store", "data")],
        prevent_initial_call=False
    )
    def handle_tab_change_for_sample_id(active_tab, current_id):
        """Handle sample ID when switching to compare tab"""
        # When switching to compare tab, if no ID is set, generate one
        if active_tab == "compare-tab" and current_id is None:
            try:
                df = get_compare_data()
                if df is not None and not df.empty:
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
            df = get_compare_data()
            if df is not None and not df.empty:
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
    
    # Hallucinations tab callback for grid updates
    @app.callback(
        [Output("hallucinations-grid", "rowData"),
         Output("hallucinations-grid", "columnDefs")],
        [Input("main-tabs-simple", "active_tab")],
        prevent_initial_call=False
    )
    def update_hallucinations_table(active_tab):
        """Update hallucinations table when tab is activated"""
        if active_tab != "hallucinations-tab":
            raise PreventUpdate
        
        global _hallucinations_flags_cache
        
        if _hallucinations_flags_cache is None:
            return [], []
        
        try:
            # Get the cached data
            df = get_hallucinations_data()
            
            if df is None or df.empty:
                return [], []
            
            # Process data using cached flags
            from utils import extract_text_from_transcription
            import pandas as pd
            
            df_processed = df.copy()
            whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
            for col in whisper_cols:
                df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
            
            # Flatten data: create one row per hallucination
            hallucination_data = []
            
            for idx, row in df_processed.iterrows():
                row_id = str(row.get('id', ''))
                
                # Only process rows that have hallucinations
                if row_id not in _hallucinations_flags_cache:
                    continue
                
                row_flags = _hallucinations_flags_cache[row_id]
                
                # Create one row per model that has a hallucination
                for model_name, flags in row_flags.items():
                    record = {
                        'id': row_id,
                        'split': row.get('split', ''),
                        'transcription': row.get('transcription', ''),
                        'model': model_name,
                        'model_transcription': flags.get('hypothesis', ''),
                        'has_hallucination': True,
                        'info': flags.get('info', '')
                    }
                    hallucination_data.append(record)
            
            df_hall = pd.DataFrame(hallucination_data)
            
            if df_hall.empty:
                return [], []
            
            # Build column definitions
            columnDefs = [
                {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
                {"field": "split", "headerName": "Split", "width": 100},
                {"field": "model", "headerName": "Model", "width": 150, "filter": True},
                {"field": "transcription", "headerName": "Reference", "width": 300, "wrapText": True},
                {"field": "model_transcription", "headerName": "Model Output", "width": 350, "wrapText": True},
                {"field": "info", "headerName": "Details", "width": 500, "wrapText": True}
            ]
            
            logger.info(f"✅ Hallucinations table: {len(df_hall)} hallucination instances found")
            return df_hall.to_dict('records'), columnDefs
            
        except Exception as e:
            logger.error(f"Error updating hallucinations table: {e}", exc_info=True)
            return [], []
    
    @app.callback(
        Output("data-grid-simple", "rowData"),
        [Input("split-filter-simple", "value"),
         Input("search-input-simple", "value")],
        prevent_initial_call=False
    )
    def load_data(split_filter, search_term):
        """Load all data once (client-side row model for fast operations)"""
        start = time.time()
        
        try:
            # Build filters
            filters = {}
            if split_filter:
                filters['split'] = split_filter
            if search_term and search_term.strip():
                filters['search_term'] = search_term.strip()
            
            # Load all data (with filters if provided)
            df = load_pixeltable_data_paginated(
                limit=None,  # Load all records
                offset=0,
                filters=filters if filters else None
            )
            
            # Convert to records
            if df is not None and not df.empty:
                # Convert any dict columns to strings for AG Grid
                for col in df.columns:
                    if df[col].dtype == 'object':
                        # Check if first non-null value is a dict
                        sample = df[col].dropna()
                        if len(sample) > 0 and isinstance(sample.iloc[0], dict):
                            # Extract 'text' key if it exists, otherwise convert to string
                            df[col] = df[col].apply(lambda x: x.get('text', str(x)) if isinstance(x, dict) and x else str(x) if x else '')
                
                rows = df.to_dict('records')
            else:
                rows = []
            
            elapsed = time.time() - start
            logger.info(f"✅ Loaded {len(rows)} records in {elapsed:.3f}s (filter={split_filter}, search={bool(search_term)})")
            
            return rows
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
            return []
