"""
Callbacks for tab content updates.
"""

from dash import Input, Output, State, callback_context, html
from dash.exceptions import PreventUpdate
import logging

from data_loader import get_data
from components.tabs.data_table import create_data_table
from components.tabs.analytics import create_analytics_tab
from components.tabs.tiny_rep_compare import create_tiny_rep_compare_tab
from components.tabs.compare import create_compare_tab
from components.tabs.hallucinations import create_hallucinations_tab
from components.tabs.audio import create_audio_tab
from components.tabs.models import create_models_tab
from components.tabs.fine_tuning import create_fine_tuning_tab
from config import app

logger = logging.getLogger(__name__)


def register_tab_callbacks():
    """Register all tab-related callbacks"""
    
    @app.callback(
        [Output("tab-content", "children"),
         Output("hallucinations-flags-store", "data")],
        [Input("main-tabs", "active_tab"),
         Input("wer-method-store", "data"),
         Input("compare-sample-id-store", "data"),
         Input("split-filter", "value"),
         Input("search-input", "value")],
        prevent_initial_call=False
    )
    def update_tab_content(active_tab, wer_method, sample_id, split_filter, search_term):
        """Update content based on active tab and filters"""
        # Check which input triggered the callback
        ctx = callback_context
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
            # If filter inputs triggered but we're not on table tab, prevent update
            if triggered_id in ['split-filter', 'search-input'] and active_tab != "table-tab":
                raise PreventUpdate
        
        try:
            hallucination_flags = None
            
            # Default WER method if not specified
            if wer_method is None:
                wer_method = 'basic'
            
            # Only load full data for tabs that need it (not table tab - it uses pagination)
            if active_tab == "table-tab":
                # For table tab, use pagination - don't load full dataset
                content = create_data_table()
            elif active_tab == "models-tab":
                # Models tab doesn't need data either
                content = create_models_tab()
            elif active_tab == "fine-tuning-tab":
                # Fine-tuning tab doesn't need data either
                content = create_fine_tuning_tab()
            else:
                # Other tabs need the full dataset
                df = get_data()
                
                # Apply filters only if they exist
                if split_filter:
                    df = df[df['split'] == split_filter]
                
                if search_term:
                    df = df[df['transcription'].str.contains(search_term, case=False, na=False)]
                
                if active_tab == "analytics-tab":
                    content = create_analytics_tab(df)
                elif active_tab == "tiny-rep-compare-tab":
                    content = create_tiny_rep_compare_tab(df)
                elif active_tab == "compare-tab":
                    content = create_compare_tab(df, wer_method, sample_id)
                elif active_tab == "hallucinations-tab":
                    content, hallucination_flags = create_hallucinations_tab(df)
                elif active_tab == "audio-tab":
                    content = create_audio_tab(df)
                else:
                    content = html.Div("Select a tab")
            
            return content, hallucination_flags
                
        except Exception as e:
            logger.error(f"Error updating tab content: {e}")
            return html.Div(f"Error: {str(e)}"), None

