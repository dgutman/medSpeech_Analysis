"""
Callbacks for the compare tab.
"""

from dash import Input, Output, State
import dash.dependencies
from dash.exceptions import PreventUpdate
import json
import logging

from data_loader import get_data
from config import app

logger = logging.getLogger(__name__)


def register_compare_callbacks():
    """Register all compare tab callbacks"""
    
    @app.callback(
        Output("wer-method-store", "data"),
        [Input({"type": "wer-method-btn", "index": dash.dependencies.ALL}, "n_clicks")],
        prevent_initial_call=True
    )
    def update_wer_method_store(n_clicks_list):
        """Update the WER method store when pill button is clicked"""
        from dash import callback_context
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



