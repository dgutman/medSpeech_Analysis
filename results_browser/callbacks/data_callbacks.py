"""
Callbacks for data loading and statistics.
"""

from dash import Input, Output, no_update
from dash.exceptions import PreventUpdate
import logging

from data_loader import get_total_count, get_split_counts
from config import app

logger = logging.getLogger(__name__)


def register_data_callbacks():
    """Register all data-related callbacks"""
    
    @app.callback(
        [Output("dataset-stats-store", "data"),
         Output("split-filter", "options")],
        [Input("main-tabs", "active_tab")],
        prevent_initial_call=False
    )
    def update_dataset_stats(active_tab):
        """Update dataset statistics and filter options"""
        try:
            # Get total count without loading all data
            total_count = get_total_count()
            logger.info(f"Total count: {total_count}")
            
            # Get split counts from database (counts all rows, not just a sample)
            try:
                splits = get_split_counts()
                logger.info(f"Split counts received in callback: {splits}")
                # Get unique split values for filter dropdown
                split_options = [{"label": split, "value": split} for split in sorted(splits.keys())]
            except Exception as e:
                logger.error(f"Error getting split counts: {e}", exc_info=True)
                splits = {}
                split_options = []
            
            from dash import html
            train_count = splits.get('train', 0)
            test_count = splits.get('test', 0)
            validation_count = splits.get('validation', 0)
            logger.info(f"Displaying counts - Train: {train_count}, Test: {test_count}, Validation: {validation_count}")
            
            stats_children = [
                html.H5(
                    f"Total Records: {total_count:,}" + 
                    (f" | Train: {train_count} | Test: {test_count} | Validation: {validation_count}" if splits else ""),
                    className="mb-0 mt-0"
                ),
                html.P("Data loaded in pages for better performance", className="mb-0 text-muted small")
            ]
            
            # Filter options - only update if table tab is active (component exists)
            if active_tab != "table-tab":
                split_options = no_update
            
            return stats_children, split_options
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}", exc_info=True)
            # Return empty stats and no_update for split-filter if component doesn't exist
            from dash import html
            if active_tab == "table-tab":
                return [html.P("Error loading data")], []
            else:
                return [html.P("Error loading data")], no_update
    
    @app.callback(
        Output("dataset-stats", "children"),
        [Input("dataset-stats-store", "data"),
         Input("main-tabs", "active_tab")],
        prevent_initial_call=False
    )
    def display_dataset_stats(stats_data, active_tab):
        """Display dataset stats in the table tab"""
        # Only show stats when table tab is active and data exists
        if active_tab == "table-tab" and stats_data:
            return stats_data
        return []

