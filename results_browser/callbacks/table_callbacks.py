"""
Callbacks for the data table tab.
"""

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import logging

from data_loader import load_pixeltable_data_paginated, get_total_count
from utils import extract_text_from_transcription
from config import app

logger = logging.getLogger(__name__)


def register_table_callbacks():
    """Register all table-related callbacks"""
    
    @app.callback(
        [Output("data-grid", "columnDefs"),
         Output("data-grid", "dashGridOptions"),
         Output("split-filter", "style"),
         Output("search-input", "style"),
         Output("filter-row", "style")],
        [Input("main-tabs", "active_tab"),
         Input("split-filter", "value"),
         Input("search-input", "value"),
         Input("dataset-stats-store", "data")],  # Trigger when stats are loaded
        prevent_initial_call=False  # Run initially to configure the grid
    )
    def update_data_grid_config(active_tab, split_filter, search_term, dataset_stats_data):
        """Update grid configuration and column definitions"""
        # Show/hide filters based on active tab
        is_table_tab = active_tab == "table-tab"
        filter_style = {"display": "block", "width": "100%"} if is_table_tab else {"display": "none"}
        filter_row_style = {"display": "flex"} if is_table_tab else {"display": "none"}
        
        # Default grid options (always return infinite model to avoid conflicts)
        default_grid_options = {
            "pagination": True,
            "paginationPageSize": 50,
            "paginationAutoPageSize": False,
            "suppressRowClickSelection": False,
            "rowSelection": "single",
            "rowModelType": "infinite",
            "cacheBlockSize": 50,
            "maxBlocksInCache": 10,
            "infiniteInitialRowCount": 0,
        }
        
        # Only update when on table tab
        if not is_table_tab:
            return [], default_grid_options, filter_style, filter_style, filter_row_style
        
        try:
            # Get total count for infinite row model
            total_count = get_total_count()
            
            # Get column definitions from a sample row
            sample_df = load_pixeltable_data_paginated(limit=1, offset=0)
            
            if sample_df is None or sample_df.empty:
                # Fallback: use minimal default columns if no data available
                columnDefs = [
                    {"field": "id", "headerName": "ID", "width": 150, "pinned": "left"},
                ]
            else:
                # Dynamically discover all columns from the data
                all_columns = list(sample_df.columns)
                
                # Try to import MODEL_COLUMNS from db_helpers for reference ordering
                try:
                    import sys
                    import os
                    # Add parent directory to path to import db_helpers
                    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    if parent_dir not in sys.path:
                        sys.path.insert(0, parent_dir)
                    from db_helpers import MODEL_COLUMNS
                    
                    # Create a priority order: standard columns first, then MODEL_COLUMNS, then others
                    priority_columns = ['id', 'split', 'transcription']
                    # Add model columns that exist in the data (handle variations like _transcription suffix)
                    model_cols_in_data = []
                    matched_columns = set()  # Track which columns we've already matched
                    
                    for model_col in MODEL_COLUMNS:
                        # Check for exact match first
                        if model_col in all_columns:
                            model_cols_in_data.append(model_col)
                            matched_columns.add(model_col)
                        else:
                            # Check for variations (e.g., whisper_tiny_transcription, whisper_tinyEn_transcription)
                            # Look for columns that start with the model column name
                            matching_cols = [col for col in all_columns 
                                          if col not in matched_columns 
                                          and (col.startswith(model_col + '_') or col == model_col)]
                            if matching_cols:
                                # Take the first match (prefer exact or shortest match)
                                matching_cols.sort(key=lambda x: (x != model_col, len(x)))
                                matched_col = matching_cols[0]
                                model_cols_in_data.append(matched_col)
                                matched_columns.add(matched_col)
                    
                    # Add other whisper columns not in MODEL_COLUMNS
                    other_whisper_cols = [col for col in all_columns 
                                        if 'whisper' in col.lower() 
                                        and col not in matched_columns 
                                        and col not in priority_columns]
                    # Add all other columns
                    other_cols = [col for col in all_columns 
                                 if col not in priority_columns 
                                 and col not in matched_columns 
                                 and col not in other_whisper_cols]
                    
                    # Build ordered column list
                    ordered_columns = []
                    for col in priority_columns:
                        if col in all_columns:
                            ordered_columns.append(col)
                    ordered_columns.extend(model_cols_in_data)
                    ordered_columns.extend(other_whisper_cols)
                    ordered_columns.extend(other_cols)
                except ImportError:
                    # If db_helpers not available, just use all columns in their natural order
                    ordered_columns = all_columns
                
                # Build column definitions dynamically
                columnDefs = []
                for col in ordered_columns:
                    col_def = {
                        "field": col,
                        "headerName": col.replace('_', ' ').title(),
                        "width": 300 if 'whisper' in col.lower() or col == 'transcription' else 150,
                        "wrapText": True if 'whisper' in col.lower() or col == 'transcription' else False
                    }
                    
                    # Pin id column to left
                    if col == 'id':
                        col_def["pinned"] = "left"
                        col_def["width"] = 150
                    
                    # Set split column width
                    if col == 'split':
                        col_def["width"] = 100
                    
                    columnDefs.append(col_def)
            
            # Update grid options with total count for infinite row model
            grid_options = {
                "pagination": True,
                "paginationPageSize": 50,
                "paginationAutoPageSize": False,
                "suppressRowClickSelection": False,
                "rowSelection": "single",
                "rowModelType": "infinite",
                "cacheBlockSize": 50,
                "maxBlocksInCache": 10,
                "infiniteInitialRowCount": total_count,
            }
            
            return columnDefs, grid_options, filter_style, filter_style, filter_row_style
            
        except Exception as e:
            logger.error(f"Error updating data grid config: {e}", exc_info=True)
            empty_style = {"display": "block", "width": "100%"} if active_tab == "table-tab" else {"display": "none"}
            empty_row_style = {"display": "flex"} if active_tab == "table-tab" else {"display": "none"}
            return [], default_grid_options, empty_style, empty_style, empty_row_style
    
    @app.callback(
        Output("data-grid", "getRowsResponse"),
        [Input("data-grid", "getRowsRequest"),
         Input("split-filter", "value"),
         Input("search-input", "value"),
         Input("data-grid", "dashGridOptions")],  # Trigger when grid options are set
        prevent_initial_call=False  # Allow initial call to handle first load
    )
    def get_rows(get_rows_request, split_filter, search_term, grid_options):
        """Server-side pagination: Load data for the requested page using infinite row model"""
        # Check what triggered this callback
        ctx = callback_context
        triggered_id = None
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id']
        
        # If no request from AG Grid yet, but grid options were just set, create initial request
        if get_rows_request is None:
            if triggered_id and 'dashGridOptions' in triggered_id:
                logger.info("Grid options set, creating initial request for rows 0-50")
                get_rows_request = {"startRow": 0, "endRow": 50}
            elif triggered_id:
                logger.info(f"getRowsRequest is None, but callback triggered by: {triggered_id}")
                # If triggered by filter changes, create request
                if 'split-filter' in triggered_id or 'search-input' in triggered_id:
                    logger.info("Filter changed, creating request for rows 0-50")
                    get_rows_request = {"startRow": 0, "endRow": 50}
                else:
                    raise PreventUpdate
            else:
                logger.debug("getRowsRequest is None, no triggers")
                raise PreventUpdate
        
        try:
            # Extract pagination info from AG Grid infinite row model request
            start_row = get_rows_request.get("startRow", 0)
            end_row = get_rows_request.get("endRow", 50)
            page_size = end_row - start_row
            
            logger.info(f"Loading rows {start_row} to {end_row} (page_size={page_size})")
            
            # Build filters for database query
            filters = {}
            if split_filter:
                filters['split'] = split_filter
            if search_term:
                filters['search_term'] = search_term
            
            # Get total count with filters applied
            total_count = get_total_count(filters=filters)
            
            # Load ONLY the requested page with filters applied at database level
            df = load_pixeltable_data_paginated(limit=page_size, offset=start_row, filters=filters)
            
            if df is None or df.empty:
                return {"rowData": [], "rowCount": total_count}
            
            # Process whisper columns to extract text
            whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
            for col in whisper_cols:
                df[col] = df[col].apply(extract_text_from_transcription)
            
            # Convert to records for AG Grid
            row_data = df.to_dict('records')
            
            logger.info(f"Returning {len(row_data)} rows (total: {total_count})")
            
            # Return response in AG Grid infinite row model format
            # rowCount should be the total number of rows (not just this page)
            # AG Grid uses this to determine pagination and if more data is available
            return {
                "rowData": row_data,
                "rowCount": total_count  # Total count for pagination
            }
            
        except Exception as e:
            logger.error(f"Error loading rows: {e}", exc_info=True)
            return {"rowData": [], "rowCount": 0}

