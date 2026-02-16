"""
Callbacks for the data table tab.
"""

from dash import Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import logging

from data_loader import load_paginated_index, load_rows_by_ids, get_total_count
from utils import extract_text_from_transcription
from config import app

logger = logging.getLogger(__name__)


def register_table_callbacks():
    """Register all table-related callbacks"""
    
    # 1) Filter visibility only - always in layout, safe to update
    @app.callback(
        [Output("split-filter", "style"),
         Output("search-input", "style"),
         Output("filter-row", "style")],
        [Input("main-tabs", "active_tab")],
        prevent_initial_call=False,
    )
    def update_filter_visibility(active_tab):
        is_table_tab = active_tab == "table-tab"
        style = {"display": "block", "width": "100%"} if is_table_tab else {"display": "none"}
        row_style = {"display": "flex"} if is_table_tab else {"display": "none"}
        return style, style, row_style
    
    # 2) Data grid config - only when table tab is active (data-grid exists)
    @app.callback(
        [Output("data-grid", "columnDefs"),
         Output("data-grid", "dashGridOptions")],
        [Input("main-tabs", "active_tab"),
         Input("split-filter", "value"),
         Input("search-input", "value"),
         Input("dataset-stats-store", "data")],
        prevent_initial_call=False,
    )
    def update_data_grid_config(active_tab, split_filter, search_term, dataset_stats_data):
        """Update grid configuration and column definitions. PreventUpdate when not on table tab so we never output to missing data-grid."""
        if active_tab != "table-tab":
            raise PreventUpdate
        
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
        
        try:
            filters = {}
            if split_filter:
                filters["split"] = split_filter
            if search_term:
                filters["search_term"] = search_term
            total_count = get_total_count(filters=filters)

            # Get column definitions from one sample row (no offset: use first id from index)
            ids = load_paginated_index(filters=filters)
            sample_df = load_rows_by_ids(ids[:1]) if ids else None

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
            
            return columnDefs, grid_options
            
        except Exception as e:
            logger.error(f"Error updating data grid config: {e}", exc_info=True)
            return [], default_grid_options

    # Populate the full row-ID list for the current filter (local pager: Pixeltable has no offset).
    @app.callback(
        Output("table-row-ids-store", "data"),
        [Input("main-tabs", "active_tab"),
         Input("split-filter", "value"),
         Input("search-input", "value")],
        prevent_initial_call=False,
    )
    def update_table_row_ids_store(active_tab, split_filter, search_term):
        """Load full ordered list of row IDs when on table tab or when filters change."""
        if active_tab != "table-tab":
            return None
        filters = {}
        if split_filter:
            filters["split"] = split_filter
        if search_term:
            filters["search_term"] = search_term
        ids = load_paginated_index(filters=filters)
        return ids

    @app.callback(
        Output("data-grid", "getRowsResponse"),
        [Input("data-grid", "getRowsRequest"),
         Input("split-filter", "value"),
         Input("search-input", "value"),
         Input("main-tabs", "active_tab")],
        [State("table-row-ids-store", "data")],
        prevent_initial_call=False,
    )
    def get_rows(get_rows_request, split_filter, search_term, active_tab, row_ids_store):
        """Server-side pagination via local pager: we have the full index in store, slice it and fetch only those rows by ID."""
        if active_tab != "table-tab":
            raise PreventUpdate

        ctx = callback_context
        triggered_id = ctx.triggered[0]["prop_id"] if ctx.triggered else None

        if get_rows_request is None:
            logger.info("getRowsRequest is None, synthesizing first block (0-50) for initial load")
            get_rows_request = {"startRow": 0, "endRow": 50}

        try:
            start_row = int(get_rows_request.get("startRow") or get_rows_request.get("start_row") or 0)
            end_row = int(get_rows_request.get("endRow") or get_rows_request.get("end_row") or 50)
            page_size = max(1, end_row - start_row)

            filters = {}
            if split_filter:
                filters["split"] = split_filter
            if search_term:
                filters["search_term"] = search_term

            # Use store if populated; otherwise load index on the fly (e.g. first run before store callback)
            if row_ids_store is not None and isinstance(row_ids_store, list):
                all_ids = row_ids_store
            else:
                all_ids = load_paginated_index(filters=filters)

            total_count = len(all_ids)
            ids_for_page = all_ids[start_row:end_row]

            logger.info(
                f"Loading page: startRow={start_row} endRow={end_row} (ids slice len={len(ids_for_page)}, total={total_count}) trigger={triggered_id!r}"
            )

            if not ids_for_page:
                return {"rowData": [], "rowCount": total_count}

            df = load_rows_by_ids(ids_for_page)
            if df is None or df.empty:
                return {"rowData": [], "rowCount": total_count}

            # Extract text for model/whisper/tiny_rep columns so we don't show [object Object]
            try:
                from db_helpers import MODEL_COLUMNS
                model_cols_in_df = [c for c in df.columns if c in MODEL_COLUMNS]
            except ImportError:
                model_cols_in_df = []
            transcription_cols = [
                c for c in df.columns
                if c in model_cols_in_df
                or "whisper" in c.lower()
                or "tiny_rep" in c.lower()
            ]
            for col in transcription_cols:
                df[col] = df[col].apply(extract_text_from_transcription)

            row_data = df.to_dict("records")
            logger.info(f"Returning {len(row_data)} rows (total: {total_count})")
            return {"rowData": row_data, "rowCount": total_count}

        except Exception as e:
            logger.error(f"Error loading rows: {e}", exc_info=True)
            return {"rowData": [], "rowCount": 0}

