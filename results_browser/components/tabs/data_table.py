"""
Data table tab component.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_ag_grid as dag


def create_data_table():
    """Create the data table tab with pagination support"""
    
    # Control Panel - Dataset Overview
    # Stats content comes from dataset-stats-store (updated by callback)
    control_panel = dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Dataset Overview", className="py-2"),
                dbc.CardBody([
                    html.Div(id="dataset-stats", className="py-1")
                ], className="py-2")
            ], className="mb-3")
        ], width=12)
    ], className="mb-3")
    
    # Note: Filter components are in the main layout (above tab-content) for callbacks to work
    # They are shown/hidden via the update_data_grid callback
    
    # Get column definitions (will be determined dynamically from data by the callback)
    # Initial empty set - will be populated by update_data_grid_config callback
    columnDefs = []
    
    return html.Div([
        control_panel,
        dcc.Loading(
            id="data-table-loading",
            type="default",
            children=dag.AgGrid(
                id="data-grid",
                columnDefs=columnDefs,
                # Don't set rowData with infinite row model - data comes via getRowsResponse
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                    "floatingFilter": False
                },
                dashGridOptions={
                    "pagination": True,
                    "paginationPageSize": 50,  # Show 50 rows per page
                    "paginationAutoPageSize": False,
                    "suppressRowClickSelection": False,
                    "rowSelection": "single",
                    "rowModelType": "infinite",  # Use infinite row model for server-side pagination
                    "cacheBlockSize": 50,  # Load 50 rows per block (same as page size)
                    "maxBlocksInCache": 10,  # Keep up to 10 blocks (500 rows) in cache
                    "infiniteInitialRowCount": 6661,  # Will be updated by callback, but needs initial value
                },
                style={"height": "600px", "width": "100%"}
            )
        )
        # Note: pagination-store is in the main layout, not here
    ])

