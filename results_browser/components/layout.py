"""
Main layout component for the results browser.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
from components.tabs.data_table import create_data_table


def create_layout():
    """Create the main application layout"""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🎤 Medical Speech Analysis Results Browser", 
                       className="text-center mb-2 text-primary")
            ])
        ]),
        
        # Main Content Tabs
        dbc.Row([
            dbc.Col([
                dbc.Tabs([
                    dbc.Tab(label="📋 Data Table", tab_id="table-tab"),
                    dbc.Tab(label="📈 Analytics", tab_id="analytics-tab"),
                    dbc.Tab(label="🔄 Tiny Rep Compare", tab_id="tiny-rep-compare-tab"),
                    dbc.Tab(label="🔍 Compare", tab_id="compare-tab"),
                    dbc.Tab(label="⚠️ Hallucinations", tab_id="hallucinations-tab"),
                    dbc.Tab(label="🎵 Audio Player", tab_id="audio-tab"),
                    dbc.Tab(label="🤖 Models", tab_id="models-tab"),
                    dbc.Tab(label="🎯 Fine Tuning", tab_id="fine-tuning-tab"),
                ], id="main-tabs", active_tab="table-tab")
            ])
        ]),

        # Filters (above table; shown when Data Table tab is active)
        dbc.Row([
            dbc.Col([
                dcc.Dropdown(
                    id="split-filter",
                    placeholder="Filter by split...",
                    clearable=True,
                    value=None,
                    options=[],
                    style={"display": "none"}
                )
            ], width=6, className="mb-2"),
            dbc.Col([
                dbc.Input(
                    id="search-input",
                    placeholder="Search transcriptions...",
                    type="text",
                    value="",
                    style={"display": "none"}
                )
            ], width=6, className="mb-2")
        ], className="mt-3", id="filter-row", style={"display": "none"}),

        # Tab Content (pre-populate with table tab since it's the default active tab)
        dbc.Row([
            dbc.Col([
                html.Div(id="tab-content", children=create_data_table())
            ])
        ], className="mt-4"),

        # Hidden stores for compare tab state
        dcc.Store(id="wer-method-store", data="basic"),
        dcc.Store(id="compare-sample-id-store", data=None),
        
        # Lightweight store for hallucination flags (just row IDs and flags, not full data)
        dcc.Store(id="hallucinations-flags-store", data=None),
        
        # Store for dataset stats (always exists for callbacks)
        dcc.Store(id="dataset-stats-store", data=None),

        # Pagination store (must be in main layout, not in tab content)
        dcc.Store(id="pagination-store", data={"page": 0, "pageSize": 50, "totalRows": 0, "loadedPages": []}),
        # Full list of row IDs for current filter (local pager: no offset in Pixeltable)
        dcc.Store(id="table-row-ids-store", data=None),

        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Medical Speech Analysis Results Browser", 
                       className="text-center text-muted")
            ])
        ], className="mt-5")
    ], fluid=True)



