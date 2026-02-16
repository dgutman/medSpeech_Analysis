"""
Hallucinations tab component for detecting and displaying hallucination instances.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
import pandas as pd
import logging

from utils import extract_text_from_transcription
from wer_utils import detect_hallucinations
from viz_cache import get_cached_visualization, cache_visualization

logger = logging.getLogger(__name__)


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
                    
                    row_model_flags[model_name] = {
                        'has_hallucination': True,
                        'repetition': hall_result['repetition']['detected'],
                        'length_anomaly': hall_result['length_anomaly']['detected'],
                        'char_repetition': hall_result['char_repetition']['detected'],
                        'insertions': hall_result['insertions']['detected'],
                        'stuttering': hall_result['stuttering']['detected'],
                        'info': '; '.join(all_issues) if all_issues else '',
                        'hypothesis': hypothesis  # Store the model transcription
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
        
        # Try to get cached stats table
        cached_stats_table = get_cached_visualization("hallucinations", df_processed, viz_type="stats_table")
        
        if cached_stats_table is not None:
            stats_table = cached_stats_table
            logger.info("Using cached hallucination stats table")
        else:
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
            margin=dict(l=20, r=20, t=50, b=5)
        )
        cache_visualization("hallucinations", df_processed, stats_table, viz_type="stats_table")
        
        # Initial table data and columns (will be updated by callback)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=stats_table, config={'displayModeBar': False})
                ], width=12)
            ], className="mb-1"),
            dbc.Row([
                dbc.Col([
                    html.P("Only messages with hallucinations are shown. Each row represents one hallucination instance. Use the Model column filter to filter by specific models.", 
                           className="text-muted small mb-1")
                ], width=12)
            ]),
            dcc.Loading(
                id="hallucinations-table-loading",
                type="default",
                children=dag.AgGrid(
                    id="hallucinations-grid",
                    columnDefs=[],  # Empty initially, populated by callback
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



