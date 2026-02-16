"""
Analytics tab component for comprehensive WER analysis.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import logging

from utils import extract_text_from_transcription
from wer_utils import calculate_wer
from viz_cache import get_cached_visualization, cache_visualization, get_data_hash

logger = logging.getLogger(__name__)


def create_analytics_tab(df):
    """Create the analytics tab with comprehensive WER analysis"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return html.Div("No data available")
    
    try:
        # Process the dataframe to extract text from transcription columns
        df_processed = df.copy()
        
        # Process whisper columns to extract text
        whisper_cols = [col for col in df.columns if 'whisper' in col.lower()]
        for col in whisper_cols:
            df_processed[col] = df_processed[col].apply(extract_text_from_transcription)
        
        # Calculate WER for each model
        model_wer_cols = {}
        for col in whisper_cols:
            wer_col = col.replace('_transcription', '_wer')
            df_processed[wer_col] = df_processed.apply(
                lambda row: calculate_wer(row[col], row['transcription']), axis=1
            )
            model_wer_cols[col] = wer_col
        
        # Dynamically discover whisper columns from the data
        # Try to use MODEL_COLUMNS from db_helpers for ordering if available
        try:
            import sys
            import os
            # Add parent directory to path to import db_helpers
            parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            from db_helpers import MODEL_COLUMNS
            
            # Find model columns that exist in the data (may have _transcription suffix or other variations)
            model_cols_in_data = []
            for model_col in MODEL_COLUMNS:
                # Check for exact match or with _transcription suffix
                if model_col in df_processed.columns:
                    model_cols_in_data.append(model_col)
                elif f"{model_col}_transcription" in df_processed.columns:
                    model_cols_in_data.append(f"{model_col}_transcription")
            
            # Add any other whisper columns not in MODEL_COLUMNS
            # IMPORTANT: df_processed also contains computed columns like *_wer; don't treat those as models.
            all_whisper_cols = [
                col for col in df_processed.columns
                if 'whisper' in col.lower()
                and not col.lower().endswith('_wer')
                and '_wer' not in col.lower()
            ]
            other_whisper_cols = [col for col in all_whisper_cols if col not in model_cols_in_data]
            
            # Order: MODEL_COLUMNS first (in their order), then other whisper columns
            available_models = model_cols_in_data + other_whisper_cols
        except ImportError:
            # If db_helpers not available, just discover all whisper columns dynamically
            available_models = [col for col in df_processed.columns if 'whisper' in col.lower()]
        
        if not available_models:
            return html.Div("No Whisper model data available for analysis")
        
        # Prepare data for visualization
        wer_data = []
        for model in available_models:
            wer_col = model_wer_cols[model]
            model_name = model.replace('whisper_', '').replace('_transcription', '').replace('_', ' ').title()
            
            for _, row in df_processed.iterrows():
                if not pd.isna(row[wer_col]):
                    # Ensure ID is a string and handle NaN/None values
                    row_id = str(row.get('id', 'unknown'))
                    if row_id == 'nan' or row_id == 'None' or pd.isna(row.get('id')):
                        row_id = f'unknown_{len(wer_data)}'  # Create unique ID for missing values
                    
                    wer_data.append({
                        'Model': model_name,
                        'WER': row[wer_col] * 100,  # Convert to percentage
                        'ID': row_id
                    })
        
        wer_df = pd.DataFrame(wer_data)
        
        if wer_df.empty:
            return html.Div("No valid WER data available for analysis")
        
        # Ensure ID column is string type
        wer_df['ID'] = wer_df['ID'].astype(str)
        
        # Remove any duplicate ID+Model combinations (keep first occurrence)
        # This is a safeguard in case duplicates somehow got through
        initial_count = len(wer_df)
        wer_df = wer_df.drop_duplicates(subset=['ID', 'Model'], keep='first')
        if len(wer_df) != initial_count:
            logger.warning(f"Removed {initial_count - len(wer_df)} duplicate ID+Model combinations in analytics tab")
        
        # Verify no duplicates remain
        duplicates = wer_df.duplicated(subset=['ID', 'Model']).sum()
        if duplicates > 0:
            logger.error(f"ERROR: Still have {duplicates} duplicate ID+Model combinations after deduplication!")
            # Force remove by resetting index and using drop_duplicates again
            wer_df = wer_df.reset_index(drop=True).drop_duplicates(subset=['ID', 'Model'], keep='first')
        
        # Reset index to ensure clean integer index (plotly can have issues with duplicate indices)
        wer_df = wer_df.reset_index(drop=True)
        
        # Try to get cached visualizations
        cached_fig_violin = get_cached_visualization("analytics", wer_df, viz_type="violin")
        cached_table_fig = get_cached_visualization("analytics", wer_df, viz_type="table")

        # Calculate mean WER for each model (for annotations)
        mean_wer_by_model = wer_df.groupby('Model')['WER'].mean()
        y_max = min(120, wer_df['WER'].max() + 10)
        annotation_y_position = y_max * 0.95  # Position annotations near the top of the plot

        # Violin plot for WER distribution
        if cached_fig_violin is not None:
            fig_violin = cached_fig_violin
            logger.info("Using cached violin plot")
        else:
            fig_violin = px.violin(
            wer_df,
            x='Model',
            y='WER',
            title="WER Distribution Across Whisper Models",
            color='Model',
            labels={"WER": "Word Error Rate (%)", "Model": "Whisper Model"},
            template="plotly_white",
            box=True,
            points="outliers"
        )
        fig_violin.update_layout(
            title_font_size=16,
            yaxis=dict(range=[0, y_max]),
            xaxis_title="",
            yaxis_title="Word Error Rate (%)",
            font=dict(size=12),
            showlegend=False
        )
        
        # Add mean WER annotations to violin plot
        annotations_violin = []
        for model in mean_wer_by_model.index:
            mean_value = mean_wer_by_model[model]
            annotations_violin.append(
                dict(
                    x=model,
                    y=annotation_y_position,
                    text=f"μ={mean_value:.1f}%",
                    showarrow=False,
                    font=dict(size=11, color='#2c3e50', family='Arial Black'),
                    bgcolor='rgba(255, 255, 255, 0.85)',
                    bordercolor='#2c3e50',
                    borderwidth=1.5,
                    borderpad=4
                )
            )
            fig_violin.update_layout(annotations=annotations_violin)
            cache_visualization("analytics", wer_df, fig_violin, viz_type="violin")
        
        # 3. Summary statistics table with improved formatting
        summary_stats = wer_df.groupby('Model')['WER'].agg([
            'count', 'mean', 'std', 'min', 'max', 'median'
        ])
        summary_stats.columns = ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median']
        
        # Sort by Mean WER (best to worst)
        summary_stats = summary_stats.sort_values('Mean')
        
        # Format values appropriately
        formatted_data = {
            'Model': summary_stats.index.tolist(),
            'Count': [f"{int(count)}" for count in summary_stats['Count']],
            'Mean WER (%)': [f"{mean:.1f}" for mean in summary_stats['Mean']],
            'Std Dev': [f"{std:.1f}" for std in summary_stats['Std Dev']],
            'Min': [f"{min_val:.1f}" for min_val in summary_stats['Min']],
            'Max': [f"{max_val:.1f}" for max_val in summary_stats['Max']],
            'Median': [f"{median:.1f}" for median in summary_stats['Median']]
        }
        
        # Color code cells based on Mean WER (green=good, yellow=medium, red=bad)
        def get_wer_color(wer_value):
            """Return color based on WER value"""
            if wer_value < 15:
                return '#d4edda'  # Light green - excellent
            elif wer_value < 25:
                return '#fff3cd'  # Light yellow - good
            elif wer_value < 35:
                return '#ffeaa7'  # Yellow - moderate
            else:
                return '#f8d7da'  # Light red - poor
        
        # Prepare cell colors for Mean WER column
        mean_colors = [get_wer_color(mean) for mean in summary_stats['Mean']]
        
        if cached_table_fig is not None:
            table_fig = cached_table_fig
            logger.info("Using cached summary table")
        else:
            table_fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Model', 'Count', 'Mean WER (%)', 'Std Dev', 'Min', 'Max', 'Median'],
                fill_color='#2c3e50',
                align='left',
                font=dict(size=13, color='white', family='Arial'),
                height=40
            ),
            cells=dict(
                values=[
                    formatted_data['Model'],
                    formatted_data['Count'],
                    formatted_data['Mean WER (%)'],
                    formatted_data['Std Dev'],
                    formatted_data['Min'],
                    formatted_data['Max'],
                    formatted_data['Median']
                ],
                fill_color=[
                    ['white'] * len(summary_stats),  # Model column
                    ['#f8f9fa'] * len(summary_stats),  # Count column
                    mean_colors,  # Mean WER - color coded
                    ['white'] * len(summary_stats),  # Std Dev
                    ['#f8f9fa'] * len(summary_stats),  # Min
                    ['white'] * len(summary_stats),  # Max
                    ['#f8f9fa'] * len(summary_stats)  # Median
                ],
                align=['left', 'center', 'center', 'center', 'center', 'center', 'center'],
                font=dict(size=12, color='black', family='Arial'),
                height=35
            )
        )])
            table_fig.update_layout(
                title=dict(
                    text="📊 WER Summary Statistics (sorted by Mean WER)",
                    font=dict(size=16, family='Arial', color='#2c3e50'),
                    x=0.5
                ),
                height=350,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            cache_visualization("analytics", wer_df, table_fig, viz_type="table")

        return dbc.Container([
            dbc.Row([
                dbc.Col([dcc.Graph(figure=fig_violin)], width=12)
            ], className="mb-4"),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=table_fig)], width=12)
            ])
        ])
        
    except Exception as e:
        import traceback
        logger.error(f"Error creating analytics tab: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return html.Div(f"Error creating analytics: {str(e)}")



