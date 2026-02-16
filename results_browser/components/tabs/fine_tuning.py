"""
Fine-tuning summary tab component for displaying comprehensive fine-tuning results.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from typing import Dict, List
import logging

from training_stats_utils import get_model_training_stats, get_training_stats_summary

logger = logging.getLogger(__name__)


def create_fine_tuning_tab():
    """Create a comprehensive fine-tuning results summary tab"""
    
    # Get fine-tuning statistics
    # Try multiple possible paths (development vs Docker)
    possible_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'fine_tuning_data'),  # Development: results_browser/fine_tuning_data
        '/app/fine_tuning_data',  # Docker path
        os.path.join(os.path.dirname(__file__), '..', '..', 'Fine-Tuning-Whisper-on-Custom-Dataset'),  # Fallback: original location
    ]
    
    base_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            base_dir = path
            break
    
    if base_dir is None:
        logger.warning(f"Fine-tuning directory not found. Tried: {possible_paths}")
        return dbc.Container([
            dbc.Alert(
                "Fine-tuning data directory not found. Please ensure the fine-tuning-data directory is available.",
                color="warning"
            )
        ], fluid=True)
    
    training_stats = get_model_training_stats(base_dir)
    training_summary = get_training_stats_summary(base_dir)
    
    if not training_summary:
        return dbc.Container([
            dbc.Alert(
                "No fine-tuning statistics found. Please ensure training data is available in the Fine-Tuning-Whisper-on-Custom-Dataset directory.",
                color="warning"
            )
        ], fluid=True)
    
    # Create summary table
    summary_df = pd.DataFrame(training_summary)
    
    # Sort by improvement (best first)
    if '_improvement' in summary_df.columns:
        summary_df = summary_df.sort_values('_improvement', ascending=False)
    
    # Create comparison table figure with training duration
    comparison_table = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "Baseline WER", "Best WER", "WER Improvement", "Improvement %", "Training Duration", "Total Epochs", "Total Steps"],
            fill_color='#2c3e50',
            align='left',
            font=dict(size=11, color='white', family='Arial', weight='bold'),
            height=40
        ),
        cells=dict(
            values=[
                summary_df['Model'],
                summary_df['Baseline WER'],
                summary_df['Best WER'],
                summary_df['WER Improvement'],
                summary_df['Improvement %'],
                summary_df['Training Duration'],
                summary_df['Total Epochs'],
                summary_df['Total Steps']
            ],
            fill_color=[
                ['white'] * len(summary_df),
                ['#f8f9fa'] * len(summary_df),
                ['white'] * len(summary_df),
                ['#e8f5e9'] * len(summary_df),  # Green tint for improvements
                ['#c8e6c9'] * len(summary_df),  # Darker green for percentages
                ['#fff3cd'] * len(summary_df),  # Yellow tint for training duration
                ['white'] * len(summary_df),
                ['#f8f9fa'] * len(summary_df)
            ],
            align='left',
            font=dict(size=10, color='black', family='Arial'),
            height=35
        )
    )])
    
    comparison_table.update_layout(
        title=dict(
            text="🎯 Fine-Tuning Results Summary",
            font=dict(size=18, family='Arial', color='#2c3e50'),
            x=0.5
        ),
        height=300 + (len(summary_df) * 35),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Create WER improvement bar chart
    improvement_data = [
        {
            'Model': s['Model'],
            'Baseline WER': s['_baseline_wer'] if s['_baseline_wer'] != float('inf') else None,
            'Best WER': s['_best_wer'] if s['_best_wer'] != float('inf') else None,
            'Improvement': s['_improvement'] if s['_improvement'] != -float('inf') else None
        }
        for s in training_summary
    ]
    improvement_df = pd.DataFrame(improvement_data)
    improvement_df = improvement_df[improvement_df['Improvement'].notna()]
    
    fig_improvement = None
    if not improvement_df.empty:
        fig_improvement = go.Figure()
        
        # Bar chart comparing baseline vs best WER
        fig_improvement.add_trace(go.Bar(
            name='Baseline WER',
            x=improvement_df['Model'],
            y=improvement_df['Baseline WER'],
            marker_color='#e74c3c',
            text=[f"{v:.4f}" for v in improvement_df['Baseline WER']],
            textposition='outside'
        ))
        
        fig_improvement.add_trace(go.Bar(
            name='Best WER (Fine-tuned)',
            x=improvement_df['Model'],
            y=improvement_df['Best WER'],
            marker_color='#27ae60',
            text=[f"{v:.4f}" for v in improvement_df['Best WER']],
            textposition='outside'
        ))
        
        fig_improvement.update_layout(
            title="WER Comparison: Baseline vs Fine-Tuned",
            xaxis_title="Model",
            yaxis_title="Word Error Rate (WER)",
            barmode='group',
            height=500,
            legend=dict(x=0.7, y=0.95),
            margin=dict(l=20, r=20, t=60, b=40)
        )
    
    # Create training curves for each model
    training_curves = []
    for base_model, stats in training_stats.items():
        if stats.get('training_stats') and stats['training_stats'].get('training_curve'):
            curve = stats['training_stats']['training_curve']
            if curve.get('eval_wers'):
                training_curves.append({
                    'model': base_model.title(),
                    'steps': curve['eval_steps'],
                    'wers': curve['eval_wers']
                })
    
    fig_curves = None
    if training_curves:
        fig_curves = go.Figure()
        
        colors = px.colors.qualitative.Set3
        for i, curve_data in enumerate(training_curves):
            fig_curves.add_trace(go.Scatter(
                x=curve_data['steps'],
                y=curve_data['wers'],
                mode='lines+markers',
                name=curve_data['model'],
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=6)
            ))
        
        fig_curves.update_layout(
            title="Training Progress: WER Over Training Steps",
            xaxis_title="Training Step",
            yaxis_title="Word Error Rate (WER)",
            height=500,
            legend=dict(x=0.7, y=0.95),
            margin=dict(l=20, r=20, t=60, b=40),
            hovermode='x unified'
        )
    
    # Create improvement percentage chart
    improvement_pct_data = [
        {
            'Model': s['Model'],
            'Improvement %': float(s['Improvement %'].replace('%', '')) if s['Improvement %'] != "N/A" else None
        }
        for s in training_summary
    ]
    improvement_pct_df = pd.DataFrame(improvement_pct_data)
    improvement_pct_df = improvement_pct_df[improvement_pct_df['Improvement %'].notna()]
    
    fig_pct = None
    if not improvement_pct_df.empty:
        fig_pct = go.Figure(data=[
            go.Bar(
                x=improvement_pct_df['Model'],
                y=improvement_pct_df['Improvement %'],
                marker_color='#3498db',
                text=[f"{v:.2f}%" for v in improvement_pct_df['Improvement %']],
                textposition='outside'
            )
        ])
        
        fig_pct.update_layout(
            title="WER Improvement Percentage by Model",
            xaxis_title="Model",
            yaxis_title="Improvement (%)",
            height=400,
            margin=dict(l=20, r=20, t=60, b=40)
        )
    
    # Training duration comparison chart
    duration_data = [
        {
            'Model': s['Model'],
            'Duration (seconds)': s['_duration_seconds'] if s['_duration_seconds'] != float('inf') else None,
            'Duration (str)': s['Training Duration']
        }
        for s in training_summary
    ]
    duration_df = pd.DataFrame(duration_data)
    duration_df = duration_df[duration_df['Duration (seconds)'].notna()]
    
    fig_duration = None
    if not duration_df.empty:
        # Convert seconds to minutes for better readability
        duration_df['Duration (minutes)'] = duration_df['Duration (seconds)'] / 60
        
        fig_duration = go.Figure(data=[
            go.Bar(
                x=duration_df['Model'],
                y=duration_df['Duration (minutes)'],
                marker_color='#9b59b6',
                text=[f"{v:.1f} min" for v in duration_df['Duration (minutes)']],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Duration: %{text}<extra></extra>'
            )
        ])
        
        fig_duration.update_layout(
            title="Training Duration by Model",
            xaxis_title="Model",
            yaxis_title="Training Duration (minutes)",
            height=400,
            margin=dict(l=20, r=20, t=60, b=40)
        )
    
    # WER vs Training Time scatter plot
    wer_time_data = [
        {
            'Model': s['Model'],
            'Best WER': s['_best_wer'] if s['_best_wer'] != float('inf') else None,
            'Training Time (minutes)': (s['_duration_seconds'] / 60) if s['_duration_seconds'] != float('inf') else None,
            'Improvement %': float(s['Improvement %'].replace('%', '')) if s['Improvement %'] != "N/A" else None
        }
        for s in training_summary
    ]
    wer_time_df = pd.DataFrame(wer_time_data)
    wer_time_df = wer_time_df[wer_time_df['Best WER'].notna() & wer_time_df['Training Time (minutes)'].notna()]
    
    fig_wer_time = None
    if not wer_time_df.empty:
        fig_wer_time = go.Figure()
        
        # Create scatter plot with size based on improvement percentage
        fig_wer_time.add_trace(go.Scatter(
            x=wer_time_df['Training Time (minutes)'],
            y=wer_time_df['Best WER'],
            mode='markers+text',
            marker=dict(
                size=[max(20, v * 2) if v else 20 for v in wer_time_df['Improvement %']],
                color=wer_time_df['Improvement %'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Improvement %"),
                line=dict(width=2, color='white')
            ),
            text=wer_time_df['Model'],
            textposition='top center',
            hovertemplate='<b>%{text}</b><br>Best WER: %{y:.4f}<br>Training Time: %{x:.1f} min<extra></extra>'
        ))
        
        fig_wer_time.update_layout(
            title="WER vs Training Time: Efficiency Comparison",
            xaxis_title="Training Time (minutes)",
            yaxis_title="Best WER (lower is better)",
            height=500,
            margin=dict(l=20, r=20, t=60, b=40),
            hovermode='closest'
        )
        
        # Invert y-axis so lower WER is at top (better)
        fig_wer_time.update_yaxes(autorange='reversed')
    
    # Build the layout
    panels = []
    
    # Summary table
    panels.append(dbc.Row([
        dbc.Col([
            dcc.Graph(figure=comparison_table, config={'displayModeBar': False})
        ], width=12)
    ], className="mb-4"))
    
    # WER comparison chart
    if fig_improvement:
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_improvement)
            ], width=12)
        ], className="mb-4"))
    
    # Training curves
    if fig_curves:
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_curves)
            ], width=12)
        ], className="mb-4"))
    
    # Improvement percentage chart
    if fig_pct:
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_pct)
            ], width=12)
        ], className="mb-4"))
    
    # Training duration chart
    if fig_duration:
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_duration)
            ], width=12)
        ], className="mb-4"))
    
    # WER vs Training Time scatter plot
    if fig_wer_time:
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_wer_time)
            ], width=12)
        ], className="mb-4"))
    
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H3("📊 Fine-Tuning Results Summary", className="mb-4"),
                html.P([
                    "This section provides a comprehensive overview of fine-tuning results for all Whisper models ",
                    "(Tiny, Base, Small, Medium, Large) on the Hani89 medical ASR dataset. Compare baseline performance ",
                    "with fine-tuned models to see accuracy improvements, training progress, and training time efficiency. ",
                    "Use the charts below to identify the best trade-offs between accuracy (WER) and training time."
                ], className="mb-4 text-muted")
            ], width=12)
        ]),
        *panels
    ], fluid=True)

