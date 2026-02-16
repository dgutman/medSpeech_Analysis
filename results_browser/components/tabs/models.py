"""
Models tab component for displaying Whisper model information and fine-tuning statistics.
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
from typing import Dict, List

from training_stats_utils import get_model_training_stats, get_training_stats_summary
import logging

logger = logging.getLogger(__name__)


def create_fine_tuning_panel(training_stats: Dict, training_summary: List[Dict]) -> List:
    """Create the fine-tuning statistics panel with tables and visualizations"""
    
    panels = []
    
    if not training_summary:
        panels.append(dbc.Row([
            dbc.Col([
                dbc.Alert(
                    "No fine-tuning statistics found. Please ensure training data is available in the Fine-Tuning-Whisper-on-Custom-Dataset directory.",
                    color="warning"
                )
            ], width=12)
        ]))
        return panels
    
    # Create comparison table
    summary_df = pd.DataFrame(training_summary)
    
    # Sort by improvement (best first)
    if '_improvement' in summary_df.columns:
        summary_df = summary_df.sort_values('_improvement', ascending=False)
    
    # Create comparison table figure
    comparison_table = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "Baseline WER", "Best WER", "WER Improvement", "Improvement %", "Total Epochs", "Total Steps"],
            fill_color='#2c3e50',
            align='left',
            font=dict(size=12, color='white', family='Arial', weight='bold'),
            height=40
        ),
        cells=dict(
            values=[
                summary_df['Model'],
                summary_df['Baseline WER'],
                summary_df['Best WER'],
                summary_df['WER Improvement'],
                summary_df['Improvement %'],
                summary_df['Total Epochs'],
                summary_df['Total Steps']
            ],
            fill_color=[
                ['white'] * len(summary_df),
                ['#f8f9fa'] * len(summary_df),
                ['white'] * len(summary_df),
                ['#e8f5e9'] * len(summary_df),  # Green tint for improvements
                ['#c8e6c9'] * len(summary_df),  # Darker green for percentages
                ['white'] * len(summary_df),
                ['#f8f9fa'] * len(summary_df)
            ],
            align='left',
            font=dict(size=11, color='black', family='Arial'),
            height=35
        )
    )])
    
    comparison_table.update_layout(
        title=dict(
            text="🎯 Fine-Tuning Results Comparison",
            font=dict(size=18, family='Arial', color='#2c3e50'),
            x=0.5
        ),
        height=300 + (len(summary_df) * 35),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    panels.append(dbc.Row([
        dbc.Col([
            dcc.Graph(figure=comparison_table, config={'displayModeBar': False})
        ], width=12)
    ], className="mb-4"))
    
    # Create WER improvement bar chart
    if any(s.get('_improvement', -float('inf')) > -float('inf') for s in training_summary):
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
            
            panels.append(dbc.Row([
                dbc.Col([
                    dcc.Graph(figure=fig_improvement)
                ], width=12)
            ], className="mb-4"))
    
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
        
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_curves)
            ], width=12)
        ], className="mb-4"))
    
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
        
        panels.append(dbc.Row([
            dbc.Col([
                dcc.Graph(figure=fig_pct)
            ], width=12)
        ], className="mb-4"))
    
    return panels


def create_models_tab():
    """Create the models information tab with details about Whisper models and fine-tuning statistics"""
    
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
        training_stats = {}
        training_summary = []
    else:
        training_stats = get_model_training_stats(base_dir)
        training_summary = get_training_stats_summary(base_dir)
    
    # Whisper model specifications
    models_data = [
        {
            "Model": "Tiny (English)",
            "Full Name": "Whisper Tiny English",
            "Parameters": "39M",
            "Size": "~75 MB",
            "Speed": "Fastest (real-time factor ~0.01-0.1x)",
            "Multi-language": False,
            "Accuracy": "Basic",
            "Use Case": "Quick prototyping, low-latency applications",
            "Notes": "English-only variant, optimized for speed"
        },
        {
            "Model": "Base (English)",
            "Full Name": "Whisper Base English",
            "Parameters": "74M",
            "Size": "~140 MB",
            "Speed": "Very Fast (real-time factor ~0.1-0.3x)",
            "Multi-language": False,
            "Accuracy": "Good",
            "Use Case": "Production applications requiring speed",
            "Notes": "English-only variant, good speed/accuracy balance"
        },
        {
            "Model": "Small (English)",
            "Full Name": "Whisper Small English",
            "Parameters": "244M",
            "Size": "~460 MB",
            "Speed": "Fast (real-time factor ~0.3-0.8x)",
            "Multi-language": False,
            "Accuracy": "Very Good",
            "Use Case": "High-accuracy English transcription",
            "Notes": "English-only variant, strong accuracy"
        },
        {
            "Model": "Medium (English)",
            "Full Name": "Whisper Medium English",
            "Parameters": "769M",
            "Size": "~1.5 GB",
            "Speed": "Moderate (real-time factor ~0.8-2x)",
            "Multi-language": False,
            "Accuracy": "Excellent",
            "Use Case": "High-quality English transcription",
            "Notes": "English-only variant, excellent accuracy"
        },
        {
            "Model": "Large",
            "Full Name": "Whisper Large",
            "Parameters": "1550M",
            "Size": "~3 GB",
            "Speed": "Slow (real-time factor ~2-5x)",
            "Multi-language": True,
            "Accuracy": "Best",
            "Use Case": "Highest quality transcription, multi-language support",
            "Notes": "Full multilingual model, best accuracy, slower inference"
        },
        {
            "Model": "Turbo",
            "Full Name": "Whisper Turbo",
            "Parameters": "1550M",
            "Size": "~3 GB",
            "Speed": "Moderate-Fast (real-time factor ~0.5-1.5x)",
            "Multi-language": True,
            "Accuracy": "Excellent",
            "Use Case": "High-quality transcription with faster inference",
            "Notes": "Optimized version of Large model, 2-3x faster while maintaining accuracy"
        }
    ]
    
    # Create DataFrame for easier manipulation
    models_df = pd.DataFrame(models_data)
    
    # Create a styled table using Plotly
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Model", "Parameters", "Size", "Inference Speed", "Multi-language", "Accuracy", "Use Case"],
            fill_color='#2c3e50',
            align='left',
            font=dict(size=12, color='white', family='Arial', weight='bold'),
            height=40
        ),
        cells=dict(
            values=[
                models_df['Model'],
                models_df['Parameters'],
                models_df['Size'],
                models_df['Speed'],
                models_df['Multi-language'].apply(lambda x: "✅ Yes" if x else "❌ No"),
                models_df['Accuracy'],
                models_df['Use Case']
            ],
            fill_color=[
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df),
                ['#f8f9fa'] * len(models_df),
                ['white'] * len(models_df)
            ],
            align='left',
            font=dict(size=11, color='black', family='Arial'),
            height=35
        )
    )])
    
    table_fig.update_layout(
        title=dict(
            text="🤖 Whisper Model Specifications",
            font=dict(size=18, family='Arial', color='#2c3e50'),
            x=0.5
        ),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    # Create detailed cards for each model
    model_cards = []
    for _, model in models_df.iterrows():
        # Color code based on model size/accuracy
        if "Tiny" in model['Model'] or "Base" in model['Model']:
            card_color = "light"
            header_color = "secondary"
        elif "Small" in model['Model'] or "Medium" in model['Model']:
            card_color = "info"
            header_color = "info"
        else:  # Large or Turbo
            card_color = "primary"
            header_color = "primary"
        
        lang_badge = dbc.Badge(
            "🌍 Multilingual" if model['Multi-language'] else "🇺🇸 English Only",
            color="success" if model['Multi-language'] else "warning",
            className="me-2"
        )
        
        card = dbc.Card([
            dbc.CardHeader([
                html.H5([
                    model['Model'],
                    " ",
                    lang_badge
                ], className="mb-0")
            ], className=f"bg-{header_color} text-white"),
            dbc.CardBody([
                html.P([
                    html.Strong("Full Name: "),
                    model['Full Name']
                ], className="mb-2"),
                html.P([
                    html.Strong("Parameters: "),
                    model['Parameters']
                ], className="mb-2"),
                html.P([
                    html.Strong("Model Size: "),
                    model['Size']
                ], className="mb-2"),
                html.P([
                    html.Strong("Inference Speed: "),
                    model['Speed']
                ], className="mb-2"),
                html.P([
                    html.Strong("Accuracy: "),
                    dbc.Badge(model['Accuracy'], color="success", className="ms-1")
                ], className="mb-2"),
                html.P([
                    html.Strong("Use Case: "),
                    model['Use Case']
                ], className="mb-2"),
                html.P([
                    html.Strong("Notes: "),
                    html.Em(model['Notes'], style={"color": "#6c757d"})
                ], className="mb-0")
            ])
        ], className="mb-3", color=card_color, outline=True)
        
        model_cards.append(dbc.Col(card, width=12, md=6, lg=4))
    
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.H4("Model Overview", className="mb-3"),
                html.P([
                    "This dataset uses OpenAI's Whisper models for automatic speech recognition (ASR). ",
                    "Whisper is a family of transformer-based models trained on multilingual and multitask ",
                    "supervised data. The models vary in size and capabilities, with trade-offs between ",
                    "accuracy, speed, and computational requirements."
                ], className="mb-4"),
                dcc.Graph(figure=table_fig, config={'displayModeBar': False})
            ], width=12)
        ], className="mb-4"),
        dbc.Row([
            dbc.Col([
                html.H4("Detailed Model Information", className="mb-3")
            ], width=12)
        ]),
        dbc.Row(model_cards),
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    html.Strong("References: "),
                    html.A("Whisper Paper", href="https://arxiv.org/abs/2212.04356", target="_blank", className="me-3"),
                    html.A("OpenAI Whisper", href="https://github.com/openai/whisper", target="_blank", className="me-3"),
                    html.A("Model Cards", href="https://huggingface.co/models?search=whisper", target="_blank")
                ], className="text-muted small")
            ], width=12)
        ]),
        # Fine-tuning Statistics Section
        dbc.Row([
            dbc.Col([
                html.Hr(className="my-5"),
                html.H4("📊 Fine-Tuning Results", className="mb-4"),
                html.P([
                    "This section shows the results of fine-tuning various Whisper models on the Hani89 medical ASR dataset. ",
                    "Compare baseline performance with fine-tuned models to see accuracy improvements."
                ], className="mb-4 text-muted")
            ], width=12)
        ]),
        *create_fine_tuning_panel(training_stats, training_summary)
    ])

