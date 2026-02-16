"""
Audio player tab component.
"""

from dash import html
import dash_bootstrap_components as dbc
import json


def create_audio_tab(df):
    """Create the audio player tab"""
    if df is None or (hasattr(df, 'empty') and df.empty) or (hasattr(df, 'shape') and df.shape[0] == 0):
        return html.Div("No data available")
    
    return dbc.Card([
        dbc.CardHeader("🎵 Audio Player"),
        dbc.CardBody([
            html.P("Audio playback functionality would be implemented here."),
            html.P("This would include audio controls and waveform visualization."),
            html.P("For now, showing sample data:"),
            html.Pre(json.dumps(df.head(3).to_dict('records'), indent=2))
        ])
    ])



