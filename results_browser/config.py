"""
Application config: creates the Dash app instance.
Exported as `app` for Gunicorn (app:server) and for app.py to attach layout/callbacks.
"""

from dash import Dash
import dash_bootstrap_components as dbc

# Create Dash app with Bootstrap theme; suppress_callback_exceptions so tab content
# (e.g. data-grid) can be targeted by callbacks even when not yet rendered.
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
