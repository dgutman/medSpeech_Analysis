from dash import dcc, html
import dash_bootstrap_components as dbc
from datetime import datetime
import logging
import os

from config import app

# Configure logging
logger = logging.getLogger(__name__)

# Add custom CSS for hallucination table styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .hallucination-yes {
                background-color: #ffcccc !important;
                text-align: center;
            }
            .hallucination-no {
                background-color: #ccffcc !important;
                text-align: center;
            }
            .flag-repetition { background-color: #ffeb3b !important; text-align: center; opacity: 0.7; }
            .flag-length-anomaly { background-color: #ff9800 !important; text-align: center; opacity: 0.7; }
            .flag-char-repetition { background-color: #f44336 !important; text-align: center; opacity: 0.7; }
            .flag-insertions { background-color: #9c27b0 !important; text-align: center; opacity: 0.7; }
            .flag-stuttering { background-color: #e91e63 !important; text-align: center; opacity: 0.7; }
            .has-details { background-color: #fff3cd !important; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Initialize Pixeltable connection at startup (before any requests)
from data_loader import initialize_connection
logger.info("Pre-loading data connections...")
initialize_connection()

# Layout
# NOTE: the app layout must be the single source of truth for component IDs used in callbacks.
# Use the modular layout so shared Stores (eg, dataset-stats-store) are always present.
from components.layout import create_layout
app.layout = create_layout()

# Register all callbacks
from callbacks import register_all_callbacks
register_all_callbacks()

# Health check endpoint
@app.server.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Define server for Gunicorn
server = app.server

if __name__ == "__main__":
    # Pixeltable: multiple processes OK, multiple threads in same process NOT. Disable reloader/threading.
    # Allow toggling via env vars (DASH_USE_RELOADER=0, DASH_THREADED=0) for Docker.
    debug = os.environ.get("DASH_DEBUG", "1") == "1"
    use_reloader = os.environ.get("DASH_USE_RELOADER", "1") == "1"
    threaded = os.environ.get("DASH_THREADED", "0") == "1"
    app.run_server(debug=debug, host="0.0.0.0", port=8050, use_reloader=use_reloader, threaded=threaded)
