"""
Callbacks package - registers all application callbacks.
"""

from callbacks.data_callbacks import register_data_callbacks
from callbacks.table_callbacks import register_table_callbacks
from callbacks.tab_callbacks import register_tab_callbacks
from callbacks.compare_callbacks import register_compare_callbacks
from callbacks.hallucinations_callbacks import register_hallucinations_callbacks


def register_all_callbacks():
    """Register all application callbacks"""
    # IMPORTANT: Register tab_callbacks FIRST so tab content (including data-grid) 
    # is created before other callbacks that reference components in tabs
    register_tab_callbacks()
    register_data_callbacks()
    register_table_callbacks()
    register_compare_callbacks()
    register_hallucinations_callbacks()
