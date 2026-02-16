"""
Precompute analytics tab visualizations and write them to the viz cache.

Run during Docker build (after preload_data.py) or at container startup so the
Analytics tab loads instantly from cache instead of computing WER on first view.

Usage:
  CACHE_DIR=/var/lib/app/cache python warm_analytics_cache.py   # build/startup
  python warm_analytics_cache.py                                 # uses CACHE_DIR from env or ./cache
"""

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # Use container cache path if set (so cache is baked into image or used at runtime)
    cache_dir = os.environ.get("CACHE_DIR", "./cache")
    os.environ.setdefault("CACHE_DIR", cache_dir)

    logger.info("Warming analytics cache (WER + violin + summary table)...")
    try:
        from data_loader import initialize_connection, get_data
        from components.tabs.analytics import create_analytics_tab
    except ImportError as e:
        logger.error("Import error: %s. Run from results_browser with PYTHONPATH=/app (or .).", e)
        sys.exit(1)

    initialize_connection()
    df = get_data()
    if df is None or (hasattr(df, "empty") and df.empty):
        logger.warning("No data available; skipping analytics cache warm-up.")
        return

    logger.info("Data loaded (%d rows). Building analytics visualizations...", len(df))
    try:
        create_analytics_tab(df)
        logger.info("Analytics cache warm-up complete.")
    except Exception as e:
        logger.warning("Analytics warm-up failed (tab will compute on first view): %s", e)


if __name__ == "__main__":
    main()
