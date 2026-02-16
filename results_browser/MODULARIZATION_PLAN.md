# Modularization Plan

## Current Status
- **app.py**: 2143 lines (TOO LARGE)
- **Created modules**: data_loader.py, utils.py, config.py, components/layout.py, components/tabs/data_table.py
- **Problem**: app.py still contains ALL code, modules are not being used

## What Needs to be Done

### 1. Remove Duplicates from app.py
- Remove data loading functions (already in data_loader.py)
- Remove utility functions (already in utils.py)
- Remove app initialization (should use config.py)

### 2. Extract Tab Components
- `create_analytics_tab()` → `components/tabs/analytics.py`
- `create_hallucinations_tab()` → `components/tabs/hallucinations.py`
- `create_models_tab()` + `create_fine_tuning_panel()` → `components/tabs/models.py`
- `create_compare_tab()` → `components/tabs/compare.py`
- `create_audio_tab()` → `components/tabs/audio.py`

### 3. Extract Callbacks
- All 8 callbacks → `callbacks/data_callbacks.py`, `callbacks/tab_callbacks.py`, etc.

### 4. Refactor app.py
- Import from modules
- Set up layout from components/layout.py
- Register callbacks from callbacks/
- Keep only: imports, app setup, callback registration, main entry point

## Target Structure
```
app.py (~100 lines) - just imports and wiring
├── config.py - app initialization
├── data_loader.py - all data loading
├── utils.py - utility functions
├── components/
│   ├── layout.py - main layout
│   └── tabs/
│       ├── data_table.py
│       ├── analytics.py
│       ├── hallucinations.py
│       ├── models.py
│       ├── compare.py
│       └── audio.py
└── callbacks/
    ├── data_callbacks.py
    ├── tab_callbacks.py
    └── filter_callbacks.py
```



