# Fix for split-filter and search-input Error

## Issue
Dash was throwing an error: "A nonexistent object was used in an `Input` of a Dash callback. The id of this object is `split-filter`"

## Root Cause
The filter components (`split-filter` and `search-input`) were not properly defined in the main layout, but callbacks were trying to use them.

## Solution
1. Added `dcc.Dropdown` with `id="split-filter"` to the main layout (line 306-313)
2. Added `dbc.Input` with `id="search-input"` to the main layout (line 314-320)
3. Both components are initialized with:
   - `value=None` / `value=""` 
   - `options=[]` for the dropdown
   - `style={"display": "none"}` to hide them by default
4. Updated `update_data_grid` callback to return filter styles to show/hide them based on active tab

## Verification
Both components are confirmed to be in the layout:
- Line 307: `id="split-filter"`
- Line 315: `id="search-input"`

## Next Steps
**IMPORTANT: Restart the Dash server** for the changes to take effect. The error will persist until the server is restarted with the new code.

If the error persists after restart:
1. Clear browser cache
2. Check that the server is running the updated app.py file
3. Verify no other instances of the app are running
