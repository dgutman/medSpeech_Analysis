"""
Tiny Rep Compare tab: one summary stat and only rows where the 5 runs disagreed.

1) Summary: % of samples where tiny did not agree with reference but all 5 runs returned the same result.
2) Table: only the rows where the 5 runs produced different results (highlighted).
"""

from dash import html
import dash_bootstrap_components as dbc
import pandas as pd
import logging
from collections import Counter

from utils import extract_text_from_transcription

logger = logging.getLogger(__name__)

TINY_REP_COLS = ["tiny_rep1", "tiny_rep2", "tiny_rep3", "tiny_rep4", "tiny_rep5"]


def _norm(s):
    """Normalize for comparison (strip, collapse whitespace)."""
    if pd.isna(s) or s is None:
        return ""
    return " ".join(str(s).strip().split())


def create_tiny_rep_compare_tab(df):
    """One stat: % where tiny disagreed with ref but all runs agreed. Table: only rows where runs disagreed."""
    if df is None or (hasattr(df, "empty") and df.empty):
        return html.Div("No data available")

    rep_cols = [c for c in TINY_REP_COLS if c in df.columns]
    if not rep_cols:
        return html.Div(
            "No Tiny Rep columns (tiny_rep1–tiny_rep5) found in the data.",
            className="text-muted"
        )

    df_work = df.copy()
    for col in rep_cols:
        df_work[col] = df_work[col].apply(extract_text_from_transcription)
    ref_col = "transcription" if "transcription" in df_work.columns else None

    def row_stats(row):
        texts = [str(row[c]).strip() if pd.notna(row[c]) else "" for c in rep_cols]
        unique = [t for t in texts if t]
        all_same = len(set(unique)) <= 1 if unique else True
        counter = Counter(unique)
        majority = counter.most_common(1)[0][0] if counter else ""
        return all_same, majority, texts

    total = 0
    disagree_with_ref_but_reps_agreed = 0  # tiny != ref, but all 5 reps same
    rows_where_reps_disagreed = []  # only these go in the table

    for idx, row in df_work.iterrows():
        all_same, majority, texts = row_stats(row)
        ref_val = row.get(ref_col) if ref_col else None
        ref_str = _norm(ref_val)
        majority_norm = _norm(majority)
        tiny_disagrees_with_ref = ref_str != majority_norm
        total += 1

        if tiny_disagrees_with_ref and all_same:
            disagree_with_ref_but_reps_agreed += 1
        if not all_same:
            rows_where_reps_disagreed.append({
                "id": row.get("id", idx),
                "reference": ref_str,
                "texts": texts,
                "majority": majority,
                "num_unique": len(set(t for t in texts if t)),
            })

    pct = (100.0 * disagree_with_ref_but_reps_agreed / total) if total else 0
    n_disagree = len(rows_where_reps_disagreed)

    summary = dbc.Card([
        dbc.CardHeader("Tiny Rep summary", className="py-2"),
        dbc.CardBody([
            html.P(
                [
                    html.Strong(f"In {pct:.1f}% of samples ({disagree_with_ref_but_reps_agreed:,} / {total:,}), "),
                    "the tiny model did not agree with the reference but all 5 runs returned the same result."
                ],
                className="mb-0"
            ),
            html.P(
                f"Below: the {n_disagree} samples where the 5 runs produced different results. Minority outputs are highlighted in red.",
                className="mb-0 mt-2 text-muted small"
            )
        ], className="py-2")
    ], className="mb-4")

    table_rows = []
    for r in rows_where_reps_disagreed:
        cells = [
            html.Td(str(r["id"])[:50], style={"maxWidth": "140px", "overflow": "hidden", "textOverflow": "ellipsis"}),
            html.Td(r["reference"][:100] + ("…" if len(r["reference"]) > 100 else ""), style={"maxWidth": "220px"}),
        ]
        for text in r["texts"]:
            is_minority = text != r["majority"] and text
            cell_style = {"backgroundColor": "#f8d7da", "color": "#721c24"} if is_minority else {}
            cells.append(html.Td(text[:70] + ("…" if len(text) > 70 else ""), style={**cell_style, "maxWidth": "180px"}))
        table_rows.append(html.Tr(cells, style={"backgroundColor": "#fff3cd"}))

    header = html.Thead(html.Tr([
        html.Th("ID"),
        html.Th("Reference"),
        *[html.Th(f"Rep{i+1}") for i in range(len(rep_cols))],
    ]))
    table = dbc.Table(
        [header, html.Tbody(table_rows)],
        bordered=True,
        size="sm",
        responsive=True,
        className="mb-0"
    )

    if n_disagree == 0:
        table_section = html.P("There are no samples where the 5 runs produced different results.", className="text-muted")
    else:
        table_section = html.Div(table, style={"overflowX": "auto"})

    return dbc.Container([
        dbc.Row(dbc.Col(summary, width=12)),
        dbc.Row(dbc.Col(table_section, width=12))
    ], fluid=True)
