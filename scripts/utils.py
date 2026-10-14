"""
Shared utilities and path resolvers for the CIC-Trap4Phish benchmark audit.
"""

import os
import re
import pandas as pd


def get_repo_root():
    """Resolve repository root dynamically."""
    if "WORKSPACE_DIR" in os.environ:
        return os.path.abspath(os.environ["WORKSPACE_DIR"])
    curr = os.path.abspath(os.path.dirname(__file__))
    while curr and curr != os.path.dirname(curr):
        if os.path.exists(os.path.join(curr, "config.yaml")) or os.path.exists(os.path.join(curr, "VERSION")):
            return curr
        curr = os.path.dirname(curr)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def sanitize_columns(columns):
    """Sanitize column names by replacing non-alphanumeric characters with underscores."""
    return [re.sub(r'[^\w]', '_', c) for c in columns]


def safe_col(df, col, default=0.0):
    """Safely extract a column as float series, returning default series if absent."""
    if col in df.columns:
        return df[col].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def resolve_data_file(fmt, filename, workspace=None, use_sample=False):
    """
    Resolve path to a format feature CSV, checking data/, root, and sample directories.
    """
    if workspace is None:
        workspace = get_repo_root()

    sample_map = {
        "Word": "Word_Sample.csv",
        "Excel": "Excel_Sample.csv",
        "PDF": "PDF_Sample.csv",
        "HTML": "HTML_Sample.csv"
    }

    if use_sample:
        sample_path = os.path.join(workspace, "data", "sample", sample_map.get(fmt, filename))
        if os.path.exists(sample_path):
            return sample_path

    # Check data/
    cand = os.path.join(workspace, "data", filename)
    if os.path.exists(cand):
        return cand

    # Check workspace root
    cand = os.path.join(workspace, filename)
    if os.path.exists(cand):
        return cand

    # Fallback to sample if full dataset is not present
    sample_path = os.path.join(workspace, "data", "sample", sample_map.get(fmt, filename))
    if os.path.exists(sample_path):
        return sample_path

    return os.path.join(workspace, "data", filename)
