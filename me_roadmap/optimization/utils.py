"""
Utility functions for the COSMIC roadmap optimisation pipeline.

Covers array reordering and loading all optimisation input CSVs from the
SmartCity-style format (3 header rows, capabilities as rows, use cases as columns).
"""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def reorder_array(new_order_indices, data_array: np.ndarray) -> np.ndarray:
    """Reorder the rows of a single array.

    Parameters:
        new_order_indices: sequence of integer row indices specifying the new order.
        data_array: array to reorder.

    Returns:
        np.ndarray: reordered copy of data_array.
    """
    return data_array[np.array(new_order_indices)]


def reorder_multiple_arrays(new_order_indices, *arrays) -> tuple:
    """Reorder the rows of multiple arrays by the same index sequence.

    Parameters:
        new_order_indices: sequence of integer row indices.
        *arrays: any number of np.ndarray objects to reorder.

    Returns:
        tuple: reordered arrays in the same order they were passed.
    """
    order = np.array(new_order_indices)
    return tuple(arr[order] for arr in arrays)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _clean_numeric(value) -> float:
    """Parse a cell value to float.

    Handles:
    - NaN/None → np.nan
    - int/float → float as-is
    - Labeled strings ("13.0 - Description") → leading number
    - Plain number strings ("13.0") → parsed float
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"^\s*(\d+\.?\d*)\s*-", value)
        if match:
            return float(match.group(1))
        try:
            return float(value.strip())
        except ValueError:
            return np.nan
    return np.nan


def _load_smartcity_csv(filepath: str) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load a SmartCity-format CSV into a mission × capability DataFrame.

    SmartCity CSVs have 3 header rows (Assignments, Use Case ID, Use Case),
    capabilities as rows, and use cases as columns.  This function loads the
    file, cleans cell values, and transposes so that the returned DataFrame
    has **use cases as the index (rows)** and **capabilities as columns**.

    Parameters:
        filepath: path to the CSV file.

    Returns:
        Tuple:
            - DataFrame with index=use_cases and columns=capabilities.
            - list of use case names (row order).
            - list of capability names (column order).
    """
    df = pd.read_csv(filepath, header=2, index_col=0)
    df.columns = df.columns.str.strip()
    df.index = df.index.str.strip()
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)
    for col in df.columns:
        df[col] = df[col].apply(_clean_numeric)
    df_T = df.T  # rows = use cases, cols = capabilities
    return df_T, df_T.index.tolist(), df_T.columns.tolist()


def load_optimization_data(
    dependency_file: str,
    readiness_file: str,
    learning_rate_file: str,
    utilization_file: str,
) -> Dict:
    """Load all optimisation input CSVs from SmartCity-format files.

    All four files must share the same set of use cases and capabilities.

    Parameters:
        dependency_file: path to dependency CSV.
        readiness_file: path to readiness CSV.
        learning_rate_file: path to learning rate CSV.
        utilization_file: path to utilization CSV.

    Returns:
        dict with keys:
            'dependency'     – np.ndarray (num_missions × num_capabilities)
            'readiness'      – np.ndarray (num_missions × num_capabilities)
            'learning_rate'  – np.ndarray (num_missions × num_capabilities)
            'utilization'    – np.ndarray (num_missions × num_capabilities)
            'mission_names'  – list[str]
            'capability_names' – list[str]
    """
    dep_df, missions, caps = _load_smartcity_csv(dependency_file)
    read_df, _, _ = _load_smartcity_csv(readiness_file)
    lr_df, _, _ = _load_smartcity_csv(learning_rate_file)
    util_df, _, _ = _load_smartcity_csv(utilization_file)

    return {
        "dependency": dep_df.to_numpy(dtype=float),
        "readiness": read_df.to_numpy(dtype=float),
        "learning_rate": lr_df.to_numpy(dtype=float),
        "utilization": util_df.to_numpy(dtype=float),
        "mission_names": missions,
        "capability_names": caps,
    }
