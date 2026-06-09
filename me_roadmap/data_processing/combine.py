"""
Roadmap Data Combiner Module

This module provides functionality to combine dependency and readiness data from CSV files
into a unified data structure for analysis and visualization.
"""
from me_roadmap.data_processing.models import RoadmapData
from typing import Dict, Tuple, Any, Optional
import pandas as pd
import numpy as np
import re


import re


def clean_value(value):
    """
    Cleans a single value to a float.

    Supports:
    - Plain numbers (int or float): returned as-is.
    - Labeled strings like "13.0 - Description": leading number extracted.
    - Plain number strings like "13.0": parsed directly.
    - NaN / None / unparseable: returned as np.nan.

    Args:
        value: Input value to clean.

    Returns:
        float: Cleaned numeric value or np.nan.
    """
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # Labeled format: "13.0 - Description"
        match = re.match(r"^\s*(\d+\.?\d*)\s*-", value)
        if match:
            return float(match.group(1))
        # Plain number string
        try:
            return float(value.strip())
        except ValueError:
            return np.nan
    return np.nan


def load_csv_files(dependency_file: str, readiness_file: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """
    Loads dependency and readiness CSV files into DataFrames and extracts contributor information.
    
    Args:
        dependency_file (str): The file path for the dependency CSV.
        readiness_file (str): The file path for the readiness CSV.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]: Dependency and readiness DataFrames, plus contributor mapping.
        
    Raises:
        FileNotFoundError: If either CSV file cannot be found.
        Exception: If there's an error reading the CSV files.
    """
    try:
        # Read the first row to get contributor information
        contributors_df = pd.read_csv(dependency_file, nrows=1, header=None)
        contributors = {}
        
        # Extract contributors from the first row (skip first column which is "Assignments --->")
        if len(contributors_df.columns) > 1:
            use_case_headers_df = pd.read_csv(dependency_file, skiprows=2, nrows=1, header=None)
            for i in range(1, min(len(contributors_df.columns), len(use_case_headers_df.columns))):
                if pd.notna(contributors_df.iloc[0, i]) and pd.notna(use_case_headers_df.iloc[0, i]):
                    use_case_name = str(use_case_headers_df.iloc[0, i]).strip()
                    contributor_name = str(contributors_df.iloc[0, i]).strip()
                    contributors[use_case_name] = contributor_name
        
        # Load the main dataframes
        dependency_df = pd.read_csv(dependency_file, header=2, index_col=0)
        readiness_df = pd.read_csv(readiness_file, header=2, index_col=0)
        print(f"✅ Successfully loaded CSV files: {dependency_file}, {readiness_file}")
        return dependency_df, readiness_df, contributors
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Please make sure the CSV files are in the correct location.")
        raise
    except Exception as e:
        print(f"❌ An error occurred while reading the CSV files: {e}")
        raise


def clean_dataframes(dependency_df: pd.DataFrame, readiness_df: pd.DataFrame, 
                    apply_value_cleaning: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans the DataFrames by removing whitespace, empty rows/columns, and optionally cleaning cell values.
    
    Args:
        dependency_df (pd.DataFrame): The dependency DataFrame to clean.
        readiness_df (pd.DataFrame): The readiness DataFrame to clean.
        apply_value_cleaning (bool): Whether to apply value cleaning to extract numbers from mixed-format cells.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned dependency and readiness DataFrames.
    """
    dep_clean = dependency_df.copy()
    read_clean = readiness_df.copy()
    
    # Clean column names and index
    dep_clean.columns = dep_clean.columns.str.strip()
    read_clean.columns = read_clean.columns.str.strip()
    dep_clean.index = dep_clean.index.str.strip()
    read_clean.index = read_clean.index.str.strip()

    # Drop empty rows and columns
    dep_clean.dropna(how='all', axis=0, inplace=True)
    dep_clean.dropna(how='all', axis=1, inplace=True)
    read_clean.dropna(how='all', axis=0, inplace=True)
    read_clean.dropna(how='all', axis=1, inplace=True)
    
    # Apply value cleaning if requested
    if apply_value_cleaning:
        print("🧹 Applying enhanced value cleaning...")
        for col in dep_clean.columns:
            dep_clean[col] = dep_clean[col].apply(clean_value)
        for col in read_clean.columns:
            read_clean[col] = read_clean[col].apply(clean_value)
    
    return dep_clean, read_clean


def align_dataframes(dependency_df: pd.DataFrame, readiness_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns DataFrames to have common use_cases and capabilities.
    
    Args:
        dependency_df (pd.DataFrame): The dependency DataFrame.
        readiness_df (pd.DataFrame): The readiness DataFrame.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Aligned dependency and readiness DataFrames.
    """
    common_use_cases = dependency_df.columns.intersection(readiness_df.columns)
    common_capabilities = dependency_df.index.intersection(readiness_df.index)

    dep_aligned = dependency_df.loc[common_capabilities, common_use_cases]
    read_aligned = readiness_df.loc[common_capabilities, common_use_cases]
    
    return dep_aligned, read_aligned


def build_combined_structure(dependency_df: pd.DataFrame, readiness_df: pd.DataFrame) -> Dict[str, Dict[str, Tuple[Any, Any]]]:
    """
    Builds the combined data structure from aligned DataFrames.
    
    Args:
        dependency_df (pd.DataFrame): The aligned dependency DataFrame.
        readiness_df (pd.DataFrame): The aligned readiness DataFrame.
        
    Returns:
        Dict[str, Dict[str, Tuple[Any, Any]]]: Combined data structure.
                Structure: {use_case: {capability: (dependency, readiness)}}
    """
    # Use pandas .stack() for efficient iteration
    dep_stack = dependency_df.stack()
    read_stack = readiness_df.stack()
    combined_data = {}
    for (capability, use_case), dep_value in dep_stack.items():
        read_value = read_stack.get((capability, use_case), None)
        if use_case not in combined_data:
            combined_data[use_case] = {}
        combined_data[use_case][capability] = (dep_value, read_value)
    return combined_data


def display_data_summary(dependency_df: pd.DataFrame, readiness_df: pd.DataFrame) -> None:
    """
    Display a summary of the loaded data for verification.

    Args:
        dependency_df (pd.DataFrame): Cleaned dependency data
        readiness_df (pd.DataFrame): Cleaned readiness data
    """
    use_cases = dependency_df.columns.tolist()
    capabilities = dependency_df.index.tolist()
    
    print("\n" + "="*60)
    print("📊 DATA SUMMARY")
    print("="*60)

    print(f"Number of use_cases: {len(use_cases)}")
    print(f"Number of capabilities: {len(capabilities)}")

    print("\nFirst few use_cases:")
    for i, use_case in enumerate(use_cases[:5]):
        print(f"  {i+1}. {use_case}")
    if len(use_cases) > 5:
        print(f"  ... and {len(use_cases)-5} more")

    print("\nFirst few capabilities:")
    for i, capability in enumerate(capabilities[:5]):
        print(f"  {i+1}. {capability}")
    if len(capabilities) > 5:
        print(f"  ... and {len(capabilities)-5} more")

    print(f"\nDependency data shape: {dependency_df.shape}")
    print(f"Readiness data shape: {readiness_df.shape}")

    # Show data completeness
    dep_completeness = (dependency_df.notna().sum().sum() / (len(use_cases) * len(capabilities))) * 100
    read_completeness = (readiness_df.notna().sum().sum() / (len(use_cases) * len(capabilities))) * 100

    print(f"\nData completeness:")
    print(f"  Dependency data: {dep_completeness:.1f}% filled")
    print(f"  Readiness data: {read_completeness:.1f}% filled")
    print("="*60)


def generate_simplified_csvs(dependency_df: pd.DataFrame, readiness_df: pd.DataFrame, 
                           output_dir: str = ".") -> None:
    """
    Generates simplified CSV files with transposed data (use_cases as rows, capabilities as columns).

    Args:
        dependency_df (pd.DataFrame): DataFrame with dependency data
        readiness_df (pd.DataFrame): DataFrame with readiness data
        output_dir (str): Directory to save the simplified CSV files
    """
    import os
    
    print("📁 Generating simplified CSV files...")

    # Transpose the dataframes so that use_cases are rows and capabilities are columns
    simplified_dependency_df = dependency_df.transpose()
    simplified_readiness_df = readiness_df.transpose()

    # Create output paths
    dep_path = os.path.join(output_dir, 'simplified_dependency.csv')
    read_path = os.path.join(output_dir, 'simplified_readiness.csv')

    # Generate the CSV files (na_rep='' ensures NaN values are saved as empty cells)
    simplified_dependency_df.to_csv(dep_path, na_rep='')
    simplified_readiness_df.to_csv(read_path, na_rep='')

    print("✅ Generated files:")
    print(f"   - {dep_path}")
    print(f"   - {read_path}")


def create_combined_roadmap(dependency_file: str, readiness_file: str, 
                          show_summary: bool = False, 
                          export_simplified: bool = False,
                          apply_value_cleaning: bool = True) -> RoadmapData:
    """
    Orchestrates the complete process of loading, cleaning, aligning, and combining roadmap data.

    This function coordinates all the steps needed to transform CSV files into a unified
    data structure for analysis and visualization.

    Args:
        dependency_file (str): The file path for the dependency CSV.
        readiness_file (str): The file path for the readiness CSV.
        show_summary (bool): Whether to display data summary for verification.
        export_simplified (bool): Whether to export simplified transposed CSV files.
        apply_value_cleaning (bool): Whether to apply enhanced value cleaning.

    Returns:
        RoadmapData: A Pydantic model containing the combined roadmap data.
                    Returns empty RoadmapData if processing fails.
    """
    try:
        dependency_df, readiness_df, contributors = load_csv_files(dependency_file, readiness_file)
        dependency_df, readiness_df = clean_dataframes(dependency_df, readiness_df, apply_value_cleaning)
        
        if show_summary:
            display_data_summary(dependency_df, readiness_df)
        
        dependency_df, readiness_df = align_dataframes(dependency_df, readiness_df)
        
        if export_simplified:
            generate_simplified_csvs(dependency_df, readiness_df)
        
        combined_dict = build_combined_structure(dependency_df, readiness_df)
        roadmap_data = RoadmapData.from_dict(combined_dict, contributors)
        
        print(f"✅ Successfully created roadmap with {roadmap_data.get_use_case_count()} use_cases and {len(roadmap_data.get_all_capabilities())} capabilities")
        return roadmap_data
        
    except Exception as e:
        print(f"❌ Error creating combined roadmap: {e}")
        return RoadmapData(use_cases={})