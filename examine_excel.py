"""
Quick script to examine the Excel file structure
"""
import pandas as pd
import os

def examine_excel_file():
    file_path = "data/raw/2025-08-01 Roadmap Workshop Spreadsheet_Populated_Alec.xlsx"
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return
    
    print("="*80)
    print("📊 EXCEL FILE STRUCTURE ANALYSIS")
    print("="*80)
    
    # Read the Excel file to get sheet names
    xl_file = pd.ExcelFile(file_path)
    
    print(f"\n📋 File: {file_path}")
    print(f"📄 Total sheets: {len(xl_file.sheet_names)}")
    
    for i, sheet_name in enumerate(xl_file.sheet_names, 1):
        print(f"\n🔍 Sheet {i}: '{sheet_name}'")
        print("-" * 60)
        
        try:
            # Read the sheet
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            print(f"   📐 Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Show first few column names
            if len(df.columns) > 0:
                print(f"   📝 First 5 columns: {list(df.columns[:5])}")
                if len(df.columns) > 5:
                    print(f"   📝 Total columns: {len(df.columns)}")
            
            # Show first few rows of data (just the first 3 rows)
            if len(df) > 0:
                print(f"   📊 Sample data (first 3 rows):")
                for idx, row in df.head(3).iterrows():
                    print(f"      Row {idx}: {list(row[:3])}{'...' if len(row) > 3 else ''}")
            
            # Show if there are any empty/NaN patterns
            total_cells = df.shape[0] * df.shape[1]
            empty_cells = df.isna().sum().sum()
            if total_cells > 0:
                fill_percentage = ((total_cells - empty_cells) / total_cells) * 100
                print(f"   📈 Data completeness: {fill_percentage:.1f}% filled")
                
        except Exception as e:
            print(f"   ❌ Error reading sheet: {e}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    examine_excel_file()