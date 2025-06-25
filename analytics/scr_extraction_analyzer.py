"""
SCR Extraction Analyzer: Statistical Analysis and Quality Assessment

This module provides comprehensive analysis tools for Symptom-Cause-Remedy (SCR) triplet
extraction results. It processes CSV files containing extracted SCR triplets and generates
detailed statistics, quality metrics, and analytical reports.

Key components:
- Data loading and consolidation from multiple CSV extraction files
- Statistical analysis of triplet distribution and content quality
- Equipment and error code coverage analysis
- Automated report generation with detailed metrics
- Console-based summary display and DataFrame inspection

Dependencies: pandas, pathlib, yaml, collections
Usage: Run as standalone script or import functions for custom analysis workflows
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import yaml
from collections import Counter
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

def load_settings() -> Dict[str, Any]:
    """
    Load configuration settings from YAML file.
    
    Reads the main configuration file to access project paths and settings
    required for SCR triplet analysis.
    
    Returns:
        Dict[str, Any]: Configuration dictionary containing all project settings
        
    Raises:
        FileNotFoundError: If settings.yaml file is not found
        yaml.YAMLError: If YAML file is malformed
    """
    config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_scr_data(extract_dir: Path) -> pd.DataFrame:
    """
    Load and consolidate all SCR triplet CSV files from extraction directory.
    
    Searches for all CSV files matching the pattern '*_scr_triplets.csv' and
    combines them into a single DataFrame with additional analytical columns.
    
    Args:
        extract_dir (Path): Directory containing SCR triplet CSV files
        
    Returns:
        pd.DataFrame: Combined DataFrame with all triplets and computed features
        
    Raises:
        FileNotFoundError: If no CSV files are found in the specified directory
    """
    csv_files = list(extract_dir.glob("*_scr_triplets.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {extract_dir}")
    
    # Load and combine all CSV files
    dataframes = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        df['source_file'] = csv_file.name
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Clean and prepare data with computed features
    combined_df['error_code'] = combined_df['symptom'].str.extract(r'([A-Z]+-\d+)')
    combined_df['error_prefix'] = combined_df['error_code'].str.split('-').str[0]
    combined_df['cause_length'] = combined_df['cause'].str.len()
    combined_df['remedy_length'] = combined_df['remedy'].str.len()
    
    return combined_df

def generate_statistics_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive statistical analysis of SCR triplet data.
    
    Computes key metrics including coverage statistics, content analysis,
    and distribution patterns across equipment types and error codes.
    
    Args:
        df (pd.DataFrame): DataFrame containing SCR triplet data
        
    Returns:
        Dict[str, Any]: Dictionary containing all computed statistics and metrics
    """
    stats = {
        'total_triplets': len(df),
        'unique_documents': df['URL'].nunique(),
        'unique_equipments': df['equipment'].nunique(),
        'unique_error_codes': df['error_code'].nunique(),
        'unique_error_prefixes': df['error_prefix'].nunique(),
        'total_pages': df['page'].nunique(),
        'pages_range': (df['page'].min(), df['page'].max()),
        'avg_triplets_per_page': len(df) / df['page'].nunique(),
        'avg_cause_length': df['cause_length'].mean(),
        'avg_remedy_length': df['remedy_length'].mean(),
        'error_prefixes_distribution': dict(df['error_prefix'].value_counts().head(10))
    }
    
    return stats

def create_summary_report(df: pd.DataFrame, stats: Dict[str, Any], output_dir: Path):
    """
    Create comprehensive text-based analysis report.
    
    Generates a detailed markdown report containing all statistical findings,
    distribution analysis, and content quality metrics.
    
    Args:
        df (pd.DataFrame): Source data for analysis
        stats (Dict[str, Any]): Pre-computed statistics dictionary
        output_dir (Path): Directory where report file will be saved
        
    Returns:
        None: Report is saved to file system
    """
    
    report_content = f"""
# SCR EXTRACTION ANALYSIS REPORT
=====================================================

## GENERAL STATISTICS
- **Total extracted triplets**: {stats['total_triplets']:,}
- **Processed documents**: {stats['unique_documents']}
- **Different equipment types**: {stats['unique_equipments']}
- **Unique error codes**: {stats['unique_error_codes']:,}
- **Unique error prefixes**: {stats['unique_error_prefixes']}

## PAGE COVERAGE
- **Total pages processed**: {stats['total_pages']:,}
- **Page range**: {stats['pages_range'][0]} to {stats['pages_range'][1]}
- **Average triplets per page**: {stats['avg_triplets_per_page']:.1f}

## CONTENT ANALYSIS
- **Average cause length**: {stats['avg_cause_length']:.0f} characters
- **Average remedy length**: {stats['avg_remedy_length']:.0f} characters

## TOP ERROR PREFIXES
"""
    
    for prefix, count in list(stats['error_prefixes_distribution'].items())[:10]:
        percentage = (count / stats['total_triplets']) * 100
        report_content += f"- **{prefix}**: {count:,} triplets ({percentage:.1f}%)\n"
    
    # Save the report
    with open(output_dir / 'scr_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print("Detailed report saved: scr_analysis_report.md")

def display_console_summary(stats: Dict[str, Any]):
    """
    Display concise analysis summary in console.
    
    Prints key statistics and findings to console for quick overview
    of extraction results and quality metrics.
    
    Args:
        stats (Dict[str, Any]): Statistics dictionary from generate_statistics_report
        
    Returns:
        None: Output is printed to console
    """
    print(f"\nSCR ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total triplets: {stats['total_triplets']:,}")
    print(f"Documents: {stats['unique_documents']}")
    print(f"Equipment types: {stats['unique_equipments']}")
    print(f"Unique error codes: {stats['unique_error_codes']:,}")
    print(f"Pages processed: {stats['total_pages']:,}")
    print(f"Triplets per page (avg): {stats['avg_triplets_per_page']:.1f}")
    
    print(f"\nTOP 10 ERROR PREFIXES:")
    for i, (prefix, count) in enumerate(list(stats['error_prefixes_distribution'].items())[:10], 1):
        percentage = (count / stats['total_triplets']) * 100
        print(f"   {i:2d}. {prefix}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nAVERAGE LENGTHS:")
    print(f"   Causes: {stats['avg_cause_length']:.0f} characters")
    print(f"   Remedies: {stats['avg_remedy_length']:.0f} characters")

def display_dataframe_sample(df: pd.DataFrame):
    """
    Display structured sample of DataFrame content for inspection.
    
    Shows first 50 rows in tabular format with truncated text fields,
    followed by detailed examples and DataFrame metadata.
    
    Args:
        df (pd.DataFrame): DataFrame containing SCR triplet data
        
    Returns:
        None: Output is printed to console
    """
    print(f"\nDATAFRAME SAMPLE - FIRST 50 ROWS")
    print(f"{'='*120}")
    
    # Display main columns from extraction CSV
    main_columns = ['URL', 'equipment', 'page', 'symptom', 'cause', 'remedy']
    
    # Check that columns exist
    available_columns = [col for col in main_columns if col in df.columns]
    
    if len(available_columns) < 6:
        print(f"Warning: Missing columns. Available columns: {list(df.columns)}")
        display_df = df[available_columns]
    else:
        display_df = df[available_columns].copy()
        
        # Truncate long texts for table display
        display_df['symptom_short'] = display_df['symptom'].str[:50] + '...'
        display_df['cause_short'] = display_df['cause'].str[:40] + '...'
        display_df['remedy_short'] = display_df['remedy'].str[:40] + '...'
        
        # Create display DataFrame with truncated texts
        clean_df = pd.DataFrame({
            'URL': display_df['URL'],
            'Equipment': display_df['equipment'].str[:20] + '...',
            'Page': display_df['page'],
            'Symptom': display_df['symptom_short'],
            'Cause': display_df['cause_short'],
            'Remedy': display_df['remedy_short']
        })
        
        # Configure pandas display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.max_colwidth', 50)
        pd.set_option('display.max_rows', 50)
        
        # Display cleaned DataFrame
        print(clean_df.head(50))
    
    print(f"\nDETAILED EXAMPLES:")
    print(f"{'─'*120}")
    
    # Display 3 complete examples line by line
    for i in range(min(3, len(df))):
        row = df.iloc[i]
        print(f"\nTRIPLET {i+1}")
        print(f"   URL:       {row.get('URL', 'N/A')}")
        print(f"   Equipment: {row.get('equipment', 'N/A')}")
        print(f"   Page:      {row.get('page', 'N/A')}")
        print(f"   Symptom:   {row.get('symptom', 'N/A')}")
        print(f"   Cause:     {row.get('cause', 'N/A')}")
        print(f"   Remedy:    {row.get('remedy', 'N/A')}")
        print(f"   {'─'*100}")
    
    print(f"\nDATAFRAME INFORMATION:")
    print(f"{'─'*60}")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Actual columns: {list(df.columns)}")
    print(f"Expected columns: ['URL', 'equipment', 'page', 'symptom', 'cause', 'remedy']")
    
    if 'page' in df.columns:
        print(f"\nQUICK STATISTICS:")
        print(f"   Pages min/max: {df['page'].min()} - {df['page'].max()}")
        if 'cause' in df.columns:
            print(f"   Cause length avg/max: {df['cause'].str.len().mean():.0f} / {df['cause'].str.len().max()}")
        if 'remedy' in df.columns:
            print(f"   Remedy length avg/max: {df['remedy'].str.len().mean():.0f} / {df['remedy'].str.len().max()}")

def main():
    """
    Main execution function for SCR extraction analysis.
    
    Orchestrates the complete analysis workflow including data loading,
    statistical computation, report generation, and console output.
    
    Returns:
        None: Results are saved to files and displayed in console
    """
    try:
        print("SCR Extraction Results Analysis")
        print("="*60)
        
        # Load configuration
        settings = load_settings()
        base_extract_dir = Path(settings["paths"]["scr_triplets"])
        
        # Output directory for analysis reports
        output_dir = Path(settings["paths"]["outputs"]) / "analytic_reports"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading data from: {base_extract_dir}")
        
        # Load data
        df = load_scr_data(base_extract_dir)
        print(f"Data loaded: {len(df):,} triplets")
        
        # Generate statistics
        print("Generating statistics...")
        stats = generate_statistics_report(df)
        
        # Display console summary
        display_console_summary(stats)
        
        # Display DataFrame sample
        display_dataframe_sample(df)
        
        # Create summary report
        print(f"\nGenerating report...")
        create_summary_report(df, stats, output_dir)
        
        print(f"\nANALYSIS COMPLETED")
        print(f"{'='*60}")
        print(f"Report generated in: {output_dir}")
        print(f"File: scr_analysis_report.md")
        
    except FileNotFoundError as e:
        print(f"Files not found: {e}")
        print(f"Please run first: python scripts/00_extract_scr_triplets.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()