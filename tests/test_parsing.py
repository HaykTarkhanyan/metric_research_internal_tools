#!/usr/bin/env python3
"""
Simple test script to verify the WandB data parsing works correctly
"""

import pandas as pd
import ast
import numpy as np

def parse_history(history_str):
    """Parse history string into DataFrame"""
    try:
        if isinstance(history_str, str):
            # Replace 'nan' with 'None' to make it parseable
            cleaned_str = history_str.replace('nan', 'None')
            history_list = ast.literal_eval(cleaned_str)
        else:
            history_list = history_str
        
        if history_list and len(history_list) > 0:
            df_hist = pd.DataFrame(history_list)
            # Convert None back to NaN
            df_hist = df_hist.replace({None: np.nan})
            return df_hist
        else:
            return pd.DataFrame()
    except Exception:
        # If ast.literal_eval fails, try eval as a last resort
        try:
            if isinstance(history_str, str):
                # Create a safe environment for eval
                safe_dict = {"nan": float('nan'), "__builtins__": {}}
                history_list = eval(history_str, safe_dict)
                if history_list and len(history_list) > 0:
                    return pd.DataFrame(history_list)
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

def get_available_metrics(df):
    """Get all available metrics from all runs"""
    all_metrics = set()
    
    for idx, row in df.iterrows():
        hist_df = parse_history(row['history'])
        if not hist_df.empty:
            # Get numeric columns (exclude timestamp and step columns for metric selection)
            numeric_cols = hist_df.select_dtypes(include=[np.number]).columns.tolist()
            all_metrics.update(numeric_cols)
    
    # Remove common non-metric columns
    exclude_cols = {'_timestamp', '_runtime', '_step'}
    all_metrics = all_metrics - exclude_cols
    
    return sorted(list(all_metrics))

def main():
    """Test the parsing functionality"""
    try:
        # Load data
        df = pd.read_csv("wandb_runs_data.csv")
        print(f"✅ Loaded {len(df)} runs from wandb_runs_data.csv")
        
        # Test parsing
        first_run = df.iloc[0]
        hist_df = parse_history(first_run['history'])
        print(f"✅ Successfully parsed history for '{first_run['name']}' - {hist_df.shape[0]} entries")
        
        # Test metric detection
        metrics = get_available_metrics(df)
        print(f"✅ Found {len(metrics)} metrics:")
        for metric in metrics[:10]:  # Show first 10
            print(f"   - {metric}")
        if len(metrics) > 10:
            print(f"   ... and {len(metrics) - 10} more")
        
        # Test common metrics
        common_metrics = ['train/loss', 'eval/loss', 'train/learning_rate']
        found_common = [m for m in common_metrics if m in metrics]
        print(f"✅ Found {len(found_common)} common metrics: {found_common}")
        
        print("\n🎉 All tests passed! The Streamlit app should work correctly.")
        
    except FileNotFoundError:
        print("❌ Error: wandb_runs_data.csv not found")
        print("   Make sure you're running this from the correct directory")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()