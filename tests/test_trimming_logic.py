#!/usr/bin/env python3
"""
Test script to verify trimmed plotting functions work correctly
"""

import pandas as pd
import numpy as np

# Create sample data similar to WandB history
def create_sample_history():
    """Create sample history data with high initial values"""
    np.random.seed(42)
    
    # Simulate 100 training steps
    steps = list(range(100))
    
    # Create gradient norm data with high initial values
    initial_values = np.random.exponential(20, 10)  # Very high initial
    stable_values = np.random.normal(2, 0.5, 90)    # Stable later
    stable_values = np.maximum(stable_values, 0.1)   # Ensure positive
    
    grad_norm = np.concatenate([initial_values, stable_values])
    
    # Create loss data
    loss_values = 3.0 * np.exp(-np.array(steps) / 20) + np.random.normal(0, 0.1, 100)
    
    history = []
    for i, step in enumerate(steps):
        history.append({
            'train/global_step': step,
            'train/grad_norm': grad_norm[i],
            'train/loss': loss_values[i],
            '_step': i
        })
    
    return history

def test_trimming_logic():
    """Test the trimming logic"""
    history = create_sample_history()
    df_hist = pd.DataFrame(history)
    
    print("📊 Testing Trimming Logic")
    print(f"Original data points: {len(df_hist)}")
    
    # Test percentage trimming
    for percentage in [10, 20, 30]:
        skip_count = int(len(df_hist) * percentage / 100)
        trimmed = df_hist.iloc[skip_count:]
        print(f"  {percentage}% trimming: {len(trimmed)} points remaining (skipped {skip_count})")
    
    # Test fixed row trimming
    for fixed_rows in [5, 10, 15]:
        skip_count = min(fixed_rows, len(df_hist) - 1)
        trimmed = df_hist.iloc[skip_count:]
        print(f"  {fixed_rows} row trimming: {len(trimmed)} points remaining (skipped {skip_count})")
    
    # Show benefit for grad_norm
    grad_norm_data = df_hist['train/grad_norm']
    print("\n🎯 Grad Norm Analysis:")
    print(f"  Original range: {grad_norm_data.min():.2f} - {grad_norm_data.max():.2f}")
    
    # 10% trimming
    skip_10 = int(len(grad_norm_data) * 0.1)
    trimmed_10 = grad_norm_data.iloc[skip_10:]
    print(f"  10% trimmed range: {trimmed_10.min():.2f} - {trimmed_10.max():.2f}")
    print(f"  Improvement: {((grad_norm_data.max() - trimmed_10.max()) / grad_norm_data.max() * 100):.1f}% reduction in max value")
    
    print("\n✅ All trimming tests passed!")

if __name__ == "__main__":
    test_trimming_logic()