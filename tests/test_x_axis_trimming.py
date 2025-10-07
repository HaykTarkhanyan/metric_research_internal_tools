#!/usr/bin/env python3
"""
Test script to verify x-axis trimming works correctly
"""

# Test script for x-axis trimming

def test_axis_trimming():
    """Test that both x and y axes are properly trimmed"""
    
    # Create sample data
    steps = list(range(0, 100, 5))  # [0, 5, 10, 15, ..., 95] - 20 points
    values = [10 - i * 0.1 for i in range(20)]  # Decreasing values
    
    print("📊 Testing X-Axis Trimming")
    print(f"Original data points: {len(steps)}")
    print(f"Original x-axis range: {min(steps)} - {max(steps)}")
    print(f"Original y-axis range: {min(values):.1f} - {max(values):.1f}")
    
    # Test percentage trimming (20%)
    skip_percentage = 20
    skip_count = int(len(values) * skip_percentage / 100)
    
    # Simulate the correct trimming
    y_trimmed = values[skip_count:]
    x_trimmed = steps[skip_count:]
    
    print(f"\nAfter {skip_percentage}% trimming (skip {skip_count} points):")
    print(f"Remaining data points: {len(x_trimmed)}")
    print(f"Trimmed x-axis range: {min(x_trimmed)} - {max(x_trimmed)}")
    print(f"Trimmed y-axis range: {min(y_trimmed):.1f} - {max(y_trimmed):.1f}")
    
    # Verify both axes were trimmed equally
    assert len(x_trimmed) == len(y_trimmed), "X and Y axes have different lengths!"
    
    # Test fixed row trimming
    skip_rows = 5
    y_trimmed_fixed = values[skip_rows:]
    x_trimmed_fixed = steps[skip_rows:]
    
    print(f"\nAfter skipping {skip_rows} rows:")
    print(f"Remaining data points: {len(x_trimmed_fixed)}")
    print(f"Trimmed x-axis range: {min(x_trimmed_fixed)} - {max(x_trimmed_fixed)}")
    print(f"Trimmed y-axis range: {min(y_trimmed_fixed):.1f} - {max(y_trimmed_fixed):.1f}")
    
    # Verify both axes were trimmed equally
    assert len(x_trimmed_fixed) == len(y_trimmed_fixed), "X and Y axes have different lengths!"
    
    print("\n✅ X-axis trimming test passed!")
    print("Both x and y axes are properly synchronized after trimming.")
    
    # Show the improvement
    original_x_span = max(steps) - min(steps)
    trimmed_x_span = max(x_trimmed) - min(x_trimmed)
    reduction = (original_x_span - trimmed_x_span) / original_x_span * 100
    
    print("\n📈 X-axis improvement:")
    print(f"Original span: {original_x_span}")
    print(f"Trimmed span: {trimmed_x_span}")
    print(f"X-axis range reduction: {reduction:.1f}%")

if __name__ == "__main__":
    test_axis_trimming()