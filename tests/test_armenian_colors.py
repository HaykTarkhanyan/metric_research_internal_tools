#!/usr/bin/env python3
"""
Test script for Armenian flag colors functionality
"""

import sys
sys.path.append('.')

from streamlit_wandb_visualizer import get_colors

def test_armenian_colors():
    """Test the Armenian flag color functionality"""
    
    print("🇦🇲 Testing Armenian Flag Colors for WandB Visualizer")
    print("=" * 50)
    
    # Test different numbers of runs
    test_cases = [1, 2, 3, 4, 5, 6]
    
    for num_runs in test_cases:
        colors = get_colors(num_runs)
        print(f"\n📊 {num_runs} run(s):")
        
        if num_runs >= 3:
            print("   Using Armenian flag colors! 🇦🇲")
            expected = ['#D90012', '#0033A0', '#F2A800']  # Red, Blue, Orange
            actual = colors[:3]
            
            for i, (expected_color, actual_color) in enumerate(zip(expected, actual)):
                color_name = ['Red', 'Blue', 'Orange'][i]
                match = "✅" if expected_color == actual_color else "❌"
                print(f"   {color_name}: {actual_color} {match}")
                
            if num_runs > 3:
                print(f"   Additional colors: {colors[3:num_runs]}")
        else:
            print("   Using default Plotly colors")
            print(f"   Colors: {colors[:num_runs]}")
    
    print(f"\n🎨 Armenian Flag Colors:")
    print(f"   🔴 Red:    #D90012")
    print(f"   🔵 Blue:   #0033A0") 
    print(f"   🟠 Orange: #F2A800")
    
    print(f"\n✅ Color scheme test completed!")

if __name__ == "__main__":
    test_armenian_colors()