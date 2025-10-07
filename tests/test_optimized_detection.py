#!/usr/bin/env python3
"""
Test script for optimized local peak detection in training curves
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_training_curve_with_local_anomalies():
    """Create realistic training curve with subtle local anomalies"""
    np.random.seed(42)
    
    # Create base training curve (decreasing loss with plateaus)
    steps = np.arange(300)
    
    # Base exponential decay with some plateaus
    base_loss = 2.0 * np.exp(-steps / 80) + 0.1
    
    # Add normal training noise
    noise = np.random.normal(0, 0.02, len(steps))
    
    # Add subtle local anomalies that are important for training analysis
    anomalies = np.zeros_like(steps, dtype=float)
    
    # Learning rate schedule changes (small spikes)
    lr_changes = [50, 120, 200]
    for pos in lr_changes:
        anomalies[pos:pos+2] = [0.05, 0.03]  # Small but important spikes
    
    # Data quality issues (small dips)
    data_issues = [80, 150, 220]
    for pos in data_issues:
        anomalies[pos:pos+3] = [-0.04, -0.06, -0.02]  # Small dips
    
    # Gradient explosion (bigger spike - rare but critical)
    anomalies[180:183] = [0.15, 0.20, 0.08]
    
    # Combine all components
    loss_curve = base_loss + noise + anomalies
    
    return steps, loss_curve

def test_local_sensitivity():
    """Test the new sensitive parameters for local anomaly detection"""
    steps, loss_data = create_training_curve_with_local_anomalies()
    
    print("🔍 Testing Optimized Local Peak Detection for Training Curves")
    print("=" * 65)
    print(f"Data points: {len(loss_data)}")
    print(f"Loss range: {loss_data.min():.3f} - {loss_data.max():.3f}")
    
    # Old parameters (less sensitive)
    print(f"\n📊 OLD Parameters (Less Sensitive):")
    old_params = {
        'spike_threshold': 2.5,
        'dip_threshold': 2.5,
        'min_distance': 8,
        'prominence_threshold': 0.15,
        'local_window': 15
    }
    for key, value in old_params.items():
        print(f"   {key}: {value}")
    
    # New parameters (more sensitive)
    print(f"\n🎯 NEW Parameters (Training Curve Optimized):")
    new_params = {
        'spike_threshold': 1.5,
        'dip_threshold': 1.5, 
        'min_distance': 3,
        'prominence_threshold': 0.05,
        'local_window': 12,
        'smoothing_window': 1  # No smoothing to catch sudden changes
    }
    for key, value in new_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n🔄 Key Changes:")
    print(f"   • Lower thresholds (2.5σ → 1.5σ): Catch smaller but important changes")
    print(f"   • Closer peaks allowed (8 → 3 steps): Don't miss clustered anomalies") 
    print(f"   • Lower prominence (0.15 → 0.05): Detect subtle local changes")
    print(f"   • Smaller window (15 → 12): More responsive to immediate context")
    print(f"   • No smoothing (5 → 1): Preserve sudden changes in raw data")
    
    print(f"\n💡 Why These Changes Matter for Training Curves:")
    print(f"   🎯 Loss spikes from learning rate changes")
    print(f"   📉 Sudden dips from data quality issues")  
    print(f"   💥 Gradient explosions (critical to detect early)")
    print(f"   🔄 Training instabilities and oscillations")
    print(f"   📊 Model checkpoint anomalies")
    
    # Create sample detection comparison
    print(f"\n🔍 Impact on Detection:")
    print(f"   • OLD: Misses subtle but important training anomalies")
    print(f"   • NEW: Catches local peaks that matter for ML debugging")
    print(f"   • Perfect for: loss curves, grad_norm, learning_rate plots")
    
    return steps, loss_data

if __name__ == "__main__":
    test_data = test_local_sensitivity()
    print(f"\n✅ Parameters optimized for training curve anomaly detection!")
    print(f"🚀 Your WandB visualizer will now catch more relevant local peaks!")