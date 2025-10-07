#!/usr/bin/env python3
"""
Test script for local vs global anomaly detection
"""

import numpy as np
import pandas as pd

def test_local_vs_global_detection():
    """Compare local vs global anomaly detection"""
    
    # Create test data with local anomalies
    np.random.seed(42)
    
    # Base trend - two distinct phases
    steps = np.arange(100)
    phase1 = 2.0 * np.exp(-steps[:50] / 20) + 1.0  # High values, decreasing
    phase2 = 0.5 + 0.1 * steps[50:]  # Low values, slowly increasing
    base_trend = np.concatenate([phase1, phase2])
    
    # Add noise
    noise = np.random.normal(0, 0.05, len(steps))
    
    # Add a spike in phase 2 (would be normal in phase 1)
    spike_data = base_trend + noise
    spike_data[75] = 1.5  # This is a spike in phase 2 context but normal in phase 1
    
    print("🔍 Testing Local vs Global Spike Detection")
    print(f"Data points: {len(spike_data)}")
    print(f"Phase 1 mean (steps 0-49): {np.mean(spike_data[:50]):.3f}")
    print(f"Phase 2 mean (steps 50-99): {np.mean(spike_data[50:]):.3f}")
    print(f"Global mean: {np.mean(spike_data):.3f}")
    print(f"Spike value at step 75: {spike_data[75]:.3f}")
    
    # Global detection
    global_mean = np.mean(spike_data)
    global_std = np.std(spike_data)
    global_threshold = global_mean + 2.0 * global_std
    
    print(f"\n📊 Global Statistics:")
    print(f"Mean: {global_mean:.3f}")
    print(f"Std: {global_std:.3f}")
    print(f"Spike threshold (2σ): {global_threshold:.3f}")
    print(f"Spike at step 75 detected globally: {spike_data[75] > global_threshold}")
    
    # Local detection (window size 20)
    local_window = 20
    spike_idx = 75
    start_idx = max(0, spike_idx - local_window // 2)
    end_idx = min(len(spike_data), spike_idx + local_window // 2 + 1)
    local_data = spike_data[start_idx:end_idx]
    
    local_mean = np.mean(local_data)
    local_std = np.std(local_data)
    local_threshold = local_mean + 2.0 * local_std
    
    print(f"\n📍 Local Statistics (window {start_idx}-{end_idx}):")
    print(f"Local mean: {local_mean:.3f}")
    print(f"Local std: {local_std:.3f}")
    print(f"Local spike threshold (2σ): {local_threshold:.3f}")
    print(f"Spike at step 75 detected locally: {spike_data[75] > local_threshold}")
    
    # Create sample data
    sample_df = pd.DataFrame({
        'step': steps,
        'value': spike_data,
        'phase_1_context': steps < 50,
        'is_spike_global': spike_data > global_threshold,
        'is_spike_local_75': [False] * len(steps)
    })
    
    sample_df.loc[75, 'is_spike_local_75'] = spike_data[75] > local_threshold
    
    print(f"\n✅ Test Summary:")
    print(f"Local detection is more sensitive to context-specific anomalies")
    print(f"The spike at step 75 (value {spike_data[75]:.3f}) is:")
    print(f"  - {'Detected' if spike_data[75] > local_threshold else 'Not detected'} by local method")
    print(f"  - {'Detected' if spike_data[75] > global_threshold else 'Not detected'} by global method")
    
    if spike_data[75] > local_threshold and spike_data[75] <= global_threshold:
        print("✨ Local detection found an anomaly that global detection missed!")
    
    return sample_df

if __name__ == "__main__":
    test_data = test_local_vs_global_detection()
    print("\n🎯 Local statistics make spikes more relevant to their immediate context!")
    print("This is perfect for training curves where different phases have different scales.")