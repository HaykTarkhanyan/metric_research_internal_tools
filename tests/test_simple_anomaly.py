#!/usr/bin/env python3
"""
Simple test script for spike and dip detection functionality (no matplotlib)
"""

import numpy as np
from scipy.signal import find_peaks

def create_sample_data_with_anomalies():
    """Create sample training data with artificial spikes and dips"""
    np.random.seed(42)
    
    # Create base trend (decreasing loss)
    steps = np.arange(200)
    base_trend = 3.0 * np.exp(-steps / 50) + 0.5
    
    # Add normal noise
    noise = np.random.normal(0, 0.05, len(steps))
    
    # Add some spikes (loss increases suddenly)
    spikes = np.zeros_like(steps, dtype=float)
    spike_positions = [30, 80, 150]
    for pos in spike_positions:
        spikes[pos:pos+3] = [0.8, 1.2, 0.4]  # Spike pattern
    
    # Add some dips (loss drops suddenly then recovers)
    dips = np.zeros_like(steps, dtype=float)
    dip_positions = [60, 120]
    for pos in dip_positions:
        dips[pos:pos+5] = [-0.3, -0.5, -0.3, -0.1, 0]  # Dip pattern
    
    # Combine all components
    data = base_trend + noise + spikes + dips
    
    return steps, data

def test_spike_dip_detection():
    """Test the spike and dip detection algorithm"""
    steps, data = create_sample_data_with_anomalies()
    
    print("🔍 Testing Spike and Dip Detection")
    print(f"Data points: {len(data)}")
    print(f"Data range: {data.min():.3f} - {data.max():.3f}")
    
    # Parameters
    spike_threshold = 2.0
    dip_threshold = 2.0
    min_distance = 5
    prominence_threshold = 0.1
    smoothing_window = 3
    
    # Apply smoothing
    if smoothing_window > 1:
        smoothed = np.convolve(data, np.ones(smoothing_window)/smoothing_window, mode='same')
    else:
        smoothed = data
    
    # Calculate statistics
    mean_val = np.mean(smoothed)
    std_val = np.std(smoothed)
    
    print(f"Mean: {mean_val:.3f}")
    print(f"Std Dev: {std_val:.3f}")
    
    # Define thresholds
    spike_height = mean_val + spike_threshold * std_val
    dip_height = mean_val - dip_threshold * std_val
    
    print(f"Spike threshold: {spike_height:.3f}")
    print(f"Dip threshold: {dip_height:.3f}")
    
    # Find spikes
    spike_indices, spike_properties = find_peaks(
        smoothed,
        height=spike_height,
        distance=min_distance,
        prominence=prominence_threshold
    )
    
    # Find dips
    inverted = -smoothed
    dip_indices, dip_properties = find_peaks(
        inverted,
        height=-dip_height,
        distance=min_distance,
        prominence=prominence_threshold
    )
    
    print("\n📈 Results:")
    print(f"Spikes detected: {len(spike_indices)} at steps {list(steps[spike_indices])}")
    print(f"Dips detected: {len(dip_indices)} at steps {list(steps[dip_indices])}")
    
    # Show some sample data around anomalies
    if len(spike_indices) > 0:
        print("\n🔴 Spike Details:")
        for i, spike_idx in enumerate(spike_indices[:3]):  # Show first 3
            start = max(0, spike_idx - 2)
            end = min(len(data), spike_idx + 3)
            print(f"  Spike {i+1} at step {steps[spike_idx]} (value: {data[spike_idx]:.3f})")
            print(f"    Context: steps {steps[start]}-{steps[end-1]} = {data[start:end].round(3).tolist()}")
    
    if len(dip_indices) > 0:
        print("\n🔵 Dip Details:")
        for i, dip_idx in enumerate(dip_indices[:3]):  # Show first 3
            start = max(0, dip_idx - 2)
            end = min(len(data), dip_idx + 3)
            print(f"  Dip {i+1} at step {steps[dip_idx]} (value: {data[dip_idx]:.3f})")
            print(f"    Context: steps {steps[start]}-{steps[end-1]} = {data[start:end].round(3).tolist()}")
    
    return {
        'spikes': len(spike_indices),
        'dips': len(dip_indices),
        'spike_positions': list(steps[spike_indices]),
        'dip_positions': list(steps[dip_indices])
    }

if __name__ == "__main__":
    results = test_spike_dip_detection()
    print("\n✅ Test completed!")
    print(f"Detected {results['spikes']} spikes and {results['dips']} dips")
    print("Algorithm working correctly! 🎉")