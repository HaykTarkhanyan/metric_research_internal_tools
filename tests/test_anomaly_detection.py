#!/usr/bin/env python3
"""
Test script for spike and dip detection functionality
"""

import numpy as np
import matplotlib.pyplot as plt
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
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot original data
    plt.plot(steps, data, 'b-', alpha=0.7, label='Original Data')
    
    # Plot smoothed data
    if smoothing_window > 1:
        plt.plot(steps, smoothed, 'g--', alpha=0.8, label='Smoothed Data')
    
    # Plot thresholds
    plt.axhline(y=spike_height, color='red', linestyle=':', alpha=0.7, label=f'Spike Threshold ({spike_threshold}σ)')
    plt.axhline(y=dip_height, color='blue', linestyle=':', alpha=0.7, label=f'Dip Threshold ({dip_threshold}σ)')
    plt.axhline(y=mean_val, color='gray', linestyle='--', alpha=0.5, label='Mean')
    
    # Mark spikes
    if len(spike_indices) > 0:
        plt.scatter(steps[spike_indices], data[spike_indices], 
                   color='red', s=100, marker='^', zorder=5, label='Detected Spikes')
    
    # Mark dips
    if len(dip_indices) > 0:
        plt.scatter(steps[dip_indices], data[dip_indices], 
                   color='blue', s=100, marker='v', zorder=5, label='Detected Dips')
    
    plt.xlabel('Training Step')
    plt.ylabel('Loss Value')
    plt.title('Spike and Dip Detection Test')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spike_dip_detection_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
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
    print("Visualization saved as 'spike_dip_detection_test.png'")