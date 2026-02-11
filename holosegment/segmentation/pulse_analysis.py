"""
Pulse analysis module for analyzing temporal pulsatility in vessels
"""

import numpy as np
from scipy import signal, fft
from skimage import measure


def analyze_pulse(frames, vessel_mask, config):
    """
    Analyze pulse characteristics in vessel regions
    
    Args:
        frames: preprocessed frames of shape (num_frames, height, width)
        vessel_mask: binary vessel mask of shape (height, width)
        config: pulse analysis configuration dict
    
    Returns:
        Dictionary containing pulse analysis results:
        - pulsatility_map: spatial map of pulsatility index
        - frequency_map: dominant frequency at each pixel
        - vessel_metrics: per-vessel pulse metrics
    """
    num_frames, height, width = frames.shape
    
    # Extract configuration
    sampling_rate = config.get('sampling_rate', 1.0)  # Hz
    frequency_range = config.get('frequency_range', [0.5, 3.0])  # Hz, typical heart rate range
    
    # Initialize result maps
    pulsatility_map = np.zeros((height, width), dtype=np.float32)
    frequency_map = np.zeros((height, width), dtype=np.float32)
    
    # Compute temporal statistics in vessel regions only
    vessel_coords = np.argwhere(vessel_mask > 0)
    
    for y, x in vessel_coords:
        # Extract temporal signal at this pixel
        temporal_signal = frames[:, y, x]
        
        # Compute pulsatility index
        # PI = (max - min) / mean
        signal_max = np.max(temporal_signal)
        signal_min = np.min(temporal_signal)
        signal_mean = np.mean(temporal_signal)
        
        if signal_mean > 1e-6:
            pulsatility_index = (signal_max - signal_min) / signal_mean
        else:
            pulsatility_index = 0.0
        
        pulsatility_map[y, x] = pulsatility_index
        
        # Compute dominant frequency using FFT
        if num_frames > 4:
            # Detrend signal
            detrended = signal.detrend(temporal_signal)
            
            # Apply window
            window = signal.windows.hann(num_frames)
            windowed = detrended * window
            
            # Compute FFT
            fft_vals = np.abs(fft.fft(windowed))
            freqs = fft.fftfreq(num_frames, d=1.0/sampling_rate)
            
            # Only consider positive frequencies in physiological range
            freq_mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
            if np.any(freq_mask):
                valid_freqs = freqs[freq_mask]
                valid_fft = fft_vals[freq_mask]
                
                # Find peak frequency
                peak_idx = np.argmax(valid_fft)
                dominant_freq = valid_freqs[peak_idx]
                frequency_map[y, x] = dominant_freq
    
    # Compute per-vessel metrics
    vessel_metrics = compute_vessel_metrics(frames, vessel_mask, pulsatility_map, frequency_map)
    
    results = {
        'pulsatility_map': pulsatility_map,
        'frequency_map': frequency_map,
        'vessel_metrics': vessel_metrics,
    }
    
    return results


def compute_vessel_metrics(frames, vessel_mask, pulsatility_map, frequency_map):
    """
    Compute aggregate metrics for each vessel
    
    Args:
        frames: preprocessed frames
        vessel_mask: binary vessel mask
        pulsatility_map: pulsatility index map
        frequency_map: dominant frequency map
    
    Returns:
        List of dictionaries containing per-vessel metrics
    """
    # Label connected components
    labeled_vessels = measure.label(vessel_mask, connectivity=2)
    num_vessels = labeled_vessels.max()
    
    vessel_metrics = []
    
    for vessel_id in range(1, num_vessels + 1):
        vessel_region = labeled_vessels == vessel_id
        
        # Calculate region properties
        props = measure.regionprops(labeled_vessels * (labeled_vessels == vessel_id).astype(int))[0]
        
        # Calculate pulse metrics
        mean_pulsatility = np.mean(pulsatility_map[vessel_region])
        std_pulsatility = np.std(pulsatility_map[vessel_region])
        mean_frequency = np.mean(frequency_map[vessel_region])
        
        # Calculate mean intensity
        mean_intensity = np.mean(frames[:, vessel_region], axis=(0, 1))
        
        metrics = {
            'vessel_id': vessel_id,
            'area': props.area,
            'centroid': props.centroid,
            'mean_pulsatility': float(mean_pulsatility),
            'std_pulsatility': float(std_pulsatility),
            'mean_frequency': float(mean_frequency),
            'mean_intensity': float(mean_intensity),
        }
        
        vessel_metrics.append(metrics)
    
    return vessel_metrics
