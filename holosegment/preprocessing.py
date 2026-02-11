"""
Preprocessing module for normalization and registration of hologram frames
"""

import numpy as np
from scipy import ndimage
from skimage import exposure


def normalize_frames(frames, method='zscore'):
    """
    Normalize frame intensities
    
    Args:
        frames: numpy array of shape (num_frames, height, width)
        method: normalization method ('zscore', 'minmax', 'percentile')
    
    Returns:
        Normalized frames
    """
    normalized = frames.copy().astype(np.float32)
    
    if method == 'zscore':
        # Z-score normalization
        mean = np.mean(normalized, axis=(1, 2), keepdims=True)
        std = np.std(normalized, axis=(1, 2), keepdims=True)
        normalized = (normalized - mean) / (std + 1e-8)
        
    elif method == 'minmax':
        # Min-max normalization to [0, 1]
        min_val = np.min(normalized, axis=(1, 2), keepdims=True)
        max_val = np.max(normalized, axis=(1, 2), keepdims=True)
        normalized = (normalized - min_val) / (max_val - min_val + 1e-8)
        
    elif method == 'percentile':
        # Percentile-based normalization
        p_low = np.percentile(normalized, 1, axis=(1, 2), keepdims=True)
        p_high = np.percentile(normalized, 99, axis=(1, 2), keepdims=True)
        normalized = np.clip(normalized, p_low, p_high)
        normalized = (normalized - p_low) / (p_high - p_low + 1e-8)
        
    return normalized


def register_frames(frames, reference_idx=0):
    """
    Register frames to a reference frame using phase correlation
    
    Args:
        frames: numpy array of shape (num_frames, height, width)
        reference_idx: index of reference frame
    
    Returns:
        Registered frames
    """
    num_frames = frames.shape[0]
    registered = np.zeros_like(frames)
    
    # Use first frame or specified frame as reference
    reference = frames[reference_idx]
    registered[reference_idx] = reference
    
    for i in range(num_frames):
        if i == reference_idx:
            continue
            
        # Calculate shift using phase correlation
        shift = estimate_shift(reference, frames[i])
        
        # Apply shift
        registered[i] = ndimage.shift(frames[i], shift, mode='nearest')
    
    return registered


def estimate_shift(reference, target):
    """
    Estimate shift between two images using phase correlation
    
    Args:
        reference: reference image
        target: target image to align
    
    Returns:
        (dy, dx) shift values
    """
    # Compute FFT
    ref_fft = np.fft.fft2(reference)
    target_fft = np.fft.fft2(target)
    
    # Compute cross-power spectrum
    cross_power = (ref_fft * np.conj(target_fft)) / (np.abs(ref_fft * np.conj(target_fft)) + 1e-8)
    
    # Inverse FFT to get correlation
    correlation = np.fft.ifft2(cross_power).real
    
    # Find peak
    peak = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Convert to shift values (handle wraparound)
    shift_y = peak[0] if peak[0] < correlation.shape[0] // 2 else peak[0] - correlation.shape[0]
    shift_x = peak[1] if peak[1] < correlation.shape[1] // 2 else peak[1] - correlation.shape[1]
    
    return (shift_y, shift_x)


def preprocess_frames(frames, config):
    """
    Apply preprocessing pipeline to frames
    
    Args:
        frames: numpy array of shape (num_frames, height, width)
        config: preprocessing configuration dict
    
    Returns:
        Preprocessed frames
    """
    # Extract configuration parameters
    normalize_method = config.get('normalize_method', 'zscore')
    do_registration = config.get('register', True)
    reference_frame = config.get('reference_frame', 0)
    
    # Step 1: Normalization
    preprocessed = normalize_frames(frames, method=normalize_method)
    
    # Step 2: Registration
    if do_registration:
        preprocessed = register_frames(preprocessed, reference_idx=reference_frame)
    
    return preprocessed
