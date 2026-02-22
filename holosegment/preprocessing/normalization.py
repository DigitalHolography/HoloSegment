"""
Normalization of moments to correct for illumination variations and enhance contrast
"""

import numpy as np

import numpy as np
from scipy.ndimage import gaussian_filter
import math


def flat_field_correction(video, gaussian_blur_ratio, border_amount=0.0):
    video = video.astype(np.float32, copy=False)

    T, H, W = video.shape

    # Blur spatially only (no blur across time)
    blurred = gaussian_filter(video, sigma=(0, gaussian_blur_ratio, gaussian_blur_ratio))
    blurred[blurred == 0] = 1e-8

    corrected = video / blurred

    if border_amount != 0:
        a = int(math.ceil(H * border_amount))
        b = int(math.floor(H * (1 - border_amount)))
        c = int(math.ceil(W * border_amount))
        d = int(math.floor(W * (1 - border_amount)))
    else:
        a, b, c, d = 0, H, 0, W

    ms = np.sum(video[:, a:b, c:d], axis=(1, 2))
    ms2 = np.sum(corrected[:, a:b, c:d], axis=(1, 2))

    scale = np.ones_like(ms)
    valid = ms2 != 0
    scale[valid] = ms[valid] / ms2[valid]

    corrected *= scale[:, None, None]

    return corrected


def normalize_video(frames, method='zscore'):
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