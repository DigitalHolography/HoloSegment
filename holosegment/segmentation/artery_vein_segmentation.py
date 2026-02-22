"""
Segmentation module for semantic segmentation
"""

import numpy as np
from skimage import filters, morphology, measure
import holosegment.segmentation.pulse_analysis as pulse_analysis
import holosegment.utils.model_utils as model_utils
  

def deep_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache):
    # Extract configuration
    use_correlation = config.get('AVCorrelationSegmentationNet', True)
    use_diasys = config.get('AVDiasysSegmentationNet', True)

    # Using this pre-artery mask, compute correlation map and diasys map if configured
    correlation_map = pulse_analysis.compute_correlation_map(M0_ff_video, pre_artery_mask) if use_correlation else None
    diasys_map = pulse_analysis.compute_diasys_image(M0_ff_video, pre_artery_mask) if use_diasys else None

    if correlation_map is not None:
        if diasys_map is not None:
            print("Using both correlation map and diasys map for artery mask segmentation.")
            model_input = np.stack([M0_ff_image, correlation_map, diasys_map], axis=-1)
        else:
            print("Using correlation map for artery mask segmentation.")
            model_input = np.stack([M0_ff_image, correlation_map], axis=-1)
    elif diasys_map is not None:
        print("Using diasys map for artery mask segmentation.")
        model_input = np.stack([M0_ff_image, diasys_map], axis=-1)

    artery_mask, vein_mask = model_utils.run_model(model_input, cache.get_av_segmentation_model(config), preprocess=True)

    return artery_mask, vein_mask

def handmade_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache):
    pass


def artery_vein_segmentation(M0_ff_video, M0_ff_image, vessel_mask, config, cache):
    """
    Perform artery vein segmentation, using the binary vessel mask
    
    Args:
        M0_ff_video: preprocessed M0_flatfield video of shape (num_frames, height, width)
        M0_ff_image: preprocessed M0_flatfield image of shape (height, width)
        vessel_mask: binary vessel mask of shape (height, width)
        config: artery mask segmentation configuration dict
    
    Returns:
        Refined artery mask of shape (height, width)
    """
   
    # Compute pre-artery mask using pulse analysis
    pre_artery_mask = pulse_analysis.compute_pre_artery_mask(M0_ff_video, vessel_mask)

    if config['AVCorrelationSegmentationNet'] or config['AVDiasysSegmentationNet']:
        print("Using deep segmentation model for artery vein segmentation.")
        artery_mask, vein_mask = deep_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    else:
        print("Use hand-made heuristics for artery vein segmentation.")
        artery_mask, vein_mask = handmade_segmentation(M0_ff_video, M0_ff_image, pre_artery_mask, config, cache)
    
    return artery_mask, vein_mask