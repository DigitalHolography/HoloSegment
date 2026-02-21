"""
Binary segmentation of retinal vessels, using deep learning models or traditional methods.
"""

import holosegment.utils.model_utils as model_utils
import numpy as np

def deep_segmentation(M0_ff_image, config, cache):
    vessel_mask = model_utils.run_model(M0_ff_image, cache.get_av_segmentation_model(config), preprocess=True)

    return vessel_mask