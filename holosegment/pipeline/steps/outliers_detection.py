from holosegment.pipeline.step import BaseStep
from holosegment.segmentation.process_masks import elliptical_mask

import numpy as np
from scipy.ndimage import convolve

class OutliersDetectionStep(BaseStep):
    requires = {"moments"}
    produces = {"outlier_frames_mask"}
    name = "preprocess"

    def _relevant_config(self, ctx):
        return {
        }

    def run(self, ctx):
        moments = ctx.require("moments")
        moment0 = moments.M0
        section_mask = elliptical_mask(sz[-2], sz[-1], 1)
        moment0_signal = np.sum(moment0 * ,axis=(1,2)) / np.count_nonzero(section_mask)

        mean_ = np.mean(moment0_signal)
        std_ = np.std(moment0_signal)

        N_ = 3
        outliers = np.abs(moment0_signal - mean_) > N_ * std_

        kernel = np.ones(8)  

        outlier_frames_mask = convolve(outliers.astype(float), kernel, mode='nearest') > 0

        ctx.set("outlier_frames_mask", outlier_frames_mask)