"""
Pipeline for Holosegment.
"""

from models.registry import ModelRegistry
from io.read_moments import Moments
from preprocessing.preprocessing import Preprocessor
from segmentation import artery_vein_segmentation
from segmentation import binary_segmentation

class Pipeline:
    def __init__(self, config, cache, model_registry):
        self.config = config
        self.cache = cache

        vessel_model = ModelRegistry.get("vessel", config["models"]["vessel"])
        self.cache.vessel_model = vessel_model

    def run(self, input_path):
        # Step 1: Load moments data
        moments = self.load_moments(input_path)
        self.cache.M0 = moments.M0  # Cache M0 data for use in segmentation

        # Step 2: Preprocess data (normalization, registration and flatfield correction)
        M0_ff_video, M0_ff_image = self.preprocess(moments)
        self.cache.M0_ff_video = M0_ff_video
        self.cache.M0_ff_image = M0_ff_image

        # Step 3: Perform binary vessel segmentation
        vessel_mask = self.segment_vessels(self.cache.M0_ff_image)
        self.cache.vessel_mask = vessel_mask

        # Step 4: Perform pulse analysis to compute correlation map and diasys map
        correlation_map, diasys_image = self.pulse_analysis(self.cache.M0_ff_video, self.cache.vessel_mask)
        self.cache.correlation_map = correlation_map
        self.cache.diasys_image = diasys_image

        # Step 5: Perform artery/vein segmentation using correlation and diasys map
        artery_mask, vein_mask = self.av_segmentation(self.cache.M0_ff_video, self.cache.M0_ff_image, self.cache.correlation_map, self.cache.diasys_image)

        return artery_mask, vein_mask
    
    def load_moments(self, input_path):
        reader = Moments(input_path)
        reader.read_moments()  # Load data into reader.M0, reader.M1, reader.M2, reader.SH
        return reader
    
    def preprocess(self, moments):
        preprocessor = Preprocessor(self.config)
        preprocessor.preprocess(moments)
        return preprocessor.M0_ff_video, preprocessor.M0_ff_image


    def segment_vessels(self, M0_ff_image):
        """
        Perform binary vessel segmentation, using the M0_ff image
        
        Args:
            M0_ff_image: preprocessed M0_flatfield image of shape (height, width)
            config: artery mask segmentation configuration dict
            cache: cache object for storing/loading models and intermediate results
        Returns:
            Refined artery mask of shape (height, width)
        """
    
        method = self.config.get('BinarySegmentationMethod', 'AI')
        if method == 'AI':
            return binary_segmentation.deep_segmentation(M0_ff_image, self.config, self.cache)[0]  # Return artery mask
        else:
            raise NotImplementedError(f"Binary segmentation method {method} not implemented.")

    def pulse_analysis(self, M0_ff_video, vessel_mask):
        # Implement pulse analysis to compute correlation map and diasys map
        pass

    def av_segmentation(self, M0_ff_video, M0_ff_image, correlation_map, diasys_image):
        """
        Perform artery vein segmentation, using the binary vessel mask
        
        Args:
            M0_ff_video: preprocessed M0_flatfield video of shape (num_frames, height, width)
            M0_ff_image: preprocessed M0_flatfield image of shape (height, width)
            correlation_map: correlation map computed from pulse analysis of shape (height, width)
            diasys_image: diasys image computed from pulse analysis of shape (height, width)
        
        Returns:
            Refined artery mask of shape (height, width)
        """
    
        # Compute pre-artery mask using pulse analysis
        if self.config['AVCorrelationSegmentationNet'] or self.config['AVDiasysSegmentationNet']:
            print("Using deep segmentation model for artery vein segmentation.")
            model = self.models.get_av_segmentation_model(self.config)
            artery_mask, vein_mask = artery_vein_segmentation.deep_segmentation(M0_ff_video, M0_ff_image, correlation_map, diasys_image, model)
        else:
            print("Use hand-made heuristics for artery vein segmentation.")
            artery_mask, vein_mask = artery_vein_segmentation.handmade_segmentation(M0_ff_video, M0_ff_image, correlation_map, diasys_image, self.config, self.cache)
        
        return artery_mask, vein_mask
