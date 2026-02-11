"""
Utility functions for deep learning models
"""

import torch

def run_model(input_data, model):
    """
    Run a segmentation model on the input data
    
    Args:
        input_data: numpy array of shape (height, width, channels)
        model: pre-trained segmentation model (e.g., PyTorch or TensorFlow)
    
    Returns:
        Segmentation mask of shape (height, width) with class labels
    """
    # Preprocess input data for model
    # This may include normalization, resizing, etc. depending on the model requirements
    preprocessed_input = preprocess_for_model(input_data)
    
    # Run inference
    with torch.no_grad():
        input_tensor = torch.from_numpy(preprocessed_input).unsqueeze(0).float()  # Add batch dimension
        output = model(input_tensor)
    
    # Post-process output to get segmentation mask
    segmentation_mask = postprocess_model_output(output)
    
    return segmentation_mask  