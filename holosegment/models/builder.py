from .model_wrapper import TorchModelWrapper, ONNXModelWrapper


def build_model_wrapper(spec, local_path):

    if spec.format == "pt":
        return TorchModelWrapper(spec, local_path)

    if spec.format == "onnx":
        return ONNXModelWrapper(spec, local_path)

    raise ValueError(f"Unsupported model format: {spec.format}")