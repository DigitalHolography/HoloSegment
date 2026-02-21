from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelSpec:
    name: str
    task: str
    hf_repo: str
    filename: str
    format: str              # "pt" or "onnx"
    input_norm: str
    output_activation: str
    revision: Optional[str] = "main"