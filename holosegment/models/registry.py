"""
Model registry to load available models from YAML.
"""

import yaml
from pathlib import Path
from holosegment.models.spec import ModelSpec


class ModelRegistryConfig:
    def __init__(self, yaml_path: Path):
        self.yaml_path = yaml_path
        self._models = self._load()

    def _load(self):
        with open(self.yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        models = {}
        for name, cfg in raw.items():
            models[name] = ModelSpec(
                name=name,
                task=cfg["task"],
                hf_repo=cfg["hf_repo"],
                filename=cfg["filename"],
                format=cfg["format"],
                input_norm=cfg["input_norm"],
                output_activation=cfg["output_activation"],
                revision=cfg.get("revision", "main"),
            )
        return models

    def get(self, name: str) -> ModelSpec:
        if name not in self._models:
            raise ValueError(f"Unknown model '{name}'")
        return self._models[name]

    def list_models(self):
        return list(self._models.keys())