import os
from pathlib import Path
from huggingface_hub import hf_hub_download


class ModelManager:
    def __init__(self, registry, cache_dir="~/.cache/doppler_seg/models"):
        self.registry = registry
        self.cache_dir = Path(os.path.expanduser(cache_dir))

    def resolve(self, model_name: str, force_update: bool = False):
        spec = self.registry.get(model_name)

        local_path = hf_hub_download(
            repo_id=spec.hf_repo,
            filename=spec.filename,
            revision=spec.revision,
            cache_dir=self.cache_dir,
            force_download=force_update,
        )

        return spec, Path(local_path)