"""Read explicit experiment configuration records without merging modern presets."""
import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a JSON object")
    required = ("d_model", "n_layers", "d_ff", "patch_size", "seq_len", "seed",
                "batch_per_gpu", "gradient_accumulation_steps", "total_steps",
                "tokenization", "quantization", "token_mixer", "training_signal",
                "dataset_repo", "description")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Incomplete experiment configuration: {missing}")
    for key in ("d_model", "n_layers", "d_ff", "patch_size", "seq_len",
                "batch_per_gpu", "gradient_accumulation_steps", "total_steps"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["tokenization"] not in ("byte", "byte_dynamic", "bpe"):
        raise ValueError("Unsupported tokenization")
    return config
