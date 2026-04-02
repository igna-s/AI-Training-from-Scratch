#!/usr/bin/env python3
"""Download LFM2.5-350M model from Hugging Face"""

from huggingface_hub import snapshot_download
import os

model_id = "LiquidAI/LFM2.5-350M"
cache_dir = os.path.abspath("models/LFM2.5-350M")

print(f"Downloading {model_id} to {cache_dir}...")
print("This may take a few minutes depending on your internet speed...\n")

try:
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        repo_type="model",
        local_dir_use_symlinks=False
    )
    print(f"\n[OK] Model downloaded successfully to {cache_dir}")
except Exception as e:
    print(f"\n[ERROR] Error downloading model: {e}")
    raise
