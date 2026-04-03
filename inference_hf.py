"""
inference_hf.py — DEPRECATED: use inference.py directly.

All HuggingFace engine code is now in inference.py (BLOQUE 6).
This file exists only for backward-compatibility imports.
"""
from inference import (  # noqa: F401
    HuggingFaceEngine,
    _HFModelWrapper,
    _HFBlockWrapper,
    download_and_save_hf_model,
    _clear_hf_cache,
)
