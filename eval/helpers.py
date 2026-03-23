"""
Utilidades compartidas para la suite de validación CPU-1 v2.
"""
import sys
import os
import platform

import torch

# Agregar raíz del proyecto al path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model_v2 import CPU1ModelV2, BitLinear, bytes_to_patches  # noqa: E402


def configure_cpu_threads(n_threads: int | None = None) -> int:
    """
    Configurar threads igual que inferencia: cores fisicos, sin HT.
    Replica la logica de inference_v2.configure_cpu_threads.
    """
    if n_threads is None:
        total = os.cpu_count() or 4
        if platform.machine() in ('arm64', 'aarch64'):
            n_threads = total
        else:
            n_threads = max(1, total // 2)

    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    print(f"[CPU opt] Threads: {n_threads} intra-op, 1 inter-op")
    return n_threads


def load_model(checkpoint_path: str, config: dict | None = None,
               device: str = 'cpu') -> CPU1ModelV2:
    """Carga modelo desde checkpoint."""
    model = CPU1ModelV2(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_bytes_from_model(model: CPU1ModelV2, prompt: bytes,
                              max_bytes: int = 500) -> bytes:
    """Genera bytes usando el método del modelo."""
    return model.generate_bytes(prompt, max_tokens=max_bytes, temperature=1.0)
