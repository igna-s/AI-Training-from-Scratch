"""
Utilidades compartidas para la suite de validacion CPU-1 v2.

Optimizaciones multicore para Intel i7 8th gen:
- KMP_AFFINITY: pin threads a cores consecutivos (menos cache thrashing)
- KMP_BLOCKTIME=0: sin sleep de threads idle (menor latencia de wakeup)
- Inter-op parallelism: pipeline de ops independientes entre capas
- torch.compile: fusion de kernels para eliminar overhead de Python
"""
import sys
import os
import platform

# Intel MKL/OpenMP: configurar ANTES de importar torch
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
os.environ.setdefault('KMP_BLOCKTIME', '0')

import torch

# Agregar raiz del proyecto al path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import CPU1ModelV2, BitLinear, bytes_to_patches  # noqa: E402


def configure_cpu_threads(n_threads: int | None = None) -> int:
    """
    Configurar threads para maximo throughput en CPU multicore.
    Usa cores fisicos para intra-op (optimo para GEMM) y
    2 threads inter-op para pipeline de ops independientes.
    """
    if n_threads is None:
        total = os.cpu_count() or 4
        if platform.machine() in ('arm64', 'aarch64'):
            n_threads = total
        else:
            n_threads = max(1, total // 2)

    # Sincronizar env vars con el runtime de OpenMP/MKL
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)

    torch.set_num_threads(n_threads)

    # Inter-op: pipeline de ops independientes (e.g., 4 output heads)
    interop = min(2, max(1, n_threads // 3))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        interop = 1

    print(f"[CPU opt] Threads: {n_threads} intra-op, {interop} inter-op")
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
    """Genera bytes usando el metodo del modelo."""
    return model.generate_bytes(prompt, max_tokens=max_bytes, temperature=1.0)


def get_model_parallel_workers() -> int:
    """Workers para forward passes concurrentes en CPU.
    Calcula logical_cores / intra_op_threads para evitar sobresuscripcion
    del thread pool ATen/OpenMP. Para i7-8xxx: 12 / 6 = 2 workers."""
    logical = os.cpu_count() or 4
    intra = torch.get_num_threads() or max(1, logical // 2)
    return max(1, logical // intra)


def get_dataloader_workers() -> int:
    """Workers para DataLoader (carga de datos I/O en paralelo).
    Conservador en Windows (spawn overhead). Retorna 0-2 workers."""
    return max(0, min(2, (os.cpu_count() or 4) // 4))


# ---------------------------------------------------------------------------
# Optimizacion de modelo para benchmarks de throughput
# ---------------------------------------------------------------------------
_compile_cache = {}


def optimize_model_for_inference(model):
    """Optimiza modelo para maximo throughput en CPU multicore.

    - torch.compile: fusiona kernels, elimina overhead Python/autograd
    - Cachea compilacion por modelo (id) para evitar recompilar entre tests
    - El modelo se distribuye entre todos los cores via ATen (OpenMP/MKL)
    - Compatible con inference_mode() para eliminar dispatch overhead

    Fallback: retorna modelo original si torch.compile no esta disponible.
    """
    model_id = id(model)
    if model_id in _compile_cache:
        return _compile_cache[model_id]

    model.eval()
    result = model

    if hasattr(torch, 'compile'):
        try:
            result = torch.compile(model)
            print("[CPU opt] torch.compile activado")
        except Exception as e:
            print(f"[CPU opt] torch.compile no disponible: {e}")

    _compile_cache[model_id] = result
    return result
