"""
Utilidades compartidas para la suite de evaluacion CPU-1 v2.

Toda la inferencia pasa por CPU1InferenceEngine de inference.py.
No hay codigo de inferencia duplicado en la suite de eval.
"""
import sys
import os
import platform
import time

# Intel MKL/OpenMP: configurar ANTES de importar torch
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
os.environ.setdefault('KMP_BLOCKTIME', '0')

import torch

# Agregar raiz del proyecto al path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from inference import (  # noqa: E402
    CPU1InferenceEngine,
    forward_with_prefetch,
    ModelInfo,
    detect_model_info,
)
from model import CPU1ModelV2, BitLinear  # noqa: E402


def create_engine(checkpoint_path: str,
                  config: dict | None = None) -> CPU1InferenceEngine:
    """Crear engine de inferencia desde checkpoint.
    Unica forma de cargar un modelo para evaluacion.
    """
    return CPU1InferenceEngine.from_checkpoint(checkpoint_path, config or {})


def get_model(engine: CPU1InferenceEngine) -> CPU1ModelV2:
    """Acceder al modelo subyacente (sin wrapper de compile)."""
    return engine._get_model()


def get_hardware_info() -> dict:
    """Info del hardware para reportes."""
    return {
        'platform': platform.platform(),
        'processor': platform.processor() or 'CPU',
        'cpu_count': os.cpu_count(),
        'torch_threads': torch.get_num_threads(),
    }


def run_check(name: str, fn, *args, **kwargs) -> dict:
    """Ejecutar un check individual con manejo de errores y timing."""
    try:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        return {'status': 'PASS', 'time_s': round(elapsed, 2), 'result': result}
    except AssertionError as e:
        return {'status': 'FAIL', 'error': str(e)}
    except Exception as e:
        return {'status': 'ERROR', 'error': f'{type(e).__name__}: {e}'}
