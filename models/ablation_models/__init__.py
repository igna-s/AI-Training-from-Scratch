"""
models/ablation_models — Arquitecturas de los runs de ablación de CPU-1.

Construye el AblationModel correspondiente a cualquier run definido en
train_ablation.RUN_CONFIGS / SMALL_RUN_CONFIGS.

Runs 50M (progresivos):
    run_01  Transformer + BPE + FP16            ← baseline absoluto
    run_02  Transformer + Byte + FP16
    run_03  MLGRU + Byte + FP16
    run_04  MLGRU + Byte + Ternary
    run_05  MLGRU + Byte + Ternary + FPResidual ← CPU-1 core  ⭐
    run_06  CPU-1 core + Bolmo
    run_07  CPU-1 core + DeleteGate
    run_08  Folded Transformer + Byte + Ternary
    run_09  CPU-1 COMPLETE (Bolmo + DeleteGate) ← propuesto    ⭐
    run_10  CPU-1 COMPLETE + PFNet
    run_11  CPU-1 COMPLETE + per-channel decay
    run_12  CPU-1 COMPLETE + state reset

Runs 10M (scaling check):
    run_13  CPU-1 COMPLETE + BPE 4K
    run_14  CPU-1 COMPLETE + byte               ← propuesto small ⭐
    run_15  CPU-1 COMPLETE + complement (BPE→byte, warm-start desde run_13)
    run_16  CPU-1 COMPLETE + raw bytes (sin teacher)

Uso:
    from models.ablation_models import build_run

    model = build_run("run_05")   # CPU-1 core
    model = build_run("run_09")   # CPU-1 COMPLETE
"""

import sys
from pathlib import Path

# Asegurar que el root del proyecto está en sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_ablation import get_run_config, build_ablation_model, ALL_RUN_CONFIGS


def build_run(run_name: str):
    """
    Construir el modelo de ablación para un run dado.

    Args:
        run_name: nombre del run (ej. "run_06", "small_02")

    Returns:
        AblationModel instanciado y listo para training/evaluación.
    """
    config = get_run_config(run_name)
    return build_ablation_model(config)


def list_runs() -> list[str]:
    """Listar todos los runs disponibles."""
    return sorted(ALL_RUN_CONFIGS.keys())


__all__ = ["build_run", "list_runs"]
