"""
models/ablation_models — Arquitecturas de los runs de ablación de CPU-1.

Cada módulo define la config exacta y una función build() que devuelve
el AblationModel correspondiente.

Runs 50M (progresivos):
    run_01  Transformer + BPE + FP16            ← baseline absoluto
    run_02  Transformer + Byte + FP16
    run_03  MLGRU + BPE + FP16
    run_04  MLGRU + Byte + FP16
    run_05  MLGRU + Byte + Ternary
    run_06  MLGRU + Byte + Ternary + FPResidual ← CPU-1 core  ⭐
    run_07  CPU-1 core + Bolmo
    run_08  CPU-1 core + DeleteGate
    run_09  Folded Transformer + Byte + Ternary
    run_10  CPU-1 COMPLETE (Bolmo + DeleteGate) ← propuesto    ⭐
    run_11  CPU-1 COMPLETE + PFNet              ← extra

Runs 10M (scaling check):
    small_01  CPU-1 COMPLETE + BPE 4K
    small_02  CPU-1 COMPLETE + byte             ← propuesto small ⭐
    small_03  CPU-1 COMPLETE + complement (BPE→byte)
    small_04  CPU-1 COMPLETE + raw bytes (sin teacher)

Uso:
    from models.ablation_models import build_run

    model = build_run("run_06")   # CPU-1 core
    model = build_run("run_10")   # CPU-1 COMPLETE
"""

import sys
from pathlib import Path

# Asegurar que el root del proyecto está en sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, ALL_RUN_CONFIGS


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
