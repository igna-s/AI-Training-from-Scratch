"""
run_10 — CPU-1 COMPLETE  ⭐ (propuesto)
=========================================
Arquitectura completa propuesta: MLGRU + Byte dinámico (Bolmo) + Ternary +
FPResidual + DeleteGate. Todos los componentes juntos.

Propósito:
    Modelo final propuesto. Combina run_07 (Bolmo) + run_08 (DeleteGate).
    run_06 → run_10 : coste total de añadir todos los extras.

Componentes vs CPU-1 core (run_06):
    + BoundaryPredictor (Bolmo): patches dinámicos alineados a palabras
    + DeleteGate (MrT5):         eliminación de estados en capa 6

Arquitectura (~50M params):
    Embedding:         AblationBytePatchEmbedding, BitEmbedding + BitLinear
    Token mixer:       AblationMLGRU(d_model=512, fp_rank=4)
    FFN:               AblationSwiGLU(fp_rank=4)
    Blocks:            12
    Heads:             4 × AblationByteHead → BitLinear(512 → 256)
    Cuantización:      Ternary {-1,0,+1}
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: Sí — embed(8) → Linear(16,32) → ReLU → Linear(32,1) → Sigmoid
    DeleteGate:        Sí — Linear(512,64) → ReLU → Linear(64,1), en capa 6

Training en 2 fases:
    Fase 1: BoundaryPredictor supervisado + CE loss. DeleteGate pass-through.
    Fase 2: DeleteGate activo (target_rate~0.4) + regularización de eliminación.

Input:  [B, S, 4] uint8 (patches dinámicos, paddeados)
Output: 4-tuple de [B, S, 256] logits

Citas:
    - MatMul-free LM (Zhu et al., 2024)
    - BitNet b1.58 (Wang et al., 2024)
    - BLT / Bolmo (Pagnoni et al., 2024)
    - MrT5 (Pagliardini et al., 2024)
    - Slender-Mamba (Yu et al., COLING 2025)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_10"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch dinámico (BoundaryPredictor)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        True,
    "delete_gate":  True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "recurrent_state_kb": 12,
    "params_approx": "~50M",
    "weight_bits":   1.58,
    "label":        "CPU-1 COMPLETE — propuesto",
}


def build() -> AblationModel:
    """Instanciar el modelo run_10 (CPU-1 COMPLETE)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 COMPLETE):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"BoundaryPredictor: {model.use_boundary}")
    print(f"DeleteGate: {model.use_delete_gate} (capa {model.delete_gate_pos})")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
