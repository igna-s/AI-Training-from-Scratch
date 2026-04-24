"""
run_08 — CPU-1 Core + DeleteGate
==================================
Añade DeleteGate a CPU-1 core. Elimina estados MLGRU redundantes en el
punto medio del modelo (~40% de estados eliminados en Fase 2).

Propósito:
    Responde: ¿DeleteGate reduce velocidad de inferencia con mínima pérdida de calidad?
    run_06 → run_08 : sin eliminación vs con eliminación de estados

DeleteGate (inspirado en MrT5):
    Ubicación:  después de layer n_layers // 2 = capa 6
    Scorer:     Linear(512, 64) → ReLU → Linear(64, 1)
    Tamaño:     ~33KB (FP16)

    Training en 2 fases:
        Fase 1 (target_rate=0.0): mask = 1 para todos → no elimina nada
        Fase 2 (target_rate>0.0): mask soft → elimina ~40% de estados
        Regularización: L = (mean(1 - mask) - target_rate)²

    Inferencia: decisión hard con threshold=0.5
        if delete_gate.forward_infer(h_state): procesar
        else: skip (estado eliminado)

    Diferencia vs MrT5 original:
        MrT5:    opera sobre tokens (Gumbel-Softmax discreto)
        CPU-1:   opera sobre estados MLGRU (continuo en training)

Arquitectura completa (~50M params):
    Embedding:    AblationBytePatchEmbedding, BitEmbedding + BitLinear
    Token mixer:  AblationMLGRU(d_model=512, fp_rank=4)
    FFN:          AblationSwiGLU(fp_rank=4)
    Blocks:       12
    Heads:        4 × AblationByteHead
    Cuantización: Ternary
    FPResidual:   Sí (fp_rank=4)
    BoundaryPredictor: No
    DeleteGate:   Sí — Linear(512,64)→ReLU→Linear(64,1), en capa 6

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - MrT5 (Pagliardini et al., 2024, arXiv 2407.00088)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_08"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch(patch_size=4)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "delete_gate_scorer": "Linear(512,64)→ReLU→Linear(64,1)",
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "params_approx": "~50M",
}


def build() -> AblationModel:
    """Instanciar el modelo run_08 (CPU-1 core + DeleteGate)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 core + DeleteGate):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    # Fase 1: sin eliminación
    logits, states, extras = model(x, training_phase=1, target_elimination=0.0)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"DeleteGate en capa: {model.delete_gate_pos}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
