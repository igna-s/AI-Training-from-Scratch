"""
small_02 — CPU-1 COMPLETE @ 10M + Byte  ⭐ (propuesto small)
==============================================================
CPU-1 COMPLETE a escala 10M con byte-level puro desde cero.
Baseline propuesto a escala pequeña — se compara con small_01 (BPE) y small_03 (complement).

Propósito:
    ¿Byte-level puro supera a BPE compact (small_01) a 10M?
    ¿Vale la pena el warm-start BPE → byte de small_03?

Arquitectura (~10M params):
    Embedding:         AblationBytePatchEmbedding(patch_size=4, d_model=320)
                       → BitEmbedding(256, 8) + BitLinear(32 → 320)
    Token mixer:       AblationMLGRU(d_model=320, fp_rank=4)
    FFN:               AblationSwiGLU(320 → 853 → 320, fp_rank=4)
    Blocks:            8
    Heads:             4 × AblationByteHead → BitLinear(320 → 256)
    Cuantización:      Ternary
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: Sí — embed(8) → Linear(16,32) → ReLU → Linear(32,1)
    DeleteGate:        Sí (capa 4)
    Estado MLGRU:      8 × 320 × 2B ≈ 5KB

Training: 2500 steps desde cero en byte data (~1h en T4, 200M tokens)

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - MatMul-free LM (Zhu et al., 2024)
    - Slender-Mamba (Yu et al., COLING 2025)
    - BLT / Bolmo (Pagnoni et al., 2024)
    - MrT5 (Pagliardini et al., 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "small_02"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "scale":        "10M",
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
    "params_approx": "~10M",
    "train_steps":  CONFIG["total_steps"],
    "label":        "propuesto small",
}


def build() -> AblationModel:
    """Instanciar el modelo small_02."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 COMPLETE @ 10M + byte):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"BoundaryPredictor: {model.use_boundary}")
    print(f"DeleteGate: {model.use_delete_gate}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
