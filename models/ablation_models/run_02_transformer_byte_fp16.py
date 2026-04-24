"""
run_02 — Transformer + Byte + FP16
====================================
Transformer FP16 con patches de 4 bytes en lugar de tokens BPE.
Aísla el impacto de la tokenización byte-level sobre un Transformer.

Propósito:
    Responde: ¿Los byte-patches degradan un Transformer FP16?
    Compara directamente con run_01 (solo cambia tokenización).

Arquitectura (~50M params):
    Embedding:    AblationBytePatchEmbedding(patch_size=4, d_model=512)
                  → BitEmbedding(256, 8) + Linear(32 → 512), FP16
    Token mixer:  SlidingWindowAttention(d=512, h=8, head_dim=64, window=512)
    FFN:          SwiGLU(512 → 1376 → 512), FP16
    Blocks:       12
    Heads:        4 × Linear(512 → 256), FP16  ← 4 heads, uno por byte del patch
    Cuantización: FP16 (sin BitLinear)
    FPResidual:   No

Input: [B, S, 4] uint8 (patches de 4 bytes)
Output: 4-tuple de [B, S, 256] logits (uno por posición del patch)

Citas:
    - MegaByte (Yu et al., 2023): byte-level patching para LMs
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_02"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "SlidingWindowAttention",
    "tokenization": "BytePatch(patch_size=4)",
    "quantization": "FP16",
    "fp_residual":  False,
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "n_heads":      CONFIG["n_heads"],
    "head_dim":     CONFIG["head_dim"],
    "window_size":  CONFIG["window_size"],
    "params_approx": "~50M",
}


def build() -> AblationModel:
    """Instanciar el modelo run_02."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import json
    print(f"Config de {RUN_NAME}:")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros totales: {total / 1e6:.1f}M")
