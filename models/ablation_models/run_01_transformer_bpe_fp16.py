"""
run_01 — Transformer + BPE + FP16
==================================
Baseline absoluto. FP16 Transformer estándar con tokenización BPE (Qwen2.5,
152K vocab). Sin cuantización, sin byte-patches, sin MLGRU.

Propósito:
    Punto de referencia más alto (mejor calidad esperada).
    Todos los demás runs se comparan contra este.

Preguntas que responde:
    run_01 → run_02 : ¿Byte-patches degradan un Transformer FP16?
    run_01 → run_03 : ¿MLGRU supera a Transformer (BPE, FP16)?

Arquitectura (~50M params):
    Embedding:    nn.Embedding(152064, 512)  ← BPE, FP16
    Token mixer:  SlidingWindowAttention(d=512, h=8, head_dim=64, window=512)
    FFN:          SwiGLU(512 → 1376 → 512), FP16
    Blocks:       12
    Heads:        Linear(512 → 152064), FP16
    Cuantización: FP16 (sin BitLinear)
    FPResidual:   No

Citas:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - Qwen2.5 tokenizer (Alibaba, 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_01"

CONFIG = get_run_config(RUN_NAME)

# Resumen de componentes clave para esta arquitectura
ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "SlidingWindowAttention",
    "tokenization": "BPE (Qwen2.5, 152K vocab)",
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
    """Instanciar el modelo run_01."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import json
    print(f"Config de {RUN_NAME}:")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros totales: {total / 1e6:.1f}M")
