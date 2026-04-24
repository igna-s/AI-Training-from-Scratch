"""
run_03 — MLGRU + BPE + FP16
==============================
MatMul-free token mixer (MLGRU) con tokenización BPE. FP16.
Aísla el impacto del mixer recurrente sin MatMul frente a Transformer.

Propósito:
    Responde: ¿MLGRU supera a Transformer (con BPE, FP16)?
    Compara directamente con run_01 (solo cambia el token mixer).

Arquitectura MLGRU (sin MatMul en mixing):
    h_t = f_t * h_{t-1} + (1 - f_t) * c_t   ← solo Hadamard
    o_t = g_t * h_t

    donde:
        f_t = sigmoid(Linear_f(x_t))   ← forget gate
        c_t = silu(Linear_c(x_t))      ← candidate
        g_t = sigmoid(Linear_g(x_t))   ← output gate

Arquitectura completa (~50M params):
    Embedding:    nn.Embedding(152064, 512), FP16
    Token mixer:  AblationMLGRU(d_model=512, fp_rank=0)
                  → 3 × Linear(512, 512), FP16, sin FPResidual
    FFN:          SwiGLU(512 → 1376 → 512), FP16
    Blocks:       12
    Heads:        Linear(512 → 152064), FP16
    Cuantización: FP16 (sin BitLinear)
    FPResidual:   No
    Estado MLGRU: 12 × 512 × 2B = 12KB fijo (sin KV cache)

Ventaja CPU sobre Transformer:
    - O(d_model) por token vs O(window_size × d_model)
    - Estado recurrente 12KB vs KV cache creciente

Citas:
    - MatMul-free LM (Zhu et al., 2024, arXiv 2406.02528)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_03"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BPE (Qwen2.5, 152K vocab)",
    "quantization": "FP16",
    "fp_residual":  False,
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "recurrent_state_kb": 12,
    "params_approx": "~50M",
}


def build() -> AblationModel:
    """Instanciar el modelo run_03."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import json
    print(f"Config de {RUN_NAME}:")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    total = sum(p.numel() for p in model.parameters())
    print(f"\nParámetros totales: {total / 1e6:.1f}M")
