"""
run_05 — MLGRU + Byte + Ternary
=================================
Añade cuantización ternaria (1.58-bit) a run_04. Sin FPResidual.
Aísla el coste de la cuantización sobre MLGRU + byte.

Propósito:
    Responde: ¿Cuánto degrada la cuantización ternaria sola?
    run_04 → run_05 : FP16 vs Ternary (todo lo demás igual)
    run_05 → run_06 : ¿FPResidual recupera esa pérdida?

Cuantización BitLinear:
    w_ternary = clamp(round(w / mean(|w|)), -1, 1)  ← {-1, 0, +1}
    x_quant   = clamp(round(x / max(|x|) * 127), -128, 127)
    out = (x_quant @ w_ternary) * (x_scale * w_scale / 127)

    Ventaja CPU: pesos en 2 bits → 16× menos ancho de banda RAM vs FP32.

Arquitectura (~50M params):
    Embedding:    AblationBytePatchEmbedding(patch_size=4, d_model=512)
                  → BitEmbedding(256, 8) + BitLinear(32 → 512)
    Token mixer:  AblationMLGRU(d_model=512, fp_rank=0)
                  → 3 × BitLinear(512, 512), sin FPResidual
    FFN:          AblationSwiGLU, 3 × BitLinear, sin FPResidual
    Blocks:       12
    Heads:        4 × AblationByteHead → BitLinear(512 → 256)
    Cuantización: Ternary (BitLinear en todo)
    FPResidual:   No  ← coste puro de cuantización

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - BitNet b1.58 (Wang et al., 2024)
    - MatMul-free LM (Zhu et al., 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_05"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch(patch_size=4)",
    "quantization": "Ternary {-1,0,+1} (BitLinear)",
    "fp_residual":  False,
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "fp_rank":      0,
    "params_approx": "~50M",
    "weight_bits":   1.58,
}


def build() -> AblationModel:
    """Instanciar el modelo run_05."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME}:")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
