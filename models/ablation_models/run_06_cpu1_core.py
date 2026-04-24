"""
run_06 — CPU-1 Core  ⭐
=========================
MLGRU + Byte + Ternary + FPResidual. Arquitectura base de CPU-1.
Añade corrección FP16 de rango bajo en paralelo con los pesos ternarios.

Propósito:
    Responde: ¿FPResidual recupera la pérdida de calidad de la cuantización?
    run_05 → run_06 : Ternary solo vs Ternary + FPResidual
    run_06 es el punto de partida para run_07, run_08, run_09, run_10.

FPResidual (corrección de bajo rango):
    out = BitLinear(x) + FP16_B(SiLU(FP16_A(x)))
    donde FP16_A: [d_model → fp_rank], FP16_B: [fp_rank → d_model]
    fp_rank=4 → 2 × 512 × 4 = 4KB por corrección (FP16)

    Analogía: LoRA aplicado en forward pass, no como fine-tuning.
    El modelo aprende qué errores de cuantización corregir.

Arquitectura (~50M params):
    Embedding:    AblationBytePatchEmbedding, BitEmbedding + BitLinear
    Token mixer:  AblationMLGRU(d_model=512, fp_rank=4)
                  → 3 × BitLinear + FPResidual en output
    FFN:          AblationSwiGLU(fp_rank=4)
                  → 3 × BitLinear + FPResidual en down_proj
    Blocks:       12
    Heads:        4 × AblationByteHead → BitLinear(512 → 256)
    Cuantización: Ternary
    FPResidual:   Sí (fp_rank=4, en mixer y FFN)
    Estado MLGRU: 12 × 512 × 2B = 12KB

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - Slender-Mamba (Yu et al., COLING 2025)
    - BitNet b1.58 (Wang et al., 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_06"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch(patch_size=4)",
    "quantization": "Ternary {-1,0,+1} (BitLinear)",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "recurrent_state_kb": 12,
    "params_approx": "~50M",
    "weight_bits":   1.58,
    "label":        "CPU-1 core",
}


def build() -> AblationModel:
    """Instanciar el modelo run_06 (CPU-1 core)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 core):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
