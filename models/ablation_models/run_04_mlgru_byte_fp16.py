"""
run_04 — MLGRU + Byte + FP16
==============================
MLGRU + byte-patches juntos, FP16. Combina los cambios de run_02 y run_03.

Propósito:
    Responde: ¿El efecto de MLGRU + byte-patches se suma o hay interacción?
    Compara con run_03 (solo cambia tokenización: BPE → byte).

Arquitectura (~50M params):
    Embedding:    AblationBytePatchEmbedding(patch_size=4, d_model=512), FP16
    Token mixer:  AblationMLGRU(d_model=512, fp_rank=0), FP16
    FFN:          SwiGLU(512 → 1376 → 512), FP16
    Blocks:       12
    Heads:        4 × Linear(512 → 256), FP16
    Cuantización: FP16 (sin BitLinear)
    FPResidual:   No
    Estado MLGRU: 12 × 512 × 2B = 12KB fijo

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - MatMul-free LM (Zhu et al., 2024)
    - MegaByte (Yu et al., 2023)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_04"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch(patch_size=4)",
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
    """Instanciar el modelo run_04."""
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
