"""
small_01 — CPU-1 COMPLETE @ 10M + BPE compact (4K vocab)
===========================================================
CPU-1 COMPLETE a escala 10M con tokenización BPE de vocabulario reducido (4096 tokens).
El vocab completo de Qwen (152K) no cabe en 10M params — embedding = 39M solo.

Propósito:
    ¿Conviene BPE incluso a 10M? Compara con small_02 (byte-level puro).
    Permite iteración rápida (~1h en T4) antes de escalar a 50M.

Constraint de escala:
    10M params → embedding 152K×320 = 48.7M — imposible.
    Solución: top-4096 tokens por frecuencia del dataset.
    Embedding 4096×320 ≈ 1.3M params → manejable.
    BoundaryPredictor desactivado: BPE ya tiene sus propios límites de subword.

Arquitectura (~10M params):
    Embedding:         nn.Embedding(4096, 320) o BitEmbedding(4096, 320)
                       (BPE compact — top-4096 tokens por frecuencia)
    Token mixer:       AblationMLGRU(d_model=320, fp_rank=4)
    FFN:               AblationSwiGLU(320 → 853 → 320, fp_rank=4)
    Blocks:            8
    Heads:             AblationBPEHead → BitLinear(320 → 4096)
    Cuantización:      Ternary
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: No (BPE tiene sus propios límites)
    DeleteGate:        Sí (capa 4)
    Estado MLGRU:      8 × 320 × 2B ≈ 5KB

Training: 2500 steps (~1h en T4, 200M tokens efectivos)

Citas:
    - MatMul-free LM (Zhu et al., 2024)
    - Slender-Mamba (Yu et al., COLING 2025)
    - MrT5 (Pagliardini et al., 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "small_01"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "scale":        "10M",
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BPE compact (4096 tokens)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "bpe_vocab":    CONFIG.get("bpe_vocab_size", 4096),
    "params_approx": "~10M",
    "train_steps":  CONFIG["total_steps"],
}


def build() -> AblationModel:
    """Instanciar el modelo small_01."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 @ 10M + BPE 4K):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    # BPE: input [B, S] token IDs
    x = torch.randint(0, 4096, (1, 64))
    logits, states, extras = model(x)
    print(f"Logits shape: {logits.shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
