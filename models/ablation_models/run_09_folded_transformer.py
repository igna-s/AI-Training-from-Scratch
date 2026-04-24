"""
run_09 — Folded Transformer + Byte + Ternary
=============================================
Transformer ternario con sliding-window attention (O(N·128)) sobre byte-patches.
Alternativa a MLGRU — compara el mejor Transformer CPU-friendly vs CPU-1 core.

Propósito:
    Responde: ¿MLGRU (run_06) supera al mejor Transformer local ternario?
    run_06 vs run_09 : MLGRU core vs Folded Transformer

    NO es parte de la cadena progresiva. Es una comparación ortogonal.

SlidingWindowAttention (window_size=128):
    Cada posición atiende a los 128 patches anteriores.
    Complejidad: O(N × 128) vs O(N²) para atención full.
    128 patches = 512 bytes de contexto local.

    Implementación:
        mask = tril(S,S) & triu(S,S, diagonal=-(window-1))
        out  = SDPA(q, k, v, attn_mask=float_mask)

    Sin KV cache de crecimiento infinito — ventana fija de 128.
    Pero SÍ tiene overhead de atención vs O(d_model) de MLGRU.

Arquitectura (~50M params):
    Embedding:    AblationBytePatchEmbedding, BitEmbedding + BitLinear
    Token mixer:  SlidingWindowAttention(d=512, h=8, head_dim=64, window=128)
                  → 4 × BitLinear (Q, K, V, O), sin FPResidual
    FFN:          AblationSwiGLU, 3 × BitLinear, sin FPResidual
    Blocks:       12
    Heads:        4 × AblationByteHead → BitLinear(512 → 256)
    Cuantización: Ternary (BitLinear en todo)
    FPResidual:   No  ← intencionalmente, para aislar el mixer
    window_size:  128 patches (512 bytes)

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - Longformer (Beltagy et al., 2020)
    - BitNet b1.58 (Wang et al., 2024)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_09"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "SlidingWindowAttention(window=128)",
    "tokenization": "BytePatch(patch_size=4)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  False,
    "pfnet":        False,
    "bolmo":        False,
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "n_heads":      CONFIG["n_heads"],
    "head_dim":     CONFIG["head_dim"],
    "window_size":  CONFIG["window_size"],
    "complexity":   "O(N × 128)",
    "params_approx": "~50M",
    "note":         "Alternativa a MLGRU — NO en cadena progresiva",
}


def build() -> AblationModel:
    """Instanciar el modelo run_09 (Folded Transformer)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (Folded Transformer):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"Mixer type: {model.blocks[0].mixer_type}")
    print(f"Window size: {model.blocks[0].token_mixer.window_size}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
