"""
run_07 — CPU-1 Core + Bolmo (dynamic byte patches)
====================================================
Añade BoundaryPredictor a CPU-1 core para patches de longitud variable.
Los patches ya no son fijos de 4 bytes; se cortan en límites naturales del texto.

Propósito:
    Responde: ¿Los patches dinámicos (Bolmo) mejoran sobre los fijos (4-byte)?
    run_06 → run_07 : patches fijos vs dinámicos

BoundaryPredictor (Bolmo):
    Arquitectura:  Embedding(256, 8) → Linear(16, 32) → ReLU → Linear(32, 1) → Sigmoid
    Input:         byte_actual + byte_siguiente (lookahead 1)
    Output:        P(boundary) ∈ [0, 1]
    Tamaño:        ~2KB → siempre en L1 cache durante inferencia

    Boundaries naturales aprendidos:
        - Espacios (0x20)
        - Puntuación (.,!?;:)
        - Inicio de caracteres UTF-8 multibyte (0xC0–0xFF)
        - Forzado cada 8 bytes máximo

    En training:  supervisado con get_natural_boundaries(byte_seq)
    En inferencia: threshold=0.5

Arquitectura completa (~50M params):
    Embedding:        AblationBytePatchEmbedding, BitEmbedding + BitLinear
    Token mixer:      AblationMLGRU(d_model=512, fp_rank=4)
    FFN:              AblationSwiGLU(fp_rank=4)
    Blocks:           12
    Heads:            4 × AblationByteHead
    Cuantización:     Ternary
    FPResidual:       Sí (fp_rank=4)
    BoundaryPredictor: Sí (embed_dim=8, hidden_dim=32)
    DeleteGate:       No

Input:  [B, S, 4] uint8 (patches de longitud variable, paddeados a 4)
Output: 4-tuple de [B, S, 256] logits

Citas:
    - BLT (Pagnoni et al., 2024, arXiv 2410.20771)
    - Bolmo (Allen AI, 2025)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_07"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch dinámico (BoundaryPredictor)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        True,
    "boundary_predictor": {"embed_dim": 8, "hidden_dim": 32, "size_kb": 2},
    "delete_gate":  False,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "params_approx": "~50M",
}


def build() -> AblationModel:
    """Instanciar el modelo run_07 (CPU-1 core + Bolmo)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 core + Bolmo):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"BoundaryPredictor: {model.use_boundary}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
