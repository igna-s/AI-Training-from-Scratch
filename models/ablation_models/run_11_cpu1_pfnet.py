"""
run_11 — CPU-1 COMPLETE + PFNet  (extra)
==========================================
CPU-1 COMPLETE (run_10) con PFNet activado: residual no lineal cache-residente.
PFNet añade una capacidad expresiva que MLGRU estructuralmente no puede representar.

Propósito:
    Responde: ¿PFNet mejora calidad sin añadir coste de RAM en inferencia?
    run_10 → run_11 : con/sin rama no lineal cache-residente

PFNet (Prefetch-Aware Nonlinear Residual):
    Arquitectura por bloque: RMSNorm → BitLinear(512, 32) → SiLU → BitLinear(32, 512)
    pfnet_hidden=32 → params: 12 × 2 × 512 × 32 = 393K total (~786KB BF16)

    Propiedad clave:
        MLGRU h_t = f*h + (1-f)*c  es una interpolación LINEAL entre estados.
        PFNet añade un camino no lineal directo que MLGRU no puede expresar.
        → Amplía la capacidad expresiva sin añadir recurrencia.

    Presupuesto de cache:
        PFNet pesos (12 capas): 393K × 1bit ≈ 49KB → caben en L2 (~512KB/core)
        Estado MLGRU (12 capas): 12 × 512 × 2B = 12KB → caben en L2
        Pesos principales (~50M): streamean desde RAM capa a capa
        → PFNet no añade latencia por RAM: ya está en L2 mientras llegan los pesos

    Output: block_out = mlgru_out + ffn_out + pfnet_out

Arquitectura completa (~50M + 393K params):
    Embedding:         AblationBytePatchEmbedding
    Token mixer:       AblationMLGRU(d_model=512, fp_rank=4)
    FFN:               AblationSwiGLU(fp_rank=4)
    PFNet por bloque:  PFNet(d_model=512, hidden=32) ← RMSNorm + 2×BitLinear
    Blocks:            12
    Heads:             4 × AblationByteHead
    Cuantización:      Ternary
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: Sí
    DeleteGate:        Sí (capa 6)

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "run_11"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch dinámico (BoundaryPredictor)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        True,
    "pfnet_hidden": CONFIG["pfnet_hidden"],
    "pfnet_params_k": 393,
    "pfnet_cache":  "L2-residente (~49KB ternary)",
    "bolmo":        True,
    "delete_gate":  True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "params_approx": "~50M + 393K PFNet",
}


def build() -> AblationModel:
    """Instanciar el modelo run_11 (CPU-1 COMPLETE + PFNet)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 COMPLETE + PFNet):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    pfnet_count = sum(
        1 for b in model.blocks if b.pfnet is not None
    )
    print(f"Bloques con PFNet: {pfnet_count}/{len(model.blocks)}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
