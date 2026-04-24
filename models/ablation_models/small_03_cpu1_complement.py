"""
small_03 — CPU-1 COMPLETE @ 10M + Complement (BPE warm-start → byte)
======================================================================
Empieza desde el checkpoint de small_01 (BPE) y continúa entrenando en
bytes. El embedding BPE se descarta y se reemplaza por uno byte fresco.
Todos los demás pesos (MLGRU, FFN, heads) se heredan del warm-start.

Propósito:
    ¿El pretraining BPE mejora la calidad byte-level final?
    Inspirado en la "byteification" de dos etapas de Bolmo.
    small_02 vs small_03 : byte puro vs BPE warm-start + byte finetune

Protocolo complement:
    1. Cargar checkpoint de small_01 (BPE entrenado 2500 steps)
    2. Descartar embedding BPE → crear AblationBytePatchEmbedding fresco
    3. Activar BoundaryPredictor (no estaba en small_01)
    4. Continuar training en byte data por 2500 steps más
       (mismo presupuesto byte que small_02)

    Carga del checkpoint:
        state = torch.load("small_01_checkpoint.pt")
        model = build()
        # swap embedding
        model.embedding = AblationBytePatchEmbedding(d_model=320, patch_size=4)
        model.load_state_dict(state, strict=False)  # ignora embedding mismatch

Arquitectura post-complement (~10M params):
    Embedding:         AblationBytePatchEmbedding(patch_size=4, d_model=320) ← NUEVA
    Token mixer:       AblationMLGRU(d_model=320, fp_rank=4) ← heredado de small_01
    FFN:               AblationSwiGLU(320 → 853 → 320, fp_rank=4) ← heredado
    Blocks:            8
    Heads:             4 × AblationByteHead ← nuevas (small_01 tenía BPEHead)
    Cuantización:      Ternary
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: Sí ← nueva
    DeleteGate:        Sí (capa 4) ← heredado

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Citas:
    - Bolmo (Allen AI, 2025): byteification en dos etapas
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "small_03"

CONFIG = get_run_config(RUN_NAME)

ARCHITECTURE_SUMMARY = {
    "run":          RUN_NAME,
    "name":         CONFIG["name"],
    "scale":        "10M",
    "token_mixer":  "MLGRU (MatMul-free)",
    "tokenization": "BytePatch dinámico (BoundaryPredictor)",
    "quantization": "Ternary {-1,0,+1}",
    "fp_residual":  True,
    "fp_rank":      CONFIG["fp_rank"],
    "pfnet":        False,
    "bolmo":        True,
    "delete_gate":  True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "d_model":      CONFIG["d_model"],
    "n_layers":     CONFIG["n_layers"],
    "d_ff":         CONFIG["d_ff"],
    "params_approx": "~10M",
    "complement_from": "small_01",
    "note":         "Embedding descartado de small_01. Resto de pesos heredados.",
}


def build() -> AblationModel:
    """
    Instanciar el modelo small_03 (arquitectura byte post-complement).

    Nota: este build() crea el modelo byte-level final. Para hacer el complement
    real (cargar pesos de small_01 y hacer el swap), usar load_from_complement().
    """
    return build_ablation_model(CONFIG)


def load_from_complement(checkpoint_path: str) -> AblationModel:
    """
    Cargar modelo small_03 con pesos heredados de small_01.

    Args:
        checkpoint_path: ruta al checkpoint de small_01

    Returns:
        AblationModel con MLGRU/FFN heredados, embedding y heads frescos.
    """
    import torch
    from ablation import AblationBytePatchEmbedding

    model = build()
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("model_state_dict", state)

    # Cargar pesos con strict=False — embedding y heads no coinciden
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[small_03] Pesos no cargados (nuevo): {len(missing)}")
    print(f"[small_03] Pesos ignorados (BPE): {len(unexpected)}")
    return model


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 complement):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
