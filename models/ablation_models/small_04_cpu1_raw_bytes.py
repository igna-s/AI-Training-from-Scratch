"""
small_04 — CPU-1 COMPLETE @ 10M + Raw Bytes (sin teacher)
===========================================================
CPU-1 COMPLETE a 10M entrenado con bytes crudos de HuggingFaceFW/fineweb.
Cero teacher: sin logprobs de Qwen, sin distilación. Solo CE loss en bytes.

Propósito:
    Responde: ¿Los logprobs del teacher Qwen en small_02 realmente ayudan?
    small_02 vs small_04 : CE + Qwen logprobs vs CE puro en bytes crudos.

    Pipeline completamente independiente del dataset cpu1-ablation-dataset:
        - small_02 usa Cukinator/cpu1-ablation-dataset (con teacher logprobs)
        - small_04 usa HuggingFaceFW/fineweb directamente (sin preprocessing)

    Si small_04 ≈ small_02 → el teacher no ayuda → simplifica el pipeline.
    Si small_02 >> small_04 → la distilación vale la pena → mantener.

Diferencias vs small_02:
    - dataset_repo:   HuggingFaceFW/fineweb  (en vez de cpu1-ablation-dataset)
    - training_signal: ce_only (igual, pero sin columna teacher_logprobs)
    - raw_bytes_only:  True → streamer lee bytes UTF-8 directo, sin teacher_mask
    - BoundaryPredictor: supervisado con get_natural_boundaries() en vez de
                         boundaries del dataset (auto-generadas en runtime)

Arquitectura (~10M params):
    Embedding:         AblationBytePatchEmbedding(patch_size=4, d_model=320)
    Token mixer:       AblationMLGRU(d_model=320, fp_rank=4)
    FFN:               AblationSwiGLU(320 → 853 → 320, fp_rank=4)
    Blocks:            8
    Heads:             4 × AblationByteHead → BitLinear(320 → 256)
    Cuantización:      Ternary
    FPResidual:        Sí (fp_rank=4)
    BoundaryPredictor: Sí (supervisado con boundaries naturales auto-generadas)
    DeleteGate:        Sí (capa 4)
    Estado MLGRU:      8 × 320 × 2B ≈ 5KB

Training: 2500 steps desde cero. CE loss puro, sin teacher_mask.

Input:  [B, S, 4] uint8
Output: 4-tuple de [B, S, 256] logits

Nota: la arquitectura del modelo es idéntica a small_02.
      La diferencia está completamente en el pipeline de datos y loss.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import get_run_config, build_ablation_model, AblationModel

RUN_NAME = "small_04"

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
    "training_signal": "CE puro (sin teacher logprobs)",
    "dataset":      "HuggingFaceFW/fineweb (bytes crudos)",
    "raw_bytes_only": True,
    "note":         "Arquitectura idéntica a small_02. Solo cambia el pipeline de datos.",
}


def build() -> AblationModel:
    """Instanciar el modelo small_04 (idéntico en arquitectura a small_02)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 COMPLETE @ 10M + raw bytes):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo")}, indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"BoundaryPredictor: {model.use_boundary}")
    print(f"DeleteGate: {model.use_delete_gate}")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"\nNOTA: arquitectura == small_02. Diferencia: datos y loss (sin teacher).")
