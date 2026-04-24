"""
run_400M — CPU-1 KERNEL-OPTIMAL  ⭐ (extra)
==========================================
Escalado a ~400M parámetros del CPU-1 COMPLETE+PFNet (run_11), con dimensiones
elegidas específicamente para los kernels AVX2 2-bit MAD del proyecto.

Propósito:
    Modelo de referencia "production-size" diseñado bajo dos restricciones:
      1. **Calidad de generación** — escalar a 400M para empezar a competir con
         baselines como LFM2.5-350M.
      2. **Compatibilidad con el kernel actual** — todas las dimensiones caben
         en los buffers estáticos de `inference/kernels.cpp`
         (MAX_D_MODEL=2048, MAX_D_FF=5120) y son múltiplos de 64 para SIMD
         AVX2 limpio.
    Adicionalmente las dimensiones se eligen para que el pipeline de mode 4
    (`forward_full_mad_pipeline`) reparta limpio entre las 6 stages
    (24 capas / 6 = 4 capas por stage exactas).

Componentes (TODO incluido — el "completo" del 1.58bits stack):
    - Token mixer: AblationMLGRU(d_model=1280, fp_rank=8) — MatMul-free
    - FFN: AblationSwiGLU(d_ff=3328, fp_rank=8)
    - PFNet por bloque: pfnet_hidden=64 (cache-resident)
    - FPResidual: rank=8 (mejor calidad que rank=4 a esta escala)
    - BoundaryPredictor (Bolmo): patches dinámicos
    - DeleteGate (MrT5): eliminación de estados en capa 12
    - Per-channel learnable decay (RWKV/HGRN2): habilitado
    - LocalByteDecoder MegaByte: d_local=192 (mayor que el 128 de 50M),
      n_heads=6, ff_ratio=4.0 → d_ff_local=768

Tabla de parámetros (target ~420M):

    Por capa:
      Mixer (3 GEMVs d_model²):    3 × 1280² = 4.92M
      FFN (3 GEMVs d_model×d_ff): 3 × 1280 × 3328 = 12.78M
      PFNet (2 × d_model × 64):    164K
      FPResidual (2 × d_model × 8): 20K
      ─────────────────────────────────────
      Total por capa:             ~17.9M

    24 capas:                    ~430M
    + Embeddings + Heads:           +5M
    + LocalByteDecoder:             +2M
    ─────────────────────────────
    Total aproximado:            ~437M  (rango 410-440M en práctica)

Tamaño en disco (2-bit packed final):
    Backbone ternary:    430M × 2 bits / 8  = ~107 MB
    Norms/scales (BF16): ~2 MB
    LocalDecoder (BF16 opcional): +1.5 MB
    ─────────────────────────────
    Inference checkpoint: ~110 MB (vs Q4 GGUF ~210 MB para 350M)

Optimizaciones para el kernel:
    - d_model=1280, d_ff=3328: ambos múltiplos de 64
    - n_layers=24 = 6 stages × 4 capas (pipeline mode 4 perfectamente balanceado)
    - pfnet_hidden=64: cabe en L1 (~512 KB total para 24 capas)
    - fp_rank=8: vector inner product de 8 elementos (1 instrucción AVX2)
    - LocalDecoder dim=192 = 3 × 64 (alignment SIMD limpio)

Throughput esperado (i7-8750H, mode 4 pipeline 6-stage calibrado):
    Por linealidad con 170M (40 MB → 521 pps full patch):
      ~107 MB / 40 MB = 2.7× más bytes por patch
      Estimado: 521 / 2.7 ≈ 195 pps full-patch
    En la práctica el pipeline puede compensar parcialmente — esperar 200-280 pps.

Input:  [B, S, 4] uint8 (patches dinámicos via Bolmo)
Output: 4-tuple de [B, S, 256] logits (MegaByte intra-patch AR)

Citas:
    - MatMul-free LM (Zhu et al., 2024)
    - BitNet b1.58 (Wang et al., 2024)
    - BLT / Bolmo (Pagnoni et al., 2024)
    - MrT5 (Pagliardini et al., 2024)
    - RWKV (Peng et al., 2023)
    - HGRN2 (Qin et al., 2024)
    - MegaByte (Yu et al., 2023)
    - Slender-Mamba (Yu et al., COLING 2025)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ablation import BASE_CONFIG, build_ablation_model, AblationModel

RUN_NAME = "run_400M_kernel_optimal"

# Take the BASE_CONFIG (training/data/loss settings) and override the
# architectural fields. We do NOT register this in RUN_CONFIGS — it's a
# standalone "extra" run, instantiable via build() below.
CONFIG = dict(BASE_CONFIG)
CONFIG.update({
    'name':                   'cpu1_400M_kernel_optimal',

    # ── Architecture: kernel-optimal 400M ────────────────────────────
    'd_model':                1280,    # multiple of 64, ≤ MAX_D_MODEL=2048
    'n_layers':               24,      # 24/6 = 4 layers per pipeline stage
    'd_ff':                   3328,    # multiple of 64, ≤ MAX_D_FF=5120
    'fp_rank':                8,
    'pfnet_hidden':           64,
    'patch_size':             4,

    # ── Token mixer & quantisation: full CPU-1 stack ────────────────
    'token_mixer':            'mlgru',
    'tokenization':           'byte_dynamic',
    'quantization':           'ternary',
    'fp_residual':            True,
    'use_boundary_predictor': True,
    'use_delete_gate':        True,
    'mlgru_per_channel_decay': True,    # RWKV/HGRN2 per-dim decay

    # ── MegaByte LocalByteDecoder: scaled up from 128 → 192 ─────────
    'use_local_decoder':      True,
    'local_decoder_dim':      192,      # 3 × 64
    'local_decoder_heads':    6,        # 192/6 = 32 head_dim
    'local_decoder_ff_ratio': 4.0,      # → d_ff_local = 768
    'local_decoder_fp16':     False,    # ternary as the rest of the stack

    # ── Training tweaks for 400M scale ──────────────────────────────
    'batch_per_gpu':               4,    # halve from 8 (2× memory pressure)
    'gradient_accumulation_steps': 4,    # double to keep effective batch
    'lr':                          2e-4, # smaller for larger model
    'min_lr':                      2e-5,
    'warmup_steps':                1000,
    'total_steps':                 20_000,

    'description': (
        'CPU-1 KERNEL-OPTIMAL 400M — escala completa con todos los '
        'componentes (MLGRU + per-channel decay + Bolmo + DeleteGate + '
        'PFNet + MegaByte LocalDecoder). Dimensiones múltiplos de 64 '
        'optimizadas para los kernels AVX2 2-bit MAD y para el pipeline '
        '6-stage del modo 4 (24 capas = 6×4 perfectamente balanceado).'
    ),
    'cites': [
        'MatMul-free LM (Zhu et al., 2024)',
        'BitNet b1.58 (Wang et al., 2024)',
        'BLT (Pagnoni et al., 2024)',
        'MrT5 (Pagliardini et al., 2024)',
        'RWKV (Peng et al., 2023)',
        'HGRN2 (Qin et al., 2024)',
        'MegaByte (Yu et al., 2023)',
        'Slender-Mamba (Yu et al., COLING 2025)',
    ],
})

ARCHITECTURE_SUMMARY = {
    "run":            RUN_NAME,
    "name":           CONFIG["name"],
    "token_mixer":    "MLGRU + per-channel decay (RWKV/HGRN2)",
    "tokenization":   "BytePatch dinámico (BoundaryPredictor / Bolmo)",
    "quantization":   "Ternary {-1,0,+1} (1.58 bit)",
    "fp_residual":    True,
    "fp_rank":        CONFIG["fp_rank"],
    "pfnet":          True,
    "pfnet_hidden":   CONFIG["pfnet_hidden"],
    "bolmo":          True,
    "delete_gate":    True,
    "delete_gate_pos": CONFIG["n_layers"] // 2,
    "local_decoder":  "MegaByte (d_local=192, n_heads=6, d_ff_local=768)",
    "d_model":        CONFIG["d_model"],
    "n_layers":       CONFIG["n_layers"],
    "d_ff":           CONFIG["d_ff"],
    "params_approx":  "~430M (target 400M)",
    "weight_bits":    1.58,
    "kernel_compat":  "AVX2 2-bit MAD (mode 4 pipeline 6-stage perfectamente balanceado)",
    "label":          "CPU-1 400M — production-size kernel-optimal",
}


def build() -> AblationModel:
    """Instanciar el modelo run_400M (CPU-1 KERNEL-OPTIMAL 400M)."""
    return build_ablation_model(CONFIG)


if __name__ == "__main__":
    import torch, json
    print(f"Config de {RUN_NAME} (CPU-1 KERNEL-OPTIMAL 400M):")
    print(json.dumps({k: v for k, v in CONFIG.items()
                      if k not in ("dataset_repo", "checkpoint_repo", "cites")},
                     indent=2))
    model = build()
    x = torch.randint(0, 256, (1, 32, 4))
    logits, states, extras = model(x)
    print(f"Logits[0] shape: {logits[0].shape}")
    print(f"BoundaryPredictor: {model.use_boundary}")
    print(f"DeleteGate: {model.use_delete_gate} (capa {model.delete_gate_pos})")
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"Backbone (ternary): {sum(p.numel() for n, p in model.named_parameters() if 'norm' not in n and 'head' not in n) * 2 / 8 / 1e6:.1f} MB (2-bit packed)")
