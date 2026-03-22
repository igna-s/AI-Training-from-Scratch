"""
Ablation experiment configuration for CPU-1 v2.

Design: 11 runs at 50M params on Kaggle 2×T4, shared dataset.
Variables under study:
  1. Token mixer:      Transformer (SlidingWindowAttention) vs MLGRU (MatMul-free)
  2. Tokenization:     BPE (~152K vocab) vs Byte-level (patches of 4 bytes)
  3. Weight quant:     FP16 vs Ternary 1.58-bit (BitLinear)
  4. Training signal:  CE only vs Distillation BPE→byte

Justified with citations (no runs needed):
  - BitEmbedding vs FP16 embedding:  Slender-Mamba (Yu et al., COLING 2025)
  - BitHead vs FP16 head:            Slender-Mamba (Yu et al., COLING 2025)
  - LUT vs INT8 matmul:              T-MAC (Wei et al., EuroSys 2025)
  - FPResidual paralelo:             MatMul-free LM (Zhu et al., 2024)
  - SwiGLU vs MLP:                   PaLM (Chowdhery et al., 2022)
  - RMSNorm vs LayerNorm:            LLaMA (Touvron et al., 2023)
  - Cosine LR schedule:              standard, multiple papers
  - KL + CE distillation loss:       Hinton et al., 2015
  - Temperature T=4.0:               Hinton et al., 2015
"""

# ============================================================
# Base config shared by ALL 11 runs (50M params)
# ============================================================

BASE_CONFIG = {
    # Architecture (50M params)
    'd_model':      512,
    'n_layers':     12,
    'd_ff':         1376,       # 8/3 × 512, rounded to multiple of 128
    'fp_rank':      4,          # FPResidual low-rank correction
    'n_heads':      8,          # for transformer runs
    'head_dim':     64,
    'pfnet_hidden': 0,          # disabled for 50M ablation

    # Training
    'batch_per_gpu':                8,
    'gradient_accumulation_steps':  2,
    'lr':                           3e-4,
    'min_lr':                       3e-5,
    'warmup_steps':                 500,
    'total_steps':                  10_000,
    'weight_decay_bitlinear':       0.1,
    'weight_decay_fpresidual':      0.0,
    'weight_decay_heads':           0.01,
    'grad_clip':                    1.0,
    'seed':                         42,

    # Distillation
    'temperature':  4.0,
    'alpha_kl':     0.9,
    'alpha_ce':     0.1,

    # Dataset
    'dataset_repo':     'Cukinator/cpu1-ablation-dataset',
    'seq_len':          2048,
    'patch_size':       4,

    # Checkpoints
    'checkpoint_repo':  'Cukinator/cpu1-ablation-checkpoints',
    'save_every_steps': 2000,
}

# ============================================================
# Per-run delta configs
# ============================================================
#
# Grid 2×2 (Mixer × Tokenization):
#
#               BPE                    Byte
# Transformer   Run 01, 03, 04         Run 02
# MLGRU         Run 05, 06, 07         Run 08, 09, 10, 11

RUN_CONFIGS = {
    'run_01': {
        'name':             'transformer_bpe_fp16_ce',
        'token_mixer':      'transformer',
        'tokenization':     'bpe',
        'quantization':     'fp16',
        'training_signal':  'ce_only',
        'fp_residual':      False,
        'description':      'Baseline absoluto — Transformer FP16 BPE sin distilación',
        'cites':            [],
    },
    'run_02': {
        'name':             'transformer_byte_fp16_ce',
        'token_mixer':      'transformer',
        'tokenization':     'byte',
        'quantization':     'fp16',
        'training_signal':  'ce_only',
        'fp_residual':      False,
        'description':      'Aislar efecto de byte-level sobre Transformer FP16',
        'cites':            ['MegaByte (Yu et al., 2023)'],
    },
    'run_03': {
        'name':             'transformer_bpe_ternary_ce',
        'token_mixer':      'transformer',
        'tokenization':     'bpe',
        'quantization':     'ternary',
        'training_signal':  'ce_only',
        'fp_residual':      True,
        'description':      'Aislar efecto de cuantización ternaria sobre transformer BPE',
        'cites':            ['BitNet b1.58 (Wang et al., 2024)',
                             'Slender-Mamba (Yu et al., 2025)'],
    },
    'run_04': {
        'name':             'transformer_bpe_ternary_distil',
        'token_mixer':      'transformer',
        'tokenization':     'bpe',
        'quantization':     'ternary',
        'training_signal':  'distillation',
        'fp_residual':      True,
        'description':      'Transformer completo con ternario y distilación',
        'cites':            ['BitNet b1.58', 'Hinton et al. 2015'],
    },
    'run_05': {
        'name':             'mlgru_bpe_fp16_ce',
        'token_mixer':      'mlgru',
        'tokenization':     'bpe',
        'quantization':     'fp16',
        'training_signal':  'ce_only',
        'fp_residual':      False,
        'description':      'Aislar efecto de MLGRU sobre BPE FP16',
        'cites':            ['MatMul-free LM (Zhu et al., 2024)'],
    },
    'run_06': {
        'name':             'mlgru_bpe_ternary_ce',
        'token_mixer':      'mlgru',
        'tokenization':     'bpe',
        'quantization':     'ternary',
        'training_signal':  'ce_only',
        'fp_residual':      True,
        'description':      'Aislar MLGRU + ternario sobre BPE',
        'cites':            ['MatMul-free LM', 'BitNet b1.58'],
    },
    'run_07': {
        'name':             'mlgru_bpe_ternary_distil',
        'token_mixer':      'mlgru',
        'tokenization':     'bpe',
        'quantization':     'ternary',
        'training_signal':  'distillation',
        'fp_residual':      True,
        'description':      'MLGRU BPE completo — comparación directa vs Run 10',
        'cites':            ['MatMul-free LM', 'BitNet b1.58', 'Hinton et al. 2015'],
    },
    'run_08': {
        'name':             'mlgru_byte_fp16_ce',
        'token_mixer':      'mlgru',
        'tokenization':     'byte',
        'quantization':     'fp16',
        'training_signal':  'ce_only',
        'fp_residual':      False,
        'description':      'Aislar byte-level sobre MLGRU FP16',
        'cites':            ['MegaByte (Yu et al., 2023)'],
    },
    'run_09': {
        'name':             'mlgru_byte_ternary_ce',
        'token_mixer':      'mlgru',
        'tokenization':     'byte',
        'quantization':     'ternary',
        'training_signal':  'ce_only',
        'fp_residual':      True,
        'description':      'CPU-1 sin distilación — aislar efecto del teacher signal',
        'cites':            [],
    },
    'run_10': {
        'name':             'mlgru_byte_ternary_distil_fpres',
        'token_mixer':      'mlgru',
        'tokenization':     'byte',
        'quantization':     'ternary',
        'training_signal':  'distillation',
        'fp_residual':      True,
        'description':      'CPU-1 COMPLETO con FPResidual — propuesta principal del paper',
        'cites':            [],
        'is_proposed':      True,
    },
    'run_11': {
        'name':             'mlgru_byte_ternary_distil_nofp',
        'token_mixer':      'mlgru',
        'tokenization':     'byte',
        'quantization':     'ternary',
        'training_signal':  'distillation',
        'fp_residual':      False,
        'description':      '100% ternario puro — sin FPResidual. '
                            'Run 10 vs 11 responde: ¿vale la pena la rama FP16?',
        'cites':            [
            'BitNet b1.58 (Wang et al., 2024)',
            'Slender-Mamba (Yu et al., 2025)',
        ],
    },
}


def get_run_config(run_name: str) -> dict:
    """Merge BASE_CONFIG with run-specific delta."""
    if run_name not in RUN_CONFIGS:
        raise ValueError(
            f"Unknown run: {run_name}. Available: {list(RUN_CONFIGS.keys())}"
        )
    run = RUN_CONFIGS[run_name]
    merged = {**BASE_CONFIG}
    merged['use_ternary'] = run['quantization'] == 'ternary'
    merged['fp_rank'] = BASE_CONFIG['fp_rank'] if run.get('fp_residual', True) else 0
    merged.update(run)
    return merged
