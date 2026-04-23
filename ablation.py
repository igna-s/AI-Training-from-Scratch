"""
Ablation study for CPU-1: configuration, models, and analysis.

Section 1 - Configuration: 10 run configs at 50M params on Lightning.ai T4.
Section 2 - Models: configurable model variants for all ablation runs.
Section 3 - Analysis: paper-ready tables (text + LaTeX).

All runs use CE-only training (no teacher) on raw text.
All runs use ternary weights or FP16 baseline for comparison.
Architecture constraint: 1.58-bit weights, no activation×activation matmul at inference.

Usage:
    from ablation import get_run_config, build_ablation_model, RUN_CONFIGS

    config = get_run_config("run_09")
    model = build_ablation_model(config)
"""

import json
import math
import argparse
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model import (
    RMSNorm,
    BitLinear,
    BitEmbedding,
    FPResidual,
    PFNet,
    BoundaryPredictor,
    DeleteGate,
    EmbeddingAligner,
    _ste_round,
    VOCAB_SIZE,
    PATCH_SIZE,
)


# =============================================================================
# Section 1: Configuration
# =============================================================================

# ============================================================
# Base config shared by ALL runs (50M params)
# ============================================================
#
# Architecture constraint:
#   - 1.58-bit ternary weights (BitLinear, BitEmbedding)
#   - No activation×activation matmul at inference (MLGRU uses Hadamard only)
#   - Transformer runs use sliding-window attention (local, CPU-friendly)
#   - FP16 runs included as reference baselines only
#
# Dataset: CE-only training on raw text (no teacher logprobs needed).
#   Current:  Cukinator/cpu1-ablation-dataset (~37.5M tokens, sequences pre-packed)
#   Future:   HuggingFaceFW/fineweb sample-10BT (1B+ tokens, requires train.py update)
# ============================================================

BASE_CONFIG = {
    # Architecture (50M params: d_model=512, n_layers=12)
    'd_model':      512,
    'n_layers':     12,
    'd_ff':         1376,       # 8/3 × 512, rounded to multiple of 128
    'fp_rank':      4,          # FPResidual low-rank correction (0 = disabled)
    'n_heads':      8,          # for transformer / folded-transformer runs
    'head_dim':     64,
    'window_size':  512,        # sliding-window attention span (in patches)
    'pfnet_hidden': 0,          # PFNet disabled for ablation runs

    # Training
    'batch_per_gpu':               8,
    'gradient_accumulation_steps': 2,
    'lr':                          3e-4,
    'min_lr':                      3e-5,
    'warmup_steps':                500,
    'total_steps':                 5_000,
    'weight_decay_bitlinear':      0.1,
    'weight_decay_fpresidual':     0.0,
    'weight_decay_heads':          0.01,
    'grad_clip':                   1.0,
    'seed':                        42,

    # Training signal — CE only for all ablation runs
    'training_signal': 'ce_only',

    # Loss hyperparameters (used by compute_loss in train.py)
    'temperature': 4.0,
    'alpha_kl':    0.9,
    'alpha_ce':    0.1,

    # Dataset
    'dataset_repo':     'Cukinator/cpu1-ablation-dataset',
    'seq_len':          5000,
    'patch_size':       4,

    # Byte head — MegaByte local decoder (intra-patch autoregression).
    # Conditions b_k on b_0..b_{k-1}; without it the 4 bytes are independent
    # which corrupts UTF-8 continuations and common bigrams.
    # Paper-canonical: h_proj outputs P × d_local, FFN ratio 4×.
    'use_local_decoder':      True,
    'local_decoder_dim':      128,
    'local_decoder_heads':    4,
    'local_decoder_ff_ratio': 4.0,     # paper: GELU FFN at 4×; SwiGLU-equivalent is 4×·2/3 ≈ 2.67× but 4× matches capacity
    'local_decoder_fp16':     False,   # opt-in: keep local decoder in FP16 even when global is ternary

    # Checkpoints
    'checkpoint_repo':  'Cukinator/cpu1-ablation-checkpoints',
    'save_every_steps': 1000,
}

# ============================================================
# Per-run delta configs — 10 progressive runs
# ============================================================
#
# Story: start from a standard FP16 Transformer baseline and add
# ONE CPU-1 design choice per run until we reach the full architecture.
# All runs: CE-only training from scratch, 50M params.
#
# Progressive chain:
#
#   run_01  Transformer + BPE + FP16            (absolute baseline)
#     ↓ swap tokenizer
#   run_02  Transformer + Byte + FP16           (+byte patches, same mixer)
#     ↓ swap mixer + keep byte
#   run_03  MLGRU + Byte + FP16                (+MatMul-free mixer AND byte)
#     ↓ quantize
#   run_04  MLGRU + Byte + Ternary             (+1.58-bit weights)
#     ↓ correct
#   run_05  MLGRU + Byte + Ternary + FPRes      (+FPResidual) ← CPU-1 core
#     ↓ branch: add components
#   run_06  CPU-1 core + Bolmo                  (+dynamic byte patches)
#   run_07  CPU-1 core + DeleteGate             (+state elimination)
#   run_08  Folded Transformer + Byte + Ternary (alternative: local attn)
#   run_09  CPU-1 core + Bolmo + DeleteGate     ← CPU-1 COMPLETE ⭐
#   run_10  CPU-1 COMPLETE + PFNet              ← +cache-resident MLP
#   run_11  CPU-1 COMPLETE + per-channel decay  ← +learnable forgetting per dim
#   run_12  CPU-1 COMPLETE + state reset        ← structural forgetting, 0 params
#   run_17  CPU-1 COMPLETE + TIDE exit          ← +per-token early exit (post-train)
#
# Key questions:
#   run_01 → run_02 : Does byte-level hurt a FP16 Transformer?
#   run_01 → run_03 : Transformer+BPE vs MLGRU+Byte — mixer AND tokenization combined
#   run_02 → run_03 : Does MLGRU improve over Transformer at byte level?
#   run_03 → run_04 : Does ternary hurt MLGRU+byte?
#   run_04 → run_05 : Does FPResidual recover ternary quality loss?
#   run_05 → run_06 : Does Bolmo improve over fixed patches?
#   run_05 → run_07 : Does DeleteGate preserve quality while pruning?
#   run_05 vs run_08 : CPU-1 core vs best local-attention Transformer
#   run_05 → run_09 : Full CPU-1 vs core — cost of all extras combined
#   run_09 → run_10 : Does PFNet improve quality without hurting inference speed?
#   run_09 → run_11 : Does per-channel learnable decay improve over fixed gate?
#   run_09 → run_12 : Does periodic state reset improve long-sequence generalization?
#   run_11 vs run_12 : Learned dissipation vs structural forgetting — which wins?
#   run_09 → run_17 : Does TIDE early exit reduce CPU-1 FLOPs without quality loss?

RUN_CONFIGS = {
    # ------------------------------------------------------------------
    # run_01: absolute baseline
    # ------------------------------------------------------------------
    'run_01': {
        'name':         'transformer_bpe_fp16',
        'token_mixer':  'transformer',
        'tokenization': 'bpe',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  'Absolute baseline — FP16 Transformer, BPE tokens, CE-only.',
        'cites':        [],
    },

    # ------------------------------------------------------------------
    # run_02: +byte patches (swap tokenizer)
    # ------------------------------------------------------------------
    'run_02': {
        'name':         'transformer_byte_fp16',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  '+Byte patches — FP16 Transformer with 4-byte patches. Isolates tokenization.',
        'cites':        ['MegaByte (Yu et al., 2023)'],
    },

    # ------------------------------------------------------------------
    # run_03: MLGRU + byte (both components together)
    # ------------------------------------------------------------------
    'run_03': {
        'name':         'mlgru_byte_fp16',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  'MLGRU + byte patches, FP16. Combined mixer+tokenization.',
        'cites':        ['MatMul-free LM', 'MegaByte (Yu et al., 2023)'],
    },

    # ------------------------------------------------------------------
    # run_04: +ternary weights
    # ------------------------------------------------------------------
    'run_04': {
        'name':         'mlgru_byte_ternary',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  False,
        'description':  '+Ternary — 1.58-bit weights on MLGRU+byte. Isolates quantization cost.',
        'cites':        ['BitNet b1.58 (Wang et al., 2024)', 'MatMul-free LM'],
    },

    # ------------------------------------------------------------------
    # run_05: +FPResidual → CPU-1 core architecture
    # ------------------------------------------------------------------
    'run_05': {
        'name':         'mlgru_byte_ternary_fpres',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  True,
        'description':  (
            '+FPResidual — low-rank FP16 correction alongside ternary weights. '
            'run_04 → run_05 answers: does FPResidual recover ternary quality loss? '
            'This is the CPU-1 core architecture.'
        ),
        'cites':        ['Slender-Mamba (Yu et al., 2025)', 'BitNet b1.58'],
    },

    # ------------------------------------------------------------------
    # run_06: CPU-1 core + Bolmo (dynamic byte patches)
    # ------------------------------------------------------------------
    'run_06': {
        'name':         'mlgru_byte_ternary_fpres_bolmo',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'description':  (
            '+Bolmo — dynamic byte patches aligned to word boundaries. '
            'CPU-1 core + boundary predictor. run_05 → run_06: fixed vs dynamic patches.'
        ),
        'cites':        ['BLT (Pagnoni et al., 2024)'],
    },

    # ------------------------------------------------------------------
    # run_07: CPU-1 core + DeleteGate (state elimination)
    # ------------------------------------------------------------------
    'run_07': {
        'name':         'mlgru_byte_ternary_fpres_deletegate',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_delete_gate': True,
        'description':  (
            '+DeleteGate — MrT5-style state elimination at midpoint (~40%). '
            'CPU-1 core + delete gate. run_05 → run_07: quality vs speed tradeoff.'
        ),
        'cites':        ['MrT5 (Pagliardini et al., 2024)'],
    },

    # ------------------------------------------------------------------
    # run_08: alternative — Folded Transformer (local sliding-window)
    # Not part of the build-up chain; compares a CPU-friendly attention
    # baseline against the CPU-1 core (run_05) at the same parameter budget.
    # window_size=128 patches = 512 bytes of context per position.
    # ------------------------------------------------------------------
    'run_08': {
        'name':         'folded_transformer_byte_ternary',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  False,
        'window_size':  128,
        'description':  (
            'Folded Transformer — ternary sliding-window attention over byte patches. '
            'O(N·128) vs O(N²). Alternative to MLGRU. '
            'run_05 vs run_08: MLGRU core vs best local-attention Transformer.'
        ),
        'cites':        [
            'Longformer (Beltagy et al., 2020)',
            'BitNet b1.58 (Wang et al., 2024)',
        ],
    },

    # ------------------------------------------------------------------
    # run_09: CPU-1 COMPLETE — all components together
    # ------------------------------------------------------------------
    'run_09': {
        'name':         'cpu1_complete',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate': True,
        'description':  (
            'CPU-1 COMPLETE — MLGRU + Bolmo (dynamic byte patches) + ternary + '
            'FPResidual + DeleteGate. Full proposed architecture.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'Slender-Mamba (Yu et al., 2025)',
        ],
        'is_proposed':  True,
    },

    # ------------------------------------------------------------------
    # run_10: CPU-1 COMPLETE + PFNet (cache-resident nonlinear residual)
    # ------------------------------------------------------------------
    # PFNet adds a small 2-layer ternary MLP (pfnet_hidden=32) as a residual
    # branch per block. At 50M (d_model=512, 12 layers):
    #   params: 12 × 2 × 512 × 32 = 393K — fits entirely in L2 cache (~786KB BF16)
    # Key property: PFNet weights are L2-resident alongside the MLGRU state,
    # so they don't add RAM streaming cost. Near-zero latency overhead.
    # MLGRU mixing h_t = f*h + (1-f)*c is a linear interpolation — PFNet adds
    # a direct nonlinear pathway that MLGRU structurally cannot express.
    # Question: does PFNet improve quality without hurting inference speed?
    # ------------------------------------------------------------------
    'run_10': {
        'name':         'cpu1_pfnet',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'pfnet_hidden': 32,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'description':  (
            'CPU-1 COMPLETE + PFNet (pfnet_hidden=32). '
            'Cache-resident nonlinear residual — quality gain, zero RAM overhead.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'Slender-Mamba (Yu et al., 2025)',
        ],
    },

    # ------------------------------------------------------------------
    # run_11: CPU-1 COMPLETE + Per-Channel Learnable Decay (RWKV / HGRN2-style)
    # ------------------------------------------------------------------
    # MLGRU's forget gate f is a scalar shared across all hidden dims.
    # This run adds d_model learnable log-decay scalars (one per channel):
    #   decay[i] = exp(-softplus(log_decay[i]))   ∈ (0, 1)
    #   h_t[i]   = decay[i] * f_t * h_{t-1}[i] + (1 - decay[i] * f_t) * c_t[i]
    # Bottom-layer channels can learn fast decay (local context), top-layer
    # channels slow decay (long-range dependencies) — emerging automatically.
    # Parameter cost: n_layers × d_model scalars = 12 × 512 = 6144 floats (~12KB).
    # Inference cost: one element-wise multiply per step — negligible.
    # Question: does per-channel memory dissipation improve quality vs fixed f?
    # Inspired by: RWKV time-decay W[i], HGRN2 layer-wise decay bounds.
    # ------------------------------------------------------------------
    'run_11': {
        'name':         'cpu1_decay_learned',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'mlgru_per_channel_decay': True,   # adds n_layers × d_model log-decay params
        'description':  (
            'CPU-1 COMPLETE + per-channel learnable decay in MLGRU (RWKV/HGRN2-style). '
            'Each hidden dim has its own decay rate — d_model × n_layers scalars total (~12KB). '
            'run_09 → run_11: does data-driven per-channel forgetting improve quality?'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'RWKV (Peng et al., 2023)',
            'HGRN2 (Qin et al., 2024)',
        ],
    },

    # ------------------------------------------------------------------
    # run_12: CPU-1 COMPLETE + Periodic State Reset (Stuffed Mamba / zero-param)
    # ------------------------------------------------------------------
    # "Stuffed Mamba" (arXiv 2410.07145) shows that oversized recurrent states
    # accumulate unbounded context, causing the model to "forget" to forget —
    # degrading performance on long sequences. The fix: periodic state reset.
    # Implementation: every `state_reset_period` tokens, multiply h_t by a
    # cosine-decayed scalar → 0 (full reset). Zero new parameters.
    # At reset_period=512 (= 2 × patch_size × context_len): the model sees a
    # clean slate at document boundaries, preventing state bleed across docs.
    # Question: does structural forgetting at fixed intervals help vs run_09?
    #   If yes → long-sequence quality matters even at 50M
    #   If no  → state accumulation is benign at this scale
    # ------------------------------------------------------------------
    'run_12': {
        'name':         'cpu1_decay_reset',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'state_reset_period': 512,   # reset h_t to 0 every N tokens (zero params)
        'description':  (
            'CPU-1 COMPLETE + periodic MLGRU state reset every 512 tokens (Stuffed Mamba-style). '
            'Zero extra parameters — pure structural forgetting heuristic. '
            'run_09 → run_12: does forced forgetting at doc boundaries improve generalization?'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'Stuffed Mamba (anonymous, 2024)',
        ],
    },

    # ------------------------------------------------------------------
    # run_17: CPU-1 COMPLETE + TIDE per-token early exit (post-training)
    # ------------------------------------------------------------------
    # TIDE (Token-Informed Depth Execution, arXiv 2603.21365) observes that most
    # tokens "converge" well before the final layer — their representation stops
    # changing significantly. A tiny 2-layer MLP classifier per block (d_model →
    # 64 → 1) is calibrated post-training on 2K wikitext samples (< 3 min on GPU)
    # to predict "has this token converged?" NO retraining of the base model.
    # At inference: converged tokens skip remaining layers → lower FLOPs, same
    # model weights, same quality (99% decode quality on DeepSeek R1 Distill 8B).
    #
    # For CPU-1 @ 50M (d_model=512, 12 layers):
    #   Exit classifiers: 12 × (512 × 64 + 64 + 64 + 1) ≈ 394K params (~800KB)
    #   Calibration only — these params are NOT trained via backprop with the model
    #   Expected: most "the", "of", "." tokens exit at layer 4-6 → 40-50% FLOPs saved
    #
    # Question: does TIDE early exit reduce CPU-1 inference FLOPs while preserving
    # byte-level perplexity? Especially relevant for CPU deployment where every
    # layer saves ~3ms at 50M scale.
    # ------------------------------------------------------------------
    'run_17': {
        'name':         'cpu1_tide_exit',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'use_tide_exit':          True,   # per-layer exit classifiers, post-training calibration
        'tide_bottleneck':        64,     # hidden dim of exit MLP (d_model → 64 → 1)
        'tide_threshold':         0.9,    # cosine-sim threshold for "converged" decision
        'description':  (
            'CPU-1 COMPLETE + TIDE per-token early exit (TIDE, arXiv 2603.21365). '
            'Tiny post-training MLP classifiers (~800KB) per layer predict token convergence. '
            'No base model retraining — calibrated on 2K samples. '
            'run_09 → run_17: does adaptive depth reduce CPU-1 FLOPs without quality loss?'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'TIDE (Jaber et al., 2026)',
        ],
    },
}


# ============================================================
# Bonus: 10M parameter runs — fast scaling-consistency check
# ============================================================
#
# Why 10M?
#   - Trains in ~1h on T4 (vs 4h for 50M) — fast iteration
#   - BPE vocab (152K tokens) can't fit in 10M params: embedding alone = 39M
#     → byte-level only at this scale
#   - Mirrors the key 50M decisions to verify trends are scale-consistent
#   - Paper can show: same ordering holds at 10M and 50M → trends generalize
#
# All three small_ runs use the CPU-1 COMPLETE architecture (MLGRU + ternary +
# FPResidual + Bolmo + DeleteGate). Only the TRAINING STRATEGY varies.
#
# Key question: which data strategy produces the best byte-level model?
#   run_13  CPU-1 COMPLETE + compact BPE (4K vocab) from scratch
#   run_14  CPU-1 COMPLETE + byte from scratch
#   run_15  CPU-1 COMPLETE + complement (BPE warm-start → byte finetune)
#   run_16  CPU-1 COMPLETE + raw bytes, no teacher
#
# BPE vocab constraint: full 152K Qwen vocab → 152K × 320 = 48.7M params, larger
#   than the entire 10M model. run_13 uses top-4096 tokens by frequency from
#   cpu1-ablation-dataset (bpe_tokenized/ split). Embedding: 4096 × 320 ≈ 1.3M params.
#
# Architecture: d_model=320, n_layers=8, d_ff=853 → ~10M params
#   MLGRU per block:  3 × 320² = 307K
#   SwiGLU per block: 3 × 320 × 853 = 819K
#   8 blocks: 9.0M | embedding+heads: 0.35M | total ≈ 10M
#
# Training: 2500 steps each (~1h on T4, 200M effective tokens).
#   run_15 starts from run_13 checkpoint and continues for 2500 byte steps —
#   same byte-step budget as run_14, but with BPE warm-start on top.
#
# "Complement" mode: run 10M first to find best design choices in <1h each,
#   then only scale up the winners to 50M — saves ~3h per eliminated run.

SMALL_BASE_CONFIG = {
    **BASE_CONFIG,
    # Architecture (~10M params)
    'd_model':      320,
    'n_layers':     8,
    'd_ff':         853,        # 8/3 × 320 = 853
    'fp_rank':      4,
    'n_heads':      8,
    'head_dim':     40,         # 320 / 8
    'window_size':  256,

    # Shorter training — 10M converges faster
    'total_steps':  2500,
    'warmup_steps': 250,
}

SMALL_RUN_CONFIGS = {
    # ------------------------------------------------------------------
    # run_13: CPU-1 COMPLETE + compact BPE (4K vocab) from scratch
    # ------------------------------------------------------------------
    # BPE at 10M: uses top-4096 tokens by frequency from cpu1-ablation-dataset
    # (bpe_tokenized/ split). Full 152K Qwen vocab is impossible at this scale.
    # BoundaryPredictor disabled — BPE tokens carry their own subword boundaries.
    # ------------------------------------------------------------------
    'run_13': {
        'name':                   'small_cpu1_bpe',
        'token_mixer':            'mlgru',
        'tokenization':           'bpe',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': False,   # BPE has its own boundaries
        'use_delete_gate':        True,
        'bpe_vocab_size':         4096,    # compact — top-4096 by frequency
        'description':            'CPU-1 COMPLETE @ 10M, compact BPE (4K vocab) from scratch. '
                                  'Does BPE tokenization outperform byte-level at 10M?',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'MrT5'],
    },

    # ------------------------------------------------------------------
    # run_14: CPU-1 COMPLETE + byte-level from scratch
    # ------------------------------------------------------------------
    'run_14': {
        'name':                   'small_cpu1_byte',
        'token_mixer':            'mlgru',
        'tokenization':           'byte_dynamic',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'description':            'CPU-1 COMPLETE @ 10M, byte-level from scratch. '
                                  'Pure byte baseline for complement comparison.',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'Bolmo', 'MrT5'],
        'is_proposed':            True,
    },

    # ------------------------------------------------------------------
    # run_15: CPU-1 COMPLETE + complement (BPE warm-start → byte finetune)
    # ------------------------------------------------------------------
    # Starts from run_13 checkpoint. The BPE embedding layer is discarded and
    # replaced with a fresh BytePatchEmbeddingV2; all other weights carry over.
    # Then continues training on byte data for 2500 steps (same byte budget as
    # run_14). Inspired by Bolmo's two-stage subword→byte byteification.
    # ------------------------------------------------------------------
    'run_15': {
        'name':                   'small_cpu1_complement',
        'token_mixer':            'mlgru',
        'tokenization':           'byte_dynamic',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'complement_from':        'run_13',   # load weights, swap embedding
        'description':            'CPU-1 COMPLETE @ 10M, complement: BPE warm-start (run_13) '
                                  '→ byte finetune. Does BPE pretraining bootstrap byte quality?',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'Bolmo', 'MrT5'],
    },

    # ------------------------------------------------------------------
    # run_16: CPU-1 COMPLETE + raw bytes, NO teacher (pure CE, no Qwen)
    # ------------------------------------------------------------------
    # Streams raw UTF-8 bytes directly from HuggingFaceFW/fineweb — no teacher
    # forward pass, no logprobs, no teacher_mask. Pure CE loss on byte patches.
    # Answers: do Qwen teacher logprobs (in run_14) actually help, or does
    # training on raw bytes alone reach the same quality?
    # Uses a separate dataset path (raw fineweb bytes) so it's truly independent
    # of the cpu1-ablation-dataset pipeline.
    # ------------------------------------------------------------------
    'run_16': {
        'name':                   'small_cpu1_raw_bytes',
        'token_mixer':            'mlgru',
        'tokenization':           'byte_dynamic',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'raw_bytes_only':         True,    # skip teacher columns, stream raw fineweb bytes
        'dataset_repo':           'HuggingFaceFW/fineweb',
        'description':            'CPU-1 COMPLETE @ 10M, raw bytes from scratch, zero teacher. '
                                  'Do Qwen logprobs in run_14 actually help vs pure CE on raw bytes?',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'Bolmo', 'MrT5'],
    },
}

# Merged registry — both 50M and 10M runs
ALL_RUN_CONFIGS = {**RUN_CONFIGS, **SMALL_RUN_CONFIGS}


def get_run_config(run_name: str) -> dict:
    """Merge base config with run-specific delta. Handles both 50M and 10M runs."""
    if run_name not in ALL_RUN_CONFIGS:
        raise ValueError(
            f"Unknown run: {run_name}. Available: {list(ALL_RUN_CONFIGS.keys())}"
        )
    run = ALL_RUN_CONFIGS[run_name]
    # 10M runs (run_13+) use SMALL_BASE_CONFIG, 50M runs (run_01-run_12) use BASE_CONFIG
    base = SMALL_BASE_CONFIG if run_name in SMALL_RUN_CONFIGS else BASE_CONFIG
    merged = {**base}
    merged['use_ternary'] = run['quantization'] == 'ternary'
    fp_residual = run.get('fp_residual', False)
    merged['fp_rank'] = base['fp_rank'] if (fp_residual and run['quantization'] == 'ternary') else 0
    # Optional architecture flags — default off unless run specifies them
    merged['use_boundary_predictor'] = run.get('use_boundary_predictor', False)
    merged['use_delete_gate'] = run.get('use_delete_gate', False)
    merged['use_embedding_alignment'] = run.get('use_embedding_alignment', False)
    merged.update(run)
    return merged


# =============================================================================
# Section 2: Configurable Models
# =============================================================================
# ============================================================
# Factory helpers — select FP16 or ternary components
# ============================================================

def make_linear(in_features: int, out_features: int,
                use_ternary: bool = True) -> nn.Module:
    if use_ternary:
        return BitLinear(in_features, out_features)
    linear = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(linear.weight, std=0.02)
    return linear


def make_embedding(vocab_size: int, embed_dim: int,
                   use_ternary: bool = True) -> nn.Module:
    if use_ternary:
        return BitEmbedding(vocab_size, embed_dim)
    return nn.Embedding(vocab_size, embed_dim)


# ============================================================
# Configurable SwiGLU FFN
# ============================================================

class AblationSwiGLU(nn.Module):
    """SwiGLU FFN, configurable for FP16 or ternary weights."""

    def __init__(self, d_model: int, d_ff: int,
                 use_ternary: bool = True, fp_rank: int = 4):
        super().__init__()
        self.gate_proj = make_linear(d_model, d_ff, use_ternary)
        self.up_proj = make_linear(d_model, d_ff, use_ternary)
        self.down_proj = make_linear(d_ff, d_model, use_ternary)
        self.fp_residual = (
            FPResidual(d_ff, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        out = self.down_proj(hidden)
        if self.fp_residual is not None:
            out = out + self.fp_residual(hidden)
        return out


# ============================================================
# Configurable byte-patch embedding
# ============================================================

class AblationBytePatchEmbedding(nn.Module):
    """4-byte patch embedding, configurable for FP16 or ternary."""

    def __init__(self, d_model: int, patch_size: int = 4,
                 bytes_per_embed: int = 8, use_ternary: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.bytes_per_embed = bytes_per_embed
        self.byte_embed = make_embedding(256, bytes_per_embed, use_ternary)
        self.patch_proj = make_linear(
            patch_size * bytes_per_embed, d_model, use_ternary
        )

    def forward(self, patches: Tensor) -> Tensor:
        """patches: [B, S, 4] uint8/long -> [B, S, d_model]"""
        B, S, P = patches.shape
        embeds = self.byte_embed(patches.long())
        embeds = embeds.reshape(B, S, P * self.bytes_per_embed)
        return self.patch_proj(embeds)


# ============================================================
# Configurable MLGRU token mixer
# ============================================================

class AblationMLGRU(nn.Module):
    """MatMul-free token mixer (MLGRU), configurable quantization."""

    def __init__(self, d_model: int, use_ternary: bool = True,
                 fp_rank: int = 4):
        super().__init__()
        self.d_model = d_model
        self.forget_proj = make_linear(d_model, d_model, use_ternary)
        self.cand_proj = make_linear(d_model, d_model, use_ternary)
        self.output_proj = make_linear(d_model, d_model, use_ternary)
        self.fp_residual = (
            FPResidual(d_model, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )
        self.register_buffer('h_init', torch.zeros(d_model))

    def forward(self, x: Tensor, h: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, S, D = x.shape
        if h is None:
            h = self.h_init.unsqueeze(0).expand(B, -1).clone()

        f = torch.sigmoid(self.forget_proj(x))
        c = F.silu(self.cand_proj(x))
        g = torch.sigmoid(self.output_proj(x))

        # torch.lerp(c, h, f) = f*h + (1-f)*c — una sola op vectorizada, sin temporales
        outputs = torch.empty_like(f)  # [B, S, D] pre-alocado
        for t in range(S):
            h = torch.lerp(c[:, t], h, f[:, t])
            outputs[:, t] = g[:, t] * h

        out = outputs  # ya es [B, S, D], sin torch.stack
        if self.fp_residual is not None:
            out = out + self.fp_residual(x)
        return out, h


# ============================================================
# Sliding-window causal attention
# ============================================================

class SlidingWindowAttention(nn.Module):
    """Multi-head causal self-attention with optional sliding window."""

    def __init__(self, d_model: int, n_heads: int = 8, head_dim: int = 64,
                 window_size: int = 512, use_ternary: bool = True,
                 fp_rank: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        inner_dim = n_heads * head_dim

        self.q_proj = make_linear(d_model, inner_dim, use_ternary)
        self.k_proj = make_linear(d_model, inner_dim, use_ternary)
        self.v_proj = make_linear(d_model, inner_dim, use_ternary)
        self.o_proj = make_linear(inner_dim, d_model, use_ternary)
        self.fp_residual = (
            FPResidual(d_model, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Build sliding-window causal mask
        # Each position attends to at most `window_size` preceding positions
        if S <= self.window_size:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            band = torch.ones(S, S, dtype=torch.bool, device=x.device).triu(
                diagonal=-(self.window_size - 1)
            )
            attn_mask = mask & band
            # SDPA expects float mask with -inf for masked positions
            float_mask = torch.zeros(S, S, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~attn_mask, float('-inf'))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        result = self.o_proj(out)
        if self.fp_residual is not None:
            result = result + self.fp_residual(x)
        return result


# ============================================================
# Configurable block (MLGRU or Transformer)
# ============================================================

class AblationBlock(nn.Module):
    """Unified block: pre-norm → mixer → residual → pre-norm → FFN → residual."""

    def __init__(self, config: dict):
        super().__init__()
        d_model = config['d_model']
        d_ff = config['d_ff']
        use_ternary = config.get('use_ternary', True)
        fp_rank = config.get('fp_rank', 4)
        pfnet_hidden = config.get('pfnet_hidden', 0)

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.ffn = AblationSwiGLU(d_model, d_ff, use_ternary, fp_rank)

        mixer = config.get('token_mixer', 'mlgru')
        if mixer == 'mlgru':
            self.mixer_type = 'mlgru'
            self.token_mixer = AblationMLGRU(d_model, use_ternary, fp_rank)
        else:
            self.mixer_type = 'transformer'
            self.token_mixer = SlidingWindowAttention(
                d_model,
                n_heads=config.get('n_heads', 8),
                head_dim=config.get('head_dim', 64),
                window_size=config.get('window_size', 512),
                use_ternary=use_ternary,
                fp_rank=fp_rank,
            )

        self.pfnet = (
            PFNet(d_model, pfnet_hidden) if pfnet_hidden > 0 else None
        )

    def forward(self, x: Tensor, h: Tensor | None = None
                ) -> tuple[Tensor, Tensor | None]:
        residual = x
        if self.mixer_type == 'mlgru':
            mixer_out, h_new = self.token_mixer(self.norm1(x), h)
        else:
            mixer_out = self.token_mixer(self.norm1(x))
            h_new = None
        x = residual + mixer_out

        residual = x
        x = residual + self.ffn(self.norm2(x))
        if self.pfnet is not None:
            x = x + self.pfnet(residual)
        return x, h_new

    def forward_one_token(self, x: Tensor, h: Tensor | None = None
                          ) -> tuple[Tensor, Tensor | None]:
        """Decode-time shortcut — x is [B, d_model] (no sequence dim).

        Avoids unsqueeze/squeeze overhead in the per-token loop in inference.py
        for all 14 blocks × every generated patch.
        """
        x_3d = x.unsqueeze(1)
        out_3d, h_new = self.forward(x_3d, h)
        return out_3d.squeeze(1), h_new


# ============================================================
# Prediction heads
# ============================================================

class AblationByteHead(nn.Module):
    """Single byte prediction head (over 256 classes)."""

    def __init__(self, d_model: int, use_ternary: bool = True):
        super().__init__()
        self.proj = make_linear(d_model, 256, use_ternary)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x) * self.scale


class AblationBPEHead(nn.Module):
    """BPE prediction head (over tokenizer vocab)."""

    def __init__(self, d_model: int, vocab_size: int,
                 use_ternary: bool = True):
        super().__init__()
        self.proj = make_linear(d_model, vocab_size, use_ternary)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# ============================================================
# MegaByte-style Local Byte Decoder (intra-patch autoregression)
# ============================================================

class LocalByteDecoder(nn.Module):
    """MegaByte (Yu et al., 2023, Listing 1) local model for byte-level prediction.

    Paper-canonical wiring:
      - Global state h_G projected to P * d_local via a single BitLinear, then
        reshaped so each of the P positions inside the patch receives its own
        slice of the global context (not a broadcast copy).
      - Byte input at patch position k is the shift-right byte embedding:
        position 0 sees a zero-padded byte, positions 1..P-1 see emb(b_{k-1}).
      - Position-specific global slice + shifted byte embedding + learned
        position embedding feed a causal transformer block over length P.

    Training: teacher-forced, parallel over P positions with a causal P×P mask.
    Inference: call generate_step once per byte in the patch.

    FP16 override: passing use_ternary=False keeps all local-decoder weights in
    FP16 while the rest of the model stays ternary. Costs ~420 KB extra vs
    ternary at d_local=128; removes quantization noise from a narrow (d=128)
    shallow (1 layer, seq_len=P) network where the noise hurts most.
    """

    def __init__(self, d_model: int, patch_size: int = 4,
                 d_local: int = 128, n_heads: int = 4,
                 ff_ratio: float = 4.0,
                 use_ternary: bool = True):
        super().__init__()
        assert d_local % n_heads == 0, "d_local must be divisible by n_heads"
        self.patch_size = patch_size
        self.d_local = d_local
        self.n_heads = n_heads
        self.head_dim = d_local // n_heads

        # Paper Listing 1: project global to P × d_local so each of the P patch
        # positions receives its own global slice, not a broadcast copy.
        self.h_proj = make_linear(d_model, patch_size * d_local, use_ternary)
        self.byte_embed = make_embedding(256, d_local, use_ternary)
        self.pos_embed = nn.Parameter(torch.randn(patch_size, d_local) * 0.02)

        self.norm1 = RMSNorm(d_local)
        self.q_proj = make_linear(d_local, d_local, use_ternary)
        self.k_proj = make_linear(d_local, d_local, use_ternary)
        self.v_proj = make_linear(d_local, d_local, use_ternary)
        self.o_proj = make_linear(d_local, d_local, use_ternary)

        self.norm2 = RMSNorm(d_local)
        d_ff_local = int(d_local * ff_ratio)
        self.ffn_gate = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_up = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_down = make_linear(d_ff_local, d_local, use_ternary)

        self.out_proj = make_linear(d_local, 256, use_ternary)
        self.out_scale = nn.Parameter(torch.ones(1))

    def _attn_ffn(self, x: Tensor) -> Tensor:
        """Causal self-attention + SwiGLU FFN over the patch axis. x: [N, L, d_local]."""
        N, L, D = x.shape
        residual = x
        y = self.norm1(x)
        q = self.q_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        x = residual + self.o_proj(y)

        residual = x
        y = self.norm2(x)
        y = self.ffn_down(F.silu(self.ffn_gate(y)) * self.ffn_up(y))
        return residual + y

    def forward(self, h: Tensor, targets: Tensor) -> tuple:
        """Teacher-forced training/eval forward.

        Args:
            h:       [B, S, d_model] global patch-level hidden state
            targets: [B, S, P] long — ground-truth bytes of the predicted patch
        Returns:
            tuple of P tensors, each [B, S, 256] — logits for byte_0 .. byte_{P-1}
        """
        B, S, _ = h.shape
        P = self.patch_size

        # Paper Listing 1: project to [B, S, P, d_local] — one slice per position.
        g = self.h_proj(h).reshape(B, S, P, self.d_local)

        # Shift-right byte targets: position 0 sees a zero-padded byte;
        # position k > 0 sees emb(target[k-1]).
        zero_pad = torch.zeros_like(targets[..., :1])
        shifted = torch.cat([zero_pad, targets[..., :-1]], dim=-1)      # [B, S, P]
        byte_embs = self.byte_embed(shifted.long())                     # [B, S, P, d_local]

        local_in = g + byte_embs + self.pos_embed                       # [B, S, P, d_local]
        local_in = local_in.reshape(B * S, P, self.d_local)
        local_out = self._attn_ffn(local_in)
        local_out = local_out.reshape(B, S, P, self.d_local)

        logits = self.out_proj(local_out) * self.out_scale              # [B, S, P, 256]
        return tuple(logits[:, :, k, :] for k in range(P))

    def generate_step(self, h: Tensor, generated: Tensor) -> Tensor:
        """Single-byte prediction within a patch (inference).

        Args:
            h:         [B, d_model] global state for the current patch
            generated: [B, k] long — bytes already sampled in this patch (k in 0..P-1)
        Returns:
            logits: [B, 256] for the next byte at position k
        """
        B = h.shape[0]
        k = generated.shape[1]
        P = self.patch_size

        g_full = self.h_proj(h).reshape(B, P, self.d_local)              # [B, P, d_local]
        g = g_full[:, :k + 1, :]                                         # [B, k+1, d_local]

        zero_pad = torch.zeros(B, 1, dtype=torch.long, device=h.device)
        shifted = torch.cat([zero_pad, generated.long()], dim=1)         # [B, k+1]
        byte_embs = self.byte_embed(shifted)                             # [B, k+1, d_local]

        local_in = g + byte_embs + self.pos_embed[:k + 1]                # [B, k+1, d_local]
        local_out = self._attn_ffn(local_in)
        return self.out_proj(local_out[:, -1, :]) * self.out_scale


# ============================================================
# Unified ablation model
# ============================================================

class AblationModel(nn.Module):
    """
    Configurable model for all 15 ablation runs.

    Determined by config:
        token_mixer:  'transformer' | 'mlgru'
        tokenization: 'bpe' | 'byte' | 'byte_dynamic'
        use_ternary:  True | False
        fp_rank:      0 (no FPResidual) | 4 (with FPResidual)

    Extended for runs 12-15:
        use_boundary_predictor:  True → BoundaryPredictor (Bolmo)
        use_delete_gate:         True → DeleteGate (MrT5)
        use_embedding_alignment: True → EmbeddingAligner (arXiv 2602)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        n_layers = config['n_layers']
        tok = config['tokenization']
        use_ternary = config.get('use_ternary', True)

        # Embedding — byte_dynamic uses same embedding as byte
        if tok in ('byte', 'byte_dynamic'):
            self.embedding = AblationBytePatchEmbedding(
                d_model,
                patch_size=config.get('patch_size', 4),
                use_ternary=use_ternary,
            )
        else:
            vocab_size = config.get('bpe_vocab_size', 152064)  # Qwen2.5 default
            self.embedding = make_embedding(vocab_size, d_model, use_ternary)

        # Transformer / MLGRU blocks
        self.blocks = nn.ModuleList([
            AblationBlock(config) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Prediction heads
        #   Byte runs: by default use MegaByte-style LocalByteDecoder so byte k
        #   is conditioned on bytes 0..k-1 within the patch. Setting
        #   config['use_local_decoder']=False falls back to 4 independent heads
        #   (kept for tightly-scoped ablations of the decoder itself).
        self.use_local_decoder = (
            tok in ('byte', 'byte_dynamic')
            and config.get('use_local_decoder', True)
        )
        if self.use_local_decoder:
            # FP16 override: if local_decoder_fp16 is set, keep the whole local
            # decoder in FP16 even when the global model is ternary. Trades
            # ~420 KB extra RAM for lower quantization noise in a narrow (d=128),
            # shallow (1 layer, seq_len=P) submodule where the noise bites most.
            local_ternary = use_ternary and not config.get('local_decoder_fp16', False)
            self.local_decoder = LocalByteDecoder(
                d_model=d_model,
                patch_size=config.get('patch_size', 4),
                d_local=config.get('local_decoder_dim', 128),
                n_heads=config.get('local_decoder_heads', 4),
                ff_ratio=config.get('local_decoder_ff_ratio', 4.0),
                use_ternary=local_ternary,
            )
            self.heads = None
        elif tok in ('byte', 'byte_dynamic'):
            self.heads = nn.ModuleList([
                AblationByteHead(d_model, use_ternary) for _ in range(4)
            ])
        else:
            vocab_size = config.get('bpe_vocab_size', 152064)
            self.heads = nn.ModuleList([
                AblationBPEHead(d_model, vocab_size, use_ternary)
            ])

        self.tokenization = tok
        self.use_gradient_checkpointing = config.get(
            'use_gradient_checkpointing', False
        )

        # === Extended modules for runs 12-15 ===
        self.use_boundary    = config.get('use_boundary_predictor', False)
        self.use_delete_gate = config.get('use_delete_gate', False)
        self.use_emb_align   = config.get('use_embedding_alignment', False)
        self.delete_gate_pos = n_layers // 2

        if self.use_boundary:
            self.boundary_predictor = BoundaryPredictor(
                embed_dim=8, hidden_dim=32
            )

        if self.use_delete_gate:
            self.delete_gate = DeleteGate(
                d_model=d_model,
                elimination_rate=0.0,
            )

        if self.use_emb_align:
            self.embedding_aligner = EmbeddingAligner(
                student_dim=d_model,
                teacher_dim=config.get('teacher_dim', 128),
            )

    def forward(
        self,
        x: Tensor,
        hidden_states: list | None = None,
        training_phase: int = 1,
        target_elimination: float = 0.0,
        teacher_hidden: Tensor | None = None,
        teacher_mask: Tensor | None = None,
        byte_targets: Tensor | None = None,
    ) -> tuple:
        """
        Args:
            x: [B, S, 4] uint8/long for byte, [B, S] long for BPE
            hidden_states: list of per-layer MLGRU states (None for transformer)
            training_phase: 1 = distillation, 2 = with delete gate
            target_elimination: fraction of states to eliminate (Phase 2)
            teacher_hidden: [B, S, teacher_dim] float — projected teacher hidden states
            teacher_mask: [B, S] bool — where teacher signal exists
            byte_targets: [B, S, 4] long — ground-truth bytes of the PREDICTED patch.
                Required by the MegaByte LocalByteDecoder for teacher forcing.
                If None and the decoder is enabled, zeros are used (benchmark / dry-run).

        Returns:
            For byte:  (tuple of 4 × [B, S, 256], new_hidden_states, extra_losses)
            For BPE:   ([B, S, vocab_size], new_hidden_states, extra_losses)
        """
        if self.tokenization in ('byte', 'byte_dynamic'):
            x_emb = self.embedding(x)
        else:
            x_emb = self.embedding(x)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states = []
        extra_losses = {}
        h = x_emb

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _fn(_h, _hs, _b=block):
                    return _b(_h, _hs)
                h, h_new = checkpoint(_fn, h, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)

                # DeleteGate at midpoint (Phase 2 only)
                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    h, _ = self.delete_gate.forward_train(
                        h, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss
        else:
            for i, block in enumerate(self.blocks):
                h, h_new = block(h, hidden_states[i])
                new_states.append(h_new)

                # DeleteGate at midpoint (Phase 2 only)
                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    h, _ = self.delete_gate.forward_train(
                        h, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss

        h_pre_norm = h  # guardar para EmbeddingAligner antes del norm
        h = self.norm(h)

        if self.tokenization in ('byte', 'byte_dynamic'):
            if self.use_local_decoder:
                if byte_targets is None:
                    # Fallback for throughput benchmarks / dry runs.
                    B, S = h.shape[0], h.shape[1]
                    P = self.config.get('patch_size', 4)
                    byte_targets = torch.zeros(
                        B, S, P, dtype=torch.long, device=h.device
                    )
                logits = self.local_decoder(h, byte_targets)
            else:
                logits = tuple(head(h) for head in self.heads)
        else:
            logits = self.heads[0](h)

        # EmbeddingAlignment (Phase 1 only)
        if (self.use_emb_align and training_phase == 1
                and teacher_hidden is not None and teacher_mask is not None):
            extra_losses['emb_align'] = self.embedding_aligner.alignment_loss(
                h_pre_norm, teacher_hidden, teacher_mask,
            )

        # BoundaryPredictor — predict word boundaries from raw bytes (training only)
        if self.use_boundary and self.training and self.tokenization == 'byte_dynamic':
            # x: [B, S, 4] → [B, S*4] raw bytes
            raw_bytes = x.reshape(x.shape[0], -1).long()
            boundary_probs = self.boundary_predictor.forward_train(raw_bytes)  # [B, S*4]
            # Natural boundary targets: space (32), newline (10), UTF-8 multibyte start (≥192)
            targets = ((raw_bytes == 32) | (raw_bytes == 10) | (raw_bytes >= 192)).float()
            extra_losses['boundary'] = F.binary_cross_entropy(
                boundary_probs, targets
            )

        return logits, new_states, extra_losses

    def count_parameters(self) -> dict:
        """Return parameter counts by component."""
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.embedding.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
        if self.use_local_decoder:
            heads = sum(p.numel() for p in self.local_decoder.parameters())
        else:
            heads = sum(p.numel() for p in self.heads.parameters())
        return {
            'total': total,
            'total_M': total / 1e6,
            'embedding': embedding,
            'blocks': blocks,
            'heads': heads,
        }


# ============================================================
# Builder function used by ablation_train.py
# ============================================================

def build_ablation_model(config: dict) -> AblationModel:
    """Build the correct model variant from a merged run config."""
    model = AblationModel(config)
    counts = model.count_parameters()
    extras = []
    tok = config.get('tokenization', '')
    if tok in ('byte', 'byte_dynamic') and config.get('use_local_decoder', True):
        extras.append('LocalDecoder')
    if config.get('use_boundary_predictor'):
        extras.append('Bolmo')
    if config.get('use_delete_gate'):
        extras.append('DeleteGate')
    if config.get('use_embedding_alignment'):
        extras.append('EmbAlign')
    ext_str = f" + {', '.join(extras)}" if extras else ""
    print(f"[Model] {config.get('name', '?')} — "
          f"{counts['total_M']:.1f}M params "
          f"(emb={counts['embedding']/1e6:.1f}M, "
          f"blocks={counts['blocks']/1e6:.1f}M, "
          f"heads={counts['heads']/1e6:.1f}M){ext_str}")
    return model


# =============================================================================
# Section 3: Analysis
# =============================================================================
def load_all_metrics(metrics_dir: str = ".") -> dict[str, dict]:
    """Load metrics_run_XX.json files from directory."""
    results = {}
    for run_name in RUN_CONFIGS:
        path = Path(metrics_dir) / f"metrics_{run_name}.json"
        if path.exists():
            with open(path) as f:
                results[run_name] = json.load(f)
        else:
            print(f"  [WARN] Missing: {path}")
    return results


def generate_text_table(results: dict[str, dict]) -> str:
    """Generate a text table for terminal display."""
    proposed = results.get('run_09', {})
    proposed_tps = proposed.get('tokens_per_sec_cpu', 1)

    header = (
        f"{'Run':<8} {'Mixer':<12} {'Tok':<6} {'Quant':<10} "
        f"{'Signal':<12} {'PPL':>7} {'TPS':>7} {'RAM':>6} {'vs Prop':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for run_name in sorted(results.keys()):
        metrics = results[run_name]
        cfg = RUN_CONFIGS.get(run_name, {})
        tps = metrics.get('tokens_per_sec_cpu', 0)
        ppl = metrics.get('perplexity', 0)
        ram = metrics.get('ram_mb', 0)
        ratio = tps / proposed_tps * 100 if proposed_tps > 0 else 0
        marker = " <-" if cfg.get('is_proposed') else ""

        row = (
            f"{run_name:<8} "
            f"{cfg.get('token_mixer', '?'):<12} "
            f"{cfg.get('tokenization', '?'):<6} "
            f"{cfg.get('quantization', '?'):<10} "
            f"{cfg.get('training_signal', '?'):<12} "
            f"{ppl:>7.2f} "
            f"{tps:>7.1f} "
            f"{ram:>6.1f} "
            f"{ratio:>7.1f}%"
            f"{marker}"
        )
        rows.append(row)

    return "\n".join(rows)


def generate_latex_table(results: dict[str, dict]) -> str:
    """Generate LaTeX table ready for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study at 50M parameters. CE-only training from scratch. "
        r"TPS = tokens per second on consumer CPU (4-core, no SIMD). "
        r"$\downarrow$ lower is better, $\uparrow$ higher is better. "
        r"Run 10 (\textbf{CPU-1 Complete}) is the proposed architecture.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lllllrrr}",
        r"\toprule",
        r"Run & Mixer & Tok. & Quant. & Signal "
        r"& PPL$\downarrow$ & TPS$\uparrow$ & RAM$\downarrow$ \\",
        r"\midrule",
    ]

    for run_name in sorted(results.keys()):
        metrics = results[run_name]
        cfg = RUN_CONFIGS.get(run_name, {})
        tps = metrics.get('tokens_per_sec_cpu', 0)
        ppl = metrics.get('perplexity', 0)
        ram = metrics.get('ram_mb', 0)
        bold = cfg.get('is_proposed', False)

        mixer = cfg.get('token_mixer', '?').replace('_', '-')
        tok = cfg.get('tokenization', '?')
        quant = cfg.get('quantization', '?').replace('_', '-')
        signal = cfg.get('training_signal', '?').replace('_', '-')

        if bold:
            row = (
                f"\\textbf{{{run_name}}} & \\textbf{{{mixer}}} & "
                f"\\textbf{{{tok}}} & \\textbf{{{quant}}} & "
                f"\\textbf{{{signal}}} & \\textbf{{{ppl:.2f}}} & "
                f"\\textbf{{{tps:.1f}}} & \\textbf{{{ram:.0f}}} \\\\"
            )
        else:
            row = (
                f"{run_name} & {mixer} & {tok} & {quant} & "
                f"{signal} & {ppl:.2f} & {tps:.1f} & {ram:.0f} \\\\"
            )
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_comparison_summary(results: dict[str, dict]) -> str:
    """Generate human-readable comparison of key design decisions."""
    lines = ["=" * 60, "ABLATION ANALYSIS — Key Comparisons", "=" * 60, ""]

    def _compare(label, run_a, run_b):
        if run_a not in results or run_b not in results:
            return f"  {label}: MISSING DATA ({run_a} or {run_b})\n"
        a = results[run_a]
        b = results[run_b]
        ppl_a, ppl_b = a.get('perplexity', 0), b.get('perplexity', 0)
        tps_a, tps_b = a.get('tokens_per_sec_cpu', 0), b.get('tokens_per_sec_cpu', 0)
        ppl_delta = ((ppl_b - ppl_a) / max(ppl_a, 1e-6)) * 100
        tps_delta = ((tps_b - tps_a) / max(tps_a, 1e-6)) * 100
        return (
            f"  {label}:\n"
            f"    {run_a}: PPL={ppl_a:.2f}, TPS={tps_a:.1f}\n"
            f"    {run_b}: PPL={ppl_b:.2f}, TPS={tps_b:.1f}\n"
            f"    Delta: PPL {ppl_delta:+.1f}%, TPS {tps_delta:+.1f}%\n"
        )

    lines.append("1. Byte patches — tokenization effect on FP16 Transformer:")
    lines.append(_compare("  run_01 (BPE) vs run_02 (Byte)", 'run_01', 'run_02'))

    lines.append("2. MLGRU+Byte vs Transformer+Byte — mixer effect at byte level:")
    lines.append(_compare("  run_02 (Transformer+Byte) vs run_03 (MLGRU+Byte)", 'run_02', 'run_03'))

    lines.append("3. Ternary quantization cost on MLGRU+Byte:")
    lines.append(_compare("  run_03 (FP16) vs run_04 (Ternary)", 'run_03', 'run_04'))

    lines.append("4. FPResidual correction on MLGRU+Byte+Ternary (→ CPU-1 core):")
    lines.append(_compare("  run_04 (no FPRes) vs run_05 (FPRes)", 'run_04', 'run_05'))

    lines.append("5. Bolmo — dynamic patches on CPU-1 core:")
    lines.append(_compare("  run_05 (fixed) vs run_06 (Bolmo)", 'run_05', 'run_06'))

    lines.append("6. DeleteGate — state elimination on CPU-1 core:")
    lines.append(_compare("  run_05 (no gate) vs run_07 (DeleteGate)", 'run_05', 'run_07'))

    lines.append("7. CPU-1 core vs Folded Transformer (MLGRU vs local-attn):")
    lines.append(_compare("  run_05 (MLGRU) vs run_08 (Folded Transformer)", 'run_05', 'run_08'))

    lines.append("8. CPU-1 COMPLETE vs CPU-1 core:")
    lines.append(_compare("  run_05 (core) vs run_09 (complete)", 'run_05', 'run_09'))

    lines.append("9. Full CPU-1 COMPLETE vs absolute baseline:")
    lines.append(_compare("  run_01 (FP16 Transformer) vs run_09 (CPU-1 Complete)", 'run_01', 'run_09'))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CPU-1 ablation results"
    )
    parser.add_argument('--metrics_dir', type=str, default='.',
                        help="Directory containing metrics_run_XX.json files")
    parser.add_argument('--results_json', type=str, default=None,
                        help="Single JSON file with all results")
    parser.add_argument('--output', type=str, default='ablation_table.tex',
                        help="Output LaTeX file")
    args = parser.parse_args()

    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)
    else:
        results = load_all_metrics(args.metrics_dir)

    if not results:
        print("No results found. Run ablation_train.py first.")
        return

    print(f"\nLoaded {len(results)}/{len(RUN_CONFIGS)} runs\n")

    # Text table
    text_table = generate_text_table(results)
    print(text_table)

    # Comparison summary
    summary = generate_comparison_summary(results)
    print(f"\n{summary}")

    # LaTeX table
    latex_table = generate_latex_table(results)
    with open(args.output, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {args.output}")

    # Save all results to single JSON
    combined_path = Path(args.metrics_dir) / "ablation_results.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
