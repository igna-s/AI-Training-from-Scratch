"""
Training entry point for the CPU-1 ablation study.

Two modes of execution:
  1. Single run — train one ablation config:
        python train_ablation.py --run run_09
        python train_ablation.py --run run_09 --dry_run
        python train_ablation.py --run run_07 --phase 2 --phase1_ckpt checkpoint_run_07_final.pt

  2. Lightning Studio orchestration — run groups of ablations in parallel,
     one process per GPU (CUDA_VISIBLE_DEVICES). Groups are processed
     sequentially; runs inside a group run concurrently:
        python train_ablation.py --orchestrate                # run all groups
        python train_ablation.py --orchestrate --groups 0 2   # only groups 0,2
        python train_ablation.py --orchestrate --runs run_05 run_09
        python train_ablation.py --orchestrate --dry-run
        python train_ablation.py --orchestrate --max-parallel 1

The orchestration spawns subprocesses of this same script with --run, so
each ablation run owns one GPU end-to-end.

  3. PyTorch inference — load an ablation checkpoint and generate bytes.
        Edit the INFER_CONFIG dict at the top of this file and flip
        TRAIN=False / INFER=True. Then:
            python train_ablation.py
        Pure PyTorch path (byte / byte_dynamic runs only). The C/C++ kernels
        in `inference/` are for the production model — this entry point is
        for evaluating ablation checkpoints directly.

The full evaluation suite lives in `eval/`.
"""

import os
import sys
import gc
import math
import json
import time
import contextlib
import argparse
import subprocess
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from utilities import (
    # Primitives / configurable blocks
    RMSNorm, BitLinear, BitEmbedding, FPResidual, PFNet,
    AblationSwiGLU, AblationBytePatchEmbedding, BolmoPatchEmbedding,
    AblationMLGRU, SlidingWindowAttention, AblationBlock,
    AblationByteHead, AblationBPEHead, LocalByteDecoder,
    BoundaryPredictor, DeleteGate, EmbeddingAligner,
    # Training infrastructure
    CUDAPrefetcher,
    load_ablation_dataset, compute_loss,
    get_lr_from_config, measure_cpu_throughput, evaluate_run,
    save_ablation_checkpoint,
    # Checkpoint codec
    unpack_compact_state,
    _unwrap_model,
)


# =============================================================================
# 0. Mode switch — flip one of the two bools to choose what this script does
# =============================================================================
# TRAIN  : run training (single run or orchestration; CLI flags still apply).
# INFER  : load a checkpoint and generate bytes from a prompt — pure PyTorch,
#          NO CLI needed (just edit INFER_CONFIG below).
#
# Rules:
#   - Exactly one of {TRAIN, INFER} must be True. Setting none or both is an
#     error and the script exits with a clear message.

TRAIN = True
INFER = False

# --- INFER_CONFIG ------------------------------------------------------------
# Used only when INFER=True. Edit and run `python train_ablation.py`.
# Works with ANY ablation checkpoint:
#   - byte / byte_dynamic runs → max_new_bytes counts bytes.
#   - bpe runs (run_01, run_13) → max_new_bytes counts BPE tokens; the
#     Qwen2.5-3B tokenizer is loaded automatically (needs `transformers`).
# For Phase-2 (DeleteGate) checkpoints, set elimination_rate>0.

INFER_CONFIG = {
    'ckpt':             'checkpoint_run_10_final.pt',
    'prompt':           'Once upon a time',
    'max_new_bytes':    256,    # bytes for byte runs, tokens for bpe runs
    'temperature':      0.8,    # <=0 → greedy
    'top_k':            0,      # 0 → disabled
    'top_p':            0.0,    # 0 → disabled
    'elimination_rate': 0.0,    # >0 only with Phase-2 (DeleteGate) ckpts
    'device':           None,   # None → cuda if available, else cpu
}


# =============================================================================
# 1. Base configuration (50M params)
# =============================================================================
# Architecture constraints across ALL runs:
#   - 1.58-bit ternary weights (BitLinear, BitEmbedding) by default
#   - No activation×activation matmul at inference (MLGRU uses Hadamard only)
#   - Transformer runs use sliding-window attention (local, CPU-friendly)
#   - FP16 runs included only as reference baselines
#
# Dataset (CE-only training; no teacher logprobs):
#   50M runs (run_01..run_12): HuggingFaceFW/fineweb sample-10BT, streamed and
#       tokenized on the fly. Mirrors create_dataset.py params (seed=42,
#       seq_len=5000, patch_size=4, MAX_EXTRACTION_TOKENS=700).
#   10M runs (run_13/run_14/run_16): Cukinator/cpu1-ablation-dataset (~37.5M tokens,
#       built by create_dataset.py). Subsets: bpe_tokenized for run_13,
#       byte_marginalized for run_14/15.
#   run_16 (10M, "no teacher"): HuggingFaceFW/fineweb directly — same path
#       as the 50M runs. Whole point is to skip the Qwen teacher.

BASE_CONFIG = {
    'd_model':      512,
    'n_layers':     12,
    'd_ff':         1376,       # 8/3 × 512, rounded to multiple of 128
    'fp_rank':      4,
    'n_heads':      8,
    'head_dim':     64,
    'window_size':  512,
    'pfnet_hidden': 0,

    # Single T4 (16 GB) per run; orchestration pins each run to one GPU via
    # CUDA_VISIBLE_DEVICES. batch=24 / accum=1 fills ~10–12 GB at d=512 /
    # seq=5000 / 12 layers under AMP+grad-checkpointing.
    'batch_per_gpu':               24,
    'gradient_accumulation_steps': 1,
    'lr':                          3e-4,
    'min_lr':                      3e-5,
    'warmup_steps':                500,
    'total_steps':                 5_000,
    'weight_decay_bitlinear':      0.1,
    'weight_decay_fpresidual':     0.0,
    'weight_decay_heads':          0.01,
    'grad_clip':                   1.0,
    'seed':                        42,

    'training_signal': 'ce_only',
    'temperature': 4.0,
    'alpha_kl':    0.9,
    'alpha_ce':    0.1,

    # 50M runs stream raw FineWeb because Chinchilla at 50M demands ~1B tokens,
    # while Cukinator/cpu1-ablation-dataset only holds ~37.5M (sized for 10M).
    'dataset_repo':     'HuggingFaceFW/fineweb',
    'seq_len':          5000,
    'patch_size':       4,

    # MegaByte LocalByteDecoder — autoregressive intra-patch chain. Without it
    # the 4 bytes of each patch are independent, which corrupts UTF-8 sequences.
    'use_local_decoder':      True,
    'local_decoder_dim':      128,
    'local_decoder_heads':    4,
    'local_decoder_ff_ratio': 4.0,
    'local_decoder_fp16':     False,

    # Checkpoints — Chinchilla-rescaled save_every (5% of horizon → ~20/run)
    'checkpoint_repo':    'Cukinator/cpu1-ablation-checkpoints',
    'save_every_steps':   1000,
    'checkpoint_fraction': 0.05,

    # Chinchilla scaling: total_tokens = chinchilla_tokens_per_param × N_params.
    # Recomputes total_steps + warmup_steps + save_every once N_params is
    # known so warmup_ratio (5%) stays constant across param scales.
    'chinchilla':                  True,
    'chinchilla_tokens_per_param': 20,
    'warmup_ratio':                0.05,

    # DeleteGate Phase-2 target elimination rate. Read by both the gate's
    # internal rate (set at Phase-2 entry) and forward_train(target_rate=...)
    # so both paths agree. Only effective in runs with use_delete_gate=True.
    'target_elimination':          0.4,

    # ---- Revisión 4 fixes (audit walkthrough §"Cambios aplicados") ---------
    # MatMul-free LM W_o output projection per layer. ON by default in
    # ablation runs (closes the audit's #1 gap).  +d_model² params per layer
    # (~6% of a 50M ablation budget at d=512×12).  Production CPU1Block does
    # NOT use this flag — its kernel layout is fixed.
    'mlgru_use_wo':                True,
    # HGRN2 state expansion factor.  1 = disabled (current capacity), 2 =
    # doubles hidden state during recurrence (audit-recommended; still fits
    # the L2 budget at d=512).  Higher values risk L2 spills.
    'mlgru_state_expansion':       1,
    # DeleteGate behaviour upgrades — closer to MrT5.
    'delete_gate_hard':            True,    # STE binary mask in forward
    'delete_gate_gumbel':          True,    # Gumbel noise during training
    'delete_gate_merge':           True,    # depthwise-conv neighbour merge
    'delete_gate_bimodal_lambda':  0.05,
    # BolmoPatchEmbedding cross-attention heads (0 = legacy gate-only path).
    'bolmo_cross_attn_heads':      2,
    # BolmoPatchEmbedding cross-patch window: how many patches each query
    # attends to (1 = own patch only, 3 = ±1 neighbour = 12 bytes visible).
    # Approximates BLT variable-length patches via wider fixed attention.
    'bolmo_cross_patch_window':    3,
    # When True, the boundary scorer's k=3 conv is left-padded only so it
    # never sees future bytes — required for strict patch-level causal
    # streaming inference. Default False keeps existing checkpoints valid.
    'bolmo_causal_boundary':       False,
    # When True, the cross-patch attention window covers past patches + own
    # only (vs. symmetric ±cpw//2). Required alongside bolmo_causal_boundary
    # for end-to-end causality. Default False matches legacy behaviour.
    'bolmo_causal_cross_patch':    False,
    # DeleteGate gather mode: when True, post-gate layers process a compact
    # [B, K, D] sequence (K = S×(1-rate)) instead of the full S.  Gives
    # ~20% wall-clock speedup at rate=0.4.  When False, uses mask mode
    # (forward_train) which keeps shape [B, S, D].  Both produce the same
    # gradients for the scorer; gather mode additionally reduces MLGRU FLOPs.
    'use_real_gather':             False,

    # Phase-2 (DeleteGate fine-tune) budget as a fraction of the Chinchilla
    # horizon used in Phase-1. 0.5 ≈ half the pre-train length — enough to
    # train the gate without re-paying the full pre-train cost.
    'phase2_budget_fraction':      0.5,
}


# =============================================================================
# 2. Per-run delta configs — 12 progressive 50M runs
# =============================================================================
#
# Story: every run on the linear chain adds EXACTLY ONE component vs the
# previous run, so any Δppl can be attributed to a single design choice.
# Branches are explicit and labelled with their parent.
#
# LINEAR CHAIN (single Δ per arrow):
#   run_01                    Transformer + BPE + FP16        (absolute baseline)
#     ↓ +byte tokenization
#   run_02a_byte_only_heads   Transformer + Byte + 4 heads
#     ↓ +LocalByteDecoder
#   run_02                    Transformer + Byte + LBD        (MegaByte head)
#     ↓ swap mixer to MLGRU
#   run_03                    MLGRU + Byte + FP16
#     ↓ +ternary quantization
#   run_04                    MLGRU + Byte + Ternary
#     ↓ +FPResidual
#   run_05                    CPU-1 core (paper-fiel MLGRU)
#     ↓ +Bolmo (boundary-aware embedding)
#   run_06                    + Bolmo
#     ↓ +DeleteGate (MrT5 real gather)
#   run_07                    CPU-1 COMPLETE ⭐
#     ↓ +PFNet
#   run_09                    + PFNet
#     ↓ +per-channel decay (HGRN2 decay)
#   run_10                    + decay (chain terminus)
#
# BRANCHES (off the linear chain):
#   run_05b_kernel_strict     ← from run_05  (no W_o — kernel-layout cost)
#   run_08                    ← from run_05  (alt: folded Transformer mixer)

RUN_CONFIGS = {
    'run_01': {
        'name':         'transformer_bpe_fp16',
        'token_mixer':  'transformer',
        'tokenization': 'bpe',
        'quantization': 'fp16',
        'fp_residual':  False,
        # Full Qwen vocab (152K × 512 = 78M) overflows the 50M budget — cap to 16K.
        # The dataset's BPE token IDs are clamped to this range at load time.
        'bpe_vocab_size': 16384,
        'description':  'Absolute baseline — FP16 Transformer, BPE tokens (16K vocab), CE-only.',
        'cites':        [],
    },
    # run_02a isolates byte tokenization alone — uses 4 independent FP16 byte
    # heads (the MegaByte paper's per-byte logits without local decoder).
    # run_02 (kept below) layers the LocalByteDecoder on top to measure the
    # effect of the MegaByte autoregressive intra-patch chain in isolation.
    'run_02a_byte_only_heads': {
        'name':         'transformer_byte_fp16_no_lbd',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'use_local_decoder': False,
        'description':  '+Byte patches (no LocalByteDecoder) — FP16 Transformer + 4 independent byte heads. Pure tokenization isolation.',
        'cites':        ['MegaByte (Yu et al., 2023)'],
    },
    'run_02': {
        'name':         'transformer_byte_fp16',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  '+Byte patches + LocalByteDecoder — adds MegaByte intra-patch autoregression on top of run_02a.',
        'cites':        ['MegaByte (Yu et al., 2023)'],
    },
    'run_03': {
        'name':         'mlgru_byte_fp16',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  'MLGRU + byte patches, FP16. Combined mixer+tokenization.',
        'cites':        ['MatMul-free LM', 'MegaByte (Yu et al., 2023)'],
    },
    'run_04': {
        'name':         'mlgru_byte_ternary',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  False,
        'description':  '+Ternary — 1.58-bit weights on MLGRU+byte. Isolates quantization cost.',
        'cites':        ['BitNet b1.58 (Wang et al., 2024)', 'MatMul-free LM'],
    },
    'run_05': {
        'name':         'mlgru_byte_ternary_fpres',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  True,
        'description':  (
            '+FPResidual — low-rank FP16 correction alongside ternary weights. '
            'CPU-1 core architecture.'
        ),
        'cites':        ['Slender-Mamba (Yu et al., 2025)', 'BitNet b1.58'],
    },
    # Kernel-strict MLGRU: same as run_05 BUT without W_o (the 4th matrix the
    # Zhu 2024 paper specifies that the production C++ kernel cannot afford
    # because of OMP scheduling). Measures the cost of adopting the kernel
    # layout in PyTorch ablation. This is the *production-faithful* MLGRU,
    # NOT the paper-faithful one — run_05 already serves as the paper-faithful
    # baseline (W_o on, parallel scan on, no extras).
    'run_05b_kernel_strict': {
        'name':         'mlgru_kernel_strict',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  True,
        'mlgru_use_wo':            False,
        'mlgru_per_channel_decay': False,
        'mlgru_state_expansion':   1,
        'state_reset_period':      0,
        'mlgru_parallel_scan':     True,
        'description':  (
            'Kernel-strict MLGRU — run_05 minus W_o, matching the layout the '
            'production C++ kernel can run. Measures the cost of dropping the '
            'paper\'s 4th matrix to fit OMP scheduling.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'Slender-Mamba (Yu et al., 2025)',
        ],
    },
    'run_06': {
        'name':         'mlgru_byte_ternary_fpres_bolmo',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        # Bolmo paper-faithful path: BolmoPatchEmbedding cross-attention +
        # boundary scorer feeding into the patch encoding (NOT just an
        # auxiliary BCE on a disconnected scorer).
        'use_bolmo_gating':       True,
        'use_boundary_predictor': False,
        'description':  (
            '+Bolmo — boundary-aware patch embedding via cross-attention '
            '(BolmoPatchEmbedding). CPU-1 core + integrated boundary signal.'
        ),
        'cites':        ['BLT (Pagnoni et al., 2024)'],
    },
    # +DeleteGate — chained from run_06 (Bolmo). Single Δ vs run_06: real
    # MrT5 gather at layer N//2. This is the new "CPU-1 COMPLETE" identity.
    'run_07': {
        'name':         'cpu1_complete',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_bolmo_gating':       True,
        'use_boundary_predictor': False,
        'use_delete_gate':        True,
        # Real MrT5 gather: post-gate layers process [B, K, D] with
        # K = S·(1 - rate), not [B, S, D] with a soft mask. Paper-faithful.
        'use_real_gather':        True,
        'description':  (
            '+DeleteGate (real gather) — MrT5-style state elimination at '
            'midpoint (~40%). Layered on run_06 Bolmo. Full proposed '
            'CPU-1 stack so far ⭐'
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
    # Sibling: alternative architecture — folded sliding-window Transformer
    # instead of MLGRU. Not in the linear chain (different mixer family).
    'run_08': {
        'name':         'folded_transformer_byte_ternary',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  False,
        'window_size':  128,
        'description':  (
            'Folded Transformer — ternary sliding-window attention over byte '
            'patches. O(N·128) vs O(N²). Alternative to MLGRU; sibling of '
            'run_05 (no Bolmo / DG / decay layered on top).'
        ),
        'cites':        [
            'Longformer (Beltagy et al., 2020)',
            'BitNet b1.58 (Wang et al., 2024)',
        ],
    },
    # +PFNet — single Δ vs run_07. Cache-resident nonlinear residual.
    'run_09': {
        'name':         'cpu1_pfnet',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'pfnet_hidden': 32,
        'use_bolmo_gating':       True,
        'use_boundary_predictor': False,
        'use_delete_gate':        True,
        'use_real_gather':        True,
        'description':  (
            'run_07 + PFNet (pfnet_hidden=32). Cache-resident nonlinear '
            'residual added per block — quality gain at zero RAM overhead.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'Slender-Mamba (Yu et al., 2025)',
        ],
    },
    # +per-channel decay — single Δ vs run_09. HGRN2 decay only (no expansion).
    'run_10': {
        'name':         'cpu1_decay_learned',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'pfnet_hidden': 32,
        'use_bolmo_gating':       True,
        'use_boundary_predictor': False,
        'use_delete_gate':        True,
        'use_real_gather':        True,
        'mlgru_per_channel_decay': True,
        'description':  (
            'run_09 + per-channel learnable decay in MLGRU (RWKV/HGRN2-style). '
            'Each hidden dim has its own decay rate — d_model × n_layers '
            'scalars total. State expansion stays at 1; isolated effect of '
            'decay alone.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'RWKV (Peng et al., 2023)',
            'HGRN2 (Qin et al., 2024)',
        ],
}


# =============================================================================
# 3. 10M-parameter scaling-consistency runs
# =============================================================================
# Why 10M?
#   - Trains in ~1h on T4 (vs 4h for 50M) — fast iteration
#   - Mirrors 50M decisions to verify trends are scale-consistent
#
# All four small_ runs use the CPU-1 COMPLETE architecture; only the training
# strategy varies (BPE-from-scratch / byte-from-scratch / BPE→byte complement /
# raw bytes no teacher).

SMALL_BASE_CONFIG = {
    **BASE_CONFIG,
    'd_model':      320,
    'n_layers':     8,
    'd_ff':         853,        # 8/3 × 320
    'fp_rank':      4,
    'n_heads':      8,
    'head_dim':     40,
    'window_size':  256,

    # Fixed budget: Cukinator has ~37.5M tokens → ~312 steps/epoch.
    # 1000 steps = ~3.2 epochs. NOT Chinchilla-scaled so the comparison
    # between Cukinator (teacher) and FineWeb (CE-only) is compute-fair.
    'chinchilla':       False,
    'total_steps':      1000,
    'warmup_steps':     100,
    'save_every_steps': 200,

    # 10M runs use the pre-processed Cukinator dataset; run_16 overrides
    # this back to FineWeb because its question is "no teacher".
    'dataset_repo': 'Cukinator/cpu1-ablation-dataset',

    # The Cukinator dataset ships Qwen2.5-3B teacher logprobs + hidden states.
    # Override BASE_CONFIG's 'ce_only' so runs 13-14 actually distill from them.
    # run_16 overrides this back to 'ce_only' (its question is "no teacher").
    'training_signal': 'distillation',
}

SMALL_RUN_CONFIGS = {
    'run_13': {
        'name':                   'small_cpu1_bpe',
        'token_mixer':            'mlgru',
        'tokenization':           'bpe',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': False,
        'use_delete_gate':        True,
        'bpe_vocab_size':         4096,
        'description':            'CPU-1 COMPLETE @ 10M, compact BPE (4K vocab) from scratch. '
                                  'Does BPE tokenization outperform byte-level at 10M?',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'MrT5'],
    },
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
    'run_16': {
        'name':                   'small_cpu1_raw_bytes',
        'token_mixer':            'mlgru',
        'tokenization':           'byte_dynamic',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'raw_bytes_only':         True,
        'dataset_repo':           'HuggingFaceFW/fineweb',
        'training_signal':        'ce_only',
        'description':            'CPU-1 COMPLETE @ 10M, raw bytes from scratch, zero teacher. '
                                  'Do Qwen logprobs in run_14 actually help vs pure CE?',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'Bolmo', 'MrT5'],
    },
}

ALL_RUN_CONFIGS = {**RUN_CONFIGS, **SMALL_RUN_CONFIGS}


def get_run_config(run_name: str) -> dict:
    """Merge base + run-specific delta. Selects SMALL_BASE_CONFIG for 10M runs."""
    if run_name not in ALL_RUN_CONFIGS:
        raise ValueError(
            f"Unknown run: {run_name}. Available: {list(ALL_RUN_CONFIGS.keys())}"
        )
    run = ALL_RUN_CONFIGS[run_name]
    base = SMALL_BASE_CONFIG if run_name in SMALL_RUN_CONFIGS else BASE_CONFIG
    merged = {**base}
    merged['use_ternary'] = run['quantization'] == 'ternary'
    fp_residual = run.get('fp_residual', False)
    merged['fp_rank'] = (
        base['fp_rank'] if (fp_residual and run['quantization'] == 'ternary') else 0
    )
    merged['use_boundary_predictor'] = run.get('use_boundary_predictor', False)
    merged['use_delete_gate'] = run.get('use_delete_gate', False)
    merged['use_embedding_alignment'] = run.get('use_embedding_alignment', False)
    merged.update(run)
    return merged


# =============================================================================
# 4. Chinchilla scaling helpers (Hoffmann et al., 2022)
# =============================================================================

def chinchilla_total_steps(config: dict, n_params: int) -> int:
    """Steps to consume `chinchilla_tokens_per_param × n_params` tokens."""
    tokens_per_param = config.get('chinchilla_tokens_per_param', 20)
    target_tokens = int(n_params) * int(tokens_per_param)
    tokens_per_step = (
        int(config['batch_per_gpu'])
        * int(config['gradient_accumulation_steps'])
        * int(config['seq_len'])
    )
    if tokens_per_step <= 0:
        raise ValueError(f"Invalid tokens_per_step from config: {tokens_per_step}")
    return max(target_tokens // tokens_per_step,
               int(config.get('warmup_steps', 100)) + 100)


def chinchilla_save_every(total_steps: int, fraction: float = 0.05) -> int:
    return max(int(total_steps * float(fraction)), 1)


def chinchilla_warmup(total_steps: int, ratio: float = 0.05) -> int:
    """Constant warmup *fraction* (not constant step count) keeps schedules
    comparable across param scales."""
    return max(int(total_steps * float(ratio)), 50)


# =============================================================================
# 5. Unified ablation model
# =============================================================================

def _gather_strides(keep_idx: Tensor) -> Tensor:
    """Per-survivor temporal stride after DeleteGate gather.

    Given keep_idx [B, K] of original positions sorted ascending, returns
    [B, K] where strides[b, k] = keep_idx[b, k] - keep_idx[b, k-1] for k>0
    and strides[b, 0] = 1 (first surviving token has no predecessor — the
    decay applied at t=0 acts on the zero-init state). Consumed by
    AblationMLGRU to raise per-channel decay to the gap power so the
    effective forget rate matches the original timescale.
    """
    B, K = keep_idx.shape
    if K == 0:
        return keep_idx.new_ones((B, 0))
    first = keep_idx.new_ones((B, 1))
    rest = keep_idx[:, 1:] - keep_idx[:, :-1]
    return torch.cat([first, rest], dim=1).clamp(min=1)


class AblationModel(nn.Module):
    """Configurable model for all ablation runs.

    Determined by config:
        token_mixer:  'transformer' | 'mlgru'
        tokenization: 'bpe' | 'byte' | 'byte_dynamic'
        use_ternary:  True | False
        fp_rank:      0 (no FPResidual) | 4 (with FPResidual)

    Optional modules (run_06+):
        use_boundary_predictor:  Bolmo
        use_delete_gate:         MrT5
        use_embedding_alignment: arXiv 2602
        use_bolmo_gating:        Bolmo integrated scorer in the embedding
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        n_layers = config['n_layers']
        tok = config['tokenization']
        use_ternary = config.get('use_ternary', True)

        self.use_bolmo_gating = config.get('use_bolmo_gating', False)
        if tok in ('byte', 'byte_dynamic'):
            if self.use_bolmo_gating:
                self.embedding = BolmoPatchEmbedding(
                    d_model,
                    patch_size=config.get('patch_size', 4),
                    use_ternary=use_ternary,
                    cross_attn_heads=config.get('bolmo_cross_attn_heads', 2),
                    cross_patch_window=config.get('bolmo_cross_patch_window', 1),
                    causal_boundary=config.get('bolmo_causal_boundary', False),
                    causal_cross_patch=config.get('bolmo_causal_cross_patch', False),
                )
            else:
                self.embedding = AblationBytePatchEmbedding(
                    d_model,
                    patch_size=config.get('patch_size', 4),
                    use_ternary=use_ternary,
                )
        else:
            vocab_size = config.get('bpe_vocab_size', 152064)
            from utilities import make_embedding
            self.embedding = make_embedding(vocab_size, d_model, use_ternary)

        self.blocks = nn.ModuleList([
            AblationBlock(config, layer_idx=i) for i in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Heads — MegaByte LocalByteDecoder by default for byte runs.
        self.use_local_decoder = (
            tok in ('byte', 'byte_dynamic')
            and config.get('use_local_decoder', True)
        )
        if self.use_local_decoder:
            local_ternary = use_ternary and not config.get('local_decoder_fp16', False)
            self.local_decoder = LocalByteDecoder(
                d_model=d_model,
                patch_size=config.get('patch_size', 4),
                d_local=config.get('local_decoder_dim', 128),
                n_heads=config.get('local_decoder_heads', 4),
                ff_ratio=config.get('local_decoder_ff_ratio', 4.0),
                use_ternary=local_ternary,
                n_layers=config.get('local_decoder_layers', 2),
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

        self.use_boundary = config.get('use_boundary_predictor', False)
        self.use_delete_gate = config.get('use_delete_gate', False)
        self.use_real_gather = config.get('use_real_gather', False)
        self.use_emb_align = config.get('use_embedding_alignment', False)
        self.delete_gate_pos = n_layers // 2

        if self.use_boundary:
            self.boundary_predictor = BoundaryPredictor(embed_dim=8, hidden_dim=32)

        if self.use_delete_gate:
            self.delete_gate = DeleteGate(
                d_model=d_model,
                elimination_rate=0.0,
                bimodal_lambda=config.get('delete_gate_bimodal_lambda', 0.05),
                hard=config.get('delete_gate_hard', True),
                gumbel=config.get('delete_gate_gumbel', True),
                merge=config.get('delete_gate_merge', True),
            )

        if self.use_emb_align:
            self.embedding_aligner = EmbeddingAligner(
                student_dim=d_model,
                teacher_dim=config.get('teacher_dim', 128),
            )
    @torch.no_grad()
    def _enforce_decay_cummax(self):
        """HGRN2 cummax: ensure deeper layers always retain >= shallow layers.

        Collects ``log_decay`` from every MLGRU block and clamps so that
        ``decay[layer_l] >= max(decay[0..l-1])`` per channel.  Operates
        in softplus-space (``log_decay``): a higher ``log_decay`` means a
        *smaller* decay (faster forget), so we enforce a running *minimum*
        to guarantee deeper layers forget no faster than shallow ones.
        """
        decays = [
            b.token_mixer.log_decay
            for b in self.blocks
            if hasattr(b, 'token_mixer')
            and hasattr(b.token_mixer, 'log_decay')
            and b.token_mixer.log_decay is not None
        ]
        if len(decays) < 2:
            return
        # Running min in log_decay space = running max in decay space.
        running_min = decays[0].data.clone()
        for d in decays[1:]:
            running_min = torch.min(running_min, d.data)
            d.data.copy_(torch.min(d.data, running_min))

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
        x_emb = self.embedding(x)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states: list = []
        extra_losses: dict = {}
        h = x_emb

        # HGRN2 cummax: enforce monotonic decay across layers so deep layers
        # always retain at least as much context as shallow ones.
        self._enforce_decay_cummax()

        # Determine if we should use real gather mode for this forward pass.
        # Conditions: use_real_gather enabled, delete_gate active, Phase 2,
        # and target_elimination > 0.  Otherwise fall back to mask mode.
        do_gather = (
            self.use_delete_gate
            and self.use_real_gather
            and training_phase == 2
            and target_elimination > 0.0
        )
        has_gathered = False
        h_pre_gate = None
        keep_idx = None
        strides = None  # gap between consecutive survivors after gather

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                if not has_gathered:
                    # Pre-gate layers (or all layers if no gather)
                    def _fn(_h, _hs, _b=block):
                        return _b(_h, _hs)
                    h, h_new = checkpoint(_fn, h, hidden_states[i],
                                          use_reentrant=False)
                    new_states.append(h_new)

                    if do_gather and i == self.delete_gate_pos:
                        h_pre_gate = h                        # save [B, S, D]
                        h, keep_idx, gate_loss = self.delete_gate.forward_gather(
                            h, target_rate=target_elimination
                        )
                        extra_losses['delete_gate'] = gate_loss
                        has_gathered = True
                        strides = _gather_strides(keep_idx)
                    elif (self.use_delete_gate
                            and i == self.delete_gate_pos
                            and training_phase == 2):
                        # Mask mode fallback (use_real_gather=False)
                        h, _ = self.delete_gate.forward_train(
                            h, target_rate=target_elimination
                        )
                        extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss
                else:
                    # Post-gate layers: h is [B, K, D] — shorter sequence.
                    # strides carry gap info so HGRN2 decay compensates.
                    def _fn_post(_h, _s, _b=block):
                        return _b(_h, None, strides=_s)
                    h, h_new = checkpoint(_fn_post, h, strides,
                                          use_reentrant=False)
                    new_states.append(h_new)
        else:
            for i, block in enumerate(self.blocks):
                if not has_gathered:
                    # Pre-gate layers (or all layers if no gather)
                    h, h_new = block(h, hidden_states[i])
                    new_states.append(h_new)

                    if do_gather and i == self.delete_gate_pos:
                        h_pre_gate = h                        # save [B, S, D]
                        h, keep_idx, gate_loss = self.delete_gate.forward_gather(
                            h, target_rate=target_elimination
                        )
                        extra_losses['delete_gate'] = gate_loss
                        has_gathered = True
                        strides = _gather_strides(keep_idx)
                    elif (self.use_delete_gate
                            and i == self.delete_gate_pos
                            and training_phase == 2):
                        # Mask mode fallback (use_real_gather=False)
                        h, _ = self.delete_gate.forward_train(
                            h, target_rate=target_elimination
                        )
                        extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss
                else:
                    # Post-gate layers: h is [B, K, D] — shorter sequence.
                    # strides carry gap info so HGRN2 decay compensates.
                    h, h_new = block(h, None, strides=strides)
                    new_states.append(h_new)

        # Scatter back before RMSNorm + decoder: restore [B, S, D]
        if has_gathered and h_pre_gate is not None:
            h = self.delete_gate.scatter_back(h, keep_idx, h_pre_gate)

        h_pre_norm = h
        h = self.norm(h)

        if self.tokenization in ('byte', 'byte_dynamic'):
            if self.use_local_decoder:
                if byte_targets is None:
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

        if (self.use_emb_align and training_phase == 1
                and teacher_hidden is not None and teacher_mask is not None):
            extra_losses['emb_align'] = self.embedding_aligner.alignment_loss(
                h_pre_norm, teacher_hidden, teacher_mask,
            )

        if self.use_boundary and self.training and self.tokenization == 'byte_dynamic':
            raw_bytes = x.reshape(x.shape[0], -1).long()
            # BoundaryPredictor returns LOGITS so we can use the autocast-safe
            # `binary_cross_entropy_with_logits`. Plain BCE on a fp16 sigmoid
            # output saturates to log(0) = -inf → NaN → GradScaler skips every
            # step and the run silently stalls.
            boundary_logits = self.boundary_predictor.forward_train(raw_bytes)
            targets = (
                (raw_bytes == 32) | (raw_bytes == 10) | (raw_bytes >= 192)
            ).float()
            extra_losses['boundary'] = F.binary_cross_entropy_with_logits(
                boundary_logits, targets
            )

        # Bolmo integrated gating: supervise the embedding's internal scorer
        # with natural-boundary targets. Reuses the 'boundary' key (mutually
        # exclusive with use_boundary_predictor by construction).
        if (self.use_bolmo_gating and self.training
                and self.tokenization in ('byte', 'byte_dynamic')):
            raw_bytes = x.reshape(x.shape[0], -1).long()
            bndry_logits = self.embedding.last_boundary_logits
            targets = (
                (raw_bytes == 32) | (raw_bytes == 10) | (raw_bytes >= 192)
            ).float()
            extra_losses['boundary'] = F.binary_cross_entropy_with_logits(
                bndry_logits, targets
            )

        return logits, new_states, extra_losses

    def count_parameters(self) -> dict:
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


def build_ablation_model(config: dict) -> AblationModel:
    """Build the right model variant from a merged run config and log a summary."""
    model = AblationModel(config)
    counts = model.count_parameters()
    extras = []
    tok = config.get('tokenization', '')
    if tok in ('byte', 'byte_dynamic') and config.get('use_local_decoder', True):
        extras.append('LocalDecoder')
    if config.get('use_bolmo_gating') or config.get('use_boundary_predictor'):
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
# 6. Single-run training loop
# =============================================================================

def train_run(run_name: str, dry_run: bool = False,
              total_steps_override: int | None = None,
              phase: int = 1,
              phase1_ckpt: str | None = None,
              use_hf_mount: bool = False) -> dict:
    """Train one ablation run end-to-end. Returns final validation metrics."""
    config = get_run_config(run_name)
    config['_current_phase'] = phase

    if total_steps_override is not None:
        config['total_steps'] = total_steps_override

    assert config['seed'] == BASE_CONFIG['seed'], "Seed mismatch"

    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() and not dry_run else 0
    if n_gpus > 0:
        torch.cuda.set_device(0)

    import random as _random
    import numpy as _np
    _random.seed(config['seed'])
    _np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    device = torch.device('cpu') if dry_run else torch.device('cuda:0')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"Description: {config['description']}")
    print(f"Device: {device} ({n_gpus} GPU{'s' if n_gpus != 1 else ''})")
    print(f"{'=' * 60}\n")

    if not dry_run:
        config['use_gradient_checkpointing'] = True
    model = build_ablation_model(config).to(device)

    # Chinchilla rescale: total_steps / warmup / save cadence from N_params.
    # Applied on both phases so Phase-2 (DeleteGate fine-tune) inherits the
    # param-scaled horizon instead of falling back to the BASE_CONFIG default.
    if (config.get('chinchilla', True)
            and total_steps_override is None
            and not dry_run):
        n_params = sum(p.numel() for p in model.parameters())
        phase_frac = float(config.get('phase2_budget_fraction', 0.5)) if phase == 2 else 1.0
        new_total = max(int(chinchilla_total_steps(config, n_params) * phase_frac),
                        int(config.get('warmup_steps', 100)) + 100)
        new_save = chinchilla_save_every(new_total, config.get('checkpoint_fraction', 0.05))
        new_warmup = chinchilla_warmup(new_total, config.get('warmup_ratio', 0.05))
        tokens_per_step = (
            config['batch_per_gpu'] * config['gradient_accumulation_steps']
            * config['seq_len']
        )
        tag = f"Phase-{phase} (×{phase_frac:g})"
        print(
            f"[Chinchilla {tag}] N={n_params/1e6:.1f}M params · "
            f"target={n_params * config.get('chinchilla_tokens_per_param', 20) * phase_frac/1e6:.1f}M tokens · "
            f"tokens/step={tokens_per_step:,} · "
            f"total_steps {config.get('total_steps')} → {new_total} · "
            f"warmup {config.get('warmup_steps')} → {new_warmup} ({config.get('warmup_ratio', 0.05)*100:.0f}% of horizon) · "
            f"save_every {new_save} ({config.get('checkpoint_fraction', 0.05)*100:.0f}% of horizon)"
        )
        config['total_steps'] = new_total
        config['warmup_steps'] = new_warmup
        config['save_every_steps'] = new_save

    # Complement warm-start: load a sibling run's final checkpoint if configured.
    # Architecture may differ (embeddings/heads), so strict=False is required;
    # only matching keys (MLGRU blocks, RMSNorm, DeleteGate) transfer.
    complement = config.get('complement_from')
    if phase == 1 and complement and not dry_run:
        ckpt_path = str(Path(complement) / f"checkpoint_{complement}_final.pt")
        if Path(ckpt_path).exists():
            print(f"[Complement] Warm-starting from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            fmt = ckpt.get('format', 'bf16')
            if fmt == 'compact_2bit':
                comp_state = unpack_compact_state(ckpt['compact'])
            else:
                comp_state = ckpt['model_state_dict']
            target = _unwrap_model(model)
            missing, unexpected = target.load_state_dict(comp_state, strict=False)
            transferred = sum(1 for k in comp_state if k not in unexpected)
            print(f"[Complement] Transferred {transferred} keys "
                  f"(skipped {len(unexpected)} incompatible, "
                  f"{len(missing)} fresh-init)")
            # Sanity check: a corrupt complement (e.g. ckpt with `_orig_mod.`
            # prefix from a compiled save) would transfer zero keys silently.
            n_target = sum(1 for _ in target.state_dict())
            if transferred < max(10, n_target // 4):
                raise RuntimeError(
                    f"[Complement] Suspicious load: only {transferred}/{n_target} "
                    f"keys transferred from {ckpt_path}. Refusing to start "
                    f"training from near-random weights. Check that the source "
                    f"checkpoint was unwrapped before save."
                )
        else:
            print(f"[Complement] WARNING: {ckpt_path} not found — training from scratch")

    # Phase 2: load Phase 1 checkpoint, activate DeleteGate, freeze EmbAligner.
    # Phase-1 final ckpt is compact_2bit (no 'model_state_dict'); intermediate
    # bf16 ckpts have it. Handle both.
    if phase == 2 and phase1_ckpt:
        print(f"[Phase 2] Loading Phase 1 checkpoint: {phase1_ckpt}")
        ckpt = torch.load(phase1_ckpt, map_location='cpu', weights_only=False)
        fmt = ckpt.get('format', 'bf16')
        if fmt == 'compact_2bit':
            p1_state = unpack_compact_state(ckpt['compact'])
        else:
            p1_state = ckpt['model_state_dict']
        target = _unwrap_model(model)
        missing, unexpected = target.load_state_dict(p1_state, strict=False)
        # Phase-1 ckpts saved from a compiled model (without unwrap) end up
        # with an `_orig_mod.` prefix in every key. strict=False silently
        # discards them all → Phase-2 trains from random init with the gate
        # active. Fail loud instead.
        n_target = sum(1 for _ in target.state_dict())
        loaded = n_target - len(missing)
        if loaded < max(10, n_target // 2):
            raise RuntimeError(
                f"[Phase 2] Refusing to start: only {loaded}/{n_target} keys "
                f"loaded from {phase1_ckpt} (missing={len(missing)}, "
                f"unexpected={len(unexpected)}). Likely the ckpt was saved "
                f"from a torch.compile-wrapped model without unwrapping; "
                f"keys carry an `_orig_mod.` prefix."
            )
        if hasattr(target, 'delete_gate'):
            target.delete_gate.set_elimination_rate(
                config.get('target_elimination', 0.4)
            )
        if hasattr(target, 'embedding_aligner'):
            for p in target.embedding_aligner.parameters():
                p.requires_grad = False
        print(f"[Phase 2] DeleteGate active, EmbAligner frozen (format={fmt})")

    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Multi-GPU] DataParallel on {n_gpus} GPUs: "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    mount_path = None
    if use_hf_mount and not dry_run:
        try:
            from hf_mount import mount_bucket
            mount_path = mount_bucket(config['dataset_repo'])
        except Exception as e:
            print(f"[hf-mount] Mount failed, falling back to download: {e}")

    train_ds = load_ablation_dataset(config, split='train', dry_run=dry_run,
                                     mount_path=mount_path)
    val_ds = load_ablation_dataset(config, split='val' if not dry_run else 'train',
                                   dry_run=dry_run, mount_path=mount_path)

    n_workers = 0 if dry_run else 6
    if (override := os.environ.get('SMOKE_N_WORKERS')) is not None:
        n_workers = int(override)
    total_batch = config['batch_per_gpu'] * max(n_gpus, 1)
    if (override := os.environ.get('SMOKE_BATCH_PER_GPU')) is not None:
        total_batch = int(override) * max(n_gpus, 1)
    is_iterable = isinstance(train_ds, torch.utils.data.IterableDataset)
    train_loader = DataLoader(
        train_ds,
        batch_size=total_batch,
        shuffle=False if is_iterable else True,
        num_workers=n_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(n_workers > 0),
        prefetch_factor=4 if n_workers > 0 else None,
        drop_last=False,
    )

    # Param-group routing for AdamW weight decay:
    #   bitlinear: ternary projections inside mixers/FFN/local_decoder attn+ffn
    #              and the LocalByteDecoder global lift (h_proj).
    #   heads:     final byte/BPE prediction projections — both the standalone
    #              AblationByte/BPEHead under `heads.*` and LocalByteDecoder's
    #              `out_proj` (functionally a head).
    #   other:     embeddings, RMSNorm, scales, FPResidual U/V — no decay.
    BITLINEAR_KEYS = (
        'forget_proj', 'cand_proj', 'output_proj',
        'gate_proj', 'up_proj', 'down_proj',
        'q_proj', 'k_proj', 'v_proj', 'o_proj',
        # Embedding's BitLinear projection (byte and Bolmo paths)
        'patch_proj',
        # LocalByteDecoder internals (not matched by the *_proj names above).
        # Bare names match both layer-0 flat attrs and `extra_layers.k.*` when
        # local_decoder_layers > 1 (paper-faithful MegaByte stack).
        'h_proj', 'ffn_gate', 'ffn_up', 'ffn_down',
    )
    bitlinear_params = []
    head_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'weight' in name and any(k in name for k in BITLINEAR_KEYS):
            bitlinear_params.append(p)
        elif (('heads.' in name and 'proj.weight' in name)
              or 'local_decoder.out_proj.weight' in name):
            head_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': bitlinear_params, 'weight_decay': config['weight_decay_bitlinear']},
        {'params': head_params,      'weight_decay': config['weight_decay_heads']},
        {'params': other_params,     'weight_decay': 0.0},
    ], lr=config['lr'], betas=(0.9, 0.95))

    use_amp = not dry_run and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None

    # torch.compile gives 10-20% throughput on T4 for 50M models. Not compatible
    # with DataParallel — skip when multi-GPU. MLGRU runs use a Python loop
    # over S=5000 timesteps, which is hostile to CUDA Graphs ("reduce-overhead")
    # — fall back to mode="default" for those, keep "reduce-overhead" for the
    # transformer runs where the graph is static.
    if (device.type == 'cuda' and not dry_run and n_gpus <= 1
            and os.environ.get('SMOKE_NO_COMPILE') != '1'):
        compile_mode = (
            "reduce-overhead"
            if config.get('token_mixer', 'mlgru') == 'transformer'
            else "default"
        )
        try:
            model = torch.compile(model, mode=compile_mode)
            print(f"[Opt] torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            print(f"[Opt] torch.compile failed ({e}), using eager mode")

    model.train()
    optim_step = 0
    micro_step = 0
    losses = []
    tok = config['tokenization']
    accum_steps = config['gradient_accumulation_steps']
    total_steps = config['total_steps']
    t_start = time.time()
    last_ckpt_time = t_start

    prefetcher = CUDAPrefetcher(train_loader, device)

    while optim_step < total_steps:
        for batch in prefetcher:
            if optim_step >= total_steps:
                break

            if tok in ('byte', 'byte_dynamic'):
                x = batch['patches']
                targets = batch['targets']
            else:
                x = batch['input_ids']
                targets = batch['labels']

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            t_ids = batch.get('teacher_ids_bpe')
            t_hidden = batch.get('teacher_hidden')

            is_last_micro = (micro_step % accum_steps == accum_steps - 1)
            amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                training_phase = config.get('_current_phase', 1)
                target_elim_max = config.get('target_elimination', 0.0)
                # 🥇 Curriculum learning: ramp target_rate gradually from 0
                # to target_elimination over the first half of Phase 2,
                # instead of jumping to 0.4 abruptly.  This forces the
                # DeleteGate to learn selective attenuation progressively.
                if training_phase == 2 and target_elim_max > 0:
                    ramp_steps = max(total_steps // 2, 1)
                    target_elim = min(
                        target_elim_max,
                        target_elim_max * optim_step / ramp_steps
                    )
                else:
                    target_elim = target_elim_max
                fwd_kwargs = (
                    {'byte_targets': targets} if tok in ('byte', 'byte_dynamic') else {}
                )
                result = model(
                    x, training_phase=training_phase,
                    target_elimination=target_elim,
                    teacher_hidden=t_hidden, teacher_mask=t_mask,
                    **fwd_kwargs,
                )
                if len(result) == 3:
                    logits, _, extra_losses = result
                else:
                    logits, _ = result
                    extra_losses = {}
                loss = compute_loss(logits, targets, t_probs, t_mask, config,
                                    teacher_ids=t_ids)
                # Auxiliary losses (DataParallel: scalars stack into dim 0 → mean)
                if 'delete_gate' in extra_losses:
                    dg = extra_losses['delete_gate']
                    loss = loss + 0.1 * (dg.mean() if dg.dim() > 0 else dg)
                if 'emb_align' in extra_losses and extra_losses['emb_align'] is not None:
                    ea = extra_losses['emb_align']
                    loss = loss + config.get('lambda_align', 0.3) * (
                        ea.mean() if ea.dim() > 0 else ea
                    )
                if 'boundary' in extra_losses:
                    bp = extra_losses['boundary']
                    loss = loss + 0.1 * (bp.mean() if bp.dim() > 0 else bp)
                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(loss.item() * accum_steps)
            del logits, result, extra_losses
            if t_probs is not None:
                del t_probs, t_mask
            if t_hidden is not None:
                del t_hidden
            del loss
            micro_step += 1

            if micro_step % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['grad_clip']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['grad_clip']
                    )
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                optim_step += 1
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr_from_config(optim_step, config)

                if optim_step % 500 == 0:
                    avg_loss = sum(losses[-100:]) / min(100, len(losses))
                    elapsed = time.time() - t_start
                    if device.type == 'cuda':
                        vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                        vram_res_mb = torch.cuda.memory_reserved() / (1024 ** 2)
                        print(f"Step {optim_step:5d}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {get_lr_from_config(optim_step, config):.2e} | "
                              f"VRAM: {vram_mb:.0f}/{vram_res_mb:.0f} MB | "
                              f"Time: {elapsed:.0f}s")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print(f"Step {optim_step:5d}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {get_lr_from_config(optim_step, config):.2e} | "
                              f"Time: {elapsed:.0f}s")

                # Save policy: step 10 (smoke test), every save_every_steps
                # (Chinchilla: 5% of horizon), or every 25 min as a safety net.
                now = time.time()
                save_every = int(config.get('save_every_steps', 0) or 0)
                step_due = save_every > 0 and optim_step > 0 and optim_step % save_every == 0
                should_save = (
                    (optim_step == 10)
                    or step_due
                    or (not dry_run and (now - last_ckpt_time) >= 25 * 60)
                )
                if should_save and not dry_run:
                    save_ablation_checkpoint(model, optimizer, run_name, {},
                                             config, optim_step)
                    last_ckpt_time = now

    # Final eval (unwrap DataParallel + torch.compile)
    model_eval = _unwrap_model(model)
    val_metrics = evaluate_run(model_eval, val_ds, config, device)

    if not dry_run:
        save_ablation_checkpoint(model, optimizer, run_name, val_metrics,
                                 config, optim_step, is_final=True)

    metrics_dir = Path(run_name)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = str(metrics_dir / f"metrics_{run_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\n[Done] Metrics saved to {metrics_path}")

    if mount_path is not None:
        from hf_mount import unmount
        unmount(mount_path)

    return val_metrics


# =============================================================================
# 6.5. PyTorch inference (all ablations: byte, byte_dynamic, bpe)
# =============================================================================
# Loads a checkpoint saved by `save_ablation_checkpoint` (either bf16 resume
# or final compact 2-bit format) and generates autoregressively. Three paths,
# selected by config['tokenization']:
#
#   byte / byte_dynamic + LocalByteDecoder:
#       prompt bytes → P-byte patches → global stack (carry MLGRU state) →
#       LocalByteDecoder produces P bytes per patch (intra-patch AR).
#   byte / byte_dynamic + 4 independent heads (no LocalDecoder):
#       same global stack, but each new patch's 4 bytes are sampled
#       independently from the 4 heads (no intra-patch chain).
#   bpe:
#       Qwen2.5-3B tokenizer (vocab clamped to bpe_vocab_size, matching
#       training). One token per step. MLGRU runs carry state; transformer
#       runs re-forward the trailing window (sliding-window attn, no KV cache).
#
# The C/C++ kernels in inference/ are NOT used here — pure PyTorch.

def load_ablation_checkpoint(ckpt_path: str) -> tuple[dict, dict]:
    """Return (state_dict_fp32, config) from a bf16 or compact_2bit checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt.get('config')
    if config is None:
        raise ValueError(f"Checkpoint {ckpt_path} has no 'config' field")

    fmt = ckpt.get('format', 'bf16')
    if fmt == 'compact_2bit':
        state = unpack_compact_state(ckpt['compact'])
    else:
        state = {
            k: (v.to(torch.float32) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in ckpt['model_state_dict'].items()
        }
    return state, config


@torch.no_grad()
def _global_pass(
    model: 'AblationModel',
    patches: Tensor,
    hidden_states: list,
    elimination_rate: float,
) -> tuple[Tensor, list]:
    """Embedding → blocks (carry MLGRU state) → optional DeleteGate → final norm.

    Returns (h_post_norm [B, S, d_model], new_hidden_states).
    """
    h = model.embedding(patches)
    new_states: list = []
    for i, block in enumerate(model.blocks):
        h, h_new = block(h, hidden_states[i])
        new_states.append(h_new)

        if (model.use_delete_gate
                and i == model.delete_gate_pos
                and elimination_rate > 0.0):
            scores = model.delete_gate.scorer(h).squeeze(-1)
            mask = torch.sigmoid(scores / model.delete_gate.temperature)
            h = h * mask.unsqueeze(-1)

    return model.norm(h), new_states


def _sample_logits(
    logits: Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> int:
    """Sample one byte from a [256] logits tensor. Greedy when temperature<=0."""
    if temperature <= 0.0:
        return int(logits.argmax().item())

    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, k=min(top_k, logits.numel())).values[-1]
        logits = torch.where(logits < kth, torch.full_like(logits, -1e30), logits)

    probs = F.softmax(logits, dim=-1)

    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        cutoff = (cum > top_p).nonzero()
        keep = (cutoff[0, 0].item() + 1) if cutoff.numel() > 0 else sorted_probs.numel()
        kept_idx = sorted_idx[:keep]
        kept_probs = sorted_probs[:keep]
        kept_probs = kept_probs / kept_probs.sum()
        choice = torch.multinomial(kept_probs, num_samples=1).item()
        return int(kept_idx[choice].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def _generate_byte(
    model: 'AblationModel',
    prompt_bytes: bytes,
    max_new: int,
    config: dict,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    elimination_rate: float,
) -> bytes:
    """Byte / byte_dynamic generation, P bytes per global step.

    Uses LocalByteDecoder if present (autoregressive intra-patch); otherwise
    falls back to the 4 independent byte heads.
    """
    P = int(config.get('patch_size', 4))
    has_local = getattr(model, 'use_local_decoder', False)

    pad = (P - (len(prompt_bytes) % P)) % P
    padded = prompt_bytes + b'\x00' * pad
    if not padded:
        padded = b'\x00' * P
    arr = torch.frombuffer(bytearray(padded), dtype=torch.uint8).long()
    patches = arr.reshape(1, -1, P).to(device)

    hidden_states: list = [None] * len(model.blocks)
    h_post, hidden_states = _global_pass(
        model, patches, hidden_states, elimination_rate
    )
    h_last = h_post[:, -1:, :]

    out = bytearray()
    while len(out) < max_new:
        new_patch = torch.zeros(1, 1, P, dtype=torch.long, device=device)
        if has_local:
            for k in range(P):
                logits_tup = model.local_decoder(h_last, new_patch)
                b = _sample_logits(logits_tup[k][0, 0], temperature, top_k, top_p)
                new_patch[0, 0, k] = b
                out.append(b)
                if len(out) >= max_new:
                    break
        else:
            for k, head in enumerate(model.heads):
                logits_k = head(h_last)[0, 0]
                b = _sample_logits(logits_k, temperature, top_k, top_p)
                new_patch[0, 0, k] = b
                out.append(b)
                if len(out) >= max_new:
                    break
        if len(out) >= max_new:
            break

        h_post, hidden_states = _global_pass(
            model, new_patch, hidden_states, elimination_rate
        )
        h_last = h_post[:, -1:, :]

    return bytes(out[:max_new])


_BPE_TOKENIZER = None

def _get_qwen_tokenizer():
    """Lazy-load Qwen2.5-3B tokenizer (matches training)."""
    global _BPE_TOKENIZER
    if _BPE_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BPE_TOKENIZER = AutoTokenizer.from_pretrained(
            'Qwen/Qwen2.5-3B', trust_remote_code=True
        )
    return _BPE_TOKENIZER


@torch.no_grad()
def _generate_bpe(
    model: 'AblationModel',
    prompt: str,
    max_new: int,
    config: dict,
    device: torch.device,
    temperature: float,
    top_k: int,
    top_p: float,
    elimination_rate: float,
) -> str:
    """BPE generation — Qwen2.5-3B tokenizer, one token per step.

    MLGRU runs carry per-layer recurrent state (incremental forward of the
    last token only). Transformer runs re-forward the full sequence each
    step (sliding-window attention bounds the cost).
    """
    tok = _get_qwen_tokenizer()
    vocab_cap = int(config.get('bpe_vocab_size', 152064))
    is_mlgru = config.get('token_mixer', 'mlgru') == 'mlgru'

    enc = tok.encode(prompt, return_tensors='pt').to(device)  # [1, S]
    enc = torch.where(enc < vocab_cap, enc, torch.zeros_like(enc))
    generated = enc.clone()

    hidden_states: list = [None] * len(model.blocks)
    if is_mlgru:
        h_post, hidden_states = _global_pass(
            model, generated, hidden_states, elimination_rate
        )

    for _ in range(max_new):
        if is_mlgru:
            last_tok = generated[:, -1:]
            h_post, hidden_states = _global_pass(
                model, last_tok, hidden_states, elimination_rate
            )
            logits = model.heads[0](h_post)[0, -1]
        else:
            window = config.get('window_size', 512)
            ctx = generated[:, -window:] if generated.shape[1] > window else generated
            h_post, _ = _global_pass(model, ctx, hidden_states, elimination_rate)
            logits = model.heads[0](h_post)[0, -1]

        if logits.shape[-1] > vocab_cap:
            logits = logits[..., :vocab_cap]

        nxt = _sample_logits(logits, temperature, top_k, top_p)
        generated = torch.cat(
            [generated, torch.tensor([[nxt]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tok.decode(generated[0].tolist(), skip_special_tokens=True)


@torch.no_grad()
def generate(
    model: 'AblationModel',
    prompt: str,
    max_new: int,
    config: dict,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    elimination_rate: float = 0.0,
) -> str:
    """Dispatch to the right generation path based on config['tokenization'].

    `max_new` counts BYTES for byte/byte_dynamic runs and TOKENS for bpe.
    Returns the full decoded text (prompt + continuation).
    """
    model.eval()
    tok = config.get('tokenization', 'byte')

    if tok in ('byte', 'byte_dynamic'):
        prompt_bytes = prompt.encode('utf-8')
        out_bytes = _generate_byte(
            model, prompt_bytes, max_new, config, device,
            temperature, top_k, top_p, elimination_rate,
        )
        return (prompt_bytes + out_bytes).decode('utf-8', errors='replace')

    if tok == 'bpe':
        return _generate_bpe(
            model, prompt, max_new, config, device,
            temperature, top_k, top_p, elimination_rate,
        )

    raise ValueError(f"Unknown tokenization: {tok!r}")


def infer_run(
    ckpt_path: str,
    prompt: str,
    max_new_bytes: int = 256,
    temperature: float = 0.8,
    top_k: int = 0,
    top_p: float = 0.0,
    elimination_rate: float = 0.0,
    device: str | None = None,
) -> str:
    """Top-level entry: load ckpt, build model, generate, return decoded text.

    `max_new_bytes` counts bytes for byte runs and tokens for BPE runs.
    """
    state, config = load_ablation_checkpoint(ckpt_path)

    dev = torch.device(
        device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
    )

    config['use_gradient_checkpointing'] = False
    model = build_ablation_model(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Infer] Missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[Infer] Unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model = model.to(dev).eval()

    print(f"[Infer] device={dev} run={config.get('name', '?')} "
          f"tok={config.get('tokenization')} mixer={config.get('token_mixer')} "
          f"T={temperature} top_k={top_k} top_p={top_p} elim={elimination_rate}")

    return generate(
        model, prompt, max_new_bytes, config, dev,
        temperature=temperature, top_k=top_k, top_p=top_p,
        elimination_rate=elimination_rate,
    )


# =============================================================================
# 7. Lightning Studio orchestration — multi-GPU group runner
# =============================================================================
#
# PARALLEL_GROUPS: each sublist is a *group* whose runs execute concurrently
# (one process per GPU). Groups are processed in order. Runs with DeleteGate
# (TWO_PHASE_RUNS, derived from use_delete_gate) chain Phase 1 → Phase 2 in
# the same GPU slot.

PARALLEL_GROUPS: list[list[str]] = [
    # Group 0: FP16 baselines (Transformer + MLGRU, BPE + byte)
    ["run_01", "run_02a_byte_only_heads", "run_02", "run_03"],
    # Group 1: Ternary progression (quant → FPRes → kernel-strict → Bolmo)
    ["run_04", "run_05", "run_05b_kernel_strict", "run_06"],
    # Group 2: Complete stack + variants (DG Phase 1→2 chained)
    ["run_07", "run_08", "run_09", "run_10"],
    # Group 3: 10M scaling runs (fixed 1000 steps, NOT Chinchilla)
    ["run_13", "run_14", "run_16"],
]

# Any run with use_delete_gate=True needs Phase 2 to actually train the gate;
# Phase 1 alone instantiates DeleteGate but never activates it (forward gates
# on training_phase==2). Derive from configs so this stays in sync.
TWO_PHASE_RUNS = {
    name for name, cfg in ALL_RUN_CONFIGS.items()
    if cfg.get('use_delete_gate', False)
}


def install_requirements() -> None:
    print("=" * 60)
    print("PASO 0: Instalando dependencias")
    print("=" * 60)
    req = Path(__file__).parent / "requirements.txt"
    if not req.exists():
        print(f"[skip] {req} no existe.")
        return
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req), "-q"],
        check=True,
    )
    print("Dependencias instaladas.\n")


def check_env() -> int:
    """Print environment info, return number of visible GPUs."""
    print("=" * 60)
    print("PASO 1: Entorno")
    print("=" * 60)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus:
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.0f} GB)")
    else:
        print("  Sin GPUs visibles — dry-run forzado.")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("\n[AVISO] HF_TOKEN no encontrado.")
        print("  En Lightning Studio: Settings → Secrets → HF_TOKEN")
        print("  Sin token los checkpoints NO se subirán a HF Hub.\n")
    else:
        print(f"\nHF_TOKEN: ****{hf_token[-4:]}")
    return n_gpus


def _build_run_cmd(run_name: str, dry_run: bool, total_steps: int | None,
                   phase: int, phase1_ckpt: str | None) -> list[str]:
    cmd = [
        sys.executable, "-u",
        str(Path(__file__).resolve()),
        "--run", run_name,
        "--phase", str(phase),
    ]
    if dry_run:
        cmd.append("--dry_run")
    if total_steps is not None:
        cmd += ["--total_steps", str(total_steps)]
    if phase1_ckpt:
        cmd += ["--phase1_ckpt", phase1_ckpt]
    if not dry_run:
        cmd.append("--hf-mount")
    return cmd


def _spawn_run(run_name: str, gpu_id: int, dry_run: bool,
               total_steps: int | None, log_dir: Path) -> dict:
    """Spawn one run as a subprocess pinned to a single GPU."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    log_fp = open(log_path, "ab", buffering=0)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id) if gpu_id >= 0 else ""
    env["PYTHONUNBUFFERED"] = "1"

    cmd = _build_run_cmd(run_name, dry_run, total_steps, phase=1, phase1_ckpt=None)
    print(f"  [spawn] {run_name}  GPU={gpu_id}  →  {log_path.name}")
    proc = subprocess.Popen(cmd, stdout=log_fp, stderr=subprocess.STDOUT, env=env)

    return {
        "name":        run_name,
        "gpu":         gpu_id,
        "proc":        proc,
        "log_fp":      log_fp,
        "log_path":    log_path,
        "phase":       1,
        "started_at":  time.time(),
        "two_phase":   run_name in TWO_PHASE_RUNS and not dry_run,
        "dry_run":     dry_run,
        "total_steps": total_steps,
    }


def _maybe_advance_phase(slot: dict) -> bool:
    """If a Phase-1 of a two-phase run finished cleanly, kick off Phase 2 in
    the same GPU slot. Returns True if the slot is still busy after this call.
    """
    proc = slot["proc"]
    if proc.poll() is None:
        return True
    if slot["two_phase"] and slot["phase"] == 1 and proc.returncode == 0:
        run_name = slot["name"]
        ckpt = str(Path(run_name) / f"checkpoint_{run_name}_final.pt")
        if not Path(ckpt).exists():
            slot["log_fp"].write(
                f"\n[runner] Phase-1 final checkpoint missing: {ckpt} — aborting Phase 2.\n".encode()
            )
            return False
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(slot["gpu"]) if slot["gpu"] >= 0 else ""
        env["PYTHONUNBUFFERED"] = "1"
        cmd = _build_run_cmd(run_name, slot["dry_run"], slot["total_steps"],
                             phase=2, phase1_ckpt=ckpt)
        slot["log_fp"].write(b"\n[runner] === Phase 2 ===\n")
        slot["proc"] = subprocess.Popen(
            cmd, stdout=slot["log_fp"], stderr=subprocess.STDOUT, env=env,
        )
        slot["phase"] = 2
        return True
    return False


def run_group(runs: list[str], n_gpus: int, max_parallel: int,
              dry_run: bool, total_steps: int | None,
              log_dir: Path) -> dict:
    """Run all `runs` of a group with at most `max_parallel` concurrent procs."""
    parallel = max(1, min(max_parallel, max(n_gpus, 1)))
    print(f"\n--- Grupo de {len(runs)} run(s), paralelismo {parallel} ---")
    pending = list(runs)
    in_flight: list[dict] = []
    free_gpus: list[int] = list(range(parallel))
    rcodes: dict[str, int] = {}

    while pending or in_flight:
        while pending and free_gpus:
            gpu = free_gpus.pop(0)
            slot = _spawn_run(pending.pop(0), gpu, dry_run, total_steps, log_dir)
            in_flight.append(slot)

        time.sleep(2.0)

        still_alive: list[dict] = []
        for slot in in_flight:
            if _maybe_advance_phase(slot):
                still_alive.append(slot)
            else:
                rc = slot["proc"].returncode
                rcodes[slot["name"]] = rc
                slot["log_fp"].close()
                free_gpus.append(slot["gpu"])
                tag = "OK" if rc == 0 else f"FAIL rc={rc}"
                elapsed = (time.time() - slot["started_at"]) / 60
                print(f"  [done]  {slot['name']:8s}  GPU={slot['gpu']}  {tag}  ({elapsed:.1f} min)")
        in_flight = still_alive

    return rcodes


def print_summary(all_rcodes: dict[str, int], log_dir: Path) -> int:
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    ok = [r for r, rc in all_rcodes.items() if rc == 0]
    fail = [r for r, rc in all_rcodes.items() if rc != 0]
    print(f"OK:    {len(ok)}/{len(all_rcodes)}  →  {', '.join(ok) or '-'}")
    print(f"FAIL:  {len(fail)}                →  {', '.join(fail) or '-'}")
    print(f"Logs:  {log_dir}/")

    out = Path("run_results.json")
    out.write_text(json.dumps({"returncodes": all_rcodes}, indent=2))
    print(f"\nReturncodes guardados en {out}")
    return 0 if not fail else 1


def orchestrate(args: argparse.Namespace) -> int:
    """Run multiple ablations across GPUs by spawning one subprocess each."""
    if not args.skip_install:
        install_requirements()
    n_gpus = check_env()

    log_dir = Path(args.log_dir)

    if args.runs:
        groups = [list(args.runs)]
    elif args.groups is not None:
        groups = [PARALLEL_GROUPS[i] for i in args.groups]
    else:
        groups = list(PARALLEL_GROUPS)

    max_parallel = (
        args.max_parallel if args.max_parallel is not None else max(n_gpus, 1)
    )
    if args.dry_run:
        max_parallel = min(max_parallel, 2)

    print("\n" + "=" * 60)
    total_runs = sum(len(g) for g in groups)
    print(f"PASO 2: {total_runs} run(s) en {len(groups)} grupo(s), max_parallel={max_parallel}")
    print("=" * 60)

    all_rcodes: dict[str, int] = {}
    for gi, runs in enumerate(groups):
        print(f"\n========== Grupo {gi}: {runs} ==========")
        rc = run_group(runs, n_gpus=max_parallel, max_parallel=max_parallel,
                       dry_run=args.dry_run, total_steps=args.steps,
                       log_dir=log_dir)
        all_rcodes.update(rc)

    return print_summary(all_rcodes, log_dir)


# =============================================================================
# 8. CLI entry point
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CPU-1 ablation runs (single run or multi-GPU orchestration)"
    )

    # Single-run mode
    p.add_argument('--run', default=None,
                   choices=list(ALL_RUN_CONFIGS.keys()),
                   help="Train this single run (single-process mode).")
    p.add_argument('--dry_run', action='store_true',
                   help="Use random data on CPU (smoke test)")
    p.add_argument('--total_steps', type=int, default=None,
                   help="Override total training steps")
    p.add_argument('--phase', type=int, default=1, choices=[1, 2],
                   help="Training phase (1=distillation, 2=delete gate)")
    p.add_argument('--phase1_ckpt', type=str, default=None,
                   help="Path to Phase 1 checkpoint (required for --phase 2)")
    p.add_argument('--hf-mount', action='store_true',
                   help="Use hf-mount for lazy dataset loading")

    # Orchestration mode
    p.add_argument('--orchestrate', action='store_true',
                   help="Spawn group of subprocess runs across visible GPUs")
    p.add_argument('--groups', nargs='+', type=int, default=None,
                   help="Indices of PARALLEL_GROUPS to run (e.g. --groups 0 2)")
    p.add_argument('--runs', nargs='+', default=None,
                   help="Ad-hoc list of runs (overrides --groups)")
    p.add_argument('--steps', type=int, default=None,
                   help="Override total_steps for orchestration mode")
    p.add_argument('--dry-run', dest='dry_run', action='store_true',
                   help="Same as --dry_run, accepted for orchestration legacy")
    p.add_argument('--skip-install', action='store_true')
    p.add_argument('--max-parallel', type=int, default=None,
                   help="Cap concurrent runs (default: # GPUs)")
    p.add_argument('--log-dir', type=str, default='run_logs')

    return p.parse_args()


if __name__ == "__main__":
    # Dispatch on the two top-of-file bools — exactly one must be True.
    if TRAIN and INFER:
        raise SystemExit(
            "Both TRAIN and INFER are True. Set exactly one (top of file)."
        )
    if not TRAIN and not INFER:
        raise SystemExit(
            "Neither TRAIN nor INFER is True. Set exactly one (top of file)."
        )

    if INFER:
        cfg = INFER_CONFIG
        if not cfg.get('ckpt'):
            raise SystemExit("INFER_CONFIG['ckpt'] is empty — set the checkpoint path.")
        text = infer_run(
            ckpt_path=cfg['ckpt'],
            prompt=cfg.get('prompt', ''),
            max_new_bytes=cfg.get('max_new_bytes', 256),
            temperature=cfg.get('temperature', 0.8),
            top_k=cfg.get('top_k', 0),
            top_p=cfg.get('top_p', 0.0),
            elimination_rate=cfg.get('elimination_rate', 0.0),
            device=cfg.get('device'),
        )
        print("\n----- generated -----")
        print(text)
        sys.exit(0)

    # TRAIN mode — existing CLI behaviour (single run or orchestration).
    args = parse_args()

    if args.orchestrate:
        sys.exit(orchestrate(args))

    if args.run is None:
        raise SystemExit(
            "Provide --run <name> for single-run mode, or --orchestrate for the "
            "multi-GPU group runner. Use -h for the full list of runs."
        )

    if args.phase == 2 and not args.phase1_ckpt:
        raise SystemExit("--phase1_ckpt is required when --phase 2")

    metrics = train_run(args.run, dry_run=args.dry_run,
                        total_steps_override=args.total_steps,
                        phase=args.phase,
                        phase1_ckpt=args.phase1_ckpt,
                        use_hf_mount=args.hf_mount)
    print(f"\nFinished: {args.run}")
    print(json.dumps(metrics, indent=2))
