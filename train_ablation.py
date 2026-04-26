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

Inference belongs in `inference/` (C/C++ kernels).
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
)


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
#   10M runs (run_13..run_15): Cukinator/cpu1-ablation-dataset (~37.5M tokens,
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
}


# =============================================================================
# 2. Per-run delta configs — 12 progressive 50M runs
# =============================================================================
#
# Story: start from a standard FP16 Transformer baseline and add ONE CPU-1
# design choice per run until reaching the full architecture.
#
# Progressive chain:
#   run_01  Transformer + BPE + FP16            (absolute baseline)
#     ↓ swap tokenizer
#   run_02  Transformer + Byte + FP16
#     ↓ swap mixer + keep byte
#   run_03  MLGRU + Byte + FP16
#     ↓ quantize
#   run_04  MLGRU + Byte + Ternary
#     ↓ correct
#   run_05  MLGRU + Byte + Ternary + FPRes      ← CPU-1 core
#     ↓ branch: add components
#   run_06  CPU-1 core + Bolmo                  (+dynamic byte patches)
#   run_07  CPU-1 core + DeleteGate             (+state elimination)
#   run_08  Folded Transformer + Byte + Ternary (alternative: local attn)
#   run_09  CPU-1 core + Bolmo + DeleteGate     ← CPU-1 COMPLETE ⭐
#   run_10  CPU-1 COMPLETE + PFNet
#   run_11  CPU-1 COMPLETE + per-channel decay
#   run_12  CPU-1 COMPLETE + state reset

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
    'run_02': {
        'name':         'transformer_byte_fp16',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'fp16',
        'fp_residual':  False,
        'description':  '+Byte patches — FP16 Transformer with 4-byte patches. Isolates tokenization.',
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
    'run_06': {
        'name':         'mlgru_byte_ternary_fpres_bolmo',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'description':  (
            '+Bolmo — dynamic byte patches aligned to word boundaries. '
            'CPU-1 core + boundary predictor.'
        ),
        'cites':        ['BLT (Pagnoni et al., 2024)'],
    },
    'run_07': {
        'name':         'mlgru_byte_ternary_fpres_deletegate',
        'token_mixer':  'mlgru',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_delete_gate': True,
        'description':  (
            '+DeleteGate — MrT5-style state elimination at midpoint (~40%). '
            'CPU-1 core + delete gate.'
        ),
        'cites':        ['MrT5 (Pagliardini et al., 2024)'],
    },
    'run_08': {
        'name':         'folded_transformer_byte_ternary',
        'token_mixer':  'transformer',
        'tokenization': 'byte',
        'quantization': 'ternary',
        'fp_residual':  False,
        'window_size':  128,
        'description':  (
            'Folded Transformer — ternary sliding-window attention over byte patches. '
            'O(N·128) vs O(N²). Alternative to MLGRU.'
        ),
        'cites':        [
            'Longformer (Beltagy et al., 2020)',
            'BitNet b1.58 (Wang et al., 2024)',
        ],
    },
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
    'run_11': {
        'name':         'cpu1_decay_learned',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'mlgru_per_channel_decay': True,
        'description':  (
            'CPU-1 COMPLETE + per-channel learnable decay in MLGRU (RWKV/HGRN2-style). '
            'Each hidden dim has its own decay rate — d_model × n_layers scalars total.'
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
    'run_12': {
        'name':         'cpu1_decay_reset',
        'token_mixer':  'mlgru',
        'tokenization': 'byte_dynamic',
        'quantization': 'ternary',
        'fp_residual':  True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'state_reset_period': 512,
        'description':  (
            'CPU-1 COMPLETE + periodic MLGRU state reset every 512 tokens (Stuffed Mamba). '
            'Zero extra parameters — pure structural forgetting heuristic.'
        ),
        'cites':        [
            'MatMul-free LM (Zhu et al., 2024)',
            'BitNet b1.58 (Wang et al., 2024)',
            'BLT (Pagnoni et al., 2024)',
            'MrT5 (Pagliardini et al., 2024)',
            'Stuffed Mamba (anonymous, 2024)',
        ],
    },
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

    'total_steps':  2500,
    'warmup_steps': 250,

    # 10M runs use the pre-processed Cukinator dataset; ~37.5M tokens is enough
    # at 10M × 20 = 200M target tokens (with ~5× cycling). run_16 overrides
    # this back to FineWeb because its question is "no teacher".
    'dataset_repo': 'Cukinator/cpu1-ablation-dataset',
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
    'run_15': {
        'name':                   'small_cpu1_complement',
        'token_mixer':            'mlgru',
        'tokenization':           'byte_dynamic',
        'quantization':           'ternary',
        'fp_residual':            True,
        'use_boundary_predictor': True,
        'use_delete_gate':        True,
        'complement_from':        'run_13',
        'description':            'CPU-1 COMPLETE @ 10M, complement: BPE warm-start (run_13) '
                                  '→ byte finetune.',
        'cites':                  ['MatMul-free LM', 'Slender-Mamba', 'Bolmo', 'MrT5'],
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
            emb_cls = (
                BolmoPatchEmbedding if self.use_bolmo_gating
                else AblationBytePatchEmbedding
            )
            self.embedding = emb_cls(
                d_model,
                patch_size=config.get('patch_size', 4),
                use_ternary=use_ternary,
            )
        else:
            vocab_size = config.get('bpe_vocab_size', 152064)
            from utilities import make_embedding
            self.embedding = make_embedding(vocab_size, d_model, use_ternary)

        self.blocks = nn.ModuleList([
            AblationBlock(config) for _ in range(n_layers)
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
        self.use_emb_align = config.get('use_embedding_alignment', False)
        self.delete_gate_pos = n_layers // 2

        if self.use_boundary:
            self.boundary_predictor = BoundaryPredictor(embed_dim=8, hidden_dim=32)

        if self.use_delete_gate:
            self.delete_gate = DeleteGate(d_model=d_model, elimination_rate=0.0)

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
        x_emb = self.embedding(x)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states: list = []
        extra_losses: dict = {}
        h = x_emb

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _fn(_h, _hs, _b=block):
                    return _b(_h, _hs)
                h, h_new = checkpoint(_fn, h, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)

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

                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    h, _ = self.delete_gate.forward_train(
                        h, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss

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
            boundary_probs = self.boundary_predictor.forward_train(raw_bytes)
            targets = (
                (raw_bytes == 32) | (raw_bytes == 10) | (raw_bytes >= 192)
            ).float()
            extra_losses['boundary'] = F.binary_cross_entropy(
                boundary_probs, targets
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
    if (config.get('chinchilla', True)
            and total_steps_override is None
            and not dry_run
            and phase == 1):
        n_params = sum(p.numel() for p in model.parameters())
        new_total = chinchilla_total_steps(config, n_params)
        new_save = chinchilla_save_every(new_total, config.get('checkpoint_fraction', 0.05))
        new_warmup = chinchilla_warmup(new_total, config.get('warmup_ratio', 0.05))
        tokens_per_step = (
            config['batch_per_gpu'] * config['gradient_accumulation_steps']
            * config['seq_len']
        )
        print(
            f"[Chinchilla] N={n_params/1e6:.1f}M params · "
            f"target={n_params * config.get('chinchilla_tokens_per_param', 20)/1e6:.1f}M tokens · "
            f"tokens/step={tokens_per_step:,} · "
            f"total_steps {config.get('total_steps')} → {new_total} · "
            f"warmup {config.get('warmup_steps')} → {new_warmup} ({config.get('warmup_ratio', 0.05)*100:.0f}% of horizon) · "
            f"save_every {new_save} ({config.get('checkpoint_fraction', 0.05)*100:.0f}% of horizon)"
        )
        config['total_steps'] = new_total
        config['warmup_steps'] = new_warmup
        config['save_every_steps'] = new_save

    # Phase 2: load Phase 1 checkpoint, activate DeleteGate, freeze EmbAligner
    if phase == 2 and phase1_ckpt:
        print(f"[Phase 2] Loading Phase 1 checkpoint: {phase1_ckpt}")
        ckpt = torch.load(phase1_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if hasattr(model, 'delete_gate'):
            model.delete_gate.set_elimination_rate(
                config.get('target_elimination', 0.4)
            )
        if hasattr(model, 'embedding_aligner'):
            for p in model.embedding_aligner.parameters():
                p.requires_grad = False
        print("[Phase 2] DeleteGate active, EmbAligner frozen")

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
    total_batch = config['batch_per_gpu'] * max(n_gpus, 1)
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

    bitlinear_params = []
    head_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'weight' in name and any(
            k in name for k in ['forget_proj', 'cand_proj', 'output_proj',
                                 'gate_proj', 'up_proj', 'down_proj',
                                 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        ):
            bitlinear_params.append(p)
        elif 'heads.' in name and 'proj.weight' in name:
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
    # with DataParallel — skip when multi-GPU.
    if device.type == 'cuda' and not dry_run and n_gpus <= 1:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Opt] torch.compile enabled for training")
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
                target_elim = config.get('target_elimination', 0.0)
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

    # Final eval (unwrap DataParallel)
    model_eval = model.module if hasattr(model, 'module') else model
    val_metrics = evaluate_run(model_eval, val_ds, config, device)

    if not dry_run:
        save_ablation_checkpoint(model, optimizer, run_name, val_metrics,
                                 config, optim_step, is_final=True)

    metrics_path = f"metrics_{run_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\n[Done] Metrics saved to {metrics_path}")

    if mount_path is not None:
        from hf_mount import unmount
        unmount(mount_path)

    return val_metrics


# =============================================================================
# 7. Lightning Studio orchestration — multi-GPU group runner
# =============================================================================
#
# PARALLEL_GROUPS: each sublist is a *group* whose runs execute concurrently
# (one process per GPU). Groups are processed in order. Runs with DeleteGate
# (run_07, run_09) chain Phase 1 → Phase 2 in the same GPU slot. run_15 needs
# to come after run_13 because it warm-starts from run_13's checkpoint.

PARALLEL_GROUPS: list[list[str]] = [
    ["run_01", "run_02", "run_03", "run_04"],
    ["run_05", "run_06", "run_07", "run_08"],
    ["run_09", "run_10", "run_11", "run_12"],
    ["run_13", "run_14", "run_16"],
    ["run_15"],
]

TWO_PHASE_RUNS = {"run_07", "run_09"}


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
        ckpt = f"checkpoint_{run_name}_final.pt"
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
