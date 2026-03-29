"""
Ablation study for CPU-1: configuration, models, and analysis.

Section 1 - Configuration: 15 run configs at 50M params on Kaggle T4.
Section 2 - Models: configurable model variants for all ablation runs.
Section 3 - Analysis: paper-ready tables (text + LaTeX).

Usage:
    from ablation import get_run_config, build_ablation_model, RUN_CONFIGS

    config = get_run_config("run_10")
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
# Base config shared by ALL 15 runs (50M params)
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
    'lambda_align': 0.3,            # EmbeddingAligner weight (runs 13, 15)
    'teacher_dim':  128,            # projected dim for EmbeddingAligner

    # Dataset
    'dataset_repo':     'Cukinator/cpu1-ablation-dataset',
    'seq_len':          5000,
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


# ============================================================
# Extended runs 12-15: Bolmo + MrT5 + EmbeddingAlignment
# ============================================================

# Run 12: Bolmo solo (parches dinámicos, sin MrT5 ni EmbAlign)
RUN_CONFIGS['run_12'] = {
    'name':                    'mlgru_byte_ternary_distil_bolmo',
    'token_mixer':             'mlgru',
    'tokenization':            'byte_dynamic',   # parches dinámicos
    'quantization':            'ternary',
    'training_signal':         'distillation',
    'fp_residual':             True,
    'use_boundary_predictor':  True,
    'use_delete_gate':         False,
    'use_embedding_alignment': False,
    'description': (
        'CPU-1 + Bolmo (parches dinámicos). '
        'Compara contra Run 10 (parches estáticos). '
        'Responde: ¿los parches dinámicos mejoran PPL o TPS?'
    ),
    'cites': ['Bolmo (Allen AI, 2025)'],
}

# Run 13: EmbeddingAlignment solo (sin Bolmo ni MrT5)
RUN_CONFIGS['run_13'] = {
    'name':                    'mlgru_byte_ternary_distil_embalign',
    'token_mixer':             'mlgru',
    'tokenization':            'byte',
    'quantization':            'ternary',
    'training_signal':         'distillation',
    'fp_residual':             True,
    'use_boundary_predictor':  False,
    'use_delete_gate':         False,
    'use_embedding_alignment': True,
    'lambda_align':            0.3,
    'description': (
        'CPU-1 + EmbeddingAlignment. '
        'Compara contra Run 10 (sin alineación). '
        'Responde: ¿alinear embeddings mejora la calidad del conocimiento transferido?'
    ),
    'cites': ['arXiv 2602 (2025) — Distilling Byte-Level Models'],
}

# Run 14: MrT5 (DeleteGate, 2 fases) — sobre Run 10
RUN_CONFIGS['run_14'] = {
    'name':                    'mlgru_byte_ternary_distil_deletegate',
    'token_mixer':             'mlgru',
    'tokenization':            'byte',
    'quantization':            'ternary',
    'training_signal':         'distillation_then_ce',  # Fase 1 distil, Fase 2 CE
    'fp_residual':             True,
    'use_boundary_predictor':  False,
    'use_delete_gate':         True,
    'target_elimination':      0.4,             # eliminar 40% de estados
    'training_phases':         2,               # entrenar en 2 fases
    'use_embedding_alignment': False,
    'description': (
        'CPU-1 + DeleteGate (MrT5 style). Training en 2 fases. '
        'Compara TPS vs Run 10: ¿cuánto más rápido en CPU con 40% menos estados? '
        'Compara PPL vs Run 10: ¿cuánta calidad se pierde?'
    ),
    'cites': ['MrT5 (Google, 2024)'],
}

# Run 15: Todo junto — Bolmo + EmbAlign + MrT5 (la propuesta final)
RUN_CONFIGS['run_15'] = {
    'name':                    'mlgru_byte_ternary_distil_full',
    'token_mixer':             'mlgru',
    'tokenization':            'byte_dynamic',
    'quantization':            'ternary',
    'training_signal':         'distillation_then_ce',
    'fp_residual':             True,
    'use_boundary_predictor':  True,
    'use_delete_gate':         True,
    'use_embedding_alignment': True,
    'target_elimination':      0.4,
    'lambda_align':            0.3,
    'training_phases':         2,
    'description': (
        'CPU-1 COMPLETO v3 — todas las técnicas integradas. '
        'Propuesta final del paper si los runs 12-14 muestran mejoras individuales. '
        'Bolmo mejora entrada, EmbAlign mejora conocimiento, MrT5 mejora velocidad.'
    ),
    'cites': [
        'Bolmo (Allen AI, 2025)',
        'MrT5 (Google, 2024)',
        'arXiv 2602 (2025)',
    ],
    'is_proposed_v3': True,
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
        if tok in ('byte', 'byte_dynamic'):
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
    ) -> tuple:
        """
        Args:
            x: [B, S, 4] uint8/long for byte, [B, S] long for BPE
            hidden_states: list of per-layer MLGRU states (None for transformer)
            training_phase: 1 = distillation, 2 = with delete gate
            target_elimination: fraction of states to eliminate (Phase 2)
            teacher_hidden: [B, S, teacher_dim] float — projected teacher hidden states
            teacher_mask: [B, S] bool — where teacher signal exists

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
            logits = tuple(head(h) for head in self.heads)
        else:
            logits = self.heads[0](h)

        # EmbeddingAlignment (Phase 1 only)
        if (self.use_emb_align and training_phase == 1
                and teacher_hidden is not None and teacher_mask is not None):
            extra_losses['emb_align'] = self.embedding_aligner.alignment_loss(
                h_pre_norm, teacher_hidden, teacher_mask,
            )

        return logits, new_states, extra_losses

    def count_parameters(self) -> dict:
        """Return parameter counts by component."""
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.embedding.parameters())
        blocks = sum(p.numel() for p in self.blocks.parameters())
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
    proposed = results.get('run_10', {})
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
        r"\caption{Ablation study at 50M parameters, 1B tokens. "
        r"TPS = tokens per second on consumer CPU (4-core). "
        r"$\downarrow$ lower is better, $\uparrow$ higher is better.}",
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

    lines.append("1. MLGRU vs Transformer (BPE FP16, CE):")
    lines.append(_compare("  Mixer effect", 'run_01', 'run_05'))

    lines.append("2. Byte vs BPE (Transformer FP16, CE):")
    lines.append(_compare("  Tokenization effect", 'run_01', 'run_02'))

    lines.append("3. Ternary vs FP16 (Transformer BPE, CE):")
    lines.append(_compare("  Quantization effect", 'run_01', 'run_03'))

    lines.append("4. Distillation vs CE (Transformer BPE ternary):")
    lines.append(_compare("  Training signal effect", 'run_03', 'run_04'))

    lines.append("5. Full proposal vs baseline:")
    lines.append(_compare("  CPU-1 vs Transformer baseline", 'run_01', 'run_10'))

    lines.append("6. FPResidual ablation:")
    lines.append(_compare("  With vs without FPResidual", 'run_11', 'run_10'))

    lines.append("7. Distillation effect on CPU-1:")
    lines.append(_compare("  CE vs distillation (MLGRU+Byte+Ternary)",
                           'run_09', 'run_10'))

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
