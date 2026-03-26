"""
Configurable model variants for the CPU-1 ablation study.

Supports all 4 combinations of {Transformer, MLGRU} × {BPE, Byte},
with configurable quantization (FP16 vs ternary) and FPResidual.

Usage:
    from ablation_models import build_ablation_model
    from ablation_config import get_run_config

    config = get_run_config('run_10')
    model = build_ablation_model(config)
"""

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_v2 import (
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
                teacher_dim=config.get('teacher_dim', 5120),
            )

    def forward(
        self,
        x: Tensor,
        hidden_states: list | None = None,
        training_phase: int = 1,
        target_elimination: float = 0.0,
    ) -> tuple:
        """
        Args:
            x: [B, S, 4] uint8/long for byte, [B, S] long for BPE
            hidden_states: list of per-layer MLGRU states (None for transformer)
            training_phase: 1 = distillation, 2 = with delete gate
            target_elimination: fraction of states to eliminate (Phase 2)

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

        h = self.norm(h)

        if self.tokenization in ('byte', 'byte_dynamic'):
            logits = tuple(head(h) for head in self.heads)
        else:
            logits = self.heads[0](h)

        # EmbeddingAlignment placeholder (Phase 1 only)
        if self.use_emb_align and training_phase == 1:
            extra_losses['emb_align'] = None  # populated externally

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
