"""
Shared utilities for CPU-1 training (production model + ablation runs).

Contents:
  1. Constants (D_MODEL, VOCAB_SIZE, etc.)
  2. Quantization primitives — RMSNorm, BitLinear, BitEmbedding, FPResidual
  3. Building blocks — SwiGLU, PFNet, BytePatchEmbeddingV2,
     MatMulFreeTokenMixer, BitHead, LocalByteDecoder, CPU1BlockV2
  4. Boundary / gating modules — BoundaryPredictor, DeleteGate, EmbeddingAligner
  5. Byte ↔ patch helpers — bytes_to_patches, bytes_to_dynamic_patches
  6. Checkpoint serialization — pack_model_compact / unpack_compact_state
  7. Training infrastructure — CUDAPrefetcher, datasets, LR schedules,
     param groups, distillation/CE loss, checkpointing, evaluation

Inference utilities live in `inference/` (C/C++ kernels).
Evaluation suite lives in `eval/`.
"""

import os
import gc
import math
import json
import time
import contextlib
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# 1. Global configuration
# =============================================================================

D_MODEL = 1024
N_LAYERS = 14
D_FF = 2730
FP_RANK = 8
PFNET_HIDDEN = 64
VOCAB_SIZE = 256
PATCH_SIZE = 4


# =============================================================================
# 2. Quantization primitives
# =============================================================================

def _ste_round(x: Tensor) -> Tensor:
    """Round con Straight-Through Estimator: forward redondea, backward pasa directo."""
    return x + (x.round() - x).detach()


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BitLinear(nn.Module):
    """Linear con cuantización ternaria {-1, 0, 1} y STE."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight
        w_scale = w.abs().mean()
        w_normalized = w / (w_scale + 1e-8)
        w_ternary = torch.clamp(_ste_round(w_normalized), -1.0, 1.0)
        x_scale = x.abs().amax(dim=-1, keepdim=True)
        x_normalized = x / (x_scale + 1e-8)
        x_quant = torch.clamp(_ste_round(x_normalized * 127.0), -128.0, 127.0)
        output = F.linear(x_quant, w_ternary)
        output = output * (x_scale * w_scale / 127.0)
        return output


class BitEmbedding(nn.Module):
    """1.58-bit ternary embedding (Slender-Mamba, COLING 2025).

    Quantizes the embedding weight matrix to {-1, 0, 1} with one FP16 scale
    per row. Keeps memory roughly equal to a int8 embedding while staying
    consistent with the rest of the ternary stack.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(vocab_size, 1))

    def forward(self, indices: Tensor) -> Tensor:
        w_scale = self.weight.abs().mean(dim=1, keepdim=True)
        w_ternary = (self.weight / (w_scale + 1e-8)).round().clamp(-1, 1)
        w_ternary = w_ternary.detach() + self.weight - self.weight.detach()  # STE
        return F.embedding(indices, w_ternary * self.scale)


class FPResidual(nn.Module):
    """Low-rank FP16 correction applied alongside a ternary BitLinear path."""

    def __init__(self, in_dim: int, out_dim: int | None = None, rank: int = FP_RANK):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fp_A = nn.Linear(in_dim, rank, bias=False)
        self.fp_B = nn.Linear(rank, out_dim, bias=False)
        nn.init.normal_(self.fp_A.weight, std=0.02)
        nn.init.zeros_(self.fp_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.fp_B(F.silu(self.fp_A(x)))


# =============================================================================
# 3. Factory helpers (shared by ablation: FP16 or ternary at construction time)
# =============================================================================

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


# =============================================================================
# 4. SwiGLU FFN (production + configurable variants)
# =============================================================================

class SwiGLU(nn.Module):
    """SwiGLU FFN — production CPU-1 version (always ternary, with FPResidual)."""

    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)
        self.up_proj = BitLinear(d_model, d_ff)
        self.down_proj = BitLinear(d_ff, d_model)
        self.down_fp_residual = FPResidual(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        return self.down_proj(hidden) + self.down_fp_residual(hidden)


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


# =============================================================================
# 5. PFNet — cache-resident nonlinear residual branch
# =============================================================================
# Per layer:  (d_model × hidden + hidden × d_model) ternary ≈ 16 KB at d=1024,h=64
# 14 layers × 16 KB = 224 KB → fits in L2 alongside the 28 KB MLGRU state.

class PFNet(nn.Module):
    """Tiny 2-layer ternary MLP residual that stays L2-resident across layers."""

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.w1 = BitLinear(d_model, hidden)
        self.w2 = BitLinear(hidden, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(self.norm(x))))


# =============================================================================
# 6. Patch embeddings — production + configurable + Bolmo-gated variants
# =============================================================================

class BytePatchEmbeddingV2(nn.Module):
    """Production byte-patch embedding: BitEmbedding + BitLinear projection."""

    def __init__(self, d_model: int = D_MODEL, patch_size: int = PATCH_SIZE,
                 bytes_per_embed: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.bytes_per_embed = bytes_per_embed
        self.byte_embed = BitEmbedding(256, bytes_per_embed)
        self.patch_proj = BitLinear(patch_size * bytes_per_embed, d_model)

    def forward(self, patches: Tensor) -> Tensor:
        B, S, P = patches.shape
        embeds = self.byte_embed(patches.long())
        embeds = embeds.reshape(B, S, P * self.bytes_per_embed)
        return self.patch_proj(embeds)


class AblationBytePatchEmbedding(nn.Module):
    """Configurable byte-patch embedding (FP16 or ternary)."""

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
        B, S, P = patches.shape
        embeds = self.byte_embed(patches.long())
        embeds = embeds.reshape(B, S, P * self.bytes_per_embed)
        return self.patch_proj(embeds)


class BolmoPatchEmbedding(nn.Module):
    """Bolmo / BLT-style byte-patch embedding with integrated boundary scoring.

    Embeds each byte, scores each position with a tiny 1D conv, and amplifies
    high-boundary positions before the patch projection. Caches pre-sigmoid
    logits in `last_boundary_logits` so the parent model can compute a BCE
    loss against natural-boundary targets.
    """

    def __init__(self, d_model: int, patch_size: int = 4,
                 bytes_per_embed: int = 8, use_ternary: bool = True):
        super().__init__()
        self.patch_size = patch_size
        self.bytes_per_embed = bytes_per_embed
        self.byte_embed = make_embedding(256, bytes_per_embed, use_ternary)
        self.patch_proj = make_linear(
            patch_size * bytes_per_embed, d_model, use_ternary
        )
        self.boundary_scorer = nn.Conv1d(
            in_channels=bytes_per_embed,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        self.last_boundary_logits: Tensor | None = None

    def forward(self, patches: Tensor) -> Tensor:
        B, S, P = patches.shape
        d_b = self.bytes_per_embed
        byte_feats = self.byte_embed(patches.long())                  # [B, S, P, d_b]

        conv_in = byte_feats.reshape(B, S * P, d_b).transpose(1, 2)   # [B, d_b, S*P]
        bndry_logits = self.boundary_scorer(conv_in).squeeze(1)       # [B, S*P]
        self.last_boundary_logits = bndry_logits

        gate = torch.sigmoid(bndry_logits).reshape(B, S, P, 1)
        byte_feats = byte_feats * (1.0 + gate)

        flat = byte_feats.reshape(B, S, P * d_b)
        return self.patch_proj(flat)


# =============================================================================
# 7. Token mixers — production MLGRU, configurable MLGRU, sliding-window attn
# =============================================================================

class MatMulFreeTokenMixer(nn.Module):
    """Production CPU-1 MLGRU (MatMul-free token mixer, Zhu et al. 2024).

    State [d_model] only; no KV cache. Always ternary, always with FPResidual.
    """

    def __init__(self, d_model: int, fp_rank: int = FP_RANK):
        super().__init__()
        self.d_model = d_model
        self.forget_proj = BitLinear(d_model, d_model)
        self.cand_proj = BitLinear(d_model, d_model)
        self.output_proj = BitLinear(d_model, d_model)
        self.fp_residual = FPResidual(d_model, d_model, rank=fp_rank)
        self.register_buffer('h_init', torch.zeros(d_model))

    def forward(self, x: Tensor, h: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        B, S, D = x.shape
        if h is None:
            h = self.h_init.unsqueeze(0).expand(B, -1).clone()

        f = torch.sigmoid(self.forget_proj(x))
        c = F.silu(self.cand_proj(x))
        g = torch.sigmoid(self.output_proj(x))

        outputs = torch.empty_like(f)
        for t in range(S):
            h = torch.lerp(c[:, t], h, f[:, t])  # f*h + (1-f)*c
            outputs[:, t] = g[:, t] * h

        return outputs + self.fp_residual(x), h


class AblationMLGRU(nn.Module):
    """Configurable MLGRU with optional per-channel decay and state reset."""

    def __init__(self, d_model: int, use_ternary: bool = True,
                 fp_rank: int = 4,
                 per_channel_decay: bool = False,
                 state_reset_period: int = 0):
        super().__init__()
        self.d_model = d_model
        self.state_reset_period = state_reset_period
        self.forget_proj = make_linear(d_model, d_model, use_ternary)
        self.cand_proj = make_linear(d_model, d_model, use_ternary)
        self.output_proj = make_linear(d_model, d_model, use_ternary)
        self.fp_residual = (
            FPResidual(d_model, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )
        self.register_buffer('h_init', torch.zeros(d_model))
        self.log_decay = (
            nn.Parameter(torch.zeros(d_model)) if per_channel_decay else None
        )

    def forward(self, x: Tensor, h: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        B, S, D = x.shape
        if h is None:
            h = self.h_init.unsqueeze(0).expand(B, -1).clone()

        f = torch.sigmoid(self.forget_proj(x))
        c = F.silu(self.cand_proj(x))
        g = torch.sigmoid(self.output_proj(x))

        if self.log_decay is not None:
            decay = torch.exp(-F.softplus(self.log_decay))
            f = decay * f

        outputs = torch.empty_like(f)
        reset = self.state_reset_period
        for t in range(S):
            if reset > 0 and t > 0 and t % reset == 0:
                h = torch.zeros_like(h)
            h = torch.lerp(c[:, t], h, f[:, t])
            outputs[:, t] = g[:, t] * h

        out = outputs
        if self.fp_residual is not None:
            out = out + self.fp_residual(x)
        return out, h


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

        if S <= self.window_size:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            mask = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            band = torch.ones(S, S, dtype=torch.bool, device=x.device).triu(
                diagonal=-(self.window_size - 1)
            )
            attn_mask = mask & band
            float_mask = torch.zeros(S, S, device=x.device, dtype=x.dtype)
            float_mask.masked_fill_(~attn_mask, float('-inf'))
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        result = self.o_proj(out)
        if self.fp_residual is not None:
            result = result + self.fp_residual(x)
        return result


# =============================================================================
# 8. Prediction heads
# =============================================================================

class BitHead(nn.Module):
    """Production byte prediction head — single ternary projection."""

    def __init__(self, d_model: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.proj = BitLinear(d_model, vocab_size)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x) * self.output_scale


class AblationByteHead(nn.Module):
    """Configurable byte head (FP16 or ternary)."""

    def __init__(self, d_model: int, use_ternary: bool = True):
        super().__init__()
        self.proj = make_linear(d_model, 256, use_ternary)
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x) * self.scale


class AblationBPEHead(nn.Module):
    """Configurable BPE head over arbitrary vocabulary."""

    def __init__(self, d_model: int, vocab_size: int,
                 use_ternary: bool = True):
        super().__init__()
        self.proj = make_linear(d_model, vocab_size, use_ternary)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


# =============================================================================
# 9. LocalByteDecoder — MegaByte-style intra-patch autoregression
# =============================================================================

class LocalByteDecoder(nn.Module):
    """MegaByte (Yu et al., 2023, Listing 1) local model for byte prediction.

    Replaces the four independent byte heads with an autoregressive chain:
        P(b_0|ctx) · P(b_1|b_0, ctx) · P(b_2|b_1, b_0, ctx) · ...

    Single global → P × d_local projection feeds a 1-layer causal transformer
    over the P-byte axis, conditioned on shifted-right byte embeddings and
    learned position embeddings.
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
            h:       [B, S, d_model] global hidden state (post final RMSNorm)
            targets: [B, S, P] long — ground-truth bytes of the predicted patch
        Returns:
            tuple of P tensors, each [B, S, 256] — logits byte_0..byte_{P-1}
        """
        B, S, _ = h.shape
        P = self.patch_size

        g = self.h_proj(h).reshape(B, S, P, self.d_local)

        zero_pad = torch.zeros_like(targets[..., :1])
        shifted = torch.cat([zero_pad, targets[..., :-1]], dim=-1)
        byte_embs = self.byte_embed(shifted.long())

        local_in = g + byte_embs + self.pos_embed
        local_in = local_in.reshape(B * S, P, self.d_local)
        local_out = self._attn_ffn(local_in)
        local_out = local_out.reshape(B, S, P, self.d_local)

        logits = self.out_proj(local_out) * self.out_scale
        return tuple(logits[:, :, k, :] for k in range(P))


# =============================================================================
# 10. CPU1BlockV2 — production block (MLGRU + SwiGLU + PFNet)
# =============================================================================

class CPU1BlockV2(nn.Module):
    """Pre-norm MLGRU block + SwiGLU FFN + PFNet nonlinear residual."""

    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF,
                 fp_rank: int = FP_RANK, pfnet_hidden: int = PFNET_HIDDEN):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.token_mixer = MatMulFreeTokenMixer(d_model, fp_rank)
        self.ffn = SwiGLU(d_model, d_ff)
        self.pfnet = PFNet(d_model, hidden=pfnet_hidden)

    def forward(self, x: Tensor, h: Tensor | None = None
                ) -> tuple[Tensor, Tensor]:
        residual = x
        mixer_out, h_new = self.token_mixer(self.norm1(x), h)
        x = residual + mixer_out
        residual = x
        x = self.ffn(self.norm2(x))
        x = residual + x + self.pfnet(residual)
        return x, h_new


# =============================================================================
# 11. AblationBlock — configurable mixer (MLGRU or sliding-window attention)
# =============================================================================

class AblationBlock(nn.Module):
    """Unified block for ablation runs: pre-norm → mixer → residual → FFN."""

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
            self.token_mixer = AblationMLGRU(
                d_model, use_ternary, fp_rank,
                per_channel_decay=config.get('mlgru_per_channel_decay', False),
                state_reset_period=config.get('state_reset_period', 0),
            )
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


# =============================================================================
# 12. BoundaryPredictor — Bolmo-style word-boundary scorer
# =============================================================================

class BoundaryPredictor(nn.Module):
    """Tiny 2-byte-context boundary scorer, used as an auxiliary training loss.

    Lookahead of 1 byte: takes byte[i] and byte[i+1], predicts whether byte[i]
    is a patch boundary. ~2 KB of weights; sized to vanish in compute against
    the main model. Supervised with `get_natural_boundaries` labels at training
    time. Inference-side decoding belongs in `inference/`.
    """

    def __init__(self, embed_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim)
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_train(
        self,
        bytes_seq: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        b_curr = self.byte_embed(bytes_seq[:, :-1].long())
        b_next = self.byte_embed(bytes_seq[:, 1:].long())
        ctx = torch.cat([b_curr, b_next], dim=-1)
        probs = self.predictor(ctx).squeeze(-1)
        last = torch.ones(bytes_seq.shape[0], 1,
                          device=bytes_seq.device, dtype=probs.dtype)
        return torch.cat([probs, last], dim=1)

    @staticmethod
    def get_natural_boundaries(byte_seq: bytes) -> list[bool]:
        """Generate boundary labels from raw bytes for supervised training.

        Boundaries: spaces, punctuation, UTF-8 multibyte starts, max 8-byte runs.
        """
        boundaries = [False] * len(byte_seq)
        last_boundary = 0
        PUNCT = set(b'.,!?;:\n\t')

        for i, b in enumerate(byte_seq):
            if i - last_boundary >= 8:
                boundaries[i] = True
                last_boundary = i
                continue

            is_boundary = (
                b == 0x20 or
                b in PUNCT or
                (b >= 0xC0) or
                (i > 0 and byte_seq[i - 1] == 0x20)
            )
            if is_boundary and i > last_boundary:
                boundaries[i] = True
                last_boundary = i

        if boundaries:
            boundaries[-1] = True
        return boundaries


# =============================================================================
# 13. DeleteGate — MrT5-style mid-stack state elimination
# =============================================================================

class DeleteGate(nn.Module):
    """Soft-masking gate placed at layer N//2 to learn to drop redundant states.

    Phase 1 training (target_rate=0): pass-through, no elimination loss.
    Phase 2 training (target_rate>0): soft mask + elimination regularization.
    """

    def __init__(self, d_model: int, elimination_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.elimination_rate = elimination_rate
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.register_buffer('temperature', torch.tensor(1.0))
        self.last_elimination_loss = torch.tensor(0.0)

    def forward_train(
        self,
        hidden: Tensor,
        target_rate: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        if target_rate == 0.0:
            mask = torch.ones(hidden.shape[:2], device=hidden.device)
            self.last_elimination_loss = torch.tensor(0.0, device=hidden.device)
            return hidden, mask

        scores = self.scorer(hidden).squeeze(-1)
        mask = torch.sigmoid(scores / self.temperature)
        self.last_elimination_loss = (
            (1 - mask).mean() - target_rate
        ).pow(2)
        return hidden * mask.unsqueeze(-1), mask

    def set_elimination_rate(self, rate: float) -> None:
        self.elimination_rate = rate


# =============================================================================
# 14. EmbeddingAligner — distillation-style cosine alignment to a teacher
# =============================================================================

class EmbeddingAligner(nn.Module):
    """Project student hiddens into the teacher dimension and align by cosine.

    Active only in Phase 1 (frozen during Phase 2 / DeleteGate training).
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def alignment_loss(
        self,
        student_hidden: Tensor,
        teacher_hidden: Tensor,
        teacher_mask: Tensor,
        aggregation: str = 'mean_pool',
    ) -> Tensor:
        student_proj = self.projection(student_hidden)
        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_hidden.float(), dim=-1)

        if teacher_mask.any():
            min_S = min(student_norm.shape[1], teacher_norm.shape[1])
            s_masked = student_norm[:, :min_S][teacher_mask[:, :min_S]]
            t_masked = teacher_norm[:, :min_S][teacher_mask[:, :min_S]]
            if s_masked.numel() > 0:
                cosine_sim = (s_masked * t_masked).sum(dim=-1)
                return (1 - cosine_sim).mean()
        return torch.tensor(0.0, device=student_hidden.device)


# =============================================================================
# 15. Byte ↔ patch helpers
# =============================================================================

def bytes_to_patches(raw: bytes) -> Tensor:
    """Convert a raw byte string to a [1, n_patches, PATCH_SIZE] tensor."""
    n = len(raw)
    pad = (PATCH_SIZE - n % PATCH_SIZE) % PATCH_SIZE
    raw = raw + b'\x00' * pad
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.reshape(1, -1, PATCH_SIZE).long()


def bytes_to_dynamic_patches(
    byte_seq: bytes,
    boundaries: list[bool],
    max_patch_size: int = 8,
    pad_byte: int = 0,
) -> list[list[int]]:
    """Pack bytes into variable-length patches (1..max_patch_size), padded."""
    patches = []
    current_patch: list[int] = []

    for b, is_boundary in zip(byte_seq, boundaries):
        current_patch.append(b)
        if is_boundary or len(current_patch) >= max_patch_size:
            padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
            patches.append(padded[:max_patch_size])
            current_patch = []

    if current_patch:
        padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
        patches.append(padded[:max_patch_size])

    return patches


# =============================================================================
# 16. Checkpoint serialization — 2-bit ternary packing
# =============================================================================
# Encoding: {-1, 0, 1} → {0b00, 0b01, 0b10}, 4 values per byte.
# 50M ablation: ~14 MB compact; 170M production: ~44 MB.

def pack_ternary_2bit(weight: Tensor) -> tuple[Tensor, Tensor]:
    """Quantize then pack a ternary weight tensor into 2-bit uint8."""
    scale = weight.abs().mean().float()
    w = (weight.float() / (scale + 1e-8)).round().clamp(-1, 1)
    enc = (w + 1).to(torch.uint8).flatten()

    pad = (-enc.numel()) % 4
    if pad:
        enc = torch.cat([enc, enc.new_zeros(pad)])

    packed = (
        enc[0::4]
        | (enc[1::4] << 2)
        | (enc[2::4] << 4)
        | (enc[3::4] << 6)
    )
    return packed.cpu(), scale.cpu()


def unpack_2bit_to_float(packed: Tensor, scale: float, shape: tuple) -> Tensor:
    """Reverse `pack_ternary_2bit` back to float32."""
    n = 1
    for s in shape:
        n *= s
    out = torch.empty(packed.numel() * 4, dtype=torch.float32)
    for i in range(4):
        out[i::4] = ((packed >> (2 * i)) & 0b11).float() - 1
    return out[:n].reshape(shape) * scale


def pack_model_compact(model: nn.Module) -> dict:
    """Build a compact state dict — 2-bit ternary + bfloat16 scalars."""
    ternary_keys: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, (BitLinear, BitEmbedding)):
            ternary_keys.add(f'{mod_name}.weight')

    compact: dict = {}
    for key, val in model.state_dict().items():
        val = val.detach()
        if key in ternary_keys:
            packed, scale = pack_ternary_2bit(val)
            compact[f'__2bit_data__{key}'] = packed
            compact[f'__2bit_scale__{key}'] = scale
            compact[f'__2bit_shape__{key}'] = list(val.shape)
        elif val.is_floating_point():
            compact[key] = val.to(torch.bfloat16).cpu()
        else:
            compact[key] = val.cpu()
    return compact


def unpack_compact_state(compact: dict) -> dict:
    """Reverse `pack_model_compact` into a float32 state dict."""
    meta: dict[str, dict] = {}
    regular: dict[str, Tensor] = {}

    for key, val in compact.items():
        if key.startswith('__2bit_data__'):
            meta.setdefault(key[len('__2bit_data__'):], {})['data'] = val
        elif key.startswith('__2bit_scale__'):
            meta.setdefault(key[len('__2bit_scale__'):], {})['scale'] = val
        elif key.startswith('__2bit_shape__'):
            meta.setdefault(key[len('__2bit_shape__'):], {})['shape'] = val
        else:
            regular[key] = (
                val.float() if isinstance(val, Tensor) and val.is_floating_point()
                else val
            )

    state: dict = dict(regular)
    for orig_key, m in meta.items():
        state[orig_key] = unpack_2bit_to_float(
            m['data'], float(m['scale']), tuple(m['shape'])
        )
    return state


# =============================================================================
# 17. CUDA prefetcher — overlaps H2D copies with GPU compute
# =============================================================================

class CUDAPrefetcher:
    """Async double-buffered batch prefetcher with a dedicated CUDA stream."""

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device
        self.use_cuda = device.type == 'cuda'
        self.stream = torch.cuda.Stream(device=device) if self.use_cuda else None
        self._iter = None
        self._next: dict | None = None

    def __iter__(self):
        self._iter = iter(self.loader)
        self._next = None
        self._preload()
        return self

    def _to_device(self, batch: dict) -> dict:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _preload(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            self._next = None
            return
        if not self.use_cuda:
            self._next = batch
            return
        with torch.cuda.stream(self.stream):
            self._next = self._to_device(batch)

    def __next__(self):
        if self._next is None:
            raise StopIteration
        if self.use_cuda:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self._next
        self._preload()
        return batch


# =============================================================================
# 18. Distillation loss — production training (Qwen teacher → CPU-1 student)
# =============================================================================

def distillation_loss(
    student_logits: tuple[Tensor, Tensor, Tensor, Tensor],
    target_bytes: Tensor,
    teacher_probs: Tensor | None,
    teacher_mask: Tensor,
    temperature: float = 4.0,
    alpha_kl: float = 0.9,
    alpha_ce: float = 0.1,
    compute_metrics: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """Byte-level distillation loss against a first-byte-marginalized teacher.

    Head 0: KL(student/T, teacher/T) * T² + CE where the teacher has signal,
            CE-only elsewhere.
    Heads 1-3: CE only (teacher BPE has no signal on intra-token bytes).
    """
    metrics: dict[str, float] = {}

    logits_0 = student_logits[0]
    targets_0 = target_bytes[:, :, 0].long()

    ce_loss_0 = F.cross_entropy(
        logits_0.reshape(-1, 256),
        targets_0.reshape(-1),
        reduction='none',
    ).reshape_as(targets_0)

    if teacher_probs is not None and teacher_mask is not None and teacher_mask.any():
        student_log_probs_T = F.log_softmax(logits_0 / temperature, dim=-1)
        teacher_probs_T = teacher_probs ** (1.0 / temperature)
        teacher_probs_T = teacher_probs_T / (
            teacher_probs_T.sum(dim=-1, keepdim=True) + 1e-8
        )

        kl = F.kl_div(
            student_log_probs_T,
            teacher_probs_T,
            reduction='none',
            log_target=False,
        ).sum(dim=-1)

        kl_scaled = kl * (temperature ** 2)
        mask_float = teacher_mask.float()
        loss_with_teacher = alpha_kl * kl_scaled + alpha_ce * ce_loss_0
        head0_loss = mask_float * loss_with_teacher + (1.0 - mask_float) * ce_loss_0

        if compute_metrics:
            n_teacher = mask_float.sum().clamp(min=1)
            metrics['loss_head0_kl'] = (kl_scaled * mask_float).sum().item() / n_teacher.item()
            metrics['teacher_coverage'] = mask_float.mean().item()
    else:
        head0_loss = ce_loss_0
        if compute_metrics:
            metrics['loss_head0_kl'] = 0.0
            metrics['teacher_coverage'] = 0.0

    total_loss = head0_loss.mean()
    if compute_metrics:
        metrics['loss_head0_ce'] = ce_loss_0.mean().item()

    heads_ce_sum = 0.0
    for i in range(1, 4):
        logits_i = student_logits[i]
        targets_i = target_bytes[:, :, i].long()
        ce_i = F.cross_entropy(
            logits_i.reshape(-1, 256),
            targets_i.reshape(-1),
            reduction='mean',
        )
        total_loss = total_loss + ce_i
        if compute_metrics:
            heads_ce_sum += ce_i.item()

    total_loss = total_loss / 4.0
    if compute_metrics:
        metrics['loss_heads_123_ce'] = heads_ce_sum / 3.0
        metrics['loss_total'] = total_loss.item()

    return total_loss, metrics


def compute_loss(
    student_logits,
    targets,
    teacher_probs,
    teacher_mask,
    config: dict,
    teacher_ids: Tensor | None = None,
) -> Tensor:
    """Ablation loss — dispatches to BPE single-head or byte 4-head as needed."""
    tok = config['tokenization']
    sig = config['training_signal']
    T = config['temperature']
    alpha_kl = config['alpha_kl']
    alpha_ce = config['alpha_ce']

    if tok == 'bpe':
        logits = student_logits
        vocab_size = logits.shape[-1]

        ce = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none',
        ).reshape(targets.shape)

        if (sig in ('distillation', 'distillation_then_ce')
                and teacher_probs is not None
                and teacher_mask is not None
                and teacher_ids is not None):
            teacher_full = torch.zeros_like(logits)
            teacher_full.scatter_(2, teacher_ids, teacher_probs.to(logits.dtype))
            teacher_full = teacher_full / teacher_full.sum(-1, keepdim=True).clamp(min=1e-8)

            student_log_T = F.log_softmax(logits / T, dim=-1)
            teacher_T = (teacher_full / T).softmax(dim=-1)
            kl = F.kl_div(
                student_log_T, teacher_T,
                reduction='none', log_target=False,
            ).sum(dim=-1) * T ** 2

            mask = teacher_mask.float()
            loss = mask * (alpha_kl * kl + alpha_ce * ce) + (1 - mask) * ce
            return loss.mean()

        return ce.mean()

    # Byte 4-head path
    if sig in ('distillation', 'distillation_then_ce') and teacher_probs is not None:
        loss, _ = distillation_loss(
            student_logits, targets,
            teacher_probs, teacher_mask,
            temperature=T, alpha_kl=alpha_kl, alpha_ce=alpha_ce,
        )
    else:
        no_mask = torch.zeros(
            targets.shape[0], targets.shape[1],
            dtype=torch.bool, device=targets.device,
        )
        loss, _ = distillation_loss(
            student_logits, targets, None, no_mask,
            temperature=T, alpha_kl=alpha_kl, alpha_ce=alpha_ce,
        )
    return loss


# =============================================================================
# 19. LR schedules
# =============================================================================

def get_lr(
    step: int,
    warmup_steps: int = 2000,
    total_steps: int = 100_000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def get_lr_from_config(step: int, config: dict) -> float:
    """Same schedule, but reads parameters from a run-config dict."""
    ws = config['warmup_steps']
    ts = config['total_steps']
    lr = config['lr']
    min_lr = config['min_lr']
    if step < ws:
        return lr * step / max(ws, 1)
    p = (step - ws) / max(ts - ws, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


# =============================================================================
# 20. Param groups (production)
# =============================================================================

def build_param_groups(model: nn.Module) -> list[dict]:
    """Split params into 4 groups: BitLinear (wd=0.1), no-decay, heads, PFNet."""
    decay_params = []
    no_decay_params = []
    head_params = []
    pfnet_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if '.pfnet.' in name:
            pfnet_params.append(param)
        elif ('head_' in name or 'byte_heads' in name) and 'weight' in name:
            head_params.append(param)
        elif any(k in name for k in ('fp_A', 'fp_B', 'norm', 'bias',
                                      'byte_embedding', 'byte_embed',
                                      'output_scale', '.scale', 'pos_embed')):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': head_params, 'weight_decay': 0.01},
        {'params': pfnet_params, 'weight_decay': 0.05, 'lr_scale': 1.5},
    ]


# =============================================================================
# 21. Production-style checkpointing (full FP32, used by train_final.py)
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    loss: float,
    path: str,
    upload_to_hf: bool = False,
    hf_repo: str = "Cukinator/cpu1-checkpoints",
):
    """Save a full resume checkpoint and optionally push to HF Hub."""
    model_to_save = model.module if hasattr(model, 'module') else model
    checkpoint = {
        'step': step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

    if upload_to_hf:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=Path(path).name,
                repo_id=hf_repo,
                repo_type="model",
            )
        except Exception as e:
            print(f"[WARN] No se pudo subir checkpoint a HF Hub: {e}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    """Load a checkpoint into model (and optionally optimizer/scaler)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint.get('step', 0)


def save_ablation_checkpoint(model: nn.Module, optimizer, run_name: str,
                             metrics: dict, config: dict, step: int,
                             is_final: bool = False) -> str:
    """Save ablation checkpoint: BF16 resume during training, 2-bit packed at end."""
    path = (
        f"checkpoint_{run_name}_final.pt"
        if is_final
        else f"checkpoint_{run_name}_step{step}.pt"
    )

    model_to_save = model.module if hasattr(model, 'module') else model

    if is_final:
        torch.save({
            'compact': pack_model_compact(model_to_save),
            'metrics': metrics,
            'step': step,
            'run_name': run_name,
            'config': config,
            'format': 'compact_2bit',
        }, path)
        print(f"[Checkpoint] Final compact saved: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    else:
        model_bf16 = {
            k: v.to(torch.bfloat16) if v.is_floating_point() else v
            for k, v in model_to_save.state_dict().items()
        }
        opt_state = optimizer.state_dict()
        for group in opt_state.get('state', {}).values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in group and group[key].is_floating_point():
                    group[key] = group[key].to(torch.bfloat16)

        torch.save({
            'model_state_dict': model_bf16,
            'optimizer_state_dict': opt_state,
            'metrics': metrics,
            'step': step,
            'run_name': run_name,
            'config': config,
            'format': 'bf16',
        }, path)
        print(f"[Checkpoint] Resume checkpoint saved: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")

    hub_repo = config.get('checkpoint_repo')
    if hub_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"ablation/{path}",
                repo_id=hub_repo,
            )
            print(f"[Checkpoint] Uploaded to HF: {path}")
        except Exception as e:
            print(f"[Checkpoint] Hub upload failed: {e}")

    if not is_final:
        import glob as _glob
        for old in _glob.glob(f"checkpoint_{run_name}_step*.pt"):
            if old != path:
                try:
                    os.remove(old)
                    print(f"[Checkpoint] Removed old: {old}")
                except OSError:
                    pass

    return path


# =============================================================================
# 22. Datasets — production distillation, ablation byte/BPE, dry-run, FineWeb
# =============================================================================

class DistillationDataset(Dataset):
    """Production training dataset: pre-computed patches + first-byte teacher.

    Supports HuggingFace Datasets / Parquet / Hub repos and a dry-run mode
    (random tensors) for smoke-testing the training loop without data.
    """

    def __init__(
        self,
        data_path: str | None = None,
        seq_len: int = 2048,
        dry_run_samples: int = 0,
        teacher_coverage: float = 0.25,
    ):
        self.seq_len = seq_len

        if dry_run_samples > 0:
            self.mode = 'dry_run'
            self.n_samples = dry_run_samples
            self.teacher_coverage = teacher_coverage
        elif data_path is not None:
            self.mode = 'hf'
            self._load_hf(data_path)
        else:
            raise ValueError("Necesitas data_path o dry_run_samples > 0")

    def _load_hf(self, data_path: str):
        from datasets import load_from_disk, load_dataset
        p = Path(data_path)
        if p.is_dir() and (p / 'dataset_info.json').exists():
            self.dataset = load_from_disk(data_path)
        elif p.suffix == '.parquet':
            self.dataset = load_dataset('parquet', data_files=str(p), split='train')
        elif p.is_dir():
            shards = sorted(p.glob('shard_*.parquet'))
            if not shards:
                shards = sorted(p.glob('**/*.parquet'))
            self.dataset = load_dataset(
                'parquet', data_files=[str(s) for s in shards], split='train',
            )
        else:
            from huggingface_hub import list_repo_files
            try:
                files = [
                    f"hf://datasets/{data_path}/{f}"
                    for f in list_repo_files(data_path, repo_type='dataset')
                    if f.startswith('byte_marginalized/') and f.endswith('.parquet')
                ]
                if files:
                    self.dataset = load_dataset('parquet', data_files=files, split='train')
                else:
                    self.dataset = load_dataset(data_path, split='train')
            except Exception:
                self.dataset = load_dataset(data_path, split='train')

    def __len__(self) -> int:
        return self.n_samples if self.mode == 'dry_run' else len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.mode == 'dry_run':
            return self._get_dry_run_item()
        return self._get_hf_item(idx)

    def _get_dry_run_item(self) -> dict[str, Tensor]:
        S = self.seq_len
        patches = torch.randint(0, 256, (S, PATCH_SIZE), dtype=torch.long)
        targets = torch.randint(0, 256, (S, PATCH_SIZE), dtype=torch.long)

        teacher_probs = torch.zeros(S, VOCAB_SIZE)
        teacher_mask = torch.rand(S) < self.teacher_coverage

        if teacher_mask.sum().item() > 0:
            for i in teacher_mask.nonzero(as_tuple=True)[0]:
                n_active = torch.randint(3, 12, (1,)).item()
                active = torch.randperm(256)[:n_active]
                teacher_probs[i, active] = torch.softmax(torch.randn(n_active), dim=0)

        teacher_hidden = torch.zeros(S, 128, dtype=torch.float32)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
            'teacher_hidden': teacher_hidden,
        }

    def _get_hf_item(self, idx: int) -> dict[str, Tensor]:
        item = self.dataset[idx]

        def _to_tensor(raw, dtype):
            if isinstance(raw, str):
                raw = json.loads(raw)
            return torch.as_tensor(raw, dtype=dtype)

        patches = _to_tensor(item['patches'], torch.long)
        targets = _to_tensor(item['targets'], torch.long)

        seq_len = patches.numel() // PATCH_SIZE
        patches = patches.view(seq_len, PATCH_SIZE)
        targets = targets.view(seq_len, PATCH_SIZE)

        if 'teacher_probs' in item and item['teacher_probs'] is not None:
            teacher_probs = _to_tensor(item['teacher_probs'], torch.float32)
            teacher_probs = teacher_probs.view(seq_len, VOCAB_SIZE)
        else:
            teacher_probs = torch.zeros(seq_len, VOCAB_SIZE)

        if 'teacher_mask' in item and item['teacher_mask'] is not None:
            teacher_mask = _to_tensor(item['teacher_mask'], torch.bool)
        else:
            teacher_mask = torch.zeros(seq_len, dtype=torch.bool)

        if 'teacher_hidden' in item and item['teacher_hidden'] is not None:
            t_dim = int(item.get('teacher_dim', 128))
            teacher_hidden = _to_tensor(item['teacher_hidden'], torch.float32)
            teacher_hidden = teacher_hidden.view(seq_len, t_dim)
        else:
            teacher_hidden = torch.zeros(seq_len, 128, dtype=torch.float32)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
            'teacher_hidden': teacher_hidden,
        }


class DryRunDataset(Dataset):
    """Random-tensor dataset for ablation smoke tests."""

    def __init__(self, config: dict, n_samples: int = 1000):
        self.config = config
        self.n_samples = n_samples
        self.seq_len = config['seq_len']
        self.tok = config['tokenization']
        self.sig = config['training_signal']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.tok in ('byte', 'byte_dynamic'):
            patches = torch.randint(0, 256, (self.seq_len, 4))
            targets = torch.randint(0, 256, (self.seq_len, 4))
            sample = {'patches': patches, 'targets': targets}
            if self.sig in ('distillation', 'distillation_then_ce'):
                sample['teacher_probs'] = torch.zeros(self.seq_len, 256)
                sample['teacher_mask'] = torch.zeros(self.seq_len, dtype=torch.bool)
            if self.config.get('use_embedding_alignment'):
                teacher_dim = self.config.get('teacher_dim', 128)
                sample['teacher_hidden'] = torch.randn(self.seq_len, teacher_dim)
        else:
            vocab = self.config.get('bpe_vocab_size', 152064)
            input_ids = torch.randint(0, vocab, (self.seq_len,))
            labels = torch.randint(0, vocab, (self.seq_len,))
            sample = {'input_ids': input_ids, 'labels': labels}
            if self.sig == 'distillation':
                sample['teacher_probs_bpe'] = torch.zeros(self.seq_len, 128)
                sample['teacher_ids_bpe'] = torch.zeros(
                    self.seq_len, 128, dtype=torch.long
                )
                sample['teacher_mask_bpe'] = torch.zeros(
                    self.seq_len, dtype=torch.bool
                )
        return sample


class ByteLevelDataset(Dataset):
    """Wraps a HuggingFace byte-level dataset (cpu1-ablation-dataset format)."""

    def __init__(self, hf_dataset, config: dict):
        self.ds = hf_dataset
        self.config = config
        self.seq_len = config['seq_len']
        self.patch_size = config.get('patch_size', 4)
        self.has_teacher = config['training_signal'] in (
            'distillation', 'distillation_then_ce'
        )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        S, P = self.seq_len, self.patch_size
        patches = torch.tensor(row['patches'], dtype=torch.long).reshape(S, P)
        targets = torch.tensor(row['targets'], dtype=torch.long).reshape(S, P)
        sample = {'patches': patches, 'targets': targets}
        if self.has_teacher and 'teacher_probs' in row:
            sample['teacher_probs'] = (
                torch.tensor(row['teacher_probs'], dtype=torch.float32)
                .reshape(S, 256)
            )
            sample['teacher_mask'] = torch.tensor(
                row['teacher_mask'], dtype=torch.bool
            )
        if 'teacher_hidden' in row and row['teacher_hidden'] is not None:
            teacher_dim = self.config.get('teacher_dim', 128)
            if teacher_dim > 0:
                sample['teacher_hidden'] = (
                    torch.tensor(row['teacher_hidden'], dtype=torch.float32)
                    .reshape(S, teacher_dim)
                )
        return sample


class BPEDataset(Dataset):
    """Wraps a HuggingFace BPE dataset (cpu1-ablation-dataset bpe_tokenized)."""
    TOP_K = 128

    def __init__(self, hf_dataset, config: dict):
        self.ds = hf_dataset
        self.seq_len = config['seq_len']
        self.has_teacher = config['training_signal'] in (
            'distillation', 'distillation_then_ce'
        )
        self.vocab_cap = int(config.get('bpe_vocab_size', 152064))

    def __len__(self):
        return len(self.ds)

    def _clamp_ids(self, t: torch.Tensor) -> torch.Tensor:
        if self.vocab_cap >= 152064:
            return t
        return torch.where(t < self.vocab_cap, t, torch.zeros_like(t))

    def __getitem__(self, idx):
        row = self.ds[idx]
        S, K = self.seq_len, self.TOP_K
        input_ids = torch.tensor(row['input_ids'], dtype=torch.long)
        labels = torch.tensor(row['labels'], dtype=torch.long)
        sample = {
            'input_ids': self._clamp_ids(input_ids),
            'labels':    self._clamp_ids(labels),
        }
        if self.has_teacher and 'teacher_probs_bpe' in row:
            sample['teacher_probs_bpe'] = (
                torch.tensor(row['teacher_probs_bpe'], dtype=torch.float32)
                .reshape(S, K)
            )
            sample['teacher_ids_bpe'] = (
                torch.tensor(row['teacher_ids_bpe'], dtype=torch.long)
                .reshape(S, K)
            )
            sample['teacher_mask_bpe'] = torch.tensor(
                row['teacher_mask_bpe'], dtype=torch.bool
            )
        return sample


class FineWebStreamingDataset(torch.utils.data.IterableDataset):
    """Stream raw HuggingFaceFW/fineweb sample-10BT and tokenize on the fly.

    50M ablation runs and run_16 use this; the pre-processed Cukinator dataset
    is too small for their Chinchilla budget. Mirrors create_dataset.py params
    (seed=42, seq_len=5000, patch_size=4, max 700 BPE tokens / 2800 bytes).
    """

    MAX_EXTRACTION_TOKENS = 700
    MAX_EXTRACTION_BYTES = 700 * 4

    def __init__(self, config: dict, split: str = 'train'):
        super().__init__()
        self.config = config
        self.tok = config['tokenization']
        self.seq_len = int(config['seq_len'])
        self.patch_size = int(config.get('patch_size', 4))
        self.seed = int(config.get('seed', 42))
        self.split = split
        self.vocab_cap = int(config.get('bpe_vocab_size', 152064))
        self.val_modulus = 20
        self.is_val = (split != 'train')
        self._tokenizer = None

    def _load_tokenizer(self):
        if self._tokenizer is None and self.tok == 'bpe':
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                'Qwen/Qwen2.5-3B', trust_remote_code=True
            )
        return self._tokenizer

    def _doc_iter(self, worker_id: int, num_workers: int):
        from datasets import load_dataset
        ds = load_dataset(
            'HuggingFaceFW/fineweb', name='sample-10BT', split='train',
            streaming=True,
        )
        worker_seed = self.seed + 1000 * worker_id
        ds = ds.shuffle(seed=worker_seed, buffer_size=10_000)
        for i, doc in enumerate(ds):
            if i % num_workers != worker_id:
                continue
            train_doc = (i % self.val_modulus) != 0
            if train_doc == self.is_val:
                continue
            text = doc.get('text', '') or ''
            if text.strip():
                yield text

    def _byte_pack(self, docs):
        S, P = self.seq_len, self.patch_size
        need = S * P + P
        buf = bytearray()
        for text in docs:
            data = text.encode('utf-8', errors='replace')
            if len(data) > self.MAX_EXTRACTION_BYTES:
                data = data[:self.MAX_EXTRACTION_BYTES]
            buf.extend(data)
            buf.append(0x0A)
            while len(buf) >= need:
                window = bytes(buf[:need])
                del buf[:S * P]
                arr = torch.tensor(list(window), dtype=torch.long)
                patches = arr[:S * P].reshape(S, P)
                targets = arr[P:S * P + P].reshape(S, P)
                yield {'patches': patches, 'targets': targets}

    def _bpe_pack(self, docs):
        S = self.seq_len
        tok = self._load_tokenizer()
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        buf: list[int] = []
        for text in docs:
            ids = tok(
                text, add_special_tokens=False,
                truncation=True, max_length=self.MAX_EXTRACTION_TOKENS,
            )['input_ids']
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= S + 1:
                seq = buf[:S + 1]
                del buf[:S]
                arr = torch.tensor(seq, dtype=torch.long)
                if self.vocab_cap < 152064:
                    arr = torch.where(arr < self.vocab_cap, arr, torch.zeros_like(arr))
                yield {'input_ids': arr[:S], 'labels': arr[1:S + 1]}

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        docs = self._doc_iter(worker_id, num_workers)
        if self.tok in ('byte', 'byte_dynamic'):
            yield from self._byte_pack(docs)
        elif self.tok == 'bpe':
            yield from self._bpe_pack(docs)
        else:
            raise ValueError(f"Unknown tokenization for streaming: {self.tok}")


def load_ablation_dataset(config: dict, split: str = 'train',
                          dry_run: bool = False,
                          mount_path: Path | None = None):
    """Resolve the right dataset class for an ablation run config.

    Routing:
      - dataset_repo == 'HuggingFaceFW/fineweb' → FineWebStreamingDataset
      - dataset_repo == 'Cukinator/cpu1-ablation-dataset' → pre-processed
        parquet shards (subset auto-selected by tokenization).
    """
    if dry_run:
        return DryRunDataset(config)

    repo = config['dataset_repo']
    tok = config['tokenization']

    if repo == 'HuggingFaceFW/fineweb':
        print(f"[fineweb] Streaming sample-10BT for split='{split}' "
              f"(seed={config.get('seed', 42)}, tokenization='{tok}')")
        return FineWebStreamingDataset(config, split=split)

    from datasets import load_dataset
    subset = 'bpe_tokenized' if tok == 'bpe' else 'byte_marginalized'

    from huggingface_hub import list_repo_files, hf_hub_download

    if mount_path is not None:
        parquet_dir = Path(mount_path) / subset
        all_files = sorted(str(f) for f in parquet_dir.glob('shard_*.parquet'))
        if not all_files:
            raise FileNotFoundError(f"No shards in {parquet_dir}")
        ds = load_dataset('parquet', data_files=all_files, split='train')
        print(f"[mount] Loaded {len(ds)} rows from {len(all_files)} shards ({subset})")
    else:
        try:
            repo_files = list(list_repo_files(repo, repo_type='dataset'))
            hub_files = [
                f"hf://datasets/{repo}/{f}"
                for f in repo_files
                if f.startswith(f'{subset}/') and f.endswith('.parquet')
            ]
            if not hub_files:
                raise FileNotFoundError(f"No shards for {subset}/ in {repo}")
            ds = load_dataset('parquet', data_files=hub_files, split='train')
            print(f"[hub] Loaded {len(ds)} rows from {len(hub_files)} shards ({subset})")
        except Exception as e:
            print(f"[hub] Fallback to load_dataset: {e}")
            ds = load_dataset(repo, subset, split=split)

    if split != 'all':
        split_info = None
        local_split = (
            Path(mount_path) / subset / 'split_info.json' if mount_path else None
        )
        if local_split and local_split.exists():
            with open(local_split) as f:
                split_info = json.load(f)
        else:
            try:
                split_path = hf_hub_download(
                    repo_id=repo, filename=f'{subset}/split_info.json',
                    repo_type='dataset',
                )
                with open(split_path) as f:
                    split_info = json.load(f)
            except Exception:
                pass

        if split_info and split in split_info:
            indices = [i for i in split_info[split] if i < len(ds)]
            ds = ds.select(indices)
            print(f"[split] {split}: {len(ds)} sequences")
        elif split == 'train':
            pass
        else:
            n = max(1, len(ds) // 20)
            ds = ds.select(range(len(ds) - n, len(ds)))
            print(f"[split] {split} (fallback last 5%): {len(ds)} sequences")

    if tok in ('byte', 'byte_dynamic'):
        return ByteLevelDataset(ds, config)
    return BPEDataset(ds, config)


# =============================================================================
# 23. CPU throughput + validation evaluation (training-time metrics)
# =============================================================================

def measure_cpu_throughput(model: nn.Module, config: dict,
                           n_tokens: int = 100) -> float:
    """Measure tokens/sec on CPU for this model variant (training-time metric).

    Heavy benchmarks live in `eval/eval_performance.py`; this is the lightweight
    end-of-run number that gets stamped into the metrics JSON.
    """
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_cpu.eval()
    tok = config['tokenization']

    if tok in ('byte', 'byte_dynamic'):
        dummy = torch.zeros(1, 1, 4, dtype=torch.long)
    else:
        dummy = torch.zeros(1, 1, dtype=torch.long)

    with torch.no_grad():
        for _ in range(5):
            model_cpu(dummy)

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_tokens):
            model_cpu(dummy)
    t1 = time.perf_counter()

    model.to(original_device)
    return n_tokens / (t1 - t0)


def evaluate_run(model: nn.Module, val_ds, config: dict, device) -> dict:
    """End-of-run validation: avg loss + perplexity + CPU tokens/sec + RAM MB.

    Full eval suite (correctness, quality, perplexity, external comparisons)
    lives in `eval/`. This is the inline metric that gets written to the
    per-run metrics JSON.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    tok = config['tokenization']

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= 100:
                break

            if tok in ('byte', 'byte_dynamic'):
                x = batch['patches'].to(device)
                targets = batch['targets'].to(device)
            else:
                x = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            t_ids = batch.get('teacher_ids_bpe')
            if t_probs is not None:
                t_probs = t_probs.to(device)
            if t_mask is not None:
                t_mask = t_mask.to(device)
            if t_ids is not None:
                t_ids = t_ids.to(device)

            amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if device.type == 'cuda' else nullcontext())
            with amp_ctx:
                fwd_kwargs = (
                    {'byte_targets': targets} if tok in ('byte', 'byte_dynamic') else {}
                )
                result = model(x, **fwd_kwargs)
                if len(result) == 3:
                    logits, _, _ = result
                else:
                    logits, _ = result
                loss = compute_loss(logits, targets, t_probs, t_mask, config,
                                    teacher_ids=t_ids)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))

    tps = measure_cpu_throughput(model, config)
    ram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    metrics = {
        'run_name': config.get('name', 'unknown'),
        'perplexity': perplexity,
        'tokens_per_sec_cpu': tps,
        'ram_mb': ram_mb,
        'val_loss': avg_loss,
        'n_params_M': sum(p.numel() for p in model.parameters()) / 1e6,
    }

    print(f"\n[Val] {config.get('name', '?')}")
    print(f"  Perplexity:     {perplexity:.2f}")
    print(f"  Tokens/sec CPU: {tps:.1f}")
    print(f"  RAM:            {ram_mb:.1f} MB")
    return metrics
