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


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DataParallel and torch.compile wrappers so state_dict keys are clean.

    `torch.compile` exposes the inner module as `_orig_mod`; `DataParallel` as
    `module`. Wrapped state_dicts carry a `_orig_mod.` prefix that silently
    breaks load_state_dict(strict=False) — a Phase-2 ckpt saved from a compiled
    model would load zero keys. Always unwrap before serializing or measuring.
    """
    seen = set()
    while id(model) not in seen:
        seen.add(id(model))
        if hasattr(model, '_orig_mod'):
            model = model._orig_mod
        elif hasattr(model, 'module') and isinstance(model.module, nn.Module):
            model = model.module
    return model


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
        # Compute the variance in fp32: with d_model=512 and post-residual
        # magnitudes that grow with depth, x.pow(2).sum() can overflow fp16
        # under autocast (>~12 in any channel pushes the sum past 65 504).
        # Preserve the input dtype on output so downstream layers see what
        # they expect.
        dtype = x.dtype
        x_f32 = x.float()
        norm = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + self.eps)
        out = x_f32 * norm * self.weight.float()
        return out.to(dtype)


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


class _TernaryEmbedFn(torch.autograd.Function):
    """Custom autograd Function for ternary embedding.

    Saves only Parameters (weight, scale) for backward — never intermediates —
    so no saved tensor ever lives in CUDA graph output buffers. w_ternary is
    recomputed during backward, which is cheap and avoids the CUDAGraphs
    overwrite error that occurs when intermediates are saved inside torch.compile.
    """

    @staticmethod
    def forward(ctx, indices, weight, scale):
        w_scale = weight.abs().mean(dim=1, keepdim=True)
        w_ternary = (weight / (w_scale + 1e-8)).round().clamp(-1, 1)
        ctx.save_for_backward(indices, weight, scale)
        return F.embedding(indices, w_ternary * scale)

    @staticmethod
    def backward(ctx, grad_output):
        indices, weight, scale = ctx.saved_tensors
        with torch.no_grad():
            w_scale = weight.abs().mean(dim=1, keepdim=True)
            w_ternary = (weight / (w_scale + 1e-8)).round().clamp(-1, 1)
        V, D = weight.shape
        grad_eff = weight.new_zeros(V, D)
        grad_eff.scatter_add_(0, indices.reshape(-1, 1).expand(-1, D),
                              grad_output.reshape(-1, D))
        # STE: d(w_ternary)/d(weight) = 1, so grad_weight = grad_eff * scale
        return None, grad_eff * scale, (grad_eff * w_ternary).sum(1, keepdim=True)


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
        return _TernaryEmbedFn.apply(indices, self.weight, self.scale)


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
        # RMSNorm before down_proj: silu(gate)*up has arbitrary magnitudes
        # and down_proj (BitLinear) needs normalized input for clean INT8
        # quantization.  Follows the Fused BitLinear protocol (Algorithm 1).
        self.down_norm = RMSNorm(d_ff) if use_ternary else None
        self.fp_residual = (
            FPResidual(d_ff, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )

    def forward(self, x: Tensor) -> Tensor:
        hidden = F.silu(self.gate_proj(x)) * self.up_proj(x)
        down_in = self.down_norm(hidden) if self.down_norm is not None else hidden
        out = self.down_proj(down_in)
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
    """Bolmo / BLT-style byte-patch embedding with boundary-aware cross-attention.

    Pipeline:
      1. Embed each byte (BitEmbedding).
      2. Score each byte position with a 1-D conv (kernel 3) — this is the
         "boundary detector" supervised against natural-boundary targets.
      3. Build an initial patch query by mean-pooling the P bytes inside.
      4. Cross-attention: each patch query attends over a window of W bytes
         (default W = 3P = 12 for cross_patch_window=3), with the boundary
         score as an additive logit bias so high-boundary bytes dominate
         the patch representation.  When cross_patch_window=1 the behaviour
         matches the original per-patch-only attention.  With window=3 the
         query sees its own P bytes plus ±1 neighbour patch — this lets the
         model learn that a linguistic boundary falls in the adjacent patch
         and adjust the representation accordingly, approximating BLT's
         variable-length byte→patch fusion without breaking LocalByteDecoder
         + DataLoader + kernel C++ (audit §"Conflicto 3").
      5. Project to d_model.

    Caches pre-sigmoid logits in `last_boundary_logits` so the parent model
    can compute a BCE loss against natural-boundary targets.
    """

    def __init__(self, d_model: int, patch_size: int = 4,
                 bytes_per_embed: int = 8, use_ternary: bool = True,
                 cross_attn_heads: int = 2,
                 cross_patch_window: int = 1,
                 causal_boundary: bool = False,
                 causal_cross_patch: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.bytes_per_embed = bytes_per_embed
        self.cross_attn_heads = cross_attn_heads
        self.cross_patch_window = cross_patch_window
        self.causal_boundary = causal_boundary
        self.causal_cross_patch = causal_cross_patch
        self.byte_embed = make_embedding(256, bytes_per_embed, use_ternary)
        self.patch_proj = make_linear(
            patch_size * bytes_per_embed, d_model, use_ternary
        )
        # Causal mode: pad only on the left so the conv at position t sees
        # bytes [t-2, t-1, t] instead of [t-1, t, t+1] — preserves causality
        # for streaming inference at the patch level.
        self._boundary_pad = (2, 0) if causal_boundary else (1, 1)
        self.boundary_scorer = nn.Conv1d(
            in_channels=bytes_per_embed,
            out_channels=1,
            kernel_size=3,
            padding=0,
        )
        # Cross-attention from patch (query) to its bytes (keys/values).
        # Tiny: bytes_per_embed=8, head_dim=4 → 64 params per matrix × 4 = 256 params.
        # Run in fp32 to keep softmax stable — this layer isn't on the
        # latency-critical path (one call per forward, not per token).
        self.q_proj = nn.Linear(bytes_per_embed, bytes_per_embed, bias=False)
        self.k_proj = nn.Linear(bytes_per_embed, bytes_per_embed, bias=False)
        self.v_proj = nn.Linear(bytes_per_embed, bytes_per_embed, bias=False)
        self.attn_norm = RMSNorm(bytes_per_embed)
        self.last_boundary_logits: Tensor | None = None

    def forward(self, patches: Tensor) -> Tensor:
        B, S, P = patches.shape
        d_b = self.bytes_per_embed
        H = self.cross_attn_heads
        hd = d_b // H
        byte_feats = self.byte_embed(patches.long())                  # [B, S, P, d_b]

        conv_in = byte_feats.reshape(B, S * P, d_b).transpose(1, 2)   # [B, d_b, S*P]
        conv_in = F.pad(conv_in, self._boundary_pad)
        bndry_logits = self.boundary_scorer(conv_in).squeeze(1)       # [B, S*P]
        self.last_boundary_logits = bndry_logits

        # Multiplicative gate (legacy behaviour) applied first — this gives
        # the cross-attention a pre-amplified value tensor and preserves
        # backward-compat with checkpoints that only had the gate.
        gate = torch.sigmoid(bndry_logits).reshape(B, S, P, 1)
        byte_feats = byte_feats * (1.0 + gate)

        # === Cross-patch attention with configurable window ===
        # When cross_patch_window=1: each patch query sees only its own P bytes
        # (original behaviour).  When cross_patch_window=3: each query sees
        # 3P=12 bytes (prev patch + own + next), letting the model detect
        # linguistic boundaries that straddle patch edges.
        cpw = self.cross_patch_window
        if cpw > 1:
            # Flatten to [B, S*P, d_b]. Symmetric mode pads ±(cpw//2)*P (sees
            # past + own + future). Causal mode pads (cpw-1)*P only on the
            # left so the window for patch t covers past patches + own and
            # never leaks future bytes.
            all_bytes = byte_feats.reshape(B, S * P, d_b)
            if self.causal_cross_patch:
                left_pad = (cpw - 1) * P
                right_pad = 0
                own_offset = left_pad                                  # own patch at end
            else:
                left_pad = (cpw // 2) * P                              # = P for cpw=3
                right_pad = left_pad
                own_offset = left_pad                                  # own patch in middle
            all_padded = F.pad(all_bytes, (0, 0, left_pad, right_pad))
            bndry_padded = F.pad(bndry_logits, (left_pad, right_pad))
            W = cpw * P                                                # window = 12 bytes

            # Unfold: for each patch, extract W contiguous bytes
            kv_windows = all_padded.unfold(1, W, P)                    # [B, S, d_b, W]
            kv_windows = kv_windows.permute(0, 1, 3, 2)               # [B, S, W, d_b]
            bndry_win = bndry_padded.unfold(1, W, P)                   # [B, S, W]

            # Normalize the full window; query = mean of own-patch bytes.
            kv_norm = self.attn_norm(kv_windows)
            q_input = kv_norm[:, :, own_offset:own_offset + P, :].mean(
                dim=2, keepdim=True
            )                                                          # [B,S,1,d_b]

            q = self.q_proj(q_input)                                   # [B,S,1,d_b]
            k = self.k_proj(kv_norm)                                   # [B,S,W,d_b]
            v = self.v_proj(kv_norm)                                   # [B,S,W,d_b]
            q = q.view(B, S, 1, H, hd).transpose(2, 3)                # [B,S,H,1,hd]
            k = k.view(B, S, W, H, hd).transpose(2, 3)                # [B,S,H,W,hd]
            v = v.view(B, S, W, H, hd).transpose(2, 3)
            # Boundary bias over the full window
            bias = bndry_win.unsqueeze(2).unsqueeze(3)                 # [B,S,1,1,W]
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd) + bias
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).transpose(2, 3).reshape(B, S, 1, d_b)
        else:
            # Original per-patch-only attention (cross_patch_window=1)
            bf_norm = self.attn_norm(byte_feats)                       # [B,S,P,d_b]
            q = self.q_proj(bf_norm.mean(dim=2, keepdim=True))         # [B,S,1,d_b]
            k = self.k_proj(bf_norm)                                   # [B,S,P,d_b]
            v = self.v_proj(bf_norm)                                   # [B,S,P,d_b]
            q = q.view(B, S, 1, H, hd).transpose(2, 3)                # [B,S,H,1,hd]
            k = k.view(B, S, P, H, hd).transpose(2, 3)                # [B,S,H,P,hd]
            v = v.view(B, S, P, H, hd).transpose(2, 3)
            bias = bndry_logits.reshape(B, S, P).unsqueeze(2).unsqueeze(3)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd) + bias
            attn = F.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, v).transpose(2, 3).reshape(B, S, 1, d_b)

        # Fuse attention context into every byte feature within the patch:
        # ctx broadcasts across the P axis, providing global-patch info to
        # each byte before the linear patch_proj.
        byte_feats = byte_feats + ctx

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
    """Configurable MLGRU — full MatMul-free + HGRN2 feature set.

    Supported (all opt-in via config):
        per_channel_decay   : HGRN2 learnable per-channel forget-gate ceiling
                              (with hierarchical init, see below).
        state_reset_period  : Stuffed-Mamba periodic hard reset.
        use_wo              : MatMul-free LM "W_o" output projection that mixes
                              dimensions of (g ⊙ h) before adding to the
                              residual stream — the 4th matrix the Zhu-2024
                              paper specifies but the production kernel
                              omits.  Adds d_model² params per layer (+6.3%
                              of a 50M ablation budget at d=512×12 layers).
        state_expansion     : HGRN2 state expansion factor N.  When N>1 the
                              hidden state lives in dim N·d_model during the
                              recurrence; an "expand" projection lifts the
                              candidate up and a "compress" projection brings
                              the gated output back to d_model.  N=2 is the
                              audit's recommended setting (still fits L2 at
                              d=512); N=1 disables it.
    """

    def __init__(self, d_model: int, use_ternary: bool = True,
                 fp_rank: int = 4,
                 per_channel_decay: bool = False,
                 state_reset_period: int = 0,
                 use_wo: bool = True,
                 state_expansion: int = 1,
                 use_parallel_scan: bool = True,
                 layer_idx: int = 0, total_layers: int = 1):
        super().__init__()
        assert state_expansion >= 1, "state_expansion must be >= 1"
        self.d_model = d_model
        self.state_dim = d_model * state_expansion
        self.state_expansion = state_expansion
        self.state_reset_period = state_reset_period
        self.use_wo = use_wo
        # Parallel scan (Heinsen log-domain, paper-faithful to the MatMul-free
        # LM "parallel scan" of Zhu et al. 2024). Disabled automatically when
        # state_reset_period > 0 — periodic hard reset breaks the linear
        # recurrence that makes the scan correct.
        self.use_parallel_scan = use_parallel_scan and state_reset_period == 0

        # Input gates always project d_model -> state_dim.  When state_expansion=1
        # this is the standard MLGRU; when >1, candidate/forget/output gates
        # produce per-channel signals in the expanded space.
        self.forget_proj = make_linear(d_model, self.state_dim, use_ternary)
        self.cand_proj = make_linear(d_model, self.state_dim, use_ternary)
        self.output_proj = make_linear(d_model, self.state_dim, use_ternary)

        # W_o: state_dim -> d_model.  Doubles as the "compress" projection
        # when state_expansion > 1.  When use_wo=False and state_expansion=1
        # we fall back to identity; when use_wo=False and state_expansion>1
        # we still need a compress (so we instantiate it but skip in the
        # ternary path's RMSNorm — see below).
        if use_wo or state_expansion > 1:
            self.w_o = make_linear(self.state_dim, d_model, use_ternary)
            # RMSNorm before W_o keeps INT8 quant clean (Fused-BitLinear
            # protocol).  Cheap: state_dim ops per token.
            self.wo_norm = RMSNorm(self.state_dim) if use_ternary else None
        else:
            self.w_o = None
            self.wo_norm = None

        self.fp_residual = (
            FPResidual(d_model, d_model, rank=fp_rank)
            if use_ternary and fp_rank > 0 else None
        )
        self.register_buffer('h_init', torch.zeros(self.state_dim))

        if per_channel_decay:
            # Hierarchical init (HGRN2): shallow layers forget fast (low decay),
            # deep layers retain longer (high decay).  Linear schedule in
            # log-space from ~0.3 (layer 0) to ~0.9 (last layer).
            ratio = layer_idx / max(total_layers - 1, 1)
            target_sp = -math.log(0.3 + 0.6 * ratio)
            init_val = math.log(math.exp(target_sp) - 1) if target_sp > 0.5 else target_sp
            # Decay lives in the expanded state space when state_expansion>1
            # so each expanded channel has its own forget-rate ceiling.
            self.log_decay = nn.Parameter(torch.full((self.state_dim,), init_val))
        else:
            self.log_decay = None

    @staticmethod
    def _parallel_scan(c: Tensor, f: Tensor, h_init: Tensor
                       ) -> tuple[Tensor, Tensor]:
        """Heinsen 2023 log-domain parallel scan for h_t = f_t·h_{t-1} + (1-f_t)·c_t.

        Mathematically equivalent to the for-loop, vectorized over the time
        axis via cumulative log-sum-exp. This is the "parallel scan" the
        MatMul-free LM paper (Zhu et al. 2024) prescribes — closes the audit
        gap between the sequential ablation loop and the paper's algorithm.

        Numerically stable in fp32 even for long sequences with f far from 1:
        positive and negative contributions of b_t = (1-f_t)·c_t are tracked
        in log-space separately, then recombined.

        Args:
            c, f:   [B, S, D] candidate and forget tensors (f in (0,1)).
            h_init: [B, D] initial state.
        Returns:
            outputs: [B, S, D] state at every step.
            h_final: [B, D] state at step S-1 (for inference carry-over).
        """
        eps = 1e-30
        b = (1.0 - f) * c                                 # [B, S, D]
        log_a = torch.log(f.clamp(min=eps))               # [B, S, D], ≤ 0
        cum_log_a = log_a.cumsum(dim=1)                   # [B, S, D]

        b_pos = F.relu(b)
        b_neg = F.relu(-b)
        # Floor for log(0) — large negative finite so logcumsumexp ignores it
        # but no inf/NaN propagation risk.
        neg_floor = -3e4  # float16 max ~65504, so -3e4 is safe
        log_b_pos = torch.where(
            b_pos > 0, torch.log(b_pos.clamp(min=eps)),
            torch.full_like(b_pos, neg_floor),
        )
        log_b_neg = torch.where(
            b_neg > 0, torch.log(b_neg.clamp(min=eps)),
            torch.full_like(b_neg, neg_floor),
        )

        # log Σ_{k≤t} exp(-cum_log_a_k)·b_k  =  logcumsumexp(log b_k - cum_log_a_k)
        # torch.logcumsumexp lowers to index_put in the FX graph and breaks
        # torch.compile — replace with cummax+cumsum which are compile-friendly.
        def _logcumsumexp(x: Tensor) -> Tensor:
            m = x.cummax(dim=1).values
            return m + torch.log(torch.cumsum(torch.exp(x - m), dim=1).clamp(min=1e-30))

        log_sum_pos = _logcumsumexp(log_b_pos - cum_log_a)
        log_sum_neg = _logcumsumexp(log_b_neg - cum_log_a)

        # h_t = exp(cum_log_a_t)·h_0 + exp(cum_log_a_t + log_sum_pos_t)
        #                            - exp(cum_log_a_t + log_sum_neg_t)
        h_from_init = torch.exp(cum_log_a) * h_init.unsqueeze(1)
        h_from_pos  = torch.exp(cum_log_a + log_sum_pos)
        h_from_neg  = torch.exp(cum_log_a + log_sum_neg)
        h_seq = h_from_init + h_from_pos - h_from_neg     # [B, S, D]
        return h_seq, h_seq[:, -1]

    def forward(self, x: Tensor, h: Tensor | None = None,
                strides: Tensor | None = None,
                ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        strides: optional [B, S] int tensor giving the temporal gap between
            consecutive tokens — used after DeleteGate.forward_gather to
            compensate the per-channel decay so f_t becomes (decay**stride)·f_t
            (a stride of 1 reduces to standard MLGRU). Only consulted when
            per_channel_decay is enabled.
        """
        B, S, D = x.shape
        if h is None:
            h = self.h_init.unsqueeze(0).expand(B, -1).clone()

        f = torch.sigmoid(self.forget_proj(x))      # [B,S,state_dim]
        c = F.silu(self.cand_proj(x))               # [B,S,state_dim]
        g = torch.sigmoid(self.output_proj(x))      # [B,S,state_dim]

        if self.log_decay is not None:
            decay = torch.exp(-F.softplus(self.log_decay))           # [state_dim]
            if strides is not None:
                # Raise the per-channel ceiling to the gap power so the
                # effective forget rate matches what would have been applied
                # over `stride` contiguous steps. Broadcasting:
                # [B,S,1] · [state_dim] → [B,S,state_dim].
                decay = decay.unsqueeze(0).unsqueeze(0).pow(
                    strides.to(decay.dtype).unsqueeze(-1)
                )
                f = decay * f
            else:
                f = decay * f

        if self.use_parallel_scan:
            h_seq, h = self._parallel_scan(c, f, h)
            outputs = g * h_seq                     # [B,S,state_dim]
        else:
            outputs = torch.empty_like(f)           # [B,S,state_dim]
            reset = self.state_reset_period
            for t in range(S):
                if reset > 0 and t > 0 and t % reset == 0:
                    h = torch.zeros_like(h)
                h = torch.lerp(c[:, t], h, f[:, t])  # f*h + (1-f)*c
                outputs[:, t] = g[:, t] * h

        # Compress / cross-dim mix back to d_model.  When neither use_wo nor
        # state_expansion is enabled, outputs is already in d_model space and
        # we add directly to the residual.
        if self.w_o is not None:
            wo_in = (
                self.wo_norm(outputs) if self.wo_norm is not None else outputs
            )
            out = self.w_o(wo_in)
        else:
            out = outputs                            # identity (state_dim==d_model)

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
        # RMSNorm before o_proj: attention output needs normalization for
        # clean INT8 quantization inside o_proj (BitLinear protocol).
        self.o_norm = RMSNorm(inner_dim) if use_ternary else None
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
        # RMSNorm before o_proj: attention output has arbitrary magnitudes
        # and o_proj (BitLinear) needs normalized input for clean INT8 quant.
        if self.o_norm is not None:
            out = self.o_norm(out)
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

class LocalByteDecoderLayer(nn.Module):
    """One MegaByte local-transformer block (attn + SwiGLU). Used by
    LocalByteDecoder when n_layers>1 — the paper specifies 2-4 layers.

    Parameter names mirror the flat attrs of layer 0 in LocalByteDecoder so
    the per-layer Fused-BitLinear protocol (RMSNorm → quant → ternary GEMV)
    stays identical across stacked layers.
    """

    def __init__(self, d_local: int, n_heads: int = 4,
                 ff_ratio: float = 4.0, use_ternary: bool = True):
        super().__init__()
        assert d_local % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_local // n_heads

        self.norm1 = RMSNorm(d_local)
        self.q_proj = make_linear(d_local, d_local, use_ternary)
        self.k_proj = make_linear(d_local, d_local, use_ternary)
        self.v_proj = make_linear(d_local, d_local, use_ternary)
        self.o_proj = make_linear(d_local, d_local, use_ternary)
        self.o_norm = RMSNorm(d_local) if use_ternary else None

        self.norm2 = RMSNorm(d_local)
        d_ff_local = int(d_local * ff_ratio)
        self.ffn_gate = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_up = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_down = make_linear(d_ff_local, d_local, use_ternary)
        self.ffn_down_norm = RMSNorm(d_ff_local) if use_ternary else None

    def forward(self, x: Tensor) -> Tensor:
        N, L, D = x.shape
        residual = x
        y = self.norm1(x)
        q = self.q_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).view(N, L, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(N, L, D)
        if self.o_norm is not None:
            y = self.o_norm(y)
        x = residual + self.o_proj(y)

        residual = x
        y = self.norm2(x)
        hidden = F.silu(self.ffn_gate(y)) * self.ffn_up(y)
        if self.ffn_down_norm is not None:
            hidden = self.ffn_down_norm(hidden)
        return residual + self.ffn_down(hidden)


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
                 use_ternary: bool = True,
                 n_layers: int = 1):
        super().__init__()
        assert d_local % n_heads == 0, "d_local must be divisible by n_heads"
        assert n_layers >= 1, "n_layers must be >= 1"
        self.patch_size = patch_size
        self.d_local = d_local
        self.n_heads = n_heads
        self.head_dim = d_local // n_heads
        self.n_layers = n_layers

        self.h_proj = make_linear(d_model, patch_size * d_local, use_ternary)
        self.byte_embed = make_embedding(256, d_local, use_ternary)
        self.pos_embed = nn.Parameter(torch.randn(patch_size, d_local) * 0.02)

        # Layer 0 lives at flat attribute names so the production C++ kernel
        # (`local_decode_step`, n_layers=1) and existing trained checkpoints
        # keep their parameter layout. Extra MegaByte-faithful layers (paper
        # specifies 2-4) live in `extra_layers` for ablation runs only.
        self.norm1 = RMSNorm(d_local)
        self.q_proj = make_linear(d_local, d_local, use_ternary)
        self.k_proj = make_linear(d_local, d_local, use_ternary)
        self.v_proj = make_linear(d_local, d_local, use_ternary)
        self.o_proj = make_linear(d_local, d_local, use_ternary)
        # Norms before BitLinears that don't already follow norm1/norm2:
        # the Fused-BitLinear protocol (BitNet b1.58, Algorithm 1) needs
        # normalized input for clean INT8 quantization. d_local=128 makes
        # these norms ~free in inference.
        self.o_norm = RMSNorm(d_local) if use_ternary else None

        self.norm2 = RMSNorm(d_local)
        d_ff_local = int(d_local * ff_ratio)
        self.ffn_gate = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_up = make_linear(d_local, d_ff_local, use_ternary)
        self.ffn_down = make_linear(d_ff_local, d_local, use_ternary)
        self.ffn_down_norm = RMSNorm(d_ff_local) if use_ternary else None

        if n_layers > 1:
            self.extra_layers = nn.ModuleList([
                LocalByteDecoderLayer(
                    d_local, n_heads, ff_ratio, use_ternary,
                )
                for _ in range(n_layers - 1)
            ])
        else:
            self.extra_layers = None

        self.out_proj = make_linear(d_local, 256, use_ternary)
        self.out_norm = RMSNorm(d_local) if use_ternary else None
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
        if self.o_norm is not None:
            y = self.o_norm(y)
        x = residual + self.o_proj(y)

        residual = x
        y = self.norm2(x)
        hidden = F.silu(self.ffn_gate(y)) * self.ffn_up(y)
        if self.ffn_down_norm is not None:
            hidden = self.ffn_down_norm(hidden)
        x = residual + self.ffn_down(hidden)

        if self.extra_layers is not None:
            for layer in self.extra_layers:
                x = layer(x)
        return x

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

        if self.out_norm is not None:
            local_out = self.out_norm(local_out)
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

    def __init__(self, config: dict, layer_idx: int = 0):
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
                use_wo=config.get('mlgru_use_wo', True),
                state_expansion=config.get('mlgru_state_expansion', 1),
                use_parallel_scan=config.get('mlgru_parallel_scan', True),
                layer_idx=layer_idx,
                total_layers=config['n_layers'],
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

    def forward(self, x: Tensor, h: Tensor | None = None,
                strides: Tensor | None = None,
                ) -> tuple[Tensor, Tensor | None]:
        residual = x
        if self.mixer_type == 'mlgru':
            mixer_out, h_new = self.token_mixer(self.norm1(x), h, strides=strides)
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
    """5-byte-context boundary scorer, used as an auxiliary training loss.

    A Conv1d over byte embeddings gives each position a 5-byte receptive
    field (2 left + center + 2 right) — wider than the prior 2-byte concat
    that could not distinguish a UTF-8 lead-byte from punctuation in
    isolation (audit §4). Still ~training-only and tiny: <10 KB of weights.

    The supervision target is heuristic (`get_natural_boundaries`), not the
    BLT entropy-patching signal — replicating that would require a separate
    LM scoring pass per byte (audit §"Conflicto 11"). Wider context narrows
    the gap by giving the scorer enough information to fit BLT-like targets
    even though we keep heuristic labels for now.
    """

    def __init__(self, embed_dim: int = 8, hidden_dim: int = 32,
                 kernel_size: int = 5):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd for symmetric padding"
        self.byte_embed = nn.Embedding(256, embed_dim)
        # Conv1d with symmetric padding -> output length matches input.
        # Uses bias and ReLU/GELU activation between layers; final 1x1 conv
        # produces LOGITS (no Sigmoid) for autocast-safe BCE-with-logits.
        pad = kernel_size // 2
        self.scorer = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1),
        )

    def forward_train(
        self,
        bytes_seq: Tensor,
        threshold: float = 0.5,
    ) -> Tensor:
        """Return per-byte boundary LOGITS (shape [B, T]).

        The last byte is forced to be a boundary; we represent that with a
        large positive logit (sigmoid(10) ≈ 0.99995) so it pairs cleanly with
        binary_cross_entropy_with_logits at the call site.
        """
        # [B, T, E] -> [B, E, T] for Conv1d, then back to [B, T].
        embeds = self.byte_embed(bytes_seq.long()).transpose(1, 2)
        logits = self.scorer(embeds).squeeze(1)         # [B, T]
        # Force last byte to be a boundary (matches prior contract). Build
        # the constant last column instead of mutating in place to keep the
        # graph autograd-clean.
        last = torch.full(
            (logits.shape[0], 1), 10.0,
            device=logits.device, dtype=logits.dtype,
        )
        return torch.cat([logits[:, :-1], last], dim=1)

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
    """MrT5-style state-elimination gate placed at layer N//2.

    Phase 1 training (target_rate=0): pass-through, no elimination loss.
    Phase 2 training (target_rate>0): two modes available:

    **Mask mode** (``forward_train``, default):
      Produces a near-binary mask via straight-through Gumbel-sigmoid,
      multiplies kept states by 1 and deleted by 0, then merges deleted
      mass into surviving neighbours via a depthwise 1-D convolution.
      Sequence length stays fixed at S — simple, no shape changes.

    **Gather mode** (``forward_gather`` + ``scatter_back``, opt-in):
      Real MrT5 gather: TopK survivors are gathered into a compact
      ``[B, K, D]`` tensor (K = S × (1 - rate)).  Post-gate layers
      process only K tokens — the MLGRU scan runs K steps instead
      of S, giving ~20% wall-clock speedup for rate=0.4.  After
      post-gate processing, ``scatter_back`` places survivors at
      their original positions; deleted positions keep the pre-gate
      representation (no refinement from post-gate layers).
      Enabled via config ``use_real_gather=True``.

    Loss components:
      1. Rate constraint  : ((1 - mask).mean() - target_rate)²
      2. Bimodality penalty (binary entropy of the soft pre-STE mask)
         to keep gradients flowing toward decisive {0,1} decisions even
         though the forward pass is already hard.

    Args:
        hard            : if True, forward uses STE hard mask {0,1};
                          backward uses the soft sigmoid gradient.
        gumbel          : if True, adds Gumbel noise during training so
                          early-Phase-2 exploration is non-deterministic.
        merge           : if True, deleted residual mass is averaged into
                          neighbours via a tied depthwise conv (kernel=3).
        bimodal_lambda  : weight on the entropy regulariser.
    """

    def __init__(self, d_model: int, elimination_rate: float = 0.0,
                 bimodal_lambda: float = 0.05,
                 hard: bool = True, gumbel: bool = True,
                 merge: bool = True):
        super().__init__()
        self.d_model = d_model
        self.elimination_rate = elimination_rate
        self.bimodal_lambda = bimodal_lambda
        self.hard = hard
        self.gumbel = gumbel
        self.merge = merge
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # Depthwise (groups=d_model) conv: keeps params O(d_model · k) instead
        # of d_model² and acts independently per channel — equivalent to a
        # MrT5 "weighted sum into neighbours" for each feature.
        if merge:
            self.merge_conv = nn.Conv1d(
                d_model, d_model, kernel_size=3, padding=1,
                groups=d_model, bias=False,
            )
            # Init as average over neighbours; trains away from this if useful.
            with torch.no_grad():
                self.merge_conv.weight.fill_(1.0 / 3.0)
        else:
            self.merge_conv = None
        self.register_buffer('temperature', torch.tensor(1.0))
        self.last_elimination_loss = torch.tensor(0.0)

    def _gumbel_noise(self, logits: Tensor) -> Tensor:
        u = torch.rand_like(logits).clamp(1e-6, 1.0 - 1e-6)
        return torch.log(u) - torch.log(1.0 - u)

    def forward_train(
        self,
        hidden: Tensor,
        target_rate: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        if target_rate == 0.0:
            mask = torch.ones(hidden.shape[:2], device=hidden.device)
            self.last_elimination_loss = torch.tensor(0.0, device=hidden.device)
            return hidden, mask

        logits = self.scorer(hidden).squeeze(-1)
        if self.training and self.gumbel:
            logits = logits + self._gumbel_noise(logits)
        soft_mask = torch.sigmoid(logits / self.temperature)

        # Forward: hard {0,1} via STE. Backward: gradient flows through soft.
        if self.hard:
            hard_mask = (soft_mask > 0.5).to(soft_mask.dtype)
            mask = hard_mask + soft_mask - soft_mask.detach()
        else:
            mask = soft_mask

        m1 = mask.unsqueeze(-1)                         # [B,S,1]
        kept = hidden * m1                              # survivors

        # Merge: redistribute the *deleted* residual mass into neighbours
        # via a depthwise conv, then add only at surviving positions
        # (deleted positions stay zeroed).  Falls back to pure mute if
        # merge is disabled.
        if self.merge_conv is not None:
            deleted = hidden * (1.0 - m1)               # mass to redistribute
            # Conv1d expects [B, C, S] → transpose around the call.
            merged = self.merge_conv(deleted.transpose(1, 2)).transpose(1, 2)
            kept = kept + merged * m1

        # 1) Rate constraint on the *soft* mask so gradients are well-defined.
        rate_loss = ((1.0 - soft_mask).mean() - target_rate).pow(2)

        # 2) Bimodality regulariser on soft mask: low entropy ⇒ decisive.
        m = soft_mask.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m))
        self.last_elimination_loss = rate_loss + self.bimodal_lambda * entropy.mean()

        return kept, mask

    def forward_gather(
        self,
        hidden: Tensor,
        target_rate: float = 0.0,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Real MrT5 gather: return compact [B, K, D] survivors + indices.

        Unlike ``forward_train`` which keeps shape [B, S, D] and zeros out
        deleted positions, this method *physically removes* deleted tokens
        so that downstream layers process a shorter sequence.

        Args:
            hidden:      [B, S, D] pre-gate hidden states.
            target_rate: fraction of tokens to eliminate (0.0 = keep all).

        Returns:
            gathered:  [B, K, D] compact tensor of survivors (K = S*(1-rate)).
            keep_idx:  [B, K]   indices into dim-1 for scatter_back.
            gate_loss: scalar   rate + bimodal loss for the scorer.
        """
        B, S, D = hidden.shape
        K = max(int(S * (1.0 - target_rate)), 1)

        if target_rate == 0.0:
            # No elimination: return full sequence
            idx = torch.arange(S, device=hidden.device).unsqueeze(0).expand(B, -1)
            self.last_elimination_loss = torch.tensor(0.0, device=hidden.device)
            return hidden, idx, self.last_elimination_loss

        logits = self.scorer(hidden).squeeze(-1)                      # [B, S]
        if self.training and self.gumbel:
            logits = logits + self._gumbel_noise(logits)
        soft_mask = torch.sigmoid(logits / self.temperature)

        # Rate + bimodal loss (gradient path for scorer)
        rate_loss = ((1.0 - soft_mask).mean() - target_rate).pow(2)
        m = soft_mask.clamp(1e-6, 1.0 - 1e-6)
        entropy = -(m * torch.log(m) + (1.0 - m) * torch.log(1.0 - m))
        self.last_elimination_loss = rate_loss + self.bimodal_lambda * entropy.mean()

        # Merge deleted mass into survivors BEFORE gathering
        # Uses STE hard mask for the merge gating, same as forward_train
        hard_mask = (soft_mask > 0.5).to(hidden.dtype)
        mask_ste = hard_mask + soft_mask - soft_mask.detach()          # STE
        m1 = mask_ste.unsqueeze(-1)                                    # [B,S,1]
        if self.merge_conv is not None:
            deleted = hidden * (1.0 - m1)
            merged = self.merge_conv(deleted.transpose(1, 2)).transpose(1, 2)
            hidden = hidden + merged * m1  # enrich survivors with neighbour info

        # TopK selection (fixed K per batch for clean batched ops)
        _, keep_idx = soft_mask.topk(K, dim=1)                        # [B, K]
        keep_idx = keep_idx.sort(dim=1).values                        # temporal order

        # Gather survivors
        gathered = hidden.gather(
            1, keep_idx.unsqueeze(-1).expand(-1, -1, D)                # [B, K, D]
        )

        return gathered, keep_idx, self.last_elimination_loss

    def scatter_back(
        self,
        processed: Tensor,
        keep_idx: Tensor,
        h_pre_gate: Tensor,
    ) -> Tensor:
        """Scatter processed survivors back to original positions.

        Deleted positions keep their pre-gate representation (no refinement
        from post-gate layers).  This preserves gradient flow for all
        positions through the CE loss.

        Args:
            processed:   [B, K, D] post-gate-layer output for survivors.
            keep_idx:    [B, K]    original position indices from forward_gather.
            h_pre_gate:  [B, S, D] pre-gate hidden states (full sequence).

        Returns:
            output: [B, S, D] reconstructed full sequence.
        """
        B, S, D = h_pre_gate.shape
        output = h_pre_gate.clone()  # deleted positions keep layer-N/2 repr
        output.scatter_(
            1,
            keep_idx.unsqueeze(-1).expand(-1, -1, D),
            processed,
        )
        return output

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
        if self.use_cuda:
            for k, v in batch.items():
                if torch.is_tensor(v):
                    v.record_stream(torch.cuda.current_stream(self.device))
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

            student_log_T = F.log_softmax(logits / T, dim=-1)
            # Temperature-scale from stored probs: raise to 1/T then renormalize.
            # Dividing probs by T and softmax-ing is wrong (treats probs as logits).
            teacher_T = teacher_full.pow(1.0 / T)
            teacher_T = teacher_T / teacher_T.sum(-1, keepdim=True).clamp(min=1e-8)
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
    model_to_save = _unwrap_model(model)
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
            api.create_repo(hf_repo, repo_type="model", exist_ok=True)
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
    """Load a checkpoint into model (and optionally optimizer/scaler).

    Checkpoints are saved in bfloat16 to save disk space, but AMP training
    requires float32 master weights for numerical stability and because
    GradScaler.step() uses the multi_tensor_adamw fused kernel which asserts
    that all tensors in a param-group share the same device *and* dtype.

    Strategy:
    - Snapshot each parameter's (device, dtype) *before* load_state_dict so we
      can restore them afterwards (load_state_dict silently mutates param dtype
      to match the checkpoint tensor, i.e. bfloat16 → bfloat16 params).
    - Force optimizer exp_avg / exp_avg_sq to float32 on the correct device,
      regardless of p.dtype, since AdamW master-weight accumulators must be f32.
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # 1. Remember original device/dtype for every parameter.
    param_meta: dict[str, tuple[torch.device, torch.dtype]] = {
        name: (p.device, p.dtype)
        for name, p in model.named_parameters()
    }

    model.load_state_dict(checkpoint['model_state_dict'])

    # 2. Restore original device/dtype (checkpoint may have bfloat16 weights).
    for name, p in model.named_parameters():
        target_dev, target_dtype = param_meta[name]
        if p.device != target_dev or p.dtype != target_dtype:
            p.data = p.data.to(device=target_dev, dtype=target_dtype)

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # 3. Move optimizer state to GPU in float32.
        #    exp_avg / exp_avg_sq must be float32 for AMP GradScaler.
        #    'step' is intentionally left on CPU as a float64 scalar (PyTorch
        #    convention — the fused kernel allows this).
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    for k, v in optimizer.state[p].items():
                        if isinstance(v, torch.Tensor) and k != 'step':
                            optimizer.state[p][k] = v.to(
                                device=p.device,
                                dtype=torch.float32,
                            )
    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint.get('step', 0)


def save_ablation_checkpoint(model: nn.Module, optimizer, run_name: str,
                             metrics: dict, config: dict, step: int,
                             is_final: bool = False,
                             scaler=None) -> str:
    """Save ablation checkpoint: BF16 resume during training, 2-bit packed at end.

    Checkpoints are saved inside a per-run subdirectory (e.g. run_01/) so that
    multiple ablation runs can execute concurrently without filename collisions.
    """
    # Create per-run directory (exist_ok=True is atomic and safe for
    # concurrent processes creating the same directory).
    run_dir = Path(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2 STEP checkpoints carry a `_phase2` infix so they don't collide
    # with Phase 1 step checkpoints — lets Phase 2 resume from its own
    # progress if interrupted. The FINAL checkpoint keeps the bare name so
    # downstream tooling (eval/export) always finds the "fully trained"
    # weights at `{run}/checkpoint_{run}_final.pt`.
    phase = int(config.get('_current_phase', 1))
    if is_final:
        filename = f"checkpoint_{run_name}_final.pt"
    else:
        infix = '_phase2' if phase == 2 else ''
        filename = f"checkpoint_{run_name}{infix}_step{step}.pt"
    path = str(run_dir / filename)

    # Strip DataParallel + torch.compile wrappers; otherwise state_dict keys
    # carry an `_orig_mod.` prefix that breaks Phase-2 / complement loads.
    model_to_save = _unwrap_model(model)

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
        new_state = {}
        for param_id, group in opt_state.get('state', {}).items():
            new_group = dict(group)
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in new_group and new_group[key].is_floating_point():
                    new_group[key] = new_group[key].to(torch.bfloat16)
            new_state[param_id] = new_group
        
        opt_state_copy = dict(opt_state)
        opt_state_copy['state'] = new_state

        payload = {
            'model_state_dict': model_bf16,
            'optimizer_state_dict': opt_state_copy,
            'metrics': metrics,
            'step': step,
            'run_name': run_name,
            'config': config,
            'format': 'bf16',
        }
        if scaler is not None:
            payload['scaler_state_dict'] = scaler.state_dict()
        torch.save(payload, path)
        print(f"[Checkpoint] Resume checkpoint saved: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")

    hub_repo = config.get('checkpoint_repo')
    if hub_repo:
        _max_retries = 3
        _retry_wait  = 300  # 5 minutes — covers HF 429 rate-limit windows
        for _attempt in range(1, _max_retries + 1):
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.create_repo(hub_repo, repo_type="model", exist_ok=True)
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=f"{run_name}/{filename}",
                    repo_id=hub_repo,
                    repo_type="model",
                )
                print(f"[Checkpoint] Uploaded to HF: {run_name}/{filename}")
                break  # success
            except Exception as e:
                if _attempt < _max_retries:
                    print(f"[Checkpoint] Hub upload failed (attempt {_attempt}/{_max_retries}): {e}")
                    print(f"[Checkpoint] Waiting {_retry_wait//60} min before retry...")
                    time.sleep(_retry_wait)
                else:
                    print(f"[Checkpoint] Hub upload failed after {_max_retries} attempts: {e}")

    if not is_final:
        import glob as _glob
        infix = '_phase2' if phase == 2 else ''
        # Phase-aware cleanup: Phase 1 only removes its own step ckpts, Phase 2
        # only removes its own (so Phase 1 finals/steps survive Phase 2 if a
        # user wants to look at them).
        pattern = f"checkpoint_{run_name}{infix}_step*.pt"
        for old in _glob.glob(str(run_dir / pattern)):
            # Phase 1 pattern matches Phase 2 step ckpts too (since they also
            # contain `_step`), so filter those out manually.
            old_name = os.path.basename(old)
            if phase == 1 and '_phase2_step' in old_name:
                continue
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
        if self.has_teacher:
            if 'teacher_probs' in row and row['teacher_probs'] is not None and len(row['teacher_probs']) > 0:
                n_valid = len(row['teacher_probs']) // 256
                tp_raw = torch.tensor(row['teacher_probs'], dtype=torch.float32).reshape(n_valid, 256)
                full_tp = torch.zeros(S, 256, dtype=torch.float32)
                mask_raw = row['teacher_mask']
                tm = torch.zeros(S, dtype=torch.bool)
                if len(mask_raw) == S:
                    # Full boolean mask [S] — number of True entries may differ
                    # from n_valid if packing used a fixed budget; clamp to avoid
                    # shape-mismatch on scatter (e.g. [5000,256] vs [1148,256]).
                    tm = torch.tensor(mask_raw, dtype=torch.bool)
                    n_true = int(tm.sum().item())
                    n_use = min(n_valid, n_true)
                    if n_use > 0:
                        full_tp[tm.nonzero(as_tuple=True)[0][:n_use]] = tp_raw[:n_use]
                else:
                    # Sparse integer indices of valid positions
                    idx_t = torch.tensor(mask_raw, dtype=torch.long)
                    n_use = min(n_valid, len(idx_t))
                    full_tp[idx_t[:n_use]] = tp_raw[:n_use]
                    tm[idx_t[:n_use]] = True
                sample['teacher_probs'] = full_tp
                sample['teacher_mask'] = tm
            else:
                sample['teacher_probs'] = torch.zeros(S, 256, dtype=torch.float32)
                sample['teacher_mask'] = torch.zeros(S, dtype=torch.bool)

        if self.config.get('use_embedding_alignment', False):
            teacher_dim = self.config.get('teacher_dim', 128)
            if teacher_dim > 0:
                if 'teacher_hidden' in row and row['teacher_hidden'] is not None and len(row['teacher_hidden']) > 0:
                    n_valid_h = len(row['teacher_hidden']) // teacher_dim
                    th_raw = torch.tensor(row['teacher_hidden'], dtype=torch.float32).reshape(n_valid_h, teacher_dim)
                    full_th = torch.zeros(S, teacher_dim, dtype=torch.float32)
                    mask = sample.get('teacher_mask', torch.zeros(S, dtype=torch.bool))
                    active = mask.nonzero(as_tuple=True)[0]
                    n_use_h = min(n_valid_h, len(active))
                    if n_use_h > 0:
                        full_th[active[:n_use_h]] = th_raw[:n_use_h]
                    sample['teacher_hidden'] = full_th
                else:
                    sample['teacher_hidden'] = torch.zeros(S, teacher_dim, dtype=torch.float32)
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
        if self.has_teacher:
            if 'teacher_probs_bpe' in row and row['teacher_probs_bpe'] is not None and len(row['teacher_probs_bpe']) > 0:
                t_probs = (
                    torch.tensor(row['teacher_probs_bpe'], dtype=torch.float32)
                    .reshape(S, K)
                )
                t_ids = (
                    torch.tensor(row['teacher_ids_bpe'], dtype=torch.long)
                    .reshape(S, K)
                )
                # Drop teacher entries whose token ID falls outside the student vocab.
                # Without this, scatter_ into logits[..., :vocab_cap] crashes with
                # an out-of-bounds index when the Qwen teacher uses its full 152K vocab.
                if self.vocab_cap < 152064:
                    out_of_vocab = t_ids >= self.vocab_cap
                    t_probs = t_probs.clone()
                    t_probs[out_of_vocab] = 0.0
                    t_ids = t_ids.clone()
                    t_ids[out_of_vocab] = 0
                sample['teacher_probs_bpe'] = t_probs
                sample['teacher_ids_bpe'] = t_ids
                sample['teacher_mask_bpe'] = torch.tensor(
                    row['teacher_mask_bpe'], dtype=torch.bool
                )
            else:
                sample['teacher_probs_bpe'] = torch.zeros(S, K, dtype=torch.float32)
                sample['teacher_ids_bpe'] = torch.zeros(S, K, dtype=torch.long)
                sample['teacher_mask_bpe'] = torch.zeros(S, dtype=torch.bool)
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
        buf_size = int(os.environ.get('SMOKE_BUFFER_SIZE', '10000'))
        ds = ds.shuffle(seed=worker_seed, buffer_size=buf_size)
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


def _enforce_dataset_disk_limit(limit_gb: float = 10.0) -> None:
    """Auto-clear HuggingFace datasets cache if it exceeds limit_gb on disk."""
    import shutil
    cache_dir = Path(os.environ.get(
        'HF_DATASETS_CACHE',
        Path.home() / '.cache' / 'huggingface' / 'datasets',
    ))
    if not cache_dir.exists():
        return
    total_bytes = sum(
        f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()
    )
    total_gb = total_bytes / (1024 ** 3)
    if total_gb > limit_gb:
        print(f"[Dataset] Cache is {total_gb:.1f} GB > {limit_gb:.0f} GB limit — clearing stale cache at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Dataset] Cache cleared.")
    else:
        print(f"[Dataset] Cache size: {total_gb:.2f} GB / {limit_gb:.0f} GB limit")


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

    _enforce_dataset_disk_limit(limit_gb=10.0)

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
            from huggingface_hub import list_repo_tree
            file_tree = list(list_repo_tree(repo, repo_type='dataset', recursive=True))
            size_map = {
                item.path: getattr(item, 'size', None)
                for item in file_tree
                if item.path.startswith(f'{subset}/') and item.path.endswith('.parquet')
            }
            sorted_paths = sorted(size_map)
            if not sorted_paths:
                raise FileNotFoundError(f"No shards for {subset}/ in {repo}")

            # Cap each subset to 5 GB so that bpe_tokenized + byte_marginalized
            # together stay within the 10 GB total disk budget, giving all runs
            # proportionally equal data coverage.
            budget_bytes = 5 * 1024 ** 3
            selected, total_bytes = [], 0
            for path in sorted_paths:
                sz = size_map[path] or 0
                if sz and total_bytes + sz > budget_bytes:
                    break
                selected.append(f"hf://datasets/{repo}/{path}")
                total_bytes += sz
            if not selected:
                selected = [f"hf://datasets/{repo}/{p}" for p in sorted_paths]

            skipped = len(sorted_paths) - len(selected)
            print(f"[hub] Using {len(selected)}/{len(sorted_paths)} shards "
                  f"({total_bytes / 1024**3:.1f} GB, skipped {skipped} to stay ≤5 GB/subset)")
            hub_files = selected
            ds = load_dataset('parquet', data_files=hub_files, split='train')
            print(f"[hub] Loaded {len(ds)} rows from {len(hub_files)} shards ({subset})")
        except Exception as e:
            print(f"[hub] Fallback to load_dataset: {e}")
            ds = load_dataset(repo, subset, split=split)

    _enforce_dataset_disk_limit(limit_gb=10.0)

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
    # Unwrap torch.compile / DataParallel before moving to CPU: an
    # OptimizedModule on CPU re-triggers Dynamo compilation on each forward.
    model = _unwrap_model(model)
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
    max_batches = int(os.environ.get('SMOKE_MAX_VAL_BATCHES', '100'))

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= max_batches:
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
