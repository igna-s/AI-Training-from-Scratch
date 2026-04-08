"""
CPU-1: ~170M parameter byte-level language model with ternary weights, MatMul-free
token mixing, and PFNet (Prefetch-Aware Nonlinear Residual).

Arquitectura:
- BytePatchEmbeddingV2: BitEmbedding ternario (4 bytes -> 1024 dims)
- MatMulFreeTokenMixer: MLGRU sin multiplicacion de matrices (Hadamard only)
- PFNet: 2-layer ternary MLP residual branch (224 KB total, L2-resident)
- BitHead: prediction heads ternarios
- KV cache: eliminado — estado MLGRU de 28KB fijo

Fuentes:
- Slender-Mamba (COLING 2025): cuantizacion de embeddings y heads
- MatMul-free LM (arXiv 2406.02528): MLGRU sin multiplicacion de matrices

El objetivo es maximizar tokens/segundo en CPU de consumo (4 cores, 16GB RAM, 32MB L3).
"""

import ctypes
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuracion global v2
# =============================================================================

D_MODEL = 1024
N_LAYERS = 14
D_FF = 2730
FP_RANK = 8
PFNET_HIDDEN = 64
VOCAB_SIZE = 256
PATCH_SIZE = 4


# =============================================================================
# Componentes compartidos
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
        self._init_weights()

    def _init_weights(self):
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


class FPResidual(nn.Module):
    """Corrección FP16 de rango bajo en paralelo con BitLinear."""

    def __init__(self, in_dim: int, out_dim: int | None = None, rank: int = FP_RANK):
        super().__init__()
        out_dim = out_dim or in_dim
        self.fp_A = nn.Linear(in_dim, rank, bias=False)
        self.fp_B = nn.Linear(rank, out_dim, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fp_A.weight, std=0.02)
        nn.init.zeros_(self.fp_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        return self.fp_B(F.silu(self.fp_A(x)))


class SwiGLU(nn.Module):
    """FFN con gating multiplicativo SwiGLU."""

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


# Memory budget for PFNet:
# Per layer:  (1024×64 + 64×1024) × 1 bit = 16 KB
# 14 layers:  14 × 16 KB = 224 KB  → fits in L2 (~256–512 KB per core)
# Main model: ~44 MB                → streams from RAM layer by layer
# Recurrent state: 14 × 1024 × 2B = 28 KB → fits in L2 alongside PFNet


class PFNet(nn.Module):
    """
    Prefetch-Aware Nonlinear Residual.

    A tiny 2-layer ternary MLP that runs in parallel with the main
    MLGRU block. Sized so all PFNet weights across all layers fit in
    CPU L2 cache (~224 KB for 14 layers at d_model=1024, hidden=64),
    allowing them to remain cache-resident while main weights stream
    from RAM during inference.

    Output is added as a residual to the block output:
        block_out = mlgru_out + ffn_out + pfnet_out

    Weights are ternary (BitLinear), activations are SiLU.
    No FPResidual correction branch — kept intentionally minimal.
    """

    def __init__(self, d_model: int, hidden: int = 64):
        super().__init__()
        self.w1 = BitLinear(d_model, hidden)
        self.w2 = BitLinear(hidden, d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(self.norm(x))))


# =============================================================================
# 1. BitEmbedding — Embedding cuantizado a ternario (Slender-Mamba)
# =============================================================================

class BitEmbedding(nn.Module):
    """
    Capa de embedding cuantizada a 1.58 bits (ternario).

    Basado en Slender-Mamba (COLING 2025).
    Cuantiza la weight matrix del embedding a {-1, 0, 1} con un scale FP16
    por fila. En lookup: embedding = ternary_row * scale_row.

    Para byte-level (vocab_size=256) la tabla es pequena (256 * embed_dim),
    pero mantener todo ternario de pies a cabeza es el principio clave.
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(vocab_size, embed_dim) * 0.02)
        self.scale = nn.Parameter(torch.ones(vocab_size, 1))

    def forward(self, indices: Tensor) -> Tensor:
        """
        Args:
            indices: cualquier shape de indices (ej: [B, S, 4] para patches)
        Returns:
            [..., embed_dim] FP16/FP32
        """
        # Cuantizar pesos en forward (STE para backward)
        w_scale = self.weight.abs().mean(dim=1, keepdim=True)  # [vocab, 1]
        w_ternary = (self.weight / (w_scale + 1e-8)).round().clamp(-1, 1)
        w_ternary = w_ternary.detach() + self.weight - self.weight.detach()  # STE

        # Lookup ternario * scale
        return F.embedding(indices, w_ternary * self.scale)


# =============================================================================
# 2. BytePatchEmbeddingV2 — con BitEmbedding en vez de nn.Embedding
# =============================================================================

class BytePatchEmbeddingV2(nn.Module):
    """
    Convierte patches de 4 bytes en embeddings de d_model dims.
    Usa BitEmbedding (ternario) en vez de nn.Embedding (FP16).

    4 bytes * 8 dims = 32 features -> BitLinear(32, d_model).
    """

    def __init__(self, d_model: int = D_MODEL, patch_size: int = PATCH_SIZE,
                 bytes_per_embed: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.bytes_per_embed = bytes_per_embed
        self.byte_embed = BitEmbedding(256, bytes_per_embed)
        self.patch_proj = BitLinear(patch_size * bytes_per_embed, d_model)

    def forward(self, patches: Tensor) -> Tensor:
        """patches: [B, S, 4] uint8/long -> [B, S, d_model]"""
        B, S, P = patches.shape
        embeds = self.byte_embed(patches.long())           # [B, S, 4, 8]
        embeds = embeds.reshape(B, S, P * self.bytes_per_embed)  # [B, S, 32]
        return self.patch_proj(embeds)                      # [B, S, d_model]


# =============================================================================
# 3. MatMulFreeTokenMixer — MLGRU sin multiplicacion de matrices
# =============================================================================

class MatMulFreeTokenMixer(nn.Module):
    """
    Reemplaza SlidingWindowAttention con un token mixer sin MatMul.

    Basado en "Scalable MatMul-free Language Modeling" (Zhu et al., 2024).
    Usa un GRU modificado (MLGRU) que solo usa productos elemento a elemento
    (Hadamard products) en vez de Q@K^T.

    Estructura por timestep:
        f_t = sigmoid(BitLinear_f(x_t))      # forget gate
        c_t = silu(BitLinear_c(x_t))         # candidate
        g_t = sigmoid(BitLinear_g(x_t))      # output gate
        h_t = f_t * h_{t-1} + (1 - f_t) * c_t   # Hadamard, no MatMul
        o_t = g_t * h_t

    Ventajas sobre atencion en CPU:
    - O(d_model) ops por token vs O(window_size * d_model)
    - Estado h: [d_model] = 4KB en FP16 -> siempre en L1
    - Zero KV cache: sin crecimiento de memoria con contexto largo
    """

    def __init__(self, d_model: int, fp_rank: int = FP_RANK):
        super().__init__()
        self.d_model = d_model

        # 3 proyecciones con BitLinear ternario
        self.forget_proj = BitLinear(d_model, d_model)
        self.cand_proj = BitLinear(d_model, d_model)
        self.output_proj = BitLinear(d_model, d_model)

        # FPResidual paralelo en output
        self.fp_residual = FPResidual(d_model, d_model, rank=fp_rank)

        # Estado inicial (buffer, no parameter)
        self.register_buffer('h_init', torch.zeros(d_model))

    def forward(
        self,
        x: Tensor,                    # [B, S, d_model]
        h: Tensor | None = None       # [B, d_model] estado previo
    ) -> tuple[Tensor, Tensor]:
        """
        Training: procesar secuencia completa con loop temporal.
        Inferencia: procesar un token con estado explicito.

        Returns: (output [B, S, d_model], h_final [B, d_model])
        """
        B, S, D = x.shape

        if h is None:
            h = self.h_init.unsqueeze(0).expand(B, -1).clone()

        # Proyecciones ternarias
        f = torch.sigmoid(self.forget_proj(x))    # [B, S, D]
        c = F.silu(self.cand_proj(x))             # [B, S, D]
        g = torch.sigmoid(self.output_proj(x))    # [B, S, D]

        # Scan temporal: h_t = f_t * h_{t-1} + (1 - f_t) * c_t
        # torch.lerp(c, h, f) = c + f*(h-c) = f*h + (1-f)*c — una sola op vectorizada
        outputs = torch.empty_like(f)  # [B, S, D] pre-alocado
        for t in range(S):
            h = torch.lerp(c[:, t], h, f[:, t])
            outputs[:, t] = g[:, t] * h

        out = outputs  # ya es [B, S, D], sin torch.stack

        # FPResidual en paralelo (lee x, no out)
        out = out + self.fp_residual(x)

        return out, h

    def forward_one_token(
        self,
        x: Tensor,   # [B, D]
        h: Tensor     # [B, D]
    ) -> tuple[Tensor, Tensor]:
        """
        Inferencia recurrente: O(D) ops, sin KV cache.
        """
        # Proyecciones para un solo vector (unsqueeze/squeeze para BitLinear)
        x_3d = x.unsqueeze(1)  # [B, 1, D]

        f = torch.sigmoid(self.forget_proj(x_3d)).squeeze(1)   # [B, D]
        c = F.silu(self.cand_proj(x_3d)).squeeze(1)            # [B, D]
        g = torch.sigmoid(self.output_proj(x_3d)).squeeze(1)   # [B, D]

        h_new = torch.lerp(c, h, f)  # f*h + (1-f)*c en una sola op
        o = g * h_new

        fp_out = self.fp_residual(x_3d).squeeze(1)  # [B, D]
        return o + fp_out, h_new


# =============================================================================
# 4. BitHead — Prediction head ternario (Slender-Mamba)
# =============================================================================

class BitHead(nn.Module):
    """
    Prediction head ternario para byte prediction.

    Slender-Mamba cuantiza el lm_head con BitLinear.
    Para vocab_size=256 (byte-level) la degradacion es minima.

    Original:  nn.Linear(2048, 256) FP16 -> 4 heads * 1MB = 4MB
    BitHead:   BitLinear(2048, 256) ternario -> 4 heads * 0.125MB = 0.5MB
    """

    def __init__(self, d_model: int, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.proj = BitLinear(d_model, vocab_size)
        self.output_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, S, d_model] -> [B, S, 256]"""
        return self.proj(x) * self.output_scale


# =============================================================================
# 5. CPU1BlockV2 — Bloque con MLGRU + SwiGLU, todo ternario
# =============================================================================

class CPU1BlockV2(nn.Module):
    """
    Bloque actualizado: MLGRU + SwiGLU + PFNet.

    Perfil de memoria en inferencia por bloque:
        Original (atencion): KV cache 516 * 2048 * 2 * 2 bytes = 4MB/bloque
        MLGRU:               Estado h = 1024 * 2 bytes = 2KB/bloque
        PFNet:               16 KB/bloque (cache-resident in L2)
    """

    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF,
                 fp_rank: int = FP_RANK, pfnet_hidden: int = PFNET_HIDDEN):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.token_mixer = MatMulFreeTokenMixer(d_model, fp_rank)
        self.ffn = SwiGLU(d_model, d_ff)
        self.pfnet = PFNet(d_model, hidden=pfnet_hidden)

    def forward(
        self,
        x: Tensor,                    # [B, S, d_model]
        h: Tensor | None = None       # [B, d_model]
    ) -> tuple[Tensor, Tensor]:
        # Pre-norm -> MLGRU -> residual
        residual = x
        mixer_out, h_new = self.token_mixer(self.norm1(x), h)
        x = residual + mixer_out
        # Pre-norm -> FFN -> residual + PFNet nonlinear residual
        residual = x
        x = self.ffn(self.norm2(x))
        x = residual + x + self.pfnet(residual)
        return x, h_new

    def forward_one_token(
        self,
        x: Tensor,   # [B, D]
        h: Tensor     # [B, D]
    ) -> tuple[Tensor, Tensor]:
        """Inferencia de 1 token a traves del bloque."""
        residual = x
        x_3d = x.unsqueeze(1)  # [B, 1, D]
        normed = self.norm1(x_3d).squeeze(1)  # [B, D]
        mixer_out, h_new = self.token_mixer.forward_one_token(normed, h)
        x = residual + mixer_out  # residual
        residual = x
        x_3d = x.unsqueeze(1)
        pfnet_out = self.pfnet(x_3d).squeeze(1)  # PFNet from pre-norm input
        x = residual + self.ffn(self.norm2(x_3d)).squeeze(1) + pfnet_out
        return x, h_new


# =============================================================================
# 6. CPU1ModelV2 — Modelo completo actualizado
# =============================================================================

class CPU1ModelV2(nn.Module):
    """
    CPU-1 v2: todo ternario, sin MatMul en token mixing, con PFNet.

    Componentes cambiados vs v1:
        BytePatchEmbedding     -> BytePatchEmbeddingV2 (BitEmbedding)
        SlidingWindowAttention -> MatMulFreeTokenMixer (MLGRU)
        nn.Linear heads        -> BitHead (ternario)
        RoPE                   -> eliminado
        PFNet                  -> nonlinear residual branch per block

    Estado de inferencia:
        MLGRU state: 14 * 1024 * 2 bytes = 28KB total -> en L2
        PFNet weights: 14 * 16KB = 224KB -> en L2
        (antes: 20 * 516 * 2048 * 2 * 2 bytes = 80MB -> no cabia en L3)
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        d_model = config.get('d_model', D_MODEL)
        n_layers = config.get('n_layers', N_LAYERS)
        d_ff = config.get('d_ff', D_FF)
        fp_rank = config.get('fp_rank', FP_RANK)
        pfnet_hidden = config.get('pfnet_hidden', PFNET_HIDDEN)

        self.d_model = d_model
        self.n_layers = n_layers

        self.patch_embedding = BytePatchEmbeddingV2(d_model)
        self.blocks = nn.ModuleList([
            CPU1BlockV2(d_model, d_ff, fp_rank, pfnet_hidden) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # 4 heads ternarios (Slender-Mamba style)
        self.byte_heads = nn.ModuleList([
            BitHead(d_model, VOCAB_SIZE) for _ in range(4)
        ])

        self.use_gradient_checkpointing = config.get(
            'use_gradient_checkpointing', False
        )

    def forward(
        self,
        patches: Tensor,                     # [B, S, 4] uint8/long
        hidden_states: list | None = None    # lista de n_layers tensores [B, d_model]
    ) -> tuple[tuple, list]:
        """
        Returns:
            logits: tuple de 4 tensores [B, S, 256]
            hidden_states: lista de n_layers tensores [B, d_model]
        """
        x = self.patch_embedding(patches)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states = []

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _block_fn(x_in, h_in, _block=block):
                    return _block(x_in, h_in)
                x, h_new = checkpoint(_block_fn, x, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)
        else:
            for i, block in enumerate(self.blocks):
                # Prefetch pesos de capa i+2 mientras procesamos capa i
                if i + 2 < len(self.blocks):
                    _ = self.blocks[i + 2].token_mixer.forget_proj.weight.data_ptr()

                x, h_new = block(x, hidden_states[i])
                new_states.append(h_new)

        x = self.norm(x)
        logits = tuple(head(x) for head in self.byte_heads)

        return logits, new_states

    def generate_bytes(
        self,
        prompt_bytes: bytes,
        max_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> bytes:
        """
        Inferencia recurrente token por token.
        Cero KV cache. Estado MLGRU de 28KB fijo.
        """
        self.eval()
        device = next(self.parameters()).device

        # Convertir prompt a patches
        patches = bytes_to_patches(prompt_bytes).to(device)
        hidden_states = None

        # Prefill: procesar prompt completo
        with torch.no_grad():
            _, hidden_states = self.forward(patches, hidden_states)

        # Generacion autoregresiva
        generated = []
        last_patch = patches[:, -1:, :]

        for _ in range(max_tokens // PATCH_SIZE):
            with torch.no_grad():
                logits, hidden_states = self.forward(last_patch, hidden_states)

            # Samplear los 4 bytes del proximo patch
            next_bytes = []
            for head_logits in logits:
                probs = F.softmax(head_logits[0, 0] / temperature, dim=-1)
                # Top-p (nucleus) sampling
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cumsum = sorted_probs.cumsum(dim=0)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                byte_val = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
                next_bytes.append(byte_val)

            generated.extend(next_bytes)
            last_patch = torch.tensor(
                next_bytes, dtype=torch.long, device=device
            ).reshape(1, 1, PATCH_SIZE)

        return bytes(generated)

    def generate_one_patch(
        self,
        x_patch: Tensor,        # [B, D]
        hidden_states: list,     # lista de n_layers [B, D]
    ) -> tuple[Tensor, list]:
        """
        Procesar 1 patch a traves de TODAS las capas.
        Maximiza reutilizacion del estado h en cache (28KB en L2).
        """
        new_states = []
        for i, block in enumerate(self.blocks):
            x_patch, h_new = block.forward_one_token(x_patch, hidden_states[i])
            new_states.append(h_new)
        return x_patch, new_states

    def prepare_for_inference(self):
        """
        Preparar pesos para inferencia CPU:
        1. Hacer pesos contiguos
        2. Transponer pesos para acceso secuencial en vector-matrix multiply
        3. Verificar alineamiento de cache line
        """
        for block in self.blocks:
            for name, param in block.named_parameters():
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()

        # Transponer pesos de BitLinear para inferencia eficiente
        _prepare_bitlinear_weights(self)

        return True


# =============================================================================
# 7. BoundaryPredictor — Parches dinámicos (Bolmo, Allen AI 2025)
# =============================================================================

class BoundaryPredictor(nn.Module):
    """
    Predice si el byte actual es el último de un parche (boundary=True).

    Basado en: Bolmo (Allen AI, 2025) — adaptado para CPU-1 v2.

    Arquitectura: muy liviana para no dominar el cómputo.
        byte_embed → Linear(16, 32) → ReLU → Linear(32, 1) → Sigmoid

    Input:  byte actual (uint8) + byte siguiente (uint8, lookahead de 1)
    Output: probabilidad de que el byte actual sea un boundary [0, 1]

    En training: supervisado con boundaries naturales del texto
        (espacios, puntuación, inicio de caracteres UTF-8 multibyte)
    En inferencia: threshold de 0.5 para decidir si cortar el parche

    Tamaño: ~2KB — completamente en L1 cache durante inferencia.

    CONCURRENCIA GPU:
        En training, correr en paralelo con BytePatchEmbeddingV2:
            boundaries = boundary_predictor(bytes_current, bytes_next)
            embeds     = patch_embedding(patches)  # en paralelo

    CONCURRENCIA CPU:
        Un byte a la vez. Tan liviano que no añade latencia medible.

    MODO ESTÁTICO (compatibilidad con runs 01-11):
        Si use_boundary_predictor=False en config → parches fijos de 4 bytes
        Los runs 01-11 existentes NO cambian.
    """

    def __init__(self, embed_dim: int = 8, hidden_dim: int = 32):
        super().__init__()
        self.byte_embed = nn.Embedding(256, embed_dim)
        # Toma byte actual + byte siguiente → contexto de 2 bytes
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward_train(
        self,
        bytes_seq: Tensor,  # [B, L] uint8 — secuencia de bytes
        threshold: float = 0.5,
    ) -> Tensor:
        """
        Predecir boundaries para toda la secuencia (training, paralelo).

        Returns: [B, L] float — probabilidades de boundary
        """
        # Lookahead de 1 byte: comparar byte[i] con byte[i+1]
        b_curr = self.byte_embed(bytes_seq[:, :-1].long())    # [B, L-1, E]
        b_next = self.byte_embed(bytes_seq[:, 1:].long())     # [B, L-1, E]

        ctx = torch.cat([b_curr, b_next], dim=-1)  # [B, L-1, 2E]
        probs = self.predictor(ctx).squeeze(-1)     # [B, L-1]

        # Último byte siempre es boundary
        last = torch.ones(bytes_seq.shape[0], 1,
                         device=bytes_seq.device, dtype=probs.dtype)
        probs = torch.cat([probs, last], dim=1)     # [B, L]

        return probs  # retornar probs, no bool (para loss)

    def forward_infer(
        self,
        byte_curr: int,
        byte_next: int,
        threshold: float = 0.5,
    ) -> bool:
        """
        Predecir si byte_curr es boundary (inferencia, 1 byte a la vez).
        CPU: <0.01ms por llamada (2KB de pesos en L1).
        """
        with torch.no_grad():
            b_c = self.byte_embed(torch.tensor([byte_curr]))
            b_n = self.byte_embed(torch.tensor([byte_next]))
            ctx = torch.cat([b_c, b_n], dim=-1)
            return self.predictor(ctx).item() > threshold

    @staticmethod
    def get_natural_boundaries(byte_seq: bytes) -> list[bool]:
        """
        Generar labels de training para BoundaryPredictor.
        Boundaries naturales: espacios, puntuación, inicio UTF-8 multibyte.

        Reglas:
        1. Espacios (0x20) → boundary DESPUÉS del espacio
        2. Puntuación ASCII (.,!?;:) → boundary DESPUÉS
        3. Inicio de caracter UTF-8 multibyte (0xC0-0xFF) → boundary ANTES
        4. Cada 8 bytes como máximo → boundary forzado (evitar parches largos)
        5. Cada 1 byte como mínimo → no hay parches de 0 bytes
        """
        boundaries = [False] * len(byte_seq)
        last_boundary = 0

        PUNCT = set(b'.,!?;:\n\t')

        for i, b in enumerate(byte_seq):
            # Forzar boundary cada 8 bytes máximo
            if i - last_boundary >= 8:
                boundaries[i] = True
                last_boundary = i
                continue

            # Boundary natural
            is_boundary = (
                b == 0x20 or          # espacio
                b in PUNCT or         # puntuación
                (b >= 0xC0) or        # inicio UTF-8 multibyte
                (i > 0 and byte_seq[i-1] == 0x20)  # después de espacio
            )

            # Mínimo 1 byte por parche
            if is_boundary and i > last_boundary:
                boundaries[i] = True
                last_boundary = i

        # Siempre terminar con boundary
        if boundaries:
            boundaries[-1] = True

        return boundaries


def bytes_to_dynamic_patches(
    byte_seq: bytes,
    boundaries: list[bool],
    max_patch_size: int = 8,
    pad_byte: int = 0,
) -> list[list[int]]:
    """
    Convertir secuencia de bytes a parches de longitud variable.

    Los parches tienen longitud entre 1 y max_patch_size bytes.
    Se paddean a max_patch_size con pad_byte para batch processing.

    Returns: lista de parches, cada uno de longitud max_patch_size (paddeado)
    """
    patches = []
    current_patch = []

    for i, (b, is_boundary) in enumerate(zip(byte_seq, boundaries)):
        current_patch.append(b)
        if is_boundary or len(current_patch) >= max_patch_size:
            # Paddear a max_patch_size
            padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
            patches.append(padded[:max_patch_size])
            current_patch = []

    if current_patch:
        padded = current_patch + [pad_byte] * (max_patch_size - len(current_patch))
        patches.append(padded[:max_patch_size])

    return patches


def bytes_to_dynamic_patches_infer(
    byte_seq: bytes,
    boundary_predictor: 'BoundaryPredictor | None',
    threshold: float = 0.5,
    max_patch_size: int = 8,
) -> list[list[int]]:
    """
    Crear parches dinámicos para inferencia usando BoundaryPredictor.
    Si boundary_predictor es None, usa parches estáticos de 4 bytes.
    """
    if boundary_predictor is None:
        # Modo estático: parches fijos de 4 bytes
        patches = []
        for i in range(0, len(byte_seq), PATCH_SIZE):
            patch = list(byte_seq[i:i + PATCH_SIZE])
            patch += [0] * (PATCH_SIZE - len(patch))
            patches.append(patch)
        return patches

    # Modo dinámico: usar BoundaryPredictor
    boundaries = []
    for i in range(len(byte_seq)):
        if i < len(byte_seq) - 1:
            is_bound = boundary_predictor.forward_infer(
                byte_seq[i], byte_seq[i + 1], threshold
            )
        else:
            is_bound = True  # último byte siempre es boundary
        boundaries.append(is_bound)

    return bytes_to_dynamic_patches(byte_seq, boundaries, max_patch_size)


# =============================================================================
# 8. DeleteGate — MrT5 (Google, 2024)
# =============================================================================

class DeleteGate(nn.Module):
    """
    Puerta de eliminación inspirada en MrT5.
    Se ubica después de la capa N del MLGRU y decide qué estados eliminar.

    Basado en: MrT5 (Google, 2024) — adaptado para MLGRU byte-level.

    Diferencia vs MrT5 original:
    - MrT5 opera sobre tokens (discreta con Gumbel-Softmax)
    - Nuestra versión opera sobre estados MLGRU (continua durante training)
    - En inferencia: decisión binaria hard (eliminar o no)

    CONCURRENCIA GPU (Fase 2 training):
        delete_scores = delete_gate(hidden_states)  # [B, S]
        mask = torch.sigmoid(delete_scores)
        hidden_filtered = hidden_states * mask.unsqueeze(-1)

    CONCURRENCIA CPU (inferencia):
        if delete_gate.should_delete(h_state):
            continue  # skip este estado
        else:
            continue_processing(h_state)

    Ubicación: después de layer N//2 (la mitad del modelo).
    """

    def __init__(self, d_model: int, elimination_rate: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.elimination_rate = elimination_rate  # 0.0 en Fase 1

        # Scorer liviano: [d_model → 64 → 1]
        self.scorer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Temperatura para annealing durante training
        self.register_buffer('temperature', torch.tensor(1.0))
        # Almacena la loss de eliminación del último forward
        self.last_elimination_loss = torch.tensor(0.0)

    def forward_train(
        self,
        hidden: Tensor,           # [B, S, d_model]
        target_rate: float = 0.0,  # fracción de estados a eliminar
    ) -> tuple[Tensor, Tensor]:
        """
        Soft masking diferenciable para training.

        Returns:
            hidden_filtered: [B, S, d_model] — estados con mask aplicada
            mask:            [B, S] float — valores entre 0 y 1

        En Fase 1 (target_rate=0.0): mask = todos 1s → sin eliminación
        En Fase 2 (target_rate>0.0): mask sparse → algunos estados reducidos
        """
        if target_rate == 0.0:
            # Fase 1: sin eliminación, pass-through puro
            mask = torch.ones(hidden.shape[:2], device=hidden.device)
            self.last_elimination_loss = torch.tensor(0.0, device=hidden.device)
            return hidden, mask

        scores = self.scorer(hidden).squeeze(-1)  # [B, S]

        # Soft masking con temperatura
        mask = torch.sigmoid(scores / self.temperature)

        # Regularización: push hacia target_rate de eliminación
        self.last_elimination_loss = (
            (1 - mask).mean() - target_rate
        ).pow(2)

        return hidden * mask.unsqueeze(-1), mask

    def forward_infer(
        self,
        h_state: Tensor,  # [d_model] — estado de UN patch
        threshold: float = 0.5,
    ) -> bool:
        """
        Decisión hard para inferencia CPU.

        Returns: True si el estado debe MANTENERSE (False = eliminar)

        CONCURRENCIA: este método es llamado en Thread 1.
        Thread 2 puede estar haciendo prefetch de la siguiente capa.
        Latencia: ~0.1ms (64 ops, completamente en L1).
        """
        with torch.no_grad():
            score = self.scorer(h_state.unsqueeze(0)).item()
            return torch.sigmoid(torch.tensor(score)).item() > threshold

    def set_elimination_rate(self, rate: float) -> None:
        """Cambiar tasa de eliminación entre Fase 1 y Fase 2."""
        self.elimination_rate = rate


# =============================================================================
# 9. EmbeddingAligner — Distilación de embeddings (arXiv 2602, 2025)
# =============================================================================

class EmbeddingAligner(nn.Module):
    """
    Alinea el espacio de embeddings del student con el del teacher.

    Basado en: "Distilling Byte-Level Models from Tokenizer-Based LMs"
    (arXiv 2602, 2025) — Embedding Alignment técnica.

    El teacher (Qwen-7B) produce embeddings de dimensión teacher_dim.
    El student (CPU-1) produce embeddings de dimensión student_dim.

    Este módulo aprende una proyección lineal:
        student_proj: [student_dim → teacher_dim]

    Y minimiza la distancia coseno entre:
        proj(student_hidden) y teacher_hidden

    Solo se activa en Fase 1 de training (cuando hay señal del teacher).
    En Fase 2 (DeleteGate activado), se congela.

    CONCURRENCIA GPU:
        La loss de alineación se calcula en paralelo con la distillation loss:
            task1: distillation_loss(logits, targets, teacher_probs)
            task2: alignment_loss(student_hidden, teacher_hidden)
            total_loss = task1 + lambda_align * task2

        lambda_align: hyperparámetro entre 0.1 y 1.0
        Recomendado: 0.3 (no dominar sobre la distillation loss)
    """

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projection = nn.Linear(student_dim, teacher_dim, bias=False)
        # Inicializar con ortogonal para estabilidad
        nn.init.orthogonal_(self.projection.weight)

    def alignment_loss(
        self,
        student_hidden: Tensor,    # [B, S, student_dim]
        teacher_hidden: Tensor,    # [B, S_teacher, teacher_dim]
        teacher_mask: Tensor,      # [B, S] bool — donde hay señal
        aggregation: str = 'mean_pool',
    ) -> Tensor:
        """
        Calcular loss de alineación de embeddings.

        El desafío: student tiene S patches (byte-level),
        teacher tiene S_teacher tokens (BPE-level).
        S != S_teacher en general.

        Solución: usar las primeras min(S, S_teacher) posiciones
        donde hay señal del teacher (teacher_mask).

        Returns: scalar loss
        """
        # Proyectar student al espacio del teacher
        student_proj = self.projection(student_hidden)  # [B, S, teacher_dim]

        # Normalizar (cosine similarity)
        student_norm = F.normalize(student_proj, dim=-1)
        teacher_norm = F.normalize(teacher_hidden.float(), dim=-1)

        # Solo calcular donde hay señal del teacher (teacher_mask)
        if teacher_mask.any():
            # Alinear dimensiones
            min_S = min(student_norm.shape[1], teacher_norm.shape[1])

            s_masked = student_norm[:, :min_S][teacher_mask[:, :min_S]]
            t_masked = teacher_norm[:, :min_S][teacher_mask[:, :min_S]]

            if s_masked.numel() > 0:
                # Cosine similarity loss: 1 - cosine(student, teacher)
                cosine_sim = (s_masked * t_masked).sum(dim=-1)
                loss = (1 - cosine_sim).mean()
            else:
                loss = torch.tensor(0.0, device=student_hidden.device)
        else:
            loss = torch.tensor(0.0, device=student_hidden.device)

        return loss


# =============================================================================
# 10. CPU1ModelV2WithGates — Modelo extendido con las 3 técnicas
# =============================================================================

class CPU1ModelV2WithGates(nn.Module):
    """
    CPU-1 v2 extendido con BoundaryPredictor y DeleteGate.

    Extiende CPU1ModelV2 sin romper compatibilidad.
    Si use_boundary=False y use_delete_gate=False → idéntico a CPU1ModelV2.

    Estructura:
        BytePatchEmbeddingV2 (estático O dinámico con BoundaryPredictor)
        ↓
        Layers 0..N//2: CPU1BlockV2 (MLGRU + SwiGLU)
        ↓
        DeleteGate (solo si use_delete_gate=True y Fase 2)
        ↓
        Layers N//2+1..N-1: CPU1BlockV2
        ↓
        RMSNorm → 4 × BitHead
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model  = config.get('d_model', D_MODEL)
        n_layers = config.get('n_layers', N_LAYERS)
        d_ff     = config.get('d_ff', D_FF)
        fp_rank  = config.get('fp_rank', FP_RANK)
        pfnet_hidden = config.get('pfnet_hidden', PFNET_HIDDEN)

        # Flags de configuración
        self.use_boundary    = config.get('use_boundary_predictor', False)
        self.use_delete_gate = config.get('use_delete_gate', False)
        self.use_emb_align   = config.get('use_embedding_alignment', False)
        self.delete_gate_pos = n_layers // 2  # mitad del modelo

        # Módulos base (iguales a CPU1ModelV2)
        self.patch_embedding = BytePatchEmbeddingV2(d_model)
        self.blocks = nn.ModuleList([
            CPU1BlockV2(d_model, d_ff, fp_rank, pfnet_hidden)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.byte_heads = nn.ModuleList([
            BitHead(d_model, VOCAB_SIZE) for _ in range(4)
        ])

        # Módulos nuevos (opcionales)
        if self.use_boundary:
            self.boundary_predictor = BoundaryPredictor(
                embed_dim=8, hidden_dim=32
            )

        if self.use_delete_gate:
            self.delete_gate = DeleteGate(
                d_model=d_model,
                elimination_rate=0.0  # Fase 1: sin eliminación
            )

        if self.use_emb_align:
            self.embedding_aligner = EmbeddingAligner(
                student_dim=d_model,
                teacher_dim=config.get('teacher_dim', 3584),  # Qwen-7B
            )

        self.use_gradient_checkpointing = config.get(
            'use_gradient_checkpointing', False
        )

    def forward(
        self,
        patches: Tensor,                  # [B, S, patch_size] uint8
        hidden_states: list | None = None,
        training_phase: int = 1,           # 1 = con distilación, 2 = con delete gate
        target_elimination: float = 0.0,
        teacher_hidden: Tensor | None = None,  # [B, S, teacher_dim] for EmbAligner
        teacher_mask: Tensor | None = None,    # [B, S] bool
    ) -> tuple:
        """
        Forward pass con soporte para ambas fases de training.

        training_phase=1: sin DeleteGate, con EmbeddingAlignment
        training_phase=2: con DeleteGate, sin EmbeddingAlignment

        Returns: (logits_tuple, hidden_states, extra_losses_dict)
        """
        x = self.patch_embedding(patches)  # [B, S, d_model]

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states  = []
        extra_losses = {}
        delete_mask = None

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _block_fn(x_in, h_in, _block=block):
                    return _block(x_in, h_in)
                x, h_new = checkpoint(_block_fn, x, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)

                # DeleteGate en la mitad del modelo (solo Fase 2)
                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    x, delete_mask = self.delete_gate.forward_train(
                        x, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss
        else:
            for i, block in enumerate(self.blocks):
                # Prefetch capa i+2
                if i + 2 < len(self.blocks):
                    _ = self.blocks[i + 2].token_mixer.forget_proj.weight.data_ptr()

                x, h_new = block(x, hidden_states[i])
                new_states.append(h_new)

                # DeleteGate en la mitad del modelo (solo Fase 2)
                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    x, delete_mask = self.delete_gate.forward_train(
                        x, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss

        x_pre_norm = x  # guardar para EmbeddingAligner antes del norm
        x = self.norm(x)
        logits = tuple(head(x) for head in self.byte_heads)

        # EmbeddingAlignment loss (solo Fase 1 si está habilitado y hay teacher_hidden)
        if self.use_emb_align and training_phase == 1 and teacher_hidden is not None:
            extra_losses['emb_align'] = self.embedding_aligner.alignment_loss(
                x_pre_norm, teacher_hidden, teacher_mask,
            )

        return logits, new_states, extra_losses

    def generate_bytes_with_gates(
        self,
        prompt_bytes: bytes,
        max_bytes: int = 200,
        temperature: float = 1.0,
        boundary_threshold: float = 0.5,
        delete_threshold: float = 0.5,
    ) -> bytes:
        """
        Inferencia CPU con BoundaryPredictor y DeleteGate activos.

        BoundaryPredictor determina cuántos bytes aceptar de cada patch.
        Es tan liviano (~2KB, <0.01ms) que se ejecuta inline — el overhead
        de crear un thread en Windows (~0.1ms) supera el cómputo en sí.
        """
        self.eval()
        hidden_states = [None] * len(self.blocks)

        # Prefill con parches dinámicos
        patches = bytes_to_dynamic_patches_infer(
            prompt_bytes,
            self.boundary_predictor if self.use_boundary else None,
            boundary_threshold,
        )
        patches_t = torch.tensor(patches, dtype=torch.long).unsqueeze(0)

        if patches_t.shape[-1] > PATCH_SIZE:
            patches_t = patches_t[:, :, :PATCH_SIZE]

        with torch.no_grad():
            _, hidden_states, _ = self.forward(patches_t, hidden_states)

        # Generación
        generated = []
        last_patch = patches_t[:, -1:, :]

        while len(generated) < max_bytes:
            with torch.no_grad():
                logits, hidden_states, _ = self.forward(
                    last_patch, hidden_states, training_phase=2
                )

            # Samplear próximos bytes (todos los heads)
            next_bytes = []
            for head_logits in logits:
                probs = F.softmax(head_logits[0, 0] / temperature, dim=-1)
                byte_val = probs.argmax().item()
                next_bytes.append(byte_val)

            # Boundary: determinar cuántos bytes aceptar del patch
            patch_len = len(next_bytes)
            if self.use_boundary and len(next_bytes) >= 2:
                for k in range(len(next_bytes) - 1):
                    if self.boundary_predictor.forward_infer(
                        next_bytes[k], next_bytes[k + 1], boundary_threshold
                    ):
                        patch_len = k + 1
                        break

            generated.extend(next_bytes[:patch_len])

            # Preparar siguiente input (pad si más corto que PATCH_SIZE)
            input_bytes = next_bytes[:patch_len]
            input_bytes += [0] * (PATCH_SIZE - len(input_bytes))
            last_patch = torch.tensor(
                input_bytes, dtype=torch.long
            ).reshape(1, 1, PATCH_SIZE)

        return bytes(generated[:max_bytes])


# =============================================================================
# 11. Utilidades de conversion bytes <-> patches
# =============================================================================

def bytes_to_patches(raw: bytes) -> Tensor:
    """
    Convierte bytes crudos a patches de 4 bytes.
    Vectorizado: sin loops Python.

    Args:
        raw: bytes de entrada
    Returns:
        [1, n_patches, 4] long
    """
    n = len(raw)
    pad = (PATCH_SIZE - n % PATCH_SIZE) % PATCH_SIZE
    raw = raw + b'\x00' * pad

    # Vectorizado: bytes -> tensor -> reshape directo
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    return t.reshape(1, -1, PATCH_SIZE).long()


# =============================================================================
# 8. Optimizaciones CPU para inferencia
# =============================================================================

def _prepare_bitlinear_weights(model: nn.Module):
    """
    Transponer y hacer contigua la matriz de pesos de todas las capas
    BitLinear ANTES de inferencia.

    En inferencia (1 token): y = x @ W.T
    Si guardamos W_T = W.T.contiguous(), el acceso es secuencial en memoria.
    Ganancia: hasta 2x en throughput de memory bandwidth para matmul 1xNxM.
    """
    for module in model.modules():
        if isinstance(module, BitLinear):
            with torch.no_grad():
                module.weight_T = module.weight.data.T.contiguous()


def assert_aligned(tensor: Tensor, alignment: int = 64):
    """Verifica que un tensor esta alineado a cache line (64 bytes en x86)."""
    offset = tensor.data_ptr() % alignment
    assert offset == 0, f"Tensor no alineado: offset={offset}"


def pack_ternary_to_int8(weight: Tensor) -> tuple[Tensor, Tensor]:
    """
    Empaqueta pesos {-1, 0, 1} en INT8 para inferencia CPU eficiente.

    PyTorch soporta matmul INT8 via torch._int_mm.
    Ventajas vs FP16:
    - 2x menos bandwidth de memoria
    - INT8 matmul en AVX2: 32 elementos/ciclo vs 8 para FP32

    Returns:
        w_int8:  pesos en INT8 [-1, 0, 1]
        w_scale: escalar de re-escalado FP32
    """
    w_scale = weight.abs().mean()
    w_ternary = (weight / (w_scale + 1e-8)).round().clamp(-1, 1)
    w_int8 = w_ternary.to(torch.int8)
    return w_int8, w_scale


def ternary_matmul_int8(x: Tensor, w_int8: Tensor, scale: float) -> Tensor:
    """
    Matmul eficiente usando INT8 acumulacion.

    Args:
        x: [1, d_in] FP32/FP16
        w_int8: [d_out, d_in] INT8
        scale: factor de escala FP32
    Returns:
        [1, d_out] FP32
    """
    x_scale = x.abs().max() / 127.0
    x_int8 = (x / (x_scale + 1e-8)).round().clamp(-128, 127).to(torch.int8)

    # INT8 matmul (requiere 2D tensors y padding a multiplo de 8 para _int_mm)
    m, k = x_int8.shape
    k2 = w_int8.shape[1]
    assert k == k2, f"Dimension mismatch: x={k}, w={k2}"

    # Pad k a multiplo de 8 si es necesario para _int_mm
    pad_k = (8 - k % 8) % 8
    if pad_k > 0:
        x_int8 = F.pad(x_int8, (0, pad_k))
        w_int8 = F.pad(w_int8, (0, pad_k))

    out = torch._int_mm(x_int8, w_int8.T).to(torch.float32)
    return out * (x_scale * scale)


def pack_model_for_int8_inference(model: CPU1ModelV2) -> dict:
    """
    Empaqueta todos los pesos BitLinear del modelo a INT8.
    Retorna un dict con los pesos empaquetados para inferencia.
    """
    packed = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w_int8, w_scale = pack_ternary_to_int8(module.weight.data)
            packed[name] = {'w_int8': w_int8, 'w_scale': w_scale}
    return packed


# =============================================================================
# 2-bit ternary packing — ~16x smaller checkpoints vs FP32
# =============================================================================
# Encoding: {-1, 0, 1} → {0b00, 0b01, 0b10}. 4 values per byte (2 bits each).
# For 170M ternary params: 170MB (INT8) → 42.5MB (2-bit) + ~2MB FP16 extras ≈ 44MB total.
# For 50M ablation: 200MB (FP32) → ~14MB total inference checkpoint.
# =============================================================================

def pack_ternary_2bit(weight: Tensor) -> tuple[Tensor, Tensor]:
    """
    Pack float32 ternary weight matrix into 2-bit uint8.

    Quantizes weight → {-1,0,1} then encodes as {0,1,2} and packs
    4 values per byte using 2-bit slots (bits 0-1, 2-3, 4-5, 6-7).

    Returns:
        packed: CPU uint8 tensor, shape (ceil(numel/4),)
        scale:  CPU float32 scalar (mean absolute value before quantization)
    """
    scale = weight.abs().mean().float()
    w = (weight.float() / (scale + 1e-8)).round().clamp(-1, 1)
    enc = (w + 1).to(torch.uint8).flatten()       # {-1,0,1} → {0,1,2}

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
    """Unpack 2-bit packed ternary back to float32."""
    n = 1
    for s in shape:
        n *= s
    out = torch.empty(packed.numel() * 4, dtype=torch.float32)
    for i in range(4):
        out[i::4] = ((packed >> (2 * i)) & 0b11).float() - 1  # 0,1,2 → -1,0,1
    return out[:n].reshape(shape) * scale


def pack_model_compact(model: nn.Module) -> dict:
    """
    Build a compact state dict for saving/uploading.

    - BitLinear.weight and BitEmbedding.weight → 2-bit packed uint8 (4 values/byte)
    - Everything else (FPResidual, RMSNorm, scales, etc.) → bfloat16
    - No optimizer state — for inference checkpoints only.

    Size for 50M model:  ~14 MB  (vs ~200 MB FP32)
    Size for 170M model: ~44 MB  (vs ~680 MB FP32)

    Usage:
        compact = pack_model_compact(model)
        torch.save({'compact': compact, 'config': config, 'step': step}, path)

    To load:
        ckpt = torch.load(path, weights_only=True)
        state = unpack_compact_state(ckpt['compact'])
        model.load_state_dict(state)
    """
    # Collect names of ternary weight tensors
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
    """
    Reconstruct a full float32 state dict from pack_model_compact() output.
    Call model.load_state_dict(unpack_compact_state(ckpt['compact'])).
    """
    # Gather metadata for packed keys
    meta: dict[str, dict] = {}
    regular: dict[str, Tensor] = {}

    for key, val in compact.items():
        if key.startswith('__2bit_data__'):
            orig = key[len('__2bit_data__'):]
            meta.setdefault(orig, {})['data'] = val
        elif key.startswith('__2bit_scale__'):
            orig = key[len('__2bit_scale__'):]
            meta.setdefault(orig, {})['scale'] = val
        elif key.startswith('__2bit_shape__'):
            orig = key[len('__2bit_shape__'):]
            meta.setdefault(orig, {})['shape'] = val
        else:
            regular[key] = val.float() if isinstance(val, Tensor) and val.is_floating_point() else val

    state: dict = dict(regular)
    for orig_key, m in meta.items():
        state[orig_key] = unpack_2bit_to_float(
            m['data'], float(m['scale']), tuple(m['shape'])
        )

    return state


def compile_for_inference(model: CPU1ModelV2, aggressive: bool = True):
    """
    Aplicar torch.compile al modelo para inferencia.

    torch.compile fusiona ops pequenas (sigmoid, mul, add) en un solo kernel.
    Especialmente efectivo para el MLGRU con muchos Hadamard consecutivos.

    Args:
        model: modelo a compilar
        aggressive: True = max-autotune (lento en compilar, rapido en runtime)
    Returns:
        modelo compilado
    """
    mode = "max-autotune" if aggressive else "reduce-overhead"
    try:
        compiled = torch.compile(model, mode=mode, backend="inductor")
        return compiled
    except Exception:
        # Fallback si torch.compile no esta disponible o falla
        return model


# =============================================================================
# 9. Ablation configs y benchmarks
# =============================================================================

def run_architecture_ablation():
    """
    Tabla de ablation para el paper.
    4 variantes a medir en el MISMO hardware CPU consumer.

    Hipotesis:
    - V3 > V2 en tokens/seg (MLGRU elimina KV cache, evita RAM stalls)
    - V4 > V3 si byte-level tiene menor perplexity por byte vs BPE
    - V4 > V1 porque byte-level + MLGRU > BPE + ternario Mamba
    """
    configs = {
        'v1_slender_mamba_bpe': {
            'arch': 'mamba2',
            'tokenizer': 'bpe',
            'quantization': 'ternary_full',
            'note': 'baseline Slender-Mamba style',
        },
        'v2_cpu1_attention': {
            'arch': 'transformer_sliding_window',
            'tokenizer': 'byte',
            'quantization': 'ternary_partial',
            'note': 'CPU-1 original del README',
        },
        'v3_cpu1_mlgru': {
            'arch': 'mlgru_matmulfree',
            'tokenizer': 'byte',
            'quantization': 'ternary_partial',
            'note': 'MLGRU reemplaza atencion',
        },
        'v4_cpu1_mlgru_full': {
            'arch': 'mlgru_matmulfree',
            'tokenizer': 'byte',
            'quantization': 'ternary_full',
            'note': 'todo ternario, sin MatMul',
        },
        'v5_cpu1_170m_pfnet': {
            'arch': 'mlgru_matmulfree_pfnet',
            'tokenizer': 'byte',
            'quantization': 'ternary_full',
            'note': '170M, PFNet nonlinear residual',
        },
    }

    metrics = [
        'tokens_per_sec_cpu',
        'latency_p50_ms',
        'ram_mb',
        'model_size_mb',
        'perplexity_fineweb',
        'cache_miss_rate',
        'state_size_kb',
    ]

    return configs, metrics


def benchmark_cpu_inference(
    model: CPU1ModelV2,
    n_patches: int = 100,
    warmup: int = 10,
) -> dict:
    """
    Benchmark de inferencia CPU: mide throughput y latencia.

    Args:
        model: modelo v2
        n_patches: cantidad de patches a generar
        warmup: patches de warmup (no contados)
    Returns:
        dict con metricas
    """
    model.eval()
    model.prepare_for_inference()
    device = next(model.parameters()).device

    # Inicializar estado
    hidden_states = [None] * len(model.blocks)
    dummy_patch = torch.randint(0, 256, (1, 1, PATCH_SIZE), device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            logits, hidden_states = model.forward(dummy_patch, hidden_states)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_patches):
            t0 = time.perf_counter()
            logits, hidden_states = model.forward(dummy_patch, hidden_states)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # ms

    latencies.sort()
    total_time_s = sum(latencies) / 1000.0
    bytes_generated = n_patches * PATCH_SIZE

    return {
        'patches_per_sec': n_patches / total_time_s,
        'bytes_per_sec': bytes_generated / total_time_s,
        'latency_p50_ms': latencies[len(latencies) // 2],
        'latency_p99_ms': latencies[int(len(latencies) * 0.99)],
        'total_time_s': total_time_s,
        'n_patches': n_patches,
    }


def benchmark_byte_vs_bpe_throughput():
    """
    Test critico: patches de bytes vs tokens BPE en CPU.

    Hipotesis: SI, porque:
    1. vocab_size=256 vs 32K -> embedding lookup 128x mas pequeno
    2. Tokenizador BPE tiene overhead de Python/Rust
    3. Patches de 4 bytes tienen longitud fija -> sin padding

    Contraargumento: secuencia en bytes es 3-5x mas larga que en tokens BPE.

    Placeholder: requiere modelo BPE baseline para comparacion directa.
    """
    pass


# =============================================================================
# 10. Generador optimizado con pre-alocacion de buffers
# =============================================================================

class CPU1GeneratorV2:
    """
    Generador optimizado para inferencia CPU.
    Pre-aloca buffers para evitar garbage collector jitter.
    """

    def __init__(self, model: CPU1ModelV2):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    def generate(
        self,
        prompt: str,
        max_bytes: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Genera texto desde un prompt string."""
        raw = prompt.encode('utf-8')
        generated = self.model.generate_bytes(
            raw, max_tokens=max_bytes,
            temperature=temperature, top_p=top_p,
        )
        # Decodificar, removiendo trailing nulls
        return generated.rstrip(b'\x00').decode('utf-8', errors='replace')

    @torch.no_grad()
    def generate_optimized(
        self,
        prompt_bytes: bytes,
        max_patches: int = 50,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> bytes:
        """
        Generacion optimizada: pre-aloca buffers y usa forward_one_token.

        Pre-aloca todos los buffers UNA VEZ antes del loop para evitar
        allocaciones que disparan el garbage collector.
        """
        # Prefill
        patches = bytes_to_patches(prompt_bytes).to(self.device)
        _, hidden_states = self.model.forward(patches, None)

        # Pre-alocar buffers
        output_buf = bytearray(max_patches * PATCH_SIZE)
        probs_buf = torch.zeros(VOCAB_SIZE, device=self.device)

        # Embedding del ultimo patch para empezar generacion
        last_patch = patches[:, -1:, :]
        x = self.model.patch_embedding(last_patch).squeeze(1)  # [B, D]

        for i in range(max_patches):
            # Forward a traves de todas las capas
            x_out, hidden_states = self.model.generate_one_patch(x, hidden_states)

            # Norm final y heads
            x_normed = self.model.norm(x_out.unsqueeze(1))  # [B, 1, D]
            next_bytes = []
            for head in self.model.byte_heads:
                logits = head(x_normed).squeeze(1).squeeze(0)  # [256]
                torch.softmax(logits / temperature, dim=-1, out=probs_buf)

                # Top-p filtering in-place
                sorted_probs, sorted_idx = probs_buf.sort(descending=True)
                cumsum = sorted_probs.cumsum(dim=0)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()

                byte_val = sorted_idx[torch.multinomial(sorted_probs, 1)].item()
                next_bytes.append(byte_val)

            for j, bv in enumerate(next_bytes):
                output_buf[i * PATCH_SIZE + j] = bv

            # Embed siguiente patch
            next_patch = torch.tensor(
                next_bytes, dtype=torch.long, device=self.device
            ).reshape(1, 1, PATCH_SIZE)
            x = self.model.patch_embedding(next_patch).squeeze(1)

        return bytes(output_buf)


# =============================================================================
# 11. Tests
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Cuenta total de parametros del modelo."""
    return sum(p.numel() for p in model.parameters())


def print_parameter_breakdown(model: CPU1ModelV2):
    """Prints a detailed parameter breakdown: embedding, per-layer (MLGRU + FFN + PFNet), heads, total."""
    total = count_parameters(model)
    embed_params = count_parameters(model.patch_embedding)
    norm_params = count_parameters(model.norm)
    head_params = sum(count_parameters(h) for h in model.byte_heads)
    block_params = count_parameters(model.blocks)

    print(f"Parameter breakdown (~{total / 1e6:.0f}M total):")
    print(f"  Embedding:       {embed_params:>12,}")
    print(f"  Blocks (x{model.n_layers}):    {block_params:>12,}")

    block = model.blocks[0]
    mixer_p = count_parameters(block.token_mixer)
    ffn_p = count_parameters(block.ffn)
    pfnet_p = count_parameters(block.pfnet)
    norms_p = count_parameters(block.norm1) + count_parameters(block.norm2)
    print(f"    Per block:")
    print(f"      MLGRU mixer: {mixer_p:>10,}")
    print(f"      SwiGLU FFN:  {ffn_p:>10,}")
    print(f"      PFNet:       {pfnet_p:>10,}")
    print(f"      Norms:       {norms_p:>10,}")

    print(f"  Final norm:      {norm_params:>12,}")
    print(f"  Heads (x4):      {head_params:>12,}")
    print(f"  Total:           {total:>12,}")


def test_bit_embedding():
    """Verifica BitEmbedding."""
    emb = BitEmbedding(256, 8)
    indices = torch.randint(0, 256, (2, 16, 4))
    out = emb(indices)
    assert out.shape == (2, 16, 4, 8), f"Shape error: {out.shape}"
    print("[OK] BitEmbedding OK")


def test_byte_patch_embedding_v2():
    """Verifica BytePatchEmbeddingV2."""
    emb = BytePatchEmbeddingV2(d_model=128)
    patches = torch.randint(0, 256, (2, 16, 4))
    out = emb(patches)
    assert out.shape == (2, 16, 128), f"Shape error: {out.shape}"
    print("[OK] BytePatchEmbeddingV2 OK")


def test_matmul_free_token_mixer():
    """Verifica MatMulFreeTokenMixer forward y forward_one_token."""
    D = 128
    mixer = MatMulFreeTokenMixer(D, fp_rank=4)
    x = torch.randn(2, 8, D)

    # Forward completo
    out, h = mixer(x)
    assert out.shape == (2, 8, D), f"Shape error: {out.shape}"
    assert h.shape == (2, D), f"State shape error: {h.shape}"

    # Forward one token
    x1 = torch.randn(2, D)
    h1 = torch.zeros(2, D)
    out1, h1_new = mixer.forward_one_token(x1, h1)
    assert out1.shape == (2, D), f"Shape error: {out1.shape}"
    assert h1_new.shape == (2, D), f"State shape error: {h1_new.shape}"

    print("[OK] MatMulFreeTokenMixer OK")


def test_bit_head():
    """Verifica BitHead."""
    head = BitHead(128, 256)
    x = torch.randn(2, 8, 128)
    out = head(x)
    assert out.shape == (2, 8, 256), f"Shape error: {out.shape}"
    print("[OK] BitHead OK")


def test_cpu1_block_v2():
    """Verifica CPU1BlockV2 forward y forward_one_token."""
    D = 128
    block = CPU1BlockV2(d_model=D, d_ff=256, fp_rank=4)
    x = torch.randn(2, 8, D)

    # Forward completo
    out, h = block(x)
    assert out.shape == (2, 8, D), f"Shape error: {out.shape}"
    assert h.shape == (2, D), f"State shape error: {h.shape}"

    # Forward one token
    x1 = torch.randn(2, D)
    h1 = torch.zeros(2, D)
    out1, h1_new = block.forward_one_token(x1, h1)
    assert out1.shape == (2, D), f"Shape error: {out1.shape}"

    print("[OK] CPU1BlockV2 OK")


def test_cpu1_model_v2():
    """Verifica CPU1ModelV2 forward pass."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)
    model.eval()

    patches = torch.randint(0, 256, (2, 8, PATCH_SIZE))

    with torch.no_grad():
        logits, states = model(patches)

    assert len(logits) == 4, f"Expected 4 heads, got {len(logits)}"
    for i, l in enumerate(logits):
        assert l.shape == (2, 8, VOCAB_SIZE), (
            f"Head {i}: shape {l.shape}, expected (2, 8, {VOCAB_SIZE})"
        )
    assert len(states) == 2, f"Expected 2 states, got {len(states)}"

    print(f"[OK] CPU1ModelV2 forward OK: 4 x {logits[0].shape}")


def test_generate_bytes():
    """Verifica generacion de bytes."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)

    generated = model.generate_bytes(b"Hello", max_tokens=16, temperature=1.0)
    assert len(generated) == 16, f"Expected 16 bytes, got {len(generated)}"
    assert all(0 <= b <= 255 for b in generated), "Invalid byte values"

    print(f"[OK] generate_bytes OK: {len(generated)} bytes generados")


def test_parameter_count_v2():
    """Verifica parametros del modelo completo v2 (~170M)."""
    model = CPU1ModelV2()
    total = count_parameters(model)
    print_parameter_breakdown(model)
    print(f"\n[OK] Test de parametros v2 paso: {total:,}")


def test_prepare_for_inference():
    """Verifica que prepare_for_inference funciona."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)
    result = model.prepare_for_inference()
    assert result is True

    # Verificar que los pesos transpuestos existen
    for module in model.modules():
        if isinstance(module, BitLinear):
            assert hasattr(module, 'weight_T'), "weight_T not created"
            assert module.weight_T.shape == module.weight.T.shape

    print("[OK] prepare_for_inference OK")


def test_pack_ternary():
    """Verifica empaquetado a INT8."""
    w = torch.randn(256, 128)
    w_int8, w_scale = pack_ternary_to_int8(w)
    assert w_int8.dtype == torch.int8
    assert w_int8.shape == (256, 128)
    assert (w_int8.abs() <= 1).all(), "Valores fuera de {-1, 0, 1}"
    print("[OK] pack_ternary_to_int8 OK")


def test_bytes_to_patches():
    """Verifica conversion bytes -> patches."""
    raw = b"Hello World!"
    patches = bytes_to_patches(raw)
    assert patches.shape == (1, 3, 4), f"Shape error: {patches.shape}"
    assert patches[0, 0, 0].item() == ord('H')
    assert patches[0, 0, 1].item() == ord('e')
    print("[OK] bytes_to_patches OK")


def test_cpu1_block_v2_one_token_consistency():
    """
    Verifica que forward_one_token produce el mismo resultado
    que forward con S=1 para un bloque.
    """
    D = 128
    block = CPU1BlockV2(d_model=D, d_ff=256, fp_rank=4)
    block.eval()

    x = torch.randn(1, D)
    h = torch.randn(1, D)

    with torch.no_grad():
        # Via forward (3D)
        x_3d = x.unsqueeze(1)
        out_3d, h_3d = block(x_3d, h)
        out_3d = out_3d.squeeze(1)

        # Via forward_one_token (2D)
        out_2d, h_2d = block.forward_one_token(x, h)

    assert torch.allclose(out_3d, out_2d, atol=1e-5), (
        f"Outputs differ: max diff = {(out_3d - out_2d).abs().max()}"
    )
    assert torch.allclose(h_3d, h_2d, atol=1e-5), (
        f"States differ: max diff = {(h_3d - h_2d).abs().max()}"
    )

    print("[OK] forward_one_token consistency OK")


def test_benchmark():
    """Verifica que el benchmark funciona."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)

    results = benchmark_cpu_inference(model, n_patches=20, warmup=5)
    assert results['patches_per_sec'] > 0
    assert results['latency_p50_ms'] > 0

    print(f"[OK] Benchmark: {results['bytes_per_sec']:.0f} bytes/sec, "
          f"p50={results['latency_p50_ms']:.1f}ms")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Componentes individuales v2")
    print("=" * 60)
    test_bit_embedding()
    test_byte_patch_embedding_v2()
    test_matmul_free_token_mixer()
    test_bit_head()
    test_cpu1_block_v2()

    print("\n" + "=" * 60)
    print("Test 2: Modelo completo v2")
    print("=" * 60)
    test_cpu1_model_v2()

    print("\n" + "=" * 60)
    print("Test 3: Generacion de bytes")
    print("=" * 60)
    test_generate_bytes()

    print("\n" + "=" * 60)
    print("Test 4: Consistencia forward_one_token")
    print("=" * 60)
    test_cpu1_block_v2_one_token_consistency()

    print("\n" + "=" * 60)
    print("Test 5: Inferencia CPU optimizations")
    print("=" * 60)
    test_prepare_for_inference()
    test_pack_ternary()

    print("\n" + "=" * 60)
    print("Test 6: Utilidades")
    print("=" * 60)
    test_bytes_to_patches()

    print("\n" + "=" * 60)
    print("Test 7: Benchmark")
    print("=" * 60)
    test_benchmark()

    print("\n" + "=" * 60)
    print("Test 8: Conteo de parametros (modelo completo)")
    print("=" * 60)
    test_parameter_count_v2()
