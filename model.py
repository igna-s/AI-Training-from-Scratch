"""
CPU-1: Modelo de lenguaje byte-level con pesos ternarios (~1B params).

Arquitectura:
- Byte-patch embedding (4 bytes -> 2048 dims)
- 20 bloques transformer con BitLinear + FPResidual
- Sliding window attention (512) con sink tokens (4)
- SwiGLU FFN con d_ff=5504
- 4 cabezas de predicción (una por byte del próximo patch)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =============================================================================
# Configuracion global
# Estos valores definen la geometria completa del modelo (~1B params).
# Cambiar cualquiera afecta el conteo de parametros y la compatibilidad
# con checkpoints existentes.
# =============================================================================

D_MODEL = 2048       # Dimension del hidden state en todo el modelo
N_LAYERS = 20        # Cantidad de bloques transformer apilados
N_HEADS = 16         # Cabezas de atencion en multi-head attention
HEAD_DIM = D_MODEL // N_HEADS  # 128 dims por cabeza (2048/16)
D_FF = 5504          # Dimension intermedia del FFN SwiGLU (8/3 * d_model, redondeado a multiplo de 128 para alinear con SIMD/tensor cores)
FP_RANK = 8          # Rango de la correccion FP16 en FPResidual (bajo = pocos params extra)
VOCAB_SIZE = 256     # Vocabulario = 256 bytes posibles (0x00-0xFF)
PATCH_SIZE = 4       # Bytes por patch: el modelo consume de a 4 bytes por posicion
WINDOW_SIZE = 512    # Ventana de atencion local en patches (512 * 4 = 2048 bytes)
N_SINKS = 4          # Sink tokens: los primeros 4 patches siempre visibles (capturan info global)


# =============================================================================
# 1. RMSNorm
# =============================================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Mas ligero que LayerNorm: no centra (no resta media), solo escala.
    Usado como pre-norm antes de cada sub-capa (atencion y FFN).

    Args:
        dim: dimension del ultimo eje del tensor a normalizar (d_model=2048)
        eps: epsilon para estabilidad numerica en la division
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # Escala aprendida, inicializada en 1
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [..., dim] tensor de cualquier batch shape
        Returns:
            tensor normalizado, misma shape que x
        """
        # rsqrt = 1/sqrt, mean sobre ultima dim = RMS del vector
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


# =============================================================================
# 2. BitLinear — Linear con pesos ternarios {-1, 0, 1} y STE
# =============================================================================

def _ste_round(x: Tensor) -> Tensor:
    """
    Round con Straight-Through Estimator (STE).
    En forward: redondea al entero mas cercano.
    En backward: el gradiente pasa directo (como si no hubiera redondeo).
    Truco: (x.round() - x).detach() tiene grad=0, asi que backward ve identidad.

    Args:
        x: tensor de cualquier shape
    Returns:
        tensor redondeado con misma shape, gradientes fluyen como si fuera identidad
    """
    return x + (x.round() - x).detach()


class BitLinear(nn.Module):
    """
    Linear con cuantización ternaria en forward.

    Pesos se mantienen en float durante training. En forward:
    1. Cuantizar pesos a ternarios via absmax + round + clip
    2. Cuantizar activaciones a int8 per-token
    3. Matmul cuantizada con rescaling
    4. STE para backward

    NO incluye RMSNorm interno — la normalización viene del pre-norm
    del bloque transformer. Esto evita doble normalización y reduce
    operaciones en inferencia CPU (7 norms menos por bloque).
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features:  dimension de entrada (ej: d_model=2048)
            out_features: dimension de salida (ej: n_heads*head_dim=2048)
        Nota: sin bias, los pesos float se mantienen para training via STE.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.weight, std=0.02)  # Inicializacion estandar para transformers

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, S, in_features] — activaciones de entrada
        Returns:
            [B, S, out_features] — resultado de la matmul cuantizada

        Flujo completo:
            1. Pesos float -> ternarios {-1, 0, 1} via absmax scaling
            2. Activaciones float -> int8 per-token via absmax scaling
            3. Matmul entera (simulada en float) + rescaling para recuperar magnitud
        En backward, STE hace que los gradientes actualicen los pesos float originales.
        """
        # --- Paso 1: Cuantizar pesos a ternarios {-1, 0, 1} ---
        w = self.weight
        w_scale = w.abs().mean()  # Factor de escala: media del valor absoluto de todos los pesos
        w_normalized = w / (w_scale + 1e-8)  # Normalizar a rango ~[-1, 1]
        w_ternary = torch.clamp(_ste_round(w_normalized), -1.0, 1.0)  # Redondear y clipear a {-1, 0, 1}

        # --- Paso 2: Cuantizar activaciones a int8 per-token ---
        x_scale = x.abs().amax(dim=-1, keepdim=True)  # [B, S, 1] — maximo absoluto por token
        x_normalized = x / (x_scale + 1e-8)  # Normalizar a [-1, 1]
        x_quant = torch.clamp(_ste_round(x_normalized * 127.0), -128.0, 127.0)  # Escalar a rango int8

        # --- Paso 3: Matmul cuantizada + restaurar escala ---
        output = F.linear(x_quant, w_ternary)  # [B, S, out_features] — matmul en dominio cuantizado
        # Restaurar magnitud: (x_quant/127 * x_scale) @ (w_ternary * w_scale)
        # = output * (x_scale * w_scale / 127)
        output = output * (x_scale * w_scale / 127.0)

        return output


# =============================================================================
# 3. FPResidual — Rama paralela FP16 de rango bajo
# =============================================================================

class FPResidual(nn.Module):
    """
    Corrección FP16 de rango bajo en paralelo con BitLinear.

    Estructura: x -> Linear(in_dim, fp_rank) -> SiLU -> Linear(fp_rank, out_dim)
    fp_B inicializado en ceros -> empieza sin efecto.

    Soporta dims diferentes para input/output (necesario para parallel
    con down_proj en SwiGLU donde input=d_ff, output=d_model).
    """

    def __init__(self, in_dim: int, out_dim: int | None = None, rank: int = FP_RANK):
        """
        Args:
            in_dim:  dimension de entrada (d_model=2048, o d_ff=5504 para down_proj)
            out_dim: dimension de salida (None = same as in_dim). Distinto para SwiGLU
                     donde down_proj va de d_ff=5504 -> d_model=2048
            rank:    rango del bottleneck (default 8 = muy pocos params extra)
        """
        super().__init__()
        out_dim = out_dim or in_dim
        self.fp_A = nn.Linear(in_dim, rank, bias=False)   # Down projection: comprime a rank dims
        self.fp_B = nn.Linear(rank, out_dim, bias=False)   # Up projection: expande a out_dim
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.fp_A.weight, std=0.02)
        # fp_B en CEROS: al inicio del training la rama FPResidual no aporta nada.
        # Esto da estabilidad inicial (solo BitLinear actua) y gradualmente
        # FPResidual aprende correcciones finas de alta precision.
        nn.init.zeros_(self.fp_B.weight)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, S, in_dim]
        Returns:
            [B, S, out_dim] — correccion FP16 de rango bajo
        Nota: se usa en PARALELO con BitLinear: output = BitLinear(x) + FPResidual(x)
        """
        return self.fp_B(F.silu(self.fp_A(x)))  # in_dim -> rank -> SiLU -> out_dim


# =============================================================================
# 4. RoPE — Rotary Positional Embeddings
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE aplicado a Q y K. Posiciones siempre absolutas.
    Precomputa frecuencias y las cachea hasta max_seq_len.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        """
        Args:
            dim:         dimension por cabeza (head_dim=128)
            max_seq_len: maximo de posiciones a cachear (se extiende automaticamente si se supera)
            base:        base para las frecuencias inversas (10000 es estandar)
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Frecuencias inversas: theta_i = 1 / (base^(2i/dim)) para i=0,1,...,dim/2-1
        # Dims bajos = frecuencias altas (capturan patrones locales)
        # Dims altos = frecuencias bajas (capturan patrones globales/posicionales)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Pre-calcular cos/sin para todas las posiciones hasta max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Construye tablas cos/sin para posiciones 0..seq_len-1."""
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2] — angulos por posicion
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, dim] — duplicar para cubrir dim completo
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = seq_len

    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """
        Aplica RoPE a x usando position_ids absolutas.

        Args:
            x: [B, n_heads, S, head_dim]
            position_ids: [B, S] o [1, S] con posiciones absolutas
        Returns:
            x rotado: misma shape que x
        """
        seq_len = position_ids.shape[-1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        # Obtener cos/sin para las posiciones pedidas
        cos = self.cos_cached[position_ids].unsqueeze(1)  # [B, 1, S, dim]
        sin = self.sin_cached[position_ids].unsqueeze(1)  # [B, 1, S, dim]

        # Rotacion compleja: x = (x1, x2) -> (x1*cos - x2*sin, x2*cos + x1*sin)
        # Equivale a multiplicar por e^(i*theta) en el plano complejo
        x1, x2 = x.chunk(2, dim=-1)  # Partir en mitades: parte "real" e "imaginaria"
        rotated = torch.cat([-x2, x1], dim=-1)  # [-x2, x1] para la rotacion
        return x * cos + rotated * sin  # Resultado: pares de dims codifican posicion relativa


# =============================================================================
# 5. BytePatchEmbedding
# =============================================================================

class BytePatchEmbedding(nn.Module):
    """
    Convierte patches de 4 bytes crudos en embeddings de d_model dims.

    Cada byte pasa por un embedding aprendido de 8 dims.
    4 bytes x 8 dims = 32 features -> Linear(32, d_model) sin bias.

    Input:  [batch, seq_len, 4] uint8
    Output: [batch, seq_len, d_model] float
    """

    def __init__(self, d_model: int = D_MODEL, patch_size: int = PATCH_SIZE,
                 byte_embed_dim: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed_dim = byte_embed_dim
        # Embedding aprendido por byte (256 valores posibles -> 8 dims)
        self.byte_embedding = nn.Embedding(256, byte_embed_dim)
        # Proyección de 4*8=32 -> d_model
        self.proj = nn.Linear(patch_size * byte_embed_dim, d_model, bias=False)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.byte_embedding.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, patches: Tensor) -> Tensor:
        """
        Args:
            patches: [B, S, 4] dtype long o uint8
        Returns:
            [B, S, d_model]
        """
        B, S, P = patches.shape
        # Embedding por byte: [B, S, 4] -> [B, S, 4, 8]
        x = self.byte_embedding(patches.long())
        # Flatten: [B, S, 4, 8] -> [B, S, 32]
        x = x.reshape(B, S, P * self.byte_embed_dim)
        # Proyectar: [B, S, 32] -> [B, S, d_model]
        return self.proj(x)


# =============================================================================
# 6. SlidingWindowAttention — Atención local + sink tokens
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    Multi-head attention con ventana deslizante + sink tokens.

    Cada token atiende a:
    - Los N_SINKS (4) primeros tokens (siempre visibles)
    - Los WINDOW_SIZE (512) tokens más recientes

    Proyecciones Q, K, V via BitLinear. Output projection via BitLinear + FPResidual.
    """

    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 head_dim: int = HEAD_DIM, window_size: int = WINDOW_SIZE,
                 n_sinks: int = N_SINKS):
        """
        Args:
            d_model:     dimension del modelo (2048)
            n_heads:     numero de cabezas de atencion (16)
            head_dim:    dimension por cabeza (128, d_model/n_heads)
            window_size: tamanio de la ventana local (512 patches)
            n_sinks:     cantidad de sink tokens que siempre son visibles (4)
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.n_sinks = n_sinks
        self.scale = head_dim ** -0.5  # 1/sqrt(head_dim) para escalar los attention scores

        # Proyecciones Q, K, V via BitLinear (ternarias, sin bias)
        # Cada una: d_model -> n_heads * head_dim (2048 -> 2048)
        self.q_proj = BitLinear(d_model, n_heads * head_dim)
        self.k_proj = BitLinear(d_model, n_heads * head_dim)
        self.v_proj = BitLinear(d_model, n_heads * head_dim)

        # Output projection: BitLinear + FPResidual en PARALELO
        # Ambos toman la misma entrada y se suman: o_proj(x) + o_fp_residual(x)
        # FPResidual aporta correccion fina FP16 a la proyeccion ternaria
        self.o_proj = BitLinear(n_heads * head_dim, d_model)
        self.o_fp_residual = FPResidual(d_model)

        # RoPE: codifica posiciones absolutas en Q y K
        self.rope = RotaryPositionalEmbedding(head_dim)

        # Cache de mascara: se construye lazy (una vez por seq_len) y se reutiliza
        self._cached_mask: Tensor | None = None
        self._cached_mask_seq_len: int = 0

    def _build_sliding_window_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Construye máscara causal con sliding window + sinks.

        Returns: [S, S] bool mask donde True = PERMITIR atención
        """
        # Mascara causal base: triangular inferior, cada token solo ve anteriores + si mismo
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

        # Mascara de ventana: token en fila i atiende a columna j si (i - j) < window_size
        # Esto limita la atencion a los ultimos 512 tokens (ventana local)
        rows = torch.arange(seq_len, device=device).unsqueeze(1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)
        window = (rows - cols) < self.window_size

        # Sinks: las primeras n_sinks columnas (posiciones 0..3) siempre visibles
        # para TODOS los tokens, sin importar la distancia. Capturan info global
        # (ej: BOS, formato del texto, idioma)
        sinks = cols < self.n_sinks

        # Combinar: un token puede atender si:
        #   (es causal) Y (esta en la ventana O es un sink)
        mask = causal & (window | sinks)
        return mask

    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        """
        Args:
            x: [B, S, d_model]
            position_ids: [B, S] posiciones absolutas
        Returns:
            [B, S, d_model]
        """
        B, S, _ = x.shape

        # Proyectar Q, K, V y reshape a multi-head: [B, S, d_model] -> [B, H, S, head_dim]
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Aplicar RoPE a Q y K: codifica posiciones absolutas en el espacio de atencion.
        # Dos tokens con misma distancia relativa tendran el mismo dot product
        # independientemente de su posicion absoluta (propiedad clave de RoPE).
        q = self.rope(q, position_ids)
        k = self.rope(k, position_ids)

        # Attention scores: Q @ K^T / sqrt(head_dim) -> [B, H, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Aplicar mascara sliding window + sinks (cacheada por seq_len para eficiencia)
        if self._cached_mask is None or self._cached_mask_seq_len != S or self._cached_mask.device != x.device:
            self._cached_mask = self._build_sliding_window_mask(S, x.device)
            self._cached_mask_seq_len = S
        mask = self._cached_mask
        # Donde mask=False, poner -inf para que softmax de 0 (no atiende)
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, S, S] — probabilidades de atencion

        # Weighted sum de values: [B, H, S, S] @ [B, H, S, head_dim] -> [B, H, S, head_dim]
        attn_output = torch.matmul(attn_weights, v)
        # Reshape multi-head -> concatenar: [B, H, S, head_dim] -> [B, S, d_model]
        attn_output = attn_output.transpose(1, 2).reshape(B, S, self.d_model)

        # Output projection: BitLinear (ternaria) + FPResidual (FP16 correccion)
        # Ambas reciben el mismo input y se suman
        return self.o_proj(attn_output) + self.o_fp_residual(attn_output)


# =============================================================================
# 7. SwiGLU FFN
# =============================================================================

class SwiGLU(nn.Module):
    """
    FFN con gating multiplicativo SwiGLU.

    gate, up: BitLinear (d_model -> d_ff)
    down: BitLinear (d_ff -> d_model) + FPResidual PARALELO (d_ff -> d_model)

    forward(x):
        gate = silu(BitLinear_gate(x))
        up = BitLinear_up(x)
        hidden = gate * up
        return BitLinear_down(hidden) + FPResidual(hidden)  # ambos toman hidden
    """

    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        """
        Args:
            d_model: dimension del modelo (2048) — entrada y salida
            d_ff:    dimension intermedia (5504) — 8/3 * d_model, expandida
        3 proyecciones BitLinear: gate, up (ambas d_model->d_ff), down (d_ff->d_model)
        + FPResidual paralelo al down para correccion fina.
        """
        super().__init__()
        self.gate_proj = BitLinear(d_model, d_ff)  # Genera la "compuerta" multiplicativa
        self.up_proj = BitLinear(d_model, d_ff)    # Genera el valor a modular
        self.down_proj = BitLinear(d_ff, d_model)  # Comprime de vuelta a d_model
        # FPResidual en PARALELO con down_proj: correccion FP16 para la compresion
        self.down_fp_residual = FPResidual(d_ff, d_model)  # d_ff -> rank=8 -> d_model

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, S, d_model]
        Returns:
            [B, S, d_model]
        SwiGLU: silu(gate(x)) * up(x) -> gating multiplicativo que permite al modelo
        "elegir" que features de up pasar, controlado por la rama gate.
        """
        gate = F.silu(self.gate_proj(x))        # [B, S, d_ff] — gate con activacion suave
        up = self.up_proj(x)                      # [B, S, d_ff] — valor sin activacion
        hidden = gate * up                        # [B, S, d_ff] — gating element-wise
        # Down projection: BitLinear (ternaria, bulk) + FPResidual (FP16, correccion fina)
        return self.down_proj(hidden) + self.down_fp_residual(hidden)


# =============================================================================
# 8. CPU1Block — Bloque transformer pre-norm
# =============================================================================

class CPU1Block(nn.Module):
    """
    Un bloque transformer con pre-norm.

    forward(x, position_ids):
        h = x + Attention(RMSNorm(x), position_ids)
        out = h + SwiGLU(RMSNorm(h))
    """

    def __init__(self, d_model: int = D_MODEL):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = SlidingWindowAttention(d_model=d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model=d_model)

    def forward(self, x: Tensor, position_ids: Tensor) -> Tensor:
        # Pre-norm -> Attention -> Residual
        h = x + self.attn(self.attn_norm(x), position_ids)
        # Pre-norm -> FFN -> Residual
        return h + self.ffn(self.ffn_norm(h))


# =============================================================================
# 9. CPU1Model — Modelo completo
# =============================================================================

class CPU1Model(nn.Module):
    """
    CPU-1: Modelo byte-level con pesos ternarios, ~1B params.

    Input:  patches [B, S, 4] uint8
    Output: tuple de 4 tensores [B, S, 256], uno por byte del próximo patch

    Cabezas de predicción: nn.Linear estándar FP16 (no BitLinear).
    Sin weight tying (shapes incompatibles con embedding).
    """

    def __init__(self, d_model: int = D_MODEL, n_layers: int = N_LAYERS,
                 use_gradient_checkpointing: bool = False):
        """
        Args:
            d_model: dimension del hidden state (2048)
            n_layers: bloques transformer (20 para ~1B params, usar menos para testing)
            use_gradient_checkpointing: si True, recomputa activaciones en backward
                                        para ahorrar VRAM (~30% menos) a costa de velocidad
        """
        super().__init__()
        self.d_model = d_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Patch embedding: convierte [B, S, 4] bytes -> [B, S, 2048] embeddings
        self.embedding = BytePatchEmbedding(d_model=d_model)

        # Stack de N bloques transformer (pre-norm, cada uno con atencion + FFN)
        self.blocks = nn.ModuleList([
            CPU1Block(d_model=d_model) for _ in range(n_layers)
        ])

        # Normalizacion final antes de las cabezas de prediccion
        self.final_norm = RMSNorm(d_model)

        # 4 cabezas de prediccion INDEPENDIENTES, una por byte del proximo patch.
        # Son nn.Linear estandar FP16 (NO BitLinear) porque necesitan maxima precision
        # para los logits finales. Cada una proyecta d_model -> 256 (un logit por byte posible).
        self.head_0 = nn.Linear(d_model, VOCAB_SIZE)  # Predice byte 0 del proximo patch
        self.head_1 = nn.Linear(d_model, VOCAB_SIZE)  # Predice byte 1
        self.head_2 = nn.Linear(d_model, VOCAB_SIZE)  # Predice byte 2
        self.head_3 = nn.Linear(d_model, VOCAB_SIZE)  # Predice byte 3

        self._init_heads()

    def _init_heads(self):
        for head in [self.head_0, self.head_1, self.head_2, self.head_3]:
            nn.init.normal_(head.weight, std=0.02)
            nn.init.zeros_(head.bias)

    def forward(self, patches: Tensor, position_ids: Tensor | None = None
                ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Args:
            patches: [B, S, 4] uint8 o long — patches de entrada
            position_ids: [B, S] posiciones absolutas (opcional, default 0..S-1)
        Returns:
            tuple de 4 tensores [B, S, 256] — logits por byte del próximo patch
        """
        B, S, _ = patches.shape

        # Posiciones por defecto
        if position_ids is None:
            position_ids = torch.arange(S, device=patches.device).unsqueeze(0).expand(B, -1)

        # Embedding
        x = self.embedding(patches)  # [B, S, d_model]

        # Transformer blocks: pasar x por las 20 capas secuencialmente
        # Con gradient checkpointing: no guarda activaciones intermedias, las recomputa
        # en backward. Ahorra ~30% VRAM pero es ~20% mas lento. Solo activo en training.
        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for block in self.blocks:
                x = checkpoint(block, x, position_ids, use_reentrant=False)
        else:
            for block in self.blocks:
                x = block(x, position_ids)

        # Final norm
        x = self.final_norm(x)  # [B, S, d_model]

        # 4 cabezas de predicción
        return (
            self.head_0(x),  # [B, S, 256]
            self.head_1(x),  # [B, S, 256]
            self.head_2(x),  # [B, S, 256]
            self.head_3(x),  # [B, S, 256]
        )


# =============================================================================
# 10. ChunkEncoder — Encoder para retrieval (solo inferencia)
# =============================================================================

class ChunkEncoder:
    """
    Encoder de chunks para in-context retrieval.
    Reutiliza las primeras 4 capas del modelo como encoder.

    Chunk size: 64 patches = 256 bytes.
    Output: mean pooling del output de capa 4 -> [d_model] por chunk.
    Sin parametros adicionales (reutiliza el modelo).
    """

    def __init__(self, model: 'CPU1Model', n_encoder_layers: int = 4,
                 chunk_size: int = 64):
        self.model = model
        self.n_encoder_layers = n_encoder_layers
        self.chunk_size = chunk_size

    @torch.no_grad()
    def encode(self, patches: Tensor) -> Tensor:
        """
        Args:
            patches: [B, S, 4] long
        Returns:
            [B, n_chunks, d_model] -- un embedding por chunk
        """
        B, S, _ = patches.shape

        # Pad a multiplo de chunk_size
        pad = (self.chunk_size - S % self.chunk_size) % self.chunk_size
        if pad > 0:
            patches = F.pad(patches, (0, 0, 0, pad))

        S_padded = patches.shape[1]
        n_chunks = S_padded // self.chunk_size

        # Embedding + primeras N capas
        x = self.model.embedding(patches)
        position_ids = torch.arange(S_padded, device=patches.device).unsqueeze(0).expand(B, -1)
        for block in self.model.blocks[:self.n_encoder_layers]:
            x = block(x, position_ids)

        # Reshape en chunks y mean pool
        x = x.reshape(B, n_chunks, self.chunk_size, -1)
        return x.mean(dim=2)  # [B, n_chunks, d_model]


# =============================================================================
# 11. InContextRetrieval — Retrieval por cosine similarity (solo inferencia)
# =============================================================================

class InContextRetrieval:
    """
    Cache de chunk embeddings con retrieval por cosine similarity.
    Top-k chunks se concatenan al contexto del decoder.

    Sin FAISS. Sin dependencias externas.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.cache_embeddings: Tensor | None = None   # [N, d_model]
        self.cache_patches: list[Tensor] = []          # patches originales

    def add(self, embeddings: Tensor, patches: Tensor, chunk_size: int = 64):
        """
        Agrega chunks al cache.

        Args:
            embeddings: [1, n_chunks, d_model]
            patches: [1, S, 4] -- patches originales correspondientes
        """
        embs = embeddings[0]  # [n_chunks, d_model]
        if self.cache_embeddings is None:
            self.cache_embeddings = embs
        else:
            self.cache_embeddings = torch.cat([self.cache_embeddings, embs], dim=0)

        S = patches.shape[1]
        n_chunks = embs.shape[0]
        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, S)
            self.cache_patches.append(patches[0, start:end])

    def retrieve(self, query: Tensor) -> list[Tensor]:
        """
        Args:
            query: [d_model] -- embedding de consulta
        Returns:
            Lista de hasta top_k chunks (Tensores [chunk_len, 4])
        """
        if self.cache_embeddings is None or len(self.cache_patches) == 0:
            return []

        q = F.normalize(query.unsqueeze(0), dim=-1)
        c = F.normalize(self.cache_embeddings, dim=-1)
        sims = (q @ c.T).squeeze(0)  # [N]

        k = min(self.top_k, len(self.cache_patches))
        _, top_idx = sims.topk(k)
        return [self.cache_patches[i] for i in top_idx.tolist()]

    def clear(self):
        """Limpia el cache."""
        self.cache_embeddings = None
        self.cache_patches = []


# =============================================================================
# Tests
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """Cuenta total de parámetros del modelo."""
    return sum(p.numel() for p in model.parameters())


def test_parameter_count():
    """Verifica que el modelo tiene ~1B parámetros."""
    model = CPU1Model()
    total = count_parameters(model)
    print(f"Parámetros totales: {total:,}")

    # Desglose por componente
    embed_params = count_parameters(model.embedding)
    head_params = sum(count_parameters(h) for h in [model.head_0, model.head_1,
                                                      model.head_2, model.head_3])
    norm_params = count_parameters(model.final_norm)
    block_params = count_parameters(model.blocks)

    print(f"  Embedding:     {embed_params:,}")
    print(f"  Blocks (x{N_LAYERS}):  {block_params:,}")
    print(f"    Por bloque:  {block_params // N_LAYERS:,}")
    print(f"  Final norm:    {norm_params:,}")
    print(f"  Heads (x4):    {head_params:,}")

    # Desglose de un bloque
    block = model.blocks[0]
    attn_p = count_parameters(block.attn)
    ffn_p = count_parameters(block.ffn)
    norms_p = count_parameters(block.attn_norm) + count_parameters(block.ffn_norm)
    print(f"\n  Desglose bloque 0:")
    print(f"    Attention:   {attn_p:,}")
    print(f"    SwiGLU FFN:  {ffn_p:,}")
    print(f"    Norms:       {norms_p:,}")

    # Verificar rango ~1B (entre 900M y 1.2B)
    assert 900_000_000 < total < 1_200_000_000, (
        f"Parámetros fuera de rango: {total:,} (esperado ~1B)"
    )
    print(f"\n[OK] Test de parámetros pasó: {total:,} está en rango [900M, 1.2B]")


def test_forward_pass():
    """Verifica forward pass con input dummy."""
    model = CPU1Model(n_layers=2)  # solo 2 capas para speed
    model.eval()

    B, S = 2, 16  # batch=2, seq=16 patches
    patches = torch.randint(0, 256, (B, S, PATCH_SIZE))

    with torch.no_grad():
        logits = model(patches)

    assert len(logits) == 4, f"Esperados 4 logits, got {len(logits)}"
    for i, l in enumerate(logits):
        assert l.shape == (B, S, VOCAB_SIZE), (
            f"Head {i}: shape {l.shape}, esperado ({B}, {S}, {VOCAB_SIZE})"
        )

    print(f"[OK] Forward pass OK: input {patches.shape} -> 4 x {logits[0].shape}")


def test_components():
    """Verifica cada componente individualmente."""
    B, S, D = 2, 8, D_MODEL

    # RMSNorm
    norm = RMSNorm(D)
    x = torch.randn(B, S, D)
    assert norm(x).shape == (B, S, D), "RMSNorm shape error"
    print("[OK] RMSNorm OK")

    # BitLinear
    bl = BitLinear(D, D)
    assert bl(x).shape == (B, S, D), "BitLinear shape error"
    print("[OK] BitLinear OK")

    # FPResidual
    fpr = FPResidual(D)
    assert fpr(x).shape == (B, S, D), "FPResidual shape error"
    print("[OK] FPResidual OK")

    # RoPE
    rope = RotaryPositionalEmbedding(HEAD_DIM)
    q = torch.randn(B, N_HEADS, S, HEAD_DIM)
    pos = torch.arange(S).unsqueeze(0).expand(B, -1)
    assert rope(q, pos).shape == (B, N_HEADS, S, HEAD_DIM), "RoPE shape error"
    print("[OK] RoPE OK")

    # BytePatchEmbedding
    emb = BytePatchEmbedding()
    patches = torch.randint(0, 256, (B, S, PATCH_SIZE))
    assert emb(patches).shape == (B, S, D), "BytePatchEmbedding shape error"
    print("[OK] BytePatchEmbedding OK")

    # SlidingWindowAttention
    attn = SlidingWindowAttention()
    out = attn(x, pos)
    assert out.shape == (B, S, D), "SlidingWindowAttention shape error"
    print("[OK] SlidingWindowAttention OK")

    # SwiGLU
    ffn = SwiGLU()
    assert ffn(x).shape == (B, S, D), "SwiGLU shape error"
    print("[OK] SwiGLU OK")

    # CPU1Block
    block = CPU1Block()
    assert block(x, pos).shape == (B, S, D), "CPU1Block shape error"
    print("[OK] CPU1Block OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Test 1: Componentes individuales")
    print("=" * 60)
    test_components()

    print("\n" + "=" * 60)
    print("Test 2: Forward pass completo (2 capas)")
    print("=" * 60)
    test_forward_pass()

    print("\n" + "=" * 60)
    print("Test 3: Conteo de parámetros (modelo completo)")
    print("=" * 60)
    test_parameter_count()
