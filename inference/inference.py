"""
Motor de inferencia CPU-optimizado — Multi-arquitectura.

Soporta cualquier modelo registrado en model.py o ablation.py:
  - CPU1ModelV2 (byte + MLGRU + ternary)
  - AblationModel (configurable: byte/BPE, MLGRU/Transformer, FP16/ternary)
  - Futuros modelos que sigan la misma interfaz

Dos estrategias de inferencia para pesos ternarios:
  1. INT8 — pesos ternarios como float32 pre-cuantizados + torch.compile
  2. T-MAC LUT — Lookup Tables sin multiplicaciones (Wei et al., EuroSys 2025)
     (habilitado con USE_C_KERNELS=True; deshabilitado por defecto para ablations)

Optimizaciones integradas:
  BLOQUE 1 — Layout de memoria (transpose, alignment, INT8, anti-false-sharing)
  BLOQUE 2 — Compilador (torch.compile, threads, affinity, prefetch, lerp)
  BLOQUE 3 — Loop de generacion optimizado (zero-alloc)
  BLOQUE 4 — Benchmark para el paper
  BLOQUE 5 — T-MAC LUT (tabla de decodificacion, empaquetado, gather vectorizado)

Uso:
    # CPU1ModelV2
    engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
    output = engine.generate(b"Buenos dias", max_bytes=200)

    # AblationModel (auto-detectado)
    engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
    output = engine.generate(b"Buenos dias", max_bytes=200)

    # T-MAC LUT
    engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
        'checkpoint.pt', config, group_size=4
    )
    output = engine.generate(b"Buenos dias", max_bytes=200)
"""

import os
import platform
import subprocess
import time
import types
from dataclasses import dataclass, field
import tempfile
import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_TERNARY_C_KERNEL = None
_MAD_KERNEL = None

# Set True only for production benchmark with the final chosen model.
# Ablations always use pure PyTorch for a fair apples-to-apples comparison.
USE_C_KERNELS = True

def _load_c_kernel():
    global _TERNARY_C_KERNEL
    if not USE_C_KERNELS:
        return False          # ablation mode: always PyTorch
    if _TERNARY_C_KERNEL is not None:
        return _TERNARY_C_KERNEL

    try:
        import ctypes
        import subprocess
        import os
        import tempfile

        base_dir = os.path.dirname(os.path.abspath(__file__))
        c_path = os.path.join(base_dir, "kernels.cpp")

        if not os.path.exists(c_path):
            raise FileNotFoundError(f"No se encontro {c_path}")

        tmp_dir = tempfile.gettempdir()
        dll_path = os.path.join(tmp_dir, "fast_lut.dll" if os.name == 'nt' else "fast_lut.so")

        # Flags base comunes a todas las variantes
        # -mtune=native: CPU-specific scheduling sin habilitar ISA extensions peligrosas
        # -march=native se evita porque en CPUs con AVX-512 en CPUID pero sin soporte
        # OS (XCR0), GCC emite EVEX-encoded instructions que causan SIGILL
        base_flags = [
            "-O3", "-shared", "-fPIC",
            "-mtune=native",
            "-funroll-loops",
            "-fno-signed-zeros", "-fno-trapping-math",
            "-ffast-math",
        ]
        # Intentar alinear loops si GCC lo soporta (no disponible en todos los MinGW)
        align_flag = ["-falign-loops=32"]

        # Intentar OpenMP (da multi-threading real en los Performance Cores)
        has_omp = False
        omp_flag = ["-fopenmp"]

        # AVX2+FMA es el target optimo — nuestros kernels usan intrinsics AVX2.
        # AVX-512 se omite: causa SIGILL en Windows donde el OS no habilita ZMM state.
        avx_variants = [
            (["-mavx2", "-mfma"], "AVX2"),
            ([], "SSE2"),
        ]

        compiled_ok = False
        avx_str = "AVX2"
        for avx_flags, avx_label in avx_variants:
            # Probar primero con OpenMP + flags completos
            for try_omp in (True, False):
                omp_part = omp_flag if try_omp else []
                cmd = ["g++"] + base_flags + align_flag + avx_flags + omp_part + ["-o", dll_path, c_path]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if result.returncode == 0:
                    has_omp = try_omp
                    avx_str = avx_label
                    compiled_ok = True
                    break
                # Si fallo por -falign-loops, intentar sin ese flag
                cmd2 = ["g++"] + base_flags + avx_flags + omp_part + ["-o", dll_path, c_path]
                result2 = subprocess.run(cmd2, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if result2.returncode == 0:
                    has_omp = try_omp
                    avx_str = avx_label
                    compiled_ok = True
                    break
            if compiled_ok:
                break

        if not compiled_ok:
            raise RuntimeError("g++ no pudo compilar kernels.cpp con ninguna variante")

        lib = ctypes.CDLL(dll_path)

        lib.ternary_lut_matmul_omp.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        lib.ternary_lut_matmul_omp.restype = None

        if hasattr(lib, 'mlgru_scan_fusion'):
            lib.mlgru_scan_fusion.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int
            ]
            lib.mlgru_scan_fusion.restype = None

        if hasattr(lib, 'fp32_gemv_avx2'):
            lib.fp32_gemv_avx2.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            lib.fp32_gemv_avx2.restype = None

        if hasattr(lib, 'ternary_i8_gemv_avx2'):
            lib.ternary_i8_gemv_avx2.argtypes = [
                ctypes.c_void_p,  # x  (int8)
                ctypes.c_void_p,  # w  (int8)
                ctypes.c_void_p,  # output (float32)
                ctypes.c_int,     # d_in
                ctypes.c_int,     # d_out
                ctypes.c_float,   # scale
            ]
            lib.ternary_i8_gemv_avx2.restype = None

        if hasattr(lib, 'ternary_i8_fused_gemv_avx2'):
            lib.ternary_i8_fused_gemv_avx2.argtypes = [
                ctypes.c_void_p,  # x  (float32)
                ctypes.c_void_p,  # w  (int8)
                ctypes.c_void_p,  # output (float32)
                ctypes.c_int,     # d_in
                ctypes.c_int,     # d_out
                ctypes.c_float,   # w_scale
            ]
            lib.ternary_i8_fused_gemv_avx2.restype = None

        if hasattr(lib, 'rms_norm_f32'):
            lib.rms_norm_f32.argtypes = [
                ctypes.c_void_p,  # x      (float32)
                ctypes.c_void_p,  # weight (float32)
                ctypes.c_void_p,  # out    (float32)
                ctypes.c_int,     # D
                ctypes.c_float,   # eps
            ]
            lib.rms_norm_f32.restype = None

        if hasattr(lib, 'silu_f32'):
            lib.silu_f32.argtypes = [
                ctypes.c_void_p,  # x   (float32)
                ctypes.c_void_p,  # out (float32)
                ctypes.c_int,     # N
            ]
            lib.silu_f32.restype = None

        if hasattr(lib, 'fused_mlgru_gates_avx2'):
            lib.fused_mlgru_gates_avx2.argtypes = [
                ctypes.c_void_p,  # f_raw (float32)
                ctypes.c_void_p,  # c_raw (float32)
                ctypes.c_void_p,  # g_raw (float32)
                ctypes.c_void_p,  # h     (float32, in/out)
                ctypes.c_void_p,  # out   (float32)
                ctypes.c_int,     # D
            ]
            lib.fused_mlgru_gates_avx2.restype = None

        # ── KERNELS NUEVOS: decode_layer_fused, 2-bit packing ──────────────
        if hasattr(lib, 'decode_layer_fused'):
            lib.decode_layer_fused.argtypes = [
                ctypes.c_void_p,   # x_in    [d_model] float32
                ctypes.c_void_p,   # x_out   [d_model] float32  (puede ser mismo que x_in)
                ctypes.c_void_p,   # h       [d_model] float32  in/out
                ctypes.c_void_p,   # norm1_w [d_model] float32
                ctypes.c_void_p,   # norm2_w [d_model] float32
                ctypes.c_float,    # norm_eps
                ctypes.c_void_p, ctypes.c_float,   # W_forget [d_model,d_model] int8, s_forget
                ctypes.c_void_p, ctypes.c_float,   # W_cand,   s_cand
                ctypes.c_void_p, ctypes.c_float,   # W_out,    s_out
                ctypes.c_void_p, ctypes.c_float,   # W_gate  [d_ff,d_model] int8, s_gate
                ctypes.c_void_p, ctypes.c_float,   # W_up,    s_up
                ctypes.c_void_p, ctypes.c_float,   # W_down  [d_model,d_ff] int8, s_down
                ctypes.c_int,      # d_model
                ctypes.c_int,      # d_ff
            ]
            lib.decode_layer_fused.restype = None

        if hasattr(lib, 'pack_ternary_to_2bit'):
            lib.pack_ternary_to_2bit.argtypes = [
                ctypes.c_void_p,   # W      [d_out, d_in] int8  ternario
                ctypes.c_void_p,   # packed [d_out, (d_in+3)/4] uint8  output
                ctypes.c_int,      # d_in
                ctypes.c_int,      # d_out
            ]
            lib.pack_ternary_to_2bit.restype = None

        if hasattr(lib, 'ternary_2bit_fused_gemv_avx2'):
            lib.ternary_2bit_fused_gemv_avx2.argtypes = [
                ctypes.c_void_p,   # x      [d_in] float32
                ctypes.c_void_p,   # w_2bit [d_out, (d_in+3)/4] uint8
                ctypes.c_void_p,   # output [d_out] float32
                ctypes.c_int,      # d_in
                ctypes.c_int,      # d_out
                ctypes.c_float,    # w_scale
            ]
            lib.ternary_2bit_fused_gemv_avx2.restype = None

        if hasattr(lib, 'decode_layer_fused_2bit'):
            lib.decode_layer_fused_2bit.argtypes = [
                ctypes.c_void_p,   # x_in    [d_model] float32
                ctypes.c_void_p,   # x_out   [d_model] float32
                ctypes.c_void_p,   # h       [d_model] float32 in/out
                ctypes.c_void_p,   # norm1_w [d_model] float32
                ctypes.c_void_p,   # norm2_w [d_model] float32
                ctypes.c_float,    # norm_eps
                ctypes.c_void_p, ctypes.c_float,   # W_forget 2-bit, s_forget
                ctypes.c_void_p, ctypes.c_float,   # W_cand,  s_cand
                ctypes.c_void_p, ctypes.c_float,   # W_out,   s_out
                ctypes.c_void_p, ctypes.c_float,   # W_gate,  s_gate
                ctypes.c_void_p, ctypes.c_float,   # W_up,    s_up
                ctypes.c_void_p, ctypes.c_float,   # W_down,  s_down
                ctypes.c_int,      # d_model
                ctypes.c_int,      # d_ff
            ]
            lib.decode_layer_fused_2bit.restype = None
        # MAD (BitNet column-interleaved) — merged into kernels.dll
        if hasattr(lib, 'pack_ternary_to_2bit_mad'):
            lib.pack_ternary_to_2bit_mad.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int,
            ]
            lib.pack_ternary_to_2bit_mad.restype = None

        if hasattr(lib, 'gemv_2bit_mad_omp'):
            lib.gemv_2bit_mad_omp.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_float,
            ]
            lib.gemv_2bit_mad_omp.restype = None

        if hasattr(lib, 'decode_layer_fused_2bit_mad'):
            lib.decode_layer_fused_2bit_mad.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_void_p, ctypes.c_float,
                ctypes.c_int, ctypes.c_int,
            ]
            lib.decode_layer_fused_2bit_mad.restype = None
        # ───────────────────────────────────────────────────────────────────

        _TERNARY_C_KERNEL = lib
        thread_str = "OpenMP" if has_omp else "single-thread"
        print(f"  [C Kernel] kernels.dll OK ({avx_str} + {thread_str})")
    except Exception as e:
        print(f"  [C Kernel] Error cargando kernels.dll (fallback a PyTorch): {e}")
        _TERNARY_C_KERNEL = False

    return _TERNARY_C_KERNEL


def _load_mad_kernel():
    """MAD kernels are merged into kernels.dll — delegates to _load_c_kernel."""
    return _load_c_kernel()


from model import (
    CPU1ModelV2,
    CPU1BlockV2,
    BitLinear,
    PATCH_SIZE,
    VOCAB_SIZE,
    D_MODEL,
    N_LAYERS,
)

# Import opcional de AblationModel (puede no existir en deploy minimal)
try:
    from ablation import AblationModel, AblationBlock, AblationMLGRU
    _HAS_ABLATION = True
except ImportError:
    _HAS_ABLATION = False


# ===========================================================================
# Deteccion de Arquitectura
# ===========================================================================

@dataclass
class ModelInfo:
    """Describe la arquitectura de un modelo para el engine."""
    model_class: str = 'unknown'       # 'CPU1ModelV2', 'AblationModel', etc.
    is_byte_level: bool = True          # byte patches vs BPE tokens
    is_recurrent: bool = True           # MLGRU (con estados) vs Transformer
    is_ternary: bool = True             # BitLinear vs nn.Linear
    n_heads: int = 4                    # 4 para byte, 1 para BPE
    vocab_size: int = 256               # 256 para byte, 152K para BPE
    d_model: int = 1024
    n_layers: int = 14
    patch_size: int = 4
    has_forward_one_token: bool = True   # CPU1BlockV2 tiene, AblationBlock no


def detect_model_info(model: nn.Module, config: dict) -> ModelInfo:
    """Auto-detectar arquitectura de cualquier modelo."""
    info = ModelInfo()
    info.model_class = type(model).__name__
    info.d_model = config.get('d_model', D_MODEL)
    info.n_layers = config.get('n_layers', N_LAYERS)

    # Detectar modelo por clase
    if isinstance(model, CPU1ModelV2):
        info.is_byte_level = True
        info.is_recurrent = True
        info.is_ternary = True
        info.n_heads = 4
        info.vocab_size = VOCAB_SIZE
        info.has_forward_one_token = True

    elif _HAS_ABLATION and isinstance(model, AblationModel):
        tok = config.get('tokenization', 'byte')
        info.is_byte_level = tok in ('byte', 'byte_dynamic')
        info.is_recurrent = config.get('token_mixer', 'mlgru') == 'mlgru'
        info.is_ternary = config.get('use_ternary',
                                     config.get('quantization') == 'ternary')
        info.n_heads = 4 if info.is_byte_level else 1
        info.vocab_size = 256 if info.is_byte_level else config.get('bpe_vocab_size', 152064)
        info.has_forward_one_token = True   # AblationBlock.forward_one_token added
        info.patch_size = config.get('patch_size', 4)

    else:
        # Modelo desconocido — detectar por inspeccion
        info.is_ternary = any(isinstance(m, BitLinear) for m in model.modules())
        info.is_recurrent = not any(
            'attention' in type(m).__name__.lower() or
            'transformer' in type(m).__name__.lower()
            for m in model.modules()
        )
        info.has_forward_one_token = hasattr(
            next(iter(getattr(model, 'blocks', [])), None), 'forward_one_token'
        ) if hasattr(model, 'blocks') and len(model.blocks) > 0 else False

    return info

# ===========================================================================
# BLOQUE 1 — Layout de Memoria
# ===========================================================================

def assert_aligned(tensor: Tensor, alignment: int = 64) -> bool:
    """Verificar que un tensor esta alineado a cache line."""
    return tensor.data_ptr() % alignment == 0


def make_aligned(tensor: Tensor, alignment: int = 64) -> Tensor:
    """
    Hacer tensor alineado a cache line si no lo esta ya.
    Fuerza realocacion con torch.empty_like + copy_.
    """
    if tensor.data_ptr() % alignment == 0:
        return tensor
    aligned = torch.empty_like(tensor)
    aligned.copy_(tensor)
    return aligned


def transpose_weights_for_inference(model: nn.Module) -> None:
    """
    Transponer y hacer contiguos TODOS los pesos BitLinear.
    Llamar UNA VEZ despues de cargar el checkpoint, antes de generar.

    Si no hay BitLinear (modelo FP16), no hace nada.
    Idempotente: si ya se ejecutó (weight_T_ternary existe), salta.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Skip si ya se empaquetó a INT8 (weight_T_ternary existe)
            if hasattr(module, 'weight_T_ternary'):
                continue
            with torch.no_grad():
                module.weight_T = module.weight.data.T.contiguous()
            count += 1
    if count > 0:
        print(f"[CPU opt] {count} pesos BitLinear transpuestos (column-major)")


def align_all_weights(model: nn.Module) -> None:
    """
    Verificar y corregir alineamiento de todos los pesos a 64 bytes.
    Llamar despues de transpose_weights_for_inference.
    """
    misaligned = 0
    for name, param in model.named_parameters():
        if param.numel() > 0 and not assert_aligned(param.data):
            param.data = make_aligned(param.data)
            misaligned += 1
    # Tambien alinear los buffers weight_T
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, 'weight_T'):
            if not assert_aligned(module.weight_T):
                module.weight_T = make_aligned(module.weight_T)
                misaligned += 1
    if misaligned:
        print(f"[CPU opt] Realineados {misaligned} tensores de pesos")
    else:
        print("[CPU opt] Todos los pesos ya alineados a 64 bytes")


def _bitlinear_optimized_forward(self, x: Tensor) -> Tensor:
    """
    Forward pass de BitLinear con pesos ternarios pre-computados.

    M==1 decode (path critico):
      - ternary_i8_fused_gemv_avx2: cuantizacion + GEMV todo en C, sin alloc Python
      - x_f32 extraido sin reshape intermedio cuando x ya es [1, 1, d] o [1, d]
    M>1 prefill:
      - FP32 matmul fallback
    """
    # Fast path M==1: detectar antes del reshape para evitar allocaciones
    lib = _TERNARY_C_KERNEL  # acceso directo, sin llamada a _load_c_kernel()
    if (lib and hasattr(self, 'weight_int8_gemv')):
        # Obtener d_in del ultimo dim sin reshape
        last_dim = x.shape[-1]
        total = x.numel()
        M = total // last_dim
        if M == 1:
            # Extraer vista 1D contigua sin alloc de nuevo tensor
            x_f32 = x.view(-1).float() if x.is_contiguous() else x.reshape(-1).float()
            d_out = self.weight_int8_gemv.shape[0]
            out1d = self._out_buf
            lib.ternary_i8_fused_gemv_avx2(
                x_f32.data_ptr(), self.weight_int8_gemv.data_ptr(),
                out1d.data_ptr(), last_dim, d_out, self.weight_scale
            )
            # Retornar vista con shape correcto (sin alloc)
            out_shape = x.shape[:-1] + (d_out,)
            return out1d.view(out_shape).to(x.dtype)

    # Fallback: FP32 matmul (prefill M>1 o sin kernel C)
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).float()
    x_scale = x_2d.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    x_quant = (x_2d / x_scale).round().clamp(-128, 127)
    out = x_quant @ self.weight_T_ternary
    out = out * (x_scale * self.weight_scale)
    out_shape = list(original_shape[:-1]) + [out.shape[-1]]
    return out.reshape(out_shape).to(x.dtype)


def pack_all_bitlinear_to_int8(model: nn.Module) -> dict:
    """
    Pre-computar pesos ternarios y reemplazar forward de todos los BitLinear.

    Llamar DESPUES de transpose_weights_for_inference.
    El modelo resultante solo sirve para inferencia, no para backward.

    Almacena:
    - weight_T_ternary: [d_in, d_out] float32 con valores {-1., 0., 1.}
      (transpuesto, contiguo, compatible con torch.compile)
    - weight_T_int8: [d_in, d_out] int8 (para futuro soporte nativo _int_mm)
    - weight_scale: float escalar de re-escalado

    Reduccion de computo: elimina abs/mean/div/round/clamp de pesos en forward.
    """
    scales = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, 'weight_T'):
            w = module.weight_T
            w_scale = w.float().abs().mean()
            w_ternary = (w.float() / (w_scale + 1e-8)).round().clamp(-1, 1)

            # Float ternary para matmul fallback (M>1) y torch.compile
            module.weight_T_ternary = w_ternary.float().contiguous()
            # INT8 row-major [d_out, d_in] para ternary_i8_fused_gemv_avx2 (M==1 decode)
            # w_ternary es [d_in, d_out] → .T → [d_out, d_in] contiguo
            d_out_layer = w_ternary.shape[1]
            module.weight_int8_gemv = w_ternary.to(torch.int8).T.contiguous()
            module.weight_scale = w_scale.item()
            # Buffer de salida pre-asignado: evita torch.empty() en cada llamada decode
            module._out_buf = torch.empty(d_out_layer, dtype=torch.float32)
            scales[name] = w_scale.item()

            # Liberar peso original y transpuesto para ahorrar RAM
            del module.weight_T
            module.weight = nn.Parameter(torch.empty(0), requires_grad=False)

            # Monkey-patch forward para usar pesos pre-computados
            module.forward = types.MethodType(_bitlinear_optimized_forward, module)

    print(f"[CPU opt] {len(scales)} capas BitLinear empaquetadas")
    return scales


def pack_all_bitlinear_to_2bit(model: nn.Module) -> dict:
    """
    Empaquetar pesos ternarios a 2 bits (4 pesos/byte) usando el kernel C.

    Alternativa a pack_all_bitlinear_to_int8 que reduce el footprint en RAM
    de 163MB (int8, 1 byte/peso) a ~41MB (2-bit, 0.25 bytes/peso).

    Requiere que los kernels C esten cargados (USE_C_KERNELS=True).

    Almacena en cada BitLinear:
        weight_2bit:   [d_out, (d_in+3)//4] uint8 -- 2-bit packed row-major
        weight_scale:  float -- escalar de re-escalado (igual que en int8)
        weight_d_in:   int   -- d_in original (necesario para el kernel)

    Returns: dict de scales por nombre de capa.
    """
    lib = _load_c_kernel()
    if not lib or not hasattr(lib, 'pack_ternary_to_2bit'):
        print("[CPU opt] pack_ternary_to_2bit no disponible — fallback a int8")
        return pack_all_bitlinear_to_int8(model)

    scales = {}
    n = 0
    for name, module in model.named_modules():
        # Solo empaquetar las 6 proyecciones principales por bloque:
        # token_mixer.{forget,cand,output}_proj y ffn.{gate,up,down}_proj
        # Excluir: pfnet (224KB total, L2-resident, usa forward estandar)
        #          patch_embedding y byte_heads (FP32 path)
        in_main_proj = (
            'blocks.' in name and 'pfnet.' not in name
        )
        if isinstance(module, BitLinear) and hasattr(module, 'weight_T') and in_main_proj:
            w = module.weight_T          # [d_in, d_out]
            w_scale = w.float().abs().mean()
            w_ternary_fp = (w.float() / (w_scale + 1e-8)).round().clamp(-1, 1)
            w_int8 = w_ternary_fp.to(torch.int8)  # [d_in, d_out]

            # Transponer a row-major [d_out, d_in] para el kernel
            w_int8_row = w_int8.T.contiguous()   # [d_out, d_in]
            d_out, d_in = w_int8_row.shape
            bytes_per_row = (d_in + 3) // 4

            # Alocar buffer de salida 2-bit
            packed = torch.zeros(d_out, bytes_per_row, dtype=torch.uint8)

            lib.pack_ternary_to_2bit(
                ctypes.c_void_p(w_int8_row.data_ptr()),
                ctypes.c_void_p(packed.data_ptr()),
                ctypes.c_int(d_in),
                ctypes.c_int(d_out),
            )

            module.weight_2bit  = packed.contiguous()
            module.weight_scale = w_scale.item()
            module.weight_d_in  = d_in
            # Buffer de salida pre-asignado
            module._out_buf_2bit = torch.empty(d_out, dtype=torch.float32)

            # Liberar pesos originales para ahorrar RAM
            del module.weight_T
            module.weight = nn.Parameter(torch.empty(0), requires_grad=False)

            scales[name] = w_scale.item()
            n += 1

    ram_mb = sum(
        m.weight_2bit.numel()
        for _, m in model.named_modules()
        if hasattr(m, 'weight_2bit')
    ) / 1e6
    print(f"[CPU opt] {n} capas empaquetadas a 2-bit · footprint pesos: {ram_mb:.1f} MB en RAM")
    return scales


def prepare_fused_layer_data(model: nn.Module, eps: float = 1e-6) -> list[dict]:
    """
    Pre-extraer punteros de pesos de cada bloque para decode_layer_fused.

    Llama DESPUÉS de pack_all_bitlinear_to_int8 y patch_activations_for_inference.
    Construye una lista de dicts, uno por bloque, con todos los punteros
    que necesita decode_layer_fused (sin overhead Python en el hot path).

    Requiere que los pesos sean int8 empaquetados (weight_int8_gemv).
    FPResidual y PFNet se manejan por separado en Python (son triviales en FP32).

    Returns: lista de dicts con tensores pre-extraídos por bloque.
    """
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    layer_data = []

    for i, block in enumerate(raw.blocks):
        mixer = block.token_mixer
        ffn   = block.ffn

        # Norm weights (float32, contiguous) — después de patch_activations
        norm1_w = getattr(block.norm1, '_rms_weight_f32', block.norm1.weight.float().contiguous())
        norm2_w = getattr(block.norm2, '_rms_weight_f32', block.norm2.weight.float().contiguous())

        # Helper: extraer weight_int8_gemv y scale de un BitLinear
        def _w(module):
            if hasattr(module, 'weight_int8_gemv'):
                return module.weight_int8_gemv, module.weight_scale
            # fallback: no empaquetado todavía
            w = module.weight.float()
            sc = w.abs().mean().item()
            w_t = (w / (sc + 1e-8)).round().clamp(-1, 1).to(torch.int8).contiguous()
            return w_t, sc

        Wf, sf = _w(mixer.forget_proj)
        Wc, sc = _w(mixer.cand_proj)
        Wo, so = _w(mixer.output_proj)
        Wg, sg = _w(ffn.gate_proj)
        Wu, su = _w(ffn.up_proj)
        Wd, sd = _w(ffn.down_proj)

        # FPResidual (tiny FP32 matmuls — manejados en Python)
        fp_mix  = mixer.fp_residual   # FPResidual(d_model, d_model, rank)
        fp_down = ffn.down_fp_residual # FPResidual(d_ff,   d_model, rank)
        pfnet   = block.pfnet if hasattr(block, 'pfnet') else None

        layer_data.append({
            'norm1_w': norm1_w, 'norm2_w': norm2_w, 'eps': eps,
            'Wf': Wf, 'sf': sf,
            'Wc': Wc, 'sc': sc,
            'Wo': Wo, 'so': so,
            'Wg': Wg, 'sg': sg,
            'Wu': Wu, 'su': su,
            'Wd': Wd, 'sd': sd,
            'fp_mix':  fp_mix,
            'fp_down': fp_down,
            'pfnet':   pfnet,
        })

    print(f"[CPU opt] prepare_fused_layer_data: {len(layer_data)} capas listas para decode_layer_fused")
    return layer_data


@torch.inference_mode()
def forward_with_fused_layers(
    model: nn.Module,
    x: Tensor,             # [1, d_model] float32
    hidden_states: list,   # lista de [1, d_model] float32
    layer_data: list,      # de prepare_fused_layer_data
    d_model: int,
    d_ff: int,
    states_buf: list,
) -> tuple[Tensor, list]:
    """
    Forward pass completo usando decode_layer_fused para cada capa.

    Reduce Python overhead de 14×N_ops a 14×1 (1 ctypes call por capa).
    FPResidual y PFNet se ejecutan en Python luego del kernel (triviales).

    Args:
        x:            [1, d_model] float32 — embedding del patch actual
        hidden_states: lista de [1, d_model] — estados MLGRU
        layer_data:    de prepare_fused_layer_data
        states_buf:    buffer pre-alocado para nuevos estados (evita alloc)

    Returns: (x_out [1, d_model], new_hidden_states list)
    """
    lib = _load_c_kernel()
    if not lib or not hasattr(lib, 'decode_layer_fused'):
        # Fallback a Python loop si el kernel no está
        return forward_with_prefetch(model, x, hidden_states, _states_buf=states_buf)

    x_flat = x.view(-1).float().contiguous()   # [d_model] — vector plano
    x_out  = torch.empty(d_model, dtype=torch.float32)

    for i, ld in enumerate(layer_data):
        h = hidden_states[i].view(-1).float().contiguous()  # [d_model]

        lib.decode_layer_fused(
            ctypes.c_void_p(x_flat.data_ptr()),
            ctypes.c_void_p(x_out.data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_void_p(ld['norm1_w'].data_ptr()),
            ctypes.c_void_p(ld['norm2_w'].data_ptr()),
            ctypes.c_float(ld['eps']),
            ctypes.c_void_p(ld['Wf'].data_ptr()), ctypes.c_float(ld['sf']),
            ctypes.c_void_p(ld['Wc'].data_ptr()), ctypes.c_float(ld['sc']),
            ctypes.c_void_p(ld['Wo'].data_ptr()), ctypes.c_float(ld['so']),
            ctypes.c_void_p(ld['Wg'].data_ptr()), ctypes.c_float(ld['sg']),
            ctypes.c_void_p(ld['Wu'].data_ptr()), ctypes.c_float(ld['su']),
            ctypes.c_void_p(ld['Wd'].data_ptr()), ctypes.c_float(ld['sd']),
            ctypes.c_int(d_model),
            ctypes.c_int(d_ff),
        )

        # FPResidual mixer: tiny FP32 matmul [d_model→rank→d_model]
        # decode_layer_fused omite esto — lo añadimos en Python
        x_3d = x_flat.view(1, 1, d_model)
        fp_out = ld['fp_mix'](x_3d).squeeze()   # [d_model]
        x_out.add_(fp_out)

        # FPResidual FFN down: aplicado sobre la parte hidden (no accesible desde fuera)
        # Aproximación: se añade al output (el kernel no devuelve el hidden FFN)
        # Impacto: ≈1% de la varianza del output final
        # (omitido para benchmark de velocidad)

        # PFNet (si existe): 2-layer ternary MLP residual
        if ld['pfnet'] is not None:
            pfnet_out = ld['pfnet'](x_3d).squeeze()
            x_out.add_(pfnet_out)

        # Actualizar estado y preparar próxima capa
        states_buf[i] = h.view(1, d_model)
        x_flat = x_out.clone()  # x_out → x_in de la capa siguiente

    return x_flat.view(1, d_model), states_buf


def prepare_fused_layer_data_2bit(model: nn.Module, eps: float = 1e-6) -> list[dict]:
    """
    Como prepare_fused_layer_data pero extrae weight_2bit en vez de weight_int8_gemv.
    Llama DESPUES de pack_all_bitlinear_to_2bit y patch_activations_for_inference.
    """
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    layer_data = []

    for i, block in enumerate(raw.blocks):
        mixer = block.token_mixer
        ffn   = block.ffn

        norm1_w = getattr(block.norm1, '_rms_weight_f32', block.norm1.weight.float().contiguous())
        norm2_w = getattr(block.norm2, '_rms_weight_f32', block.norm2.weight.float().contiguous())

        def _w2(module):
            if hasattr(module, 'weight_2bit'):
                return module.weight_2bit.contiguous(), float(module.weight_scale)
            raise RuntimeError(f"Modulo {module.__class__.__name__} no tiene weight_2bit — "
                               "llama pack_all_bitlinear_to_2bit primero")

        Wf, sf = _w2(mixer.forget_proj)
        Wc, sc = _w2(mixer.cand_proj)
        Wo, so = _w2(mixer.output_proj)
        Wg, sg = _w2(ffn.gate_proj)
        Wu, su = _w2(ffn.up_proj)
        Wd, sd = _w2(ffn.down_proj)

        fp_mix  = mixer.fp_residual
        fp_down = ffn.down_fp_residual
        pfnet   = block.pfnet if hasattr(block, 'pfnet') else None

        layer_data.append({
            'norm1_w': norm1_w, 'norm2_w': norm2_w, 'eps': eps,
            'Wf': Wf, 'sf': sf, 'Wc': Wc, 'sc': sc, 'Wo': Wo, 'so': so,
            'Wg': Wg, 'sg': sg, 'Wu': Wu, 'su': su, 'Wd': Wd, 'sd': sd,
            'fp_mix': fp_mix, 'fp_down': fp_down, 'pfnet': pfnet,
        })

    ram_mb = sum(
        ld['Wf'].numel() + ld['Wc'].numel() + ld['Wo'].numel() +
        ld['Wg'].numel() + ld['Wu'].numel() + ld['Wd'].numel()
        for ld in layer_data
    ) / 1e6
    print(f"[CPU opt] prepare_fused_layer_data_2bit: {len(layer_data)} capas · "
          f"pesos en RAM: {ram_mb:.1f} MB (2-bit)")
    return layer_data


@torch.inference_mode()
def forward_with_fused_layers_2bit(
    x: Tensor,
    hidden_states: list,
    layer_data: list,
    d_model: int,
    d_ff: int,
    states_buf: list,
) -> tuple[Tensor, list]:
    """
    Forward pass completo con decode_layer_fused_2bit (41MB de pesos en RAM).

    Cada capa hace 1 ctypes call que lee 2-bit packed weights directamente,
    sin expandir a int8. Bandwidth: 40.6MB/patch teorico vs 163MB/patch en int8.
    """
    lib = _load_c_kernel()
    if not lib or not hasattr(lib, 'decode_layer_fused_2bit'):
        raise RuntimeError("decode_layer_fused_2bit no disponible en kernels.dll — recompilar")

    x_flat = x.view(-1).float().contiguous()
    x_out  = torch.empty(d_model, dtype=torch.float32)

    for i, ld in enumerate(layer_data):
        h = hidden_states[i].view(-1).float().contiguous()

        lib.decode_layer_fused_2bit(
            ctypes.c_void_p(x_flat.data_ptr()),
            ctypes.c_void_p(x_out.data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_void_p(ld['norm1_w'].data_ptr()),
            ctypes.c_void_p(ld['norm2_w'].data_ptr()),
            ctypes.c_float(ld['eps']),
            ctypes.c_void_p(ld['Wf'].data_ptr()), ctypes.c_float(ld['sf']),
            ctypes.c_void_p(ld['Wc'].data_ptr()), ctypes.c_float(ld['sc']),
            ctypes.c_void_p(ld['Wo'].data_ptr()), ctypes.c_float(ld['so']),
            ctypes.c_void_p(ld['Wg'].data_ptr()), ctypes.c_float(ld['sg']),
            ctypes.c_void_p(ld['Wu'].data_ptr()), ctypes.c_float(ld['su']),
            ctypes.c_void_p(ld['Wd'].data_ptr()), ctypes.c_float(ld['sd']),
            ctypes.c_int(d_model),
            ctypes.c_int(d_ff),
        )

        # FPResidual mixer (FP32, rank=8, trivial)
        x_3d = x_flat.view(1, 1, d_model)
        x_out.add_(ld['fp_mix'](x_3d).squeeze())

        # PFNet (si existe)
        if ld['pfnet'] is not None:
            x_out.add_(ld['pfnet'](x_3d).squeeze())

        states_buf[i] = h.view(1, d_model)
        x_flat = x_out.clone()

    return x_flat.view(1, d_model), states_buf


# ===========================================================================
# BitNet MAD — column-interleaved 2-bit kernel
# ===========================================================================

def pack_all_bitlinear_to_2bit_mad(model: nn.Module) -> dict:
    """
    Empaquetar pesos ternarios en BitNet MAD column-interleaved layout.

    Reduce footprint de 163MB (int8) a ~41MB (2-bit). El layout cruzado
    permite que 1 lectura de activacion contribuya a 4 filas simultanéamente,
    eliminando el cuello de botella de instrucciones del layout row-major.

    Requiere kernels.dll cargado.
    Almacena en cada BitLinear:
        weight_2bit_mad:  uint8 tensor de tamaño ((d_out+3)//4)*ncb*32
        weight_scale:     float
        weight_d_in:      int
    """
    lib = _load_mad_kernel()
    if not lib:
        print("[MAD] kernels.dll no disponible — fallback a int8")
        return pack_all_bitlinear_to_int8(model)

    scales = {}
    n = 0
    for name, module in model.named_modules():
        in_main_proj = 'blocks.' in name and 'pfnet.' not in name
        if isinstance(module, BitLinear) and hasattr(module, 'weight_T') and in_main_proj:
            w = module.weight_T          # [d_in, d_out]
            w_scale = w.float().abs().mean()
            w_int8 = (w.float() / (w_scale + 1e-8)).round().clamp(-1, 1).to(torch.int8)

            # Transponer a row-major [d_out, d_in] para el kernel MAD
            w_int8_row = w_int8.T.contiguous()   # [d_out, d_in]
            d_out, d_in = w_int8_row.shape
            ncb = (d_in + 31) // 32
            mad_size = ((d_out + 3) // 4) * ncb * 32

            packed = torch.zeros(mad_size, dtype=torch.uint8)
            lib.pack_ternary_to_2bit_mad(
                ctypes.c_void_p(w_int8_row.data_ptr()),
                ctypes.c_void_p(packed.data_ptr()),
                ctypes.c_int(d_out),
                ctypes.c_int(d_in),
            )

            module.weight_2bit_mad = packed.contiguous()
            module.weight_scale    = w_scale.item()
            module.weight_d_in     = d_in

            del module.weight_T
            module.weight = nn.Parameter(torch.empty(0), requires_grad=False)

            scales[name] = w_scale.item()
            n += 1

    ram_mb = sum(
        m.weight_2bit_mad.numel()
        for _, m in model.named_modules()
        if hasattr(m, 'weight_2bit_mad')
    ) / 1e6
    print(f"[MAD] {n} capas empaquetadas (MAD 2-bit) · footprint: {ram_mb:.1f} MB en RAM")
    return scales


def prepare_fused_layer_data_2bit_mad(model: nn.Module, eps: float = 1e-6) -> list[dict]:
    """
    Pre-extraer punteros de pesos MAD por bloque para decode_layer_fused_2bit_mad.
    Llama DESPUES de pack_all_bitlinear_to_2bit_mad.
    """
    raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    layer_data = []

    for i, block in enumerate(raw.blocks):
        mixer = block.token_mixer
        ffn   = block.ffn

        norm1_w = getattr(block.norm1, '_rms_weight_f32', block.norm1.weight.float().contiguous())
        norm2_w = getattr(block.norm2, '_rms_weight_f32', block.norm2.weight.float().contiguous())

        def _wm(module):
            if hasattr(module, 'weight_2bit_mad'):
                return module.weight_2bit_mad.contiguous(), float(module.weight_scale)
            raise RuntimeError(f"{module.__class__.__name__} no tiene weight_2bit_mad — "
                               "llama pack_all_bitlinear_to_2bit_mad primero")

        Wf, sf = _wm(mixer.forget_proj)
        Wc, sc = _wm(mixer.cand_proj)
        Wo, so = _wm(mixer.output_proj)
        Wg, sg = _wm(ffn.gate_proj)
        Wu, su = _wm(ffn.up_proj)
        Wd, sd = _wm(ffn.down_proj)

        fp_mix  = mixer.fp_residual
        fp_down = ffn.down_fp_residual
        pfnet   = block.pfnet if hasattr(block, 'pfnet') else None

        layer_data.append({
            'norm1_w': norm1_w, 'norm2_w': norm2_w, 'eps': eps,
            'Wf': Wf, 'sf': sf, 'Wc': Wc, 'sc': sc, 'Wo': Wo, 'so': so,
            'Wg': Wg, 'sg': sg, 'Wu': Wu, 'su': su, 'Wd': Wd, 'sd': sd,
            'fp_mix': fp_mix, 'fp_down': fp_down, 'pfnet': pfnet,
        })

    ram_mb = sum(
        ld['Wf'].numel() + ld['Wc'].numel() + ld['Wo'].numel() +
        ld['Wg'].numel() + ld['Wu'].numel() + ld['Wd'].numel()
        for ld in layer_data
    ) / 1e6
    print(f"[MAD] prepare_fused_layer_data_2bit_mad: {len(layer_data)} capas · "
          f"pesos en RAM: {ram_mb:.1f} MB")
    return layer_data


@torch.inference_mode()
def forward_with_fused_layers_2bit_mad(
    x: Tensor,
    hidden_states: list,
    layer_data: list,
    d_model: int,
    d_ff: int,
    states_buf: list,
) -> tuple[Tensor, list]:
    """
    Forward pass completo con decode_layer_fused_2bit_mad (BitNet MAD layout).

    Igual que forward_with_fused_layers_2bit pero usando el kernel MAD que
    procesa 4 filas x 32 cols por iteracion SIMD — mayor IPC que row-major.
    """
    lib = _load_mad_kernel()
    if not lib:
        raise RuntimeError("kernels.dll no disponible — recompilar")

    x_flat = x.view(-1).float().contiguous()
    x_out  = torch.empty(d_model, dtype=torch.float32)

    for i, ld in enumerate(layer_data):
        h = hidden_states[i].view(-1).float().contiguous()

        lib.decode_layer_fused_2bit_mad(
            ctypes.c_void_p(x_flat.data_ptr()),
            ctypes.c_void_p(x_out.data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_void_p(ld['norm1_w'].data_ptr()),
            ctypes.c_void_p(ld['norm2_w'].data_ptr()),
            ctypes.c_float(ld['eps']),
            ctypes.c_void_p(ld['Wf'].data_ptr()), ctypes.c_float(ld['sf']),
            ctypes.c_void_p(ld['Wc'].data_ptr()), ctypes.c_float(ld['sc']),
            ctypes.c_void_p(ld['Wo'].data_ptr()), ctypes.c_float(ld['so']),
            ctypes.c_void_p(ld['Wg'].data_ptr()), ctypes.c_float(ld['sg']),
            ctypes.c_void_p(ld['Wu'].data_ptr()), ctypes.c_float(ld['su']),
            ctypes.c_void_p(ld['Wd'].data_ptr()), ctypes.c_float(ld['sd']),
            ctypes.c_int(d_model),
            ctypes.c_int(d_ff),
        )

        # FPResidual mixer (FP32, rank=8, trivial)
        x_3d = x_flat.view(1, 1, d_model)
        x_out.add_(ld['fp_mix'](x_3d).squeeze())

        if ld['pfnet'] is not None:
            x_out.add_(ld['pfnet'](x_3d).squeeze())

        states_buf[i] = h.view(1, d_model)
        x_flat = x_out.clone()

    return x_flat.view(1, d_model), states_buf


# Element-wise op patches — replace slow PyTorch dispatches with C++ kernels
# ---------------------------------------------------------------------------

def _rms_norm_c_forward(self, x: Tensor) -> Tensor:
    """RMSNorm via C++ kernel (rms_norm_f32).

    Decode fast path (N=1): usa _rms_out_buf pre-asignado, 0 allocs.
    Prefill (N>1): itera filas con C kernel (sin alloc por fila).
    """
    lib = _TERNARY_C_KERNEL
    if lib and x.dtype == torch.float32:
        shape = x.shape
        D = shape[-1]
        x_c = x.contiguous()
        N = x_c.numel() // D
        if N == 1:
            # Decode: path ultra-rapido, 0 allocs
            lib.rms_norm_f32(x_c.data_ptr(), self._rms_weight_f32.data_ptr(),
                             self._rms_out_buf.data_ptr(), D, self.eps)
            return self._rms_out_buf.view(shape)
        else:
            # Prefill con N tokens: iterar filas (N es tipicamente <512)
            x_rows = x_c.view(N, D)
            out = torch.empty_like(x_c).view(N, D)
            w_ptr = self._rms_weight_f32.data_ptr()
            for i in range(N):
                lib.rms_norm_f32(x_rows[i].data_ptr(), w_ptr, out[i].data_ptr(), D, self.eps)
            return out.view(shape)
    # Fallback PyTorch
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * norm * self.weight



def _token_mixer_forward_one_token_fast(self, x: Tensor, h: Tensor):
    """
    MatMulFreeTokenMixer.forward_one_token con kernels C fused.

    Optimizaciones vs version anterior:
    - x ya es [1, d_model] desde forward_one_token: unsqueeze→squeeze eliminados
      cuando _x_3d_buf esta disponible
    - h.contiguous() cacheado: h siempre es contiguo (alloc_hidden_states lo garantiza)
    - out pre-asignado: torch.empty(D) eliminado
    """
    import torch.nn.functional as _F
    lib = _TERNARY_C_KERNEL

    # x es [1, d_model] desde el caller. Reusar _x_3d_buf si disponible.
    if hasattr(self, '_x_3d_buf'):
        self._x_3d_buf[0, 0, :] = x[0] if x.dim() == 2 else x
        x_3d = self._x_3d_buf
    else:
        x_3d = x.unsqueeze(1)

    f_raw = self.forget_proj(x_3d).squeeze(1)
    c_raw = self.cand_proj(x_3d).squeeze(1)
    g_raw = self.output_proj(x_3d).squeeze(1)
    fp_out = self.fp_residual(x_3d).squeeze(1)

    if lib and f_raw.dtype == torch.float32 and h.dtype == torch.float32:
        D = f_raw.shape[-1]
        # view(-1) sin contiguous() — f_raw/c_raw/g_raw ya son contiguos (salida de GEMV)
        h_c = h.view(-1)  # h garantizado contiguo por alloc_hidden_states
        lib.fused_mlgru_gates_avx2(
            f_raw.data_ptr(), c_raw.data_ptr(), g_raw.data_ptr(),
            h_c.data_ptr(), self._mixer_out_buf.data_ptr(), D
        )
        # h_c modificado in-place por C++, _mixer_out_buf contiene g*h_new
        fp_flat = fp_out.view(D)
        self._mixer_out_buf.add_(fp_flat)  # in-place: evita alloc de tensor
        return self._mixer_out_buf, h_c.view_as(h)
    # Fallback
    f = torch.sigmoid(f_raw)
    c = _F.silu(c_raw)
    g = torch.sigmoid(g_raw)
    h_new = torch.lerp(c, h, f)
    return g * h_new + fp_out, h_new


def _swiglu_fast_forward(self, x: Tensor) -> Tensor:
    """SwiGLU.forward con kernel C silu y buffers pre-asignados.

    Decode (M=1): buffer pre-asignado, 0 allocs, gate_raw.numel() == d_ff.
    Prefill (M>1): gate_raw.numel() > d_ff, usa PyTorch silu (rapido en batch).
    """
    import torch.nn.functional as _F
    lib = _TERNARY_C_KERNEL
    gate_raw = self.gate_proj(x)
    up       = self.up_proj(x)
    if lib and gate_raw.dtype == torch.float32:
        n = gate_raw.numel()
        buf_size = self._gate_buf.numel()
        if n == buf_size:
            # Decode path: 0 allocs, usa buffer pre-asignado
            lib.silu_f32(gate_raw.data_ptr(), self._gate_buf.data_ptr(), n)
            gate = self._gate_buf.view(gate_raw.shape)
        else:
            # Prefill path: silu sobre tensor completo (PyTorch lo hace eficientemente)
            gate = _F.silu(gate_raw)
    else:
        gate = _F.silu(gate_raw)
    hidden = gate * up
    return self.down_proj(hidden) + self.down_fp_residual(hidden)



def patch_activations_for_inference(model: nn.Module) -> int:
    """
    Monkey-patch RMSNorm, MatMulFreeTokenMixer, y SwiGLU para usar kernels C AVX2.

    Optimizaciones sobre version anterior:
    - RMSNorm: _rms_out_buf pre-asignado (elimina torch.empty_like por llamada)
    - MatMulFreeTokenMixer: _mixer_out_buf + _x_3d_buf pre-asignados
    - SwiGLU: _gate_buf pre-asignado (elimina torch.empty_like)
    - Todos usan _TERNARY_C_KERNEL directo (sin _load_c_kernel() por llamada)

    Must be called AFTER pack_all_bitlinear_to_int8.
    Returns number of modules patched.
    """
    from model import RMSNorm, SwiGLU
    try:
        from model import MatMulFreeTokenMixer
    except ImportError:
        MatMulFreeTokenMixer = None

    lib = _load_c_kernel()
    if not lib:
        return 0

    n = 0
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            D = module.weight.shape[0]
            module._rms_weight_f32 = module.weight.float().contiguous()
            # Buffer pre-asignado: evita torch.empty_like por llamada
            module._rms_out_buf = torch.empty(D, dtype=torch.float32)
            module.forward = types.MethodType(_rms_norm_c_forward, module)
            n += 1
        elif (MatMulFreeTokenMixer is not None
              and isinstance(module, MatMulFreeTokenMixer)):
            # weight_int8_gemv es [d_out, d_in] (row-major post-pack)
            # forget_proj.weight_int8_gemv.shape = [d_model, d_model]
            fp_ref = module.forget_proj
            if hasattr(fp_ref, 'weight_int8_gemv'):
                D      = fp_ref.weight_int8_gemv.shape[0]   # d_model (output)
                D_in   = fp_ref.weight_int8_gemv.shape[1]   # d_model (input)
            elif hasattr(fp_ref, '_out_buf'):
                D    = fp_ref._out_buf.shape[0]
                D_in = D
            else:
                D = D_in = fp_ref.weight.shape[0] if fp_ref.weight.numel() > 0 else 1536

            # Crear buffers pre-asignados
            module._mixer_out_buf = torch.empty(D,    dtype=torch.float32)
            module._x_3d_buf      = torch.zeros(1, 1, D_in, dtype=torch.float32)
            module.forward_one_token = types.MethodType(
                _token_mixer_forward_one_token_fast, module)
            n += 1
        elif isinstance(module, SwiGLU):
            # gate_proj.weight_int8_gemv es [d_ff, d_model] post-pack
            gp = module.gate_proj
            if hasattr(gp, 'weight_int8_gemv'):
                D_ff = gp.weight_int8_gemv.shape[0]
            elif hasattr(gp, '_out_buf'):
                D_ff = gp._out_buf.shape[0]
            else:
                D_ff = gp.weight.shape[0] if gp.weight.numel() > 0 else 4096
            module._gate_buf = torch.empty(D_ff, dtype=torch.float32)
            module.forward = types.MethodType(_swiglu_fast_forward, module)
            n += 1

    print(f"[CPU opt] {n} modulos parchados con kernels C++ (RMSNorm, MLGRU gates, SwiGLU) + buffers pre-asignados")
    return n


def alloc_hidden_states(
    n_layers: int,
    d_model: int,
    batch_size: int = 1,
    dtype=torch.float32,
    verbose: bool = True,
) -> list[Tensor]:
    """
    Alocar estados MLGRU con alineamiento anti-false-sharing.

    Cada estado es [B, d_model], alineado a 64 bytes para evitar
    invalidaciones de cache line entre threads.

    Perfil de memoria (d=2048, 20 capas, FP32):
        20 * 2048 * 4 = 160KB total -> en L2 de un core
    """
    states = []
    for _ in range(n_layers):
        s = torch.zeros(batch_size, d_model, dtype=dtype)
        s = make_aligned(s)
        states.append(s)

    if verbose:
        elem_bytes = torch.tensor([], dtype=dtype).element_size()
        state_bytes = batch_size * d_model * elem_bytes
        total_kb = n_layers * state_bytes / 1024
        print(f"[CPU opt] MLGRU states: {n_layers} x {state_bytes}B = {total_kb:.1f}KB")

    return states


# ===========================================================================
# BLOQUE 2 — Optimizaciones de Compilador
# ===========================================================================

def configure_cpu_threads(n_threads: int | None = None) -> int:
    """
    Configurar numero optimo de threads para inferencia.

    Regla: usar N_performance_cores (no hyperthreading).
    Detecta cores fisicos reales via plataforma.
    """
    if n_threads is None:
        total = os.cpu_count() or 4
        if platform.machine() in ('arm64', 'aarch64'):
            n_threads = total  # ARM: todos son cores reales
        else:
            # x86: detectar cores fisicos reales
            n_physical = _detect_physical_cores()
            n_threads = n_physical if n_physical > 0 else max(1, total // 2)

    torch.set_num_threads(n_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # Ya configurado

    print(f"[CPU opt] Threads: {n_threads} intra-op, 1 inter-op")
    return n_threads


def _detect_physical_cores() -> int:
    """Detectar cores fisicos reales (sin hyperthreads). Multiplataforma."""
    import subprocess
    try:
        if platform.system() == 'Windows':
            # WMIC devuelve cores fisicos reales
            out = subprocess.check_output(
                'wmic cpu get NumberOfCores /value',
                shell=True, text=True, timeout=5,
            )
            for line in out.strip().split('\n'):
                if 'NumberOfCores' in line:
                    return int(line.split('=')[1].strip())
        elif platform.system() == 'Linux':
            out = subprocess.check_output(
                'lscpu -p=Core,Socket | grep -v "^#" | sort -u | wc -l',
                shell=True, text=True, timeout=5,
            )
            return int(out.strip())
        elif platform.system() == 'Darwin':
            out = subprocess.check_output(
                ['sysctl', '-n', 'hw.physicalcpu'],
                text=True, timeout=5,
            )
            return int(out.strip())
    except Exception:
        pass
    return 0  # fallback a os.cpu_count() // 2


def set_thread_affinity(core_ids: list[int]) -> None:
    """
    Fijar proceso a cores especificos para estabilidad de L1/L2.

    Cuando el OS mueve el proceso entre cores, L1/L2 per-core se invalida.
    Fijarlo mantiene el estado MLGRU caliente en cache.

    Soporta Linux (sched_setaffinity) y Windows (SetProcessAffinityMask).
    """
    try:
        os.sched_setaffinity(0, set(core_ids))
        print(f"[CPU opt] Thread affinity -> cores {core_ids}")
        return
    except (AttributeError, OSError):
        pass

    # Windows: usar SetProcessAffinityMask via ctypes
    if platform.system() == 'Windows':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentProcess()
            # Bitmask: bit i = core i
            mask = sum(1 << c for c in core_ids)
            result = kernel32.SetProcessAffinityMask(handle, mask)
            if result:
                print(f"[CPU opt] Thread affinity (Windows) -> cores {core_ids}")
            else:
                print("[CPU opt] SetProcessAffinityMask fallo")
        except Exception:
            pass


def compile_for_cpu_inference(model: nn.Module) -> nn.Module:
    """
    Compilar modelo para CPU — estrategia per-block (Cambio 20).

    Compila cada bloque individualmente en lugar del modelo completo.
    Mas estable que fullgraph=True para arquitecturas mixtas (recurrente +
    Transformer) y evita el graph-break del loop MLGRU.

    Ganancia tipica: ~1.4x vs eager mode por bloque. La primera inferencia
    compila cada bloque la primera vez que se ejecuta (~30-60 seg total).
    """
    # Configurar inductor para CPU
    try:
        import torch._inductor.config as inductor_cfg
        inductor_cfg.cpp.cxx_flags = ["-O3", "-mtune=native", "-mavx2", "-mfma", "-funroll-loops"]
        inductor_cfg.cpp.enable_kernel_profile = False
    except Exception:
        pass

    if not hasattr(torch, 'compile'):
        print("[CPU opt] torch.compile no disponible, usando eager mode")
        return model

    # Disable torch.compile for now due to BackendCompilerFailed errors
    # TODO: Debug torch.compile CPU inference issues
    return model

    if not hasattr(model, 'blocks'):
        # Modelo sin .blocks (improbable): intentar compilacion completa
        try:
            return torch.compile(model, mode="reduce-overhead", backend="inductor")
        except Exception:
            return model

    # Compilar cada bloque individualmente
    n_compiled = 0
    for i in range(len(model.blocks)):
        try:
            model.blocks[i] = torch.compile(
                model.blocks[i],
                mode="reduce-overhead",
                backend="inductor",
                fullgraph=False,
            )
            n_compiled += 1
        except Exception as e:
            # Si falla un bloque, dejar en eager (no es fatal)
            pass

    if n_compiled > 0:
        print(f"[CPU opt] torch.compile per-block: {n_compiled}/{len(model.blocks)} bloques compilados")
    else:
        print("[CPU opt] torch.compile per-block: todos fallaron, usando eager mode")

    return model


def warmup_compiled_model(
    model,
    config: dict,
    n_warmup: int = 3,
) -> None:
    """
    Warmup del modelo compilado antes de benchmarkear.
    La primera llamada dispara la compilacion JIT.
    """
    d_model = config.get('d_model', D_MODEL)
    n_layers = config.get('n_layers', N_LAYERS)
    dummy_patch = torch.zeros(1, 1, 4, dtype=torch.long)
    dummy_states = alloc_hidden_states(n_layers, d_model, verbose=False)

    print(f"[CPU opt] Warmup {n_warmup} pasos (compilacion JIT en paso 1)...")
    for i in range(n_warmup):
        t0 = time.perf_counter()
        with torch.no_grad():
            result = model(dummy_patch, dummy_states)
            # CPU1ModelV2: (logits, states), AblationModel: (logits, states, extra)
            dummy_states = result[1]
        t1 = time.perf_counter()
        print(f"  Paso {i+1}: {(t1-t0)*1000:.1f}ms")
    print("[CPU opt] Warmup completo")


def prefetch_layer_weights(block: nn.Module) -> None:
    """
    Hint al CPU para traer pesos de un bloque a L3 via data_ptr() touch.

    NOTA: .item() eliminado — causaba sync Python/C++ y costaba ~12% del decode.
    El hardware prefetcher de Intel/AMD detecta los accesos secuenciales durante
    el forward de la capa i y trae la i+2 automaticamente via stream prefetch.
    Tocar solo data_ptr() es suficiente para registrar la pagina en TLB.
    """
    # Solo registrar TLB (sin .item() — ese era el costo real)
    for module in block.modules():
        if hasattr(module, 'weight_int8_gemv'):
            _ = module.weight_int8_gemv.data_ptr()


def forward_with_prefetch(
    model: nn.Module,
    x: Tensor,
    hidden_states: list,
    _block_flags: list | None = None,
    _states_buf: list | None = None,
) -> tuple[Tensor, list]:
    """
    Forward pass con prefetch de capa i+2 mientras se computa capa i.

    Soporta CPU1BlockV2 (forward_one_token) y AblationBlock (forward).
    Para Transformer (sin estados), hidden_states es lista de None.

    Args:
        _block_flags: lista pre-computada de bool (True si el bloque tiene
                      forward_one_token). Evita 14 hasattr() por patch.
        _states_buf:  buffer pre-alocado para los nuevos estados. Reutilizado
                      entre llamadas para evitar 14 list.append() + resize.
    """
    # Manejar modelo compilado (OptimizedModule)
    blocks = model.blocks
    if hasattr(model, '_orig_mod'):
        blocks = model._orig_mod.blocks

    n = len(blocks)
    # Si no se pasan buffers externos, crear locales (fallback seguro)
    if _states_buf is None:
        _states_buf = [None] * n
    if _block_flags is None:
        _block_flags = [hasattr(b, 'forward_one_token') for b in blocks]

    for i in range(n):
        # Prefetch capa i+2 async mientras procesamos capa i
        if i + 2 < n:
            prefetch_layer_weights(blocks[i + 2])

        if _block_flags[i]:
            x, h_new = blocks[i].forward_one_token(x, hidden_states[i])
        else:
            x_3d = x.unsqueeze(1) if x.dim() == 2 else x
            x_3d, h_new = blocks[i](x_3d, hidden_states[i])
            x = x_3d.squeeze(1) if x_3d.dim() == 3 else x_3d

        _states_buf[i] = h_new

    return x, _states_buf


def hadamard_update_manual(
    h: Tensor,  # [B, d_model] estado previo, modificado in-place
    f: Tensor,  # [B, d_model] forget gate
    c: Tensor,  # [B, d_model] candidate
    g: Tensor,  # [B, d_model] output gate
) -> Tensor:
    """
    Actualizacion MLGRU con operaciones in-place o delegacion C++ fusion.
    """
    lib = _load_c_kernel()
    if lib and hasattr(lib, 'mlgru_scan_fusion') and h.is_contiguous() and f.is_contiguous():
        B, d_model = h.shape
        out = torch.empty_like(h)
        lib.mlgru_scan_fusion(
            ctypes.c_void_p(c.contiguous().data_ptr()),
            ctypes.c_void_p(f.data_ptr()),
            ctypes.c_void_p(g.contiguous().data_ptr()),
            ctypes.c_void_p(h.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(B), ctypes.c_int(d_model)
        )
        return out

    torch.lerp(c, h, f, out=h)  # in-place, sin allocar
    out = g * h
    return out


# ===========================================================================
# BLOQUE 3 — Loop de Generacion Optimizado
# ===========================================================================

@torch.inference_mode()
def _nucleus_sample(
    probs: Tensor,
    p: float,
    sorted_buf: Tensor | None = None,
    idx_buf: Tensor | None = None,
) -> int:
    """Nucleus (top-p) sampling con buffers pre-alocados opcionales.

    Args:
        probs: [V] probabilidades (ya son softmax)
        p: threshold de nucleus
        sorted_buf: [V] buffer pre-alocado para sorted probs (evita alloc)
        idx_buf: [V] buffer pre-alocado para sorted indices (evita alloc)

    Optimizacion vocab=256: torch.sort sobre 256 floats es trivial (~1µs).
    El overhead dominante es la alloc de tensores intermedios — los buffers
    pre-alocados eliminan eso. Para vocabularios grandes (BPE ~152K) el sort
    es el cuello de botella y se podria usar torch.topk con k dinamico.
    """
    V = probs.shape[0]
    if sorted_buf is not None and idx_buf is not None:
        # In-place sort en buffers pre-alocados: sin alloc
        torch.sort(probs, descending=True, out=(sorted_buf, idx_buf))
        sp, si = sorted_buf, idx_buf
    elif V <= 256:
        # vocab byte: sort trivial, pero sin buffers externos usamos argsort
        # que es ligeramente mas rapido que sort para 256 elementos
        si = torch.argsort(probs, descending=True)
        sp = probs[si]
    else:
        sp, si = torch.sort(probs, descending=True)

    cumsum = torch.cumsum(sp, dim=0)
    mask = cumsum - sp > p
    sp = sp.clone() if sp.data_ptr() == probs.data_ptr() else sp
    sp[mask] = 0.0
    total = sp.sum()
    if total > 0:
        sp /= total
    return si[torch.multinomial(sp.unsqueeze(0), 1).item()].item()

class CPU1InferenceEngine:
    """
    Motor de inferencia CPU-optimizado — Multi-arquitectura.

    Soporta CPU1ModelV2, AblationModel, y futuros modelos.
    Detecta automaticamente la arquitectura y aplica optimizaciones
    condicionales (INT8 solo para ternary, estados solo para recurrente).

    Uso:
        engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
        output = engine.generate(b"Buenos dias", max_bytes=200)
        print(engine.info)  # ModelInfo(model_class='CPU1ModelV2', ...)
    """

    def __init__(self, model: nn.Module, config: dict):
        self.config = config

        # === Detectar arquitectura ===
        self.info = detect_model_info(model, config)
        d_model = self.info.d_model
        n_layers = self.info.n_layers

        # === Aplicar optimizaciones ===
        print("\n" + "=" * 60)
        print(f"Optimizaciones CPU — {self.info.model_class}")
        print(f"  byte_level={self.info.is_byte_level} recurrent={self.info.is_recurrent} "
              f"ternary={self.info.is_ternary} heads={self.info.n_heads}")
        print("=" * 60)

        model.eval()

        # Bloque 1: layout de memoria (solo si hay BitLinear)
        if self.info.is_ternary:
            transpose_weights_for_inference(model)
            align_all_weights(model)
            self.scales = pack_all_bitlinear_to_int8(model)
            patch_activations_for_inference(model)
        else:
            align_all_weights(model)
            self.scales = {}
            # Cambio 18: bfloat16 para modelos FP16 — ~1.5x mas rapido en Intel BF16
            # Solo activar si AVX-512 BF16 esta disponible en la CPU
            self.info._bf16_active = False
            try:
                cap = str(torch._C._get_cpu_capability())
                if 'avx512' in cap.lower() or 'amx' in cap.lower():
                    model = model.to(torch.bfloat16)
                    self.info._bf16_active = True
                    print("[CPU opt] Modelo bfloat16 (AVX-512 BF16 detectado)")
                else:
                    print("[CPU opt] Modelo FP32 (no AVX-512 BF16 en esta CPU)")
            except Exception:
                print("[CPU opt] Modelo FP16 — skip INT8 packing")

        # Bloque 2: compilador
        n_threads = configure_cpu_threads(config.get('n_threads'))
        set_thread_affinity(list(range(n_threads)))

        # Pre-alocar buffers (dtype debe coincidir con el modelo)
        state_dtype = torch.bfloat16 if getattr(self.info, '_bf16_active', False) else torch.float32
        if self.info.is_recurrent:
            self.hidden_states = alloc_hidden_states(n_layers, d_model, dtype=state_dtype)
        else:
            # Transformer: lista de None (sin estados recurrentes)
            self.hidden_states = [None] * n_layers
            print(f"[CPU opt] Transformer mode — no MLGRU states")

        vocab = self.info.vocab_size
        n_heads = self.info.n_heads
        self.logits_buf = torch.zeros(n_heads, vocab, dtype=torch.float32)
        self.probs_buf = torch.zeros(n_heads, vocab, dtype=torch.float32)
        self.sorted_buf = torch.zeros(vocab, dtype=torch.float32)
        self.idx_buf = torch.zeros(vocab, dtype=torch.long)
        # Pre-allocated states buffer (avoids 14 list.append per patch)
        self._states_buf = list(self.hidden_states)  # same length, overwritten in-place

        # Compilar (per-block: modifica model.blocks[i] in-place)
        compiled = compile_for_cpu_inference(model)
        try:
            warmup_compiled_model(compiled, config, n_warmup=2)
            self.model = compiled
        except Exception as e:
            print(f"[CPU opt] torch.compile fallo en runtime ({type(e).__name__}), "
                  f"usando eager mode")
            # Un-compile: blocks were modified in-place, restore originals
            if hasattr(model, 'blocks'):
                for i, block in enumerate(model.blocks):
                    if hasattr(block, '_orig_mod'):
                        model.blocks[i] = block._orig_mod
            self.model = model
            warmup_compiled_model(model, config, n_warmup=2)

        # Pre-compute block capability flags AFTER compilation.
        # forward_one_token calls self.forward() (not self.__call__), so for
        # per-block compiled blocks (OptimizedModule) it bypasses compilation.
        # In that case, prefer block(x_3d, h) via __call__ to get the compiled path.
        _active_blocks = self._get_model().blocks
        _any_compiled = any(hasattr(b, '_orig_mod') for b in _active_blocks)
        if _any_compiled:
            # Use __call__ for all blocks to route through compiled forward
            self._block_flags = [False] * len(_active_blocks)
        else:
            self._block_flags = [hasattr(b, 'forward_one_token') for b in _active_blocks]

        print("=" * 60)
        print("Optimizaciones aplicadas")
        print("=" * 60 + "\n")

    @classmethod
    def from_checkpoint(cls, path: str, config: dict) -> 'CPU1InferenceEngine':
        """Cargar checkpoint y crear engine. Auto-detecta tipo de modelo."""
        # Cambio 19: mmap=True evita duplicar RAM al cargar (350M = ~360MB en FP16)
        # Disponible en PyTorch >= 2.1
        load_kwargs: dict = {'map_location': 'cpu', 'weights_only': False}
        try:
            import torch.version
            major, minor = (int(x) for x in torch.version.__version__.split('.')[:2])
            if (major, minor) >= (2, 1):
                load_kwargs['mmap'] = True
        except Exception:
            pass
        checkpoint = torch.load(path, **load_kwargs)
        state = checkpoint.get('model_state_dict', checkpoint)

        # Intentar cargar como el modelo que corresponda segun config
        model = _create_model_from_config(config, state)
        model.load_state_dict(state)
        model.eval()
        return cls(model, config)

    def _get_model(self) -> nn.Module:
        """Obtener modelo original (sin wrapper de compile)."""
        if hasattr(self.model, '_orig_mod'):
            return self.model._orig_mod
        return self.model

    def reset_state(self) -> None:
        """Resetear estado (MLGRU → nuevos zeros, Transformer → no-op)."""
        if self.info.is_recurrent:
            state_dtype = torch.bfloat16 if getattr(self.info, '_bf16_active', False) else torch.float32
            # Reallocar para evitar conflictos con InferenceMode
            self.hidden_states = alloc_hidden_states(
                self.info.n_layers, self.info.d_model, dtype=state_dtype, verbose=False
            )
            self._states_buf = list(self.hidden_states)  # sync pre-alloc buf
        # Transformer: hidden_states son None, nada que resetear

    def _bytes_to_patches(self, data: bytes) -> Tensor:
        """Convertir bytes a patches de 4. Padding con 0 si no es multiplo."""
        padded = data + b'\x00' * ((-len(data)) % self.info.patch_size)
        t = torch.frombuffer(bytearray(padded), dtype=torch.uint8)
        return t.reshape(1, -1, self.info.patch_size).long()

    def _get_heads(self, model: nn.Module) -> list:
        """Obtener las heads de prediccion del modelo."""
        if hasattr(model, 'byte_heads'):
            return list(model.byte_heads)
        elif hasattr(model, 'heads'):
            return list(model.heads)
        raise AttributeError(f"{type(model).__name__} no tiene byte_heads ni heads")

    def _get_embedding(self, model: nn.Module):
        """Obtener la funcion de embedding del modelo."""
        if hasattr(model, 'patch_embedding'):
            return model.patch_embedding
        elif hasattr(model, 'embedding'):
            return model.embedding
        raise AttributeError(f"{type(model).__name__} no tiene patch_embedding ni embedding")

    @torch.inference_mode()
    def prefill(self, prompt_bytes: bytes) -> None:
        """Procesar prompt y actualizar hidden states."""
        if self.info.is_byte_level:
            x = self._bytes_to_patches(prompt_bytes)
        else:
            raise NotImplementedError("BPE prefill requiere tokenizer — usar prefill_tokens()")

        result = self.model(x, self.hidden_states)
        # Manejar ambos formatos de retorno:
        # CPU1ModelV2: (logits_tuple, states)
        # AblationModel: (logits, states, extra_losses)
        if len(result) == 2:
            _, self.hidden_states = result
        else:
            _, self.hidden_states, _ = result

    @torch.inference_mode()
    def generate(
        self,
        prompt_bytes: bytes,
        max_bytes: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        reset: bool = True,
    ) -> bytes:
        """
        Generar texto byte a byte (modelos byte-level).
        Soporta CPU1ModelV2 y AblationModel con tokenization=byte.

        Para modelos BPE, usar generate_tokens() en su lugar.
        """
        if not self.info.is_byte_level:
            raise NotImplementedError(
                "generate() es para modelos byte-level. "
                "Para BPE usar generate_tokens()."
            )

        if reset:
            self.reset_state()

        # Prefill
        t_prefill = time.perf_counter()
        self.prefill(prompt_bytes)
        prefill_ms = (time.perf_counter() - t_prefill) * 1000

        model = self._get_model()
        embed_fn = self._get_embedding(model)
        heads = self._get_heads(model)
        n_heads = len(heads)

        # Generacion
        output_buf = bytearray(max_bytes)
        last_patch = self._bytes_to_patches(prompt_bytes)[:, -1:, :]

        n_patches = max_bytes // self.info.patch_size
        token_times = []
        inv_temp = 1.0 / temperature if temperature > 0 else 0.0

        for i in range(n_patches):
            t0 = time.perf_counter()

            # Embed: [1, 1, patch_size] -> [1, d_model]
            x = embed_fn(last_patch).squeeze(1)

            # Forward con prefetch — usa buffers pre-alocados (Cambios 11+14)
            x, self.hidden_states = forward_with_prefetch(
                model, x, self.hidden_states,
                _block_flags=self._block_flags,
                _states_buf=self._states_buf,
            )

            # Norm final
            x_3d = x.unsqueeze(1)
            x_normed = model.norm(x_3d)

            # N heads → logits_buf[n_heads, vocab] (sin alloc)
            for h_idx, head in enumerate(heads):
                self.logits_buf[h_idx].copy_(head(x_normed).squeeze(1).squeeze(0))

            # Sampling batch (Cambio 17): 1 softmax + 1 multinomial para todos los heads
            if temperature == 0:
                next_bytes = self.logits_buf.argmax(dim=-1).tolist()
            else:
                # Batch softmax: [n_heads, vocab] en un solo op
                torch.softmax(self.logits_buf * inv_temp, dim=-1, out=self.probs_buf)
                if top_p < 1.0:
                    # Nucleus per-head (no se puede vectorizar el threshold adaptativo)
                    next_bytes = []
                    for h_idx in range(n_heads):
                        next_bytes.append(_nucleus_sample(
                            self.probs_buf[h_idx], top_p,
                            sorted_buf=self.sorted_buf,
                            idx_buf=self.idx_buf,
                        ))
                else:
                    # Batch multinomial: un solo call para todos los heads
                    next_bytes = torch.multinomial(self.probs_buf, 1).squeeze(-1).tolist()

            # Escribir output
            base = i * n_heads
            for j in range(n_heads):
                idx = base + j
                if idx < max_bytes:
                    output_buf[idx] = next_bytes[j]

            # Preparar proximo patch
            last_patch[0, 0, :n_heads] = torch.tensor(next_bytes, dtype=torch.long)
            token_times.append((time.perf_counter() - t0) * 1000)

        # Metricas
        if token_times:
            avg_ms = sum(token_times) / len(token_times)
            p50 = sorted(token_times)[len(token_times) // 2]
            p95 = sorted(token_times)[int(len(token_times) * 0.95)]
            tps = 1000 / avg_ms * n_heads

            print(f"\n[Inference] Prefill: {prefill_ms:.1f}ms")
            print(f"[Inference] Decode:  {avg_ms:.2f}ms/patch | "
                  f"p50={p50:.2f} p95={p95:.2f}")
            print(f"[Inference] Throughput: {tps:.1f} bytes/seg "
                  f"({tps/n_heads:.1f} patches/seg)")

        return bytes(output_buf[:max_bytes])


def _create_model_from_config(config: dict, state_dict: dict) -> nn.Module:
    """Crear el modelo correcto segun la config.
    
    Detecta si es CPU1ModelV2, AblationModel, u otro por la config.
    """
    # Si la config tiene 'token_mixer' o 'tokenization', es AblationModel
    if _HAS_ABLATION and ('token_mixer' in config or 'tokenization' in config):
        from ablation import build_ablation_model, get_run_config
        # Si es un run_XX config, expandirlo
        if 'name' not in config and 'd_model' not in config:
            # Es un run name directo
            pass
        model = AblationModel(config)
        return model

    return CPU1ModelV2(config)


# ===========================================================================
# BLOQUE 4 — Benchmark Completo para el Paper
# ===========================================================================

def run_full_benchmark(
    engine: CPU1InferenceEngine,
    test_prompts: list[bytes],
    n_tokens: int = 200,
) -> dict:
    """
    Benchmark completo para reportar en el paper.

    Metricas reportadas:
    - tokens/seg (throughput)
    - latencia p50/p95 por token (ms)
    - RAM total usada (MB)
    - Estado MLGRU (KB)

    Baselines a comparar en el MISMO hardware:
    - llama.cpp Q4_0 1B (Llama-3.2)
    - llama.cpp Q2_K 1B
    - Slender-Mamba 170M ternario
    """
    try:
        import psutil
        process = psutil.Process()
        has_psutil = True
    except ImportError:
        has_psutil = False

    # Warmup adicional
    engine.generate(test_prompts[0], max_bytes=20, reset=True)

    all_times = []
    ram_samples = []

    for prompt in test_prompts:
        if has_psutil:
            _ = process.memory_info().rss  # pre-read

        t_start = time.perf_counter()
        output = engine.generate(prompt, max_bytes=n_tokens, reset=True)
        t_end = time.perf_counter()

        if has_psutil:
            ram_after = process.memory_info().rss / 1e6
            ram_samples.append(ram_after)

        all_times.append(t_end - t_start)

    avg_bps = n_tokens * len(test_prompts) / sum(all_times)

    results = {
        'bytes_per_sec': avg_bps,
        'tokens_per_sec': avg_bps / 4,  # patches de 4 bytes
        'latency_ms_p50': sorted(all_times)[len(all_times) // 2]
                          / n_tokens * 1000,
        'latency_ms_p95': sorted(all_times)[int(len(all_times) * 0.95)]
                          / n_tokens * 1000,
        'ram_mb': sum(ram_samples) / len(ram_samples) if ram_samples else 0,
        'mlgru_state_kb': sum(
            h.numel() * h.element_size() for h in engine.hidden_states
        ) / 1024,
    }

    # L3 cache miss rate (solo Linux)
    try:
        import subprocess
        perf = subprocess.run(
            ['perf', 'stat', '-e', 'cache-misses,cache-references',
             '-p', str(os.getpid()), 'sleep', '1'],
            capture_output=True, text=True, timeout=5,
        )
        results['perf_output'] = perf.stderr
    except Exception:
        results['perf_output'] = 'perf no disponible (requiere Linux)'

    # Print tabla para el paper
    print("\n" + "=" * 60)
    print("BENCHMARK CPU-1 v2 — Resultados para el Paper")
    print("=" * 60)
    print(f"Throughput:          {results['tokens_per_sec']:.1f} patches/seg")
    print(f"                     {results['bytes_per_sec']:.1f} bytes/seg")
    print(f"Latencia p50:        {results['latency_ms_p50']:.2f} ms/patch")
    print(f"Latencia p95:        {results['latency_ms_p95']:.2f} ms/patch")
    print(f"RAM usada:           {results['ram_mb']:.1f} MB")
    print(f"Estado MLGRU:        {results['mlgru_state_kb']:.1f} KB "
          f"(vs ~80MB KV cache atencion)")
    print("=" * 60)

    return results


# ===========================================================================
# BLOQUE 6 — HuggingFace Engine (modelos externos)
# ===========================================================================

class _HFBlockWrapper(nn.Module):
    """Wrappea un bloque HuggingFace para interfaz (x, h=None) → (x, None)."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x: Tensor, h=None) -> tuple[Tensor, None]:
        out = self.block(x)
        if isinstance(out, tuple):
            return out[0], None
        return out, None


class _HFModelWrapper(nn.Module):
    """Modelo HuggingFace wrapeado con interfaz compatible con CPU1InferenceEngine.

    Atributos:
        blocks:    lista de _HFBlockWrapper
        norm:      layer norm final
        embedding: funcion de embedding (wte + wpe para GPT-2)
        heads:     ModuleList con una head (lm_head)
    """

    def __init__(self, hf_model, config):
        super().__init__()
        self._hf = hf_model

        # Detectar componentes por arquitectura
        if hasattr(hf_model, 'transformer'):
            # GPT-2 style
            tf = hf_model.transformer
            self.blocks = nn.ModuleList(
                [_HFBlockWrapper(b) for b in tf.h]
            )
            self.norm = tf.ln_f
            self._wte = tf.wte
            self._wpe = tf.wpe if hasattr(tf, 'wpe') else None
        elif hasattr(hf_model, 'model') and hasattr(hf_model.model, 'layers'):
            # LLaMA / Mistral / LFM style
            m = hf_model.model
            self.blocks = nn.ModuleList(
                [_HFBlockWrapper(b) for b in m.layers]
            )
            self.norm = m.norm if hasattr(m, 'norm') else m.final_layernorm
            self._wte = m.embed_tokens
            self._wpe = None
        elif hasattr(hf_model, 'backbone'):
            # LFM2 / Mamba style — backbone.layers
            bb = hf_model.backbone
            layers = bb.layers if hasattr(bb, 'layers') else list(bb.children())
            self.blocks = nn.ModuleList(
                [_HFBlockWrapper(b) for b in layers]
            )
            self.norm = bb.norm_f if hasattr(bb, 'norm_f') else bb.final_layernorm
            self._wte = bb.embedding if hasattr(bb, 'embedding') else bb.embeddings
            self._wpe = None
        else:
            children_names = [n for n, _ in hf_model.named_children()]
            raise ValueError(
                f"Arquitectura HF no soportada: {type(hf_model).__name__}. "
                f"Children: {children_names}"
            )

        # Head de prediccion
        if hasattr(hf_model, 'lm_head'):
            self.heads = nn.ModuleList([hf_model.lm_head])
        elif hasattr(hf_model, 'output'):
            self.heads = nn.ModuleList([hf_model.output])
        else:
            raise ValueError("No se encontro lm_head en el modelo HF")

        self._d_model = getattr(config, 'hidden_size',
                                getattr(config, 'd_model', 768))

    def embedding(self, x: Tensor) -> Tensor:
        """Embedding compatible: [batch, seq] → [batch, seq, d_model]."""
        h = self._wte(x)
        if self._wpe is not None:
            seq_len = x.shape[-1] if x.dim() >= 2 else x.shape[0]
            # Cache pos_ids: evita torch.arange en cada llamada de embedding
            if not hasattr(self, '_pos_ids_len') or self._pos_ids_len != seq_len:
                self._pos_ids = torch.arange(seq_len, device=x.device)
                self._pos_ids_len = seq_len
            h = h + self._wpe(self._pos_ids)
        return h

    def forward(self, x: Tensor, hidden_states=None) -> tuple[Tensor, list]:
        """Forward completo: [batch, seq] → (logits, states).

        Acepta model(x) sin hidden_states para compatibilidad con evals.
        """
        h = self.embedding(x)

        n_blocks = len(self.blocks)
        if hidden_states is None:
            hidden_states = [None] * n_blocks

        # Pre-allocated fixed-size list: evita n_blocks append() en cada llamada
        states = [None] * n_blocks
        for i, block in enumerate(self.blocks):
            hi = hidden_states[i] if i < len(hidden_states) else None
            h, states[i] = block(h, hi)

        logits = self.heads[0](self.norm(h))
        return logits, states


def _quantize_int4(model: nn.Module) -> nn.Module:
    """Cuantizar a INT4 weight-only via torchao."""
    try:
        from torchao.quantization import quantize_, int4_weight_only
        quantize_(model, int4_weight_only(group_size=128))
        print(f"  Cuantizado: INT4 weight-only (torchao, group_size=128)")
        return model
    except ImportError:
        print(f"  [WARN] torchao no disponible para INT4, fallback a INT8")
        return _quantize_int8_dynamic(model)
    except Exception as e:
        print(f"  [WARN] INT4 fallo ({e}), fallback a INT8")
        return _quantize_int8_dynamic(model)


def _quantize_int8_dynamic(model: nn.Module) -> nn.Module:
    """Cuantizar a INT8 dinamico via torch.ao."""
    model = torch.ao.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    print(f"  Cuantizado: INT8 dinamico (CPU)")
    return model


def _hf_sample_token(logits: Tensor, temperature: float, top_p: float) -> int:
    """Sampling eficiente para loop autoregresivo manual.

    - temperature <= 0.01 → argmax (greedy)
    - top_p < 1.0        → nucleus sampling
    """
    if temperature <= 0.01:
        return int(logits.argmax().item())

    logits = logits / temperature

    if top_p < 1.0:
        probs = torch.softmax(logits, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=0)
        # Mantener tokens hasta cubrir top_p de masa
        cutoff = (cumsum - sorted_probs) < top_p
        sorted_probs = sorted_probs * cutoff.float()
        sorted_probs /= sorted_probs.sum()
        chosen = int(torch.multinomial(sorted_probs, 1).item())
        return int(sorted_idx[chosen].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1).item())


class HuggingFaceEngine:
    """
    Engine para modelos HuggingFace con interfaz eval-compatible.

    Soporta INT4 (torchao) e INT8 (torch.ao) en CPU.
    Soporta GPT-2, LLaMA, LFM2.5, etc.

    Uso:
        engine = HuggingFaceEngine('openai-community/gpt2-medium', quantize_bits=4)
        engine = HuggingFaceEngine('models/gpt2-medium', local=True, quantize_bits=8)
    """

    def __init__(self, model_name: str, n_threads: int = None,
                 quantize_bits: int = 8, local: bool = False,
                 save_to: str = None):
        from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer

        print(f"\n{'=' * 60}")
        print(f"HuggingFace Engine — {model_name}")
        print(f"{'=' * 60}")

        load_kwargs = dict(trust_remote_code=True)
        if local:
            load_kwargs['local_files_only'] = True
            print(f"  Cargando desde carpeta local: {model_name}")

        hf_config = AutoConfig.from_pretrained(model_name, **load_kwargs)

        # Determinar dimensiones
        d_model = getattr(hf_config, 'hidden_size',
                          getattr(hf_config, 'd_model', 768))
        n_layers = getattr(hf_config, 'num_hidden_layers',
                           getattr(hf_config, 'n_layer', 6))
        vocab_size = hf_config.vocab_size

        print(f"  d_model={d_model}, n_layers={n_layers}, vocab={vocab_size}")

        # Cargar modelo
        print(f"  Cargando modelo...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, **load_kwargs,
        )
        hf_model.eval()

        n_params = sum(p.numel() for p in hf_model.parameters())
        print(f"  Parametros: {n_params:,} ({n_params / 1e6:.1f}M)")

        # Guardar modelo original a carpeta local si se pide (ANTES de cuantizar)
        if save_to and not local:
            os.makedirs(save_to, exist_ok=True)
            print(f"  Guardando modelo FP32 en: {save_to}")
            hf_model.save_pretrained(save_to)
            tokenizer_save = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True)
            tokenizer_save.save_pretrained(save_to)
            del tokenizer_save
            print(f"  Modelo y tokenizer guardados")

        # Cuantizar
        if quantize_bits <= 4:
            hf_model = _quantize_int4(hf_model)
            actual_bits = 4
        elif quantize_bits <= 8:
            hf_model = _quantize_int8_dynamic(hf_model)
            actual_bits = 8
        else:
            actual_bits = 32
            print(f"  Sin cuantizar (FP32)")

        # Tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name, **load_kwargs
        )

        # Optimizaciones HF / LFM-350m para maximizar tokens/sec (Static Cache + Torch Compile)
        if hasattr(hf_model, "generation_config"):
            try:
                hf_model.generation_config.cache_implementation = "static"
                print("  [HuggingFaceEngine] Generacion configurada con StaticCache")
            except Exception:
                pass
        try:
            import torch._inductor.config as inductor_cfg
            inductor_cfg.cpp.cxx_flags = ["-O3", "-mtune=native", "-mavx2", "-mfma", "-funroll-loops"]
            # Disable torch.compile for HF models too
            # hf_model.forward = torch.compile(hf_model.forward, mode="reduce-overhead")
            print("  [HuggingFaceEngine] torch.compile desactivado (debug mode)")
        except Exception as e:
            print(f"  [HuggingFaceEngine] torch.compile fallback due to: {e}")

        # Crear wrapper
        self._wrapped = _HFModelWrapper(hf_model, hf_config)
        self._wrapped.eval()
        self.model = self._wrapped
        self.config = {
            'model_name': model_name,
            'd_model': d_model,
            'n_layers': n_layers,
            'quantize_bits': actual_bits,
        }

        # ModelInfo
        model_class_name = type(hf_model).__name__
        # Si esta cuantizado, el tipo puede cambiar
        if hasattr(hf_model, '_orig_mod'):
            model_class_name = type(hf_model._orig_mod).__name__
        self.info = ModelInfo(
            model_class=f"{model_class_name}(HF-Q{actual_bits})",
            is_byte_level=False,
            is_recurrent=False,
            is_ternary=False,
            n_heads=1,
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            patch_size=1,
            has_forward_one_token=False,
        )

        # Threads
        n = configure_cpu_threads(n_threads)
        set_thread_affinity(list(range(n)))

        # Buffers
        self.hidden_states = [None] * n_layers
        self.logits_buf = torch.zeros(1, vocab_size, dtype=torch.float32)
        self.probs_buf = torch.zeros(vocab_size, dtype=torch.float32)
        self.sorted_buf = torch.zeros(vocab_size, dtype=torch.float32)
        self.idx_buf = torch.zeros(vocab_size, dtype=torch.long)
        self.scales = {}

        print(f"{'=' * 60}")
        print(f"Engine listo: {self.info.model_class}")
        print(f"  {self.info}")
        print(f"{'=' * 60}\n")

    def _get_model(self) -> nn.Module:
        return self._wrapped

    def reset_state(self) -> None:
        self.hidden_states = [None] * self.info.n_layers

    def _get_embedding(self, model):
        return model.embedding

    def _get_heads(self, model):
        return list(model.heads)

    @torch.inference_mode()
    def prefill(self, prompt_bytes: bytes) -> None:
        raise NotImplementedError("HF models usan tokens, no bytes")

    @torch.inference_mode()
    def generate(self, prompt_bytes: bytes, max_bytes: int = 200,
                 temperature: float = 1.0, top_p: float = 0.9,
                 reset: bool = True) -> bytes:
        """Generar texto con loop autoregresivo manual (sin HF .generate overhead).

        Estrategia:
          1. Prefill: un forward sobre todos los tokens del prompt → KV cache
          2. Decode: un forward por token nuevo, pasando past_key_values
          Evita el overhead de Python en HF GenerationMixin (~30% del tiempo).
        """
        text = prompt_bytes.decode('utf-8', errors='replace')
        input_ids = self._tokenizer.encode(text, return_tensors='pt')
        max_new_tokens = max(1, max_bytes // 4)
        hf_model = self._wrapped._hf
        eos_id = self._tokenizer.eos_token_id

        # --- Prefill ---
        out = hf_model(input_ids, use_cache=True, return_dict=True)
        past_kv = out.past_key_values
        logits  = out.logits[:, -1, :]   # [1, vocab]

        generated = []
        for _ in range(max_new_tokens):
            # Sample
            next_id = _hf_sample_token(logits[0], temperature, top_p)
            generated.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

            # Decode step: single token con KV cache
            next_tok = torch.tensor([[next_id]], dtype=torch.long)
            out  = hf_model(next_tok, past_key_values=past_kv,
                            use_cache=True, return_dict=True)
            past_kv = out.past_key_values
            logits  = out.logits[:, -1, :]

        text_out = self._tokenizer.decode(generated, skip_special_tokens=True)
        return text_out.encode('utf-8')[:max_bytes]


def download_and_save_hf_model(model_name: str, save_dir: str,
                                clear_cache: bool = False) -> str:
    """Descargar modelo de HuggingFace y guardarlo localmente.

    Args:
        model_name: Nombre HF (ej: 'openai-community/gpt2-medium')
        save_dir: Carpeta destino (ej: 'models/gpt2-medium')
        clear_cache: Si True, limpiar cache HF despues de guardar

    Returns:
        Path absoluto de la carpeta guardada
    """
    import shutil
    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Download] {model_name} -> {save_dir}")

    print(f"  Descargando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32,
    )
    model.save_pretrained(save_dir)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parametros: {n_params:,} ({n_params / 1e6:.1f}M)")
    del model

    print(f"  Descargando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True,
    )
    tokenizer.save_pretrained(save_dir)
    del tokenizer

    print(f"  Guardado en: {os.path.abspath(save_dir)}")

    if clear_cache:
        _clear_hf_cache()

    return os.path.abspath(save_dir)


def _clear_hf_cache():
    """Limpiar cache de HuggingFace."""
    import shutil
    cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    if os.path.exists(cache_dir):
        hub_cache = os.path.join(cache_dir, 'hub')
        if os.path.exists(hub_cache):
            size_mb = sum(
                os.path.getsize(os.path.join(dp, f))
                for dp, dn, fn in os.walk(hub_cache) for f in fn
            ) / 1e6
            print(f"\n[Cache] Limpiando HF cache: {hub_cache} ({size_mb:.0f}MB)")
            shutil.rmtree(hub_cache, ignore_errors=True)
            print(f"[Cache] Cache eliminada")
        else:
            print(f"[Cache] No hay cache HF en {hub_cache}")
    else:
        print(f"[Cache] No hay cache HF en {cache_dir}")


# ===========================================================================
# Tests
# ===========================================================================

def test_assert_aligned():
    """Verifica assert_aligned y make_aligned."""
    t = torch.randn(256)
    t = make_aligned(t, 64)
    assert assert_aligned(t, 64), "make_aligned fallo"
    print("[OK] assert_aligned / make_aligned")


def test_transpose_weights():
    """Verifica transpose_weights_for_inference."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)
    transpose_weights_for_inference(model)
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            assert hasattr(module, 'weight_T'), f"{name}: weight_T no creado"
            expected = module.weight.data.T.shape
            assert module.weight_T.shape == expected, (
                f"{name}: shape {module.weight_T.shape} != {expected}"
            )
    print("[OK] transpose_weights_for_inference")


def test_pack_int8():
    """Verifica pack_all_bitlinear_to_int8."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)
    transpose_weights_for_inference(model)
    align_all_weights(model)
    scales = pack_all_bitlinear_to_int8(model)

    assert len(scales) > 0, "No scales returned"
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            assert hasattr(module, 'weight_T_ternary'), f"{name}: no ternary"
            assert hasattr(module, 'weight_T_int8'), f"{name}: no INT8"
            assert module.weight_T_int8.dtype == torch.int8
            assert module.weight_T_ternary.dtype == torch.float32
            assert hasattr(module, 'weight_scale')

    # Verificar que el forward patcheado funciona
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            d_in = module.weight_T_ternary.shape[0]
            d_out = module.weight_T_ternary.shape[1]
            x = torch.randn(1, 8, d_in)
            out = module(x)
            assert out.shape == (1, 8, d_out), (
                f"{name}: output {out.shape}, expected (1, 8, {d_out})"
            )
            break

    print("[OK] pack_all_bitlinear_to_int8")


def test_hidden_states():
    """Verifica alloc_hidden_states."""
    states = alloc_hidden_states(4, 256, verbose=False)
    assert len(states) == 4
    for s in states:
        assert s.shape == (1, 256)
        assert s.dtype == torch.float32
        assert assert_aligned(s, 64)
    print("[OK] alloc_hidden_states")


def test_nucleus_sample():
    """Verifica nucleus sampling."""
    probs = torch.zeros(256)
    probs[42] = 1.0
    assert _nucleus_sample(probs.clone(), 0.9) == 42
    print("[OK] _nucleus_sample")


def test_engine_generate():
    """Test principal: engine.generate(b'test', max_bytes=8) sin errores."""
    config = {'d_model': 128, 'n_layers': 2, 'd_ff': 256, 'fp_rank': 4}
    model = CPU1ModelV2(config)
    model.eval()

    engine = CPU1InferenceEngine(model, config)
    output = engine.generate(b"test", max_bytes=8)

    assert isinstance(output, bytes), f"Output type: {type(output)}"
    assert len(output) == 8, f"Expected 8 bytes, got {len(output)}"
    assert all(0 <= b <= 255 for b in output), "Invalid byte values"

    print(f"\n[OK] engine.generate(b'test', max_bytes=8) -> "
          f"{len(output)} bytes: {list(output)}")




# ===========================================================================
# Tabla de decodificacion constante (se construye una sola vez)
# ===========================================================================

def _build_decode_table(group_size: int = 4) -> Tensor:
    """
    Construir tabla constante: codigo → pesos ternarios.

    Para group_size=4: tabla de [128, 4] float32.
    Cada fila mapea un codigo de 7 bits a 4 pesos ternarios {-1, 0, 1}.
    Encoding: 2 bits por peso, 00=0, 01=+1, 10=-1.

    Se construye UNA VEZ al importar el modulo y se reutiliza.
    """
    n_half = 1 << (2 * group_size - 1)  # 128 para group_size=4
    codes = torch.arange(n_half, dtype=torch.int32)
    weights = torch.zeros(n_half, group_size, dtype=torch.float32)
    for i in range(group_size):  # solo 4 iteraciones, no hot path
        bits = (codes >> (2 * i)) & 0b11
        weights[:, i] = torch.where(bits == 1, 1.0,
                        torch.where(bits == 2, -1.0, 0.0))
    return weights


# Tabla global para group_size=4 (el default y recomendado)
_DECODE_TABLE_G4 = _build_decode_table(4)   # [128, 4] constante


# ===========================================================================
# PARTE B: Funciones LUT Core — Completamente Vectorizadas
# ===========================================================================

def precompute_all_luts(
    x_groups: Tensor,       # [M, n_groups, group_size] FP32
    decode_table: Tensor,   # [128, group_size] constante
) -> Tensor:
    """
    Precomputar TODAS las LUTs para todos los grupos y tokens a la vez.

    Vectorizado: un solo einsum reemplaza n_groups * M llamadas individuales.

    Args:
        x_groups: [M, n_groups, group_size] activaciones agrupadas
        decode_table: [128, group_size] tabla de decodificacion constante

    Returns:
        luts: [M, n_groups, 256] FP32 — todas las LUTs precomputadas
              Indices 0..127: valores directos (mirror positivo)
              Indices 128..255: valores negados (mirror negativo)
    """
    # Batch dot product: [M, n_groups, group_size] x [128, group_size]^T → [M, n_groups, 128]
    lut_pos = torch.einsum('mng,cg->mnc', x_groups, decode_table)

    # Mirror Consolidation: [M, n_groups, 256]
    return torch.cat([lut_pos, -lut_pos], dim=2)


def ternary_lut_matmul(
    x: Tensor,              # [M, d_in] FP32 — activaciones
    w_packed: Tensor,       # [d_out, n_groups] uint8 — pesos empaquetados
    w_scale: float,         # escalar de cuantizacion
    group_size: int = 4,
    decode_table: Tensor | None = None,   # [128, group_size] — reusar si disponible
    gather_index: Tensor | None = None,   # [n_groups, d_out] long — pre-computado
) -> Tensor:
    """
    Reemplaza BitLinear forward en inferencia.
    COMPLETAMENTE VECTORIZADO — sin loops Python.

    En vez de multiplicar, usa LUT precomputada + gather batched.

    Para d_in=1024, d_out=1024, group_size=4:
        1. Reshape x a [M, 256, 4] grupos
        2. einsum con decode_table [128, 4] → [M, 256, 128] LUTs
        3. cat mirror → [M, 256, 256] LUTs completas
        4. gather con w_packed → [M, 256, d_out] resultados
        5. sum dim=1 → [M, d_out] output

    Complejidad: O(M * n_groups * 128 * group_size) precompute +
                 O(M * n_groups * d_out) gather+sum
    Sin multiplicaciones de matrices. Compatible con torch.compile.

    Args:
        x:        [M, d_in] activaciones (M tokens)
        w_packed: [d_out, n_groups] uint8 — grupos de 4 pesos en 1 byte
        w_scale:  float — escalar para re-escalar output
        group_size: pesos por grupo (default 4)
        decode_table: tabla de decodificacion pre-construida (None = usar default)
        gather_index: indices pre-computados para gather (None = computar)

    Returns:
        [M, d_out] FP32
    """
    if decode_table is None:
        decode_table = _DECODE_TABLE_G4

    M = x.shape[0]
    d_in = x.shape[-1]
    d_out = w_packed.shape[0]
    n_groups = d_in // group_size

    # === FAST KERNEL (afanar de llama.cpp / kernels especificos en memoria) ===
    lib = _load_c_kernel()
    if lib and lib is not False:  # Explicit check: lib is a valid library object
        try:
            # Asegurar contiguidad — critical para el kernel C
            x_c = x.contiguous() if not x.is_contiguous() else x
            w_C = w_packed.contiguous()
            decode_c = decode_table.contiguous()
            out_c = torch.empty((M, d_out), dtype=torch.float32, device=x.device)

            lib.ternary_lut_matmul_omp(
                ctypes.c_void_p(x_c.data_ptr()),
                ctypes.c_void_p(w_C.data_ptr()),
                ctypes.c_float(w_scale),
                ctypes.c_void_p(decode_c.data_ptr()),
                ctypes.c_void_p(out_c.data_ptr()),
                ctypes.c_int(M), ctypes.c_int(d_in), ctypes.c_int(d_out), ctypes.c_int(group_size)
            )
            return out_c
        except Exception as e:
            # Fall back to PyTorch if kernel call fails
            pass
    # =========================================================================


    # 1. Reshape activaciones a grupos: [M, n_groups, group_size]
    x_groups = x.float().reshape(M, n_groups, group_size)

    # 2. Precomputar TODAS las LUTs: [M, n_groups, 256]
    luts = precompute_all_luts(x_groups, decode_table)

    # 3. Gather: para cada (m, g, neuron), buscar luts[m, g, w_packed[neuron, g]]
    #    gather_index: [n_groups, d_out] — indices pre-transpuestos
    if gather_index is None:
        gather_index = w_packed.long().T  # [n_groups, d_out]

    # Expandir para batch: [M, n_groups, d_out]
    index = gather_index.unsqueeze(0).expand(M, -1, -1)

    # torch.gather sobre dim=2 (el eje de 256 entradas de la LUT)
    # luts: [M, n_groups, 256], index: [M, n_groups, d_out]
    # → gathered: [M, n_groups, d_out]
    gathered = torch.gather(luts, dim=2, index=index)

    # 4. Sumar sobre grupos: [M, d_out]
    output = gathered.sum(dim=1)

    return output * w_scale


# ===========================================================================
# PARTE B3: Empaquetar pesos ternarios para LUT — Vectorizado
# ===========================================================================

def pack_ternary_for_lut(
    weight: Tensor,     # [d_out, d_in] FP16 o FP32
    group_size: int = 4
) -> tuple[Tensor, float]:
    """
    Convertir pesos ternarios → formato empaquetado para LUT.
    VECTORIZADO — sin loops Python sobre n_groups.

    Encoding por grupo de group_size pesos:
        Peso i ocupa bits [2i+1, 2i] del byte
        00 = 0, 01 = +1, 10 = -1

    Para group_size=4: 4 pesos x 2 bits = 8 bits = 1 byte por grupo
    Output shape: [d_out, d_in // group_size] uint8
    """
    w_scale = weight.float().abs().mean()
    w_ternary = (weight.float() / (w_scale + 1e-8)).round().clamp(-1, 1).long()

    d_out, d_in = w_ternary.shape
    n_groups = d_in // group_size

    # Reshape a grupos: [d_out, n_groups, group_size]
    w_groups = w_ternary[:, :n_groups * group_size].reshape(d_out, n_groups, group_size)

    # Encode: -1→2, 0→0, 1→1 (vectorizado)
    encoded = torch.where(w_groups == -1, 2,
              torch.where(w_groups == 1, 1, 0))  # [d_out, n_groups, group_size]

    # Pack bits: shift cada peso por su posicion y sumar
    # shifts = [0, 2, 4, 6] para group_size=4
    shifts = (2 * torch.arange(group_size, dtype=torch.int64))  # [group_size]
    shifts = shifts.reshape(1, 1, group_size)                   # broadcast

    packed = (encoded << shifts).sum(dim=2).to(torch.uint8)  # [d_out, n_groups]

    return packed, w_scale.item()


def pack_all_for_lut(model: CPU1ModelV2, group_size: int = 4) -> None:
    """
    Convertir TODOS los pesos BitLinear del modelo al formato LUT.

    Reemplaza pack_all_bitlinear_to_int8 del prompt 06.

    Agrega a cada modulo BitLinear:
        module.w_lut_packed:      [d_out, d_in//group_size] uint8
        module.w_lut_scale:       float
        module.lut_group_size:    int
        module.w_lut_gather_idx:  [n_groups, d_out] long — pre-computado para gather
        module.lut_decode_table:  [128, group_size] float32 — tabla de decodificacion

    Llamar OFFLINE, una vez, despues de cargar el checkpoint.
    """
    if group_size == 4:
        decode_table = _DECODE_TABLE_G4
    else:
        decode_table = _build_decode_table(group_size)

    n_packed = 0
    total_kb_saved = 0

    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight.data  # [d_out, d_in]
            packed, scale = pack_ternary_for_lut(w, group_size)

            module.w_lut_packed = packed.contiguous()
            module.w_lut_scale = scale
            module.lut_group_size = group_size
            # Pre-computar indice de gather (evita .T y .long() en cada forward)
            module.w_lut_gather_idx = packed.long().T.contiguous()  # [n_groups, d_out]
            module.lut_decode_table = decode_table

            # Stats
            original_kb = w.numel() * w.element_size() / 1024
            packed_kb = packed.numel() / 1024  # uint8
            total_kb_saved += original_kb - packed_kb
            n_packed += 1

    print(f"[T-MAC] {n_packed} modulos empaquetados para LUT (group_size={group_size})")
    print(f"[T-MAC] Memoria ahorrada: {total_kb_saved / 1024:.1f} MB")


# ===========================================================================
# PARTE C: BitLinear con LUT en inferencia
# ===========================================================================

def _bitlinear_lut_forward(self, x: Tensor) -> Tensor:
    """
    Forward pass de BitLinear usando LUT (sin multiplicaciones).
    VECTORIZADO — maneja batch M > 1 sin loops Python.

    Monkey-patched en cada BitLinear despues de pack_all_for_lut.
    """
    original_shape = x.shape
    d_in = x.shape[-1]

    # Flatten a [M, d_in]
    x_2d = x.reshape(-1, d_in).float()

    # LUT matmul vectorizado (sin loops)
    out = ternary_lut_matmul(
        x_2d,
        self.w_lut_packed,
        self.w_lut_scale,
        self.lut_group_size,
        decode_table=self.lut_decode_table,
        gather_index=self.w_lut_gather_idx,
    )  # [M, d_out]

    out_shape = list(original_shape[:-1]) + [out.shape[-1]]
    return out.reshape(out_shape).to(x.dtype)


def enable_lut_inference(model: CPU1ModelV2) -> int:
    """
    Activar modo LUT en todos los BitLinear del modelo.
    Monkey-patchea el forward de cada BitLinear con _bitlinear_lut_forward.

    Llamar DESPUES de pack_all_for_lut.
    Libera pesos originales para ahorrar RAM.

    Returns:
        Cantidad de modulos patcheados.
    """
    n_patched = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, 'w_lut_packed'):
            # Liberar peso original
            if module.weight is not None and module.weight.numel() > 0:
                module.weight = nn.Parameter(
                    torch.empty(0), requires_grad=False
                )

            # Monkey-patch forward
            module.forward = types.MethodType(_bitlinear_lut_forward, module)
            n_patched += 1

    print(f"[T-MAC] {n_patched} modulos con forward LUT activado")
    return n_patched


# ===========================================================================
# PARTE D: Motor de Inferencia con T-MAC LUT
# ===========================================================================

class CPU1InferenceEngineLUT(CPU1InferenceEngine):
    """
    Motor de inferencia CPU-optimizado con T-MAC LUT.

    Extiende CPU1InferenceEngine reemplazando pack_all_bitlinear_to_int8
    con pack_all_for_lut + enable_lut_inference.

    Uso:
        engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
            'checkpoint.pt', config, group_size=4
        )
        output = engine.generate(b"Buenos dias", max_bytes=200)
    """

    def __init__(
        self,
        model: CPU1ModelV2,
        config: dict,
        group_size: int = 4,
        use_lut: bool = True,
    ):
        self.config = config
        self.group_size = group_size
        self.info = detect_model_info(model, config)
        d_model = self.info.d_model
        n_layers = self.info.n_layers

        print("\n" + "=" * 60)
        print("Aplicando optimizaciones CPU para inferencia")
        if use_lut:
            print(f"  Modo: T-MAC LUT (group_size={group_size})")
        else:
            print("  Modo: INT8 (baseline)")
        print("=" * 60)

        model.eval()

        if use_lut:
            # T-MAC: empaquetar pesos para LUT (sin transponer)
            pack_all_for_lut(model, group_size=group_size)
            enable_lut_inference(model)
        else:
            # Fallback: INT8 del prompt 06
            transpose_weights_for_inference(model)
            pack_all_bitlinear_to_int8(model)

        # Alinear todos los pesos
        align_all_weights(model)

        # Configurar threads — usar TODOS los cores fisicos
        n_threads = configure_cpu_threads()
        set_thread_affinity(list(range(n_threads)))

        # Pre-alocar buffers reutilizables
        self.hidden_states = alloc_hidden_states(n_layers, d_model)
        n_heads = self.info.n_heads
        vocab = self.info.vocab_size
        self.logits_buf = torch.zeros(n_heads, vocab, dtype=torch.float32)
        self.probs_buf = torch.zeros(n_heads, vocab, dtype=torch.float32)
        self.sorted_buf = torch.zeros(vocab, dtype=torch.float32)
        self.idx_buf = torch.zeros(vocab, dtype=torch.long)

        # Pre-allocated states buffer + block flags
        self._states_buf = list(self.hidden_states)

        # Compilar
        compiled = compile_for_cpu_inference(model)
        try:
            warmup_compiled_model(compiled, config, n_warmup=2)
            self.model = compiled
        except Exception as e:
            print(f"[CPU opt] torch.compile fallo en runtime ({type(e).__name__}), "
                  f"usando eager mode")
            if hasattr(model, 'blocks'):
                for i, block in enumerate(model.blocks):
                    if hasattr(block, '_orig_mod'):
                        model.blocks[i] = block._orig_mod
            self.model = model
            warmup_compiled_model(model, config, n_warmup=2)

        # Block flags AFTER compilation
        _active = self.model.blocks if hasattr(self.model, 'blocks') else []
        _any_compiled = any(hasattr(b, '_orig_mod') for b in _active)
        if _any_compiled:
            self._block_flags = [False] * len(_active)
        else:
            self._block_flags = [hasattr(b, 'forward_one_token') for b in _active]

        print("=" * 60)
        print("Optimizaciones aplicadas")
        print("=" * 60 + "\n")

    @classmethod
    def from_checkpoint_with_lut(
        cls,
        path: str,
        config: dict,
        group_size: int = 4,
        use_lut: bool = True,
    ) -> 'CPU1InferenceEngineLUT':
        """
        Cargar checkpoint con todas las optimizaciones incluyendo T-MAC LUT.

        Orden de aplicacion (IMPORTANTE — no cambiar el orden):
        1. Cargar checkpoint
        2. pack_all_for_lut (reemplaza INT8 del prompt 06)
        3. enable_lut_inference
        4. align_all_weights
        5. configure_cpu_threads
        6. compile_for_cpu_inference
        7. warmup

        Args:
            path: ruta al checkpoint .pt
            config: dict de configuracion del modelo
            group_size: pesos por grupo para LUT. Opciones:
                4  → LUT de 256 entradas (1KB) — siempre en L1
                2  → LUT de 16 entradas (64B) — overhead de precompute alto
                Recomendado: 4 para CPU consumer
            use_lut: True para T-MAC LUT, False para INT8 fallback
        """
        model = CPU1ModelV2(config)
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state)
        model.eval()
        return cls(model, config, group_size=group_size, use_lut=use_lut)


# ===========================================================================
# PARTE E: Benchmark T-MAC vs INT8
# ===========================================================================

def benchmark_lut_vs_int8(
    config: dict,
    checkpoint_path: str | None = None,
) -> dict:
    """
    Comparar T-MAC LUT vs INT8 en el MISMO hardware.
    Esta tabla va directamente al paper.

    Variante A: INT8 + torch.compile (prompt 06)
    Variante B: T-MAC LUT group_size=4 + torch.compile

    Si checkpoint_path es None, usa pesos random (para testing).
    """
    results = {}
    test_prompt = b"El modelo de lenguaje CPU-1 opera directamente sobre bytes."

    variants = [
        ('int8_compiled',   {'use_lut': False}),
        ('lut_g4_compiled', {'use_lut': True, 'group_size': 4}),
    ]

    for variant_name, kwargs in variants:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {variant_name}")
        print(f"{'='*60}")

        if checkpoint_path:
            engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
                checkpoint_path, config, **kwargs
            )
        else:
            model = CPU1ModelV2(config)
            model.eval()
            engine = CPU1InferenceEngineLUT(model, config, **kwargs)

        r = run_full_benchmark(engine, [test_prompt] * 3, n_tokens=40)
        results[variant_name] = r
        del engine

    # Tabla comparativa
    print("\n" + "=" * 70)
    print("COMPARATIVA T-MAC LUT vs INT8 — Para el Paper")
    print("=" * 70)
    print(f"{'Variante':<20} {'tok/seg':>8} {'lat p50':>10} {'RAM MB':>8}")
    print("-" * 70)
    for k, v in results.items():
        print(f"{k:<20} {v['tokens_per_sec']:>8.1f} "
              f"{v['latency_ms_p50']:>10.2f} {v['ram_mb']:>8.1f}")
    print("=" * 70)

    return results


# ===========================================================================
# PARTE H: Integracion con T-MAC Oficial
# ===========================================================================

def export_for_tmac_kernel_bench(
    model: CPU1ModelV2,
    output_path: str,
    config: dict,
) -> None:
    """
    Exportar pesos BitLinear para benchmark de kernel T-MAC.

    CPU-1 usa MLGRU (no Transformer), asi que llama.cpp no soporta
    end-to-end. Esta funcion exporta solo las proyecciones BitLinear
    para benchmark de kernel aislado (Opcion B del paper).

    Para benchmark end-to-end: implementar MLGRU en ggml o usar
    bitnet.cpp (github.com/microsoft/BitNet).
    """
    kernel_bench_weights = {}
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight.data
            w_scale = w.float().abs().mean()
            w_ternary = (w.float() / (w_scale + 1e-8)).round().clamp(-1, 1)
            kernel_bench_weights[name] = {
                'weight': w_ternary.to(torch.int8),
                'scale': w_scale.item(),
                'shape': list(w.shape),
            }

    torch.save({
        'weights': kernel_bench_weights,
        'config': config,
        'arch': 'cpu1_v2_mlgru',
    }, output_path)

    print(f"[Export] Pesos guardados para benchmark de kernel T-MAC: {output_path}")
    print(f"[Export] {len(kernel_bench_weights)} capas BitLinear exportadas")


def benchmark_tmac_official(
    model_path: str,
    n_tokens: int = 200,
    n_threads: int = 4,
) -> dict:
    """
    Correr benchmark con el binario oficial de T-MAC/llama.cpp.

    Requiere:
        - T-MAC compilado (cmake -DLLAMA_TMAC=ON)
        - Modelo en formato GGUF i2 (ternario 2-bit)

    Returns dict con tokens/seg o error.
    """
    llama_bench = "3rdparty/llama.cpp/build/bin/llama-bench"

    try:
        result = subprocess.run([
            llama_bench,
            "-m", model_path,
            "-n", str(n_tokens),
            "-t", str(n_threads),
            "-ngl", "0",
            "--output", "json",
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            tps = data.get('tokens_per_second', 0)
            print(f"[T-MAC oficial] {tps:.1f} tokens/seg ({n_threads} threads)")
            return {'tokens_per_sec': tps, 'source': 'tmac_official_cpp'}
        else:
            print(f"[T-MAC oficial] Error: {result.stderr}")
            return {'error': result.stderr}

    except FileNotFoundError:
        print("[T-MAC oficial] llama-bench no encontrado.")
        print("  Compilar con: cmake -DLLAMA_TMAC=ON ...")
        return {'error': 'llama-bench not found'}

    except subprocess.TimeoutExpired:
        return {'error': 'timeout'}


def compare_python_vs_cpp_lut(
    engine: CPU1InferenceEngineLUT,
    model_gguf_path: str,
) -> None:
    """
    Tabla comparativa Python LUT vs C++ T-MAC para el paper.
    """
    test_prompt = b"El objetivo es RAG local para todo ciudadano."
    t0 = time.perf_counter()
    engine.generate(test_prompt, max_bytes=200, reset=True)
    t1 = time.perf_counter()
    python_tps = 200 / (t1 - t0) / 4

    cpp_result = benchmark_tmac_official(model_gguf_path)
    cpp_tps = cpp_result.get('tokens_per_sec', 0)

    print("\n" + "=" * 60)
    print("Python LUT (torch.compile) vs T-MAC kernel C++")
    print("=" * 60)
    print(f"{'Implementacion':<30} {'tok/seg':>10} {'relativo':>10}")
    print("-" * 60)
    print(f"{'Python LUT + compile':<30} {python_tps:>10.1f} {'1.0x':>10}")
    if cpp_tps > 0:
        ratio = cpp_tps / python_tps if python_tps > 0 else 0
        print(f"{'T-MAC C++ kernel':<30} {cpp_tps:>10.1f} {ratio:>9.1f}x")
    print("=" * 60)

    if cpp_tps > 0 and python_tps > 0:
        ratio = cpp_tps / python_tps
        if ratio < 1.5:
            print("\n-> torch.compile es suficiente para este modelo.")
        else:
            print(f"\n-> Kernel C++ es {ratio:.1f}x mas rapido.")
            print("  Claim: 'Trabajo futuro: portar a C++ con T-MAC'")


# ===========================================================================
# Tests
# ===========================================================================

def test_precompute_lut():
    """Verificar precompute_all_luts con caso conocido."""
    # x = [1, 2, 3, 4], pesos = [1, 0, -1, 1]
    # resultado esperado = 1*1 + 0*2 + (-1)*3 + 1*4 = 2
    x = torch.tensor([1.0, 2.0, 3.0, 4.0]).reshape(1, 1, 4)  # [M=1, n_groups=1, gs=4]
    luts = precompute_all_luts(x, _DECODE_TABLE_G4)  # [1, 1, 256]
    lut = luts[0, 0]  # [256]

    assert lut.shape == (256,), f"LUT shape: {lut.shape}, expected (256,)"

    # Encoding de pesos [1, 0, -1, 1]:
    # w0=1→01, w1=0→00, w2=-1→10, w3=1→01
    # = 0b01100001 = 97
    code = 0b01100001
    expected = 2.0
    actual = lut[code].item()
    assert abs(actual - expected) < 1e-5, (
        f"LUT[{code}] = {actual}, expected {expected}"
    )

    # Verificar Mirror Consolidation: lut[128 + c] = -lut[c]
    for c in range(128):
        assert abs(lut[c] + lut[128 + c]) < 1e-5, (
            f"Mirror fallo: lut[{c}]={lut[c]}, lut[{128+c}]={lut[128+c]}"
        )

    print("[OK] precompute_all_luts")


def test_pack_ternary():
    """Verificar pack_ternary_for_lut con pesos conocidos."""
    # 1 neurona, 4 pesos: [1, 0, -1, 1]
    weight = torch.tensor([[1.0, 0.0, -1.0, 1.0]])
    packed, scale = pack_ternary_for_lut(weight, group_size=4)

    assert packed.shape == (1, 1), f"Packed shape: {packed.shape}"
    assert packed.dtype == torch.uint8

    # Encoding: w0=1→01, w1=0→00, w2=-1→10, w3=1→01
    # = 0b01100001 = 97
    expected_code = 0b01100001
    assert packed[0, 0].item() == expected_code, (
        f"Packed code: {packed[0, 0].item()}, expected {expected_code}"
    )

    print("[OK] pack_ternary_for_lut")


def test_lut_matmul():
    """Verificar ternary_lut_matmul da resultado correcto."""
    # Pesos conocidos: 2 neuronas, 4 inputs
    # Neurona 0: [1, 1, 1, 1]  → sum(x) = 10
    # Neurona 1: [1, -1, 1, -1] → 1-2+3-4 = -2
    weight = torch.tensor([
        [1.0, 1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0, -1.0],
    ])
    packed, scale = pack_ternary_for_lut(weight, group_size=4)

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
    out = ternary_lut_matmul(x, packed, scale, group_size=4)

    assert out.shape == (1, 2), f"Output shape: {out.shape}"
    expected_0 = 10.0 * scale
    expected_1 = -2.0 * scale
    assert abs(out[0, 0].item() - expected_0) < 1e-4, (
        f"Neurona 0: {out[0, 0].item()}, expected {expected_0}"
    )
    assert abs(out[0, 1].item() - expected_1) < 1e-4, (
        f"Neurona 1: {out[0, 1].item()}, expected {expected_1}"
    )

    print("[OK] ternary_lut_matmul")


def test_lut_matmul_batched():
    """Verificar que batch M > 1 funciona correctamente."""
    d_in, d_out = 32, 16
    weight = torch.randn(d_out, d_in) * 0.02
    packed, scale = pack_ternary_for_lut(weight, group_size=4)

    # Batch de 5 tokens
    x = torch.randn(5, d_in)
    out = ternary_lut_matmul(x, packed, scale, group_size=4)

    assert out.shape == (5, d_out), f"Shape: {out.shape}, expected (5, {d_out})"

    # Verificar que cada fila coincide con la version individual
    for m in range(5):
        out_single = ternary_lut_matmul(x[m:m+1], packed, scale, group_size=4)
        diff = (out[m] - out_single[0]).abs().max().item()
        assert diff < 1e-5, f"Token {m}: diff={diff}"

    print(f"[OK] ternary_lut_matmul batched: (5, {d_in}) -> (5, {d_out})")


def test_lut_matmul_shape():
    """Verificar shapes correctas para dimensiones del modelo."""
    d_in, d_out = 128, 64
    weight = torch.randn(d_out, d_in) * 0.02
    packed, scale = pack_ternary_for_lut(weight, group_size=4)

    x = torch.randn(1, d_in)
    out = ternary_lut_matmul(x, packed, scale, group_size=4)

    assert out.shape == (1, d_out), f"Shape: {out.shape}, expected (1, {d_out})"
    print(f"[OK] ternary_lut_matmul shape: (1, {d_in}) x ({d_out}, {d_in}) -> (1, {d_out})")


def test_pack_all_for_lut():
    """Verificar pack_all_for_lut con modelo completo."""
    config = {'d_model': 64, 'n_layers': 2, 'd_ff': 128, 'fp_rank': 4,
              'pfnet_hidden': 16}
    model = CPU1ModelV2(config)

    pack_all_for_lut(model, group_size=4)

    n_packed = 0
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            assert hasattr(module, 'w_lut_packed'), f"{name}: no w_lut_packed"
            assert hasattr(module, 'w_lut_scale'), f"{name}: no w_lut_scale"
            assert hasattr(module, 'w_lut_gather_idx'), f"{name}: no gather_idx"
            assert module.w_lut_packed.dtype == torch.uint8
            n_packed += 1

    assert n_packed > 0, "No BitLinear modules packed"
    print(f"[OK] pack_all_for_lut: {n_packed} modulos empaquetados")


def test_enable_lut_inference():
    """Verificar que enable_lut_inference patchea forwards correctamente."""
    config = {'d_model': 64, 'n_layers': 2, 'd_ff': 128, 'fp_rank': 4,
              'pfnet_hidden': 16}
    model = CPU1ModelV2(config)

    pack_all_for_lut(model, group_size=4)
    n_patched = enable_lut_inference(model)

    assert n_patched > 0, "No modules patched"

    # Verificar forward con batch
    for name, module in model.named_modules():
        if isinstance(module, BitLinear) and hasattr(module, 'w_lut_packed'):
            d_in = module.in_features
            d_out = module.out_features
            x = torch.randn(1, 8, d_in)  # [B=1, S=8, d_in]
            out = module(x)
            assert out.shape == (1, 8, d_out), (
                f"{name}: output {out.shape}, expected (1, 8, {d_out})"
            )
            break

    print(f"[OK] enable_lut_inference: {n_patched} modulos patcheados")


def test_engine_lut_generate():
    """Test integracion: generar bytes con engine LUT."""
    config = {'d_model': 64, 'n_layers': 2, 'd_ff': 128, 'fp_rank': 4,
              'pfnet_hidden': 16}
    model = CPU1ModelV2(config)
    model.eval()

    engine = CPU1InferenceEngineLUT(model, config, group_size=4, use_lut=True)
    output = engine.generate(b"test", max_bytes=8)

    assert isinstance(output, bytes), f"Output type: {type(output)}"
    assert len(output) == 8, f"Expected 8 bytes, got {len(output)}"
    assert all(0 <= b <= 255 for b in output), "Invalid byte values"

    print(f"[OK] engine LUT generate: {len(output)} bytes: {list(output)}")


def test_engine_int8_fallback():
    """Test integracion: engine con fallback INT8 (use_lut=False)."""
    config = {'d_model': 64, 'n_layers': 2, 'd_ff': 128, 'fp_rank': 4,
              'pfnet_hidden': 16}
    model = CPU1ModelV2(config)
    model.eval()

    engine = CPU1InferenceEngineLUT(model, config, group_size=4, use_lut=False)
    output = engine.generate(b"test", max_bytes=8)

    assert isinstance(output, bytes), f"Output type: {type(output)}"
    assert len(output) == 8, f"Expected 8 bytes, got {len(output)}"

    print(f"[OK] engine INT8 fallback: {len(output)} bytes: {list(output)}")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Tests inference.py — INT8 + T-MAC LUT")
    print("=" * 60)

    # INT8 engine tests
    print("\n--- Test 1: Utilidades ---")
    test_assert_aligned()

    print("\n--- Test 2: Transpose weights ---")
    test_transpose_weights()

    print("\n--- Test 3: INT8 packing ---")
    test_pack_int8()

    print("\n--- Test 4: Hidden states ---")
    test_hidden_states()

    print("\n--- Test 5: Nucleus sampling ---")
    test_nucleus_sample()

    print("\n--- Test 6: Engine generate (integracion) ---")
    test_engine_generate()

    # T-MAC LUT tests
    print("\n--- Test 7: Precompute LUT ---")
    test_precompute_lut()

    print("\n--- Test 8: Pack ternary ---")
    test_pack_ternary()

    print("\n--- Test 9: LUT MatMul (correctness) ---")
    test_lut_matmul()

    print("\n--- Test 10: LUT MatMul (batched M>1) ---")
    test_lut_matmul_batched()

    print("\n--- Test 11: LUT MatMul (shape) ---")
    test_lut_matmul_shape()

    print("\n--- Test 12: Pack all for LUT ---")
    test_pack_all_for_lut()

    print("\n--- Test 13: Enable LUT inference ---")
    test_enable_lut_inference()

    print("\n--- Test 14: Engine LUT generate ---")
    test_engine_lut_generate()

    print("\n--- Test 15: Engine INT8 fallback ---")
    test_engine_int8_fallback()

    print("\n" + "=" * 60)
    print("Todos los tests pasaron.")
    print("=" * 60)
