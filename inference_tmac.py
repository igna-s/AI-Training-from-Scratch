"""
CPU-1 v2: Inferencia T-MAC — Reemplazar BitLinear MatMul con LUT.

Basado en T-MAC (Wei et al., EuroSys 2025, arXiv:2407.00088).
Reemplaza las multiplicaciones de las proyecciones BitLinear con
Lookup Tables (LUT) precomputadas.

Para pesos ternarios {-1, 0, 1}, la LUT elimina toda multiplicacion:
  w = 0  → resultado = 0
  w = 1  → resultado = x
  w = -1 → resultado = -x

Solo lookup + suma. Sin MatMul.

No modifica model_v2.py ni el training. Solo cambia como se ejecutan
los pesos ternarios en inferencia.

Implementacion COMPLETAMENTE VECTORIZADA — sin loops Python en el hot path.
Compatible con torch.compile (no graph breaks).

Uso:
    engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
        'checkpoint.pt', config, group_size=4
    )
    output = engine.generate(b"Buenos dias", max_bytes=200)
"""

import os
import subprocess
import time
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from model_v2 import (
    CPU1ModelV2,
    CPU1BlockV2,
    BitLinear,
    PATCH_SIZE,
    VOCAB_SIZE,
    D_MODEL,
    N_LAYERS,
)
from inference_v2 import (
    CPU1InferenceEngine,
    assert_aligned,
    make_aligned,
    align_all_weights,
    alloc_hidden_states,
    configure_cpu_threads,
    set_thread_affinity,
    compile_for_cpu_inference,
    warmup_compiled_model,
    forward_with_prefetch,
    pack_all_bitlinear_to_int8,
    transpose_weights_for_inference,
    run_full_benchmark,
    _nucleus_sample,
)


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
        d_model = config.get('d_model', D_MODEL)
        n_layers = config.get('n_layers', N_LAYERS)

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
        self.logits_buf = torch.zeros(4, VOCAB_SIZE, dtype=torch.float32)
        self.probs_buf = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        self.sorted_buf = torch.zeros(VOCAB_SIZE, dtype=torch.float32)
        self.idx_buf = torch.zeros(VOCAB_SIZE, dtype=torch.long)

        # Compilar
        compiled = compile_for_cpu_inference(model)
        try:
            warmup_compiled_model(compiled, config, n_warmup=2)
            self.model = compiled
        except Exception as e:
            print(f"[CPU opt] torch.compile fallo en runtime ({type(e).__name__}), "
                  f"usando eager mode")
            self.model = model
            warmup_compiled_model(model, config, n_warmup=2)

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
    print("Tests inference_tmac.py — T-MAC LUT Integration")
    print("=" * 60)

    print("\n--- Test 1: Precompute LUT ---")
    test_precompute_lut()

    print("\n--- Test 2: Pack ternary ---")
    test_pack_ternary()

    print("\n--- Test 3: LUT MatMul (correctness) ---")
    test_lut_matmul()

    print("\n--- Test 4: LUT MatMul (batched M>1) ---")
    test_lut_matmul_batched()

    print("\n--- Test 5: LUT MatMul (shape) ---")
    test_lut_matmul_shape()

    print("\n--- Test 6: Pack all for LUT ---")
    test_pack_all_for_lut()

    print("\n--- Test 7: Enable LUT inference ---")
    test_enable_lut_inference()

    print("\n--- Test 8: Engine LUT generate ---")
    test_engine_lut_generate()

    print("\n--- Test 9: Engine INT8 fallback ---")
    test_engine_int8_fallback()

    print("\n" + "=" * 60)
    print("Todos los tests pasaron.")
    print("=" * 60)
