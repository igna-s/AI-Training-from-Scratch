#!/usr/bin/env python3
"""
benchmark_packed.py — Benchmark realista con checkpoint simulado entrenado.

3 modos comparables:
  int8_python  — INT8 packing + Python layer loop (baseline)
  int8_fused   — INT8 packing + decode_layer_fused (1 ctypes call/capa)
  2bit_fused   — 2-bit packing (41MB RAM) + decode_layer_fused

Uso:
    python inference/benchmark_packed.py                   # 170M, todos los modos
    python inference/benchmark_packed.py --mode int8_fused # solo un modo
    python inference/benchmark_packed.py --d_model 512 --n_layers 12  # 50M ablacion
"""

import sys
import time
import argparse
import ctypes
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import inference.inference as _inf_mod
_inf_mod.USE_C_KERNELS = True

from inference.inference import (
    _load_c_kernel,
    transpose_weights_for_inference,
    align_all_weights,
    pack_all_bitlinear_to_int8,
    pack_all_bitlinear_to_2bit,
    patch_activations_for_inference,
    prepare_fused_layer_data,
    prepare_fused_layer_data_2bit,
    forward_with_fused_layers,
    forward_with_fused_layers_2bit,
    forward_with_prefetch,
    alloc_hidden_states,
    configure_cpu_threads,
    _nucleus_sample,
)
from model import CPU1ModelV2, BitLinear


# ── Cuantizar pesos a ternario (simular entrenamiento) ───────────────────────
def quantize_model_to_ternary(model: nn.Module) -> int:
    n = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, BitLinear):
                w = module.weight.float()
                scale = w.abs().mean().clamp(min=1e-8)
                module.weight.data = (w / scale).round().clamp(-1, 1).to(module.weight.dtype)
                n += 1
    return n


# ── Helpers comunes ──────────────────────────────────────────────────────────
def _reset_states(n_layers, d_model):
    return alloc_hidden_states(n_layers, d_model, verbose=False)


def _run_benchmark(run_patch_fn, reset_fn, n_patches=300, n_warmup=10) -> dict:
    reset_fn()
    print(f"  Warmup {n_warmup} pasos...", end=" ", flush=True)
    for _ in range(n_warmup):
        run_patch_fn()
    print("OK")

    reset_fn()
    times = []
    for _ in range(n_patches):
        t0 = time.perf_counter()
        run_patch_fn()
        times.append(time.perf_counter() - t0)

    times_s = sorted(times)
    n = len(times_s)
    avg_ms = sum(times) / n * 1000
    return {
        "patches_per_sec": 1000 / avg_ms,
        "bytes_per_sec":   4000 / avg_ms,
        "avg_ms":  avg_ms,
        "p50_ms":  times_s[n // 2]     * 1000,
        "p95_ms":  times_s[int(n*.95)] * 1000,
    }


# ── MODO 1: int8_python — baseline actual ────────────────────────────────────
def bench_int8_python(config, n_patches, n_warmup):
    print("\n[Mode: int8_python — INT8 + Python loop]")
    model = CPU1ModelV2(config).eval()
    n_q = quantize_model_to_ternary(model)
    print(f"  {n_q} capas cuantizadas")

    transpose_weights_for_inference(model)
    align_all_weights(model)
    pack_all_bitlinear_to_int8(model)
    patch_activations_for_inference(model)
    configure_cpu_threads(config.get("n_threads"))

    d_model = config["d_model"]
    n_layers = config["n_layers"]
    dummy   = torch.zeros(1, 1, 4, dtype=torch.long)
    hs      = _reset_states(n_layers, d_model)
    sb      = list(hs)
    bf      = [hasattr(b, 'forward_one_token') for b in model.blocks]

    def embed_once():
        return model.patch_embedding(dummy).squeeze(1)

    def run():
        nonlocal hs, sb
        x = embed_once()
        x, hs = forward_with_prefetch(model, x, hs, _block_flags=bf, _states_buf=sb)

    def reset():
        nonlocal hs, sb
        hs = _reset_states(n_layers, d_model)
        sb = list(hs)

    return _run_benchmark(run, reset, n_patches, n_warmup)


# ── MODO 2: int8_fused — INT8 + decode_layer_fused ───────────────────────────
def bench_int8_fused(config, n_patches, n_warmup):
    print("\n[Mode: int8_fused — INT8 + decode_layer_fused (C loop)]")
    lib = _load_c_kernel()
    if not lib or not hasattr(lib, 'decode_layer_fused'):
        print("  decode_layer_fused no disponible en kernels.dll — saltando")
        return None

    model = CPU1ModelV2(config).eval()
    quantize_model_to_ternary(model)
    transpose_weights_for_inference(model)
    align_all_weights(model)
    pack_all_bitlinear_to_int8(model)
    patch_activations_for_inference(model)
    configure_cpu_threads(config.get("n_threads"))

    d_model = config["d_model"]
    d_ff    = config["d_ff"]
    n_layers = config["n_layers"]
    dummy   = torch.zeros(1, 1, 4, dtype=torch.long)
    layer_data = prepare_fused_layer_data(model)
    hs = _reset_states(n_layers, d_model)
    sb = list(hs)

    # Verificar RAM usada por pesos int8
    int8_mb = sum(
        m.weight_int8_gemv.numel()
        for _, m in model.named_modules()
        if hasattr(m, 'weight_int8_gemv')
    ) / 1e6
    print(f"  Pesos int8 en RAM: {int8_mb:.1f} MB")

    def run():
        nonlocal hs, sb
        x = model.patch_embedding(dummy).squeeze(1)
        x, hs = forward_with_fused_layers(
            model, x, hs, layer_data, d_model, d_ff, sb
        )

    def reset():
        nonlocal hs, sb
        hs = _reset_states(n_layers, d_model)
        sb = list(hs)

    return _run_benchmark(run, reset, n_patches, n_warmup)


# ── MODO 3: 2bit_fused — 2-bit + decode_layer_fused ─────────────────────────
def bench_2bit_fused(config, n_patches, n_warmup):
    print("\n[Mode: 2bit_fused -- 2-bit packed (41MB RAM) + decode_layer_fused_2bit]")
    lib = _load_c_kernel()
    if not lib or not hasattr(lib, 'decode_layer_fused_2bit'):
        print("  decode_layer_fused_2bit no disponible -- asegurate de recompilar kernels.dll")
        return None

    model = CPU1ModelV2(config).eval()
    quantize_model_to_ternary(model)
    transpose_weights_for_inference(model)
    align_all_weights(model)
    scales = pack_all_bitlinear_to_2bit(model)
    # patch_activations para que FPResidual y PFNet tengan sus buffers
    patch_activations_for_inference(model)
    configure_cpu_threads(config.get("n_threads"))

    d_model  = config["d_model"]
    d_ff     = config["d_ff"]
    n_layers = config["n_layers"]
    dummy    = torch.zeros(1, 1, 4, dtype=torch.long)
    layer_data = prepare_fused_layer_data_2bit(model)
    hs  = _reset_states(n_layers, d_model)
    sb  = list(hs)

    bit2_mb = sum(
        m.weight_2bit.numel()
        for _, m in model.named_modules()
        if hasattr(m, 'weight_2bit')
    ) / 1e6
    print(f"  Pesos 2-bit en RAM: {bit2_mb:.1f} MB")

    def run():
        nonlocal hs, sb
        x = model.patch_embedding(dummy).squeeze(1)
        x, hs = forward_with_fused_layers_2bit(x, hs, layer_data, d_model, d_ff, sb)

    def reset():
        nonlocal hs, sb
        hs = _reset_states(n_layers, d_model)
        sb = list(hs)

    return _run_benchmark(run, reset, n_patches, n_warmup)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model",   type=int, default=1024)
    parser.add_argument("--n_layers",  type=int, default=14)
    parser.add_argument("--d_ff",      type=int, default=0)
    parser.add_argument("--fp_rank",   type=int, default=8)
    parser.add_argument("--pfnet",     type=int, default=64)
    parser.add_argument("--n_patches", type=int, default=200)
    parser.add_argument("--warmup",    type=int, default=8)
    parser.add_argument("--mode",      type=str, default="all",
                        choices=["all", "int8_python", "int8_fused", "2bit_fused"])
    args = parser.parse_args()

    d_ff = args.d_ff if args.d_ff > 0 else int(args.d_model * 8 / 3 // 128 * 128)
    config = {
        "d_model": args.d_model, "n_layers": args.n_layers,
        "d_ff": d_ff, "fp_rank": args.fp_rank, "pfnet_hidden": args.pfnet,
    }

    # Contar params
    _m = CPU1ModelV2(config)
    n_params = sum(p.numel() for p in _m.parameters()) / 1e6
    del _m

    print("=" * 65)
    print(f"CPU-1 Benchmark — {n_params:.1f}M params — d_model={args.d_model} n_layers={args.n_layers}")
    print("=" * 65)

    results = {}
    modes = ["int8_python", "int8_fused", "2bit_fused"] if args.mode == "all" else [args.mode]

    for mode in modes:
        if mode == "int8_python":
            r = bench_int8_python(config, args.n_patches, args.warmup)
        elif mode == "int8_fused":
            r = bench_int8_fused(config, args.n_patches, args.warmup)
        elif mode == "2bit_fused":
            r = bench_2bit_fused(config, args.n_patches, args.warmup)
        if r:
            results[mode] = r

    # Tabla comparativa
    print("\n" + "=" * 65)
    print("RESUMEN COMPARATIVO")
    print("=" * 65)
    print(f"{'Modo':<20} {'Patches/seg':>12} {'Bytes/seg':>11} {'p50 ms':>9} {'p95 ms':>9}")
    print("-" * 65)
    for mode, r in results.items():
        print(f"{mode:<20} {r['patches_per_sec']:>11.1f}  {r['bytes_per_sec']:>10.0f}"
              f"  {r['p50_ms']:>8.2f}  {r['p95_ms']:>8.2f}")

    ref_pps = 41.0  # LFM2-350M Q4 medido
    print(f"\n{'LFM2-350M Q4 (ref)':<20} {ref_pps:>11.1f}")

    if results:
        best_mode = max(results, key=lambda m: results[m]['patches_per_sec'])
        best_pps  = results[best_mode]['patches_per_sec']
        print(f"\nMejor modo: {best_mode} -> {best_pps:.1f} patches/seg "
              f"({best_pps/ref_pps:.1f}x vs LFM2)")

        if args.mode == "all" and len(results) > 1:
            modes_list = list(results.keys())
            if "int8_python" in results and "int8_fused" in results:
                speedup = results["int8_fused"]["patches_per_sec"] / results["int8_python"]["patches_per_sec"]
                print(f"  int8_fused vs int8_python: {speedup:.2f}× (Python overhead eliminado)")

    print("=" * 65)


if __name__ == "__main__":
    main()
