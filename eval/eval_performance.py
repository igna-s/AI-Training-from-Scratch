"""
Eval — Performance: throughput, latencia, jitter, scaling, RAM.

Adaptivo segun arquitectura:
  - MLGRU: usa forward_with_prefetch con estados
  - Transformer: forward_with_prefetch con None states
  - Byte/BPE: dummy input adaptado al formato

Tests 06, 07, 08 fusionados en un solo benchmark run.
Test 22 (CPU raw frozen) eliminado — el engine ya tiene pesos frozen.
"""
import os
import statistics
import time

import torch

from helpers import get_model, forward_with_prefetch


def _make_single_step_input(info):
    """Crear input dummy de 1 paso para decode benchmark."""
    if info.is_byte_level:
        return torch.zeros(1, 1, info.patch_size, dtype=torch.long)
    else:
        return torch.zeros(1, 1, dtype=torch.long)


def _embed_and_squeeze(model, engine, dummy):
    """Embed + squeeze adaptado al modelo."""
    embed_fn = engine._get_embedding(model)
    x = embed_fn(dummy)
    if x.dim() == 3:
        x = x.squeeze(1)
    return x


def _benchmark_decode(engine, n_steps: int = 500,
                      n_warmup: int = 20) -> list[float]:
    """Medir latencia por paso usando forward_with_prefetch."""
    model = get_model(engine)
    dummy = _make_single_step_input(engine.info)

    with torch.inference_mode():
        # Reset state dentro de inference_mode
        engine.reset_state()

        for _ in range(n_warmup):
            x = _embed_and_squeeze(model, engine, dummy)
            x, engine.hidden_states = forward_with_prefetch(
                model, x, engine.hidden_states
            )

        # Reset state para benchmark limpio
        engine.reset_state()
        times = []
        for _ in range(n_steps):
            t0 = time.perf_counter()
            x = _embed_and_squeeze(model, engine, dummy)
            x, engine.hidden_states = forward_with_prefetch(
                model, x, engine.hidden_states
            )
            times.append(time.perf_counter() - t0)

    return times


def check_throughput_latency_jitter(engine) -> dict:
    """Tests 06 + 07 + 08 fusionados."""
    times = _benchmark_decode(engine, n_steps=500, n_warmup=20)

    median_time = sorted(times)[len(times) // 2]
    tps = 1.0 / median_time if median_time > 0 else 0

    times_ms = sorted([t * 1000 for t in times])
    n = len(times_ms)
    p50 = times_ms[n // 2]
    p95 = times_ms[int(n * 0.95)]
    p99 = times_ms[int(n * 0.99)]

    mean_t = statistics.mean(times)
    std_t = statistics.stdev(times)
    cv = std_t / mean_t if mean_t > 0 else 0

    psize = engine.info.patch_size if engine.info.is_byte_level else 1
    bps = tps * psize

    print(f"    Throughput: {tps:.1f} steps/s ({bps:.1f} bytes/s)")
    print(f"    Latency:   p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    print(f"    Jitter:    CV={cv:.3f}")

    assert tps > 10, f"Throughput muy bajo: {tps:.1f} steps/s"
    assert p50 < 100, f"p50 muy alto: {p50:.1f}ms"
    assert cv < 0.3, f"Jitter muy alto: CV={cv:.3f}"

    return {
        'throughput_sps': round(tps, 1),
        'throughput_bps': round(bps, 1),
        'p50_ms': round(p50, 2),
        'p95_ms': round(p95, 2),
        'p99_ms': round(p99, 2),
        'jitter_cv': round(cv, 4),
    }


def check_context_scaling(engine) -> dict:
    """Throughput constante con contextos largos (MLGRU = O(1), Transformer ≠)."""
    info = engine.info
    model = get_model(engine)
    ctx_lengths = [0, 512, 2048]
    results = {}

    with torch.inference_mode():
        for ctx_len in ctx_lengths:
            engine.reset_state()
            if ctx_len > 0:
                if info.is_byte_level:
                    ctx = torch.zeros(1, ctx_len, info.patch_size, dtype=torch.long)
                else:
                    ctx = torch.zeros(1, ctx_len, dtype=torch.long)
                result = model(ctx, engine.hidden_states)
                engine.hidden_states = result[1]

            dummy = _make_single_step_input(info)
            t0 = time.perf_counter()
            for _ in range(100):
                x = _embed_and_squeeze(model, engine, dummy)
                x, engine.hidden_states = forward_with_prefetch(
                    model, x, engine.hidden_states
                )
            tps = 100 / (time.perf_counter() - t0)
            results[ctx_len] = round(tps, 1)
            print(f"    ctx={ctx_len:5d}: {tps:.1f} steps/s")

    baseline = results[0]
    max_degradation = 0.2 if info.is_recurrent else 0.5
    for ctx_len, tps in results.items():
        if baseline > 0 and ctx_len > 0:
            degradation = (baseline - tps) / baseline
            assert degradation < max_degradation, \
                f"Throughput degrada {degradation * 100:.1f}% con ctx={ctx_len}"

    return results


def check_ram(engine) -> dict:
    """RAM delta durante inferencia (sin memory leaks)."""
    import psutil
    process = psutil.Process(os.getpid())

    model = get_model(engine)
    ram_before = process.memory_info().rss / 1e6

    with torch.inference_mode():
        engine.reset_state()
        dummy = _make_single_step_input(engine.info)
        for _ in range(100):
            x = _embed_and_squeeze(model, engine, dummy)
            x, engine.hidden_states = forward_with_prefetch(
                model, x, engine.hidden_states
            )

    ram_after = process.memory_info().rss / 1e6
    ram_delta = ram_after - ram_before

    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"    RAM modelo: {model_mb:.1f}MB | delta: {ram_delta:.1f}MB")
    assert ram_delta < 50, f"Memory leak: {ram_delta:.1f}MB durante inferencia"

    return {'model_mb': round(model_mb, 1), 'delta_mb': round(ram_delta, 1),
            'total_mb': round(ram_after, 1)}


# =========================================================================
# Entry point
# =========================================================================

def run_performance(engine) -> dict:
    """Ejecutar todos los checks de performance."""
    from helpers import run_check

    print(f"\n--- Performance ({engine.info.model_class}) ---")
    results = {}
    checks = [
        ('throughput_latency_jitter', check_throughput_latency_jitter),
        ('context_scaling',           check_context_scaling),
        ('ram',                       check_ram),
    ]
    for name, fn in checks:
        r = run_check(name, fn, engine)
        results[name] = r
        print(f"  {r['status']:5s}  {name}")

    return results
