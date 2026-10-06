"""
Eval — Performance: throughput, latencia, jitter, scaling, RAM.

Soporta dos modos:
  Produccion (CPU1InferenceEngine): embed → forward_with_prefetch (puede usar kernels)
  Ablacion  (AblationEngine, _is_ablation=True):
      forward completo PyTorch puro — sin kernels customizados nuestros.
      Esto garantiza la comparativa limpia pedida por el usuario.

Tests 06, 07, 08 fusionados en un solo benchmark run.
"""
import os
import statistics
import time

import torch

from helpers import get_model, forward_with_prefetch


def _make_single_step_input(info):
    """Input dummy de 1 paso para decode benchmark."""
    if info.is_byte_level:
        return torch.zeros(1, 1, info.patch_size, dtype=torch.long)
    else:
        return torch.zeros(1, 1, dtype=torch.long)


def _embed_and_squeeze(model, engine, dummy):
    """Embed + squeeze para modo produccion."""
    embed_fn = engine._get_embedding(model)
    x = embed_fn(dummy)
    if x.dim() == 3:
        x = x.squeeze(1)
    return x


def _benchmark_decode(engine, n_steps: int = 500,
                      n_warmup: int = 20) -> list[float]:
    """
    Medir latencia por paso.

    Modo ablacion (_is_ablation=True):
        Usa engine._ablation_step() — PyTorch puro, sin kernels customizados.
        Para MLGRU lleva estado entre pasos (realista).
        Para Transformer no lleva estado.

    Modo produccion:
        Usa forward_with_prefetch (puede usar kernels optimizados).
    """
    model = get_model(engine)
    dummy = _make_single_step_input(engine.info)
    is_ablation = getattr(engine, '_is_ablation', False)

    with torch.inference_mode():
        engine.reset_state()

        if is_ablation:
            # ── Ablacion: PyTorch puro ──────────────────────────────────
            for _ in range(n_warmup):
                _, h_new = engine._ablation_step(model, dummy)
                engine.hidden_states = h_new

            engine.reset_state()
            times = []
            for _ in range(n_steps):
                t0 = time.perf_counter()
                _, h_new = engine._ablation_step(model, dummy)
                engine.hidden_states = h_new
                times.append(time.perf_counter() - t0)
        else:
            # ── Produccion: embed + forward_with_prefetch ───────────────
            for _ in range(n_warmup):
                x = _embed_and_squeeze(model, engine, dummy)
                x, engine.hidden_states = forward_with_prefetch(
                    model, x, engine.hidden_states
                )

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

    backend = "pytorch_pure" if getattr(engine, '_is_ablation', False) else "inference_engine"
    print(f"    Throughput: {tps:.1f} steps/s ({bps:.1f} bytes/s)  [{backend}]")
    print(f"    Latency:   p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    print(f"    Jitter:    CV={cv:.3f}")

    assert tps > 1, f"Throughput muy bajo: {tps:.1f} steps/s"
    assert p50 < 5000, f"p50 muy alto: {p50:.1f}ms"
    assert cv < 0.5, f"Jitter muy alto: CV={cv:.3f}"

    return {
        'throughput_sps':  round(tps, 1),
        'throughput_bps':  round(bps, 1),
        'p50_ms':          round(p50, 2),
        'p95_ms':          round(p95, 2),
        'p99_ms':          round(p99, 2),
        'jitter_cv':       round(cv, 4),
        'backend':         backend,
    }


def check_context_scaling(engine) -> dict:
    """Throughput constante con contextos largos (MLGRU = O(1), Transformer ≠)."""
    info = engine.info
    model = get_model(engine)
    ctx_lengths = [0, 512, 2048]
    results = {}
    is_ablation = getattr(engine, '_is_ablation', False)
    if is_ablation and not info.is_recurrent:
        return {'skipped': True, 'reason': 'Transformer ablation sin KV cache: el test de pasos aislados no conserva contexto'}

    with torch.inference_mode():
        for ctx_len in ctx_lengths:
            engine.reset_state()
            if ctx_len > 0:
                if info.is_byte_level:
                    ctx = torch.zeros(1, ctx_len, info.patch_size, dtype=torch.long)
                else:
                    ctx = torch.zeros(1, ctx_len, dtype=torch.long)

                if is_ablation:
                    _, h_new = engine._ablation_step(model, ctx) if ctx_len == 1 else \
                               (None, model(ctx, hidden_states=None)[1])
                    engine.hidden_states = h_new if info.is_recurrent else None
                else:
                    result = model(ctx, engine.hidden_states)
                    engine.hidden_states = result[1]

            dummy = _make_single_step_input(info)
            t0 = time.perf_counter()
            for _ in range(100):
                if is_ablation:
                    _, h_new = engine._ablation_step(model, dummy)
                    engine.hidden_states = h_new
                else:
                    x = _embed_and_squeeze(model, engine, dummy)
                    x, engine.hidden_states = forward_with_prefetch(
                        model, x, engine.hidden_states
                    )
            tps = 100 / (time.perf_counter() - t0)
            results[ctx_len] = round(tps, 1)
            print(f"    ctx={ctx_len:5d}: {tps:.1f} steps/s")

    baseline = results[0]
    max_degradation = 0.20 if info.is_recurrent else 0.80
    for ctx_len, tps in results.items():
        if baseline > 0 and ctx_len > 0:
            degradation = (baseline - tps) / baseline
            assert degradation < max_degradation, \
                f"Throughput degrada {degradation * 100:.1f}% con ctx={ctx_len} " \
                f"(limite {'20' if info.is_recurrent else '80'}%)"

    return {'throughput_by_ctx': results, 'baseline_sps': baseline}


def check_ram(engine) -> dict:
    """RAM delta durante inferencia (sin memory leaks)."""
    import psutil
    process = psutil.Process(os.getpid())
    model = get_model(engine)
    is_ablation = getattr(engine, '_is_ablation', False)

    ram_before = process.memory_info().rss / 1e6

    with torch.inference_mode():
        engine.reset_state()
        dummy = _make_single_step_input(engine.info)
        for _ in range(100):
            if is_ablation:
                _, h_new = engine._ablation_step(model, dummy)
                engine.hidden_states = h_new
            else:
                x = _embed_and_squeeze(model, engine, dummy)
                x, engine.hidden_states = forward_with_prefetch(
                    model, x, engine.hidden_states
                )

    ram_after = process.memory_info().rss / 1e6
    ram_delta = ram_after - ram_before
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"    RAM modelo: {model_mb:.1f}MB | delta: {ram_delta:.1f}MB")
    assert ram_delta < 200, f"Memory leak: {ram_delta:.1f}MB durante inferencia"

    return {'model_mb': round(model_mb, 1), 'delta_mb': round(ram_delta, 1),
            'total_mb': round(ram_after, 1)}


# =========================================================================
# Entry point
# =========================================================================

def run_performance(engine) -> dict:
    """Ejecutar todos los checks de performance."""
    from helpers import run_check

    backend = "pytorch_pure" if getattr(engine, '_is_ablation', False) else "inference_engine"
    print(f"\n--- Performance ({engine.info.model_class}) [{backend}] ---")
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
