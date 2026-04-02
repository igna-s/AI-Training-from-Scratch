"""
Test 09 — Throughput vs Longitud de Contexto

Verificar que el throughput NO degrada con contextos largos.
MLGRU deberia ser constante en memoria con cualquier longitud.
Un Transformer degradaria por el KV cache.

Metrica: tokens/seg para prefill lengths [0, 512, 2048, 8192]
Criterio: variacion < 20% entre el contexto corto y largo

Concurrencia: Prefill de los 4 contextos en paralelo con ThreadPoolExecutor
(la fase costosa de preparacion). El benchmark de generacion se ejecuta
estrictamente secuencial para mediciones de timing precisas.
"""
import os
import time
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_throughput_vs_context(model):
    model.eval()
    model.cpu()

    ctx_lengths = [0, 512, 2048, 8192]

    # --- Fase 1: Prefill en paralelo (no es benchmark de timing) ---
    def _prefill(ctx_len):
        if ctx_len > 0:
            ctx = torch.zeros(1, ctx_len, 4, dtype=torch.uint8)
            with torch.inference_mode():
                _, states = model(ctx)
            return states
        return None

    logical = os.cpu_count() or 4
    intra = torch.get_num_threads() or max(1, logical // 2)
    n_workers = max(1, logical // intra)

    prefilled = {}
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_prefill, c): c for c in ctx_lengths}
        for future in as_completed(futures):
            ctx = futures[future]
            prefilled[ctx] = future.result()

    # --- Fase 2: Benchmark secuencial (timing preciso, sin contention) ---
    results = {}
    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)

    for ctx_len in ctx_lengths:
        states = prefilled[ctx_len]
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(100):
                _, states = model(dummy, states)
        tps = 100 / (time.perf_counter() - t0)
        results[ctx_len] = tps
        print(f"  ctx={ctx_len:5d}: {tps:.1f} tok/seg")

    baseline = results[0]
    for ctx_len, tps in results.items():
        degradation = (baseline - tps) / baseline
        assert degradation < 0.2, \
            f"Throughput degrada {degradation*100:.1f}% con ctx={ctx_len}"

    return results


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_throughput_vs_context(model)
