"""
Test 09 — Throughput vs Longitud de Contexto

Verificar que el throughput NO degrada con contextos largos.
MLGRU deberia ser constante en memoria con cualquier longitud.
Un Transformer degradaria por el KV cache.

Metrica: tokens/seg para prefill lengths [0, 512, 2048, 8192]
Criterio: variacion < 20% entre el contexto corto y largo
"""
import time
import torch


def test_throughput_vs_context(model):
    model.eval()
    model.cpu()
    results = {}

    for ctx_len in [0, 512, 2048, 8192]:
        # Hacer prefill con ctx_len patches
        if ctx_len > 0:
            ctx = torch.zeros(1, ctx_len, 4, dtype=torch.uint8)
            with torch.no_grad():
                _, states = model(ctx)
        else:
            states = None

        # Medir throughput de generacion
        dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
        t0 = time.perf_counter()
        with torch.no_grad():
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
