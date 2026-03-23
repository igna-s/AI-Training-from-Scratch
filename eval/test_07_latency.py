"""
Test 07 — Latencia por Token (percentiles)

Medir distribucion de latencia por token: p50, p95, p99.
El p99 importa para experiencia de usuario (RAG interactivo).

Metrica: ms por patch (p50, p95, p99)
Criterio: p50 < 100ms, p99 < 500ms en CPU consumer
"""
import time
import torch


def test_latency_distribution(model, n_tokens: int = 500):
    model.eval()
    model.cpu()
    states = None
    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    token_times = []

    with torch.no_grad():
        for _ in range(n_tokens):
            t0 = time.perf_counter()
            _, states = model(dummy, states)
            t1 = time.perf_counter()
            token_times.append((t1 - t0) * 1000)

    token_times.sort()
    p50 = token_times[int(0.50 * n_tokens)]
    p95 = token_times[int(0.95 * n_tokens)]
    p99 = token_times[int(0.99 * n_tokens)]

    print(f"[Test 07] Latencia p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
    assert p50 < 100, f"p50 muy alto: {p50:.1f}ms"
    assert p99 < 500, f"p99 muy alto: {p99:.1f}ms"
    return {'p50': p50, 'p95': p95, 'p99': p99}


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_latency_distribution(model)
