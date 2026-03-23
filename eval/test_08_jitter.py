"""
Test 08 — Jitter de Latencia (estabilidad)

El jitter (varianza de latencia) importa para RAG interactivo.
Un modelo rapido pero con jitter alto es inutilizable.

Metrica: coeficiente de variacion (std/mean)
Criterio: CV < 0.3 (30% de variacion es aceptable)

Causa comun de jitter alto: allocaciones en el generate loop.
Si falla este test: revisar pre-allocacion de buffers (prompt 06).
"""
import time
import statistics
import torch


def test_latency_jitter(model, n_tokens: int = 200):
    model.eval()
    model.cpu()
    states = None
    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    times = []

    with torch.no_grad():
        for _ in range(n_tokens):
            t0 = time.perf_counter()
            _, states = model(dummy, states)
            times.append(time.perf_counter() - t0)

    mean = statistics.mean(times)
    std = statistics.stdev(times)
    cv = std / mean

    print(f"[Test 08] Jitter CV={cv:.3f} (mean={mean*1000:.2f}ms std={std*1000:.2f}ms)")
    assert cv < 0.3, f"Jitter muy alto: CV={cv:.3f}"
    return cv


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_latency_jitter(model)
