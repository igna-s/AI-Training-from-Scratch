"""
Test 06 — Throughput Baseline CPU

Medir tokens (patches) por segundo en CPU consumer.
Esta es la metrica central del paper.

Metrica: patches/seg, bytes/seg
Criterio: > 10 patches/seg en CPU 4-core (minimo usable)
          > 50 patches/seg es buen resultado para el paper
          > 100 patches/seg es resultado excelente

Optimizacion multicore:
  - torch.compile: fusiona kernels, elimina overhead de Python entre ops
  - torch.inference_mode: desactiva autograd dispatch (5-10% menos overhead)
  - El modelo se distribuye automaticamente entre todos los cores via
    el thread pool ATen controlado por torch.set_num_threads (OpenMP/MKL)
  - KMP_AFFINITY pina threads a cores consecutivos (menos cache thrashing)
"""
import time
import torch

try:
    from helpers import optimize_model_for_inference
except ImportError:
    optimize_model_for_inference = lambda m: m


def test_cpu_throughput(model, n_tokens: int = 200, n_runs: int = 5):
    model = optimize_model_for_inference(model)
    model.cpu()

    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    states = None

    # Warmup (incluye compilacion JIT de torch.compile si aplica)
    with torch.inference_mode():
        for _ in range(10):
            _, states = model(dummy, states)

    # Benchmark: cada run mide n_tokens generaciones secuenciales
    times = []
    for _ in range(n_runs):
        states = None
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_tokens):
                _, states = model(dummy, states)
        t1 = time.perf_counter()
        times.append(n_tokens / (t1 - t0))

    tps = sorted(times)[len(times) // 2]  # mediana
    print(f"[Test 06] Throughput CPU: {tps:.1f} patches/seg ({tps*4:.1f} bytes/seg)")
    assert tps > 10, f"Throughput muy bajo: {tps:.1f} patches/seg"
    return tps


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_cpu_throughput(model)
