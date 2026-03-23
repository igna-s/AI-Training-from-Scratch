"""
Test 19 — Benchmark Completo vs Baselines

Tabla comparativa completa para el paper.
Compara CPU-1 contra baselines en el MISMO hardware.

Baselines:
  - llama.cpp Q4_0 1B (Llama-3.2)  -> descargar y compilar
  - llama.cpp Q2_K 1B
  - Slender-Mamba 170M (si tiene inference engine)

Metrica: tokens/seg, RAM, perplexity
Nota: si los baselines no estan disponibles, reportar solo CPU-1
y citar los numeros del paper original.
"""
import subprocess
import platform
import torch

from test_06_cpu_throughput import test_cpu_throughput
from test_11_perplexity import test_perplexity


def _get_model_ram(model) -> float:
    """RAM del modelo en MB (aproximacion por parametros)."""
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6


def _benchmark_llamacpp(model_name: str) -> float:
    """Intenta correr llama.cpp benchmark. Retorna 0 si no esta disponible."""
    try:
        result = subprocess.run(
            ['llama-bench', '-m', model_name, '-n', '128', '-p', '0'],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'tok/s' in line:
                    parts = line.split()
                    for j, p in enumerate(parts):
                        if 'tok/s' in p and j > 0:
                            try:
                                return float(parts[j-1])
                            except ValueError:
                                pass
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0.0


def _get_hardware_info() -> dict:
    """Obtiene info del hardware para el reporte."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'cpu': platform.processor() or 'CPU consumer',
    }
    try:
        info['cpu_count'] = torch.get_num_threads()
    except Exception:
        import os
        info['cpu_count'] = os.cpu_count()
    return info


def test_full_benchmark(model, test_dataset, hardware_info: dict | None = None):
    if hardware_info is None:
        hardware_info = _get_hardware_info()

    results = {}

    # CPU-1 (nuestro modelo)
    tps = test_cpu_throughput(model)
    ppl = test_perplexity(model, test_dataset) if test_dataset is not None else 0
    results['cpu1_python_lut'] = {
        'tokens_per_sec': tps,
        'ram_mb': _get_model_ram(model),
        'perplexity': ppl,
        'n_params_M': sum(p.numel() for p in model.parameters()) / 1e6,
    }

    # llama.cpp baselines (opcional)
    for baseline in ['llama3.2-1b-q4_0', 'llama3.2-1b-q2_k']:
        tps = _benchmark_llamacpp(baseline)
        if tps > 0:
            results[baseline] = {'tokens_per_sec': tps, 'source': 'llamacpp'}

    # Imprimir tabla
    print(f"\n{'='*65}")
    print("BENCHMARK COMPARATIVO — Para el Paper")
    print(f"Hardware: {hardware_info.get('cpu', 'CPU consumer')}")
    print(f"{'='*65}")
    print(f"{'Modelo':<25} {'Params':>7} {'TPS':>8} {'RAM':>8} {'PPL':>8}")
    print("-" * 65)
    for name, r in results.items():
        params = r.get('n_params_M', 0)
        params_str = f"{params:.0f}M" if params else "?"
        print(f"{name:<25} "
              f"{params_str:>7} "
              f"{r.get('tokens_per_sec', 0):>8.1f} "
              f"{r.get('ram_mb', 0):>7.0f}M "
              f"{r.get('perplexity', 0):>8.2f}")
    print(f"{'='*65}")

    return results


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', default=None)
    args = parser.parse_args()

    model = load_model(args.model)
    test_ds = None
    if args.dataset:
        from datasets import load_from_disk
        test_ds = load_from_disk(args.dataset)['test']
    test_full_benchmark(model, test_ds)
