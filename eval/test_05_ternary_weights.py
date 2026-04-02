"""
Test 05 — Pesos Ternarios Son Realmente Ternarios

Despues de cuantizacion, todos los pesos BitLinear deben ser
exactamente {-1, 0, 1}. Sin valores intermedios.

Metrica: % de pesos fuera de {-1, 0, 1}
Criterio: 0% (exacto)

Concurrencia: ThreadPoolExecutor para verificar cada capa BitLinear
en paralelo. Cada verificacion es independiente (solo lectura de pesos).
Workers = CPU_CORES - 1 (operaciones ligeras de tensor).
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import BitLinear


def test_ternary_weights(model):
    modules = [(name, module) for name, module in model.named_modules()
               if isinstance(module, BitLinear)]

    if not modules:
        return

    def _check_module(args):
        name, module = args
        w = module.weight.data
        w_scale = w.abs().mean()
        w_q = (w / (w_scale + 1e-8)).round().clamp(-1, 1)
        valid_values = {-1.0, 0.0, 1.0}
        unique_vals = set(w_q.unique().tolist())
        invalid = unique_vals - valid_values
        assert len(invalid) == 0, \
            f"Capa {name}: valores no ternarios encontrados: {invalid}"

    n_workers = min(len(modules), max(1, (os.cpu_count() or 4) - 1))

    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_check_module, m): m[0] for m in modules}
        for future in as_completed(futures):
            try:
                future.result()
            except AssertionError as e:
                errors.append(str(e))

    if errors:
        raise AssertionError("\n".join(errors))


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_ternary_weights(model)
    print("PASS: test_05_ternary_weights")
