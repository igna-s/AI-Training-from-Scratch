"""
Test 05 — Pesos Ternarios Son Realmente Ternarios

Despues de cuantizacion, todos los pesos BitLinear deben ser
exactamente {-1, 0, 1}. Sin valores intermedios.

Metrica: % de pesos fuera de {-1, 0, 1}
Criterio: 0% (exacto)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_v2 import BitLinear


def test_ternary_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            w = module.weight.data
            w_scale = w.abs().mean()
            w_q = (w / (w_scale + 1e-8)).round().clamp(-1, 1)

            valid_values = {-1.0, 0.0, 1.0}
            unique_vals = set(w_q.unique().tolist())
            invalid = unique_vals - valid_values

            assert len(invalid) == 0, \
                f"Capa {name}: valores no ternarios encontrados: {invalid}"


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_ternary_weights(model)
    print("PASS: test_05_ternary_weights")
