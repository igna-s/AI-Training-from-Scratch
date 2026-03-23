"""
Test 10 — Memoria RAM en Inferencia

Medir RAM total durante inferencia.
Para el paper: comparar vs baseline FP16.

Metrica: MB de RAM
Criterio para 50M: < 150MB
Criterio para 170M: < 500MB
"""
import os
import torch
import psutil


def test_ram_usage(model, n_tokens: int = 100):
    process = psutil.Process(os.getpid())

    model.eval()
    model.cpu()

    ram_before = process.memory_info().rss / 1e6

    states = None
    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    with torch.no_grad():
        for _ in range(n_tokens):
            _, states = model(dummy, states)

    ram_after = process.memory_info().rss / 1e6
    ram_delta = ram_after - ram_before

    model_size = sum(p.numel() * 2 for p in model.parameters()) / 1e6

    print(f"[Test 10] RAM modelo: {model_size:.1f}MB")
    print(f"[Test 10] RAM delta inferencia: {ram_delta:.1f}MB")
    print(f"[Test 10] RAM total: {ram_after:.1f}MB")

    assert ram_delta < 50, \
        f"Demasiadas allocaciones durante inferencia: {ram_delta:.1f}MB"
    return {'model_mb': model_size, 'delta_mb': ram_delta, 'total_mb': ram_after}


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_ram_usage(model)
