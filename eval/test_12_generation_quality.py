"""
Test 12 — Generacion No Degenera

El modelo no debe generar texto degenerado (repeticion infinita,
solo espacios, solo bytes nulos, solo ASCII non-printable).

Metrica: % de bytes printables en output, ratio de repeticion
Criterio: > 50% bytes printables, ratio de repeticion < 0.3
"""
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import bytes_to_patches


def test_generation_quality(model, n_chars: int = 500):
    model.eval()
    model.cpu()

    prompt = b"The history of artificial intelligence"

    # Prefill
    patches = bytes_to_patches(prompt)
    states = None
    with torch.inference_mode():
        _, states = model(patches, states)

    # Generar
    generated = []
    last_patch = patches[:, -1:, :]

    with torch.inference_mode():
        for _ in range(n_chars // 4):
            logits, states = model(last_patch, states)
            next_bytes = [logits[i][0, 0].argmax().item() for i in range(4)]
            generated.extend(next_bytes)
            last_patch = torch.tensor(next_bytes, dtype=torch.uint8).reshape(1, 1, 4)

    gen_bytes = bytes(generated[:n_chars])

    # Chequear printables
    printable = sum(32 <= b < 127 for b in gen_bytes)
    pct_print = printable / len(gen_bytes)

    # Chequear repeticion (rolling window de 8 bytes)
    repeats = sum(
        gen_bytes[i:i+8] == gen_bytes[i-8:i]
        for i in range(8, len(gen_bytes))
    ) / max(len(gen_bytes) - 8, 1)

    print(f"[Test 12] Printable: {pct_print*100:.1f}%  Repeticion: {repeats*100:.1f}%")
    print(f"[Test 12] Muestra: {gen_bytes[:100]}")

    assert pct_print > 0.5, \
        f"Solo {pct_print*100:.1f}% bytes printables — modelo degenerado"
    assert repeats < 0.3, \
        f"Repeticion alta: {repeats*100:.1f}% — modelo en loop"
    return {'printable_pct': pct_print, 'repeat_ratio': repeats}


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_generation_quality(model)
