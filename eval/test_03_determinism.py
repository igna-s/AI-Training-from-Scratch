"""
Test 03 — Estado MLGRU es Deterministico

Con el mismo input y el mismo estado inicial, el output debe ser identico.
Critico para inferencia reproducible.

Metrica: max diferencia absoluta entre dos runs
Criterio: == 0.0 (identico bit a bit)
"""
import torch


def test_determinism(model):
    torch.manual_seed(42)
    patches = torch.randint(0, 256, (1, 64, 4), dtype=torch.uint8)

    model.eval()
    with torch.inference_mode():
        out1, states1 = model(patches, hidden_states=None)
        out2, states2 = model(patches, hidden_states=None)

    for i in range(4):
        diff = (out1[i] - out2[i]).abs().max().item()
        assert diff == 0.0, \
            f"Head {i}: output no deterministico, diff={diff}"


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_determinism(model)
    print("PASS: test_03_determinism")
