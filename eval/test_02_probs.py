"""
Test 02 — Distribucion de Probabilidades Valida

Las 4 heads deben producir distribuciones validas (suman a 1, no negativas).

Metrica: max error de normalizacion
Criterio: < 1e-5 de error en suma
"""
import torch


def test_probability_distributions(model):
    model.eval()
    patches = torch.randint(0, 256, (1, 128, 4), dtype=torch.uint8)
    with torch.inference_mode():
        logits, _ = model(patches)
    for i, head_logits in enumerate(logits):
        probs = torch.softmax(head_logits, dim=-1)
        sums = probs.sum(dim=-1)
        max_err = (sums - 1.0).abs().max().item()
        assert max_err < 1e-5, \
            f"Head {i}: suma de probs = {sums.mean():.6f} (esperado 1.0)"


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_probability_distributions(model)
    print("PASS: test_02_probs")
