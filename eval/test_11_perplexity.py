"""
Test 11 — Perplexity en Held-Out FineWeb

Perplexity en el split de test del dataset compartido.
Metrica estandar de calidad de lenguaje.

Metrica: perplexity (exp(loss))
Criterio: < 50 para 50M params (aceptable para el paper)
          < 30 para 170M params

Nota: comparar entre runs de ablation, no contra modelos mas grandes.
"""
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_perplexity(model, test_dataset, max_batches: int = 200):
    model.eval()
    total_loss = 0
    n_batches = 0

    for batch in DataLoader(test_dataset, batch_size=4):
        if n_batches >= max_batches:
            break
        patches = batch['patches']
        targets = batch['targets']

        with torch.no_grad():
            logits, _ = model(patches)
            loss = sum(
                F.cross_entropy(
                    logits[i].reshape(-1, 256),
                    targets[:, :, i].long().reshape(-1)
                ) for i in range(4)
            ) / 4

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    perplexity = math.exp(avg_loss)

    print(f"[Test 11] Perplexity: {perplexity:.2f}")
    assert perplexity < 200, \
        f"Perplexity inaceptable: {perplexity:.2f} (modelo no aprendio nada)"
    return perplexity


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    model = load_model(args.model)
    dataset = load_from_disk(args.dataset)
    test_perplexity(model, dataset['test'])
