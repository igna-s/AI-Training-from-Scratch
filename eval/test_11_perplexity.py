"""
Test 11 — Perplexity en Held-Out FineWeb

Perplexity en el split de test del dataset compartido.
Metrica estandar de calidad de lenguaje.

Metrica: perplexity (exp(loss))
Criterio: < 50 para 50M params (aceptable para el paper)
          < 30 para 170M params

Nota: comparar entre runs de ablation, no contra modelos mas grandes.

Concurrencia:
  1. DataLoader con num_workers para carga de datos I/O en paralelo
     (prefetch mientras el modelo computa el batch actual).
  2. Loss vectorizado: una sola cross_entropy sobre las 4 heads concatenadas
     en vez de 4 operaciones separadas. Aprovecha mejor OpenMP/MKL al operar
     sobre tensores mas grandes.
"""
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_perplexity(model, test_dataset, max_batches: int = 200):
    model.eval()
    total_loss = 0
    n_batches = 0

    # DataLoader con workers paralelos para solapar I/O con compute
    n_workers = max(0, min(2, (os.cpu_count() or 4) // 4))
    loader_kwargs = {'batch_size': 4}
    if n_workers > 0:
        loader_kwargs['num_workers'] = n_workers
        loader_kwargs['prefetch_factor'] = 2

    for batch in DataLoader(test_dataset, **loader_kwargs):
        if n_batches >= max_batches:
            break
        patches = batch['patches']
        targets = batch['targets']

        with torch.inference_mode():
            logits, _ = model(patches)
            # Loss vectorizado: concatenar 4 heads en una sola cross_entropy
            # Matematicamente equivalente a mean(CE_0..CE_3) cuando B*S es igual
            all_logits = torch.cat(
                [logits[i].reshape(-1, 256) for i in range(4)], dim=0
            )
            all_targets = torch.cat(
                [targets[:, :, i].long().reshape(-1) for i in range(4)], dim=0
            )
            loss = F.cross_entropy(all_logits, all_targets)

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
