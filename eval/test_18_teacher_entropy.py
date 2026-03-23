"""
Test 18 — Distribucion del Teacher es Informativa

Las distribuciones del teacher no deben ser uniformes (alta entropia)
ni completamente deterministas (entropia cero).
Alta entropia -> teacher confundido -> senal inutil
Baja entropia -> posiciones faciles -> poca ganancia de distilacion

Metrica: entropia promedio de teacher_probs donde teacher_mask=True
Criterio: entre 0.5 y 5.0 bits (rango informativo)
"""
import torch
from torch.utils.data import DataLoader


def test_teacher_entropy(dataset, max_batches: int = 100):
    entropies = []

    for i, batch in enumerate(DataLoader(dataset, batch_size=16)):
        if i >= max_batches:
            break
        probs = batch['teacher_probs']  # [B, S, 256]
        mask = batch['teacher_mask']    # [B, S]

        # Entropia solo donde hay teacher
        masked_probs = probs[mask]  # [N, 256]
        if masked_probs.numel() == 0:
            continue

        # H = -sum(p * log2(p))
        safe_probs = masked_probs.clamp(min=1e-10)
        entropy = -(safe_probs * safe_probs.log2()).sum(dim=-1)
        entropies.extend(entropy.tolist())

    avg_entropy = sum(entropies) / max(len(entropies), 1)
    print(f"[Test 18] Entropia teacher: {avg_entropy:.2f} bits")
    assert 0.5 < avg_entropy < 5.0, \
        f"Entropia fuera de rango informativo: {avg_entropy:.2f} bits"
    return avg_entropy


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    dataset = load_from_disk(args.dataset)
    test_teacher_entropy(dataset['test'])
