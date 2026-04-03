"""
Eval — Dataset: checks que solo necesitan el dataset (sin modelo).

Checks:
  teacher_coverage  — % de posiciones con teacher signal (> 15%)
  teacher_entropy   — Entropia promedio del teacher (0.5-5.0 bits)

No cargan ningun modelo. Validan la calidad del dataset de distilacion.
"""
import os

import torch
from torch.utils.data import DataLoader


def check_teacher_coverage(dataset) -> dict:
    """Test 17 — Coverage del teacher signal en el dataset."""
    total = 0
    with_teacher = 0

    n_workers = max(0, min(2, (os.cpu_count() or 4) // 4))
    loader_kwargs = {'batch_size': 32}
    if n_workers > 0:
        loader_kwargs['num_workers'] = n_workers
        loader_kwargs['prefetch_factor'] = 2

    for batch in DataLoader(dataset, **loader_kwargs):
        mask = batch['teacher_mask']
        total += mask.numel()
        with_teacher += mask.sum().item()

    coverage = with_teacher / max(total, 1) * 100
    print(f"    Teacher coverage: {coverage:.1f}%")
    assert coverage > 15, f"Coverage muy bajo: {coverage:.1f}%"
    return {'coverage_pct': round(coverage, 1)}


def check_teacher_entropy(dataset, max_batches: int = 100) -> dict:
    """Test 18 — Entropia del teacher en rango informativo."""
    entropies = []

    n_workers = max(0, min(2, (os.cpu_count() or 4) // 4))
    loader_kwargs = {'batch_size': 16}
    if n_workers > 0:
        loader_kwargs['num_workers'] = n_workers
        loader_kwargs['prefetch_factor'] = 2

    for i, batch in enumerate(DataLoader(dataset, **loader_kwargs)):
        if i >= max_batches:
            break
        probs = batch['teacher_probs']
        mask = batch['teacher_mask']

        masked_probs = probs[mask]
        if masked_probs.numel() == 0:
            continue

        safe_probs = masked_probs.clamp(min=1e-10)
        entropy = -(safe_probs * safe_probs.log2()).sum(dim=-1)
        entropies.extend(entropy.tolist())

    avg_entropy = sum(entropies) / max(len(entropies), 1)
    print(f"    Teacher entropy: {avg_entropy:.2f} bits")
    assert 0.5 < avg_entropy < 5.0, \
        f"Entropia fuera de rango: {avg_entropy:.2f} bits"
    return {'entropy_bits': round(avg_entropy, 2)}


# =========================================================================
# Entry point
# =========================================================================

def run_dataset_checks(dataset) -> dict:
    """Ejecutar checks de dataset. No necesita modelo."""
    from helpers import run_check

    print("\n--- Dataset ---")
    results = {}
    checks = [
        ('teacher_coverage', check_teacher_coverage),
        ('teacher_entropy',  check_teacher_entropy),
    ]
    for name, fn in checks:
        r = run_check(name, fn, dataset)
        results[name] = r
        print(f"  {results[name]['status']:5s}  {name}")

    return results
