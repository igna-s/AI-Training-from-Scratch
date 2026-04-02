"""
Test 17 — Coverage del Teacher Signal

Verificar que la marginalizacion BPE->byte produjo suficiente
senal del teacher para training util.

Metrica: % de posiciones con teacher_mask=True
Criterio: > 15% (si es menor, la distilacion no aporta)
          Rango tipico esperado: 20-35% para ingles

Concurrencia: DataLoader con num_workers para carga de datos I/O en paralelo.
El procesamiento por batch es trivial (sum de mask), asi que el cuello de
botella es la lectura de datos que se solapa con el compute.
"""
import os
from torch.utils.data import DataLoader


def test_teacher_coverage(dataset):
    total = 0
    with_teacher = 0

    # DataLoader con workers paralelos para solapar I/O
    n_workers = max(0, min(2, (os.cpu_count() or 4) // 4))
    loader_kwargs = {'batch_size': 32}
    if n_workers > 0:
        loader_kwargs['num_workers'] = n_workers
        loader_kwargs['prefetch_factor'] = 2

    for batch in DataLoader(dataset, **loader_kwargs):
        mask = batch['teacher_mask']  # [B, S] bool
        total += mask.numel()
        with_teacher += mask.sum().item()

    coverage = with_teacher / max(total, 1) * 100
    print(f"[Test 17] Teacher coverage: {coverage:.1f}%")
    assert coverage > 15, \
        f"Coverage muy bajo: {coverage:.1f}% — revisar marginalizacion"
    return coverage


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    dataset = load_from_disk(args.dataset)
    test_teacher_coverage(dataset['test'])
