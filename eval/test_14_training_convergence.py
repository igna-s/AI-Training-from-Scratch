"""
Test 14 — Loss Decrece Durante Training

Verificar que el modelo realmente aprende — la loss debe bajar
en los primeros 100 steps de training.

Metrica: ratio loss[100] / loss[1]
Criterio: < 0.7 (baja al menos 30% en los primeros 100 steps)

Util para detectar bugs como: LR muy bajo, gradientes muertos,
BitLinear mal implementado (STE no fluye).
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def test_training_convergence(model, dataset, n_steps: int = 100):
    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    losses = []
    for i, batch in enumerate(DataLoader(dataset, batch_size=4)):
        if i >= n_steps:
            break
        patches = batch['patches'].to(device)
        targets = batch['targets'].to(device)

        logits, _ = model(patches)
        loss = sum(
            F.cross_entropy(
                logits[j].reshape(-1, 256),
                targets[:, :, j].long().reshape(-1)
            ) for j in range(4)
        ) / 4

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    ratio = losses[-1] / losses[0]
    print(f"[Test 14] Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (ratio={ratio:.3f})")
    assert ratio < 0.7, \
        f"Loss no bajo suficiente en {n_steps} steps: ratio={ratio:.3f}"
    return ratio


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
    test_training_convergence(model, dataset['train'])
