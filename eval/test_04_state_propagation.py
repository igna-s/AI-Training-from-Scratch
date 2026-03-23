"""
Test 04 — Estado MLGRU se Propaga Correctamente

Procesar una secuencia larga de una vez vs en chunks debe dar
el mismo output (el estado se pasa correctamente entre chunks).

Metrica: max diferencia absoluta
Criterio: < 1e-4 (diferencia numerica aceptable por acumulacion FP16)
"""
import torch


def test_state_propagation(model):
    torch.manual_seed(0)
    full_seq = torch.randint(0, 256, (1, 128, 4), dtype=torch.uint8)

    # Procesar completo
    model.eval()
    with torch.no_grad():
        out_full, _ = model(full_seq)

    # Procesar en chunks de 32
    states = None
    chunk_outputs = []
    with torch.no_grad():
        for i in range(0, 128, 32):
            chunk = full_seq[:, i:i+32, :]
            out_chunk, states = model(chunk, states)
            chunk_outputs.append(out_chunk[0])  # head 0

    out_chunked = torch.cat(chunk_outputs, dim=1)
    diff = (out_full[0] - out_chunked).abs().max().item()
    assert diff < 1e-4, f"State propagation error: {diff:.6f}"


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_state_propagation(model)
    print("PASS: test_04_state_propagation")
