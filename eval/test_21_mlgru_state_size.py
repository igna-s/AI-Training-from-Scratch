"""
Test 21 — Estado MLGRU: Tamano Real en RAM

Verificar que el estado MLGRU es realmente ~80KB para 20 capas.
Este es el argumento clave del paper vs KV cache de atencion.

Metrica: KB del estado total
Criterio: < 1MB (vs ~80MB del KV cache de Transformer equivalente)
"""
import torch


def test_mlgru_state_size(model):
    model.eval()

    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    with torch.inference_mode():
        _, states = model(dummy)

    state_bytes = sum(s.numel() * s.element_size() for s in states)
    state_kb = state_bytes / 1024

    # KV cache equivalente (Transformer con window=512, 20 capas, d=2048)
    kv_cache_kb = 20 * 512 * 2048 * 2 * 2 / 1024  # layers x window x d x kv x bytes

    print(f"[Test 21] Estado MLGRU: {state_kb:.1f} KB")
    print(f"[Test 21] KV cache equiv: {kv_cache_kb:.1f} KB")
    print(f"[Test 21] Reduccion: {kv_cache_kb / state_kb:.0f}x menos memoria de estado")

    assert state_kb < 1024, f"Estado MLGRU demasiado grande: {state_kb:.1f} KB"
    assert kv_cache_kb / state_kb > 100, \
        "La reduccion de estado deberia ser >100x"

    return {'state_kb': state_kb, 'kv_cache_kb': kv_cache_kb}


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_mlgru_state_size(model)
