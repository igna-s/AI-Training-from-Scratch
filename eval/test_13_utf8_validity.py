"""
Test 13 — UTF-8 Valido en Output

El modelo byte-level puede generar bytes invalidos en UTF-8.
Para RAG en espanol/ingles, la mayoria del output debe ser UTF-8 valido.

Metrica: % de ventanas de 64 bytes decodificables como UTF-8
Criterio: > 70%
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import CPU1ModelV2


def _is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def test_utf8_validity(model, n_chars: int = 1000):
    generated = model.generate_bytes(b"Buenos dias", max_tokens=n_chars)

    windows = [generated[i:i+64] for i in range(0, len(generated)-64, 32)]
    if len(windows) == 0:
        print("[Test 13] No hay suficientes bytes generados para evaluar")
        return 0.0

    valid_utf8 = sum(1 for w in windows if _is_valid_utf8(w))
    pct = valid_utf8 / len(windows)

    print(f"[Test 13] UTF-8 valido: {pct*100:.1f}%")
    assert pct > 0.7, f"Demasiados bytes UTF-8 invalidos: {pct*100:.1f}%"
    return pct


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_utf8_validity(model)
