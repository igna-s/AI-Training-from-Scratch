"""
Test 01 — Forward Pass Sin Crash

Verificar que el modelo corre sin errores en todas las combinaciones
de batch_size y seq_len relevantes.

Metrica: pass/fail
Criterio: ningun AssertionError, RuntimeError, ni NaN en logits
"""
import torch


def test_forward_no_crash(model):
    model.eval()
    with torch.no_grad():
        for batch in [1, 2, 4]:
            for seq_len in [1, 64, 512, 2048]:
                patches = torch.zeros(batch, seq_len, 4, dtype=torch.uint8)
                logits, states = model(patches)
                assert len(logits) == 4
                for l in logits:
                    assert l.shape == (batch, seq_len, 256), \
                        f"Shape incorrecto: {l.shape} (esperado {(batch, seq_len, 256)})"
                    assert not torch.isnan(l).any(), \
                        f"NaN en logits batch={batch} seq={seq_len}"
                    assert not torch.isinf(l).any(), \
                        f"Inf en logits batch={batch} seq={seq_len}"


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_forward_no_crash(model)
    print("PASS: test_01_forward")
