"""
Test 01 — Forward Pass Sin Crash

Verificar que el modelo corre sin errores en todas las combinaciones
de batch_size y seq_len relevantes.

Metrica: pass/fail
Criterio: ningun AssertionError, RuntimeError, ni NaN en logits

Concurrencia: ThreadPoolExecutor para las 12 combinaciones independientes
de (batch_size, seq_len). PyTorch libera el GIL durante ops de tensor,
permitiendo solapamiento real entre forward passes concurrentes.
Workers = logical_cores / intra_op_threads (evita sobresuscripcion ATen).
"""
import os
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_forward_no_crash(model):
    model.eval()
    configs = [(b, s) for b in [1, 2, 4] for s in [1, 64, 512, 2048]]

    def _check_config(args):
        batch, seq_len = args
        patches = torch.zeros(batch, seq_len, 4, dtype=torch.uint8)
        with torch.inference_mode():
            logits, states = model(patches)
        assert len(logits) == 4
        for l in logits:
            assert l.shape == (batch, seq_len, 256), \
                f"Shape incorrecto: {l.shape} (esperado {(batch, seq_len, 256)})"
            assert not torch.isnan(l).any(), \
                f"NaN en logits batch={batch} seq={seq_len}"
            assert not torch.isinf(l).any(), \
                f"Inf en logits batch={batch} seq={seq_len}"

    # Workers = logical_cores / intra_op_threads para no sobresuscribir ATen
    logical = os.cpu_count() or 4
    intra = torch.get_num_threads() or max(1, logical // 2)
    n_workers = max(1, logical // intra)

    errors = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_check_config, cfg): cfg for cfg in configs}
        for future in as_completed(futures):
            try:
                future.result()
            except AssertionError as e:
                errors.append(str(e))

    if errors:
        raise AssertionError("\n".join(errors))


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    test_forward_no_crash(model)
    print("PASS: test_01_forward")
