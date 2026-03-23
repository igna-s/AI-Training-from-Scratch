"""
Test 16 — Comparativa MLGRU vs Transformer en CPU

Comparar Run 01 (Transformer) vs Run 05 (MLGRU) en throughput CPU.
Cuantifica el argumento central del paper: MLGRU > Transformer en CPU.

Metrica: tokens/seg CPU, RAM de estado (KV cache vs MLGRU state)
Criterio: MLGRU debe ser al menos 2x mas rapido en contextos largos
"""
import time
import torch


def _measure_tps_after_prefill(model, ctx_len: int, n_tokens: int = 100):
    """Mide tokens/seg despues de un prefill de ctx_len patches."""
    model.eval()
    model.cpu()

    if ctx_len > 0:
        ctx = torch.zeros(1, ctx_len, 4, dtype=torch.uint8)
        with torch.no_grad():
            _, states = model(ctx)
    else:
        states = None

    dummy = torch.zeros(1, 1, 4, dtype=torch.uint8)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_tokens):
            _, states = model(dummy, states)
    return n_tokens / (time.perf_counter() - t0)


def test_mlgru_vs_transformer(transformer_model, mlgru_model,
                               ctx_lengths=None):
    if ctx_lengths is None:
        ctx_lengths = [0, 512, 2048]

    print("\n[Test 16] MLGRU vs Transformer en CPU:")
    print(f"{'Contexto':>10} {'Transformer':>12} {'MLGRU':>10} {'Speedup':>8}")
    print("-" * 45)

    for ctx in ctx_lengths:
        tps_t = _measure_tps_after_prefill(transformer_model, ctx)
        tps_m = _measure_tps_after_prefill(mlgru_model, ctx)
        speedup = tps_m / max(tps_t, 0.01)
        print(f"{ctx:>10} {tps_t:>12.1f} {tps_m:>10.1f} {speedup:>7.1f}x")

    # En contextos largos, MLGRU debe ganar claramente
    tps_t_long = _measure_tps_after_prefill(transformer_model, 2048)
    tps_m_long = _measure_tps_after_prefill(mlgru_model, 2048)
    speedup_long = tps_m_long / max(tps_t_long, 0.01)

    assert speedup_long > 2.0, \
        f"MLGRU deberia ser >2x mas rapido en ctx=2048, got {speedup_long:.2f}x"

    return speedup_long


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer', required=True)
    parser.add_argument('--mlgru', required=True)
    args = parser.parse_args()
    t_model = load_model(args.transformer)
    m_model = load_model(args.mlgru)
    test_mlgru_vs_transformer(t_model, m_model)
