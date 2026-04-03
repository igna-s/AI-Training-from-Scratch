"""
Test 25 — Benchmark Externo Optimizado (torch.compile + INT8)

Compara el mismo modelo HuggingFace antes y después de aplicar
optimizaciones CPU: torch.compile, INT8 dynamic quantization, weight packing.

Metrica: speedup optimizado vs baseline
Criterio: speedup >= 1.2x (al menos 20% de mejora)
"""
import os
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)

PROMPT = "The history of artificial intelligence began in the mid-twentieth century"
N_TOKENS = 100
N_RUNS = 3
N_WARMUP = 2


def _bench_tps(model, input_ids, n_tokens, n_runs):
    tps_list = []
    for _ in range(n_runs):
        past_kv, cur = None, input_ids.clone()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_tokens):
                inp = cur if past_kv is None else cur[:, -1:]
                out = model(inp, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                cur = torch.cat([cur, out.logits[:, -1:].argmax(dim=-1)], dim=1)
        tps_list.append(n_tokens / (time.perf_counter() - t0))
    return statistics.median(tps_list)


def test_benchmark_external_optimized(model_path: str, n_tokens: int = N_TOKENS,
                                       n_runs: int = N_RUNS, quantize: bool = True):
    print(f"[Test 25] Cargando {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids

    for _ in range(N_WARMUP):
        with torch.inference_mode():
            model(input_ids)

    # Baseline
    tps_baseline = _bench_tps(model, input_ids, n_tokens, n_runs)
    print(f"[Test 25] Baseline: {tps_baseline:.1f} tok/s")

    # Optimizar
    try:
        from cpu_optimizations import optimize_model_for_cpu_inference
        model_opt = optimize_model_for_cpu_inference(
            model,
            compile=True,
            quantize=quantize,
            transpose_weights=True,
            warmup=True,
            vocab_size=tokenizer.vocab_size,
        )
    except Exception as e:
        print(f"[Test 25] WARN: optimize_model_for_cpu_inference falló: {e}")
        model_opt = model

    tps_opt = _bench_tps(model_opt, input_ids, n_tokens, n_runs)
    speedup = tps_opt / max(tps_baseline, 0.1)
    print(f"[Test 25] Optimizado: {tps_opt:.1f} tok/s | speedup={speedup:.2f}x")

    assert speedup >= 1.2, f"Speedup insuficiente: {speedup:.2f}x (minimo 1.2x)"

    return {
        'tps_baseline': round(tps_baseline, 1),
        'tps_optimized': round(tps_opt, 1),
        'speedup': round(speedup, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-tokens', type=int, default=N_TOKENS)
    args = parser.parse_args()
    test_benchmark_external_optimized(args.model, args.n_tokens)
