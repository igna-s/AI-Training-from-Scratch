"""
Test 24 — Benchmark Externo HuggingFace (Transformers baseline)

Mide TPS de un modelo HuggingFace en CPU con transformers puro (float32).
Sirve como baseline honesto antes de aplicar optimizaciones.

Metrica: tok/s (decode con KV cache)
Criterio: reportar (sin umbral fijo, depende del modelo)
"""
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT = "The history of artificial intelligence began in the mid-twentieth century"
N_TOKENS = 200
N_RUNS = 3
N_WARMUP = 2


def _configure_threads():
    import platform
    import psutil
    n = psutil.cpu_count(logical=False) or (os.cpu_count() or 4) // 2
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)
    return n


def test_benchmark_external_hf(model_path: str, n_tokens: int = N_TOKENS,
                                 n_runs: int = N_RUNS):
    n_threads = _configure_threads()
    print(f"[Test 24] Cargando {model_path} [float32, {n_threads} threads]...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, trust_remote_code=True
    )
    model.eval()
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
    n_params_m = sum(p.numel() for p in model.parameters()) / 1e6

    for _ in range(N_WARMUP):
        with torch.inference_mode():
            model(input_ids)

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

    tps = statistics.median(tps_list)
    print(f"[Test 24] {model_path} float32: {tps:.1f} tok/s ({n_params_m:.0f}M params)")

    return {'tps': round(tps, 1), 'n_params_m': round(n_params_m, 1), 'model': model_path}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-tokens', type=int, default=N_TOKENS)
    args = parser.parse_args()
    test_benchmark_external_hf(args.model, args.n_tokens)
