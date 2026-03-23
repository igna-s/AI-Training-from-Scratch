"""
Benchmark de modelos externos (HuggingFace) en CPU.
Genera reporte markdown en resultados/{modelo}.md

Uso:
    python eval/benchmark_external.py --model models/distilgpt2
    python eval/benchmark_external.py --model openai-community/gpt2
"""
import argparse
import math
import os
import platform
import re
import statistics
import sys
import time

import psutil
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Agregar eval/ al path para importar helpers
_EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if _EVAL_DIR not in sys.path:
    sys.path.insert(0, _EVAL_DIR)
from helpers import configure_cpu_threads

# Configurar threads igual que inferencia (cores fisicos, sin HT)
configure_cpu_threads()


def get_hardware_info() -> dict:
    proc = platform.processor() or "Unknown CPU"
    cores = os.cpu_count() or 0
    ram_gb = psutil.virtual_memory().total / 1e9
    return {
        'cpu': proc,
        'cores': cores,
        'ram_gb': ram_gb,
        'platform': platform.platform(),
    }


def benchmark_correctness(model, tokenizer):
    """Tests 01-03: forward sin crash, probs validas, determinismo."""
    results = {}

    # Test 01: Forward sin crash
    test01_pass = True
    test01_detail = ""
    try:
        for batch in [1, 2]:
            for seq_len in [1, 64, 512]:
                input_ids = torch.randint(0, tokenizer.vocab_size, (batch, seq_len))
                with torch.no_grad():
                    out = model(input_ids)
                logits = out.logits
                assert logits.shape == (batch, seq_len, tokenizer.vocab_size), \
                    f"Shape {logits.shape}"
                assert not torch.isnan(logits).any(), "NaN en logits"
                assert not torch.isinf(logits).any(), "Inf en logits"
        test01_detail = "Sin NaN/Inf en todas las combinaciones"
    except Exception as e:
        test01_pass = False
        test01_detail = str(e)
    results['forward'] = {'pass': test01_pass, 'detail': test01_detail}

    # Test 02: Distribuciones de probabilidad validas
    test02_pass = True
    test02_detail = ""
    try:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 128))
        with torch.no_grad():
            logits = model(input_ids).logits
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        max_err = (sums - 1.0).abs().max().item()
        # Vocab grande (50K+) tiene mas error numerico que vocab 256
        threshold = 1e-4 if tokenizer.vocab_size > 1000 else 1e-5
        assert max_err < threshold, f"Error normalizacion: {max_err}"
        test02_detail = f"Max error normalizacion: {max_err:.2e}"
    except Exception as e:
        test02_pass = False
        test02_detail = str(e)
    results['probs'] = {'pass': test02_pass, 'detail': test02_detail}

    # Test 03: Determinismo
    test03_pass = True
    test03_detail = ""
    try:
        torch.manual_seed(42)
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, 64))
        model.eval()
        with torch.no_grad():
            out1 = model(input_ids).logits
            out2 = model(input_ids).logits
        diff = (out1 - out2).abs().max().item()
        assert diff == 0.0, f"Diff: {diff}"
        test03_detail = f"Diff max: {diff}"
    except Exception as e:
        test03_pass = False
        test03_detail = str(e)
    results['determinism'] = {'pass': test03_pass, 'detail': test03_detail}

    # Test 04: State propagation (KV cache: full vs chunked)
    test04_pass = True
    test04_detail = ""
    try:
        torch.manual_seed(0)
        full_input = torch.randint(0, tokenizer.vocab_size, (1, 128))
        with torch.no_grad():
            out_full = model(full_input).logits

        # Chunked
        chunk_outputs = []
        past_kv = None
        for i in range(0, 128, 32):
            chunk = full_input[:, i:i+32]
            with torch.no_grad():
                out = model(chunk, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                chunk_outputs.append(out.logits)
        out_chunked = torch.cat(chunk_outputs, dim=1)
        diff = (out_full - out_chunked).abs().max().item()
        assert diff < 1e-4, f"State propagation error: {diff:.6f}"
        test04_detail = f"Diff max (full vs chunked): {diff:.2e}"
    except Exception as e:
        test04_pass = False
        test04_detail = str(e)
    results['state_propagation'] = {'pass': test04_pass, 'detail': test04_detail}

    # Test 05: Ternary weights — N/A para modelos externos
    results['ternary'] = {'pass': None, 'detail': 'N/A (no BitLinear)'}

    return results


def benchmark_throughput(model, input_ids, n_tokens=200, n_runs=3):
    """Mide tokens/seg con KV cache."""
    results = []
    for _ in range(n_runs):
        past_kv = None
        cur = input_ids.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_tokens):
                if past_kv is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past_kv,
                                use_cache=True)
                past_kv = out.past_key_values
                cur = torch.cat([cur, out.logits[:, -1:].argmax(dim=-1)], dim=1)
        elapsed = time.perf_counter() - t0
        results.append(n_tokens / elapsed)
    return sorted(results)[len(results) // 2]  # mediana


def benchmark_latency(model, input_ids, n_tokens=500):
    """Mide latencia por token: p50, p95, p99, mean, std, CV."""
    token_times = []
    past_kv = None
    cur = input_ids.clone()
    with torch.no_grad():
        for i in range(n_tokens):
            inp = cur if past_kv is None else cur[:, -1:]
            t0 = time.perf_counter()
            out = model(inp, past_key_values=past_kv, use_cache=True)
            t1 = time.perf_counter()
            past_kv = out.past_key_values
            cur = torch.cat([cur, out.logits[:, -1:].argmax(dim=-1)], dim=1)
            if i > 0:  # skip prefill
                token_times.append((t1 - t0) * 1000)

    token_times.sort()
    n = len(token_times)
    mean = statistics.mean(token_times)
    std = statistics.stdev(token_times)
    return {
        'p50': token_times[int(0.50 * n)],
        'p95': token_times[int(0.95 * n)],
        'p99': token_times[int(0.99 * n)],
        'mean': mean,
        'std': std,
        'cv': std / mean,
    }


def benchmark_context_scaling(model, tokenizer, max_ctx=None):
    """Throughput vs longitud de contexto."""
    if max_ctx is None:
        max_ctx = getattr(model.config, 'n_positions',
                          getattr(model.config, 'max_position_embeddings', 1024))

    ctx_lengths = [c for c in [0, 128, 512] if c < max_ctx]
    results = {}

    for ctx_len in ctx_lengths:
        if ctx_len > 0:
            ctx_ids = torch.randint(0, 1000, (1, ctx_len))
            with torch.no_grad():
                out = model(ctx_ids, use_cache=True)
                past_kv = out.past_key_values
        else:
            past_kv = None

        dummy = torch.zeros(1, 1, dtype=torch.long)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                out = model(dummy, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                dummy = out.logits[:, -1:].argmax(dim=-1)
        tps = 100 / (time.perf_counter() - t0)
        results[ctx_len] = tps

    return results


def _kv_cache_bytes(past_kv):
    """Bytes totales del KV cache."""
    total = 0
    for layer_kv in past_kv:
        for t in layer_kv:
            if t is not None:
                total += t.numel() * t.element_size()
    return total


def _get_arch_info(model):
    """Info de arquitectura del modelo."""
    cfg = model.config
    return {
        'n_layers': getattr(cfg, 'n_layer', getattr(cfg, 'num_hidden_layers', '?')),
        'd_model': getattr(cfg, 'n_embd', getattr(cfg, 'hidden_size', '?')),
        'n_heads': getattr(cfg, 'n_head', getattr(cfg, 'num_attention_heads', '?')),
        'max_positions': getattr(cfg, 'n_positions',
                                 getattr(cfg, 'max_position_embeddings', '?')),
    }


def benchmark_ram(model, process):
    """RAM del modelo y del proceso."""
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    rss_before = process.memory_info().rss / 1e6

    # Generar 200 tokens para medir delta
    past_kv = None
    dummy = torch.zeros(1, 1, dtype=torch.long)
    with torch.no_grad():
        for _ in range(200):
            out = model(dummy, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            dummy = out.logits[:, -1:].argmax(dim=-1)

    rss_after = process.memory_info().rss / 1e6

    return {
        'model_mb': model_mb,
        'process_mb': rss_after,
        'ram_delta_mb': rss_after - rss_before,
    }


def benchmark_kv_cache(model):
    """Mide KV cache real a distintas longitudes de contexto."""
    max_pos = getattr(model.config, 'n_positions',
                      getattr(model.config, 'max_position_embeddings', 1024))

    lengths = [t for t in [1, 64, 200, 512] if t < max_pos]
    results = {}

    for n_tokens in lengths:
        past_kv = None
        dummy = torch.zeros(1, 1, dtype=torch.long)
        with torch.no_grad():
            for _ in range(n_tokens):
                out = model(dummy, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
                dummy = out.logits[:, -1:].argmax(dim=-1)

        kv_mb = _kv_cache_bytes(past_kv) / 1e6
        results[n_tokens] = kv_mb

    return results


def benchmark_perplexity(model, tokenizer, texts=None, max_length=512):
    """Perplexity en textos de referencia."""
    if texts is None:
        texts = [
            "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",
            "In computer science, a hash table is a data structure that implements an associative array, also called a dictionary or simply map; it is an abstract data type that maps keys to values. A hash table uses a hash function to compute an index, also called a hash code, into an array of buckets or slots, from which the desired value can be found.",
            "Buenos dias, hoy vamos a hablar sobre el aprendizaje automatico y sus aplicaciones en la vida cotidiana. El machine learning es una rama de la inteligencia artificial que se enfoca en el desarrollo de algoritmos que pueden aprender de los datos sin ser programados explicitamente.",
        ]

    total_loss = 0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors='pt', truncation=True,
                              max_length=max_length)
        input_ids = encodings['input_ids']
        seq_len = input_ids.shape[1]
        if seq_len < 2:
            continue

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            # HF devuelve loss promediada sobre tokens
            total_loss += outputs.loss.item() * (seq_len - 1)
            total_tokens += seq_len - 1

    avg_loss = total_loss / total_tokens
    return {
        'loss': avg_loss,
        'perplexity': math.exp(avg_loss),
        'n_tokens': total_tokens,
    }


def benchmark_prefill(model, tokenizer, lengths=None):
    """Velocidad de prefill (procesar prompt completo)."""
    if lengths is None:
        lengths = [32, 128, 512]

    max_pos = getattr(model.config, 'n_positions',
                      getattr(model.config, 'max_position_embeddings', 1024))
    lengths = [l for l in lengths if l < max_pos]

    results = {}
    for length in lengths:
        input_ids = torch.randint(0, 1000, (1, length))
        # Warmup
        with torch.no_grad():
            _ = model(input_ids)
        # Measure
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(input_ids)
            times.append(time.perf_counter() - t0)
        median_ms = sorted(times)[len(times) // 2] * 1000
        tps = length / (median_ms / 1000)
        results[length] = {'ms': median_ms, 'tok_per_sec': tps}

    return results


def generate_sample(model, tokenizer,
                    prompt="The history of artificial intelligence"):
    """Genera texto de muestra (greedy y sampling)."""
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    with torch.no_grad():
        out_greedy = model.generate(
            input_ids, max_new_tokens=100, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
        out_sample = model.generate(
            input_ids, max_new_tokens=100, do_sample=True,
            temperature=0.8, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)

    return {
        'greedy': tokenizer.decode(out_greedy[0], skip_special_tokens=True),
        'sampling': tokenizer.decode(out_sample[0], skip_special_tokens=True),
    }


def benchmark_generation_quality(model, tokenizer, n_tokens=500):
    """Calidad de generacion: % printable, repeticion, UTF-8 validity."""
    input_ids = tokenizer.encode(
        "The history of artificial intelligence", return_tensors='pt')

    with torch.no_grad():
        out = model.generate(
            input_ids, max_new_tokens=n_tokens, do_sample=True,
            temperature=0.8, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id)

    gen_text = tokenizer.decode(out[0], skip_special_tokens=True)
    gen_bytes = gen_text.encode('utf-8', errors='replace')

    # % printable ASCII
    printable = sum(32 <= b < 127 for b in gen_bytes)
    pct_printable = printable / max(len(gen_bytes), 1)

    # Repeticion (rolling window de 8 bytes)
    repeats = 0
    for i in range(8, len(gen_bytes)):
        if gen_bytes[i:i+8] == gen_bytes[i-8:i]:
            repeats += 1
    repeat_ratio = repeats / max(len(gen_bytes) - 8, 1)

    # UTF-8 validity (ventanas de 64 bytes)
    windows = [gen_bytes[i:i+64] for i in range(0, len(gen_bytes) - 64, 32)]
    if windows:
        valid_utf8 = sum(1 for w in windows if _is_valid_utf8(w))
        utf8_pct = valid_utf8 / len(windows)
    else:
        utf8_pct = 1.0

    # Unique tokens ratio (diversidad)
    tokens = gen_text.split()
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)

    return {
        'pct_printable': pct_printable,
        'repeat_ratio': repeat_ratio,
        'utf8_pct': utf8_pct,
        'unique_token_ratio': unique_ratio,
        'generated_bytes': len(gen_bytes),
        'generated_tokens': len(tokens),
    }


def _is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def write_markdown(path, model_name, n_params, hw, arch, correctness, tps,
                   latency, ctx, ram, kv_cache, ppl, prefill, gen_quality,
                   sample):
    """Escribe el reporte markdown."""

    # Helper para formatear resultado
    def _r(val):
        if val is None:
            return 'N/A'
        return 'PASS' if val else 'FAIL'

    # Pre-calcular pass/fail para cada test
    t = {}
    t['01'] = correctness['forward']['pass']
    t['02'] = correctness['probs']['pass']
    t['03'] = correctness['determinism']['pass']
    t['04'] = correctness['state_propagation']['pass']
    t['05'] = correctness['ternary']['pass']  # None = N/A
    t['06'] = tps['median'] > 10
    t['07_p50'] = latency['p50'] < 100
    t['07_p99'] = latency['p99'] < 500
    t['08'] = latency['cv'] < 0.3
    ctx_sorted = sorted(ctx.items())
    ctx_deg = (ctx_sorted[0][1] - ctx_sorted[-1][1]) / ctx_sorted[0][1] \
        if len(ctx_sorted) >= 2 else 0
    t['09'] = ctx_deg < 0.2
    t['10'] = ram['ram_delta_mb'] < 50
    t['11'] = ppl['perplexity'] < 200
    t['12_print'] = gen_quality['pct_printable'] > 0.5
    t['12_repeat'] = gen_quality['repeat_ratio'] < 0.3
    t['13'] = gen_quality['utf8_pct'] > 0.7
    t['14'] = None  # N/A
    t['15'] = None  # N/A
    t['16'] = None  # N/A
    t['17'] = None  # N/A
    t['18'] = None  # N/A
    t['19'] = None  # N/A (este reporte ES el benchmark)
    t['20'] = None  # N/A
    kv_sorted = sorted(kv_cache.items())
    per_token_kb = (kv_sorted[-1][1] - kv_sorted[0][1]) / \
        (kv_sorted[-1][0] - kv_sorted[0][0]) * 1000 if len(kv_sorted) >= 2 else 0
    t['21'] = kv_sorted[-1][1] < 1024 if kv_sorted else None  # < 1GB

    n_pass = sum(1 for v in t.values() if v is True)
    n_fail = sum(1 for v in t.values() if v is False)
    n_na = sum(1 for v in t.values() if v is None)

    lines = [
        f"# Benchmark: {model_name}",
        "",
        f"**Fecha:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Parametros:** {n_params/1e6:.1f}M",
        f"**Arquitectura:** {arch['n_layers']} layers, d_model={arch['d_model']}, "
        f"{arch['n_heads']} heads, max_pos={arch['max_positions']}",
        f"**Hardware:** {hw['cpu']} ({hw['cores']} cores, "
        f"{hw['ram_gb']:.0f}GB RAM)",
        f"**OS:** {hw['platform']}",
        "",
        "---",
        "",
        "## Resumen",
        "",
        f"**{n_pass} PASS / {n_fail} FAIL / {n_na} N/A** (de 21 tests)",
        "",
        "| # | Test | Resultado | Valor | Criterio |",
        "|---|---|---|---|---|",
        f"| 01 | Forward sin crash | {_r(t['01'])} | {correctness['forward']['detail']} | sin NaN/Inf |",
        f"| 02 | Probs validas | {_r(t['02'])} | {correctness['probs']['detail']} | error < 1e-4 |",
        f"| 03 | Determinismo | {_r(t['03'])} | {correctness['determinism']['detail']} | diff == 0.0 |",
        f"| 04 | State propagation | {_r(t['04'])} | {correctness['state_propagation']['detail']} | diff < 1e-4 |",
        f"| 05 | Pesos ternarios | {_r(t['05'])} | {correctness['ternary']['detail']} | 0% fuera de {{-1,0,1}} |",
        f"| 06 | Throughput CPU | {_r(t['06'])} | **{tps['median']:.1f}** tok/seg | > 10 |",
        f"| 07 | Latencia p50 | {_r(t['07_p50'])} | {latency['p50']:.2f} ms | < 100ms |",
        f"| 07 | Latencia p99 | {_r(t['07_p99'])} | {latency['p99']:.2f} ms | < 500ms |",
        f"| 08 | Jitter (CV) | {_r(t['08'])} | {latency['cv']:.3f} | < 0.3 |",
        f"| 09 | Scaling contexto | {_r(t['09'])} | {ctx_deg*100:.1f}% degradacion | < 20% |",
        f"| 10 | RAM inferencia | {_r(t['10'])} | {ram['ram_delta_mb']:.1f} MB delta | < 50MB |",
        f"| 11 | Perplexity | {_r(t['11'])} | **{ppl['perplexity']:.2f}** | < 200 |",
        f"| 12 | Bytes printables | {_r(t['12_print'])} | {gen_quality['pct_printable']*100:.1f}% | > 50% |",
        f"| 12 | Repeticion | {_r(t['12_repeat'])} | {gen_quality['repeat_ratio']*100:.1f}% | < 30% |",
        f"| 13 | UTF-8 valido | {_r(t['13'])} | {gen_quality['utf8_pct']*100:.1f}% | > 70% |",
        f"| 14 | Training convergence | {_r(t['14'])} | — | ratio < 0.7 |",
        f"| 15 | FPResidual ablation | {_r(t['15'])} | — | reportar |",
        f"| 16 | MLGRU vs Transformer | {_r(t['16'])} | — | > 2x speedup |",
        f"| 17 | Teacher coverage | {_r(t['17'])} | — | > 15% |",
        f"| 18 | Teacher entropy | {_r(t['18'])} | — | 0.5-5.0 bits |",
        f"| 19 | Benchmark completo | {_r(t['19'])} | — | reportar |",
        f"| 20 | Scaling 50M/170M | {_r(t['20'])} | — | PPL baja |",
        f"| 21 | Estado/KV cache | {_r(t['21'])} | {kv_sorted[-1][1]:.2f} MB ({kv_sorted[-1][0]} tok) | < 1GB |",
        "",
        "---",
        "",
        "## Detalle por Categoria",
        "",
        "### CAT 1: Correctitud (Tests 01-05)",
        "",
        "| Test | Resultado | Detalle |",
        "|---|---|---|",
        f"| 01 Forward sin crash | {_r(correctness['forward']['pass'])} | {correctness['forward']['detail']} |",
        f"| 02 Probs validas | {_r(correctness['probs']['pass'])} | {correctness['probs']['detail']} |",
        f"| 03 Determinismo | {_r(correctness['determinism']['pass'])} | {correctness['determinism']['detail']} |",
        f"| 04 State propagation | {_r(correctness['state_propagation']['pass'])} | {correctness['state_propagation']['detail']} |",
        f"| 05 Pesos ternarios | {_r(correctness['ternary']['pass'])} | {correctness['ternary']['detail']} |",
        "",
        "### CAT 2: Velocidad en CPU (Tests 06-10)",
        "",
        "#### Test 06 — Throughput",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Tokens/seg (mediana, {tps['n_tokens']} tok, {tps['n_runs']} runs) | **{tps['median']:.1f}** |",
        f"| Mejor run | {tps['best']:.1f} |",
        f"| Peor run | {tps['worst']:.1f} |",
        "",
        "#### Prefill Speed",
        "",
        "| Longitud | ms | tok/seg |",
        "|---|---|---|",
    ]

    for length, data in sorted(prefill.items()):
        lines.append(f"| {length} | {data['ms']:.1f} | {data['tok_per_sec']:.0f} |")

    lines.extend([
        "",
        "#### Test 07 — Latencia por Token",
        "",
        "| Percentil | ms |",
        "|---|---|",
        f"| p50 | {latency['p50']:.2f} |",
        f"| p95 | {latency['p95']:.2f} |",
        f"| p99 | {latency['p99']:.2f} |",
        f"| mean | {latency['mean']:.2f} |",
        f"| std | {latency['std']:.2f} |",
        "",
        "#### Test 08 — Jitter",
        "",
        f"| CV (std/mean) | **{latency['cv']:.3f}** |",
        "|---|---|",
        f"| mean | {latency['mean']:.2f} ms |",
        f"| std | {latency['std']:.2f} ms |",
        "",
        "#### Test 09 — Throughput vs Contexto",
        "",
        "| Contexto (tokens) | tok/seg | vs baseline |",
        "|---|---|---|",
    ])

    baseline_tps = None
    for ctx_len, ctx_tps in ctx_sorted:
        if baseline_tps is None:
            baseline_tps = ctx_tps
            lines.append(f"| {ctx_len} | {ctx_tps:.1f} | baseline |")
        else:
            delta = (ctx_tps - baseline_tps) / baseline_tps * 100
            lines.append(f"| {ctx_len} | {ctx_tps:.1f} | {delta:+.1f}% |")

    lines.append("")
    lines.append(f"Degradacion total: **{ctx_deg*100:.1f}%**")

    lines.extend([
        "",
        "#### Test 10 — RAM Inferencia",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Peso modelo (params) | {ram['model_mb']:.1f} MB |",
        f"| RAM delta inferencia | {ram['ram_delta_mb']:.1f} MB |",
        f"| RAM proceso total | {ram['process_mb']:.0f} MB |",
        "",
        "### CAT 3: Calidad (Tests 11-13)",
        "",
        "#### Test 11 — Perplexity",
        "",
        "| Metrica | Valor |",
        "|---|---|",
        f"| Loss (CE) | {ppl['loss']:.4f} |",
        f"| Perplexity | **{ppl['perplexity']:.2f}** |",
        f"| Tokens evaluados | {ppl['n_tokens']} |",
        "",
        "#### Test 12 — Generacion no degenera",
        "",
        "| Metrica | Valor | Criterio |",
        "|---|---|---|",
        f"| Bytes printables | {gen_quality['pct_printable']*100:.1f}% | > 50% |",
        f"| Ratio repeticion | {gen_quality['repeat_ratio']*100:.1f}% | < 30% |",
        f"| Diversidad tokens | {gen_quality['unique_token_ratio']*100:.1f}% | — |",
        f"| Bytes generados | {gen_quality['generated_bytes']} | — |",
        "",
        "#### Test 13 — UTF-8 valido",
        "",
        f"| UTF-8 valido | **{gen_quality['utf8_pct']*100:.1f}%** | > 70% |",
        "|---|---|---|",
        "",
        "### CAT 6: Estado / KV Cache (Test 21)",
        "",
        "| Tokens en contexto | KV cache (MB) |",
        "|---|---|",
    ])

    for n_tok, mb in kv_sorted:
        lines.append(f"| {n_tok} | {mb:.2f} |")

    if per_token_kb > 0:
        lines.append("")
        lines.append(f"Crecimiento KV cache: ~**{per_token_kb:.2f} KB/token**")

    lines.extend([
        "",
        "---",
        "",
        "## Muestras de Generacion",
        "",
        "### Greedy (temperature=0)",
        "",
        "```",
        sample['greedy'][:500],
        "```",
        "",
        "### Sampling (temperature=0.8, top_p=0.9)",
        "",
        "```",
        sample['sampling'][:500],
        "```",
        "",
    ])

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark de modelo externo en CPU"
    )
    parser.add_argument('--model', required=True,
                        help='Path local o nombre HuggingFace')
    parser.add_argument('--n-tokens', type=int, default=200,
                        help='Tokens para throughput (default: 200)')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Runs para throughput (default: 3)')
    parser.add_argument('--output-dir', default=None,
                        help='Directorio de salida (default: resultados/)')
    args = parser.parse_args()

    # Directorio de salida
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = args.output_dir or os.path.join(root, 'resultados')
    os.makedirs(out_dir, exist_ok=True)

    process = psutil.Process(os.getpid())
    hw = get_hardware_info()

    # Cargar modelo
    print(f"Cargando {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()
    model.cpu()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parametros: {n_params/1e6:.1f}M")

    prompt = "The history of artificial intelligence began"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    arch = _get_arch_info(model)

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(input_ids)

    # Benchmarks
    print("[1/9] Correctitud (forward, probs, determinismo)...")
    correctness = benchmark_correctness(model, tokenizer)
    for name, r in correctness.items():
        print(f"  -> {name}: {'PASS' if r['pass'] else 'FAIL'}")

    print("[2/9] Perplexity...")
    ppl = benchmark_perplexity(model, tokenizer)
    print(f"  -> PPL={ppl['perplexity']:.2f} (loss={ppl['loss']:.4f})")

    print("[3/9] Throughput...")
    tps_median = benchmark_throughput(model, input_ids, args.n_tokens, args.n_runs)
    # Run again to get all values for reporting
    tps_all = []
    for _ in range(args.n_runs):
        past_kv = None
        cur = input_ids.clone()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(args.n_tokens):
                if past_kv is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past_kv,
                                use_cache=True)
                past_kv = out.past_key_values
                cur = torch.cat([cur, out.logits[:, -1:].argmax(dim=-1)], dim=1)
        tps_all.append(args.n_tokens / (time.perf_counter() - t0))
    tps = {
        'median': sorted(tps_all)[len(tps_all) // 2],
        'best': max(tps_all),
        'worst': min(tps_all),
        'n_tokens': args.n_tokens,
        'n_runs': args.n_runs,
    }
    print(f"  -> {tps['median']:.1f} tok/seg (best={tps['best']:.1f}, worst={tps['worst']:.1f})")

    print("[4/9] Prefill speed...")
    prefill = benchmark_prefill(model, tokenizer)
    for length, data in sorted(prefill.items()):
        print(f"  -> len={length}: {data['ms']:.1f}ms ({data['tok_per_sec']:.0f} tok/seg)")

    print("[5/9] Latencia + jitter...")
    latency = benchmark_latency(model, input_ids)
    print(f"  -> p50={latency['p50']:.2f}ms  p99={latency['p99']:.2f}ms  CV={latency['cv']:.3f}")

    print("[6/9] Scaling de contexto...")
    ctx = benchmark_context_scaling(model, tokenizer)
    for c, t in sorted(ctx.items()):
        print(f"  -> ctx={c}: {t:.1f} tok/seg")

    print("[7/9] RAM...")
    ram = benchmark_ram(model, process)
    print(f"  -> modelo={ram['model_mb']:.1f}MB  proceso={ram['process_mb']:.0f}MB")

    print("[8/9] KV cache scaling...")
    kv_cache = benchmark_kv_cache(model)
    for n_tok, mb in sorted(kv_cache.items()):
        print(f"  -> {n_tok} tokens: {mb:.2f}MB")

    print("[9/9] Calidad de generacion...")
    gen_quality = benchmark_generation_quality(model, tokenizer)
    print(f"  -> printable={gen_quality['pct_printable']*100:.1f}%  "
          f"repeticion={gen_quality['repeat_ratio']*100:.1f}%  "
          f"utf8={gen_quality['utf8_pct']*100:.1f}%")

    print("Generando muestras...")
    sample = generate_sample(model, tokenizer)

    # Nombre de archivo
    safe_name = re.sub(r'[/\\:*?"<>|]', '_', args.model.rstrip('/').split('/')[-1])
    out_path = os.path.join(out_dir, f"{safe_name}.md")

    write_markdown(out_path, args.model, n_params, hw, arch, correctness, tps,
                   latency, ctx, ram, kv_cache, ppl, prefill, gen_quality,
                   sample)
    print(f"\nReporte guardado: {out_path}")


if __name__ == "__main__":
    main()
