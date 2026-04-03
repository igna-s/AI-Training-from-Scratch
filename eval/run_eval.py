"""
Runner unificado — Suite de Evaluacion CPU-1 v2

Toda la inferencia pasa por CPU1InferenceEngine (inference.py).
Los tests se agrupan en 5 modulos que comparten el engine.

Uso:
  python eval/run_eval.py --model checkpoints/run_10.pt
  python eval/run_eval.py --model checkpoints/run_10.pt --threads 1
  python eval/run_eval.py --model checkpoints/run_10.pt --threads 6 --skip-external
  python eval/run_eval.py --model checkpoints/run_10.pt --dataset path/
"""
import os
import sys
import json
import time
import platform
import argparse
from pathlib import Path

# Agregar eval/ y raiz al path
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import torch
from helpers import create_engine, get_hardware_info
from eval_correctness import run_correctness
from eval_performance import run_performance
from eval_quality import run_quality
from eval_dataset import run_dataset_checks
from eval_external import run_external


# =========================================================================
# Tabla de checks
# =========================================================================

CHECK_INFO = {
    'forward_shapes':  ('Forward sin crash',     'sin NaN/Inf'),
    'probs':           ('Probs validas',         'error < 1e-4'),
    'determinism':     ('Determinismo',          'diff == 0.0'),
    'state_prop':      ('State propagation',     'diff < 1e-4'),
    'ternary_weights': ('Pesos ternarios',       '{-1, 0, 1}'),
    'state_size':      ('Estado MLGRU',          '< 1MB'),
    'throughput_latency_jitter': ('Throughput + Latencia + Jitter',
                                  '> 10 tok/s, p50<100ms, CV<0.3'),
    'context_scaling':  ('Scaling contexto',     '< 20% degradacion'),
    'ram':              ('RAM inferencia',       'delta < 50MB'),
    'generation':      ('Generacion',            '> 50% print, < 30% rep'),
    'utf8':            ('UTF-8 valido',          '> 70%'),
    'perplexity':      ('Perplexity',            '< 200'),
    'teacher_coverage': ('Teacher coverage',     '> 15%'),
    'teacher_entropy':  ('Teacher entropy',      '0.5-5.0 bits'),
    'llama_cpp':       ('llama.cpp GGUF',        '> 20 tok/s'),
}


def _format_value(result) -> str:
    if result is None:
        return ''
    if isinstance(result, (int, float)):
        return f'{result:.2f}' if isinstance(result, float) else str(result)
    if isinstance(result, dict):
        if result.get('skipped'):
            return f"skip: {result.get('reason', '')}"[:40]
        # Claves importantes primero
        for key in ['throughput_sps', 'throughput_pps', 'p50_ms', 'jitter_cv',
                     'perplexity', 'utf8_pct', 'printable_pct', 'delta_mb',
                     'coverage_pct', 'entropy_bits', 'best_decode_tps',
                     'state_kb', 'max_diff', 'max_error', 'modules_checked',
                     'reduction_x', 'configs_tested', 'deterministic']:
            if key in result:
                v = result[key]
                return f'{v:.2f}' if isinstance(v, float) else str(v)
        for v in result.values():
            if isinstance(v, (int, float)):
                return f'{v:.2f}' if isinstance(v, float) else str(v)
    return str(result)[:40]


def _format_detail(result) -> list[str]:
    """Formatear resultado detallado para el reporte markdown."""
    if result is None:
        return []
    if not isinstance(result, dict):
        return [str(result)]
    if result.get('skipped'):
        return [f"*Skip: {result.get('reason', '')}*"]
    lines = []
    for k, v in result.items():
        if k in ('skipped', 'reason'):
            continue
        if isinstance(v, float):
            lines.append(f"  - `{k}`: {v:.4f}")
        elif isinstance(v, dict):
            lines.append(f"  - `{k}`:")
            for k2, v2 in v.items():
                lines.append(f"    - `{k2}`: {v2}")
        elif isinstance(v, list):
            lines.append(f"  - `{k}`: {v}")
        else:
            lines.append(f"  - `{k}`: {v}")
    return lines


def print_table(results: dict):
    sep = f"|{'-' * 4}|{'-' * 32}|{'-' * 8}|{'-' * 42}|{'-' * 26}|"
    hdr = f"| {'#':>2} | {'Check':<30} | {'':^6} | {'Valor':<40} | {'Criterio':<24} |"
    print(sep)
    print(hdr)
    print(sep)

    for i, (name, info) in enumerate(CHECK_INFO.items(), 1):
        desc, criterio = info
        if name in results:
            r = results[name]
            status = r['status']
            value = _format_value(r.get('result'))
        else:
            status = 'N/A'
            value = ''
        print(f"| {i:>2} | {desc:<30} | {status:^6} | {value:<40} | {criterio:<24} |")

    print(sep)


def save_reports(model_path: str, results: dict, engine_info, hw: dict,
                 n_threads: int, total_time: float,
                 passed: int, failed: int, skipped: int):
    model_name = Path(model_path).stem
    thread_suffix = f'_{n_threads}t' if n_threads else ''

    # === JSON ===
    report = {
        'meta': {
            'model': model_path,
            'model_class': engine_info.model_class if engine_info else 'unknown',
            'date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'hardware': hw,
            'threads': n_threads or 'auto',
            'total_time_s': round(total_time, 1),
        },
        'architecture': {
            'model_class': engine_info.model_class,
            'is_byte_level': engine_info.is_byte_level,
            'is_recurrent': engine_info.is_recurrent,
            'is_ternary': engine_info.is_ternary,
            'n_heads': engine_info.n_heads,
            'vocab_size': engine_info.vocab_size,
            'd_model': engine_info.d_model,
            'n_layers': engine_info.n_layers,
        } if engine_info else {},
        'summary': {'passed': passed, 'failed': failed, 'skipped': skipped},
        'checks': results,
    }

    resultados_dir = Path(ROOT) / 'resultados'
    resultados_dir.mkdir(exist_ok=True)
    json_path = resultados_dir / f'{model_name}{thread_suffix}.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    # === Markdown Ultra-Detallado ===
    md_path = resultados_dir / f'{model_name}{thread_suffix}.md'

    lines = [
        f'# Eval Report: {model_name}',
        '',
        '---',
        '',
        '## Environment',
        '',
        f'| Field | Value |',
        f'|---|---|',
        f'| **Date** | {time.strftime("%Y-%m-%d %H:%M:%S")} |',
        f'| **OS** | {platform.platform()} |',
        f'| **Processor** | {hw.get("processor", "CPU")} |',
        f'| **CPU Cores** | {hw.get("cpu_count", "?")} |',
        f'| **Threads Used** | {n_threads or "auto"} |',
        f'| **PyTorch** | {torch.__version__} |',
        f'| **Total Time** | {total_time:.1f}s |',
        '',
    ]

    if engine_info:
        lines += [
            '## Model Architecture',
            '',
            f'| Property | Value |',
            f'|---|---|',
            f'| **Class** | `{engine_info.model_class}` |',
            f'| **Byte-level** | {engine_info.is_byte_level} |',
            f'| **Recurrent (MLGRU)** | {engine_info.is_recurrent} |',
            f'| **Ternary weights** | {engine_info.is_ternary} |',
            f'| **Prediction heads** | {engine_info.n_heads} |',
            f'| **Vocab size** | {engine_info.vocab_size:,} |',
            f'| **d_model** | {engine_info.d_model} |',
            f'| **n_layers** | {engine_info.n_layers} |',
            '',
        ]

    lines += [
        '## Summary',
        '',
        f'**{passed} PASS / {failed} FAIL / {skipped} N/A** '
        f'(de {len(CHECK_INFO)} checks)',
        '',
        '---',
        '',
        '## Results Table',
        '',
        '| # | Check | Result | Value | Criteria | Time |',
        '|---|---|---|---|---|---|',
    ]

    for i, (name, info) in enumerate(CHECK_INFO.items(), 1):
        desc, criterio = info
        if name in results:
            r = results[name]
            status = r['status']
            value = _format_value(r.get('result'))
            t = f"{r.get('time_s', '')}s" if 'time_s' in r else ''
        else:
            status = 'N/A'
            value = ''
            t = ''
        emoji = '✅' if status == 'PASS' else '❌' if status in ('FAIL', 'ERROR') else '⏭'
        lines.append(f'| {i} | {desc} | {emoji} {status} | {value} | {criterio} | {t} |')

    lines += ['', '---', '', '## Detailed Results', '']

    for name, info in CHECK_INFO.items():
        desc, _ = info
        if name not in results:
            continue
        r = results[name]
        status = r['status']
        emoji = '✅' if status == 'PASS' else '❌' if status in ('FAIL', 'ERROR') else '⏭'

        lines.append(f'### {emoji} {desc}')
        lines.append('')
        lines.append(f'- **Status:** {status}')
        if 'time_s' in r:
            lines.append(f'- **Time:** {r["time_s"]}s')
        if 'error' in r:
            lines.append(f'- **Error:** `{r["error"]}`')
        if 'result' in r:
            detail = _format_detail(r['result'])
            lines.extend(detail)
        lines.append('')

    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        print(f"Markdown: {md_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar markdown: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Suite de Evaluacion CPU-1 v2 — Unificada"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--model',
                       help='Path al checkpoint .pt (CPU1/Ablation)')
    group.add_argument('--hf-model',
                       help='Nombre HuggingFace (ej: distilgpt2)')
    parser.add_argument('--config', default=None,
                        help='Path a config JSON (opcional)')
    parser.add_argument('--dataset', default=None,
                        help='Path al dataset')
    parser.add_argument('--gguf-path', default=None,
                        help='Path directo a .gguf')
    parser.add_argument('--skip-external', action='store_true',
                        help='Saltar benchmark llama.cpp')
    parser.add_argument('--threads', type=int, default=None,
                        help='Forzar N threads (1=single core, 6=all cores)')
    parser.add_argument('--quantize-bits', type=int, default=8,
                        help='Bits para cuantizacion HF (8=INT8, 32=FP32)')
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    # Pasar threads al engine
    if args.threads:
        config['n_threads'] = args.threads

    # Nombre para display y reportes
    if args.hf_model:
        # Para carpeta local, usar nombre de carpeta; para HF, quitar slashes
        if os.path.isdir(args.hf_model):
            model_display = Path(args.hf_model).name
            model_path_for_report = model_display
        else:
            model_display = args.hf_model
            model_path_for_report = args.hf_model.replace('/', '_')
        # Agregar bits al nombre del reporte
        model_path_for_report = f"{model_path_for_report}_q{args.quantize_bits}"
    else:
        model_display = args.model
        model_path_for_report = args.model

    hw = get_hardware_info()
    print(f"\n{'=' * 60}")
    print(f"  Suite de Evaluacion CPU-1 v2")
    print(f"  Modelo:   {model_display}")
    print(f"  Hardware: {hw['processor']} ({hw['cpu_count']} cores)")
    print(f"  Threads:  {args.threads or 'auto'}")
    if args.dataset:
        print(f"  Dataset:  {args.dataset}")
    print(f"{'=' * 60}")

    t_total = time.perf_counter()

    # === Crear engine UNA VEZ ===
    t0 = time.perf_counter()
    if args.hf_model:
        from inference import HuggingFaceEngine
        is_local = os.path.isdir(args.hf_model)
        engine = HuggingFaceEngine(
            args.hf_model,
            n_threads=args.threads,
            quantize_bits=args.quantize_bits,
            local=is_local,
        )
    else:
        engine = create_engine(args.model, config)
    print(f"\n  Engine creado en {time.perf_counter() - t0:.1f}s")
    print(f"  Arquitectura: {engine.info}")

    # === Cargar dataset ===
    test_ds = None
    if args.dataset:
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(args.dataset)
            test_ds = dataset['test'] if 'test' in dataset else dataset
        except Exception as e:
            print(f"  [WARN] No se pudo cargar dataset: {e}")

    # === Ejecutar grupos ===
    all_results = {}

    all_results.update(run_correctness(engine))
    all_results.update(run_performance(engine))
    all_results.update(run_quality(engine, test_ds))

    if test_ds is not None:
        all_results.update(run_dataset_checks(test_ds))

    if not args.skip_external:
        all_results.update(run_external(args.gguf_path, args.threads))

    total_time = time.perf_counter() - t_total

    # === Resumen ===
    passed = sum(1 for r in all_results.values() if r['status'] == 'PASS')
    failed = sum(1 for r in all_results.values() if r['status'] in ('FAIL', 'ERROR'))
    skipped = len(CHECK_INFO) - len(all_results)

    print(f"\n{'=' * 60}")
    print(f"  {passed} PASS / {failed} FAIL / {skipped} N/A "
          f"(de {len(CHECK_INFO)} checks)")
    print(f"  Total: {total_time:.1f}s")
    print(f"{'=' * 60}")
    print_table(all_results)

    save_reports(model_path_for_report, all_results, engine.info, hw,
                 args.threads, total_time, passed, failed, skipped)

    return all_results


if __name__ == '__main__':
    main()
