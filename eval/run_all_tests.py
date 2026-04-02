"""
Runner unificado — Suite de Validacion CPU-1 v2

Corre los 21 tests y genera reporte como tabla.
Uso:
  python eval/run_all_tests.py --model checkpoints/run_10.pt
  python eval/run_all_tests.py --model checkpoints/run_10.pt --dataset path/
  python eval/run_all_tests.py --model checkpoints/run_10.pt \\
      --model-transformer checkpoints/run_01.pt \\
      --model-no-fp checkpoints/run_11.pt
"""
import os
import sys

# Intel MKL/OpenMP: DEBE configurarse antes de importar torch
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
os.environ.setdefault('KMP_BLOCKTIME', '0')

import argparse
import json
import time
import platform
from pathlib import Path

# Agregar raiz del proyecto y eval/ al path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

import torch
from model import CPU1ModelV2
from helpers import configure_cpu_threads

# Configurar threads igual que inferencia (cores fisicos, sin HT)
configure_cpu_threads()

# Importar todos los tests
from test_01_forward import test_forward_no_crash
from test_02_probs import test_probability_distributions
from test_03_determinism import test_determinism
from test_04_state_propagation import test_state_propagation
from test_05_ternary_weights import test_ternary_weights
from test_06_cpu_throughput import test_cpu_throughput
from test_07_latency import test_latency_distribution
from test_08_jitter import test_latency_jitter
from test_09_context_scaling import test_throughput_vs_context
from test_10_ram_usage import test_ram_usage
from test_11_perplexity import test_perplexity
from test_12_generation_quality import test_generation_quality
from test_13_utf8_validity import test_utf8_validity
from test_14_training_convergence import test_training_convergence
from test_15_fpresidual_ablation import test_fpresidual_ablation
from test_16_mlgru_vs_transformer import test_mlgru_vs_transformer
from test_17_teacher_coverage import test_teacher_coverage
from test_18_teacher_entropy import test_teacher_entropy
from test_19_full_benchmark import test_full_benchmark
from test_20_scaling import test_scaling
from test_21_mlgru_state_size import test_mlgru_state_size


# =============================================================================
# Descripciones y criterios para la tabla
# =============================================================================

TEST_INFO = {
    '01': ('Forward sin crash',       'sin NaN/Inf'),
    '02': ('Probs validas',           'error < 1e-4'),
    '03': ('Determinismo',            'diff == 0.0'),
    '04': ('State propagation',       'diff < 1e-4'),
    '05': ('Pesos ternarios',         '0% fuera de {-1,0,1}'),
    '06': ('Throughput CPU',          '> 10 tok/s'),
    '07': ('Latencia p50/p99',        'p50<100ms p99<500ms'),
    '08': ('Jitter (CV)',             'CV < 0.3'),
    '09': ('Scaling contexto',        'degradacion < 20%'),
    '10': ('RAM inferencia',          'delta < 50MB'),
    '11': ('Perplexity',              '< 200'),
    '12': ('Generacion',              '>50% print, <30% rep'),
    '13': ('UTF-8 valido',            '> 70%'),
    '14': ('Training convergence',    'ratio < 0.7'),
    '15': ('FPResidual ablation',     'reportar'),
    '16': ('MLGRU vs Transformer',    '> 2x speedup'),
    '17': ('Teacher coverage',        '> 15%'),
    '18': ('Teacher entropy',         '0.5-5.0 bits'),
    '19': ('Benchmark completo',      'reportar'),
    '20': ('Scaling 50M/170M',        'PPL baja'),
    '21': ('Estado/KV cache',         '< 1GB'),
}


def load_model(model_path: str, config: dict | None = None,
               device: str = 'cpu') -> CPU1ModelV2:
    model = CPU1ModelV2(config)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _format_value(result) -> str:
    """Extrae un valor representativo del resultado para la tabla."""
    if result is None:
        return ''
    if isinstance(result, (int, float)):
        if isinstance(result, float):
            if result > 100:
                return f'{result:.1f}'
            return f'{result:.4f}'
        return str(result)
    if isinstance(result, dict):
        # Intentar extraer la metrica mas relevante
        for key in ['perplexity', 'tps', 'median_tps', 'ratio',
                     'speedup', 'cv', 'p50_ms', 'delta_mb',
                     'degradation_pct', 'utf8_pct', 'printable_pct']:
            if key in result:
                v = result[key]
                return f'{v:.2f}' if isinstance(v, float) else str(v)
        # Fallback: primer valor numerico
        for v in result.values():
            if isinstance(v, (int, float)):
                return f'{v:.2f}' if isinstance(v, float) else str(v)
    return str(result)[:30]


def print_table(results: dict):
    """Imprime tabla de resultados estilo paper."""
    print()
    hdr = f"| {'#':>2} | {'Test':<25} | {'Resultado':^9} | {'Valor':<30} | {'Criterio':<22} |"
    sep = f"|{'-'*4}|{'-'*27}|{'-'*11}|{'-'*32}|{'-'*24}|"
    print(sep)
    print(hdr)
    print(sep)

    for num in sorted(TEST_INFO.keys(), key=lambda x: int(x)):
        name, criterio = TEST_INFO[num]
        key = [k for k in results if k.startswith(f'{num}_')]
        if key:
            r = results[key[0]]
            status = r['status']
            value = _format_value(r.get('result'))
        else:
            status = 'N/A'
            value = ''
        print(f"| {num:>2} | {name:<25} | {status:^9} | {value:<30} | {criterio:<22} |")

    print(sep)


def run_all(model_path: str, config: dict | None = None,
            dataset_path: str | None = None,
            model_transformer_path: str | None = None,
            model_no_fp_path: str | None = None,
            model_50m_path: str | None = None,
            model_170m_path: str | None = None) -> dict:
    model = load_model(model_path, config)

    # Cargar dataset si esta disponible
    test_ds = None
    train_ds = None
    if dataset_path:
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            test_ds = dataset['test'] if 'test' in dataset else dataset
            train_ds = dataset['train'] if 'train' in dataset else dataset
        except Exception as e:
            print(f"[WARN] No se pudo cargar dataset: {e}")

    # Cargar modelos adicionales (opcionales)
    model_transformer = None
    model_no_fp = None
    model_50m = None
    model_170m = None
    if model_transformer_path:
        try:
            model_transformer = load_model(model_transformer_path, config)
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo transformer: {e}")
    if model_no_fp_path:
        try:
            model_no_fp = load_model(model_no_fp_path, config)
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo sin FPResidual: {e}")
    if model_50m_path:
        try:
            model_50m = load_model(model_50m_path, config)
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo 50M: {e}")
    if model_170m_path:
        try:
            model_170m = load_model(model_170m_path, config)
        except Exception as e:
            print(f"[WARN] No se pudo cargar modelo 170M: {e}")

    # =========================================================================
    # Definir todos los tests
    # =========================================================================
    TESTS = [
        # Solo modelo
        ('01_forward',     lambda: test_forward_no_crash(model)),
        ('02_probs',       lambda: test_probability_distributions(model)),
        ('03_determinism', lambda: test_determinism(model)),
        ('04_state_prop',  lambda: test_state_propagation(model)),
        ('05_ternary',     lambda: test_ternary_weights(model)),
        ('06_throughput',  lambda: test_cpu_throughput(model)),
        ('07_latency',     lambda: test_latency_distribution(model)),
        ('08_jitter',      lambda: test_latency_jitter(model)),
        ('09_ctx_scaling', lambda: test_throughput_vs_context(model)),
        ('10_ram',         lambda: test_ram_usage(model)),
        ('12_generation',  lambda: test_generation_quality(model)),
        ('13_utf8',        lambda: test_utf8_validity(model)),
        ('21_state_size',  lambda: test_mlgru_state_size(model)),
    ]

    # Tests que necesitan dataset
    if test_ds is not None:
        TESTS.extend([
            ('11_perplexity',      lambda: test_perplexity(model, test_ds)),
            ('17_teacher_cov',     lambda: test_teacher_coverage(test_ds)),
            ('18_teacher_entropy', lambda: test_teacher_entropy(test_ds)),
            ('19_benchmark',       lambda: test_full_benchmark(model, test_ds)),
        ])
    if train_ds is not None:
        TESTS.append(
            ('14_convergence', lambda: test_training_convergence(model, train_ds)),
        )

    # Tests multi-modelo
    if model_no_fp is not None and test_ds is not None:
        TESTS.append(
            ('15_fpresidual', lambda: test_fpresidual_ablation(model, model_no_fp, test_ds)),
        )
    if model_transformer is not None:
        TESTS.append(
            ('16_mlgru_vs_tf', lambda: test_mlgru_vs_transformer(model_transformer, model)),
        )
    if model_50m is not None and model_170m is not None and test_ds is not None:
        TESTS.append(
            ('20_scaling', lambda: test_scaling(model_50m, model_170m, test_ds)),
        )

    # Ordenar por numero de test
    TESTS.sort(key=lambda t: int(t[0].split('_')[0]))

    results = {}

    print(f"\n{'='*60}")
    print(f"  Suite de Validacion CPU-1 v2 — 21 tests")
    print(f"  Modelo: {model_path}")
    if dataset_path:
        print(f"  Dataset: {dataset_path}")
    extras = []
    if model_transformer_path:
        extras.append(f"transformer={model_transformer_path}")
    if model_no_fp_path:
        extras.append(f"no-fp={model_no_fp_path}")
    if model_50m_path:
        extras.append(f"50m={model_50m_path}")
    if model_170m_path:
        extras.append(f"170m={model_170m_path}")
    if extras:
        print(f"  Modelos extra: {', '.join(extras)}")
    print(f"{'='*60}\n")

    passed = failed = skipped = 0

    for test_name, test_fn in TESTS:
        try:
            t0 = time.perf_counter()
            result = test_fn()
            elapsed = time.perf_counter() - t0
            results[test_name] = {
                'status': 'PASS', 'time_s': round(elapsed, 2),
                'result': result
            }
            print(f"  PASS  {test_name:<25} ({elapsed:.1f}s)")
            passed += 1
        except AssertionError as e:
            results[test_name] = {'status': 'FAIL', 'error': str(e)}
            print(f"  FAIL  {test_name:<25} {e}")
            failed += 1
        except Exception as e:
            results[test_name] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ERR   {test_name:<25} {e}")
            failed += 1

    # Contar tests no ejecutados (N/A)
    all_nums = set(TEST_INFO.keys())
    run_nums = {k.split('_')[0] for k in results}
    skipped = len(all_nums - run_nums)

    # =========================================================================
    # Tabla de resultados
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  {passed} PASS / {failed} FAIL / {skipped} N/A (de 21 tests)")
    print(f"{'='*60}")
    print_table(results)

    # Guardar reporte JSON
    report_path = Path(model_path).parent / 'validation_report.json'
    try:
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReporte JSON: {report_path}")
    except Exception:
        fallback = Path('validation_report.json')
        with open(fallback, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nReporte JSON: {fallback}")

    # Guardar reporte markdown
    _save_markdown_report(model_path, results, passed, failed, skipped)

    return results


def _save_markdown_report(model_path: str, results: dict,
                          passed: int, failed: int, skipped: int):
    """Genera reporte markdown en resultados/."""
    model_name = Path(model_path).stem
    resultados_dir = Path(ROOT) / 'resultados'
    resultados_dir.mkdir(exist_ok=True)
    md_path = resultados_dir / f'{model_name}.md'

    n_params = '?'
    for r in results.values():
        if isinstance(r.get('result'), dict) and 'n_params_M' in r['result']:
            n_params = f"{r['result']['n_params_M']:.1f}M"
            break

    lines = [
        f'# Benchmark: {model_name}',
        '',
        f'**Fecha:** {time.strftime("%Y-%m-%d %H:%M")}',
        f'**Parametros:** {n_params}',
        f'**Hardware:** {platform.processor()} ({os.cpu_count()} cores)',
        f'**OS:** {platform.platform()}',
        '',
        '---',
        '',
        '## Resumen',
        '',
        f'**{passed} PASS / {failed} FAIL / {skipped} N/A** (de 21 tests)',
        '',
        '| # | Test | Resultado | Valor | Criterio |',
        '|---|---|---|---|---|',
    ]

    for num in sorted(TEST_INFO.keys(), key=lambda x: int(x)):
        name, criterio = TEST_INFO[num]
        key = [k for k in results if k.startswith(f'{num}_')]
        if key:
            r = results[key[0]]
            status = r['status']
            value = _format_value(r.get('result'))
        else:
            status = 'N/A'
            value = ''
        lines.append(f'| {num} | {name} | {status} | {value} | {criterio} |')

    lines.append('')

    try:
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        print(f"Reporte markdown: {md_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar reporte markdown: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suite de Validacion Completa CPU-1 v2"
    )
    parser.add_argument('--model', required=True,
                        help='Path al checkpoint del modelo principal')
    parser.add_argument('--config', default=None,
                        help='Path a config JSON (opcional)')
    parser.add_argument('--dataset', default=None,
                        help='Path al dataset (para tests 11,14,15,17,18,19,20)')

    # Modelos adicionales para tests comparativos
    parser.add_argument('--model-transformer', default=None,
                        help='Checkpoint Transformer para test 16 (MLGRU vs Transformer)')
    parser.add_argument('--model-no-fp', default=None,
                        help='Checkpoint sin FPResidual para test 15')
    parser.add_argument('--model-50m', default=None,
                        help='Checkpoint 50M para test 20 (scaling)')
    parser.add_argument('--model-170m', default=None,
                        help='Checkpoint 170M para test 20 (scaling)')
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    run_all(
        args.model, config, args.dataset,
        model_transformer_path=args.model_transformer,
        model_no_fp_path=args.model_no_fp,
        model_50m_path=args.model_50m,
        model_170m_path=args.model_170m,
    )
