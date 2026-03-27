"""
Runner unificado — Suite de Validacion CPU-1 v2

Corre todos los tests y genera reporte para el paper.
Uso: python eval/run_all_tests.py --model checkpoints/run_10.pt [--dataset path] [--config config.json]
"""
import argparse
import json
import sys
import os
import time
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
from test_17_teacher_coverage import test_teacher_coverage
from test_18_teacher_entropy import test_teacher_entropy
from test_21_mlgru_state_size import test_mlgru_state_size


def load_model(model_path: str, config: dict | None = None,
               device: str = 'cpu') -> CPU1ModelV2:
    model = CPU1ModelV2(config)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_all(model_path: str, config: dict | None = None,
            dataset_path: str | None = None) -> dict:
    model = load_model(model_path, config)

    # Cargar dataset si esta disponible
    test_ds = None
    if dataset_path:
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(dataset_path)
            test_ds = dataset['test'] if 'test' in dataset else dataset
        except Exception as e:
            print(f"[WARN] No se pudo cargar dataset: {e}")

    # Tests que solo necesitan el modelo
    TESTS = [
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
        ])

    # Ordenar por numero de test
    TESTS.sort(key=lambda t: t[0])

    results = {}

    print(f"\n{'='*60}")
    print(f"Suite de Validacion CPU-1 v2")
    print(f"Modelo: {model_path}")
    if dataset_path:
        print(f"Dataset: {dataset_path}")
    print(f"{'='*60}\n")

    passed = failed = 0

    for test_name, test_fn in TESTS:
        try:
            t0 = time.perf_counter()
            result = test_fn()
            elapsed = time.perf_counter() - t0
            results[test_name] = {
                'status': 'PASS', 'time_s': round(elapsed, 2),
                'result': result
            }
            print(f"  PASS {test_name:<25} ({elapsed:.1f}s)")
            passed += 1
        except AssertionError as e:
            results[test_name] = {'status': 'FAIL', 'error': str(e)}
            print(f"  FAIL {test_name:<25} FAIL: {e}")
            failed += 1
        except Exception as e:
            results[test_name] = {'status': 'ERROR', 'error': str(e)}
            print(f"  ERR  {test_name:<25} ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"Resultado: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    # Guardar reporte
    report_path = Path(model_path).parent / 'validation_report.json'
    try:
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Reporte guardado: {report_path}")
    except Exception as e:
        print(f"[WARN] No se pudo guardar reporte: {e}")
        # Intentar guardar en directorio actual
        fallback = Path('validation_report.json')
        with open(fallback, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Reporte guardado: {fallback}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Suite de Validacion Completa CPU-1 v2"
    )
    parser.add_argument('--model', required=True,
                        help='Path al checkpoint del modelo')
    parser.add_argument('--config', default=None,
                        help='Path a config JSON (opcional)')
    parser.add_argument('--dataset', default=None,
                        help='Path al dataset (opcional, para tests 11,17,18)')
    args = parser.parse_args()

    config = None
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    run_all(args.model, config, args.dataset)
