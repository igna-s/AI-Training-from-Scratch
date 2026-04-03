"""
Runner maestro — Descarga modelos HF, corre eval suite completa para los 3.

Modelos:
  1. CPU-1 170M (checkpoint local, ya existente)
  2. GPT-2 Medium 355M (openai-community/gpt2-medium)
  3. LFM2.5-350M (LiquidAI/LFM2.5-350M)

Cuantizacion: INT4 y INT8 para cada modelo HF.
Threads: 1 y 6 para cada configuracion.

Resultados se guardan en resultados/.

Uso:
  python eval/run_all_models.py
  python eval/run_all_models.py --skip-download     # si ya se descargaron
  python eval/run_all_models.py --only gpt2          # solo GPT-2
  python eval/run_all_models.py --only lfm            # solo LFM2.5
  python eval/run_all_models.py --only cpu1           # solo CPU-1
"""
import os
import sys
import time
import subprocess
import argparse

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(EVAL_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Configuracion de modelos
MODELS = {
    'cpu1': {
        'name': 'CPU-1 170M',
        'checkpoint': os.path.join(ROOT, 'models', 'cpu1-170M', 'checkpoint.pt'),
        'type': 'checkpoint',
    },
    'gpt2': {
        'name': 'GPT-2 Medium 355M',
        'hf_id': 'openai-community/gpt2-medium',
        'local_dir': os.path.join(ROOT, 'models', 'gpt2-medium'),
        'type': 'hf',
    },
    'lfm': {
        'name': 'LFM2.5-350M',
        'hf_id': 'LiquidAI/LFM2.5-350M',
        'local_dir': os.path.join(ROOT, 'models', 'LFM2.5-350M'),
        'type': 'hf',
    },
}

THREAD_CONFIGS = [1, 6]
QUANT_BITS = [4, 8]  # INT4 y INT8


def download_models(only: str = None):
    """Descargar modelos HF a models/."""
    from inference import download_and_save_hf_model

    for key, cfg in MODELS.items():
        if cfg['type'] != 'hf':
            continue
        if only and key != only:
            continue

        local_dir = cfg['local_dir']
        if os.path.exists(local_dir) and os.listdir(local_dir):
            print(f"\n[Skip] {cfg['name']} ya existe en {local_dir}")
            continue

        download_and_save_hf_model(
            cfg['hf_id'], local_dir, clear_cache=False
        )

    # Limpiar cache despues de descargar todos
    from inference import _clear_hf_cache
    _clear_hf_cache()


def run_eval_subprocess(args_list: list[str], label: str) -> bool:
    """Ejecutar run_eval.py como subproceso."""
    cmd = [sys.executable, os.path.join(EVAL_DIR, 'run_eval.py')] + args_list
    print(f"\n{'=' * 70}")
    print(f"  RUNNING: {label}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'=' * 70}")

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            timeout=600,  # 10 min max por eval
        )
        elapsed = time.perf_counter() - t0
        status = 'OK' if result.returncode == 0 else f'FAIL (rc={result.returncode})'
        print(f"\n  [{status}] {label} — {elapsed:.1f}s")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"\n  [TIMEOUT] {label}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] {label}: {e}")
        return False


def run_all_evals(only: str = None):
    """Correr evaluaciones para todos los modelos y configuraciones."""
    results_summary = []

    for key, cfg in MODELS.items():
        if only and key != only:
            continue

        if cfg['type'] == 'checkpoint':
            # CPU-1: checkpoint .pt
            checkpoint = cfg['checkpoint']
            if not os.path.exists(checkpoint):
                print(f"\n[SKIP] {cfg['name']} — checkpoint no encontrado: {checkpoint}")
                results_summary.append((cfg['name'], 'SKIP', 'checkpoint not found'))
                continue

            for threads in THREAD_CONFIGS:
                label = f"{cfg['name']} — {threads}T"
                ok = run_eval_subprocess([
                    '--model', checkpoint,
                    '--threads', str(threads),
                    '--skip-external',
                ], label)
                results_summary.append((label, 'OK' if ok else 'FAIL', ''))

        elif cfg['type'] == 'hf':
            # HF: cargar desde carpeta local
            local_dir = cfg['local_dir']
            use_local = os.path.exists(local_dir) and os.listdir(local_dir)

            if use_local:
                model_arg_name = local_dir
            else:
                model_arg_name = cfg['hf_id']

            for bits in QUANT_BITS:
                for threads in THREAD_CONFIGS:
                    label = f"{cfg['name']} Q{bits} — {threads}T"
                    args_list = [
                        '--hf-model', model_arg_name,
                        '--threads', str(threads),
                        '--quantize-bits', str(bits),
                        '--skip-external',
                    ]
                    ok = run_eval_subprocess(args_list, label)
                    results_summary.append((label, 'OK' if ok else 'FAIL', ''))

    return results_summary


def print_summary(results: list[tuple]):
    """Imprimir resumen final."""
    print(f"\n\n{'=' * 70}")
    print(f"  RESUMEN FINAL")
    print(f"{'=' * 70}")
    print(f"  {'Config':<45} {'Status':<10}")
    print(f"  {'-' * 45} {'-' * 10}")
    for label, status, note in results:
        emoji = '✅' if status == 'OK' else '❌' if status == 'FAIL' else '⏭'
        extra = f" ({note})" if note else ''
        print(f"  {emoji} {label:<43} {status}{extra}")
    print(f"{'=' * 70}")

    n_ok = sum(1 for _, s, _ in results if s == 'OK')
    n_fail = sum(1 for _, s, _ in results if s == 'FAIL')
    n_skip = sum(1 for _, s, _ in results if s == 'SKIP')
    print(f"  Total: {n_ok} OK / {n_fail} FAIL / {n_skip} SKIP")
    print(f"  Resultados en: {os.path.join(ROOT, 'resultados')}")


def main():
    parser = argparse.ArgumentParser(
        description="Descarga modelos HF y corre eval suite para los 3 modelos"
    )
    parser.add_argument('--skip-download', action='store_true',
                        help='No descargar modelos (usar existentes)')
    parser.add_argument('--only', choices=['cpu1', 'gpt2', 'lfm'],
                        help='Solo ejecutar un modelo especifico')
    parser.add_argument('--download-only', action='store_true',
                        help='Solo descargar, no evaluar')
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print(f"  Suite Completa — CPU-1 vs GPT-2 Medium vs LFM2.5-350M")
    print(f"  Quantizacion: INT4 y INT8")
    print(f"  Threads: {THREAD_CONFIGS}")
    print(f"{'=' * 70}")

    t_total = time.perf_counter()

    # 1. Descargar modelos HF
    if not args.skip_download:
        print(f"\n\n{'#' * 70}")
        print(f"  FASE 1: DESCARGA DE MODELOS")
        print(f"{'#' * 70}")
        download_models(args.only)

    if args.download_only:
        print("\n  --download-only: Modelos descargados. Saliendo.")
        return

    # 2. Correr evaluaciones
    print(f"\n\n{'#' * 70}")
    print(f"  FASE 2: EVALUACIONES")
    print(f"{'#' * 70}")
    results = run_all_evals(args.only)

    # 3. Resumen
    total_time = time.perf_counter() - t_total
    print_summary(results)
    print(f"\n  Tiempo total: {total_time / 60:.1f} minutos")


if __name__ == '__main__':
    main()
