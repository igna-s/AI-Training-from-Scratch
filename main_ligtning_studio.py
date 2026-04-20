"""
Entry point para Lightning Studio.
Instala dependencias y entrena todos los ablation runs en secuencia.

Uso:
    python main.py                        # corre todos los runs de RUNS_TO_TRAIN
    python main.py --runs run_06 run_10   # corre solo esos
    python main.py --dry-run              # test rápido sin GPU (10 pasos)
"""

import os
import sys
import json
import time
import argparse
import subprocess

# ============================================================
# CONFIG — cambiar acá cuáles runs ejecutar
# ============================================================

RUNS_TO_TRAIN = [
    "run_01",
    "run_02",
    "run_03",
    "run_04",
    "run_05",
    "run_06",
    "run_07",   # dos fases (DeleteGate)
    "run_08",
    "run_09",   # dos fases (DeleteGate)
    "run_10",
    "run_11",   # per-channel learnable decay (RWKV/HGRN2)
    "run_12",   # periodic state reset (Stuffed Mamba, zero params)
    "run_17",   # TIDE per-token early exit (post-training calibration)
    # "run_13",
    # "run_14",
    # "run_15",
    # "run_16",
]

# Runs con DeleteGate que necesitan dos fases
TWO_PHASE_RUNS = {"run_07", "run_09"}

# ============================================================
# PASO 0: Instalar dependencias
# ============================================================

def install_requirements():
    print("=" * 60)
    print("PASO 0: Instalando dependencias")
    print("=" * 60)
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "-q"],
        check=True,
    )
    print("Dependencias instaladas.\n")

# ============================================================
# PASO 1: Verificar entorno
# ============================================================

def check_env():
    import torch
    print("=" * 60)
    print("PASO 1: Entorno")
    print("=" * 60)
    print(f"Python:  {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.0f} GB)")

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("\n[AVISO] HF_TOKEN no encontrado en el entorno.")
        print("  Configurarlo en Lightning Studio: Settings → Secrets → HF_TOKEN")
        print("  Los checkpoints NO se subirán a HuggingFace Hub.\n")
    else:
        print(f"\nHF_TOKEN: {'*' * 8}{hf_token[-4:]}")

# ============================================================
# PASO 2: Entrenar
# ============================================================

def run_all(runs, total_steps_override=None, dry_run=False, stop_on_failure=False):
    from train import train_run

    results = {}
    errors  = {}

    print("\n" + "=" * 60)
    print(f"PASO 2: Entrenando {len(runs)} run(s)")
    print("=" * 60)

    for run_name in runs:
        run_start = time.time()
        print(f"\n{'─' * 60}")
        print(f"Run: {run_name}")
        print(f"{'─' * 60}")

        try:
            if run_name in TWO_PHASE_RUNS and not dry_run:
                # Phase 1: CE pretraining
                print(f"[Phase 1] CE pretraining...")
                train_run(
                    run_name,
                    dry_run=False,
                    total_steps_override=total_steps_override,
                    phase=1,
                    phase1_ckpt=None,
                    use_hf_mount=True,
                )
                phase1_ckpt = f"checkpoint_{run_name}_final.pt"
                print(f"[Phase 1] Listo → {phase1_ckpt}")

                # Phase 2: DeleteGate compression
                print(f"[Phase 2] DeleteGate compression...")
                metrics = train_run(
                    run_name,
                    dry_run=False,
                    total_steps_override=total_steps_override,
                    phase=2,
                    phase1_ckpt=phase1_ckpt,
                    use_hf_mount=True,
                )
            else:
                metrics = train_run(
                    run_name,
                    dry_run=dry_run,
                    total_steps_override=10 if dry_run else total_steps_override,
                    phase=1,
                    use_hf_mount=not dry_run,
                )

            elapsed = time.time() - run_start
            results[run_name] = {**metrics, '_elapsed_min': round(elapsed / 60, 1)}
            print(f"\n[OK] {run_name} — {elapsed / 60:.1f} min")

        except Exception as e:
            elapsed = time.time() - run_start
            errors[run_name] = str(e)
            print(f"\n[ERROR] {run_name} falló ({elapsed / 60:.1f} min): {e}")
            if stop_on_failure:
                print("stop_on_failure=True — abortando.")
                break

    return results, errors


def print_summary(results, errors, total):
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(f"Completados: {len(results)}/{total}")
    for rn, m in results.items():
        bpb     = m.get('final_bpb', m.get('bpb', '?'))
        elapsed = m.get('_elapsed_min', '?')
        print(f"  [OK]  {rn:12s}  bpb={bpb}  {elapsed} min")
    if errors:
        print(f"\nFallidos: {len(errors)}")
        for rn, err in errors.items():
            print(f"  [ERR] {rn}: {err}")

    out = "run_results.json"
    with open(out, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)
    print(f"\nResultados guardados en {out}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs",     nargs="+", default=None,
                        help="Runs a entrenar (default: todos en RUNS_TO_TRAIN)")
    parser.add_argument("--steps",    type=int,  default=None,
                        help="Override total_steps para todos los runs")
    parser.add_argument("--dry-run",  action="store_true",
                        help="10 pasos con datos sintéticos, sin GPU requerido")
    parser.add_argument("--skip-install", action="store_true",
                        help="No correr pip install")
    parser.add_argument("--stop-on-failure", action="store_true",
                        help="Abortar si un run falla")
    args = parser.parse_args()

    runs = args.runs if args.runs else RUNS_TO_TRAIN

    if not args.skip_install:
        install_requirements()

    check_env()

    results, errors = run_all(
        runs,
        total_steps_override=args.steps,
        dry_run=args.dry_run,
        stop_on_failure=args.stop_on_failure,
    )

    print_summary(results, errors, total=len(runs))
    sys.exit(1 if errors else 0)
