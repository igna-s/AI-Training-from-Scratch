"""
Entrena todos los ablation runs de CPU-1 secuencialmente.
Corre en Kaggle T4 GPU.

Configurar RUNS_TO_TRAIN al principio para elegir cuáles correr.
Los runs con DeleteGate (run_08, run_10) se entrenan en dos fases automáticamente.
"""

import os
import sys
import json
import shutil
import time

# Secrets de Kaggle -> env vars
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = secrets.get_secret("HF_TOKEN")

# ============================================================
# CONFIG — cambiar acá
# ============================================================

# Runs a entrenar en orden. Comentar los que ya están listos.
RUNS_TO_TRAIN = [
    "run_01",
    "run_02",
    "run_03",
    "run_04",
    "run_05",
    "run_06",
    "run_07",
    "run_08",   # dos fases (DeleteGate)
    "run_09",
    "run_10",   # dos fases (DeleteGate)
    "run_11",
    # "small_01",
    # "small_02",
    # "small_03",
    # "small_04",
]

TOTAL_STEPS = None          # None = usar default del run, o int para override
STOP_ON_FAILURE = False     # False = seguir con el siguiente run si uno falla

# ============================================================
# PASO 0: Setup
# ============================================================
print("=" * 60)
print("PASO 0: Setup")
print("=" * 60)

src_dir = "/kaggle/input/cpu1-source-code"
dst_dir = "/kaggle/working/src"
os.makedirs(dst_dir, exist_ok=True)
for f in os.listdir(src_dir):
    if f.endswith(".py"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
        print(f"  Copied {f}")

sys.path.insert(0, dst_dir)
os.chdir(dst_dir)

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"  GPUs: {n_gpus}")
    for i in range(n_gpus):
        mem_gb = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.0f} GB)")

from train import train_run

# Runs que necesitan dos fases (tienen DeleteGate)
TWO_PHASE_RUNS = {"run_08", "run_10"}

# ============================================================
# PASO 1: Entrenar runs en secuencia
# ============================================================
results = {}
errors  = {}

for run_name in RUNS_TO_TRAIN:
    run_start = time.time()
    print("\n" + "=" * 60)
    print(f"Entrenando: {run_name}")
    print("=" * 60)

    try:
        if run_name in TWO_PHASE_RUNS:
            print(f"\n--- {run_name} — Phase 1 (CE pretraining) ---")
            train_run(
                run_name,
                dry_run=False,
                total_steps_override=TOTAL_STEPS,
                phase=1,
                phase1_ckpt=None,
                use_hf_mount=False,
            )
            phase1_ckpt = f"checkpoint_{run_name}_final.pt"
            print(f"Phase 1 done → {phase1_ckpt}")

            print(f"\n--- {run_name} — Phase 2 (DeleteGate) ---")
            metrics = train_run(
                run_name,
                dry_run=False,
                total_steps_override=TOTAL_STEPS,
                phase=2,
                phase1_ckpt=phase1_ckpt,
                use_hf_mount=False,
            )
        else:
            metrics = train_run(
                run_name,
                dry_run=False,
                total_steps_override=TOTAL_STEPS,
                phase=1,
                use_hf_mount=False,
            )

        elapsed = time.time() - run_start
        results[run_name] = {**metrics, '_elapsed_min': round(elapsed / 60, 1)}
        print(f"\n[OK] {run_name} completado en {elapsed / 60:.1f} min")

    except Exception as e:
        elapsed = time.time() - run_start
        errors[run_name] = str(e)
        print(f"\n[ERROR] {run_name} falló después de {elapsed / 60:.1f} min: {e}")
        if STOP_ON_FAILURE:
            print("STOP_ON_FAILURE=True — abortando.")
            break

# ============================================================
# Resumen final
# ============================================================
print("\n" + "=" * 60)
print("RESUMEN")
print("=" * 60)
print(f"Completados: {len(results)}/{len(RUNS_TO_TRAIN)}")
for rn, m in results.items():
    bpb     = m.get('final_bpb', m.get('bpb', '?'))
    elapsed = m.get('_elapsed_min', '?')
    print(f"  [OK] {rn:10s}  bpb={bpb}  {elapsed} min")
if errors:
    print(f"\nFallidos: {len(errors)}")
    for rn, err in errors.items():
        print(f"  [ERR] {rn}: {err}")

with open("run_all_results.json", "w") as f:
    json.dump({"results": results, "errors": errors}, f, indent=2)
print("\nResultados guardados en run_all_results.json")
