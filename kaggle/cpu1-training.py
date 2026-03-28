"""
Entrena un ablation run de CPU-1.
Lee el dataset desde HuggingFace Hub (streaming pipeline).
Corre en Kaggle T4 GPU.

Cambiar RUN_NAME para entrenar diferentes runs.
"""

import os
import sys
import shutil

# Secrets de Kaggle -> env vars
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = secrets.get_secret("HF_TOKEN")

# ============================================================
# CONFIG — cambiar aca
# ============================================================
RUN_NAME = "run_10"         # cual run entrenar
TOTAL_STEPS = None          # None = usar default (10_000), o int para override
PHASE = 1                   # 1 = distillation, 2 = delete gate
PHASE1_CKPT = None          # path al checkpoint de phase 1 (solo para PHASE=2)

# ============================================================
# PASO 0: Setup
# ============================================================
print("=" * 60)
print("PASO 0: Setup")
print("=" * 60)

# Copiar source files del dataset al working dir
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

# ============================================================
# PASO 1: Entrenar
# ============================================================
print("\n" + "=" * 60)
print(f"PASO 1: Entrenar {RUN_NAME}")
print("=" * 60)

from train import train_run

metrics = train_run(
    RUN_NAME,
    dry_run=False,
    total_steps_override=TOTAL_STEPS,
    phase=PHASE,
    phase1_ckpt=PHASE1_CKPT,
    use_hf_mount=False,   # hf-mount no funciona en Kaggle (no NFS); usa HF Hub API
)

import json
print("\n" + "=" * 60)
print(f"DONE — {RUN_NAME}")
print("=" * 60)
print(json.dumps(metrics, indent=2))
