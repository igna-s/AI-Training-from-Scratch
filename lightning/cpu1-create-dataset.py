"""
Crea el dataset de ablation y lo sube directo a HF Hub via streaming.
Corre en Lightning.ai Studio con T4 (1 GPU).

Diferencias vs Kaggle:
  - HF_TOKEN desde env var o archivo .env en el root del repo
  - No copia archivos — usa el repo clonado directamente
  - Single T4: 1 teacher en cuda:0
"""

import os
import sys

# ============================================================
# PASO 0: Setup
# ============================================================
print("=" * 60)
print("PASO 0: Setup")
print("=" * 60)

# Detectar root del repo (este script vive en lightning/, repo en ../)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)

# Cargar HF_TOKEN desde .env si no está en el entorno
env_path = os.path.join(REPO_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))
    print("  .env loaded")

if not os.environ.get("HF_TOKEN"):
    raise EnvironmentError(
        "HF_TOKEN no encontrado. Ponlo en .env o en la env var HF_TOKEN del Studio."
    )
print("  HF_TOKEN OK")

import subprocess
print("\n  Installing bitsandbytes...")
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("  bitsandbytes installed")

# El repo está clonado — agregar al path directamente
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)
print(f"  REPO_DIR: {REPO_DIR}")
print(f"  sys.path[0]: {sys.path[0]}")

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"  GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = props.total_memory / (1024**3)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.0f} GB)")

# ============================================================
# PASO 1: Crear dataset -> HF Hub (stream + checkpoint)
# ============================================================
print("\n" + "=" * 60)
print("PASO 1: Crear dataset -> HF Hub (streaming con checkpoint)")
print("=" * 60)

from create_dataset import build_full_dataset

build_full_dataset(
    n_docs=850_000,     # 850k docs x ~600 tok/doc => ~100k seqs de 5000 tok
    stream_to_hub=True, # chunk-by-chunk a HF Hub; checkpointing automatico al Hub
    chunk_size=3000,    # 3000 docs/chunk
    shard_offset=0,     # retoma automaticamente desde checkpoint en HF
)

print("\n" + "=" * 60)
print("DONE — Dataset en HF Hub, listo para entrenar")
print("=" * 60)
