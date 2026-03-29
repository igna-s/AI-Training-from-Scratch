"""
Crea el dataset de ablation y lo sube directo a HF Bucket via hf-mount.
Corre en Kaggle T4 GPU (Qwen-7B teacher extraction).
"""

import os
import sys
import shutil

# ============================================================
# PASO 0: Setup
# ============================================================
print("=" * 60)
print("PASO 0: Setup")
print("=" * 60)

# Secrets de Kaggle -> env vars
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
os.environ['HF_TOKEN'] = secrets.get_secret("HF_TOKEN")
print("  HF_TOKEN loaded from Kaggle Secrets")

import subprocess
print("\n  Installing bitsandbytes...")
subprocess.run(["pip", "install", "-q", "-U", "bitsandbytes>=0.46.1"], check=True)
print("  bitsandbytes installed")

# Copiar source files del dataset al working dir
# Kaggle monta datasets en distintas rutas según el contexto
src_dir = None
for candidate in ["/kaggle/input/cpu1-source-code", "/kaggle/input/datasets"]:
    if os.path.isdir(candidate):
        src_dir = candidate
        break
if src_dir is None:
    # Buscar cualquier subdirectorio en /kaggle/input
    for d in os.listdir("/kaggle/input"):
        p = os.path.join("/kaggle/input", d)
        if os.path.isdir(p):
            src_dir = p
            break
if src_dir is None:
    raise FileNotFoundError("No se encontró el dataset en /kaggle/input")
print(f"  Source dir: {src_dir}")

# Log completo de la estructura de /kaggle/input
print("\n  === /kaggle/input tree ===")
for root, dirs, files in os.walk("/kaggle/input"):
    level = root.replace("/kaggle/input", "").count(os.sep)
    indent = "  " * (level + 2)
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  {f}")
print("  === end tree ===\n")

dst_dir = "/kaggle/working/src"
os.makedirs(dst_dir, exist_ok=True)

# Buscar archivos .py en src_dir o en sus subdirectorios (recursivo)
py_files = []
actual_src = src_dir
for root, dirs, files in os.walk(src_dir):
    found = [f for f in files if f.endswith(".py")]
    if found:
        actual_src = root
        py_files = found
        break

print(f"  actual_src: {actual_src}")
print(f"  py_files encontrados: {py_files}")

if not py_files:
    raise FileNotFoundError(f"No se encontraron .py en {src_dir}")

for f in py_files:
    shutil.copy2(os.path.join(actual_src, f), os.path.join(dst_dir, f))
    print(f"  Copied {f}")

print(f"\n  sys.path inserting: {dst_dir}")
print(f"  dst_dir contents: {os.listdir(dst_dir)}")
sys.path.insert(0, dst_dir)
os.chdir(dst_dir)

import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    n_gpus = torch.cuda.device_count()
    print(f"  GPUs: {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)} ({mem_gb:.0f} GB)")

# ============================================================
# PASO 1: Crear dataset -> HF Bucket (hf-mount)
# ============================================================
print("\n" + "=" * 60)
print("PASO 1: Crear dataset -> HF Bucket (hf-mount)")
print("=" * 60)

# Verificar que se cargó la versión correcta del create_dataset
import create_dataset as _cd
_cd_path = getattr(_cd, '__file__', 'unknown')
_cd_mtime = os.path.getmtime(_cd_path) if os.path.exists(_cd_path) else 0
from datetime import datetime
print(f"  create_dataset.py loaded from: {_cd_path}")
print(f"  create_dataset.py mtime: {datetime.fromtimestamp(_cd_mtime)}")
print(f"  has gc import: {hasattr(_cd, 'gc')}")

# Monkey-patch para log de cada save en el multi-GPU worker
_orig_append = _cd._append_to_parquet
_save_counter = [0]
def _logging_append(records, path):
    _save_counter[0] += 1
    print(f"\n  >>> SAVE #{_save_counter[0]}: {len(records)} docs -> {path}", flush=True)
    return _orig_append(records, path)
_cd._append_to_parquet = _logging_append

from create_dataset import build_full_dataset

build_full_dataset(
    n_docs=850_000,   # 850k docs × 600 tok/doc avg ≈ 100k training seqs of 5000 tok (500M distilled tokens)
    stream_to_hub=True,
    chunk_size=500,
)

print("\n" + "=" * 60)
print("DONE — Dataset en HF Bucket, listo para entrenar")
print("=" * 60)
