"""Genera train_kaggle.ipynb con deteccion de GPU y fallback graceful."""
import json
import pathlib

REPO = pathlib.Path("D:/GitHub/1.58bits")
model_src = (REPO / "model_v2.py").read_text(encoding="utf-8")
train_src = (REPO / "train.py").read_text(encoding="utf-8")


def code_cell(source_lines, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }


def md_cell(text, cell_id):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": [text]}


def to_lines(src):
    return src.splitlines(keepends=True)


# ============================================================
# Celda 1: Detectar GPU via nvidia-smi (ANTES de importar torch)
# ============================================================
gpu_detect_src = """\
import os, sys, subprocess, pathlib, re

# Detectar GPUs ANTES de importar torch
r = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap",
                    "--format=csv,noheader"], capture_output=True, text=True)
gpu_lines = [l.strip() for l in r.stdout.strip().split("\\n") if l.strip()]
n_gpus = len(gpu_lines)

print("Python:", sys.version.split()[0])
print("GPUs detectadas:", n_gpus)

GPU_OK = False
NEED_DOWNGRADE = False

for i, line in enumerate(gpu_lines):
    parts = line.split(", ")
    name = parts[0]
    cc = float(parts[1]) if len(parts) > 1 else 0
    ok = "OK" if cc >= 7.0 else "INCOMPATIBLE"
    print(f"  GPU {i}: {name} (cc {cc}) [{ok}]")
    if cc < 7.0:
        NEED_DOWNGRADE = True

if n_gpus == 0:
    print("ERROR: No se detectaron GPUs -> Settings -> GPU T4 x2")
elif NEED_DOWNGRADE:
    print()
    print("=" * 60)
    print("GPU(s) con compute capability < 7.0 (P100 = sm_60)")
    print("PyTorch moderno de Kaggle no soporta esta GPU.")
    print()
    print("ACCION REQUERIDA:")
    print("  1. Abrir este notebook en la UI de Kaggle")
    print("  2. Settings -> Accelerator -> GPU T4 x2")
    print("  3. Click Save & Run All (Commit)")
    print()
    print("Los archivos se escribiran pero el dry-run se saltea.")
    print("=" * 60)
else:
    GPU_OK = True
    if n_gpus >= 2:
        print("OK: 2 GPUs compatibles, listas para DDP")
    else:
        print("OK: GPU compatible (single-GPU mode)")
"""

# ============================================================
# Celda 2: Importar torch (solo si GPU OK)
# ============================================================
torch_import_src = """\
import torch

if GPU_OK:
    print("PyTorch:", torch.__version__)
    print("CUDA   :", torch.version.cuda)
    # Test rapido
    try:
        t = torch.zeros(1, device="cuda:0")
        del t
        print("OK: CUDA funcional")
    except Exception as e:
        print("ERROR CUDA:", e)
        GPU_OK = False
else:
    print("PyTorch:", torch.__version__, "(GPU no compatible, solo escritura de archivos)")
"""

# ============================================================
# Celda HF token
# ============================================================
hf_src = """\
try:
    from kaggle_secrets import UserSecretsClient
    hf_token = UserSecretsClient().get_secret("HF_TOKEN")
    os.environ["HF_TOKEN"] = hf_token
    print("OK: HF_TOKEN configurado")
except Exception as e:
    print("AVISO: HF_TOKEN no encontrado:", e)
    print("  Los checkpoints solo se guardaran localmente en /kaggle/working/")
"""

# ============================================================
# Celda verificar archivos
# ============================================================
ckpt_src = """\
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
pathlib.Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
print("Checkpoints ->", CHECKPOINT_DIR)

for f in ["model_v2.py", "train.py"]:
    exists = pathlib.Path("/kaggle/working/" + f).exists()
    print(("OK" if exists else "ERROR") + ": " + f)
"""

# ============================================================
# Celda dry-run
# ============================================================
dry_src = """\
if not GPU_OK:
    print("SKIP: GPU no disponible o incompatible.")
    print("  -> Settings -> Accelerator -> GPU T4 x2 -> Run All")
else:
    cmd_dry = [
        "torchrun",
        "--nproc_per_node", str(n_gpus),
        "--master_port", "12355",
        "/kaggle/working/train.py",
        "--dry-run",
        "--dry-run-samples", "200",
        "--total-steps", "30",
        "--log-every", "5",
        "--batch-per-gpu", "2",
        "--gradient-accumulation-steps", "1",
        "--num-workers", "0",
        "--output-dir", CHECKPOINT_DIR,
    ]
    print("Comando:", " ".join(cmd_dry))
    print("-" * 60)
    result = subprocess.run(cmd_dry)
    print("-" * 60)
    if result.returncode == 0:
        print("OK: Dry-run completado")
    else:
        print("ERROR: Dry-run fallo (exit code", result.returncode, ")")
"""

# ============================================================
# Celda training real (comentado)
# ============================================================
train_real_src = """\
# ENTRENAMIENTO REAL -- ajustar DATA_PATH y descomentar

# DATA_PATH = "/kaggle/input/cpu1-dataset"
# RESUME_FROM = None

# cmd_train = [
#     "torchrun",
#     "--nproc_per_node", str(n_gpus),
#     "--master_port", "12355",
#     "/kaggle/working/train.py",
#     "--data-path", DATA_PATH,
#     "--total-steps", "100000",
#     "--warmup-steps", "2000",
#     "--log-every", "100",
#     "--save-every", "500",
#     "--batch-per-gpu", "4",
#     "--gradient-accumulation-steps", "2",
#     "--gradient-checkpointing",
#     "--num-workers", "4",
#     "--upload-hf",
#     "--output-dir", CHECKPOINT_DIR,
# ]
# # if RESUME_FROM:
# #     cmd_train += ["--resume-from", CHECKPOINT_DIR + "/" + RESUME_FROM]
# result = subprocess.run(cmd_train)
# print("Exit code:", result.returncode)

print("Celda comentada -- descomentar cuando el dataset este listo")
"""

# ============================================================
# ENSAMBLAR NOTEBOOK
# ============================================================
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": [
        md_cell(
            "# CPU-1: Byte-Level LM - Training en 2xT4\n\n"
            "Entrena CPU-1 (~170M parametros ternarios) con distillation desde Qwen-14B.\n\n"
            "**Requisito:** Settings -> Accelerator -> GPU T4 x2\n\n"
            "Si Kaggle asigna P100 (API push), el notebook escribe los archivos\n"
            "pero saltea el entrenamiento. Correr manualmente con T4 x2 desde la UI.",
            "cell-title",
        ),
        code_cell(to_lines(gpu_detect_src), "cell-gpu-detect"),
        code_cell(to_lines(torch_import_src), "cell-torch"),
        md_cell(
            "## Escribir archivos del modelo y entrenamiento", "cell-md-write"
        ),
        code_cell(
            ["%%writefile /kaggle/working/model_v2.py\n"] + to_lines(model_src),
            "cell-write-model",
        ),
        code_cell(
            ["%%writefile /kaggle/working/train.py\n"] + to_lines(train_src),
            "cell-write-train",
        ),
        code_cell(to_lines(hf_src), "cell-hf"),
        code_cell(to_lines(ckpt_src), "cell-ckpt"),
        md_cell(
            "## Dry-run -- verificar setup antes de datos reales", "cell-md-dry"
        ),
        code_cell(to_lines(dry_src), "cell-dryrun"),
        md_cell(
            "## Entrenamiento Real\n\n"
            "1. Adjuntar dataset (Dataset -> Add Data)\n"
            "2. Ajustar `DATA_PATH`\n"
            "3. Descomentar la celda siguiente",
            "cell-md-train",
        ),
        code_cell(to_lines(train_real_src), "cell-train"),
    ],
}

out = REPO / "kaggle" / "train_kaggle.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

nb2 = json.loads(out.read_text(encoding="utf-8"))
print(f"Notebook: {out}")
print(f"  Celdas: {len(nb2['cells'])}")
print(f"  model_v2.py: {len(model_src.splitlines())} lines")
print(f"  train.py:    {len(train_src.splitlines())} lines")
print("OK")
