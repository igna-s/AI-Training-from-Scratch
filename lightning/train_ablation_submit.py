"""
Lanza un ablation run en Lightning.ai Studio (T4 no-interruptible).

Uso:
    python lightning/train_ablation_submit.py --run run_01
    python lightning/train_ablation_submit.py --run run_02 --dry-run

Requisitos:
    pip install lightning-sdk
    LIGHTNING_API_KEY + HF_TOKEN en .env
"""

import argparse
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)

# ─── Cargar .env ────────────────────────────────────────────────────────────
env_path = os.path.join(REPO_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

from lightning_sdk import Studio, Machine
from lightning_sdk.machine import CloudProvider

# ─── Constantes ─────────────────────────────────────────────────────────────
TEAMSPACE   = "model-performance-assessment-project"
USER        = "ignacioschwindtdev"
REMOTE_ROOT = "/teamspace/studios/this_studio"

FILES_TO_UPLOAD = [
    ("train.py",   "train.py"),
    ("model.py",   "model.py"),
    ("ablation.py","ablation.py"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True,
                   help="Ablation run ID, e.g. run_01 … run_15")
    p.add_argument("--dry-run", action="store_true",
                   help="Pasar --dry_run a train.py (datos sintéticos, CPU)")
    p.add_argument("--total-steps", type=int, default=None,
                   help="Overridear total_steps en el config")
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--phase1-ckpt", type=str, default=None,
                   help="Ruta remota al checkpoint de fase 1 (solo --phase 2)")
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run  # ej: run_01

    studio_name = f"ablation-{run_id.replace('_', '-')}"

    print(f"Conectando/creando studio '{studio_name}'...")
    studio = Studio(
        name=studio_name,
        teamspace=TEAMSPACE,
        user=USER,
        cloud_provider=CloudProvider.GCP,
        create_ok=True,
    )
    print(f"  Status: {studio.status}")

    # Setear HF_TOKEN en el studio
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        studio.set_env({"HF_TOKEN": hf_token})
        print("  HF_TOKEN seteado")

    # Arrancar con T4 no-interruptible
    if studio.status not in ("Running",):
        print("Iniciando T4 no-interruptible (GCP)...")
        studio.start(machine=Machine.T4, interruptible=False)
        for _ in range(120):
            if studio.status == "Running":
                break
            print(f"  Esperando... ({studio.status})")
            time.sleep(5)

    print(f"  Status: {studio.status} | Machine: {studio.machine}")

    # Subir archivos del repo
    print("\nSubiendo archivos...")
    for local_name, remote_name in FILES_TO_UPLOAD:
        local  = os.path.join(REPO_DIR, local_name)
        remote = f"{REMOTE_ROOT}/{remote_name}"
        print(f"  {local_name} -> {remote}")
        studio.upload_file(local, remote)
    print("  OK")

    # Construir comando
    train_cmd_parts = [
        "python train.py",
        "--mode ablation",
        f"--run {run_id}",
    ]
    if args.dry_run:
        train_cmd_parts.append("--dry_run")
    if args.total_steps is not None:
        train_cmd_parts.append(f"--total_steps {args.total_steps}")
    if args.phase == 2:
        train_cmd_parts.append("--phase 2")
        if args.phase1_ckpt:
            train_cmd_parts.append(f"--phase1_ckpt {args.phase1_ckpt}")

    train_cmd = " ".join(train_cmd_parts)

    CMD = (
        "pip install -q -U bitsandbytes>=0.46.1 datasets huggingface_hub && "
        f"cd {REMOTE_ROOT} && "
        f"{train_cmd}"
    )

    print(f"\nLanzando (detached): {train_cmd}")
    studio.run_and_detach(CMD)
    print(f"\nCorriendo en: https://lightning.ai/{USER}/{TEAMSPACE}/studios/{studio_name}")
    print(f"Checkpoints -> HF: Cukinator/cpu1-ablation-checkpoints  (cada 25 min)")


if __name__ == "__main__":
    main()
