"""
Script mínimo: asume studio ya Running.
Solo sube archivos y lanza el training.

Uso:
    python lightning/upload_and_run.py --run run_01
"""
import argparse
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)

# Cargar .env
env_path = os.path.join(REPO_DIR, ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from lightning_sdk import Studio, Machine
from lightning_sdk.machine import CloudProvider

TEAMSPACE   = "model-performance-assessment-project"
USER        = "ignacioschwindtdev"
REMOTE_ROOT = "/teamspace/studios/this_studio"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--phase", type=int, default=1, choices=[1, 2])
    p.add_argument("--phase1-ckpt", type=str, default=None)
    args = p.parse_args()

    studio_name = f"ablation-{args.run.replace('_', '-')}"
    print(f"Conectando a '{studio_name}'...")
    studio = Studio(
        name=studio_name,
        teamspace=TEAMSPACE,
        user=USER,
        cloud_provider=CloudProvider.GCP,
        create_ok=True,
    )
    print(f"  Status: {studio.status}")

    # Setear HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        studio.set_env({"HF_TOKEN": hf_token})
        print("  HF_TOKEN OK")

    # Subir archivos (remote_path relativo — evita que os.path.normpath rompa slashes en Windows)
    print("Subiendo archivos...")
    for fname in ["train.py", "model.py", "ablation.py"]:
        local = os.path.join(REPO_DIR, fname)
        print(f"  {fname}...")
        studio.upload_file(local, fname)   # queda en ~/ del studio (home dir)
    print("  OK")

    # Construir comando
    parts = ["python train.py", "--mode ablation", f"--run {args.run}"]
    if args.dry_run:
        parts.append("--dry_run")
    if args.total_steps:
        parts.append(f"--total_steps {args.total_steps}")
    if args.phase == 2:
        parts.append("--phase 2")
        if args.phase1_ckpt:
            parts.append(f"--phase1_ckpt {args.phase1_ckpt}")

    CMD = (
        "pip install -q -U bitsandbytes>=0.46.1 datasets huggingface_hub && "
        "cd ~ && "
        + " ".join(parts)
    )

    print(f"\nLanzando: {' '.join(parts)}")
    studio.run_and_detach(CMD)
    print(f"\nRunning en: https://lightning.ai/{USER}/{TEAMSPACE}/studios/{studio_name}")
    print("Checkpoints -> HF Hub cada 25 min")

if __name__ == "__main__":
    main()
