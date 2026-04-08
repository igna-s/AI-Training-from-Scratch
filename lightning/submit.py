"""
Sube el código y lanza cpu1-create-dataset.py en Lightning.ai Studio.
Usa T4 interruptible en GCP (~$0.34/hr).

Uso:
    python lightning/submit.py

Requiere:
    pip install lightning-sdk
    LIGHTNING_API_KEY + HF_TOKEN en .env
"""

import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)

# Cargar .env
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

STUDIO_NAME = "cpu1-dataset-gcp"
TEAMSPACE   = "model-performance-assessment-project"
USER        = "ignacioschwindtdev"
REMOTE_ROOT = "/teamspace/studios/this_studio"

print(f"Conectando/creando studio '{STUDIO_NAME}' en GCP...")
studio = Studio(
    name=STUDIO_NAME,
    teamspace=TEAMSPACE,
    user=USER,
    cloud_provider=CloudProvider.GCP,
    create_ok=True,
)
print(f"  Status: {studio.status}")

# Poner HF_TOKEN en el studio
hf_token = os.environ.get("HF_TOKEN", "")
if hf_token:
    studio.set_env({"HF_TOKEN": hf_token})
    print("  HF_TOKEN seteado")

# Arrancar en T4 interruptible GCP (~$0.34/hr)
if studio.status not in ("Running",):
    print("Iniciando T4 interruptible (GCP ~$0.34/hr)...")
    studio.start(machine=Machine.T4, interruptible=True)
    # Esperar a que esté Running antes de subir archivos
    for _ in range(60):
        if studio.status == "Running":
            break
        print(f"  Esperando... ({studio.status})")
        time.sleep(5)

print(f"  Status: {studio.status} | Machine: {studio.machine}")

# Subir archivos
print("\nSubiendo archivos...")
for local, remote in [
    (os.path.join(REPO_DIR, "create_dataset.py"),           f"{REMOTE_ROOT}/create_dataset.py"),
    (os.path.join(SCRIPT_DIR, "cpu1-create-dataset.py"),    f"{REMOTE_ROOT}/lightning/cpu1-create-dataset.py"),
]:
    print(f"  {os.path.basename(local)} -> {remote}")
    studio.upload_file(local, remote)
print("  OK")

# Correr
CMD = (
    "pip install -q -U bitsandbytes>=0.46.1 && "
    f"cd {REMOTE_ROOT} && "
    "python lightning/cpu1-create-dataset.py"
)
print(f"\nLanzando (detached)...")
studio.run_and_detach(CMD)
print(f"Corriendo en: https://lightning.ai/{USER}/{TEAMSPACE}/studios/{STUDIO_NAME}")
