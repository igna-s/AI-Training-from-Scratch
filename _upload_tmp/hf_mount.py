"""
hf-mount: lazy filesystem mount for HuggingFace buckets/repos.

En vez de descargar el dataset completo (~5 GB) al disco,
hf-mount monta un bucket/repo HF como filesystem local via NFS.
Los archivos se fetchean on-demand (solo los bytes accedidos).

Buckets = read-write (para create_dataset.py)
Repos   = read-only  (alternativa para train.py)

Uso:
    from hf_mount import mount_bucket, unmount

    path = mount_bucket("Cukinator/cpu1-ablation-dataset")
    # ... escribir/leer archivos en path ...
    unmount(path)

Requiere: Linux (Kaggle OK). No necesita root (usa NFS backend).
Install automatico si no esta presente.
"""

import os
import shutil
import subprocess
from pathlib import Path

HF_MOUNT_DIR = Path('/tmp/hf_mount_dataset')


def _ensure_installed() -> bool:
    """Instalar hf-mount si no esta presente. Retorna True si disponible."""
    if shutil.which('hf-mount'):
        return True
    try:
        subprocess.run(
            'curl -fsSL https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh',
            shell=True, check=True, capture_output=True,
        )
        # El installer pone el binario en ~/.local/bin/
        local_bin = Path.home() / '.local' / 'bin'
        if str(local_bin) not in os.environ.get('PATH', ''):
            os.environ['PATH'] = f"{local_bin}:{os.environ.get('PATH', '')}"
        return shutil.which('hf-mount') is not None
    except subprocess.CalledProcessError as e:
        print(f"[hf-mount] Install failed: {e}")
        return False


def _get_token() -> str | None:
    """Buscar HF token en env vars (HF_TOKEN o Hugginface)."""
    return os.environ.get('HF_TOKEN') or os.environ.get('Hugginface')


def _run_mount(kind: str, name: str, mount_path: Path,
               token: str | None = None) -> Path:
    """Montar un bucket o repo HF."""
    if not _ensure_installed():
        raise RuntimeError(
            "No se pudo instalar hf-mount. "
            "Instalar manualmente: curl -fsSL "
            "https://raw.githubusercontent.com/huggingface/hf-mount/main/install.sh | sh"
        )

    mount_path = Path(mount_path)
    mount_path.mkdir(parents=True, exist_ok=True)

    token = token or _get_token()
    cmd = ['hf-mount', 'start']
    if token:
        cmd += ['--hf-token', token]
    cmd += [kind, name, str(mount_path)]

    subprocess.run(cmd, check=True)
    print(f"[hf-mount] Mounted {kind} {name} -> {mount_path}")
    return mount_path


def mount_bucket(bucket: str,
                 mount_path: Path = HF_MOUNT_DIR,
                 token: str | None = None) -> Path:
    """Montar un HF Bucket (read-write).

    Ideal para create_dataset.py: escribe parquets directo al bucket.
    No hace falta upload_to_hub() — los archivos van directo a HF storage.

    Args:
        bucket: nombre del bucket (e.g. "Cukinator/cpu1-ablation-dataset")
        mount_path: donde montar localmente
        token: HF token (o se lee de HF_TOKEN/Hugginface env var)
    """
    return _run_mount('bucket', bucket, mount_path, token)


def mount_repo(repo: str,
               mount_path: Path = HF_MOUNT_DIR,
               token: str | None = None) -> Path:
    """Montar un HF repo como read-only.

    Alternativa a mount_bucket para lectura.

    Args:
        repo: repo HF (e.g. "datasets/Cukinator/cpu1-ablation-dataset")
        mount_path: donde montar localmente
        token: HF token (o se lee de HF_TOKEN/Hugginface env var)
    """
    return _run_mount('repo', repo, mount_path, token)


def unmount(mount_path: Path = HF_MOUNT_DIR):
    """Desmontar bucket/repo previamente montado."""
    try:
        subprocess.run(['hf-mount', 'stop', str(mount_path)],
                       check=True, capture_output=True)
        print(f"[hf-mount] Unmounted {mount_path}")
    except Exception as e:
        print(f"[hf-mount] Unmount warning: {e}")
