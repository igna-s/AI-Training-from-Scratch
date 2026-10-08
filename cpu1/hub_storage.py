"""Optional lazy HF mounts, recovered from the historical hf_mount helper.

Install hf-mount separately from https://github.com/huggingface/hf-mount.
Credentials are passed through the child environment, never command arguments.
"""
import os
from pathlib import Path
import shutil
import subprocess
import sys

HF_MOUNT_DIR = Path("/tmp/hf_mount_dataset")


def _run_mount(kind: str, name: str, mount_path: str | Path, token: str | None = None) -> Path:
    executable = shutil.which("hf-mount")
    if sys.platform == "win32" or not executable:
        raise RuntimeError("Install hf-mount on Linux/macOS, or use standard Hub streaming")
    path = Path(mount_path)
    path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    credential = token or env.get("HF_TOKEN") or env.get("Hugginface")
    if credential:
        env["HF_TOKEN"] = credential
    command = [executable, "start", kind, name, str(path)]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"hf-mount failed with exit code {result.returncode}")
    print(f"[hf-mount] Mounted {kind} {name} -> {path}")
    return path


def mount_bucket(bucket: str, mount_path: str | Path = HF_MOUNT_DIR, token: str | None = None) -> Path:
    return _run_mount("bucket", bucket, mount_path, token)


def mount_repo(repo: str, mount_path: str | Path = HF_MOUNT_DIR, token: str | None = None) -> Path:
    return _run_mount("repo", repo, mount_path, token)


def unmount(mount_path: str | Path = HF_MOUNT_DIR) -> None:
    executable = shutil.which("hf-mount")
    if not executable:
        raise RuntimeError("hf-mount is not installed")
    result = subprocess.run([executable, "stop", str(mount_path)], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"hf-mount stop failed with exit code {result.returncode}")
