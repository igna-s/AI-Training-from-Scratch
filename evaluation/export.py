"""
Desempaqueta los checkpoints compactos (2-bit ternary) del ablation study,
guarda versiones float32 listas para usar y las sube a HuggingFace.

Repo destino: Cukinator/cpu1-ablations-final
Modelos locales: models/ablations_final/{run_name}/

Uso:
  python -m evaluation.export
  python -m evaluation.export --runs run_02 run_03
  python -m evaluation.export --skip-upload   # solo exporta localmente
  python -m evaluation.export --skip-export   # solo sube los ya exportados
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

HF_REPO_SRC  = "Cukinator/cpu1-ablation-checkpoints"
HF_REPO_DEST = "Cukinator/cpu1-ablations-final"
MODELS_DIR   = ROOT / "local" / "models" / "ablation_models"
EXPORT_DIR   = ROOT / "local" / "models" / "ablations_final"

AVAILABLE_RUNS = [
    "run_01",
    "run_02",
    "run_02a_byte_only_heads",
    "run_03",
    "run_04",
    "run_05",
    "run_05b_kernel_strict",
    "run_06",
    "run_07",
    "run_08",
    "run_09",
    "run_10",
    "run_13",
    "run_14",
    "run_15",
    "run_16",
    # Round-2 re-runs at 15 tok/param
    "run_04_r2",
    "run_05_r2",
    "run_07_r2",
    "run_08_r2",
    "run_09_r2",
    "run_10_r2",
    "run_13_r2",
    "run_14_r2",
    "run_15_r2",
    "run_16_r2",
    # Round-3 cold-start rescue (bf16 AMP + lr_scale=2 + CE-only + 50 tok/p)
    "run_04_r3",
    "run_07_r3",
    "run_14_r3",
    "run_15_r3",
]

USAGE_README = (Path(__file__).parent.parent / "docs" / "model_cards"
                / "cpu1-ablations-final.md").read_text(encoding="utf-8")


def get_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("HF_TOKEN"):
                    token = line.split("=", 1)[1].strip().strip('"')
                    break
    return token or None


def export_run(run_name: str) -> Path | None:
    """Desempaqueta el checkpoint compacto y guarda un .pt float32."""
    import torch
    from training.ablation import load_ablation_checkpoint

    src = MODELS_DIR / run_name / f"checkpoint_{run_name}_final.pt"
    if not src.exists():
        print(f"  [missing] {src}")
        return None

    out_dir = EXPORT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model.pt"

    if out_path.exists():
        print(f"  [skip] {run_name} ya exportado ({out_path.stat().st_size / 1e6:.1f} MB)")
        return out_path

    print(f"  Exportando {run_name}...")
    t0 = time.perf_counter()

    try:
        state, config = load_ablation_checkpoint(str(src))
    except Exception as e:
        print(f"  [ERROR] load: {e}")
        return None

    # Métricas del checkpoint original
    import torch as _torch
    raw = _torch.load(str(src), map_location="cpu", weights_only=False)
    metrics = raw.get("metrics", {})
    step    = raw.get("step", "?")

    payload = {
        "state_dict": state,
        "config":     config,
        "metrics":    metrics,
        "step":       step,
        "run_name":   run_name,
        "format":     "fp32",
        "source_repo": HF_REPO_SRC,
    }

    import torch as _torch
    _torch.save(payload, out_path)
    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  [ok] {run_name} → {out_path}  ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return out_path


def upload_run(run_name: str, token: str | None) -> bool:
    """Sube models/ablations_final/{run_name}/ al repo HF."""
    from huggingface_hub import HfApi

    out_path = EXPORT_DIR / run_name / "model.pt"
    if not out_path.exists():
        print(f"  [skip upload] {run_name} — archivo no encontrado")
        return False

    api = HfApi(token=token)

    print(f"  Subiendo {run_name} ({out_path.stat().st_size / 1e6:.1f} MB)...")
    t0 = time.perf_counter()
    try:
        api.upload_file(
            path_or_fileobj=str(out_path),
            path_in_repo=f"{run_name}/model.pt",
            repo_id=HF_REPO_DEST,
            repo_type="model",
            commit_message=f"Add {run_name} (float32, unpacked)",
        )
        elapsed = time.perf_counter() - t0
        print(f"  [ok] {run_name} subido en {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"  [ERROR] upload {run_name}: {e}")
        return False


def ensure_repo(token: str | None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=HF_REPO_DEST, repo_type="model")
        print(f"  Repo existente: {HF_REPO_DEST}")
    except Exception:
        print(f"  Creando repo: {HF_REPO_DEST} ...")
        api.create_repo(
            repo_id=HF_REPO_DEST,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
        print(f"  [ok] Repo creado")


def upload_readme(token: str | None):
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                         delete=False, encoding="utf-8") as f:
            f.write(USAGE_README)
            tmp = f.name
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo="README.md",
            repo_id=HF_REPO_DEST,
            repo_type="model",
            commit_message="Add README",
        )
        os.unlink(tmp)
        print("  [ok] README.md subido")
    except Exception as e:
        print(f"  [warn] README upload: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Exporta y sube checkpoints ablation desempaquetados"
    )
    parser.add_argument("--runs", nargs="+", default=None)
    parser.add_argument("--skip-upload", action="store_true",
                        help="Solo exporta localmente, no sube a HF")
    parser.add_argument("--skip-export", action="store_true",
                        help="Solo sube los .pt ya exportados (no re-exporta)")
    args = parser.parse_args()

    runs  = args.runs or AVAILABLE_RUNS
    token = get_hf_token()

    print(f"\n{'=' * 65}")
    print(f"  Export ablations -- {HF_REPO_SRC} -> {HF_REPO_DEST}")
    print(f"  Runs: {runs}")
    print(f"  HF_TOKEN: {'****' + token[-4:] if token else 'None'}")
    print(f"  Local export dir: {EXPORT_DIR}")
    print(f"{'=' * 65}\n")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # — Fase 1: exportar localmente —
    if not args.skip_export:
        print("## Exportando checkpoints (2-bit → float32)\n")
        for run in runs:
            export_run(run)

    # — Fase 2: subir a HF —
    if not args.skip_upload:
        if not token:
            print("\n[ERROR] No hay HF_TOKEN — no se puede subir.")
            return

        print("\n## Subiendo a HuggingFace\n")
        ensure_repo(token)
        upload_readme(token)

        ok = 0
        for run in runs:
            if upload_run(run, token):
                ok += 1

        print(f"\n{'=' * 65}")
        print(f"  Listo — {ok}/{len(runs)} modelos subidos a {HF_REPO_DEST}")
        print(f"{'=' * 65}\n")
    else:
        print(f"\nExportacion local completa → {EXPORT_DIR}")


if __name__ == "__main__":
    main()
