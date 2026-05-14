"""
Desempaqueta los checkpoints compactos (2-bit ternary) del ablation study,
guarda versiones float32 listas para usar y las sube a HuggingFace.

Repo destino: Cukinator/cpu1-ablations-final
Modelos locales: models/ablations_final/{run_name}/

Uso:
  python eval/export_ablations.py
  python eval/export_ablations.py --runs run_02 run_03
  python eval/export_ablations.py --skip-upload   # solo exporta localmente
  python eval/export_ablations.py --skip-export   # solo sube los ya exportados
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
MODELS_DIR   = ROOT / "models" / "ablation_models"
EXPORT_DIR   = ROOT / "models" / "ablations_final"

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
    "run_10",
    "run_13",
]

USAGE_README = """\
---
language:
  - en
license: apache-2.0
tags:
  - pytorch
  - text-generation
  - 1.58bit
  - ternary
  - byte-level
  - mlgru
  - ablation
library_name: pytorch
pipeline_tag: text-generation
model_type: custom
---

# CPU-1 Ablation Study — Ready-to-use Checkpoints

Repo: `Cukinator/cpu1-ablations-final`
Source (compact 2-bit): `Cukinator/cpu1-ablation-checkpoints`

Each checkpoint is a standard PyTorch `.pt` file with **float32 weights** —
no unpacking needed. Compatible with `train_ablation.py` from the
[1.58bits repo](https://github.com/Cukinator/1.58bits).

11 progressive ablation runs, each adding one component vs the previous.
Most runs are **~50M parameters**; run_13 is a small 10M variant.

## Quick start

```python
import torch, sys
sys.path.insert(0, "/path/to/1.58bits")
from train_ablation import build_ablation_model, generate

ckpt = torch.load("run_02/model.pt", map_location="cpu")
model = build_ablation_model(ckpt["config"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

text = generate(model, "The quick brown fox", 128, ckpt["config"], torch.device("cpu"))
print(text)
```

## Ablation chain

| Run | Architecture | Params | Val Loss | Perplexity |
|-----|-------------|--------|----------|-----------|
| run_01 | Transformer + BPE + FP16 (baseline) | 54.7M | 4.66 | 106.1 |
| run_02a | Transformer + Byte + 4 heads (no LBD) | 38.5M | 2.31 | 10.1 |
| run_02 | Transformer + Byte + LocalByteDecoder | 38.8M | 1.72 | 5.56 |
| run_03 | MLGRU + Byte + FP16 | 38.8M | 1.87 | 6.49 |
| run_04 | MLGRU + Byte + Ternary | 38.9M | 5.57 | 261.7 |
| run_05 | + FPResidual | 39.0M | 5.55 | 257.7 |
| run_05b | MLGRU kernel strict | 35.8M | 5.59 | 268.8 |
| run_06 | + BolmoPatchEmbedding | 39.0M | 5.56 | 258.8 |
| run_07 | + DeleteGate (CPU-1 complete) | 39.0M | 5.56 | 258.8 |
| run_10 | + learned per-channel decay | 39.4M | 5.53 | 253.1 |
| run_13 | Small 10M BPE model | 12.5M | 30.5 | — |

> **Note**: ternary runs (04–10) were trained with only 2 tokens/param
> (ablation budget). High perplexity reflects under-training, not architecture
> failure. FP16 runs (01–03) are valid references.
"""


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
    from train_ablation import load_ablation_checkpoint

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
