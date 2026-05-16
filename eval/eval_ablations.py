"""
Descarga y evalúa los checkpoints del ablation study desde HuggingFace.

Repo: Cukinator/cpu1-ablation-checkpoints
Checkpoints disponibles: run_01..run_13 (11 runs finales)

Para cada checkpoint:
  - Descarga desde HuggingFace → models/ablation_models/
  - Extrae métricas del entrenamiento (val_loss, perplexity, n_params)
  - Mide throughput CPU (patches/seg)
  - Genera una muestra de texto
  - Guarda JSON individual + tabla resumen Markdown

Uso:
  python eval/eval_ablations.py
  python eval/eval_ablations.py --skip-download
  python eval/eval_ablations.py --runs run_05 run_07
  python eval/eval_ablations.py --only-download
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

HF_REPO = "Cukinator/cpu1-ablation-checkpoints"
MODELS_DIR = ROOT / "models" / "ablation_models"
RESULTADOS_DIR = ROOT / "resultados"

# Runs disponibles en HuggingFace (confirmados)
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

# Prompts de generación — byte runs usan el primero, bpe runs el segundo
BYTE_PROMPT = "The quick brown fox"
BPE_PROMPT = "The quick brown fox"
MAX_NEW = 128  # bytes / tokens a generar


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


def download_checkpoint(run_name: str, token: str | None) -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"{run_name}/checkpoint_{run_name}_final.pt"
    local_dir = MODELS_DIR / run_name
    local_dir.mkdir(parents=True, exist_ok=True)

    dest = local_dir / f"checkpoint_{run_name}_final.pt"
    if dest.exists():
        print(f"  [skip] {run_name} ya existe ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"  Descargando {run_name}...")
    t0 = time.perf_counter()
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        local_dir=str(MODELS_DIR),
        token=token,
    )
    elapsed = time.perf_counter() - t0
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  OK — {size_mb:.1f} MB en {elapsed:.1f}s  ->  {path}")
    return Path(path)


def evaluate_checkpoint(run_name: str, ckpt_path: Path) -> dict:
    """Carga el checkpoint, mide throughput y genera texto."""
    import torch
    from train_ablation import (
        load_ablation_checkpoint,
        build_ablation_model,
        generate,
        measure_cpu_throughput,
    )

    result = {"run_name": run_name, "ckpt_path": str(ckpt_path)}

    # — Cargar checkpoint —
    t_load = time.perf_counter()
    try:
        state, config = load_ablation_checkpoint(str(ckpt_path))
    except Exception as e:
        result["error"] = f"load_checkpoint: {e}"
        return result
    result["load_s"] = round(time.perf_counter() - t_load, 2)

    # — Métricas del entrenamiento (guardadas en el .pt) —
    try:
        raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        train_metrics = raw.get("metrics", {})
        result["train_metrics"] = {
            k: (round(v, 4) if isinstance(v, float) else v)
            for k, v in train_metrics.items()
        }
        result["step"] = raw.get("step", "?")
        result["format"] = raw.get("format", "?")
    except Exception as e:
        result["train_metrics"] = {}
        result["step_error"] = str(e)

    # — Construir modelo —
    config["use_gradient_checkpointing"] = False
    try:
        model = build_ablation_model(config)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            result["missing_keys"] = len(missing)
        if unexpected:
            result["unexpected_keys"] = len(unexpected)
    except Exception as e:
        result["error"] = f"build_model: {e}"
        return result

    model = model.eval()
    device = torch.device("cpu")

    # — Parámetros y RAM estimada —
    n_params = sum(p.numel() for p in model.parameters())
    result["n_params_M"] = round(n_params / 1e6, 2)
    result["ram_mb"] = round(
        sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6, 1
    )

    # — Throughput CPU —
    tok = config.get("tokenization", "byte")
    try:
        t0 = time.perf_counter()
        tps = measure_cpu_throughput(model, config, n_tokens=50)
        result["throughput_patches_per_sec"] = round(tps, 2)
        result["throughput_bytes_per_sec"] = round(tps * config.get("patch_size", 4), 2)
        result["throughput_time_s"] = round(time.perf_counter() - t0, 2)
    except Exception as e:
        result["throughput_error"] = str(e)

    # — Generación de texto —
    prompt = BYTE_PROMPT if tok in ("byte", "byte_dynamic") else BPE_PROMPT
    try:
        t0 = time.perf_counter()
        text = generate(
            model, prompt, MAX_NEW, config, device,
            temperature=0.8, top_k=40, top_p=0.0,
        )
        result["generation_time_s"] = round(time.perf_counter() - t0, 2)
        result["generated_text"] = ''.join(
            c if c.isprintable() or c == '\n' else '�' for c in text
        )
        result["generated_bytes"] = len(text.encode("utf-8"))
    except Exception as e:
        result["generation_error"] = str(e)

    return result


def save_json(run_name: str, data: dict) -> Path:
    RESULTADOS_DIR.mkdir(exist_ok=True)
    path = RESULTADOS_DIR / f"ablation_{run_name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


def save_summary_markdown(all_results: list[dict]) -> Path:
    RESULTADOS_DIR.mkdir(exist_ok=True)
    path = RESULTADOS_DIR / "ablation_summary.md"

    lines = [
        "# Ablation Study — Resultados CPU",
        "",
        f"Repo: `{HF_REPO}`  |  Fecha: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Tabla comparativa",
        "",
        "| Run | Nombre | Params M | RAM MB | Val Loss | Perplexity | Patches/s | Bytes/s |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for r in all_results:
        if "error" in r and "train_metrics" not in r:
            lines.append(f"| {r['run_name']} | ERROR | — | — | — | — | — | — |")
            continue
        tm = r.get("train_metrics", {})
        name = tm.get("run_name", r["run_name"])
        params = r.get("n_params_M", "?")
        ram = r.get("ram_mb", "?")
        val_loss = tm.get("val_loss", "?")
        ppl = tm.get("perplexity", "?")
        pps = r.get("throughput_patches_per_sec", "?")
        bps = r.get("throughput_bytes_per_sec", "?")

        def fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else str(v)

        lines.append(
            f"| {r['run_name']} | {name} | {fmt(params)} | {fmt(ram)} "
            f"| {fmt(val_loss)} | {fmt(ppl)} | {fmt(pps)} | {fmt(bps)} |"
        )

    lines += ["", "## Muestras generadas", ""]
    for r in all_results:
        if "generated_text" in r:
            lines += [
                f"### {r['run_name']}",
                "",
                f"> {r['generated_text'][:300]}",
                "",
            ]
        elif "generation_error" in r:
            lines += [
                f"### {r['run_name']}",
                "",
                f"*Error: {r['generation_error']}*",
                "",
            ]

    lines += ["", "## Errores / Warnings", ""]
    any_error = False
    for r in all_results:
        for key in ("error", "missing_keys", "unexpected_keys",
                    "throughput_error", "generation_error"):
            if key in r:
                lines.append(f"- **{r['run_name']}** `{key}`: {r[key]}")
                any_error = True
    if not any_error:
        lines.append("*Ninguno*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Descarga y evalúa checkpoints del ablation study"
    )
    parser.add_argument("--skip-download", action="store_true",
                        help="No descargar (usar checkpoints ya existentes)")
    parser.add_argument("--only-download", action="store_true",
                        help="Solo descargar, no evaluar")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Evaluar sólo estos runs (ej: run_05 run_07)")
    args = parser.parse_args()

    runs = args.runs or AVAILABLE_RUNS
    token = get_hf_token()

    print(f"\n{'=' * 65}")
    print(f"  Ablation Eval — {HF_REPO}")
    print(f"  Runs: {runs}")
    print(f"  HF_TOKEN: {'****' + token[-4:] if token else 'None'}")
    print(f"{'=' * 65}\n")

    # === FASE 1: descarga ===
    ckpt_paths: dict[str, Path] = {}
    if not args.skip_download:
        print("## Descargando checkpoints\n")
        for run_name in runs:
            try:
                p = download_checkpoint(run_name, token)
                ckpt_paths[run_name] = p
            except Exception as e:
                print(f"  [ERROR] {run_name}: {e}")
    else:
        print("## Usando checkpoints locales\n")
        for run_name in runs:
            p = MODELS_DIR / run_name / f"checkpoint_{run_name}_final.pt"
            if p.exists():
                ckpt_paths[run_name] = p
                print(f"  [ok] {run_name} -> {p}")
            else:
                print(f"  [missing] {run_name} — {p}")

    if args.only_download:
        print("\n--only-download: listo.")
        return

    # === FASE 2: evaluación ===
    print(f"\n## Evaluando {len(ckpt_paths)} modelos\n")
    all_results = []
    t_total = time.perf_counter()

    for run_name, ckpt_path in ckpt_paths.items():
        print(f"\n{'─' * 65}")
        print(f"  Evaluando: {run_name}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"{'─' * 65}")
        t0 = time.perf_counter()

        result = evaluate_checkpoint(run_name, ckpt_path)
        result["eval_time_s"] = round(time.perf_counter() - t0, 1)

        if "error" in result:
            print(f"  [ERROR] {result['error']}")
        else:
            tm = result.get("train_metrics", {})
            print(f"  Params:      {result.get('n_params_M', '?')} M")
            print(f"  RAM:         {result.get('ram_mb', '?')} MB")
            print(f"  Val Loss:    {tm.get('val_loss', '?')}")
            print(f"  Perplexity:  {tm.get('perplexity', '?')}")
            print(f"  Throughput:  {result.get('throughput_patches_per_sec', '?')} patches/s")
            if "generated_text" in result:
                preview = result["generated_text"][:120].replace("\n", " ")
                print(f"  Generado:    {preview!r}")

        json_path = save_json(run_name, result)
        print(f"  JSON: {json_path}")
        all_results.append(result)

    # === FASE 3: resumen ===
    md_path = save_summary_markdown(all_results)
    elapsed = time.perf_counter() - t_total

    print(f"\n\n{'=' * 65}")
    print(f"  Listo — {len(all_results)} modelos evaluados en {elapsed:.0f}s")
    print(f"  Resumen: {md_path}")
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
