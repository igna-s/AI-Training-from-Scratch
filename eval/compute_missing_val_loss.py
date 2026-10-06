"""
compute_missing_val_loss.py
===========================
Calcula val_loss para todos los runs que tienen `val_loss_available: false`
en sus JSONs de resultados, o que directamente no tienen JSON todavía.
Luego actualiza el README de HuggingFace.

Uso:
  python eval/compute_missing_val_loss.py
  python eval/compute_missing_val_loss.py --runs run_04_r2 run_07_r2
  python eval/compute_missing_val_loss.py --force          # recalcula todos
  python eval/compute_missing_val_loss.py --upload-readme  # sube a HF al final
  python eval/compute_missing_val_loss.py --no-full-eval   # solo val_loss
  python eval/compute_missing_val_loss.py --readme-only    # solo regenerar README
"""
import os
import sys
import json
import math
import time
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import torch
import torch.nn.functional as F

MODELS_FP32    = ROOT / "models" / "ablations_final"
MODELS_COMPACT = ROOT / "models" / "ablation_models"
RESULTADOS_DIR = ROOT / "resultados"
HF_README      = ROOT / "hf_readmes" / "cpu1-ablations-final.md"
HF_REPO_MODELS = "Cukinator/cpu1-ablations-final"

LN2          = math.log(2)
UNIFORM_BYTE = math.log(256)  # 5.545 nats

# ─────────────────────────────────────────────────────────────────────────────
# Todos los runs conocidos
# ─────────────────────────────────────────────────────────────────────────────
ALL_KNOWN_RUNS = [
    # 50M r1
    "run_01", "run_02a_byte_only_heads", "run_02", "run_03",
    "run_04", "run_05", "run_05b_kernel_strict", "run_06",
    "run_07", "run_08", "run_09", "run_10",
    # 10M r1
    "run_13", "run_14", "run_15", "run_16",
    # r2
    "run_04_r2", "run_05_r2", "run_07_r2", "run_08_r2",
    "run_09_r2", "run_10_r2",
    "run_13_r2", "run_14_r2", "run_15_r2", "run_16_r2",
    # v3
    "run_01_v3", "run_02a_byte_only_heads_v3", "run_02_v3", "run_03_v3",
    "run_04_v3", "run_05_v3", "run_05b_kernel_strict_v3", "run_06_v3",
    "run_08_v3", "run_13_v3", "run_14_v3", "run_15_v3",
    # r3 (si existen en disco)
    "run_04_r3", "run_07_r3", "run_14_r3", "run_15_r3",
]


def find_checkpoint(run_name: str) -> Path | None:
    fp32 = MODELS_FP32 / run_name / "model.pt"
    if fp32.exists():
        return fp32
    compact = MODELS_COMPACT / run_name / f"checkpoint_{run_name}_final.pt"
    if compact.exists():
        return compact
    return None


def load_existing_json(run_name: str) -> dict | None:
    p = RESULTADOS_DIR / f"ablation_{run_name}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def needs_val_loss(run_name: str) -> bool:
    j = load_existing_json(run_name)
    if j is None:
        return True
    lm = j.get("loss_metrics", {})
    return not lm.get("val_loss_available", False)


# ─────────────────────────────────────────────────────────────────────────────
# Carga del modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_config(ckpt_path: Path):
    from train_ablation import load_ablation_checkpoint, build_ablation_model

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    fmt = raw.get("format", "bf16")
    if fmt == "fp32":
        config = raw["config"]
        state  = raw["state_dict"]
    else:
        state, config = load_ablation_checkpoint(str(ckpt_path))

    config["use_gradient_checkpointing"] = False
    model = build_ablation_model(config)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config, raw


# ─────────────────────────────────────────────────────────────────────────────
# Cálculo de val_loss usando load_ablation_dataset de utilities
# ─────────────────────────────────────────────────────────────────────────────

def _load_hf_token() -> str | None:
    """Lee HF_TOKEN desde env o .env"""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_p = ROOT / ".env"
    if env_p.exists():
        for line in env_p.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN"):
                val = line.split("=", 1)[1].strip().strip('"').strip("'")
                if val:
                    return val
    return None


def compute_val_loss(model, config: dict, n_batches: int = 300,
                     batch_size: int = 4) -> float | None:
    """Calcula CE loss usando Cukinator (evita descargar FineWeb multi-GB)."""
    from utilities import load_ablation_dataset

    tok     = config.get("tokenization", "byte")
    is_byte = tok in ("byte", "byte_dynamic")
    vocab   = 256 if is_byte else int(config.get("bpe_vocab_size", 16384))

    # Inyectar HF token
    token = _load_hf_token()
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
        except Exception:
            pass

    # Forzar Cukinator (ya parcialmente cacheado, mucho mas liviano que FineWeb)
    cfg_val = dict(config)
    cfg_val["dataset_repo"] = "Cukinator/cpu1-ablation-dataset"
    if tok == "bpe" and "bpe_vocab_size" not in cfg_val:
        cfg_val["bpe_vocab_size"] = 4096

    print(f"  Cargando Cukinator dataset (tok={tok}, split=val)...")
    val_ds = None
    for split in ("val", "train"):
        try:
            val_ds = load_ablation_dataset(cfg_val, split=split)
            print(f"  Dataset cargado (split={split})")
            break
        except Exception as e:
            print(f"  [WARN] split='{split}' fallo: {e}")
    if val_ds is None:
        print("  [ERROR] No se pudo cargar dataset")
        return None

    loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    # Limitar seq_len por batch para que CPU no tarde demasiado
    # (seq_len=5000 × batch=4 = 20K tokens → >30s/batch en CPU)
    MAX_STEPS = 256  # pasos de patch — 256×4=1024 bytes máximo por ejemplo

    total_loss = 0.0
    n_done     = 0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for batch in loader:
            if n_done >= n_batches:
                break
            try:
                patches = batch["patches"] if is_byte else batch["input_ids"]
                targets = batch["targets"] if is_byte else batch["labels"]

                # Truncar secuencias largas para velocidad en CPU
                if patches.dim() >= 2 and patches.shape[1] > MAX_STEPS:
                    patches = patches[:, :MAX_STEPS]
                    targets = targets[:, :MAX_STEPS]

                result  = model(patches, **({'byte_targets': targets} if is_byte else {}))
                logits  = result[0]


                if is_byte:
                    if isinstance(logits, (tuple, list)):
                        nh = len(logits)
                        all_logits  = torch.cat([logits[i].reshape(-1, 256) for i in range(nh)], dim=0)
                        all_targets = torch.cat([targets[..., i].long().reshape(-1) for i in range(nh)], dim=0)
                    elif logits.dim() == 4:
                        all_logits  = logits.reshape(-1, logits.shape[-1])
                        all_targets = targets.long().reshape(-1)
                    else:
                        all_logits  = logits.reshape(-1, vocab)
                        all_targets = targets.long().reshape(-1)
                else:
                    all_logits  = logits.reshape(-1, vocab)
                    all_targets = targets.long().reshape(-1)

                mask = (all_targets >= 0) & (all_targets < vocab)
                if mask.sum() == 0:
                    continue

                loss = F.cross_entropy(all_logits[mask], all_targets[mask])
                total_loss += loss.item()
                n_done += 1

                if n_done % 50 == 0:
                    avg = total_loss / n_done
                    print(f"    batch {n_done}/{n_batches}  "
                          f"loss={avg:.4f}  ppl={math.exp(min(avg,88)):.2f}  "
                          f"t={time.perf_counter()-t0:.0f}s")
            except Exception as e:
                print(f"    [WARN] batch {n_done} error: {e}")

    if n_done == 0:
        print("  [ERROR] Ningun batch procesado")
        return None

    val_loss = total_loss / n_done
    print(f"\n  val_loss  = {val_loss:.6f} nats  ({n_done} batches, {time.perf_counter()-t0:.0f}s)")
    if is_byte:
        print(f"  PPL       = {math.exp(min(val_loss,88)):.3f}")
        print(f"  BPB       = {val_loss/LN2:.5f}  (delta_uniform = {val_loss-UNIFORM_BYTE:+.5f})")
    return val_loss



# ─────────────────────────────────────────────────────────────────────────────
# Métricas derivadas
# ─────────────────────────────────────────────────────────────────────────────

def make_loss_metrics(val_loss: float, is_byte: bool) -> dict:
    ppl = math.exp(min(val_loss, 88.0))
    r = {
        "val_loss":           round(val_loss, 6),
        "perplexity":         round(ppl, 3),
        "ppl_type":           "byte-level" if is_byte else "token-level",
        "val_loss_available": True,
    }
    if is_byte:
        r["bpb"]           = round(val_loss / LN2, 5)
        r["delta_uniform"] = round(val_loss - UNIFORM_BYTE, 5)
    return r


# ─────────────────────────────────────────────────────────────────────────────
# Eval completo para runs sin JSON
# ─────────────────────────────────────────────────────────────────────────────

def run_full_eval(run_name: str, ckpt_path: Path, model, config: dict, raw: dict,
                  val_loss: float | None) -> dict:
    from eval_ablations import (AblationInfo, AblationEngine,
                                extract_loss_metrics)
    from eval_correctness import run_correctness
    from eval_performance import run_performance
    from eval_quality     import run_quality

    info   = AblationInfo.from_config(run_name, config)
    engine = AblationEngine(model, config, info)

    record = {
        "run_name":      run_name,
        "ckpt_path":     str(ckpt_path),
        "ckpt_format":   raw.get("format", "?"),
        "step":          raw.get("step", "?"),
        "tok":           info.tok,
        "quant":         info.quant,
        "eff_bits":      1.58 if info.is_ternary else 16.0,
        "is_byte_level": info.is_byte_level,
        "is_recurrent":  info.is_recurrent,
        "n_params_M":    round(sum(p.numel() for p in model.parameters()) / 1e6, 3),
        "ram_fp32_mb":   round(sum(p.numel() * 4 for p in model.parameters()) / 1e6, 1),
    }

    if val_loss is not None:
        record["loss_metrics"] = make_loss_metrics(val_loss, info.is_byte_level)
    else:
        record["loss_metrics"] = extract_loss_metrics(raw, info)

    t0 = time.perf_counter()
    print(f"  Correctness...")
    record["correctness"] = run_correctness(engine)
    print(f"  Correctness OK ({time.perf_counter()-t0:.0f}s)")

    t0 = time.perf_counter()
    print(f"  Performance...")
    record["performance"] = run_performance(engine)
    print(f"  Performance OK ({time.perf_counter()-t0:.0f}s)")

    t0 = time.perf_counter()
    print(f"  Quality...")
    record["quality"] = run_quality(engine, dataset=None)
    print(f"  Quality OK ({time.perf_counter()-t0:.0f}s)")

    record["dataset_eval"]  = {"skipped": True, "reason": "sin dataset local"}
    record["external_eval"] = {"skipped": True, "reason": "sin GGUF (no aplica a ablation)"}
    return record


# ─────────────────────────────────────────────────────────────────────────────
# README update
# ─────────────────────────────────────────────────────────────────────────────

def update_readme() -> None:
    from eval_ablations import build_readme

    # Leer TODOS los JSONs existentes
    all_results = []
    for p in sorted(RESULTADOS_DIR.glob("ablation_*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                all_results.append(json.load(f))
        except Exception as e:
            print(f"  [WARN] No se pudo leer {p.name}: {e}")

    if not all_results:
        print("  [WARN] No hay resultados para generar README")
        return

    readme_txt = build_readme(all_results)
    HF_README.parent.mkdir(parents=True, exist_ok=True)
    HF_README.write_text(readme_txt, encoding="utf-8")
    print(f"  README actualizado: {HF_README}  ({len(all_results)} runs)")


def upload_readme_to_hf() -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [WARN] huggingface_hub no instalado")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        env_p = ROOT / ".env"
        if env_p.exists():
            for line in env_p.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN"):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        print("  [WARN] No hay HF_TOKEN")
        return

    api = HfApi(token=token)
    print(f"  Subiendo README a {HF_REPO_MODELS}...")
    try:
        api.upload_file(
            path_or_fileobj = str(HF_README),
            path_in_repo    = "README.md",
            repo_id         = HF_REPO_MODELS,
            repo_type       = "model",
            commit_message  = (
                f"Update README: computed val_loss for missing runs "
                f"{time.strftime('%Y-%m-%d')}"
            ),
        )
        print("  README subido OK")
    except Exception as e:
        print(f"  [ERROR upload] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calcula val_loss para runs con datos faltantes y actualiza README HF"
    )
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Subset de runs (default: todos los que falten)")
    parser.add_argument("--force", action="store_true",
                        help="Recalcular val_loss aunque ya exista")
    parser.add_argument("--n-batches", type=int, default=300,
                        help="Batches de validacion por run (default: 300)")
    parser.add_argument("--no-full-eval", action="store_true",
                        help="Solo val_loss, sin correctness/performance/quality")
    parser.add_argument("--upload-readme", action="store_true",
                        help="Subir README actualizado a HuggingFace al final")
    parser.add_argument("--readme-only", action="store_true",
                        help="Solo regenerar README desde JSONs existentes")
    args = parser.parse_args()

    if args.readme_only:
        print("\nRegenerando README desde JSONs existentes...")
        update_readme()
        if args.upload_readme:
            upload_readme_to_hf()
        print("Listo.")
        return

    # Detectar qué runs necesitan procesamiento
    candidate_runs = args.runs or ALL_KNOWN_RUNS
    runs_with_ckpt = [(r, find_checkpoint(r)) for r in candidate_runs]
    runs_with_ckpt = [(r, p) for r, p in runs_with_ckpt if p is not None]

    if args.force:
        to_process = runs_with_ckpt
    else:
        to_process = [(r, p) for r, p in runs_with_ckpt if needs_val_loss(r)]

    if not to_process:
        print("\nNo hay runs con val_loss faltante.")
        print("Actualizando README con datos existentes...")
        update_readme()
        if args.upload_readme:
            upload_readme_to_hf()
        return

    print(f"\n{'═'*70}")
    print(f"  compute_missing_val_loss — {len(to_process)} runs a procesar")
    print(f"  n_batches={args.n_batches}  full_eval={not args.no_full_eval}")
    print(f"{'═'*70}")
    for r, p in to_process:
        has_json = load_existing_json(r) is not None
        tag = "[JSON existe]" if has_json else "[JSON nuevo ]"
        print(f"  {tag} {r:<40} → {p.name}")

    t_total = time.perf_counter()

    for run_name, ckpt_path in to_process:
        print(f"\n{'─'*70}")
        print(f"  RUN: {run_name}  ({ckpt_path})")
        print(f"{'─'*70}")
        t0 = time.perf_counter()

        try:
            model, config, raw = load_model_and_config(ckpt_path)
        except Exception as e:
            print(f"  [ERROR carga] {e}")
            continue

        tok     = config.get("tokenization", "byte")
        is_byte = tok in ("byte", "byte_dynamic")

        # ── Calcular val_loss ───────────────────────────────────────────────
        print(f"\n  Calculando val_loss ({args.n_batches} batches)...")
        val_loss = compute_val_loss(model, config, n_batches=args.n_batches)

        # ── Actualizar o crear JSON ─────────────────────────────────────────
        existing_json = load_existing_json(run_name)
        has_json      = existing_json is not None

        if args.no_full_eval and has_json:
            # Solo parchear loss_metrics en el JSON existente
            existing_json["loss_metrics"] = (
                make_loss_metrics(val_loss, is_byte)
                if val_loss is not None
                else {"val_loss_available": False, "error": "compute failed"}
            )
            record = existing_json
        else:
            # Eval completo (si no hay JSON o si queremos datos frescos)
            record = run_full_eval(run_name, ckpt_path, model, config, raw, val_loss)
            record["eval_time_s"] = round(time.perf_counter() - t0, 1)

        # Guardar JSON
        out_p = RESULTADOS_DIR / f"ablation_{run_name}.json"
        RESULTADOS_DIR.mkdir(exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False, default=str)
        print(f"  JSON guardado: {out_p}")

        # Resumen consola
        lm = record.get("loss_metrics", {})
        if lm.get("val_loss_available"):
            print(f"\n  ── Resumen {run_name} ({record.get('eval_time_s', '?')}s) ──")
            print(f"  val_loss   = {lm['val_loss']:.6f}  [{lm['ppl_type']}]")
            print(f"  perplexity = {lm['perplexity']}")
            if "bpb" in lm:
                print(f"  BPB        = {lm['bpb']:.5f}  (Δuniform = {lm['delta_uniform']:+.5f})")
        else:
            print(f"  [WARN] val_loss no disponible")

    # ── Regenerar README ────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print("  Actualizando README...")
    update_readme()

    if args.upload_readme:
        upload_readme_to_hf()

    elapsed = time.perf_counter() - t_total
    print(f"\n{'═'*70}")
    print(f"  Listo — {len(to_process)} runs en {elapsed:.0f}s")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
