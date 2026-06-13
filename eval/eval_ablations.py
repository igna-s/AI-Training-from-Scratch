"""
eval_ablations.py — Suite de evaluación completa para todos los modelos ablation.

Usa los módulos eval existentes de la carpeta (no duplica lógica):
  eval_correctness  → run_correctness(engine)
  eval_performance  → run_performance(engine)    [PyTorch puro, sin kernels nuestros]
  eval_quality      → run_quality(engine, dataset=None)
  eval_dataset      → run_dataset_checks(dataset) [skip: sin dataset local]
  eval_external     → skip (no hay GGUF de ablation)

El adaptador AblationEngine hace que cualquier AblationModel sea compatible
con la interfaz que usan esos módulos (diseñada para CPU1InferenceEngine).

Criterio homogéneo de métricas:
  perplexity = exp(val_loss)  para TODOS los runs sin excepción
  bpb        = val_loss / ln(2)  solo runs byte-level
  Referencia uniforme byte: ln(256) = 5.545 nats → PPL=256 → BPB=8.0

  BPE runs (run_01, run_13, *_r2): PPL es TOKEN-level.
  Byte runs (todos los demás):     PPL es BYTE-level.

Casos especiales:
  generation (eval_quality): BPE runs → N/A
  state_propagation (eval_correctness): Transformer → N/A
  state_size / ternary_weights: idem condicionados por arquitectura

Fuentes de checkpoints (en orden de preferencia):
  1. models/ablation_models/{run}/checkpoint_{run}_final.pt  (compact_2bit)
  2. models/ablations_final/{run}/model.pt                   (fp32)

Uso:
  python eval/eval_ablations.py                          # todos los runs locales
  python eval/eval_ablations.py --runs run_02 run_07     # runs concretos
  python eval/eval_ablations.py --no-performance         # omitir benchmark CPU
  python eval/eval_ablations.py --update-readme          # actualizar HF readme
"""

import os
import sys
import json
import math
import time

# Forzar UTF-8 en stdout/stderr para que los caracteres Unicode (═, ─, ⭐, etc.)
# funcionen en Windows con cp1252 por defecto.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import argparse
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch

MODELS_COMPACT  = ROOT / "models" / "ablation_models"
MODELS_FP32     = ROOT / "models" / "ablations_final"
RESULTADOS_DIR  = ROOT / "resultados"
HF_README       = ROOT / "hf_readmes" / "cpu1-ablations-final.md"
HF_REPO_MODELS  = "Cukinator/cpu1-ablations-final"

LN2             = math.log(2)
UNIFORM_BYTE    = math.log(256)   # 5.545 nats — entropía uniforme byte


# ─────────────────────────────────────────────────────────────────────────────
# Metadatos de cada run (para las tablas del README)
# ─────────────────────────────────────────────────────────────────────────────

RUN_META = {
    # ── 50M, Round 1 (2 tok/param) ──
    "run_01":                 {"label": "Transformer + BPE 16K + FP16",             "group": "50M", "round": "r1"},
    "run_02a_byte_only_heads":{"label": "Transformer + Byte + 4 heads (no LBD)",    "group": "50M", "round": "r1"},
    "run_02":                 {"label": "Transformer + Byte + LocalByteDecoder",     "group": "50M", "round": "r1"},
    "run_03":                 {"label": "MLGRU + Byte + FP16",                       "group": "50M", "round": "r1"},
    "run_04":                 {"label": "MLGRU + Byte + Ternary",                    "group": "50M", "round": "r1"},
    "run_05":                 {"label": "+ FPResidual",                              "group": "50M", "round": "r1"},
    "run_05b_kernel_strict":  {"label": "MLGRU kernel-strict (no W_o)",              "group": "50M", "round": "r1"},
    "run_06":                 {"label": "+ Bolmo patch embedding",                   "group": "50M", "round": "r1"},
    "run_07":                 {"label": "+ DeleteGate (CPU-1 completo) ⭐",          "group": "50M", "round": "r1"},
    "run_08":                 {"label": "Folded Transformer + Byte + Ternary",       "group": "50M", "round": "r1"},
    "run_09":                 {"label": "+ PFNet",                                   "group": "50M", "round": "r1"},
    "run_10":                 {"label": "+ decay per-canal aprendido",               "group": "50M", "round": "r1"},
    # ── 10M, Round 1 ──
    "run_13":                 {"label": "Small CPU-1 + BPE 4K",                     "group": "10M", "round": "r1"},
    "run_14":                 {"label": "Small CPU-1 byte (Qwen distill.)",          "group": "10M", "round": "r1"},
    "run_15":                 {"label": "Small CPU-1 byte + hidden distill.",        "group": "10M", "round": "r1"},
    "run_16":                 {"label": "Small CPU-1 raw bytes, no teacher",         "group": "10M", "round": "r1"},
    # ── Round 2 (15 tok/param) ──
    "run_04_r2":              {"label": "run_04 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_05_r2":              {"label": "run_05 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_07_r2":              {"label": "run_07 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_08_r2":              {"label": "run_08 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_09_r2":              {"label": "run_09 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_10_r2":              {"label": "run_10 @ 15 tok/param",                    "group": "50M", "round": "r2"},
    "run_13_r2":              {"label": "run_13 @ 15 tok/param",                    "group": "10M", "round": "r2"},
    "run_14_r2":              {"label": "run_14 @ 15 tok/param",                    "group": "10M", "round": "r2"},
    "run_15_r2":              {"label": "run_15 @ 15 tok/param",                    "group": "10M", "round": "r2"},
    "run_16_r2":              {"label": "run_16 @ 15 tok/param",                    "group": "10M", "round": "r2"},
    # ── Round 3 (v3) ──
    "run_01_v3":              {"label": "run_01 v3",                                "group": "50M", "round": "v3"},
    "run_02_v3":              {"label": "run_02 v3",                                "group": "50M", "round": "v3"},
    "run_02a_byte_only_heads_v3": {"label": "run_02a v3",                           "group": "50M", "round": "v3"},
    "run_03_v3":              {"label": "run_03 v3",                                "group": "50M", "round": "v3"},
    "run_04_v3":              {"label": "run_04 v3",                                "group": "50M", "round": "v3"},
    "run_05_v3":              {"label": "run_05 v3",                                "group": "50M", "round": "v3"},
    "run_05b_kernel_strict_v3": {"label": "run_05b v3",                             "group": "50M", "round": "v3"},
    "run_06_v3":              {"label": "run_06 v3",                                "group": "50M", "round": "v3"},
    "run_08_v3":              {"label": "run_08 v3",                                "group": "50M", "round": "v3"},
    "run_13_v3":              {"label": "run_13 v3",                                "group": "10M", "round": "v3"},
    "run_14_v3":              {"label": "run_14 v3",                                "group": "10M", "round": "v3"},
    "run_15_v3":              {"label": "run_15 v3",                                "group": "10M", "round": "v3"},
}


# ─────────────────────────────────────────────────────────────────────────────
# AblationInfo — compatible con la interfaz ModelInfo de los eval_* módulos
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AblationInfo:
    """
    Equivalente a ModelInfo (inference.py) pero para AblationModel.
    Todos los eval_* módulos acceden solo a atributos, nunca a isinstance.
    """
    run_name:      str
    model_class:   str    # config['name']
    tok:           str    # byte / byte_dynamic / bpe
    quant:         str    # ternary / fp16
    is_byte_level: bool
    is_recurrent:  bool   # True → MLGRU (estado persistente), False → Transformer
    is_ternary:    bool
    use_local_dec: bool   # LocalByteDecoder
    patch_size:    int    # bytes por patch
    vocab_size:    int    # 256 (byte) / bpe_vocab_size
    n_heads:       int    # patch_size para byte, 1 para BPE
    d_model:       int
    n_layers:      int

    @staticmethod
    def from_config(run_name: str, config: dict) -> "AblationInfo":
        tok   = config.get("tokenization", "byte")
        quant = config.get("quantization", "fp16")
        mixer = config.get("token_mixer", "mlgru")
        patch = int(config.get("patch_size", 4))
        is_byte = tok in ("byte", "byte_dynamic")
        use_lbd = is_byte and config.get("use_local_decoder", True)
        vocab = 256 if is_byte else int(config.get("bpe_vocab_size", 16384))
        # n_heads: siempre patch_size para byte (LocalDecoder se desempaqueta
        # en _get_logits_list de eval_correctness a P tensores [B,S,256])
        n_heads = patch if is_byte else 1
        return AblationInfo(
            run_name      = run_name,
            model_class   = config.get("name", run_name),
            tok           = tok,
            quant         = quant,
            is_byte_level = is_byte,
            is_recurrent  = (mixer == "mlgru"),
            is_ternary    = (quant == "ternary"),
            use_local_dec = use_lbd,
            patch_size    = patch,
            vocab_size    = vocab,
            n_heads       = n_heads,
            d_model       = int(config.get("d_model", 512)),
            n_layers      = int(config.get("n_layers", 12)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# AblationEngine — adapter para usar AblationModel con eval_* módulos
# ─────────────────────────────────────────────────────────────────────────────

class AblationEngine:
    """
    Adapter que hace que AblationModel sea compatible con la interfaz
    CPU1InferenceEngine que usan eval_correctness, eval_performance, eval_quality.

    Los módulos de eval solo acceden a:
      engine.info            → AblationInfo (compatible con ModelInfo)
      engine.hidden_states   → estado recurrente
      engine._get_model()    → AblationModel
      engine._get_embedding(model) → fn(x) → x  (identidad, embedding está en el model)
      engine.reset_state()   → limpia estado
      engine.generate(...)   → generación byte-level
      engine._ablation_step(model, dummy) → (logits, new_states)  [performance]
      engine._is_ablation    → True (marca para eval_performance)
    """

    _is_ablation = True   # detectado por eval_performance para usar PyTorch puro

    def __init__(self, model, config: dict, info: AblationInfo):
        self._model = model
        self.config = config
        self.info   = info
        self.hidden_states: list | None = None

    # ── Interfaz básica ──────────────────────────────────────────────────────

    def _get_model(self):
        return self._model

    def reset_state(self):
        self.hidden_states = None

    def _get_embedding(self, model):
        """
        Devuelve una función identidad.
        Para AblationModel, el embedding forma parte del forward completo;
        no hay paso 'solo embedding' separable sin reescribir el modelo.
        eval_performance nunca llama a esto en modo ablation (_is_ablation=True),
        pero si alguien lo llama por error devuelve algo razonable.
        """
        def _identity(x):
            return x
        return _identity

    # ── Paso autoregressivo — PyTorch puro, SIN kernels nuestros ─────────────

    def _ablation_step(self, model, dummy: torch.Tensor):
        """
        Un paso de forward completo en PyTorch puro.

        Garantías:
          - Solo usa model.forward() de PyTorch estándar.
          - No llama a forward_with_prefetch ni a ningún kernel C/SIMD nuestro.
          - Para MLGRU: lleva self.hidden_states entre pasos (real O(1)).
          - Para Transformer: hidden_states siempre None (sin estado persistente).
        """
        result = model(
            dummy,
            hidden_states = self.hidden_states if self.info.is_recurrent else None,
            training_phase       = 1,
            target_elimination   = 0.0,
        )
        logits, new_states = result[0], result[1]
        return logits, (new_states if self.info.is_recurrent else None)

    # ── Generación ───────────────────────────────────────────────────────────

    def generate(self, prompt: bytes, max_bytes: int = 256,
                 temperature: float = 1.0, top_p: float = 0.9,
                 reset: bool = True) -> bytes:
        """
        Generación byte-level usando _generate_byte de train_ablation.
        BPE runs: devuelve b'' (generate_tokens no implementado).
        """
        if not self.info.is_byte_level:
            return b""
        from train_ablation import _generate_byte
        return _generate_byte(
            self._model, prompt, max_bytes, self.config,
            torch.device("cpu"),
            temperature=temperature, top_k=40, top_p=top_p,
            elimination_rate=0.0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Localización y carga de checkpoints
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoint(run_name: str) -> Path | None:
    """Busca checkpoint local (compact_2bit primero, fp32 como fallback)."""
    candidates = [
        MODELS_COMPACT / run_name / f"checkpoint_{run_name}_final.pt",
        MODELS_FP32    / run_name / "model.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def discover_runs() -> list[str]:
    """Lista todos los runs con checkpoint local disponible."""
    return [r for r in RUN_META if find_checkpoint(r) is not None]


def load_run(ckpt_path: Path, run_name: str) -> tuple:
    """
    Carga el checkpoint y construye el modelo. Soporta tres formatos:

      compact_2bit  → load_ablation_checkpoint desempaqueta el ternary
      bf16          → load_ablation_checkpoint castea a fp32
      fp32          → carga directa desde raw['state_dict']
                      (formato de cpu1-ablations-final y export_ablations.py)

    Devuelve (engine, raw_ckpt, config, missing_keys, unexpected_keys).
    """
    from train_ablation import load_ablation_checkpoint, build_ablation_model

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    fmt = raw.get("format", "bf16")
    if fmt == "fp32":
        # Descargado de Cukinator/cpu1-ablations-final o exportado localmente.
        # Claves: state_dict, config, metrics, step, run_name, format, source_repo
        config = raw["config"]
        state  = raw["state_dict"]
    else:
        # compact_2bit o bf16: load_ablation_checkpoint lo desempaqueta / castea.
        state, config = load_ablation_checkpoint(str(ckpt_path))

    config["use_gradient_checkpointing"] = False
    model = build_ablation_model(config)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    info   = AblationInfo.from_config(run_name, config)
    engine = AblationEngine(model, config, info)
    return engine, raw, config, missing, unexpected


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de pérdida (del checkpoint guardado)
# ─────────────────────────────────────────────────────────────────────────────

def extract_loss_metrics(raw_ckpt: dict, info: AblationInfo) -> dict:
    """
    Extrae val_loss del checkpoint y calcula métricas derivadas.

    Criterio homogéneo:
      perplexity = exp(val_loss) para TODOS los runs (byte-level o token-level).
      bpb        = val_loss / ln(2)  solo runs byte-level.
      delta_uniform = val_loss - ln(256)  solo runs byte-level.
    """
    stored = raw_ckpt.get("metrics", {}) or {}
    val_loss = None
    for key in ("val_loss", "loss", "final_val_loss"):
        if key in stored and isinstance(stored[key], (float, int)):
            val_loss = float(stored[key])
            break

    if val_loss is None:
        return {"val_loss_available": False}

    ppl = math.exp(min(val_loss, 88.0))
    result = {
        "val_loss":            round(val_loss, 6),
        "perplexity":          round(ppl, 3),
        "ppl_type":            "byte-level" if info.is_byte_level else "token-level",
        "val_loss_available":  True,
    }
    if info.is_byte_level:
        result["bpb"]           = round(val_loss / LN2, 5)
        result["delta_uniform"] = round(val_loss - UNIFORM_BYTE, 5)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Guardado de resultados
# ─────────────────────────────────────────────────────────────────────────────

def _f(v, dec=2, fb="—"):
    if v is None:
        return fb
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def save_run_json(run_name: str, data: dict) -> Path:
    RESULTADOS_DIR.mkdir(exist_ok=True)
    p = RESULTADOS_DIR / f"ablation_{run_name}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return p


def save_summary_markdown(all_results: list[dict]) -> Path:
    RESULTADOS_DIR.mkdir(exist_ok=True)
    path = RESULTADOS_DIR / "ablation_summary.md"

    lines = [
        "# Ablation Study — Evaluación completa",
        "",
        f"Generado: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Criterio: PPL = exp(val_loss) | BPB = val_loss / ln(2) (byte-only)",
        f"Referencia uniforme byte: ln(256) = {UNIFORM_BYTE:.3f} nats → BPB = 8.000",
        f"Performance: PyTorch puro, sin kernels customizados (ablation mode).",
        "",
        "## Correctness",
        "",
        "| Run | fwd_shapes | probs | determinism | state_prop | ternary | state_size |",
        "|-----|:----------:|:-----:|:-----------:|:----------:|:-------:|:----------:|",
    ]
    for r in all_results:
        nm = r["run_name"]
        C  = r.get("correctness", {})
        s  = lambda k: C.get(k, {}).get("status", "—")
        lines.append(f"| {nm} | {s('forward_shapes')} | {s('probs')} | "
                     f"{s('determinism')} | {s('state_propagation')} | "
                     f"{s('ternary_weights')} | {s('state_size')} |")

    lines += [
        "",
        "## Performance (PyTorch puro)",
        "",
        "| Run | P/s | Bytes/s | p50 ms | p95 ms | CV | ctx_scaling | RAM Δ MB |",
        "|-----|----:|-------:|-------:|-------:|---:|:-----------:|---------:|",
    ]
    for r in all_results:
        nm  = r["run_name"]
        P   = r.get("performance", {})
        tl  = P.get("throughput_latency_jitter", {}).get("result", {})
        cs  = P.get("context_scaling", {}).get("status", "—")
        ram = P.get("ram", {}).get("result", {})
        lines.append(
            f"| {nm} "
            f"| {_f(tl.get('throughput_sps'),1)} "
            f"| {_f(tl.get('throughput_bps'),0)} "
            f"| {_f(tl.get('p50_ms'),1)} "
            f"| {_f(tl.get('p95_ms'),1)} "
            f"| {_f(tl.get('jitter_cv'),4)} "
            f"| {cs} "
            f"| {_f(ram.get('delta_mb'),1)} |"
        )

    lines += [
        "",
        "## Quality",
        "",
        "| Run | Val Loss | PPL | BPB | Δ uniforme | gen% print | utf8% | gen status |",
        "|-----|--------:|----:|----:|----------:|-----------:|------:|:----------:|",
    ]
    for r in all_results:
        nm  = r["run_name"]
        lm  = r.get("loss_metrics", {})
        Q   = r.get("quality", {})
        gn  = Q.get("generation", {}).get("result", {})
        u8  = Q.get("utf8", {}).get("result", {})
        gs  = Q.get("generation", {}).get("status", "—")
        is_byte = r.get("is_byte_level", True)
        lines.append(
            f"| {nm} "
            f"| {_f(lm.get('val_loss'),4)} "
            f"| {_f(lm.get('perplexity'),2)} "
            f"| {_f(lm.get('bpb'),4) if is_byte else '—'} "
            f"| {_f(lm.get('delta_uniform'),4) if is_byte else '—'} "
            f"| {_f(gn.get('printable_pct'),3) if not gn.get('skipped') else '—'} "
            f"| {_f(u8.get('utf8_pct'),3) if not u8.get('skipped') else '—'} "
            f"| {gs} |"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Generación del README actualizado
# ─────────────────────────────────────────────────────────────────────────────

def build_readme(all_results: list[dict]) -> str:
    """Reconstruye el README con tablas recalculadas desde los datos reales."""
    by_run = {r["run_name"]: r for r in all_results}
    date_str = time.strftime("%Y-%m-%d")

    yaml_block = ""
    if HF_README.exists():
        orig = HF_README.read_text(encoding="utf-8")
        end = orig.find("---", 3)
        if end > 0:
            yaml_block = orig[:end + 3]

    lines = []
    if yaml_block:
        lines.append(yaml_block)

    lines += [
        "",
        "# CPU-1 Ablation Study — Ready-to-use Checkpoints (fp32 unpacked)",
        "",
        f"Repo:    `Cukinator/cpu1-ablations-final`",
        f"Source:  [`Cukinator/cpu1-ablation-checkpoints`]"
        f"(https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)",
        f"Code:    [github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits)",
        "",
        f"> **Tablas recalculadas {date_str}** — criterio homogéneo:",
        f"> `perplexity = exp(val_loss)` para todos los runs sin excepción.",
        f"> `BPB = val_loss / ln(2)` — solo byte-level.",
        f"> Referencia uniforme byte: **ln(256) = {UNIFORM_BYTE:.3f} nats → PPL=256 → BPB=8.000**",
        f"> BPE runs (run\\_01, run\\_13, \\*\\_r2): PPL es **token-level**.",
        "",
    ]

    def _lm(nm, k, dec=4):
        return _f(by_run.get(nm, {}).get("loss_metrics", {}).get(k), dec)

    def _perf(nm):
        r = by_run.get(nm, {})
        tl = r.get("performance", {}).get("throughput_latency_jitter", {}).get("result", {})
        return _f(tl.get("throughput_sps"), 1), _f(tl.get("throughput_bps"), 0)

    def _params(nm):
        return _f(by_run.get(nm, {}).get("n_params_M"), 1)

    def _ram(nm):
        return _f(by_run.get(nm, {}).get("ram_fp32_mb"), 1)

    # ── 50M Round 1 ──────────────────────────────────────────────────────────
    r1_50m_order = ["run_01","run_02a_byte_only_heads","run_02","run_03","run_04",
                    "run_05","run_05b_kernel_strict","run_06","run_07","run_08",
                    "run_09","run_10"]
    r1_50m = [nm for nm in r1_50m_order if nm in by_run]

    lines += [
        "## Ablation chain — 50M runs (round 1, 2 tok/param)",
        "",
        "| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Δ uniform | P/s | Bytes/s | Nivel |",
        "|-----|-------------|--------:|------------:|---------:|----:|----:|----------:|----:|--------:|-------|",
    ]
    for nm in r1_50m:
        meta = RUN_META.get(nm, {})
        r = by_run[nm]
        is_b = r.get("is_byte_level", True)
        nivel = "byte" if is_b else "token"
        pps, bps = _perf(nm)
        lines.append(
            f"| {nm} | {meta.get('label','?')} "
            f"| {_params(nm)} | {_ram(nm)} "
            f"| {_lm(nm,'val_loss',4)} | {_lm(nm,'perplexity',2)} "
            f"| {_lm(nm,'bpb',4) if is_b else '—'} "
            f"| {_lm(nm,'delta_uniform',4) if is_b else '—'} "
            f"| {pps} | {bps} | {nivel} |"
        )

    # ── 10M Round 1 ──────────────────────────────────────────────────────────
    r1_10m_order = ["run_13","run_14","run_15","run_16"]
    r1_10m = [nm for nm in r1_10m_order if nm in by_run]
    lines += [
        "",
        "## Small runs — 10M (round 1, 2 tok/param)",
        "",
        "| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Nivel |",
        "|-----|-------------|--------:|------------:|---------:|----:|----:|-------|",
    ]
    for nm in r1_10m:
        meta = RUN_META.get(nm, {})
        r = by_run[nm]
        is_b = r.get("is_byte_level", True)
        nivel = "byte" if is_b else "token"
        lines.append(
            f"| {nm} | {meta.get('label','?')} "
            f"| {_params(nm)} | {_ram(nm)} "
            f"| {_lm(nm,'val_loss',4)} | {_lm(nm,'perplexity',2)} "
            f"| {_lm(nm,'bpb',4) if is_b else '—'} | {nivel} |"
        )

    # ── Round 2 ──────────────────────────────────────────────────────────────
    r2_to_r1 = {
        "run_04_r2":"run_04","run_05_r2":"run_05","run_07_r2":"run_07",
        "run_08_r2":"run_08","run_09_r2":"run_09","run_10_r2":"run_10",
        "run_13_r2":"run_13","run_14_r2":"run_14","run_15_r2":"run_15",
        "run_16_r2":"run_16",
    }
    r2_runs = [nm for nm in by_run if nm.endswith("_r2")]
    if r2_runs:
        lines += [
            "",
            "## Round 2 — re-runs a 15 tok/param",
            "",
            "| Run | Arquitectura | Params M | Loss r1 | PPL r1 | Loss r2 | PPL r2 | Δ loss | BPB r2 | Nivel |",
            "|-----|-------------|--------:|--------:|-------:|--------:|-------:|-------:|-------:|-------|",
        ]
        for nm in sorted(r2_runs):
            meta = RUN_META.get(nm, {})
            r = by_run[nm]
            is_b = r.get("is_byte_level", True)
            nivel = "byte" if is_b else "token"
            r1_nm = r2_to_r1.get(nm)
            r1_lm = by_run.get(r1_nm, {}).get("loss_metrics", {})
            r2_lm = r.get("loss_metrics", {})
            vl1 = r1_lm.get("val_loss")
            vl2 = r2_lm.get("val_loss")
            if vl1 is not None and vl2 is not None:
                delta = vl2 - vl1
                delta_s = f"+{delta:.5f}" if delta >= 0 else f"{delta:.5f}"
            else:
                delta_s = "—"
            lines.append(
                f"| {nm} | {meta.get('label','?')} "
                f"| {_params(nm)} "
                f"| {_f(vl1,4)} | {_f(r1_lm.get('perplexity'),2)} "
                f"| {_lm(nm,'val_loss',4)} | {_lm(nm,'perplexity',2)} "
                f"| {delta_s} "
                f"| {_lm(nm,'bpb',4) if is_b else '—'} | {nivel} |"
            )

    # ── Round 3 ──────────────────────────────────────────────────────────────
    v3_runs = [nm for nm in by_run if nm.endswith("_v3")]
    if v3_runs:
        lines += [
            "",
            "## Round 3 — v3 runs (bf16 AMP + lr×2 + 50 tok/p)",
            "",
            "| Run | Arquitectura | Params M | Val Loss | PPL | BPB | Step | P/s | Bytes/s | Nivel |",
            "|-----|-------------|--------:|--------:|----:|----:|-----:|----:|--------:|-------|",
        ]
        for nm in sorted(v3_runs):
            meta = RUN_META.get(nm, {})
            r = by_run[nm]
            is_b = r.get("is_byte_level", True)
            nivel = "byte" if is_b else "token"
            pps, bps = _perf(nm)
            lines.append(
                f"| {nm} | {meta.get('label','?')} "
                f"| {_params(nm)} "
                f"| {_lm(nm,'val_loss',4)} | {_lm(nm,'perplexity',2)} "
                f"| {_lm(nm,'bpb',4) if is_b else '—'} "
                f"| {_f(r.get('step'))} | {pps} | {bps} | {nivel} |"
            )
            
        lines += [
            "",
            "**Hallazgo clave de V3:** El cuello de botella de los modelos ternarios en *cold-start* fue resuelto en la **v3** (usando bf16 AMP y `lr_scale=2.0` en BitLinear). Esto permitió romper el plateau de entropía uniforme (~5.55 nats) en el que caían las versiones anteriores, logrando convergencia estable y permitiendo que generen texto imprimible (`PASS` en los tests de generación).",
        ]

    lines += [
        "",
        "## Licencia",
        "",
        "Apache-2.0. Same as the source code at "
        "[github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits).",
        "",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HuggingFace: descarga y upload
# ─────────────────────────────────────────────────────────────────────────────

def _get_hf_token() -> str | None:
    """Lee HF_TOKEN del entorno o del archivo .env del repo."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("HF_TOKEN"):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    return token or None


def download_from_hf(runs_to_try: list[str]) -> list[str]:
    """
    Descarga checkpoints hacia disco:
      1. Intenta fp32 desde Cukinator/cpu1-ablations-final → models/ablations_final/
      2. Si da 404, intenta compact_2bit desde Cukinator/cpu1-ablation-checkpoints
         → models/ablation_models/ (load_ablation_checkpoint lo desempaqueta)
    Saltea los que ya existen.
    Devuelve lista de runs con checkpoint disponible.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [ERROR] huggingface_hub no instalado — pip install huggingface-hub")
        return []

    HF_REPO_COMPACT = "Cukinator/cpu1-ablation-checkpoints"

    MODELS_FP32.mkdir(parents=True, exist_ok=True)
    MODELS_COMPACT.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"  Descargando modelos (fp32 primero, compact como fallback)")
    print(f"{'=' * 70}")

    available = []
    for run_name in runs_to_try:
        # ── ¿Ya tenemos algo? ────────────────────────────────────────────────
        fp32_dst    = MODELS_FP32    / run_name / "model.pt"
        compact_dst = MODELS_COMPACT / run_name / f"checkpoint_{run_name}_final.pt"

        if fp32_dst.exists():
            size_mb = fp32_dst.stat().st_size / 1e6
            print(f"  [cache fp32]    {run_name:<30} ({size_mb:.0f} MB)")
            available.append(run_name)
            continue
        if compact_dst.exists():
            size_mb = compact_dst.stat().st_size / 1e6
            print(f"  [cache compact] {run_name:<30} ({size_mb:.0f} MB)")
            available.append(run_name)
            continue

        # ── Intento 1: fp32 desde cpu1-ablations-final ───────────────────────
        fp32_dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Descargando (fp32) {run_name}...", end="", flush=True)
        try:
            hf_hub_download(
                repo_id    = HF_REPO_MODELS,
                filename   = f"{run_name}/model.pt",
                repo_type  = "model",
                local_dir  = str(MODELS_FP32),
                local_dir_use_symlinks = False,
            )
            size_mb = fp32_dst.stat().st_size / 1e6
            print(f" {size_mb:.0f} MB  OK")
            available.append(run_name)
            continue
        except Exception:
            print(f" no disponible — probando compact...")

        # ── Intento 2: compact_2bit desde cpu1-ablation-checkpoints ──────────
        compact_dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Descargando (compact) {run_name}...", end="", flush=True)
        try:
            hf_hub_download(
                repo_id    = HF_REPO_COMPACT,
                filename   = f"{run_name}/checkpoint_{run_name}_final.pt",
                repo_type  = "model",
                local_dir  = str(MODELS_COMPACT),
                local_dir_use_symlinks = False,
            )
            size_mb = compact_dst.stat().st_size / 1e6
            print(f" {size_mb:.0f} MB  OK")
            available.append(run_name)
        except Exception as e:
            print(f" FALLO: {e}")

    print(f"  Total disponibles: {len(available)}/{len(runs_to_try)}")
    return available


    return available


def upload_readme_to_hf(readme_path: Path) -> None:
    """Sube el README actualizado al repo HuggingFace."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [WARN] huggingface_hub no instalado — no se sube el README")
        return

    token = _get_hf_token()
    if not token:
        print("  [WARN] No hay HF_TOKEN — no se puede subir el README")
        return

    api = HfApi(token=token)
    commit_msg = f"Update README: recalculated metrics {time.strftime('%Y-%m-%d')}"
    print(f"\n  Subiendo README a {HF_REPO_MODELS} ...", flush=True)
    try:
        api.upload_file(
            path_or_fileobj = str(readme_path),
            path_in_repo    = "README.md",
            repo_id         = HF_REPO_MODELS,
            repo_type       = "model",
            commit_message  = commit_msg,
        )
        print(f"  README subido OK")
    except Exception as e:
        print(f"  [ERROR upload README] {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Suite completa de evaluación para todos los modelos ablation"
    )
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Subset de runs (ej: run_02 run_07)")
    parser.add_argument("--no-performance", action="store_true",
                        help="Omitir eval_performance (más rápido)")
    parser.add_argument("--update-readme", action="store_true",
                        help="Reconstruir hf_readmes/cpu1-ablations-final.md con métricas reales")
    parser.add_argument("--download", action="store_true",
                        help="Descargar checkpoints faltantes desde HF (Cukinator/cpu1-ablations-final)")
    parser.add_argument("--upload-readme", action="store_true",
                        help="Subir el README actualizado al repo HuggingFace después del eval")
    args = parser.parse_args()

    # Importar módulos de eval (usamos TODOS)
    from eval_correctness import run_correctness
    from eval_performance import run_performance
    from eval_quality     import run_quality
    # eval_dataset y eval_external se saltan para ablation models:
    #   eval_dataset → requiere dataset local con teacher signal
    #   eval_external → requiere GGUF + llama.cpp (no aplica a ablation)

    # ── Descarga desde HF si se pidió ─────────────────────────────────────
    if args.download:
        target_runs = args.runs or list(RUN_META.keys())
        download_from_hf(target_runs)

    runs = args.runs or discover_runs()
    if not runs:
        print("  No se encontraron checkpoints localmente (usa --download para descargar).")
        return

    print(f"\n{'═' * 70}")
    print(f"  Ablation Eval Suite")
    print(f"  Módulos: correctness + performance (PyTorch puro) + quality")
    print(f"  Runs: {len(runs)}")
    print(f"  Criterio: PPL = exp(val_loss) | BPB = val_loss / ln(2) (byte-only)")
    print(f"{'═' * 70}")

    all_results = []
    t_total = time.perf_counter()

    for run_name in runs:
        ckpt_path = find_checkpoint(run_name)
        if ckpt_path is None:
            print(f"\n  [SKIP] {run_name} — checkpoint no encontrado")
            continue

        print(f"\n{'─' * 70}")
        print(f"  RUN: {run_name}  →  {ckpt_path}")
        print(f"{'─' * 70}")
        t0 = time.perf_counter()

        # ── Carga ──────────────────────────────────────────────────────────
        try:
            engine, raw, config, missing, unexpected = load_run(ckpt_path, run_name)
        except Exception as e:
            print(f"  [ERROR carga] {e}")
            all_results.append({"run_name": run_name, "error": f"load: {e}"})
            continue

        model = engine._get_model()
        info  = engine.info

        record: dict = {
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
        if missing:
            record["missing_keys"] = len(missing)
        if unexpected:
            record["unexpected_keys"] = len(unexpected)

        # ── Métricas de pérdida (del checkpoint) ───────────────────────────
        record["loss_metrics"] = extract_loss_metrics(raw, info)

        # ── Correctness ────────────────────────────────────────────────────
        record["correctness"] = run_correctness(engine)

        # ── Performance (eval_performance, modo ablation = PyTorch puro) ───
        if not args.no_performance:
            record["performance"] = run_performance(engine)
        else:
            record["performance"] = {"skipped": True}

        # ── Quality (eval_quality: generation + utf8 + perplexity) ─────────
        # dataset=None → skip check_perplexity (usamos el del checkpoint)
        record["quality"] = run_quality(engine, dataset=None)

        # ── eval_dataset: skip (sin dataset local) ─────────────────────────
        record["dataset_eval"] = {"skipped": True, "reason": "sin dataset local"}

        # ── eval_external: skip (sin GGUF de ablation) ────────────────────
        record["external_eval"] = {"skipped": True, "reason": "sin GGUF (no aplica a ablation)"}

        record["eval_time_s"] = round(time.perf_counter() - t0, 1)

        # ── Resumen consola ────────────────────────────────────────────────
        lm  = record["loss_metrics"]
        tl  = record.get("performance", {}).get("throughput_latency_jitter", {}).get("result", {})
        c_ok = sum(1 for v in record["correctness"].values()
                   if isinstance(v, dict) and v.get("status") == "PASS")
        c_n  = len(record["correctness"])
        q_ok = sum(1 for v in record["quality"].values()
                   if isinstance(v, dict) and v.get("status") == "PASS")

        print(f"\n  ── Resumen {run_name} ({record['eval_time_s']}s) ──")
        print(f"  Params:     {record['n_params_M']} M  |  RAM fp32: {record['ram_fp32_mb']} MB")
        if lm.get("val_loss_available"):
            print(f"  Val Loss:   {lm['val_loss']} nats  [{lm['ppl_type']}]")
            print(f"  PPL:        {lm['perplexity']}")
            if "bpb" in lm:
                print(f"  BPB:        {lm['bpb']} bits/byte  (Δ uniform = {lm.get('delta_uniform','?')})")
        if tl:
            print(f"  Throughput: {tl.get('throughput_sps','?')} P/s  "
                  f"({tl.get('throughput_bps','?')} bytes/s)  "
                  f"p50={tl.get('p50_ms','?')}ms  [pytorch_pure]")
        print(f"  Checks:     correctness {c_ok}/{c_n} PASS  |  quality {q_ok}/? PASS")

        json_p = save_run_json(run_name, record)
        print(f"  JSON:       {json_p}")
        all_results.append(record)

    # ── Resumen global ─────────────────────────────────────────────────────
    if all_results:
        md_p = save_summary_markdown(all_results)
        print(f"\n\n  Resumen Markdown: {md_p}")

    if args.update_readme and all_results:
        HF_README.parent.mkdir(parents=True, exist_ok=True)
        readme_txt = build_readme(all_results)
        HF_README.write_text(readme_txt, encoding="utf-8")
        print(f"  README local: {HF_README}")

    if args.upload_readme and all_results:
        if not args.update_readme:
            # Generar el README aunque no se haya pedido guardar localmente
            HF_README.parent.mkdir(parents=True, exist_ok=True)
            readme_txt = build_readme(all_results)
            HF_README.write_text(readme_txt, encoding="utf-8")
        upload_readme_to_hf(HF_README)

    elapsed = time.perf_counter() - t_total
    total_pass = sum(
        sum(1 for v in r.get("correctness", {}).values()
            if isinstance(v, dict) and v.get("status") == "PASS")
        for r in all_results
    )
    print(f"\n{'═' * 70}")
    print(f"  Listo — {len(all_results)} runs  |  {elapsed:.0f}s total")
    print(f"  Correctness PASS total: {total_pass}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
