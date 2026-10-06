"""Strict loading of custom CPU-1 checkpoints for the common audit."""
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class AblationInfo:
    """
    Architecture metadata used by the checkpoint audit.
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
        # en _get_logits_list de evaluation.correctness a P tensores [B,S,256])
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


class AblationEngine:
    """Loaded model and recurrent carry for correctness checks and timing."""

    def __init__(self, model, config: dict, info: AblationInfo):
        self._model = model
        self.config = config
        self.info = info
        self.hidden_states = None

    def _get_model(self):
        return self._model

    def reset_state(self):
        self.hidden_states = None

    def _ablation_step(self, model, dummy: torch.Tensor):
        result = model(
            dummy,
            hidden_states=self.hidden_states if self.info.is_recurrent else None,
            training_phase=1,
            target_elimination=0.0,
        )
        return result[0], result[1] if self.info.is_recurrent else None


def load_run(ckpt_path: Path, run_name: str) -> tuple:
    """
    Carga el checkpoint y construye el modelo. Soporta tres formatos:

      compact_2bit  → load_ablation_checkpoint desempaqueta el ternary
      bf16          → load_ablation_checkpoint castea a fp32
      fp32          → carga directa desde raw['state_dict']
                      (formato de cpu1-ablations-final y export_ablations.py)

    Devuelve (engine, raw_ckpt, config, missing_keys, unexpected_keys).
    """
    from training.ablation import load_ablation_checkpoint, build_ablation_model

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
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint incompatible: missing={missing}, unexpected={unexpected}")
    model.eval()

    info   = AblationInfo.from_config(run_name, config)
    engine = AblationEngine(model, config, info)
    return engine, raw, config, missing, unexpected
