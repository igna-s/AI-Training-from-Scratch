"""Inspect all recorded/preset architectures on meta tensors, without downloads.

This checks constructors and module inventories, not training quality, GPU
compatibility, paper equivalence, or the numerical correctness of a checkpoint.
"""
import argparse
from collections import Counter
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import torch

from cpu1.components import (
    AblationMLGRU, BitEmbedding, BitLinear, BolmoPatchEmbedding,
    BoundaryPredictor, DeleteGate, EmbeddingAligner, FPResidual,
    LocalByteDecoder, MatMulFreeTokenMixer, PFNet, SlidingWindowAttention,
)
from training import ablation
from training.configuration import load_config
from training.final import CPU1ModelV2, CPU1ModelV2WithGates


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(model, config):
    modules = list(model.modules())
    decoder = next((m for m in modules if isinstance(m, LocalByteDecoder)), None)
    embedding = getattr(model, "embedding", getattr(model, "patch_embedding", None))
    mixers = [m for m in modules if isinstance(
        m, (AblationMLGRU, MatMulFreeTokenMixer, SlidingWindowAttention))]
    shapes = {key: list(value.shape) for key, value in model.state_dict().items()}
    counts = Counter(type(m).__name__ for m in modules)
    selected = (BitLinear, BitEmbedding, FPResidual, PFNet, DeleteGate,
                BoundaryPredictor, EmbeddingAligner)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "state_shape_sha256": hashlib.sha256(
            json.dumps(shapes, sort_keys=True).encode()).hexdigest(),
        "tokenization_label": config.get("tokenization", "byte"),
        "quantization_label": config.get("quantization", "ternary"),
        "embedding": type(embedding).__name__,
        "mixers": dict(Counter(type(m).__name__ for m in mixers)),
        "local_decoder_layers": decoder.n_layers if decoder else 0,
        "local_decoder_dim": decoder.d_local if decoder else None,
        "module_counts": {cls.__name__: counts[cls.__name__] for cls in selected},
        "learned_decay_layers": sum(getattr(m, "log_decay", None) is not None for m in mixers),
        "state_expansion": sorted({m.state_expansion for m in mixers if isinstance(m, AblationMLGRU)}),
        "parallel_scan_layers": sum(getattr(m, "use_parallel_scan", False) for m in mixers),
        "reset_periods": sorted({m.state_reset_period for m in mixers if isinstance(m, AblationMLGRU)}),
        "wo_layers": sum(getattr(m, "w_o", None) is not None for m in mixers),
        "gather_configured": bool(config.get("use_real_gather", False)),
        "training_signal": config.get("training_signal"),
        "recorded_phase": config.get("_current_phase"),
        "dimensions": {key: config.get(key) for key in ("d_model", "n_layers", "d_ff", "patch_size")},
    }
    if isinstance(embedding, BolmoPatchEmbedding):
        result["boundary_attention"] = {
            "fixed_patch_size": embedding.patch_size,
            "cross_patch_window": embedding.cross_patch_window,
            "causal_boundary": embedding.causal_boundary,
            "causal_cross_patch": embedding.causal_cross_patch,
        }
    return result


def construct(factory, config):
    with torch.device("meta"), contextlib.redirect_stdout(io.StringIO()):
        model = factory(config)
    return inventory(model, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/architecture-review.json"))
    args = parser.parse_args()
    # ROCm and CUDA availability are deliberately not tested by this CPU/meta
    # review. Avoid GPU initialization in the AMD module's import-time selector.
    with patch.object(torch.cuda, "is_available", return_value=False):
        amd = importlib.import_module("training.ablation_amd")
    manifest_path = ROOT / "experiments/published/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    published = []
    for record in manifest["experiments"]:
        path = ROOT / "experiments/published" / record["config"]
        config = load_config(path)
        first = construct(ablation.build_ablation_model, config)
        second = construct(amd.build_ablation_model, config)
        config_matches = sha256(path) == record["config_sha256"]
        counts_match = first["parameters"] == record["parameters"]
        backends_match = first == second
        if not (config_matches and counts_match and backends_match):
            raise RuntimeError(f"Recorded architecture mismatch: {record['run']}")
        published.append({"run": record["run"], "config_sha256": sha256(path),
                          "checkpoint_sha256": record["checkpoint_sha256"],
                          "saved_step": record["saved_step"],
                          "record_matches": True, "cuda_amd_inventory_matches": True,
                          **first})
    presets = []
    for backend in (ablation, amd):
        for run in sorted(backend.ALL_RUN_CONFIGS):
            config = backend.get_run_config(run)
            presets.append({"backend": backend.__name__, "run": run,
                            "configured_tokens_per_parameter": config.get("chinchilla_tokens_per_param"),
                            **construct(backend.build_ablation_model, config)})
    by_backend = {}
    for record in presets:
        comparable = {key: value for key, value in record.items()
                      if key not in ("backend", "configured_tokens_per_parameter")}
        by_backend.setdefault(record["backend"], {})[record["run"]] = comparable
    preset_inventories_match = by_backend[ablation.__name__] == by_backend[amd.__name__]
    if not preset_inventories_match:
        raise RuntimeError("Current CUDA/AMD preset architecture inventories differ")
    large_config = dict(d_model=1024, n_layers=14, d_ff=2730, patch_size=4,
                        fp_rank=8, pfnet_hidden=64)
    development = []
    for factory in (CPU1ModelV2, CPU1ModelV2WithGates):
        for legacy in (False, True):
            config = {**large_config, "legacy_heads": legacy}
            development.append({"class": factory.__name__, "legacy_heads": legacy,
                                **construct(factory, config)})
    config = {**large_config, "use_boundary_predictor": True,
              "use_delete_gate": True, "use_embedding_alignment": True}
    development.append({"class": "CPU1ModelV2WithGates", "all_optional_gates": True,
                        **construct(CPU1ModelV2WithGates, config)})
    summary = {
        "published_records": len(published),
        "published_backend_constructions": 2 * len(published),
        "preset_constructions": len(presets),
        "development_constructions": len(development),
        "published_parameters_range": [min(r["parameters"] for r in published),
                                       max(r["parameters"] for r in published)],
        "published_embedding_counts": dict(Counter(r["embedding"] for r in published)),
        "published_mixer_counts": dict(Counter(next(iter(r["mixers"])) for r in published)),
        "published_local_decoder_layers": dict(Counter(str(r["local_decoder_layers"]) for r in published)),
        "published_with_module": {cls.__name__: sum(r["module_counts"][cls.__name__] > 0 for r in published)
                                  for cls in (BitLinear, FPResidual, PFNet, DeleteGate, BoundaryPredictor, EmbeddingAligner)},
        "published_with_learned_decay": sum(r["learned_decay_layers"] > 0 for r in published),
        "all_published_records_match": True,
        "all_published_cuda_amd_inventories_match": True,
        "all_current_cuda_amd_preset_inventories_match": preset_inventories_match,
    }
    source_paths = ["cpu1/components.py", "training/ablation.py", "training/ablation_amd.py",
                    "training/final.py", "training/configuration.py", "inference/kernels.cpp",
                    "inference/benchmark.c"]
    report = {
        "scope": "Offline meta constructors and module inventory; no checkpoint downloads, numerical inference, GPU runs, or retraining",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "torch_version": torch.__version__,
        "source_sha256": {path: sha256(ROOT/path) for path in source_paths},
        "manifest_sha256": sha256(manifest_path),
        "reviewer_sha256": sha256(Path(__file__)),
        "summary": summary, "published": published, "presets": presets,
        "development": development,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
