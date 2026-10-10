# Setup, loading, and training

Run the examples from the repository root. For model results and project scope, see the [main README](../README.md).

The loading and generation examples use the pinned `run_02` byte Transformer export and were checked on CPU. GPU training recipes require a compatible accelerator environment; the model evaluation does not validate those jobs.

## Install and inspect

Clone the source repository before running these examples. Model weights and datasets are hosted on Hugging Face. Examples below use PowerShell and keep caches on a writable D: drive; adjust paths for your machine. The full evaluator requires a `--storage` path on D:.

The reviewed environment used **Python 3.12.10 and PyTorch 2.6.0**, running on CPU. Dependency versions are [recorded](../experiments/evaluation/model-audit-2026-10-06/environment.json); the project has no complete dependency lockfile.

```powershell
git clone https://github.com/igna-s/AI-Training-from-Scratch.git D:/ai-training
Set-Location D:/ai-training
New-Item -ItemType Directory -Force D:/ai-training-cache/hf, D:/ai-training-cache/datasets | Out-Null
$env:HF_HOME = "D:/ai-training-cache/hf"
$env:HF_DATASETS_CACHE = "D:/ai-training-cache/datasets"
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/audit.txt
```

After installation, these inspections do **not** download model weights, datasets, or the teacher:

```powershell
# Constructs all recorded/preset architectures with meta tensors.
python -m evaluation.architecture_review

# Demonstrates the existing parallel-scan mismatch.
python -m evaluation.diagnostics

# Copies the published evaluation records for local analysis.
python -m evaluation.audit --storage D:/cpu1-readme-audit --restore-evidence
```

For GPU training, install the PyTorch build appropriate to your accelerator and operating system using the [official selector](https://pytorch.org/get-started/locally/), then install `requirements/training.txt`. `training.ablation_amd` provides the ROCm-oriented entry point. The [architecture inventory](architecture-review.md) checks constructors on CPU/meta tensors; it does not validate a fresh CUDA or ROCm training job.

## Load and score one export

This example downloads **one approximately 155 MB file**, `run_02/model.pt`, from the pinned collection. It loads the byte Transformer baseline and computes teacher-forced next-patch loss. No teacher model is needed. Set cache variables **before** importing Hugging Face libraries.

```python
import os
os.environ["HF_HOME"] = "D:/ai-training-cache/hf"

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from training.ablation import build_ablation_model

torch.set_num_threads(1)
path = hf_hub_download(
    "Cukinator/cpu1-ablations-final", "run_02/model.pt",
    revision="65951289352c1a036b12ba6d888af037e7c90b13",
    local_dir="D:/ai-training-cache/active",
)
# Load the pinned project checkpoint and enforce its exact architecture.
payload = torch.load(path, map_location="cpu", weights_only=False)
config = dict(payload["config"])
config["use_gradient_checkpointing"] = False
model = build_ablation_model(config).eval()
model.load_state_dict(payload["state_dict"], strict=True)
del payload

p = config["patch_size"]
raw = list(b"The history of artificial intelligence is long.")
patches = torch.tensor(raw[:len(raw) // p * p]).reshape(1, -1, p)
x, targets = patches[:, :-1], patches[:, 1:]
with torch.inference_mode():
    heads, _, _ = model(x, byte_targets=targets,
                       training_phase=1, target_elimination=0.0)
    logits = torch.stack(heads, dim=2)
    nll = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
assert torch.isfinite(nll)
print("Mean NLL per byte:", nll.item())
```

For a byte-generation smoke check with this loaded CPU model:

```python
from evaluation.audit import generation

torch.manual_seed(42)
sample = generation(model, config, "The history of artificial intelligence", length=128)
print(sample.decode("utf-8", errors="replace"))
```

Generation can produce invalid UTF-8 and weak text; replacement decoding is only a display choice. The helper samples local bytes sequentially with temperature 1.0, top-k 40, and top-p 0.9. Transformer context is recomputed; recurrent variants carry state and remain affected by the documented scan defect.

## Training and teacher data

`training.ablation` and `training.ablation_amd` support named presets, explicit JSON configurations, synthetic CPU runs, checkpoint resume, and grouped GPU execution. [Recorded configurations](../experiments/README.md) describe the published models; current presets can differ from them.

Preview a recorded AMD recipe without starting training:

```powershell
python -m runners.lightning ablation --backend amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json
```

The runner prints a plan until `--execute` is supplied. It runs inside the selected platform; it does not provision cloud resources. Kaggle and Lightning dataset recipes preserve configured document targets, which are not verified processed totals.

For a new experiment, copy the chosen configuration, set `checkpoint_repo` to your own destination or `null`, and use a fresh run name. Real training requires suitable hardware. When `checkpoint_repo` is configured, the trainer can automatically download an intermediate checkpoint to resume and upload newly saved checkpoints. Use `--skip-install` after preparing dependencies. Synthetic `--dry_run` uses CPU data with the configured model and sequence length; a small step count does not reduce their memory cost.

The two-phase deletion path requires an appropriate phase-1 checkpoint and explicit phase handling. The final filename alone does not prove phase-2 training or an achieved deletion rate. The shared evaluation disables deletion and does not measure compression benefits.

[data_pipeline/prepare.py](../data_pipeline/prepare.py) extracts pretrained `Qwen/Qwen2.5-3B` signals over FineWeb. Byte supervision maps token candidates to their first byte at selected boundaries; BPE supervision retains top-k teacher entries. Optional hidden-state alignment uses projected features. This is teacher-assisted student training, rather than training Qwen from scratch.

The [dataset review](../docs/model_cards/cpu1-ablation-dataset.md) inspected **three fixed shards**, not all 917. It found a reader alignment defect, mask-format problems, and UTF-8 marginalization limitations. Byte/BPE shard ranges both omit IDs 0060 and 0061, and unequal subsets must not be paired by shard index. The card reports service row totals; an independent count of every row, dataset-wide teacher coverage, and reconstruction from the original documents are not established.

The precomputed loader's nominal **5 GiB selection / 10 GiB cache cleanup are not hard quotas**: its fallback can request an entire subset, and later cleanup does not cover every cache or temporary file. Without split metadata, the fallback validation slice overlaps the training data. Review these behaviors before starting a large run. The small WikiText audit uses a separate corpus.
