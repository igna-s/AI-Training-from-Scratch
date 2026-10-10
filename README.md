# AI Training from Scratch

Personal investigations I ran with AMD cloud credits: training small language models from scratch, exploring low-bit and byte-level architectures, and running ablation studies.

This repository brings together custom PyTorch models, training scripts, teacher-data preparation, CPU evaluation, and recorded experiments. **CPU-1** is the main documented experiment family. The student code constructs randomly initialized models and also supports resume or transfer from other experiment checkpoints. Some routes train with cross-entropy alone, while others use signals from the pretrained **Qwen2.5-3B** teacher.

**Status: experimental research, with known correctness defects.** Published checkpoints can be inspected and loaded, but the available evidence does not establish production readiness, reliable streaming inference, or a controlled improvement over the cited architectures.

[Model exports](https://huggingface.co/Cukinator/cpu1-ablations-final) · [Training checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints) · [Teacher dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) · [Architecture review](docs/architecture-review.md) · [Full evaluation](docs/model_cards/cpu1-ablations-final.md)

## What is implemented

The architecture review checks **all 34 published configuration records**, the **30 current presets in each ablation entry point**, and the separate larger-model constructors. Published models range from **10,691,371 to 54,735,360 parameters**. Names such as “10M” or “50M” in older code are approximate family labels.

| Component | Implementation and recorded scope |
|---|---|
| Transformer baselines | Causal sliding-window self-attention with RMSNorm and SwiGLU; 8 published exports, using bytes or a truncated BPE vocabulary |
| MLGRU variants | Custom gated recurrent mixer with floating-point or simulated ternary projections; 26 exports. The current parallel scan fails equivalence with serial and chunked execution |
| Byte representation | 29 exports use fixed **four-byte patches**. 27 have a **two-layer local autoregressive byte decoder**; 2 have four independent byte heads |
| Ternary weights | 26 exports contain `BitLinear`/`BitEmbedding` modules. PyTorch simulates quantization in floating point; the published exports store FP32 tensors |
| Boundary-aware embedding | 14 exports use `BolmoPatchEmbedding`: a boundary scorer and cross-attention over fixed patches. The `byte_dynamic` label does not establish variable-length patching in these models |
| Residual branches | 21 exports contain a floating-point low-rank `FPResidual`; 2 contain the separate **ternary** `PFNet` branch |
| Deletion and alignment | 15 exports contain `DeleteGate`; 3 contain teacher-hidden alignment. The common model audit runs phase 1 with deletion disabled |
| Decay/state options | Learned per-channel decay is present in one export; all 34 recorded configurations use state expansion factor 1. Expanded state and periodic reset are additional source options |

For byte models, global input patch `i` predicts patch `i+1`. The local decoder receives the global representation and the **preceding target bytes** within that next patch. Correct likelihood scoring requires shifted `byte_targets`; omitting them supplies zero placeholders.

The separate `CPU1ModelV2` path uses `MatMulFreeTokenMixer`, rather than the ablation mixer, and defaults to width 1,024, 14 layers, FFN width 2,730, and a one-layer local decoder: **164,869,729 parameters**. This is a development implementation; no corresponding large trained checkpoint is established by the 34-export audit. Its real-data loader and optional gates have additional limits documented in the [architecture review](docs/architecture-review.md#larger-model-path).

The code adapts ideas from several papers. “BitNet,” “MEGABYTE,” “Bolmo,” “MrT5,” “HGRN2,” and “T-MAC” references identify inspiration; they do not certify complete reproductions or comparable results. The review traces each name to the actual modules and flags unsupported claims.

## Measured results

The frozen **2026-10-06 CPU audit** evaluates every published FP32 export on **32 short WikiText test passages**. All 34 checkpoints were subsequently rechecked locally during the README review: hashes, strict loading, parameter counts, losses, and causal/chunk statuses matched the recorded evidence.

| Check | Result |
|---|---|
| Strict state loading | 34/34 loaded with no missing or unexpected keys |
| Finite loss | 30/34; 4 produced nonfinite loss |
| Prefix causality | 16 PASS, 14 FAIL, 4 ERROR |
| Recurrent chunk equivalence | 26/26 recurrent exports FAIL; 8 Transformer exports N/A |

The numerical failures are `run_01_v3`, `run_02_v3`, `run_02a_byte_only_heads_v3`, and `run_08_v3`. Checks overlap; successful loading does not imply numerical correctness.

Representative **byte** results:

| Export | Saved step | Mean NLL/byte | Bits/byte | Prefix causality | Chunk equivalence |
|---|---:|---:|---:|---|---|
| `run_02` | 647 | 1.9702 | 2.8424 | PASS | N/A |
| `run_03_v3` | 6,187 | 1.7640 | 2.5449 | PASS | FAIL |
| `run_04_v3` | 6,074 | 1.5515 | 2.2384 | PASS | FAIL |
| `run_14_v3` | 9,775 | 0.8811 | 1.2712 | FAIL | FAIL |

These are small-corpus diagnostics. A model that fails causality can benefit from future input, so its low loss does not establish autoregressive prediction quality. A causality pass also does not resolve the recurrent scan defect. Different saved steps, parameter counts, signals, and budgets prevent treating these rows as a controlled component ranking.

BPE scores use a different unit and a lossy vocabulary mapping: Qwen IDs outside the student vocabulary become ID 0. In the audit sample this affected **14.26%** of IDs for the 16K vocabulary and **31.77%** for the 4K vocabulary. BPE perplexity cannot be compared directly with byte perplexity.

The [full model card](docs/model_cards/cpu1-ablations-final.md) includes all 34 rows, generated samples, and timing conditions. [Frozen evidence](experiments/evaluation/README.md) preserves the exact corpus, environment, per-run records, and recurrence diagnostic. [README validation](experiments/evaluation/readme-validation-2026-10-10.json) records the later recheck and its scope.

## Published artifacts

Sizes below are checkpoint/Parquet file totals in **decimal GB**, measured at the reviewed revisions.

| Repository | Contents | Reviewed size |
|---|---|---:|
| [cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final/tree/65951289352c1a036b12ba6d888af037e7c90b13) | 34 custom `<run>/model.pt` FP32 exports | 4.178 GB |
| [cpu1-ablation-checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/tree/fb402d025e843ed00fb990966bf2430c32eb958c) | 1,346 `.pt` files across 34 run folders, including intermediate checkpoints and 21 finals | 215.178 GB |
| [cpu1-ablation-dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset/tree/bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf) | 917 Parquet files: 480 byte and 437 BPE shards | 68.019 GB |

The 34-export model evaluation does not cover every historical intermediate checkpoint. The [manifest](experiments/published/manifest.json) binds each export to a configuration, SHA-256, saved step, parameter count, and pinned model revision. An `_r2` or `_v3` suffix can identify different settings; consult the record instead of inferring architecture from the name.

Checkpoint formats are project-specific: intermediate files can carry BF16 parameters and optimizer state; packed finals use **two bits per ternary value**, scales, and additional floating-point tensors; the public inference exports are FP32. “1.58-bit” refers to `log2(3)`, not the actual bit cost of a complete saved model. These artifacts do not load through `AutoModelForCausalLM.from_pretrained()`.

## Install and inspect

The source repository is currently **private** and requires GitHub access. The companion Hugging Face repositories are public. Examples below use PowerShell and keep caches on a writable D: drive; adjust paths for your machine. The full evaluator requires a `--storage` path on D:.

The reviewed environment used **Python 3.12.10 and PyTorch 2.6.0**, running on CPU. Dependency versions are [recorded](experiments/evaluation/model-audit-2026-10-06/environment.json); the project has no complete dependency lockfile.

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

# Restores the frozen evidence into ignored local outputs.
python -m evaluation.audit --storage D:/cpu1-readme-audit --restore-evidence
```

For GPU training, install the PyTorch build appropriate to your accelerator and operating system using the [official selector](https://pytorch.org/get-started/locally/), then install `requirements/training.txt`. `training.ablation_amd` provides the ROCm-oriented entry point. The architecture review checks its constructors on CPU/meta tensors; it does not validate a fresh CUDA or ROCm training job.

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

`training.ablation` and `training.ablation_amd` support named presets, explicit JSON configurations, synthetic CPU runs, checkpoint resume, and grouped GPU execution. [Recorded configurations](experiments/README.md) describe the published models; current presets can differ from them.

Preview a recorded AMD recipe without starting training:

```powershell
python -m runners.lightning ablation --backend amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json
```

The runner prints a plan until `--execute` is supplied. It runs inside the selected platform; it does not provision cloud resources. Kaggle and Lightning dataset recipes preserve configured document targets, which are not verified processed totals.

For a new experiment, copy the chosen configuration, set `checkpoint_repo` to your own destination or `null`, and use a fresh run name. Real training requires suitable hardware. When `checkpoint_repo` is configured, the trainer can automatically download an intermediate checkpoint to resume and upload newly saved checkpoints. Use `--skip-install` after preparing dependencies. Synthetic `--dry_run` uses CPU data with the configured model and sequence length; a small step count does not reduce their memory cost.

The two-phase deletion path requires an appropriate phase-1 checkpoint and explicit phase handling. The final filename alone does not prove phase-2 training or an achieved deletion rate. The shared evaluation disables deletion and does not measure compression benefits.

[data_pipeline/prepare.py](data_pipeline/prepare.py) extracts pretrained `Qwen/Qwen2.5-3B` signals over FineWeb. Byte supervision maps token candidates to their first byte at selected boundaries; BPE supervision retains top-k teacher entries. Optional hidden-state alignment uses projected features. This is teacher-assisted student training, rather than training Qwen from scratch.

The [dataset review](docs/model_cards/cpu1-ablation-dataset.md) inspected **three fixed shards**, not all 917. It found a reader alignment defect, mask-format problems, and UTF-8 marginalization limitations. Byte/BPE shard ranges both omit IDs 0060 and 0061, and unequal subsets must not be paired by shard index. Full row totals, teacher coverage, and reconstruction from the original documents are not established.

The precomputed loader's nominal **5 GiB selection / 10 GiB cache cleanup are not hard quotas**: its fallback can request an entire subset, and later cleanup does not cover every cache or temporary file. Without split metadata, the fallback validation slice overlaps the training data. Review these behaviors before starting a large run. The small WikiText audit uses a separate corpus.

## Evidence and project layout

| Path | Purpose |
|---|---|
| [cpu1/](cpu1/) | Quantization simulation, model components, dataset adapters, checkpoint codec |
| [training/](training/) | Ablation backends, explicit configuration loading, larger model development |
| [data_pipeline/](data_pipeline/) | Teacher extraction and dataset preparation |
| [evaluation/](evaluation/) | Strict loading, CPU audit, export, dataset inspection, architecture review |
| [experiments/](experiments/) | 34 recorded configurations and selected evaluation evidence |
| [runners/](runners/) | Kaggle/Lightning recipe previews and execution |
| [inference/](inference/) | Experimental C/C++ kernels and synthetic benchmark drivers |
| [tests/](tests/) | Offline evaluator, checkpoint, publication, and provenance checks |
| [docs/](docs/) | Model/data cards, architecture review, recovered experiment history |

The C++ drivers use synthetic weights. Their kernel or multi-stream throughput is not verified end-to-end generation speed for a published checkpoint; trained-weight loading and numerical parity need separate work. PyTorch timing records distinguish single-input forwards from actual byte generation.

The [provenance review](docs/provenance/README.md) documents 148 recovered historical paths and the retained `cpu1-experiment-2026-10-08` snapshot. A later rewrite changed six commit subjects and 28 IDs without changing their trees; the original history map describes the recovery snapshot. Available artifacts do not bind every checkpoint to an exact training source, job, dependency lock, or teacher/tokenizer revision. Deterministic end-to-end retraining is not established.

Checkpoints and datasets live on Hugging Face. `local/`, `outputs/`, environments, credentials, and generated training folders are ignored and absent from a clean checkout. Keep upload credentials outside tracked files. The full evaluator can be rerun selectively with `python -m evaluation.audit --storage D:/cpu1-readme-audit --threads 1 --runs run_02 --force`; this downloads the selected checkpoint and the small scoring corpus.

## References and licensing

Architectural inspirations: [BitNet b1.58](https://arxiv.org/abs/2402.17764), [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528), [MEGABYTE](https://arxiv.org/abs/2305.07185), [Bolmo](https://arxiv.org/abs/2512.15586), [MrT5](https://arxiv.org/abs/2410.20771), [HGRN2](https://arxiv.org/abs/2404.07904), [T-MAC](https://arxiv.org/abs/2407.00088), and [Distilling Token-Trained Models into Byte-Level Models](https://arxiv.org/abs/2602.01007).

The companion model repositories declare Apache-2.0 and the companion dataset declares MIT. This GitHub source tree has **no project license file**; those artifact declarations do not assign a source-code license here. Included WikiText excerpts retain their [upstream attribution and terms](experiments/evaluation/model-audit-2026-10-06/CORPUS_LICENSE.txt).
