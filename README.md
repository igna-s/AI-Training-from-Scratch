# CPU-1: byte-patch and ternary language-model research

CPU-1 studies language models intended for CPU inference: four-byte patches, recurrent MLGRU token mixing, ternary projections, and small floating-point modules. This repository contains the PyTorch reference implementation, ablation training and dataset pipelines, checkpoint codecs, native C/C++ kernel experiments, and reproducible evaluation evidence.

The published collection contains **34 research checkpoints with 10.7M–54.7M parameters**. Larger configurations in the source are development targets. These models are not instruction-tuned assistants, and the audit found numerical, causality, and recurrent-state defects that limit their use.

## Contents

- [Published artifacts](#published-artifacts)
- [Verified status and results](#verified-status-and-results)
- [Architecture](#architecture)
- [Ablation experiments](#ablation-experiments)
- [Installation and storage](#installation-and-storage)
- [Load, score, and generate](#load-score-and-generate)
- [Training](#training)
- [Datasets and teacher signals](#datasets-and-teacher-signals)
- [Checkpoint formats and export](#checkpoint-formats-and-export)
- [Reproduce the evaluation](#reproduce-the-evaluation)
- [Native CPU kernels](#native-cpu-kernels)
- [Experiment provenance and recovered history](#experiment-provenance-and-recovered-history)
- [Repository map](#repository-map)
- [Current limitations and next work](#current-limitations-and-next-work)
- [Research references](#research-references)
- [License](#license)

## Published artifacts

| Artifact | Contents at the audited revision | Documentation |
|---|---|---|
| [FP32 exports](https://huggingface.co/Cukinator/cpu1-ablations-final) | 34 `<run>/model.pt` files; 4.178 GB total | [Complete model card](docs/model_cards/cpu1-ablations-final.md) |
| [Training archive](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints) | 1,346 `.pt` files across 34 folders: 21 finals and 1,325 steps; 215.178 GB / 200.400 GiB | [Archive card and inventory](docs/model_cards/cpu1-ablation-checkpoints.md) |
| [Teacher dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) | Byte and BPE Parquet subsets with precomputed teacher signals | [Local dataset documentation](docs/model_cards/cpu1-ablation-dataset.md) |

These are custom PyTorch payloads, not Transformers model packages. Use the model factory in this repository. The archive is much larger than the exports; download one named file rather than an entire repository snapshot.

The audit pins export revision `65951289352c1a036b12ba6d888af037e7c90b13` and archive revision `fb402d025e843ed00fb990966bf2430c32eb958c`. Later README publications do not change the assessed weights. Exact paths, sizes, and LFS hashes are in [inventory.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/inventory.json).

## Verified status and results

The **2026-10-06 CPU audit** attempted every published FP32 export under a common protocol. Failures overlap; they are not separate groups of models.

| Check | Observed result |
|---|---|
| Strict checkpoint loading | 34/34 loaded with no missing or unexpected state keys |
| Finite loss on the shared corpus | 30/34; four exports produced nonfinite loss |
| Prefix causality | 16 PASS, 14 FAIL, 4 ERROR |
| Recurrent chunk equivalence | All 26 MLGRU exports failed; 8 Transformer exports were N/A |
| Packed/FP32 example | The `run_04` pair loaded strictly with finite outputs and identical decoded state dictionaries |

The numerical failures are `run_01_v3`, `run_02_v3`, `run_02a_byte_only_heads_v3`, and `run_08_v3`. They loaded successfully; their outputs/loss failed numerical evaluation.

Representative results illustrate the different units and checks. The [complete card](docs/model_cards/cpu1-ablations-final.md) contains all 34 rows and generated samples.

| Export | Saved step | Mean NLL | Byte PPL | Bits/byte | Prefix causality | Chunk equivalence |
|---|---:|---:|---:|---:|---|---|
| `run_01` | 912 | 5.3742 per mapped BPE token | N/A | N/A | PASS | N/A |
| `run_02` | 647 | 1.9702 per byte | 7.1723 | 2.8424 | PASS | N/A |
| `run_03_v3` | 6,187 | 1.7640 per byte | 5.8357 | 2.5449 | PASS | FAIL |
| `run_04_v3` | 6,074 | 1.5515 per byte | 4.7187 | 2.2384 | PASS | FAIL |
| `run_06_v3` | 4,237 | 1.0572 per byte | 2.8784 | 1.5253 | FAIL | FAIL |
| `run_14_v3` | 9,775 | 0.8811 per byte | 2.4136 | 1.2712 | FAIL | FAIL |

These are diagnostics on **32 short WikiText test passages**, not full benchmark scores. Low full-sequence loss from a model that fails causality may benefit from future input and does not establish causal prediction quality. A recurrent model can pass prefix causality yet fail equivalence between full-sequence and incremental execution. Check both columns before interpreting a result.

BPE maps Qwen token IDs outside the student vocabulary to ID 0. The mapped fraction in this sample was about 14.26% for the 16K vocabulary and 31.77% for the 4K vocabulary. Token PPL, byte PPL, and bits/byte measure different units. Different parameter counts, saved steps, signals, and budgets prevent treating the table as a controlled component ranking.

Timing uses PyTorch FP32 on an Intel Core i7-8750H with one thread. For example, `run_04_v3` measured 3.49 isolated forward steps/s and 21.74–25.91 generated bytes/s across three prompts. A forward step excludes autoregressive local-byte sampling; it is not a generation rate. See its [raw result](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/run_04_v3.json) for conditions and samples.

## Architecture

The saved checkpoint `config` selects the actual model. The byte path is:

```mermaid
flowchart LR
    A[UTF-8 bytes] --> B[Four-byte patches]
    B --> C[Patch embedding]
    C --> D[Global blocks: MLGRU or attention]
    D --> E[Final RMSNorm]
    E --> F[LocalByteDecoder or independent byte heads]
    F --> G[Next-patch byte distributions]
```

Global input patch `i` predicts patch `i+1`. The local decoder predicts each byte from the global representation and the preceding target bytes within that next patch. Shifted targets enter through `byte_targets`; future bytes within the target patch must not enter an earlier byte prediction. Without `byte_targets`, the decoder receives zero placeholders, suitable for shape checks but not teacher-forced likelihood scoring.

| Component | Role in the implementation |
|---|---|
| `BitLinear` / `BitEmbedding` | Simulate ternary weights with scales and straight-through gradients; Python uses floating-point operations |
| MLGRU | Intended recurrence `h[t] = f[t] * h[t-1] + (1-f[t]) * c[t]`; retains per-layer state |
| Sliding-window attention | Transformer mixer alternative; the ablation inference wrapper has no persistent KV cache |
| RMSNorm / SwiGLU | Normalization and feed-forward computation |
| `FPResidual` | Optional low-rank floating-point mixer correction |
| `PFNet` | Optional small nonlinear residual, sized by `pfnet_hidden` |
| `BolmoPatchEmbedding` | Boundary scoring and byte cross-attention, optionally across adjacent patches |
| `LocalByteDecoder` | Autoregressive next-patch prediction; usually width 128 and four heads |
| Independent byte heads | Control predicting patch positions independently |
| `DeleteGate` | Second-phase state selection/merging with hard/soft and gather controls |
| `EmbeddingAligner` | Optional projected teacher hidden-state supervision |

The `byte_dynamic` label can select boundary-aware embedding while the model still accepts fixed four-byte patches. Inspect gates and the actual forward path before assuming variable-length patching or a measured compression ratio.

The ternary alphabet has `log2(3) ≈ 1.585` bits of information per value. The codec packs each ternary value into **two bits**, with scales and other floating-point tensors stored separately. FP32 exports use four bytes per parameter. Packed disk size, unpacked parameter RAM, peak process RAM, and native throughput are distinct measurements.

## Ablation experiments

The main family uses width 512, 12 layers, and FFN width 1,376. The small family uses width 320, 8 layers, and FFN width 853. Counts come from loaded published models; historical “50M” and “10M” labels are approximate design names.

| Base run | Representation / mixer | Main distinction | Actual parameters |
|---|---|---|---:|
| `run_01` | 16K BPE / Transformer | Floating-point baseline | 54.735M |
| `run_02a_byte_only_heads` | Bytes / Transformer | Independent byte heads | 38.501M |
| `run_02` | Bytes / Transformer | LocalByteDecoder | 38.830M |
| `run_03` | Bytes / MLGRU | Floating-point recurrent control | 38.830M |
| `run_04` | Bytes / MLGRU | Ternary projections | 38.854M |
| `run_05` | Bytes / MLGRU | Adds FPResidual | 38.994M |
| `run_05b_kernel_strict` | Bytes / MLGRU | Removes the MLGRU output projection | 35.842M |
| `run_06` | Boundary-aware bytes / MLGRU | Adds Bolmo embedding | 38.994M |
| `run_07` | Boundary-aware bytes / MLGRU | Adds DeleteGate | 39.029M |
| `run_08` | Bytes / Transformer | Ternary attention control | 38.854M |
| `run_09` | Boundary-aware bytes / MLGRU | Adds PFNet, hidden width 32 | 39.428M |
| `run_10` | Boundary-aware bytes / MLGRU | Adds learned per-channel decay | 39.434M |
| `run_13` | 4K BPE / MLGRU | Small model, teacher distillation | 12.548M |
| `run_14` | Boundary-aware bytes / MLGRU | Small model, teacher distillation | 10.691M |
| `run_15` | Boundary-aware bytes / MLGRU | Adds teacher hidden-state alignment | 10.732M |
| `run_16` | Boundary-aware bytes / MLGRU | Small CE-only control on raw FineWeb | 10.691M |

Published exports include selected `_r2` and `_v3` names. Current training presets include selected `_r2` and `_r3` names; **published names and current CLI presets are not identical**. Use `--config experiments/published/configs/<run>.json` to select one of the 34 recorded published configurations; without it, the CLI uses current presets. See [the manifest](experiments/published/manifest.json). A suffix does not prove that architecture, data exposure, or training settings stayed constant.

Source budgets use `chinchilla_tokens_per_param` (including 2, 15, and 50 in different rounds). These are configured targets, not measured exposure. Complete saved configs and steps are in [the audit records](https://huggingface.co/Cukinator/cpu1-ablations-final/tree/d38678e7c5d18577f36edade64495237c72a44b5/evaluation).

## Installation and storage

The audit used Python 3.12.10 and PyTorch 2.6.0+cu124 on Windows 11, running on the CPU. Recorded dependencies include Transformers 5.3.0, tokenizers 0.22.2, huggingface_hub 1.19.0, datasets 4.8.3, and NumPy 2.3.5. See [environment.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/environment.json) for the full setup. The repository does not provide a dependency lockfile.

These PowerShell examples keep the checkout, downloads, and caches on **D:**:

```powershell
git clone https://github.com/igna-s/New-1.58.git D:/cpu1/New-1.58
Set-Location D:/cpu1/New-1.58
New-Item -ItemType Directory -Force D:/cpu1/hf, D:/cpu1/datasets, D:/cpu1/temp, D:/cpu1/pip-cache | Out-Null
$env:HF_HOME = "D:/cpu1/hf"
$env:HF_HUB_CACHE = "D:/cpu1/hf/hub"
$env:HF_DATASETS_CACHE = "D:/cpu1/datasets"
$env:TEMP = "D:/cpu1/temp"
$env:TMP = "D:/cpu1/temp"
$env:PIP_CACHE_DIR = "D:/cpu1/pip-cache"
$env:HF_HUB_DISABLE_XET = "1"
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/audit.txt
```

This is enough for loading and CPU auditing without the teacher. For GPU training, first install the PyTorch build for your OS and accelerator using the [official installation selector](https://pytorch.org/get-started/locally/), then install the relevant dependency set:

```powershell
python -m pip install -r requirements/training.txt
# Both CUDA and ROCm entry points use this dependency set.
```

Training dependencies are in `requirements/training.txt`; CPU evaluation adds `requirements/audit.txt`, and teacher extraction uses `requirements/data.txt`. Optional teacher quantization requires a suitable `bitsandbytes` or `torchao` installation. PyTorch is installed separately. Both ablation scripts otherwise install requirements when launched; use `--skip-install` after setup. ROCm uses the `torch.cuda` API here; device/OS support depends on the installed build.

Set cache variables before importing Hugging Face libraries; see their [environment variable reference](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables). The audit enforces D: storage, runs one model at a time, and deletes its downloaded `model.pt` after saving results. Manual examples retain one checkpoint until removed. Training writes checkpoints/logs relative to the working directory, so use the D: checkout to keep those outputs there.

Public downloads need no write token. Publishing and training uploads use your own `HF_TOKEN` through the environment or Hugging Face authentication; keep credentials outside tracked files. Scripts report only whether the token is configured, without revealing its characters. The card uploader hides interactive token entry. Private key files, credential JSONs, and local secret directories are also ignored.

## Load, score, and generate

Run examples from the repository root. `AutoModelForCausalLM.from_pretrained()` cannot load these checkpoints. This loads the audited `run_04` FP32 export and scores correctly shifted byte targets:

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from training.ablation import build_ablation_model

torch.set_num_threads(1)
path = hf_hub_download(
    "Cukinator/cpu1-ablations-final", "run_04/model.pt",
    revision="65951289352c1a036b12ba6d888af037e7c90b13",
    local_dir="D:/cpu1/active", cache_dir="D:/cpu1/hf/hub",
)
# Trusted checkpoint from the pinned project source.
payload = torch.load(path, map_location="cpu", weights_only=False)
config = dict(payload["config"])
config["use_gradient_checkpointing"] = False
model = build_ablation_model(config)
model.load_state_dict(payload["state_dict"], strict=True)
model.eval()
del payload

patch_size = config["patch_size"]
raw = list(b"The history of artificial intelligence is long.")
patches = torch.tensor(raw[:len(raw) // patch_size * patch_size]).reshape(1, -1, patch_size)
x, targets = patches[:, :-1], patches[:, 1:]
with torch.inference_mode():
    heads, states, extra = model(
        x, byte_targets=targets, training_phase=1, target_elimination=0.0,
    )
    logits = torch.stack(heads, dim=2)  # [batch, input patches, byte positions, 256]
    nll = F.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
assert torch.isfinite(nll)
print("Mean NLL per byte:", nll.item())
```

This is a loadable example, not a quality recommendation. For a packed final, replace payload extraction with `state, config = load_ablation_checkpoint(path)` from `training.ablation`, then build and load `state` strictly. Download the named archive file only after releasing/removing the previous model. [Loading evidence](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/loading_examples.json) verifies only the `run_04` export/archive pair.

Sample bytes with the loaded CPU model using the audit helper:

```python
from evaluation.audit import generation
torch.manual_seed(42)
generated = generation(model, config, "The history of artificial intelligence", length=128)
print(generated.decode("utf-8", errors="replace"))
```

The helper prefills recurrent state once and carries it thereafter; Transformer variants retain and recompute context instead. Sampling uses temperature 1.0, top-k 40, top-p 0.9, and sequential local-byte prediction. Prompt bytes are zero-padded to a patch boundary. Invalid UTF-8 can occur; replacement decoding is only for display, and the audit retains raw bytes as hex. Inspect samples and state failures before interpreting generation results.

Training scripts also expose inference through top-of-file `TRAIN`, `INFER`, and `INFER_CONFIG` settings. That built-in path is separate from the helper used for fresh audit measurements.

## Training

### Individual ablations

The standard entry point is [training/ablation.py](training/ablation.py). With dependencies installed:

```powershell
# Synthetic CPU data; checks training plumbing.
python -m training.ablation --run run_04 --dry_run --total_steps 2 --skip-install

# Dataset and budget selected by the preset.
python -m training.ablation --run run_04 --skip-install

# Resume a local intermediate checkpoint.
python -m training.ablation --run run_04 --resume_ckpt D:/cpu1/checkpoint_run_04_step640.pt --skip-install
```

Dry runs use synthetic CPU data with the preset model and sequence length; `--total_steps` limits steps, not memory. For a small CPU check, reduce the preset sequence length and batch settings first. Synthetic runs do not establish quality or real-data throughput. Normal training selects CUDA/ROCm devices, uses AdamW parameter groups and mixed precision, and supports multiple GPUs. `--total_steps` overrides the calculated budget. Presets and optimizer controls are in `BASE_CONFIG`, `ALL_RUN_CONFIGS`, and `get_run_config()`.

Without `--resume_ckpt`, normal training can automatically download the latest matching phase's step checkpoint from its configured checkpoint repository. For a separate fresh experiment, set `checkpoint_repo` and the output run directory appropriately. Training saves under `<run>/`, records metrics, and attempts Hub uploads using configured authentication. Finals are inference artifacts; intermediate checkpoints can include optimizer state.

### DeleteGate phases

Runs in `TWO_PHASE_RUNS` automatically chain ordinary training into DeleteGate training. The gate is active in phase 2; a phase-1 gate's presence does not prove training or its elimination target. Start phase 2 explicitly with:

```powershell
python -m training.ablation --run run_07 --phase 2 --phase1_ckpt D:/cpu1/run_07/checkpoint_run_07_final.pt --skip-install
```

Inspect saved `_current_phase`; the final filename alone does not distinguish phases. The common audit disables elimination (`training_phase=1`, `target_elimination=0.0`), so it does not measure gate compression or phase-2 speedups.

### Orchestration and AMD

```powershell
# Cap execution at one training process.
python -m training.ablation --orchestrate --runs run_04 run_05 --max-parallel 1 --skip-install

# Indices refer to PARALLEL_GROUPS in the source.
python -m training.ablation --orchestrate --groups 2 3 --max-parallel 1 --skip-install

# ROCm-oriented entry point with its dependencies installed.
python -m training.ablation_amd --run run_04 --skip-install
```

Orchestration assigns visible GPU slots, processes groups in order, chains required phases, and logs to `run_logs/` unless `--log-dir` changes it. `--steps` sets orchestration budgets; single-run mode uses `--total_steps`. Non-dry orchestrated subprocesses automatically request `--hf-mount` and fall back to download if mounting fails. Child processes reuse the parent environment and skip dependency installation. Training can retain multiple checkpoints with one active process; the one-file cleanup policy belongs to the audit.

### Larger CPU1ModelV2 path

[training/final.py](training/final.py) constructs `CPU1ModelV2` separately from ablations. Defaults in `cpu1/components.py` are width 1,024, 14 layers, FFN width 2,730, FPResidual rank 8, and PFNet hidden width 64. They do not identify a published, evaluated production model.

```powershell
# Synthetic check; still constructs the larger model.
python -m training.final --dry-run --dry-run-samples 2 --seq-len 8 --batch-per-gpu 1 --total-steps 1 --output-dir D:/cpu1/final-smoke

# Inspect this separate development CLI.
python -m training.final --help
```

Phase 1 selects CE-only FineWeb by default, but its DataLoader currently requests `shuffle=True` for an iterable streaming dataset; that path needs a loader correction before normal training. The larger training path was not validated by the export audit.

Its options use hyphens: `--resume-from`, `--data-path`, `--phase2`, and `--phase1-ckpt`, unlike ablation underscore options. `--upload-hf` enables phase-1 uploads. Precomputed distillation data is selected through `--data-path`; the teacher is not run automatically. Consult `--help` and phase-2 code for separate output behavior.

## Datasets and teacher signals

`load_ablation_dataset()` in [cpu1/components.py](cpu1/components.py) routes presets as follows:

| Route | Representation and supervision |
|---|---|
| `HuggingFaceFW/fineweb` | Streams `sample-10BT`; preset selects byte/BPE adaptation and CE training |
| `cpu1-ablation-dataset` / `byte_marginalized` | Patches, shifted targets, teacher probabilities/masks, optional projected hidden states |
| `cpu1-ablation-dataset` / `bpe_tokenized` | Token IDs, shifted labels, teacher top-k probabilities/IDs/masks |
| Dry run | Synthetic tensors matching the interface |

At revision `bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf`, the dataset has 480 byte Parquet files (0.687 GB compressed) and 437 BPE files (67.332 GB compressed). IDs 0060 and 0061 are absent in both ranges. Unequal subsets must not be zipped by row or shard index. These are file-listing measurements; row counts and teacher coverage were not fully audited.

Only `train` is declared by the dataset card. The adapter can use `split_info.json`; without it, validation selects the last 5% while training retains the full dataset. That fallback does **not** establish an independent held-out set. Fresh model evaluation uses a separate WikiText test corpus.

The non-mounted precomputed loader selects shards up to a nominal 5 GiB per subset and deletes the configured datasets cache if its size exceeds 10 GiB. Use a dedicated `HF_DATASETS_CACHE` directory for this loader. Byte and BPE runs can therefore see different coverage. `--hf-mount` selects a mounted-shard path; it does not establish document equivalence.

Inspect one row without downloading the complete dataset:

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_DATASETS_CACHE"] = "D:/cpu1/datasets"
from datasets import load_dataset
rows = load_dataset(
    "Cukinator/cpu1-ablation-dataset", "byte_marginalized",
    revision="bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf",
    split="train", streaming=True, cache_dir="D:/cpu1/datasets",
)
print(next(iter(rows)).keys())
```

### Building teacher data

[data_pipeline/prepare.py](data_pipeline/prepare.py) extracts `Qwen/Qwen2.5-3B` signals over FineWeb and builds both representations, with local, Hub-streaming, and mounted workflows. Writer settings include four-byte patches, 5,000-step sequences, top-128 BPE teacher entries, and projected hidden width 128. Inspect rows and `teacher_dim`; not every stored shard has been checked against these settings.

Byte supervision marginalizes BPE candidates to their first byte at mapped boundaries and uses coverage masks. It is not a complete joint teacher distribution for all four bytes. BPE student vocabularies also filter/map out-of-range IDs.

```powershell
# Alignment/marginalization tests; no full teacher extraction.
python -m data_pipeline.prepare

# Full extraction/build/publication; needs teacher compute and Hub write auth.
python -m data_pipeline.prepare --n_docs 50000 --stream-to-hub

# Convert extracted logprobs into local training data.
python -m data_pipeline.prepare --logprobs-dir D:/cpu1/logprobs --output-dir D:/cpu1/dataset
```

The [platform recipes](docs/provenance/README.md#current-platform-recipes) print a plan until `--execute` is supplied. Kaggle preserves the 500-document chunk recipe and Lightning the 3,000-document recipe, both with a configured 850,000-document target. These targets do not establish the number actually processed. The full pipeline can publish as it builds; it is not a read-only preview. `--skip_extraction` reuses extraction output, `--only_upload` publishes existing data, and `--shard-offset` sets numbering. Full-pipeline output defaults to `./shared_dataset`. Current package-based launchers live in `runners/`; the original platform wrappers and notebooks are preserved in Git history. Optional mounts use `cpu1/hub_storage.py` and require a separately installed hf-mount on Linux/macOS.

## Checkpoint formats and export

| Payload | Intended use | Loading |
|---|---|---|
| `format="fp32"` | Unpacked parameters, config, step, metrics, run name, source metadata | `torch.load()`, factory, strict `state_dict` loading |
| `format="compact_2bit"` | Packed ternary final with scales and remaining float tensors | `load_ablation_checkpoint()` to decode, then build/load |
| Floating-point final | Baseline inference artifact | Inspect `format`; use checkpoint loader |
| Intermediate step | Model/config and, where saved, optimizer/resumption state | Matching training entry point |

Inspect payloads instead of assuming every final is packed or every step interchangeable. Historical `metrics` may be empty or use older protocols; fresh evidence is separate.

[evaluation/export.py](evaluation/export.py) reads local finals at `local/models/ablation_models/<run>/checkpoint_<run>_final.pt` and writes `local/models/ablations_final/<run>/model.pt`:

```powershell
python -m evaluation.export --runs run_04 --skip-upload
```

The source final must exist locally. The exporter does not recreate every current Hub export automatically. Without `--skip-upload`, it can publish models and a README; use the complete audit/card workflow for fresh results. Model files and cache/build outputs are excluded by `.gitignore`.

## Reproduce the evaluation

Use the activated CPU environment from installation:

```powershell
# Pinned collection, one model file at a time.
python -m evaluation.audit --storage D:/cpu1-readme-audit --threads 1

# Force fresh evaluation of named exports.
python -m evaluation.audit --storage D:/cpu1-readme-audit --threads 1 --runs run_04 run_14_v3 --force

# Restore the published metadata/corpus only, without downloading any model.
python -m evaluation.audit --storage D:/cpu1-readme-audit --restore-evidence

# Generate local cards from complete evidence; preserves this root README.
python -m evaluation.audit --storage D:/cpu1-readme-audit --cards

# Publish both model cards and evidence with owner write authentication.
python -m evaluation.audit --storage D:/cpu1-readme-audit --upload

# Evaluator regression checks; no model downloads.
python -m unittest discover -s tests -v

# Isolate the scan defect without a checkpoint.
python -m evaluation.diagnostics
```

A clean checkout contains source and documentation, not model weights or generated reports. `--restore-evidence` downloads the published JSONs and corpus license for card regeneration without evaluating models. For a new audit, the model/corpus revisions remain pinned. Existing records are reused when the evaluator hash matches; `--force` reruns them. Card generation requires all 34 records at the pinned revision with consistent evaluator/corpus hashes. Upload verifies audited model paths, sizes, and LFS hashes still match before publishing. These commands update `outputs/audit/` and local Hub templates.

Scoring takes the first 32 non-heading WikiText test paragraphs with at least 260 UTF-8 bytes, truncating to 260 bytes. Byte scoring uses the first patch as context, shifted next-patch targets, and correct local teacher forcing. Loss is weighted by target count. BPE uses the training vocabulary mapping. Batch size is 1, no teacher is loaded, and elimination is disabled.

Correctness covers shapes, probabilities, determinism, recurrent carry, and prefix causality. Timing separates 10 warmup / 100 measured single-input forwards from real generation. Byte samples use three prompts, 128 bytes each, seeds 42–44, temperature 1.0, top-k 40, and top-p 0.9. BPE generation is N/A because its lossy mapping is not comparable to byte sampling.

| Evidence | Contents |
|---|---|
| [inventory.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/inventory.json) | Model revisions, sizes, and LFS hashes |
| [corpus.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/corpus.json) | Exact passages, upstream revision, selection |
| [environment.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/environment.json) | Packages, hardware, threads, source metadata |
| [Per-run JSON files](https://huggingface.co/Cukinator/cpu1-ablations-final/tree/d38678e7c5d18577f36edade64495237c72a44b5/evaluation) | Configs, loading, losses, failures, timings, raw generated bytes |
| [implementation_diagnostics.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/implementation_diagnostics.json) | Parallel scan versus serial and chunked execution |
| [loading_examples.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/loading_examples.json) | Sequential packed/FP32 `run_04` smoke checks |
| [publication.json](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/publication.json) | Published Hub card commit IDs |

The Qwen tokenizer's resolved commit was not captured (`null` in the environment record). Models and corpus are pinned, but this missing pin limits exact BPE reproducibility. Dependencies are recorded rather than locked.

The current tree contains the common audit and its supporting tools. Historical runners, experimental variants and 34 older training reports are preserved in Git history; they use different protocols and are not fresh evaluation results. Historical evidence remains on Hugging Face; restored reports and newly generated results stay under ignored `outputs/audit/`.

## Native CPU kernels

[inference/kernels.cpp](inference/kernels.cpp) implements experimental packed-weight kernels and forward paths. [inference/benchmark.c](inference/benchmark.c) is a C++-compiled driver using **random weights and synthetic inputs**. It does not load published trained checkpoints, and the Python audit does not call it.

Build from the root with a GCC/MinGW toolchain supporting AVX2, FMA, and OpenMP:

```powershell
g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math -std=c++11 -x c++ -o benchmark.exe inference/benchmark.c inference/kernels.cpp -lm
./benchmark.exe --mode 2 --threads 1 --patches 100 --warmup 20 --local 1
```

| Mode | Benchmark path |
|---:|---|
| 0 | Full forward with OpenMP |
| 1 | Per-layer forward |
| 2 | Single-thread full forward |
| 3 | Multiple independent streams via `--batch` |
| 4 | Layer pipeline via `--chunk`, `--stages`, `--pin`, `--calibrate` |

The driver accepts `--d_model`, `--n_layers`, `--d_ff`, and `--fp_rank`; defaults differ from the larger PyTorch model. `--local 1` includes local decoding. Multi-stream aggregate and pipeline patch throughput are not end-to-end trained-model generation. [Build notes](inference/links.md) cover further experiments; trained-checkpoint loading and numerical parity need separate verification.

## Experiment provenance and recovered history

The first cleanup filtered experimental code too broadly. The correction restores **148 historical paths**: extraction/marginalization, early architectures, ablation implementations, Kaggle/Lightning code and notebooks, historical evaluation/inference, and 34 training reports plus their summary. The 87 original commits and the security commit retain their messages, authors, dates and relationships after remapping; historical hashes change. The current tree keeps the organized packages.

[The provenance review](docs/provenance/README.md) explains every category. [The restored-file manifest](docs/provenance/restored-files.json) lists all recovered paths, their hashes and source links, plus the 109 paths still excluded. [The history map](docs/provenance/history-map.json) maps original commit IDs to their public replacements. Downloaded weights/caches, papers, compiled outputs and credentials remain excluded.

[The published configuration manifest](experiments/published/manifest.json) binds 34 configuration records to pinned checkpoint hashes, saved steps, audited parameter counts and result URLs. The `_v3` records preserve the 150 tokens/parameter setting, consistent with the AMD preset. This does not establish each job's source commit or platform. The records came from the audit loader, which disabled gradient checkpointing; GPU training enables it. They are not an untouched copy of the checkpoint configuration bytes.

```powershell
# Print cloud recipes without starting work.
python -m runners.kaggle dataset
python -m runners.lightning dataset
python -m runners.lightning ablation --backend amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json

# Inspect three pinned dataset shards and a synthetic tokenizer case, using D:.
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer
```

The dataset inspection reproduced a dense-signal reader mismatch (1,333 nonzero stored distributions became 369) and short boolean masks treated as indices. The tokenizer test also reproduced replacement-character decoding for partial UTF-8 tokens. See [the dataset card](docs/model_cards/cpu1-ablation-dataset.md) for sample scope and evidence. These defects remain visible; the recovery did not repair data or trained behavior, retrain models, or re-date the October 6 model evaluation. Exact end-to-end reconstruction remains limited by missing job/source/environment and per-shard extraction records.

## Repository map

```text
cpu1/
  components.py             Shared modules, adapters, losses, checkpoint codecs
  hub_storage.py            Optional HF mounts; credentials in child environment
training/
  ablation.py               Main training/inference entry point and presets
  ablation_amd.py           ROCm-oriented variant
  final.py                  Larger CPU1ModelV2 training path
  configuration.py          Explicit published configuration loader
data_pipeline/
  prepare.py                Teacher extraction and dataset preparation
runners/                    Kaggle/Lightning recipes; CUDA or AMD training
experiments/published/      34 config records and pinned checkpoint manifest
evaluation/
  audit.py                  Common scoring, correctness, timing and generation
  checkpoints.py            Strict checkpoint loader and model metadata
  correctness.py            Architecture-aware correctness checks
  cards.py                  Model-card generation and publication
  export.py                 Packed-to-FP32 export
  diagnostics.py            Isolated recurrence diagnostic
  verify_checkpoints.py     Sequential packed/FP32 loading example
  helpers.py                Check result handling
  dataset.py                Pinned shard/reader/tokenizer inspection
inference/                  Native kernels, synthetic benchmark, build notes
tests/                      Offline evaluator and integration tests
docs/model_cards/           Three canonical model/dataset cards
docs/provenance/            Recovery report, file manifest and commit map
requirements/               Training, audit and data dependency sets
README.md                   Project guide and verified status
.gitignore                  Local assets, credentials, reports and build outputs
```

`local/` holds private checkpoints, reference material, and archived files on the developer machine. `outputs/` holds generated reports. Both directories are ignored and absent from a clean GitHub checkout. Training run folders, virtual environments, caches, and model/dataset binaries are also ignored. Published evaluation evidence and model weights are linked from Hugging Face rather than duplicated in this repository.

## Current limitations and next work

`AblationMLGRU._parallel_scan()` differs from its defining serial recurrence and chunked execution. With the saved seed/shape, maximum differences were 19.1005 and 11.4713 against tolerance `1e-4`. The `cummax`/`cumsum` construction does not correctly rescale cumulative terms when its stabilizing maximum changes. This is an implementation defect, not a successful streaming test.

Saved Bolmo configurations with noncausal boundary/cross-patch context allow future patches to change prefix logits in 14 exports. Fixing causality flags or recurrence changes behavior, so repairs need fresh evaluation rather than inheriting these scores. Existing weights may need retraining.

Next milestones: resolve and validate teacher array positional layout and UTF-8 conversion; correct and validate recurrence; make the boundary path causal in full and incremental execution; define independent, comparable splits; rerun quality/generation; and establish trained-checkpoint parity for native kernels. This collection currently has no demonstrated larger-model production result, reliable gate speedup, external leaderboard result, or packed native generation rate.

## Research references

The project draws ideas from these papers; experimental local code should not be assumed to reproduce their implementations or results:

- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528): recurrent mixing and low-bit projections.
- [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185): global/local byte prediction.
- [Bolmo](https://arxiv.org/abs/2512.15586): byte-level modeling and boundary experiments.
- [MrT5](https://arxiv.org/abs/2410.20771): learned deletion/merging.
- [HGRN2](https://arxiv.org/abs/2404.07904): gated recurrence and state expansion ideas.
- [T-MAC](https://arxiv.org/abs/2407.00088): lookup-based low-bit CPU computation.
- [Distilling token models into byte-level models](https://arxiv.org/abs/2602.01007): supervision across representations.

References link to their original publications; PDFs and personal working notes are excluded from Git. Config comments are design context; measurements come from evaluation records.

## License

The two Hugging Face model repositories declare Apache-2.0; companion dataset metadata declares MIT. This GitHub checkout has no standalone `LICENSE` file, so Hub labels alone do not establish a repository-wide source license. FineWeb, Qwen, papers, and other assets retain upstream terms. WikiText excerpts retain CC-BY-SA-3.0/GFDL terms documented in [CORPUS_LICENSE.txt](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/d38678e7c5d18577f36edade64495237c72a44b5/evaluation/CORPUS_LICENSE.txt).
