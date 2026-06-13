---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- 1.58-bit
- ternary
- cpu-1
- ablation
---

# CPU-1 Ablation Study — Complete Checkpoint Archive

**Repo:** `Cukinator/cpu1-ablation-checkpoints`
**Unpacked weights:** [`Cukinator/cpu1-ablations-final`](https://huggingface.co/Cukinator/cpu1-ablations-final)
**Source code:** [`github.com/Cukinator/1.58bits`](https://github.com/Cukinator/1.58bits)

---

This repository is the **complete checkpoint archive** for the CPU-1 ablation study:
a systematic, one-component-at-a-time dissection of the design choices behind a
1.58-bit ternary language model optimised for CPU inference.

**1346 checkpoint files · 200.4 GB total · 34 run folders**

Two checkpoint flavours are stored per run:

| File pattern | Format | Size | Purpose |
|---|---|---|---|
| `checkpoint_<run>_final.pt` | `compact_2bit` (2-bit packed ternary + bf16 scales) | ~3–104 MB | Final inference weights — load directly with `load_ablation_checkpoint()` |
| `checkpoint_<run>_step<N>.pt` | bf16 model + bf16 optimizer state | ~20–313 MB | Mid-training resume point |
| `checkpoint_<run>_phase2_step<N>.pt` | bf16 model + bf16 optimizer state | same | Phase-2 (DeleteGate fine-tune) resume point |

> **Just want to run inference?** Use [`Cukinator/cpu1-ablations-final`](https://huggingface.co/Cukinator/cpu1-ablations-final) — plain float32 `.pt` files, no unpacking needed.

---

## Architecture overview

| Scale | `d_model` | `n_layers` | `d_ff` | `n_heads` | Params |
|---|---|---|---|---|---|
| **50M** (runs 01–10) | 512 | 12 | 1376 | 8 | ~50M |
| **10M** (runs 13–16) | 320 | 8 | 853 | 8 | ~10M |

All runs use:
- **1.58-bit ternary weights** (`BitLinear` / `BitEmbedding`) by default — FP16 runs are explicit baselines
- **Byte-level patch tokenisation** (patch_size=4) except `run_01` / `run_13` which use BPE
- **Chinchilla-scaled training budget**: 2 tok/param (r1), 15 tok/param (r2), dense step checkpoints (r3)

---

## 50M-parameter ablation chain (`d_model=512, n_layers=12`)

Each run adds **exactly one component** to the previous:

```
run_01  Transformer + BPE 16K + FP16          ← absolute baseline
  │
  ├─ run_02a  + byte patches (no LocalByteDecoder)
  │     └─ run_02  + LocalByteDecoder (MegaByte intra-patch)
  │           └─ run_03  swap Transformer → MLGRU  (FP16)
  │                 └─ run_04  + ternary quantisation (1.58-bit)
  │                       └─ run_05  + FPResidual (CPU-1 core)
  │                             ├─ run_05b  − W_o  (production kernel layout)
  │                             ├─ run_08   swap MLGRU → folded Transformer
  │                             └─ run_06  + BolmoPatchEmbedding
  │                                   └─ run_07  + DeleteGate  ⭐ CPU-1 COMPLETE
  │                                         └─ run_09  + PFNet (pfnet_hidden=32)
  │                                               └─ run_10  + per-channel decay
  │
  ├─ Round 2: run_04_r2, run_07_r2  (same architecture, 15 tok/param)
  └─ Round 3: run_XX_v3/            (dense step-by-step checkpoints)
```

### Round 1 — 2 tok/param

| Run | Internal name | Steps saved | Max step | `_final` size | Step ckpt size | Total | Description |
|---|---|---|---|---|---|---|---|
| `run_01` | `transformer_bpe_fp16` | 21 | 900 | 104.4 MB | 313.3 MB | 6684 MB | Transformer + BPE 16K vocab + FP16. Absolute baseline. |
| `run_02a_byte_only_heads` | `transformer_byte_fp16_no_lbd` | 21 | 640 | 73.5 MB | 220.5 MB | 4703 MB | +Byte patches, 4 independent byte heads (no LocalByteDecoder). Tokenisation isolation. |
| `run_02` | `transformer_byte_fp16` | 21 | 640 | 74.1 MB | 222.3 MB | 4743 MB | +LocalByteDecoder — MegaByte autoregressive intra-patch chain over run_02a. |
| `run_03` | `mlgru_byte_fp16` | 21 | 640 | 74.1 MB | 222.4 MB | 4744 MB | Swap Transformer → MLGRU. FP16, byte patches + LocalByteDecoder. |
| `run_04` | `mlgru_byte_ternary` | 21 | 640 | 9.4 MB | 74.2 MB | 1567 MB | +Ternary quantisation (1.58-bit). Isolates BitNet cost on MLGRU+byte. |
| `run_05` | `mlgru_byte_ternary_fpres` | 21 | 640 | 9.7 MB | 74.5 MB | 1574 MB | +FPResidual (low-rank FP16 correction). CPU-1 core architecture. |
| `run_05b_kernel_strict` | `mlgru_kernel_strict` | 21 | 600 | 8.9 MB | 68.4 MB | 1446 MB | Branch from run_05: remove W_o to match production C++ OMP kernel layout. |
| `run_06` | `mlgru_byte_ternary_fpres_bolmo` | 21 | 640 | 9.7 MB | 74.5 MB | 1574 MB | +BolmoPatchEmbedding — boundary-aware patch encoding via cross-attention. |
| `run_07` | `cpu1_complete` | 21 | 640 | 9.8 MB | 74.5 MB | 1575 MB | +DeleteGate (real MrT5 gather, ~40% elimination at layer N//2). CPU-1 COMPLETE. ⭐ |
| `run_08` | `folded_transformer_byte_ternary` | 21 | 640 | 9.4 MB | 74.2 MB | 1567 MB | Branch from run_05: swap MLGRU → ternary sliding-window Transformer (window=128). |
| `run_09` | `cpu1_pfnet` | 21 | 640 | 9.9 MB | 75.3 MB | 1591 MB | +PFNet (pfnet_hidden=32): cache-resident nonlinear residual per block. |
| `run_10` | `cpu1_decay_learned` | 21 | 640 | 9.9 MB | 75.3 MB | 1592 MB | +Per-channel learnable decay in MLGRU (RWKV/HGRN2-style). Chain terminus. |

### Round 2 — 15 tok/param

| Run | Steps saved | Max step | `_final` size | Step ckpt size | Total | Description |
|---|---|---|---|---|---|---|
| `run_04_r2` | 21 | 4,840 | 9.4 MB | 74.2 MB | 1567 MB | run_04 architecture at 15 tok/param — ternary MLGRU baseline with real budget. |
| `run_07_r2` | 21 | 4,860 | — (no final) | 74.5 MB | 1565 MB | run_07 (CPU-1 COMPLETE) at 15 tok/param. No `_final` — training stopped early. ⭐ |

### Round 3 — step-by-step checkpoints (50M)

Dense step-by-step checkpoints from the v3 training run. Full **bf16 model + bf16 optimizer state** per step — no `_final.pt`.

| Folder | Step checkpoints | Max step | Per-file size | Total size |
|---|---|---|---|---|
| `run_01_v3/` | 72 | 12,173 | 313.3 MB | 22560 MB |
| `run_02_v3/` | 73 | 10,963 | 222.3 MB | 16231 MB |
| `run_02a_byte_only_heads_v3/` | 75 | 16,327 | 220.5 MB | 16534 MB |
| `run_03_v3/` | 71 | 6,187 | 222.4 MB | 15788 MB |
| `run_04_v3/` | 71 | 6,074 | 222.5 MB | 15800 MB |
| `run_05_v3/` | 71 | 4,390 | 223.4 MB | 15861 MB |
| `run_05b_kernel_strict_v3/` | 71 | 2,573 | 205.3 MB | 14579 MB |
| `run_06_v3/` | 71 | 4,237 | 223.4 MB | 15862 MB |
| `run_08_v3/` | 71 | 3,858 | 222.5 MB | 15799 MB |

> **Round 3 total (9 folders):** 145.5 GB

---

## 10M-parameter runs (`d_model=320, n_layers=8`)

All use **CPU-1 COMPLETE architecture**. Training strategy is the only variable.

**Datasets:**
- `run_13`, `run_14`, `run_15`: `Cukinator/cpu1-ablation-dataset` (~37.5M tokens, Qwen2.5-3B teacher logprobs + hidden states)
- `run_16`: `HuggingFaceFW/fineweb` directly (CE-only, no teacher — the control)

### Round 1 — 2 tok/param

| Run | Internal name | Steps saved | Max step | `_final` size | Step ckpt size | Total | Description |
|---|---|---|---|---|---|---|---|
| `run_13` | `small_cpu1_bpe` | 21 | 600 | 3.2 MB | 71.9 MB | 1513 MB | CPU-1 @ 10M, BPE 4K vocab, distillation from Qwen2.5-3B logprobs. Does BPE beat bytes at 10M? |
| `run_14` | `small_cpu1_byte` | 21 | 600 | 2.8 MB | 20.5 MB | 432 MB | CPU-1 @ 10M, byte-level, distillation (logprobs only). Pure byte baseline. ⭐ |
| `run_15` | `small_cpu1_byte_hidden` | 21 | 600 | 2.9 MB | 20.5 MB | 434 MB | run_14 + EmbeddingAligner (hidden-state distillation). Does aligning hidden reps help? |
| `run_16` | `small_cpu1_raw_bytes` | 21 | 600 | 2.8 MB | 20.5 MB | 432 MB | CPU-1 @ 10M, byte-level, zero teacher (CE-only on FineWeb). Do Qwen logprobs in run_14 actually help? |

### Round 2 — 15 tok/param

| Run | Steps saved | Max step | `_final` size | Step ckpt size | Total | Description |
|---|---|---|---|---|---|---|
| `run_13_r2` | 21 | 1,560 | 3.2 MB | 71.9 MB | 1513 MB | run_13 at 15 tok/param — BPE + ternary with sufficient budget. |
| `run_14_r2` | 21 | 1,320 | 2.8 MB | 20.5 MB | 432 MB | run_14 at 15 tok/param — byte baseline with sufficient budget. ⭐ |
| `run_15_r2` | 21 | 1,340 | 2.9 MB | 20.5 MB | 434 MB | run_15 at 15 tok/param — byte + hidden distillation with sufficient budget. |
| `run_16_r2` | 21 | 1,320 | 2.8 MB | 20.5 MB | 432 MB | run_16 at 15 tok/param — zero-teacher bytes with sufficient budget. |

### Round 3 — step-by-step checkpoints (10M)

Dense step-by-step checkpoints from the v3 training run. Full **bf16 model + bf16 optimizer state** per step — no `_final.pt`.

| Folder | Step checkpoints | Max step | Per-file size | Total size |
|---|---|---|---|---|
| `run_13_v3/` | 72 | 9,290 | 71.9 MB | 5175 MB |
| `run_14_v3/` | 75 | 9,775 | 61.3 MB | 4597 MB |
| `run_15_v3/` | 70 | 2,350 | 61.5 MB | 4307 MB |

> **Round 3 total (3 folders):** 13.7 GB

---

## Quick start

### Load a `compact_2bit` final checkpoint

```python
import sys
sys.path.insert(0, "/path/to/1.58bits")
from train_ablation_amd import load_ablation_checkpoint, build_ablation_model, generate
import torch

state, config = load_ablation_checkpoint("run_07/checkpoint_run_07_final.pt")
model = build_ablation_model(config)
model.load_state_dict(state, strict=False)
model.eval()

output = generate(model, "Once upon a time", max_new_bytes=128, config=config, device=torch.device("cpu"))
print(output)
```

### Resume training from a step checkpoint

```bash
python train_ablation_amd.py --run run_07 --resume_from run_07/checkpoint_run_07_step640.pt
```

### Download with huggingface_hub

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Cukinator/cpu1-ablation-checkpoints",
    filename="run_07/checkpoint_run_07_final.pt",
    repo_type="model",
)
```

---

## Related repositories

| Repo | Contents |
|---|---|
| [`Cukinator/cpu1-ablation-checkpoints`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints) | **This repo** — raw training checkpoints (`compact_2bit` finals + bf16 step files) |
| [`Cukinator/cpu1-ablations-final`](https://huggingface.co/Cukinator/cpu1-ablations-final) | Unpacked float32 weights — ready for `model.load_state_dict()` without any helper |
| [`Cukinator/cpu1-ablation-dataset`](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) | Pre-processed training dataset with Qwen2.5-3B teacher logprobs + hidden states |

---

## License

Apache-2.0. See [github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits).
