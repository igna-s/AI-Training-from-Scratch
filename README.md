# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen2.5-3B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

---

## Architecture (v2 — 170M)

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| `n_layers` | 14 |
| `d_ff` | 2730 (PyTorch) / 2688 (C benchmark, rounded to multiple of 64) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| `pfnet_hidden` | 64 (PFNet hidden dim, 224 KB total across all layers) |
| `mlgru_state` | 28 KB fp16 / 56 KB fp32 (14 × 1024 × {2, 4} bytes — fits in L2) |
| **Total parameters** | **~170M** |

### Key Components

- **BitEmbedding** — Ternary embedding table {-1, 0, 1} with per-row FP16 scale (Slender-Mamba, COLING 2025).
- **BytePatchEmbeddingV2** — Converts 4-byte patches to `d_model` dims: BitEmbedding (8 dims/byte) → concat (32 dims) → BitLinear → d_model.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE).
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with BitLinear, initialized to zero.
- **PFNet** — 2-layer ternary MLP residual per block (hidden=64). All PFNet weights across 14 layers (224 KB) fit in L2 alongside the MLGRU state (28 KB fp16), staying cache-resident while main weights stream from RAM.
- **MatMulFreeTokenMixer (MLGRU)** — GRU-style recurrence with Hadamard products only, no Q@K^T (arXiv 2406.02528). O(d_model) per token, zero KV cache.
- **SwiGLU FFN** — Gated feed-forward with fully ternary weights.
- **BitHead** — 4 ternary prediction heads (one per byte in next patch) via MegaByte LocalByteDecoder.
- **RMSNorm** — Pre-norm before token mixer and FFN.

### Memory at Inference

| Component | v1 (Attention) | v2 (MLGRU + PFNet) |
|---|---|---|
| Token mixer state | 80 MB (KV cache) | **28 KB fp16** / **56 KB fp32** — L2 |
| PFNet weights | — | **224 KB** (all layers) — L2 |
| Active layer weights | — | ~3.1 MB (1 layer) — L3 |
| Full model weights | — | ~44 MB — RAM streaming |

### Extended Architecture (v3)

CPU-1 v3 extends the base architecture with three optional modules:

- **BoundaryPredictor** (Bolmo-inspired): Replaces fixed 4-byte patches with dynamic patches that align with natural language boundaries. Adds ~2KB of parameters.
- **DeleteGate** (MrT5-inspired): Placed after layer N//2, learns to discard uninformative states before the remaining layers. Reduces active states by up to 40%, activated only in training Phase 2.
- **EmbeddingAligner**: Aligns the student's hidden space with the teacher's embedding space via cosine similarity loss. Trained in Phase 1 only, frozen in Phase 2.

All three modules are optional and controlled by config flags:
```
use_boundary_predictor: false
use_delete_gate: false
use_embedding_alignment: false
```

### Byte Head — MegaByte Local Decoder

All byte-level runs (run_02 onward) predict the 4 bytes of each patch through a **MegaByte local decoder** ([Yu et al., 2023](papers/MEGABYTE-Predicting-byte-sequences.pdf)) rather than four independent heads. This replaces the conditional independence assumption `P(b₀ b₁ b₂ b₃ | ctx) = ∏ P(bᵢ | ctx)` with the chain rule `∏ P(bᵢ | ctx, b₀..b_{i-1})`, which is essential for correctly modeling UTF-8 continuation bytes and common bigrams.

The local decoder uses: per-position global slice projection, shift-right byte input, a 1-layer causal transformer (SwiGLU FFN 4×, RMSNorm, d_local=128, 4 heads), and teacher forcing in training / autoregressive generation at inference.

---

## Ablation Results

> All ablation runs evaluated with homogeneous criteria: `PPL = exp(val_loss)`, `BPB = val_loss / ln(2)` (byte-level only).
> Uniform byte reference: **ln(256) = 5.545 nats → PPL=256 → BPB=8.000**.
> BPE runs (run_01, run_13): PPL is **token-level**.

### 50M Runs — Round 1 (2 tok/param)

| Run | Architecture | Params M | Val Loss | PPL | BPB | Δ uniform | P/s | Bytes/s |
|-----|-------------|--------:|---------:|----:|----:|----------:|----:|--------:|
| run_01 | Transformer + BPE 16K + FP16 | 54.7 | 4.6648 | 106.14 | — | — | 78.0 | 78 |
| run_02a | Transformer + Byte + 4 heads (no LBD) | 38.5 | 2.3079 | 10.05 | 3.33 | −3.24 | 85.9 | 344 |
| run_02 | Transformer + Byte + LocalByteDecoder | 38.8 | 1.7154 | 5.56 | 2.47 | −3.83 | 86.1 | 344 |
| run_03 | MLGRU + Byte + FP16 | 38.8 | 1.8697 | 6.49 | 2.70 | −3.68 | 59.9 | 240 |
| run_04 | MLGRU + Byte + Ternary | 38.9 | 5.5671 | 261.68 | 8.03 | +0.02 | 8.0 | 32 |
| run_05 | + FPResidual | 39.0 | 5.5516 | 257.65 | 8.01 | +0.01 | 7.6 | 30 |
| run_05b | MLGRU kernel-strict (no W_o) | 35.8 | 5.5940 | 268.82 | 8.07 | +0.05 | 8.7 | 35 |
| run_06 | + Bolmo patch embedding | 39.0 | 5.5561 | 258.81 | 8.02 | +0.01 | 7.8 | 31 |
| run_07 | + DeleteGate (CPU-1 complete) ⭐ | 39.0 | 5.5561 | 258.81 | 8.02 | +0.01 | 7.8 | 31 |
| run_08 | Folded Transformer + Byte + Ternary | 38.9 | 5.5621 | 260.37 | 8.02 | +0.02 | 8.0 | 32 |
| run_09 | + PFNet | 39.4 | 5.5330 | 252.91 | 7.98 | −0.01 | 7.3 | 29 |
| run_10 | + per-channel decay | 39.4 | 5.5337 | 253.08 | 7.98 | −0.01 | 7.1 | 28 |

### 10M Runs — Round 1 (2 tok/param)

| Run | Architecture | Params M | Val Loss | PPL | BPB |
|-----|-------------|--------:|---------:|----:|----:|
| run_13 | Small CPU-1 + BPE 4K | 12.5 | 30.5375 | 1.83×10¹³ | — |
| run_14 | Small CPU-1 byte (Qwen distill) | 10.7 | 5.5755 | 263.88 | 8.04 |
| run_15 | Small CPU-1 byte + hidden distill | 10.7 | 5.5755 | 263.88 | 8.04 |
| run_16 | Small CPU-1 raw bytes, no teacher | 10.7 | 5.5754 | 263.85 | 8.04 |

### Round 2 — Re-runs at 15 tok/param

| Run | Architecture | Params M | Loss r1 | PPL r1 | Loss r2 | PPL r2 | Δ loss |
|-----|-------------|--------:|--------:|-------:|--------:|-------:|-------:|
| run_04_r2 | run_04 @ 15 tok/p | 38.9 | 5.5671 | 261.68 | 5.5894 | 267.57 | +0.022 |
| run_07_r2 | run_07 @ 15 tok/p | 39.0 | 5.5561 | 258.81 | 5.5434 | 255.54 | −0.013 |
| run_13_r2 | run_13 @ 15 tok/p | 12.5 | 30.54 | 1.83×10¹³ | 25.43 | 1.10×10¹¹ | −5.11 |
| run_14_r2 | run_14 @ 15 tok/p | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | 0.000 |
| run_15_r2 | run_15 @ 15 tok/p | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | 0.000 |
| run_16_r2 | run_16 @ 15 tok/p | 10.7 | 5.5754 | 263.85 | 5.5754 | 263.85 | 0.000 |

### Round 3 — V3 Runs (bf16 AMP + lr×2 + 50 tok/param)

> Training was interrupted by cloud credit exhaustion (AMD MI300X). Results below are from partial checkpoints.

| Run | Architecture | Params M | Val Loss | PPL | BPB | Progress (steps) |
|-----|-------------|--------:|--------:|----:|----:|-----:|
| run_01_v3 | Transformer + BPE 16K + FP16 | 54.7 | — | — | — | 12,173 (diverged NaN) |
| run_02_v3 | Transformer + Byte + LBD + FP16 | 38.8 | — | — | — | 10,963 (diverged NaN) |
| run_02a_v3 | Transformer + Byte + 4 heads + FP16 | 38.5 | — | — | — | 16,327 (diverged NaN) |
| run_03_v3 | MLGRU + Byte + FP16 | 38.8 | 3.7841 | 43.99 | 5.46 | 6,187 |
| **run_04_v3** | **MLGRU + Byte + Ternary** | **38.9** | **2.5988** | **13.45** | **3.75** | **6,074** |
| run_05_v3 | + FPResidual | 39.0 | 2.7034 | 14.93 | 3.90 | 4,390 |
| run_05b_v3 | MLGRU kernel-strict | 35.8 | 3.2144 | 24.89 | 4.64 | 2,573 |
| **run_06_v3** | **+ Bolmo patch embedding** | **39.0** | **2.1493** | **8.58** | **3.10** | **4,237** |
| run_08_v3 | Folded Transformer + Ternary | 38.9 | — | — | — | 3,858 (diverged NaN) |
| run_13_v3 | Small CPU-1 + BPE 4K | 12.5 | 5.4113 | 223.92 | — | 9,290 |
| **run_14_v3** | **Small CPU-1 byte** | **10.7** | **2.2977** | **9.95** | **3.31** | **9,775** |
| run_15_v3 | Small CPU-1 byte + hidden distill | 10.7 | 2.8020 | 16.48 | 4.04 | 2,350 |

### Key Findings

**V3 broke the ternary cold-start plateau.** In rounds 1 and 2, every ternary model collapsed to the uniform-output baseline `ln(256) ≈ 5.545 nats` — the model was effectively outputting random bytes. The V3 corrections (bf16 AMP instead of fp16, `lr_scale=2.0` for BitLinear weights, CE-only training signal, 50 tok/param budget) produced **real convergence for the first time**:

| Run | V1 Loss | V1 PPL | V3 Loss | V3 PPL | Improvement |
|-----|--------:|-------:|--------:|-------:|-------:|
| run_04 (MLGRU+Byte+Ternary) | 5.57 | 261.7 | **2.60** | **13.45** | −4.26 nats |
| run_06 (+Bolmo) | 5.56 | 258.8 | **2.15** | **8.58** | −4.93 nats |
| run_14 (Small CPU-1 10M) | 5.58 | 263.9 | **2.30** | **9.95** | −5.05 nats |

**The FP16 ablation chain works:** `run_01..run_03` converge as designed and produce clean architecture comparisons. Byte tokenization + LocalByteDecoder beats raw BPE; MLGRU is a small step back from Transformer at the same scale but provides O(d_model) recurrence with zero KV cache.

**Cold-start ternary at <50 tok/param does not work at this scale.** A weight-evolution audit showed >99.9% of BitLinear weights remain in their random-init ternary state after the full training run at 2–15 tok/param. The STE quantization step filters out gradient updates that are too small to cross the ternary boundary. This matches the literature: BitNet b1.58 succeeds at cold-start only at ≥700M params with ≥30 tok/param (industrial compute).

**1.58-bit inference speed only wins above ~700M parameters.** At 38M params, even with ideal SIMD kernels (BitNet.cpp/T-MAC class), the ternary model runs at ~0.30–0.50× the throughput of FP16 Transformer. The 8× memory advantage of ternary only translates to speedup when weight loading from RAM is the bottleneck — at 38M, weights fit in L3 cache. Above ~700M, the crossover flips and ternary wins by a factor that grows with model size.

**Practical value of 1.58-bit at small scale is memory footprint, not speed.** A 38M ternary model is ~10 MB packed vs 76 MB FP16 — valuable for deployment to memory-constrained edge devices (microcontrollers, browsers via WASM, mobile inference).

---

## Ablation Design

The ablation tells a story: starting from a standard FP16 Transformer baseline, we add **one CPU-1 design choice per run** until we reach the full architecture.

```
LINEAR CHAIN (single Δ per arrow):
run_01:   Transformer  + BPE   + FP16        ← absolute baseline (16K vocab cap)
          ↓ +byte tokenization (no LocalByteDecoder)
run_02a:  Transformer + Byte + FP16 (4 indep. heads)
          ↓ +LocalByteDecoder
run_02:   Transformer  + Byte  + FP16        ← +MegaByte intra-patch chain
          ↓ swap mixer
run_03:   MLGRU        + Byte  + FP16        ← +MatMul-free mixer
          ↓ quantize
run_04:   MLGRU        + Byte  + Ternary     ← +1.58-bit weights
          ↓ +FPResidual
run_05:   MLGRU        + Byte  + Ternary + FPRes  ← CPU-1 core ⭐
          ↓ +Bolmo (byte_dynamic patches)
run_06:   + Bolmo                            ← +boundary-aware patch embedding
          ↓ +DeleteGate (real gather)
run_07:   CPU-1 COMPLETE ⭐                  ← +MrT5-style state elimination
          ↓ +PFNet
run_09:   + PFNet (pfnet_hidden=32)          ← +cache-resident nonlinear residual
          ↓ +per-channel decay
run_10:   + decay (RWKV/HGRN2-style)         ← chain terminus

BRANCHES (off the linear chain):
run_05b_kernel_strict  ← from run_05  (no W_o — kernel-layout faithful)
run_08                 ← from run_05  (Folded Transformer + Byte + Ternary, w=128)
```

All runs: CE-only training from scratch under **ablation-scaled Chinchilla** (`tokens ≈ 2 × N_params`), same seed (42). Each run's `total_steps` is recomputed from its actual parameter count.

### 10M Scale Runs (run_13 to run_16)

Four **~10M parameter models** with the **same CPU-1 COMPLETE architecture**, varying only the training data strategy:

| Run | Training strategy | Key question |
|---|---|---|
| `run_13` | **Compact BPE** (4K vocab), from scratch | Does subword tokenization beat byte-level at 10M? |
| `run_14` | **Byte-level**, distillation from Qwen logprobs | Pure byte baseline with teacher |
| `run_15` | **Byte + hidden distillation** (EmbeddingAligner) | Does hidden-state alignment help over just logprobs? |
| `run_16` | **Raw bytes**, no teacher | Do Qwen teacher logprobs actually help? |

---

## Project Structure

```
1.58bits/
│
│  Core (training)
├── utilities.py             # Shared primitives: BitLinear, BitEmbedding, RMSNorm,
│                            #   FPResidual, PFNet, MLGRU, LocalByteDecoder,
│                            #   BoundaryPredictor, DeleteGate, EmbeddingAligner.
│                            #   Plus datasets, loss functions, LR schedules,
│                            #   checkpoint serialization (2-bit pack/unpack).
├── train_final.py           # Production CPU-1 170M training (CPU1ModelV2)
├── train_ablation.py        # Ablation training + multi-GPU orchestration
├── train_ablation_amd.py    # ROCm-adapted ablation for AMD MI300X
├── create_dataset.py        # Teacher logprob extraction + BPE→byte marginalization
├── main.py                  # Dry-run validation suite for training fixes
│
│  Inference (C / C++ only — no Python engine)
├── inference/
│   ├── benchmark.c          # Standalone driver — times global pass + LocalByteDecoder
│   ├── kernels.cpp          # AVX2 + FMA kernels: GEMV, RMSNorm, SiLU, MLGRU, PFNet,
│   │                        #   FPResidual, forward_full_mad_{st,batch,pipeline}
│   ├── benchmark.exe        # Compiled driver (regenerate with instructions in links.md)
│   └── links.md             # Build instructions + paper references
│
│  Kaggle
├── kaggle/
│   ├── cpu1-create-dataset.py        # Build + upload shared dataset (T4)
│   ├── upload_dataset.py             # Local → Hub uploader for pre-built shards
│   └── create-dataset-metadata.json  # Kaggle kernel metadata
│
│  Evaluation
├── eval/
│   ├── run_eval.py          # Unified runner — 14 checks across 5 modules
│   ├── eval_ablations.py    # Batch evaluation over all ablation models
│   ├── compute_missing_val_loss.py  # Fill in missing val_loss from checkpoints
│   ├── export_ablations.py  # Unpack compact_2bit → fp32, upload to HF Hub
│   ├── helpers.py           # Shared engine + hardware introspection
│   ├── eval_correctness.py  # forward_shapes, probs, determinism, ternary_weights
│   ├── eval_performance.py  # throughput, latency, jitter, RAM
│   ├── eval_quality.py      # generation, UTF-8 validity, perplexity
│   ├── eval_dataset.py      # teacher_coverage, teacher_entropy (no model needed)
│   └── eval_external.py     # llama-bench native benchmark on GGUFs
│
│  Models
├── models/
│   ├── cpu1-170M/               # CPU-1 170M checkpoint (in training)
│   ├── LFM2.5-350M/             # LFM2.5-350M (HF safetensors) — benchmark reference
│   ├── gpt2-medium/             # GPT-2 medium (HF safetensors) — baseline reference
│   ├── ablation_models/         # Trained compact_2bit final checkpoints per run
│   │                            #   Mirrored to Cukinator/cpu1-ablation-checkpoints
│   └── ablations_final/         # fp32-unpacked ready-to-use checkpoints per run
│                                #   Mirrored to Cukinator/cpu1-ablations-final
│
│  Results & Reference
├── resultados/              # Benchmark reports (markdown + json) per model config
├── papers/                  # Reference PDFs (see References table below)
├── hf_readmes/              # README templates for HuggingFace repos
└── Prompts/                 # Design documents per pipeline stage
```

---

## Training Pipeline

### Full Training Pipeline (170M)

```
Raw Text Corpus
      │
      ▼
┌─────────────────────────────┐
│  create_dataset.py          │  Teacher logprob extraction
│  (Qwen2.5-3B, 4-bit NF4)   │  + BPE→byte marginalization
│                             │  → Parquet with patches, targets,
│                             │    teacher_probs, teacher_mask
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  train_final.py             │  Student training with distillation
│  (AMP, gradient accum.)     │  → Checkpoints → HuggingFace Hub
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  inference/benchmark.c      │  CPU-optimized generation
│  + inference/kernels.cpp    │  T-MAC LUT + layer pipeline
└─────────────────────────────┘
```

### Ablation Training

```bash
# Single ablation run
python train_ablation.py --run run_10

# Dry run (no GPU, synthetic data)
python train_ablation.py --run run_01 --dry_run --total_steps 10

# With HF-mount (lazy dataset streaming, no full download)
python train_ablation.py --run run_10 --hf-mount

# Full AMD MI300X parallel orchestration (16 runs)
HF_STAGGER_SECS=120 python train_ablation_amd.py --orchestrate
```

Runs with `use_delete_gate=True` (run_07, run_09, run_10, and 10M scale runs) train in two phases: Phase 1 (CE pretraining, DeleteGate disabled) → Phase 2 (DeleteGate activated, elimination rate anneals to 40%). Plain `python train_ablation.py --run run_07` auto-chains both phases.

### Checkpoint Formats

| Format | When | Contents | Size (50M ternary) |
|---|---|---|---:|
| **Resume** (intermediate) | Every N steps | BF16 model + optimizer | ~214 MB |
| **Final** (inference) | End of training | 2-bit packed, no optimizer | **~9 MB** |

The final inference checkpoint uses 2-bit ternary packing: {-1, 0, 1} → {0, 1, 2}, 4 values per byte. Load with `utilities.unpack_compact_state()`.

### Dataset

All ablation runs use CE training. 50M runs stream from `HuggingFaceFW/fineweb` `sample-10BT`; 10M runs use `Cukinator/cpu1-ablation-dataset` (two subsets: `byte_marginalized/` for byte runs, `bpe_tokenized/` for BPE runs). Both paths use the same params as `create_dataset.py` (seed=42, seq_len=5000, patch_size=4).

---

## CPU Inference

Production inference uses the standalone C driver `inference/benchmark.c` linked against `inference/kernels.cpp`. The kernels are optimized for the MLGRU architecture: OpenMP, AVX2 + FMA SIMD, deep data dependency unrolling, `thread_local` scratch reuse, and a layer-pipeline mode.

### CPU-1 170M Performance — C Benchmark

> Hardware: Intel i7-8750H (6C/12T, Coffee Lake, AVX2 + FMA, no AVX-512). Model: CPU-1 170M, ~40.4 MB 2-bit packed weights.

| Mode | Description | Total pps | Per-stream pps | vs Single-thread |
|---|---|---:|---:|---:|
| 2 — single-thread (ST) | 1 thread reference | 188 | **188** | 1.00× |
| 3 — batch (6 threads) | 6 threads, batch=6 | **726** | 121 | 0.64× |
| **4 — pipeline (6 stages)** ⭐ | 6 threads, DP-calibrated layer split | **521** | **521** | **2.77×** |

> **Single-user latency**: mode 4 wins 4.3× per stream vs mode 3 (521 vs 121 pps).
> **Multi-user throughput**: mode 3 wins 1.4× total vs mode 4 (726 vs 521 pps).

Each patch produces 4 bytes: 1 global pass (14-layer backbone) + 4 sequential LocalByteDecoder steps.

### 400M Kernel-Optimal Config

The same C kernel handles a ~430M parameter scale-up via CLI flags:

| Mode | Total pps | Per-stream pps | vs Single-thread |
|---|---:|---:|---:|
| 2 — single-thread | 73 | 73 | 1.00× |
| 3 — batch (6 threads) | **270** | 45 | 0.62× |
| **4 — pipeline (6 stages)** ⭐ | **193** | **193** | **2.65×** |

```bash
# Compile:
g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math \
    -std=c++11 -x c++ -o benchmark.exe \
    inference/benchmark.c inference/kernels.cpp -lm

# Mode 4 — recommended single-stream pipeline:
./benchmark.exe --mode 4 --patches 80 --warmup 30 \
                --chunk 128 --pin 1 --stages 6 --calibrate 1 --local 1

# Mode 3 — max aggregate throughput:
./benchmark.exe --mode 3 --batch 6 --patches 100 --warmup 30 --threads 6 --local 1
```

---

## Evaluation Suite

14 checks across 5 modules. All inference goes through a single engine created once and reused across all checks. The suite is multi-architecture: it auto-detects the model type and adapts each check.

```bash
# Basic (11 checks, no dataset needed)
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt

# With dataset (adds perplexity + dataset quality checks)
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt --dataset path/

# Against a trained ablation checkpoint
python eval/run_eval.py --model models/ablations_final/run_07/model.pt
```

| Module | Checks | Description |
|---|---|---|
| `eval_correctness` | forward_shapes, probs, determinism, state_prop, ternary_weights, state_size | Model properties and correctness |
| `eval_performance` | throughput+latency+jitter, context_scaling, ram | Speed and memory benchmarks |
| `eval_quality` | generation+utf8, perplexity | Output quality evaluation |
| `eval_dataset` | teacher_coverage, teacher_entropy | Dataset quality (no model needed) |
| `eval_external` | llama_cpp | Native llama-bench benchmark |

---

## Ready-to-Use Checkpoints

All trained ablation models are unpacked from compact_2bit to FP32 and available on HuggingFace Hub:

- **Compact (2-bit packed):** [Cukinator/cpu1-ablation-checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)
- **FP32 (ready to use):** [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final)

To test locally with PyTorch:

```python
from utilities import build_ablation_model, unpack_compact_state
import torch

# Load from fp32-unpacked checkpoint
ckpt = torch.load('models/ablations_final/run_07/model.pt', map_location='cpu')
# Build model from config stored in checkpoint, then load weights
```

Or use the built-in inference mode in `train_ablation_amd.py`:

```python
TRAIN = False
INFER = True
INFER_CONFIG = {
    'ckpt':             'path/to/model.pt',
    'prompt':           'Once upon a time',
    'max_new_bytes':    256,
    'temperature':      0.8,
    'elimination_rate': 0.0,    # Use 0.4 for run_07/run_09 (DeleteGate)
    'device':           'cuda', # or 'cpu'
}
```

---

## Design Decisions

**Why byte-level?**
No tokenizer required. Processes any UTF-8 text, code, or binary data. The small vocabulary (256) makes prediction heads and embeddings negligibly small.

**Why MLGRU instead of attention?**
On CPU, attention's KV cache (80 MB for 20 layers) doesn't fit in L3 and causes constant cache misses. MLGRU's recurrent state is 28 KB total, fits in L2, and costs O(d_model) per token with Hadamard products only.

**Why PFNet?**
MLGRU's mixing `h_t = f*h + (1-f)*c` is a linear interpolation — all nonlinearity lives in the gates. PFNet adds a direct nonlinear pathway (2-layer ternary MLP with SiLU) as residual. Sized at 224 KB to fit in L2 alongside the recurrent state.

**Why fully ternary weights?**
{-1, 0, 1} weights reduce model size to ~44 MB and enable integer arithmetic (INT8 MAC or T-MAC LUT). FPResidual low-rank branches provide precision recovery.

**Why distill from BPE models?**
Strong pretrained BPE models (like Qwen2.5-3B) encode rich language knowledge. Distillation transfers this to the byte-level student more efficiently than training from scratch. First-byte marginalization solves the BPE-to-byte alignment problem.

---

## Size Comparison

Theoretical comparison if CPU-1 were scaled to 350M parameters, vs LFM2.5-350M (LiquidAI):

| Model | Format | Size | Ratio |
|---|---|---|---|
| LFM2.5-350M | BF16 (baseline) | ~700 MB | 1.0× |
| LFM2.5-350M | Q4_K_M (GGUF) | ~219 MB | 3.2× smaller |
| **CPU-1 350M** | **2-bit ternary packed** | **~92 MB** | **7.6× smaller** |

Beyond size, CPU-1 has **zero matrix multiplications at inference** (T-MAC LUT + Hadamard products only), while LFM2.5 requires standard Q4 matmul for both its convolution and GQA attention blocks.

---

## References

| Paper | What it contributes to CPU-1 |
|---|---|
| [Scalable MatMul-free LM (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) | MLGRU token mixer — Hadamard-only, no Q@K^T |
| [Slender-Mamba (COLING 2025)](https://aclanthology.org/2025.coling-main.316/) | Ternary embedding and prediction heads |
| [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) | LUT-based zero-multiply inference on CPU |
| [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) | 1.58-bit weight quantization with STE |
| [MEGABYTE (Yu et al., 2023)](papers/MEGABYTE-Predicting-byte-sequences.pdf) | Intra-patch byte autoregression (LocalByteDecoder) |
| [Bolmo (arXiv 2410.20771)](https://arxiv.org/abs/2410.20771) | Dynamic byte patches via boundary prediction |
| [MrT5](papers/mrt5-dynamic-token-merging-byte-level-lm.pdf) | Token deletion gate — prunes 40% of states |
| [Distilling Token-Trained Models into Byte-Level](papers/distilling-token-models-into-byte-level.pdf) | Embedding alignment distillation loss |
| [HGRN2 (arXiv 2404.07904)](https://arxiv.org/abs/2404.07904) | Per-channel learnable decay with hierarchical init |
| [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531) | KL + CE distillation loss |
| [LFM2 (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404) | External benchmark only (LiquidAI hybrid CNN+GQA) |

### Papers Directory (`papers/`)

| File | Paper |
|---|---|
| `scalable-matmul-free-language-modeling.pdf` | Scalable MatMul-free Language Modeling |
| `slender-mamba-fully-quantized-1.58bits.pdf` | Slender-Mamba: Fully Quantized Mamba in 1.58 Bits |
| `tmac-cpu-table-lookup-low-bit-llm.pdf` | T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM |
| `MEGABYTE-Predicting-byte-sequences.pdf` | MegaByte: Predicting Million-byte Sequences |
| `bolmo-byteifying-next-gen-language-models.pdf` | Bolmo: Byteifying the Next Generation of Language Models |
| `mrt5-dynamic-token-merging-byte-level-lm.pdf` | MrT5: Dynamic Token Merging for Byte-Level LMs |
| `distilling-token-models-into-byte-level.pdf` | Distilling Token-Trained Models into Byte-Level Models |
| `hgrn2-gated-linear-rnns-state-expansion.pdf` | HGRN2: Gated Linear RNNs with State Expansion |
| `stuffed-mamba-oversized-states-inability-to-forget.pdf` | Stuffed Mamba: Oversized States and the Inability to Forget |
| `lfm2-liquid-foundation-models-technical-report.pdf` | LFM2 Technical Report |
| `efficient-llm-inference-on-cpus.pdf` | Efficient LLM Inference on CPUs |
| `challenging-gpu-dominance-cpu-on-device-llm.pdf` | Challenging GPU Dominance: CPUs for On-Device LLM |
| `gemma3-technical-report.pdf` | Gemma 3 Technical Report |
| `tide-token-informed-depth-execution-early-exit.pdf` | TIDE: Token-Informed Depth Execution for Early Exit |

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
```
torch>=2.6.0
transformers>=4.47.0
datasets>=3.2.0
huggingface_hub>=0.27.0
bitsandbytes>=0.45.0           # 4-bit NF4 for Qwen2.5-3B teacher
llama-cpp-python==0.3.19       # CPU-AVX2 wheel for eval/eval_external.py
```

For AMD MI300X (ROCm):
```bash
pip install -r requirements_amd.txt
```

---

## Status

### Completed

- ✅ Architecture — 170M model with MLGRU + PFNet + ternary + MegaByte LocalByteDecoder
- ✅ Extended techniques (v3) — BoundaryPredictor, DeleteGate, EmbeddingAligner, two-phase training
- ✅ Dataset pipeline — Teacher logprob extraction + BPE→byte marginalization
- ✅ Ablation system — 12 50M runs + 4 10M runs, three rounds (r1/r2/r3)
- ✅ C inference kernels — 6-stage pipeline (mode 4: 521 pps) and 6-stream batch (mode 3: 726 pps)
- ✅ 400M kernel-optimal config — 193 pps (mode 4), 270 pps (mode 3)
- ✅ Evaluation suite — 14 checks in 5 modules, multi-architecture adaptive
- ✅ V3 ternary convergence fix — bf16 AMP + lr_scale=2.0 breaks the cold-start plateau

### In Progress

- 🔄 170M production model training (`models/cpu1-170M/`)
- 🔄 V3 training completion (interrupted by cloud credit exhaustion)

### Path Forward

The V3 corrections demonstrated that **warm-start is not the only path**: cold-start ternary at 50 tok/param with bf16 AMP and `lr_scale=2.0` produces real convergence. The best next step is completing the interrupted V3 training runs, particularly run_06_v3 (val_loss 2.15, PPL 8.58 at only 6.8% progress) which is the strongest ternary result.

---

## License

Apache-2.0
