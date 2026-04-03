# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen2.5-3B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

---

## Architecture (v2 — 170M)

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| `n_layers` | 14 |
| `d_ff` | 2730 (SwiGLU, ~8/3 × d_model) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| `pfnet_hidden` | 64 (PFNet hidden dim, 224 KB total across all layers) |
| `mlgru_state` | 28 KB (14 × 1024 × 2 bytes — fixed, fits in L2) |
| **Total parameters** | **~170M** |

### Key Components

- **BitEmbedding** — Ternary embedding table {-1, 0, 1} with per-row FP16 scale (Slender-Mamba, COLING 2025).
- **BytePatchEmbeddingV2** — Converts 4-byte patches to `d_model` dims: BitEmbedding (8 dims/byte) → concat (32 dims) → BitLinear → d_model.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE).
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with BitLinear, initialized to zero.
- **PFNet** — 2-layer ternary MLP residual per block (hidden=64). All PFNet weights across 14 layers (224 KB) fit in L2 alongside the MLGRU state (28 KB), staying cache-resident while main weights stream from RAM.
- **MatMulFreeTokenMixer (MLGRU)** — GRU-style recurrence with Hadamard products only, no Q@K^T (arXiv 2406.02528). O(d_model) per token, zero KV cache.
- **SwiGLU FFN** — Gated feed-forward with fully ternary weights.
- **BitHead** — 4 ternary prediction heads (one per byte in next patch).
- **RMSNorm** — Pre-norm before token mixer and FFN.

### Memory at Inference

| Component | v1 (Attention) | v2 (MLGRU + PFNet) |
|---|---|---|
| Token mixer state | 80 MB (KV cache) | **28 KB** (14 × 1024 × 2B) — L2 |
| PFNet weights | — | **224 KB** (all layers) — L2 |
| Active layer weights | — | ~3.1 MB (1 layer) — L3 |
| Full model weights | — | ~44 MB — RAM streaming |

### Extended Architecture (v3)

CPU-1 v3 extends the base architecture with three optional modules:

- **BoundaryPredictor** (Bolmo-inspired): Replaces fixed 4-byte patches with dynamic patches that align with natural language boundaries. Adds ~2KB of parameters, runs in Thread 2 during CPU inference with zero added latency.

- **DeleteGate** (MrT5-inspired): Placed after layer N//2, learns to discard uninformative states before the remaining layers. Reduces active states by up to 40%, activated only in training Phase 2.

- **EmbeddingAligner** (arXiv 2602): Aligns the student's hidden space with the teacher's embedding space via cosine similarity loss. Trained in Phase 1 only, frozen in Phase 2.

All three modules are optional and controlled by config flags:
```
use_boundary_predictor: false  # set true for dynamic patches
use_delete_gate: false         # set true for MrT5-style compression
use_embedding_alignment: false # set true for embedding distillation
```

### References

- [Slender-Mamba (COLING 2025)](https://aclanthology.org/2025.coling-main.316/) — Ternary embeddings and prediction heads.
- [Scalable MatMul-free Language Modeling (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) — MLGRU token mixer.
- [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) — LUT-based inference for low-bit LLMs.
- [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) — 1.58-bit weight quantization.
- [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531) — KL + CE distillation loss, T=4.0.
- [Bolmo (Allen AI, 2025)](https://arxiv.org/abs/2410.20771) — Dynamic byte-level patches with boundary prediction.
- [MrT5 (Google, 2024)](https://arxiv.org/abs/2512.15586) — Token deletion gate for efficient inference.
- [Distilling Byte-Level Models (arXiv 2602, 2025)](https://arxiv.org/abs/2602.01007) — Embedding alignment distillation.

---

## Ablation Study — 15 Runs, Shared Dataset

The ablation validates each design decision independently at 50M parameters, using a single shared dataset (teacher runs once).

### Variables Under Study

| Variable | Baseline | Proposed |
|---|---|---|
| Token mixer | Transformer (SlidingWindowAttention) | MLGRU (MatMul-free) |
| Tokenization | BPE (Qwen, ~152K vocab) | Byte-level (patches of 4) |
| Weight quantization | FP16 | Ternary 1.58-bit (BitLinear) |
| Training signal | CE only | KD: BPE→byte distillation |
| Patch type | Fixed 4-byte | Dynamic (Bolmo) |
| Embedding distillation | None | Cosine alignment (arXiv 2602) |
| State compression | None | DeleteGate (MrT5) |

### Experimental Design (15 runs)

```
Run 01: Transformer + BPE  + FP16    + CE     ← absolute baseline
Run 02: Transformer + Byte + FP16    + CE     ← isolate byte-level
Run 03: Transformer + BPE  + Ternary + CE     ← isolate quantization
Run 04: Transformer + BPE  + Ternary + Distil ← transformer complete
Run 05: MLGRU       + BPE  + FP16    + CE     ← isolate MLGRU
Run 06: MLGRU       + BPE  + Ternary + CE     ← MLGRU + ternary
Run 07: MLGRU       + BPE  + Ternary + Distil ← MLGRU BPE complete
Run 08: MLGRU       + Byte + FP16    + CE     ← isolate byte on MLGRU
Run 09: MLGRU       + Byte + Ternary + CE     ← without distillation
Run 10: MLGRU       + Byte + Ternary + Distil + FPResidual  ← PROPOSED (CPU-1)
Run 11: MLGRU       + Byte + Ternary + Distil − FPResidual  ← 100% ternary pure
Run 12: Run 10 + BoundaryPredictor (Bolmo)    ← dynamic patches
Run 13: Run 10 + EmbeddingAligner (arXiv 2602) ← embedding distillation
Run 14: Run 10 + DeleteGate (MrT5, 2 phases)  ← state compression
Run 15: Run 10 + all three                    ← PROPOSED v3 (CPU-1 complete)
```

Run 10 vs 11 answers: is the FP16 correction branch worth it?
Run 10 vs 12: effect of dynamic patches (Bolmo).
Run 10 vs 13: effect of embedding alignment.
Run 10 vs 14: effect of DeleteGate (CPU speed).
Run 10 vs 15: combined effect (v3 proposal).

### Justified with Citations (no runs needed)

| Decision | Citation |
|---|---|
| BitEmbedding / BitHead | Slender-Mamba (Yu et al., COLING 2025) |
| LUT vs INT8 matmul | T-MAC (Wei et al., EuroSys 2025) |
| FPResidual parallel branch | MatMul-free LM (Zhu et al., 2024) |
| SwiGLU | PaLM (Chowdhery et al., 2022) |
| RMSNorm | LLaMA (Touvron et al., 2023) |
| KL + CE loss, T=4.0 | Hinton et al., 2015 |

---

## Pipeline

### Full Training Pipeline (170M model)

```
Raw Text Corpus
      │
      ▼
┌─────────────────────────────┐
│  create_dataset.py          │  Teacher logprob extraction
│  (Transformers or Ollama    │  + BPE→byte marginalization
│   backend, 4-bit NF4)       │  → Parquet with patches, targets,
│                             │    teacher_probs, teacher_mask
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  train.py                   │  Student training with distillation
│  (AMP, gradient             │  → Checkpoints → HuggingFace Hub
│   accumulation)             │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  inference.py               │  CPU-optimized generation
│                             │  INT8 packing  or  T-MAC LUT
└─────────────────────────────┘
```

### Ablation Pipeline (50M × 15 runs on Kaggle T4)

#### 1. Build Shared Dataset (once)

```bash
# Full pipeline: extract teacher logprobs + build byte and BPE datasets
python create_dataset.py --n_docs 50000

# Build only byte-marginalization from existing logprobs
python create_dataset.py --logprobs-dir ./logprobs --output-dir ./dataset
```

Teacher (Qwen2.5-3B in 4-bit NF4 or FP8) runs **once**. Produces:
- `byte_marginalized/` — patches, targets, teacher_probs, teacher_mask, teacher_hidden
- `bpe_tokenized/` — input_ids, labels, teacher_probs_bpe, teacher_mask_bpe
- Uploaded to `Cukinator/cpu1-ablation-dataset` on HuggingFace Hub

**Why not use existing distillation datasets?** We evaluated public teacher-logit datasets ([arcee-ai/LLama-405B-Logits](https://huggingface.co/datasets/arcee-ai/LLama-405B-Logits), [hbfreed/Dolci-Instruct-RL-Completions](https://huggingface.co/datasets/hbfreed/Dolci-Instruct-RL-Completions), [eyad-silx/Base-logits](https://huggingface.co/datasets/eyad-silx/Base-logits), among others). None are suitable because: (1) all contain instruction/math/code data, not general web text needed for pretraining ablations; (2) none include intermediate hidden states required by the EmbeddingAligner (runs 13, 15); (3) none provide byte-level marginalized probabilities — our model operates at byte level, requiring BPE→byte probability conversion that must be computed from the raw teacher logprobs.

#### 2. Train Ablation Runs

```bash
# Single ablation run (GPU)
python train.py --mode ablation --run run_10

# Dry run (no GPU, synthetic data)
python train.py --mode ablation --run run_01 --dry_run --total_steps 10

# With HF-mount (lazy dataset streaming, no full download)
python train.py --mode ablation --run run_10 --hf-mount
```

#### Two-Phase Training (runs 14–15 only)

For runs using DeleteGate (MrT5), training proceeds in two phases:

- **Phase 1 (Distillation):** Standard knowledge distillation from Qwen2.5-3B. DeleteGate disabled (elimination_rate=0). EmbeddingAligner active.
  `python train.py --mode ablation --run run_14 --phase 1`

- **Phase 2 (Compression):** DeleteGate activated. EmbeddingAligner frozen. Loss changes from KL+CE to CE-only.
  `python train.py --mode ablation --run run_14 --phase 2 --phase1_ckpt <path>`

#### 3. Analyze Results

```bash
python ablation.py --metrics_dir .
```

Generates a text table for terminal and a LaTeX table for the paper (`ablation_table.tex`).

---

### Kaggle Workflow

The `kaggle/` directory contains scripts ready to run directly in Kaggle notebooks (T4 GPU):

| Script | Purpose |
|---|---|
| `kaggle/cpu1-create-dataset.py` | Runs Qwen2.5-3B teacher (Batch 5 aligned for 4GB VRAM free), builds and uploads the shared dataset to HF Hub |
| `kaggle/cpu1-training.py` | Trains a specific ablation run; configure `RUN_NAME` and `PHASE` at the top |

Both scripts read `HF_TOKEN` from Kaggle Secrets automatically. To push an updated script:

```bash
kaggle kernels push -p kaggle/  # uses kernel-metadata.json / training-metadata.json
```

---

## Inference

Two engines, both in `inference.py`:

```python
# INT8 — ternary weights as quantized float32 + torch.compile
engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
output = engine.generate(b"Buenos dias", max_bytes=200)

# T-MAC LUT — zero multiplications: precomputed lookup table + gather
engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
    'checkpoint.pt', config, group_size=4
)
output = engine.generate(b"Buenos dias", max_bytes=200)
```

---

## Project Structure

```
1.58bits/
│
│  Core
├── model.py              # Full CPU-1 architecture: MLGRU + PFNet + FPResidual +
│                         #   BoundaryPredictor + DeleteGate + EmbeddingAligner
├── train.py              # Training loop — two modes:
│                         #   default: 170M production model
│                         #   --mode ablation --run run_XX: ablation run
├── ablation.py           # Ablation system: BASE_CONFIG + 15 RUN_CONFIGS,
│                         #   configurable model variants, analysis + LaTeX tables
├── create_dataset.py     # Full dataset pipeline: teacher logprob extraction,
│                         #   BPE→byte marginalization, HuggingFace Hub upload
├── inference.py          # CPU inference: INT8 packing + T-MAC LUT engine
├── hf_mount.py           # Lazy HF filesystem mount via NFS — fetches dataset
│                         #   on-demand without downloading the full ~5 GB
│
│  Kaggle
├── kaggle/
│   ├── cpu1-create-dataset.py   # Kaggle script: build + upload shared dataset (T4)
│   ├── cpu1-training.py         # Kaggle script: train ablation run (T4)
│   ├── create-dataset-metadata.json
│   ├── kernel-metadata.json
│   └── training-metadata.json
│
│  Evaluation
├── eval/
│   ├── run_all_tests.py         # Runs all 21 tests, saves report to resultados/
│   ├── verify_all.py            # Quick correctness check
│   ├── eval_cpu.py              # CPU performance benchmark
│   ├── benchmark_external.py    # External model comparison
│   ├── bpe_vs_bytebatch_benchmark.py
│   ├── helpers.py
│   ├── test_01_forward.py … test_21_mlgru_state_size.py
│   └── test_26_benchmark_llama_cpp.py  # llama-bench native benchmark
│
│  llama.cpp
├── llamacpp/                # llama.cpp source + builds
│   ├── build_optimized/     #   GCC + AVX2 + FMA + Ninja (-march=native -O3)
│   └── build_ninja/         #   Previous Ninja build
│
│  Models
├── models/
│   ├── cpu1-170M/              # CPU-1 170M (ternary, byte-level, MLGRU) — in training
│   ├── LFM2-350M-GGUF/        # LFM2-350M Q4_K_M (219 MB) — mlGRU architecture
│   └── distilgpt2-GGUF/       # distilgpt2 Q4_K_M (60 MB) — transformer baseline
│
│  Results & Reference
├── checkpoints/          # Training checkpoints
├── resultados/           # Benchmark reports (markdown) per model
├── papers/               # Reference PDFs
└── Prompts/              # Design documents per pipeline stage
```

---

## Evaluation Suite

### Why this architecture

The original eval suite had **26 separate test files** (`test_01_forward.py` through `test_26_benchmark_llama_cpp.py`), each reimplementing its own inference loop — loading the model raw, calling `model.forward()` directly, and bypassing all the optimizations in `inference.py` (INT8 packing, torch.compile, weight transposition, zero-alloc generation). The benchmarks measured the unoptimized model, not what actually runs in production.

Additionally:
- The model was loaded up to **26 times** when running the full suite
- Tests 06, 07, 08 each ran separate benchmark loops (200-500 tokens) to measure throughput, latency, and jitter — **three runs for what's a single measurement**
- Tests 12, 13 generated text independently (2 generation calls for quality + UTF-8 checks)
- Tests 14-16, 19-20 were ablation comparisons that don't belong in an inference eval suite
- Tests 22 reimplemented `_freeze_bitlinear_for_inference()` — code that already exists in `inference.py`
- **None of the tests worked with `AblationModel`** — they were hardcoded for `CPU1ModelV2`

### Current design

14 checks across 5 modules. All inference goes through `CPU1InferenceEngine` from `inference.py` — zero duplicated inference code. The engine is created **once** and reused across all checks.

The suite is **multi-architecture**: it auto-detects the model type via `ModelInfo` and adapts each check. Ternary weight checks are skipped for FP16 models, state propagation is skipped for Transformers, generation quality is skipped for BPE models, etc.

```bash
# Basic (11 checks, no dataset needed)
python eval/run_eval.py --model checkpoints/run_10.pt

# With dataset (adds perplexity + dataset quality checks)
python eval/run_eval.py --model checkpoints/run_10.pt --dataset path/

# Skip external benchmarks (llama.cpp)
python eval/run_eval.py --model checkpoints/run_10.pt --skip-external
```

| Module | Checks | Description |
|---|---|---|
| `eval_correctness` | forward_shapes, probs, determinism, state_prop, ternary_weights, state_size | Model properties and correctness |
| `eval_performance` | throughput+latency+jitter (fused), context_scaling, ram | Speed and memory benchmarks |
| `eval_quality` | generation+utf8 (fused), perplexity | Output quality evaluation |
| `eval_dataset` | teacher_coverage, teacher_entropy | Dataset quality (no model needed) |
| `eval_external` | llama_cpp | Native llama-bench benchmark |

### Supported architectures

| Architecture | Runs | Adapted checks |
|---|---|---|
| MLGRU + Byte + Ternary | 09, 10, 11, 12-15, CPU-1 170M | All 14 checks |
| MLGRU + Byte + FP16 | 08 | Skip ternary_weights |
| MLGRU + BPE + Ternary | 06, 07 | Skip generation/utf8 (BPE) |
| MLGRU + BPE + FP16 | 05 | Skip ternary + generation |
| Transformer + BPE + Ternary | 03, 04 | Skip state_prop, state_size, generation |
| Transformer + BPE + FP16 | 01 | Skip ternary + states + generation |
| Transformer + Byte + FP16 | 02 | Skip ternary + states |

---

## llama.cpp Optimization Analysis — LFM2-350M on i7-8750H

Systematic optimization and benchmarking of the LFM2-350M model running via llama.cpp on consumer CPU hardware (Intel i7-8750H, 6C/12T, Coffee Lake).

### Build Optimization

The llama.cpp project was compiled from source with full CPU optimization:

```
cmake -G Ninja -B build_optimized \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="-march=native -O3" \
  -DCMAKE_CXX_FLAGS="-march=native -O3" \
  -DGGML_NATIVE=ON
```

Verified via `CMakeCache.txt`: `-march=native` activates **AVX2 + FMA** on this CPU (Coffee Lake does **not** support AVX-512). Build completed successfully (272/272 targets).

### Runtime Configurations Tested

| Config | Threads | mmap | flash-attn | KV cache | Priority | Batch |
|---|---|---|---|---|---|---|
| ULTRA | 1 | OFF | ON | q8_0 | REALTIME | 2048 |
| OPTIMAL | 1 | OFF | OFF | q8_0 | HIGH | 2048 |
| BASELINE | 1 | default | OFF | f16 | default | default |

Methodology: 900 tokens prefill + 100 tokens decode × 3 repetitions (matching [LiquidAI's official benchmark](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)).

### Benchmark Results

| Model | Quant | Decode (tok/s) | Config | Notes |
|---|---|---|---|---|
| **distilgpt2 (82M)** | Q4_K_M | **141.7** | ULTRA | Transformer — parallelizable |
| **LFM2-350M** | Q4_K_M | **~41** | ULTRA | mlGRU — sequential decode |

### Comparison vs Official Benchmarks

| Device | Inference | Framework | Prefill | Decode | Bandwidth |
|---|---|---|---|---|---|
| AMD Ryzen AI Max 395+ | CPU | llama.cpp (Q4) | 2,900 | 313 | ~128 GB/s |
| Apple M5 Max | GPU | Mirai (bf16) | 44,800 | 564 | ~546 GB/s |
| Apple M1 Max | GPU | Mirai (bf16) | 10,400 | 328 | ~200 GB/s |
| Apple A18 Pro | GPU | Mirai (bf16) | 1,953 | 73 | ~75 GB/s |
| Raspberry Pi5 | CPU | Cactus (int8) | 200 | 30 | ~8 GB/s |
| **i7-8750H (this repo)** | **CPU** | **llama.cpp (Q4)** | **—** | **~41** | **~38 GB/s** |

### Root Cause Analysis

Two hardware/architecture bottlenecks limit decode throughput:

**1. mlGRU sequential dependency.** The LFM2 architecture uses mlGRU (MatMul-free GRU) for token mixing. Each decode step depends on the hidden state from the previous step — this is inherently sequential and cannot be parallelized. By contrast, transformer models (like distilgpt2) with KV-cache allow parallel computation within each decode step. This architectural difference explains the **3.6× gap** between distilgpt2 (141.7 tok/s) and LFM2-350M (~41 tok/s) despite LFM2 being only 2.7× larger.

**2. Memory bandwidth ceiling.** Decode throughput is memory-bandwidth-bound (streaming weights from RAM each token). This system has **mismatched DIMMs** (2667 MHz + 2400 MHz), forcing the memory controller to operate at the slower speed with potential additional timing penalties. Theoretical peak: ~38.4 GB/s. The AMD Ryzen AI Max 395+ has ~128 GB/s (DDR5) — a **3.4× advantage** in raw bandwidth, compounded by AVX-512 (2× SIMD width) and superior Zen5 IPC.

**Why do Apple chips achieve 300+ tok/s?** They run on **GPU** (Metal) with **Mirai** framework using **bf16** weights — not llama.cpp on CPU. The M5 Max has ~546 GB/s bandwidth (14× more than this system). The sequential nature of mlGRU remains, but each sequential step executes in microseconds instead of milliseconds due to the massive bandwidth advantage.

### Key Finding

> The build_optimized and previous build_ninja builds already had `-march=native` active — **AVX2 was never missing**. The performance ceiling is imposed by the mlGRU architecture and memory bandwidth, not by compiler flags or runtime configuration.

### Paths to 100+ tok/s

| Strategy | Expected Impact | Effort |
|---|---|---|
| **Upgrade to LFM2.5-350M** | ~2× faster decode (hybrid arch, less sequential) | Download new GGUF |
| Re-quantize to Q4_0 | +5–15% (simpler kernel, less compute overhead) | Run `llama-quantize` |
| Use llamafile | +10–30% prefill, marginal decode gain (tinyBLAS kernels) | Download binary |
| Match RAM DIMMs (2×2667) | +10% bandwidth (eliminate timing penalty) | Hardware upgrade |
| **All combined** | **~100–140 tok/s estimate** | — |

For this project's CPU-1 model (fully ternary MLGRU), the bottleneck analysis confirms the architecture choice is sound for the target use case: streaming generation on CPU with minimal memory footprint (28 KB recurrent state vs 80 MB KV cache). The tradeoff is lower peak tok/s compared to transformer architectures, compensated by constant-memory inference regardless of context length.

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
Strong pretrained BPE models (like Qwen2.5-3B) encode rich language knowledge. Distillation transfers this to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch. First-byte marginalization solves the BPE-to-byte alignment problem.

**Why hf_mount?**
The ablation dataset is ~5 GB. Downloading it in full on each Kaggle session wastes time and disk quota. `hf_mount` uses NFS to mount the HF bucket and fetches only the bytes actually read — compatible with Kaggle's Linux environment, no root required.

---

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
```
torch==2.6.0+cu124          # CUDA 12.4 build (compatible with 12.6 driver)
transformers==5.3.0
datasets==4.8.4
huggingface_hub==1.8.0
bitsandbytes==0.49.2        # 4-bit NF4 for teacher model
pyarrow==23.0.1
numpy==2.4.3
tqdm==4.67.3
```

---

## Status

### Implemented

- **Architecture** — 170M model: d_model=1024, 14 layers, MLGRU + PFNet + fully ternary (`model.py`)
- **Extended techniques (v3)** — BoundaryPredictor (Bolmo), DeleteGate (MrT5), EmbeddingAligner (arXiv 2602), two-phase training
- **Dataset pipeline** — Teacher logprob extraction + BPE→byte marginalization + HF Hub upload (`create_dataset.py`)
- **HF-mount** — Lazy NFS-based dataset access, no full download needed (`hf_mount.py`)
- **Training** — AMP, gradient accumulation, distillation loss, both main and ablation modes in one script (`train.py`)
- **Ablation system** — 15-run experimental design, configurable variants, shared dataset, LaTeX table generation (`ablation.py`)
- **CPU inference** — Multi-architecture engine with auto-detection (`ModelInfo`), INT8 packing, torch.compile, T-MAC LUT (`inference.py`)
- **Evaluation suite** — 14 checks in 5 modules, multi-architecture adaptive, single engine load, fused benchmarks (`eval/`)
- **Kaggle scripts** — Ready-to-run notebooks for dataset creation and training on T4 (`kaggle/`)

### Models

| Model | Size | Architecture | Status |
|---|---|---|---|
| CPU-1 170M | ~170M params (~44 MB ternary) | MLGRU + Byte + Ternary + FPResidual + PFNet | 🔄 Training |
| LFM2-350M (GGUF) | 219 MB (Q4_K_M) | mlGRU (LiquidAI) | ✅ Benchmark reference |
| distilgpt2 (GGUF) | 60 MB (Q4_K_M) | Transformer | ✅ Baseline reference |

### Pending

- Run the 15 ablation experiments on Kaggle T4 and collect metrics
- Scale winning ablation config to 170M and train on full dataset
- T-MAC C++ kernel integration (bitnet.cpp / llama.cpp)

---

## License

MIT
