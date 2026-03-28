# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen-7B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

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

Teacher (Qwen-7B in 4-bit NF4) runs **once**. Produces:
- `byte_marginalized/` — patches, targets, teacher_probs, teacher_mask
- `bpe_tokenized/` — input_ids, labels, teacher_probs_bpe, teacher_mask_bpe
- Uploaded to `Cukinator/cpu1-ablation-dataset` on HuggingFace Hub

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

- **Phase 1 (Distillation):** Standard knowledge distillation from Qwen-7B. DeleteGate disabled (elimination_rate=0). EmbeddingAligner active.
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
| `kaggle/cpu1-create-dataset.py` | Runs Qwen-7B teacher, builds and uploads the shared dataset to HF Hub |
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
│   └── test_01_forward.py … test_21_mlgru_state_size.py
│
│  Results & Reference
├── models/distilgpt2/    # Local distilgpt2 (82M) used as eval baseline
├── checkpoints/          # Training checkpoints
├── resultados/           # Benchmark reports (markdown) per model
├── papers/               # Reference PDFs
└── Prompts/              # Design documents per pipeline stage
```

---

## Evaluation Suite

21 test files across 6 categories (16 run automatically, 5 require manual invocation). Run with:

```bash
python eval/run_all_tests.py --model models/distilgpt2
```

| Category | Tests |
|---|---|
| Correctness | 01 forward, 02 probs, 03 determinism, 04 state propagation, 05 ternary weights |
| CPU speed | 06 throughput, 07 latency, 08 jitter, 09 context scaling, 10 RAM |
| Quality | 11 perplexity, 12 generation, 13 UTF-8 validity |
| Training | 14 convergence |
| Ablation | 15 FPResidual*, 16 MLGRU vs Transformer*, 17 teacher coverage, 18 teacher entropy |
| Full | 19 benchmark*, 20 scaling*, 21 MLGRU state / KV cache |

\* Run manually (not included in `run_all_tests.py`).

**Baseline (distilgpt2, 82M FP32, local CPU):**

| Metric | Value |
|---|---|
| Throughput | 43.8 tok/s |
| Latency p50 | 22.3 ms |
| Perplexity | 65.4 |
| RAM delta | 11.5 MB |
| UTF-8 valid | 100% |

Full report: [`resultados/distilgpt2.md`](resultados/distilgpt2.md)

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
Strong pretrained BPE models (Qwen-7B) encode rich language knowledge. Distillation transfers this to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch. First-byte marginalization solves the BPE-to-byte alignment problem.

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
bitsandbytes==0.49.2        # 4-bit NF4 for Qwen-7B teacher
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
- **CPU inference** — INT8 packing with `torch.compile` + T-MAC LUT zero-multiplication engine (`inference.py`)
- **Evaluation suite** — 21 tests: correctness, speed, quality, training, ablation metrics (`eval/`)
- **Kaggle scripts** — Ready-to-run notebooks for dataset creation and training on T4 (`kaggle/`)

### Pending

- Run the 15 ablation experiments on Kaggle T4 and collect metrics
- Scale winning ablation config to 170M and train on full dataset
- T-MAC C++ kernel integration (bitnet.cpp / llama.cpp)

---

## License

MIT
