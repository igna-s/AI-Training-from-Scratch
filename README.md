# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen-14B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

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

- **BoundaryPredictor** (Bolmo-inspired): Replaces fixed 4-byte patches with
  dynamic patches that align with natural language boundaries. Adds ~2KB of
  parameters, runs in Thread 2 during CPU inference with zero added latency.

- **DeleteGate** (MrT5-inspired): Placed after layer N//2, learns to discard
  uninformative states before the remaining layers. Reduces active states by
  up to 40%, activated only in training Phase 2.

- **EmbeddingAligner** (arXiv 2602): Aligns the student's hidden space with
  the teacher's embedding space via cosine similarity loss. Trained in Phase 1
  only, frozen in Phase 2.

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
│  extract_logprobs.py        │  Teacher logprob extraction
│  (Transformers or Ollama    │  → Parquet with top-k logprobs per position
│   backend, 4-bit NF4)       │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  byte_marginalization.py    │  BPE → Byte alignment
│                             │  → Dataset of (patches, targets,
│                             │    teacher_probs, teacher_mask)
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  train.py                   │  Student training with distillation
│  (DDP, AMP, gradient        │  → Checkpoints
│   accumulation)             │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  inference_v2.py            │  CPU-optimized generation
│  inference_tmac.py          │  → INT8 packing / T-MAC LUT
└─────────────────────────────┘
```

### Ablation Pipeline (50M × 11 runs)

#### 1. Build Shared Dataset (once)

```bash
# Full pipeline: extract teacher logprobs + build both BPE and byte datasets
python build_shared_dataset.py --n_docs 50000

# Resume after interruption
python build_shared_dataset.py --n_docs 50000

# Upload existing dataset
python build_shared_dataset.py --only_upload
```

Teacher (Qwen-14B in 4-bit NF4) runs **once**. Produces:
- `byte_marginalized/` — patches, targets, teacher_probs, teacher_mask
- `bpe_tokenized/` — input_ids, labels, teacher_probs_bpe, teacher_mask_bpe
- Uploaded to `Cukinator/cpu1-ablation-dataset`

#### 2. Train Ablation Runs

```bash
# Single GPU
python ablation_train.py --run run_10

# Kaggle 2×T4
torchrun --nproc_per_node=2 ablation_train.py --run run_10

# Dry run (no GPU, random data)
python ablation_train.py --run run_01 --dry_run --total_steps 10
```

#### Two-Phase Training (runs 14-15 only)

For runs using DeleteGate (MrT5), training proceeds in two phases:

- **Phase 1 (Distillation):** Standard knowledge distillation from Qwen-14B.
  DeleteGate disabled (elimination_rate=0). EmbeddingAligner active.
  Command: `torchrun ... --run run_14 --phase 1`

- **Phase 2 (Compression):** DeleteGate activated. EmbeddingAligner frozen.
  Loss changes from KL+CE to CE-only (sequence length varies with gate).
  Command: `torchrun ... --run run_14 --phase 2 --phase1_ckpt <path>`

Suggested order to maximize early signal:
```bash
# Day 1: extremes
ablation_train.py --run run_01   # baseline
ablation_train.py --run run_10   # proposed

# Day 2: complete 2×2 grid
ablation_train.py --run run_02   # Transformer + Byte
ablation_train.py --run run_05   # MLGRU + BPE
ablation_train.py --run run_08   # MLGRU + Byte FP16

# Day 3-4: quantization + distillation
ablation_train.py --run run_03
ablation_train.py --run run_06
ablation_train.py --run run_09

# Day 5: complete table
ablation_train.py --run run_04
ablation_train.py --run run_07

# Day 6: FPResidual ablation
ablation_train.py --run run_11

# Day 7-8: Extended techniques (requires runs 01-11 working)
ablation_train.py --run run_13                    # EmbAlign (1 phase)
ablation_train.py --run run_12                    # Bolmo (1 phase)
ablation_train.py --run run_14 --phase 1          # DeleteGate Phase 1
ablation_train.py --run run_14 --phase 2 --phase1_ckpt <path>  # Phase 2

# Day 9-10: Full v3 (2 phases)
ablation_train.py --run run_15 --phase 1
ablation_train.py --run run_15 --phase 2 --phase1_ckpt <path>
```

#### 3. Analyze Results

```bash
python ablation_analysis.py --metrics_dir .
```

Generates:
- Text table for terminal
- LaTeX table for paper (`ablation_table.tex`)
- Key comparisons (MLGRU effect, byte effect, quantization effect, etc.)

---

## Project Structure

```
1.58bits/
│
│  Dataset
├── build_shared_dataset.py   # Genera el dataset compartido: corre el teacher (Qwen-14B),
│                             #   extrae logprobs + hidden states, y lo sube a HuggingFace
├── extract_logprobs.py       # Extrae logprobs del teacher para distillación (BPE→byte mapping)
├── byte_marginalization.py   # Convierte probabilidades BPE del teacher a probabilidades
│                             #   a nivel de byte (marginalización). Soporta parches dinámicos (Bolmo)
│
│  Modelo principal (170M params)
├── model_v2.py               # Arquitectura completa CPU-1: MLGRU + byte-level + ternary
│                             #   + PFNet + FPResidual + BoundaryPredictor + DeleteGate
│                             #   + EmbeddingAligner. El corazón del proyecto
├── train.py                  # Loop de entrenamiento del modelo principal de 170M
│
│  Inferencia
├── inference_v2.py           # Inferencia estándar del modelo v2 (generación, sampling)
├── inference_tmac.py         # Inferencia optimizada con T-MAC: usa LUT (lookup tables)
│                             #   en vez de multiplicaciones para pesos ternarios
├── eval_cpu.py               # Benchmarks rápidos de rendimiento en CPU
│
│  Ablación (15 runs × 50M params)
├── ablation_config.py        # BASE_CONFIG + RUN_CONFIGS para los 15 runs
├── ablation_models.py        # Variantes configurables (Transformer/MLGRU × BPE/Byte)
├── ablation_train.py         # Loop de entrenamiento unificado para cualquier run
├── ablation_analysis.py      # Resultados → tablas texto + LaTeX para el paper
│
│  Evaluación y referencia
├── eval/                     # Suite de 21 tests (correctitud, velocidad, calidad)
├── resultados/               # Reportes de benchmark en markdown
├── papers/                   # Papers de referencia
└── Prompts/                  # System prompts (design docs)
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
Strong pretrained BPE models (Qwen-14B) encode rich language knowledge. Distillation transfers this to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch. First-byte marginalization solves the BPE-to-byte alignment problem.

---

## Dependencies

```bash
pip install torch
pip install transformers datasets huggingface_hub
pip install bitsandbytes          # 4-bit NF4 for teacher
pip install pyarrow numpy
```

---

## Status

### Implemented

- **Architecture** — 170M model: d_model=1024, 14 layers, MLGRU + PFNet + fully ternary (`model_v2.py`)
- **Data pipeline** — Teacher logprob extraction + BPE→byte marginalization (`extract_logprobs.py`, `byte_marginalization.py`)
- **Training** — DDP, AMP, gradient accumulation, distillation loss with temperature scaling (`train.py`)
- **CPU inference** — INT8 packing with `torch._int_mm`, weight transposition, `torch.compile` (`inference_v2.py`)
- **T-MAC LUT inference** — Zero-multiplication engine: ternary weights → precomputed LUT, lookup + accumulate only (`inference_tmac.py`)
- **Ablation system** — 15-run experimental design, configurable models (Transformer/MLGRU × BPE/Byte × FP16/Ternary × CE/Distil), unified training loop, shared dataset pipeline, LaTeX table generation
- **Extended techniques (v3)** — BoundaryPredictor (Bolmo), DeleteGate (MrT5), EmbeddingAligner (arXiv 2602), two-phase training support

### Pending

- Run 15 ablation experiments on Kaggle 2×T4 and collect metrics
- Scale winning ablation config to 170M and train on full dataset
- T-MAC C++ kernel integration (bitnet.cpp / llama.cpp)

---

## License

MIT
