# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~1B parameter byte-level language model trained via knowledge distillation from a larger teacher (e.g. Qwen-14B), using ternary weight quantization {-1, 0, 1}.

---

## Overview

CPU-1 is a research implementation exploring three ideas simultaneously:

1. **Byte-level processing** — No tokenizer needed. Operates on raw UTF-8 bytes in fixed 4-byte patches.
2. **Ternary weight quantization** — All core projections use {-1, 0, 1} weights (BitLinear), dramatically reducing memory and enabling integer arithmetic.
3. **Knowledge distillation from BPE models** — A novel alignment strategy maps BPE teacher distributions to byte-patch student distributions.

The result is an efficient, compact language model with a full training pipeline: teacher logprob extraction → byte marginalization → student training → inference.

---

## Architecture

| Hyperparameter | Value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 20 |
| `n_heads` | 16 |
| `head_dim` | 128 |
| `d_ff` | 5504 (SwiGLU, ≈8/3 × d_model) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `window_size` | 512 patches |
| `n_sinks` | 4 |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| **Total parameters** | **~1B** |

### Key Components

- **BytePatchEmbedding** — Embeds each 4-byte patch: 8 dims/byte → Linear projection to `d_model`.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE) for gradient flow.
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with each BitLinear layer, initialized to zero and gradually refined during training.
- **SlidingWindowAttention** — Local 512-patch window + 4 global sink tokens; memory stays O(1) with sequence length.
- **RoPE** — Rotary Positional Embeddings up to 8192 patches (32KB of context).
- **SwiGLU FFN** — Gated feed-forward with ternary weights.
- **4 Prediction Heads** — Independent FP16 linear layers, one per byte in the next patch.
- **ChunkEncoder / InContextRetrieval** — Retrieval encoder using the first 4 transformer blocks + cosine-similarity based chunk lookup.

---

## Pipeline

```
Raw Text Corpus
      │
      ▼
┌─────────────────────────────┐
│  extract_logprobs.py        │  Teacher logprob extraction
│  (Ollama API or Transformers│  → Parquet with top-k logprobs per position
│   backend with 4-bit NF4)   │
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
│  (DDP, AMP, gradient        │  → Checkpoints to HuggingFace Hub
│   accumulation)             │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  inference.py               │  Generation & benchmarking
│                             │  → Streaming output, benchmarks,
│                             │    interactive REPL
└─────────────────────────────┘
```

### Step 1 — Extract Teacher Logprobs

```bash
python extract_logprobs.py \
  --backend transformers \
  --model Qwen/Qwen2.5-14B-Instruct \
  --input corpus.txt \
  --output logprobs.parquet \
  --top_k 100
```

Two backends available:
- **`transformers`** (recommended) — Full forward pass, complete vocabulary, 4-bit NF4 quantization (~8GB VRAM for 14B).
- **`ollama`** — Local Ollama API, incremental prefixes, up to 20 logprobs per position.

Output Parquet columns: `doc_id`, `text_bytes`, `positions` (top-k logprobs per token), `n_tokens`.

### Step 2 — Byte Marginalization

```bash
python byte_marginalization.py \
  --input logprobs.parquet \
  --output dataset/ \
  --seq_len 2048
```

Solves the core alignment problem between variable-length BPE tokens and fixed 4-byte patches. Three cases:

| Case | Condition | Teacher Signal |
|---|---|---|
| `ALIGNED` | Patch boundary = BPE boundary | Yes — full first-byte marginal |
| `INSIDE_TOKEN` | Boundary inside a BPE token | No — deterministic bytes |
| `MULTI_TOKEN` | Multiple BPE tokens per patch | First byte only |

Output dataset format:
```
patches:       [seq_len, 4]   uint8    # Input patches
targets:       [seq_len, 4]   uint8    # Next patch (shifted)
teacher_probs: [seq_len, 256] float32  # P(byte_0 | context)
teacher_mask:  [seq_len]      bool     # Whether teacher signal exists
```

### Step 3 — Train

```bash
# Single GPU
python train.py --dataset dataset/ --output checkpoints/

# Distributed (e.g. Kaggle 2×T4)
torchrun --nproc_per_node=2 train.py --dataset dataset/ --output checkpoints/
```

**Distillation Loss**:
- **Head 0** (first byte): 90% KL divergence + 10% cross-entropy at temperature T=4.0.
- **Heads 1–3** (remaining bytes): Pure cross-entropy (BPE teacher has no signal for internal bytes).
- **Temperature compensation** (T²): Rescales KL loss to account for gradient scaling.

**Optimizer**:
- BitLinear weights → `weight_decay=0.1`
- FPResidual, norms, biases → `weight_decay=0.0`
- Prediction heads → `weight_decay=0.01`
- Cosine annealing LR with linear warmup.

### Step 4 — Inference

```bash
# Single generation
python inference.py --checkpoint checkpoints/model.pt --prompt "Hello, world"

# Interactive REPL
python inference.py --checkpoint checkpoints/model.pt --interactive

# Throughput benchmark
python inference.py --checkpoint checkpoints/model.pt --benchmark
```

Sampling options: greedy, temperature, top-k, top-p (nucleus).

---

## Project Structure

```
1.58bits/
├── model.py                  # Full model architecture
├── train.py                  # Training loop with knowledge distillation
├── inference.py              # Text generation & benchmarking
├── extract_logprobs.py       # Teacher logprob extraction
└── byte_marginalization.py   # BPE→byte alignment & dataset construction
```

---

## Dependencies

```bash
pip install torch torchvision
pip install transformers datasets huggingface_hub
pip install bitsandbytes          # 4-bit NF4 quantization for teacher
pip install pyarrow               # Parquet I/O
pip install numpy requests
```

> For Ollama backend: install [Ollama](https://ollama.ai) separately.

---

## Design Decisions

**Why byte-level?**
No tokenizer required. The model can process any UTF-8 text, code, or binary data without vocabulary limitations. Byte patches (4 bytes each) keep sequence lengths manageable while preserving full fidelity.

**Why ternary weights?**
{-1, 0, 1} weights reduce model size and enable integer MAC operations. FPResidual low-rank branches provide precision recovery without giving up the benefits of quantization.

**Why distill from BPE models?**
Strong pretrained BPE models (e.g. Qwen-14B) encode rich language knowledge. Distillation transfers this knowledge to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch.

**The alignment problem**
BPE tokens have variable byte lengths (1–15 bytes); student patches are fixed at 4 bytes. Boundaries rarely align. The marginalization step handles all three alignment cases and extracts a valid byte-level distribution wherever possible (~20–30% of positions), using pure cross-entropy elsewhere.

---

## Status

This is an active research project. The pipeline is complete end-to-end. Ongoing work:
- KV cache implementation for faster autoregressive generation
- Scaling experiments (layer count, model width)
- Evaluation on standard language modeling benchmarks

---

## License

MIT
