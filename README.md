# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~1B parameter byte-level language model trained via knowledge distillation from a larger teacher (e.g. Qwen-14B), using fully ternary weights {-1, 0, 1} and MatMul-free token mixing.

---

## Overview

CPU-1 is a research implementation exploring three ideas simultaneously:

1. **Byte-level processing** — No tokenizer needed. Operates on raw UTF-8 bytes in fixed 4-byte patches.
2. **Fully ternary weights** — Embeddings, token mixer, FFN, and prediction heads all use {-1, 0, 1} weights (BitLinear / BitEmbedding / BitHead), dramatically reducing memory and enabling integer arithmetic.
3. **Knowledge distillation from BPE models** — A novel alignment strategy maps BPE teacher distributions to byte-patch student distributions.

The result is an efficient, compact language model optimized for CPU inference (4 cores, 16GB RAM, 32MB L3), with a full training pipeline: teacher logprob extraction → byte marginalization → student training → inference.

---

## Architecture (v2)

| Hyperparameter | Value |
|---|---|
| `d_model` | 2048 |
| `n_layers` | 20 |
| `d_ff` | 5504 (SwiGLU, ≈8/3 × d_model) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| `mlgru_state` | 80 KB (20 × 2048 × 2 bytes — fixed, fits in L2) |
| **Total parameters** | **~1B** |

### Key Components

- **BitEmbedding** — Ternary embedding table {-1, 0, 1} with per-row FP16 scale. Lookup: `embed = ternary_row * scale` (Slender-Mamba, COLING 2025).
- **BytePatchEmbeddingV2** — Converts 4-byte patches to `d_model` dims: BitEmbedding (8 dims/byte) → concat (32 dims) → BitLinear → d_model.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE) for gradient flow.
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with BitLinear layers, initialized to zero.
- **MatMulFreeTokenMixer (MLGRU)** — Replaces attention entirely. GRU-style recurrence using only Hadamard products, no Q@K^T (arXiv 2406.02528):
  ```
  f_t = sigmoid(BitLinear_f(x_t))        # forget gate
  c_t = silu(BitLinear_c(x_t))           # candidate
  g_t = sigmoid(BitLinear_g(x_t))        # output gate
  h_t = f_t * h_{t-1} + (1 - f_t) * c_t # Hadamard only
  o_t = g_t * h_t
  ```
  Cost per token: O(d_model) vs O(window_size × d_model) for attention. Zero KV cache.
- **SwiGLU FFN** — Gated feed-forward with fully ternary weights.
- **BitHead** — 4 ternary prediction heads (one per byte in the next patch). Size: 4 × 0.125 MB vs 4 × 1 MB for FP16 (Slender-Mamba style).
- **RMSNorm** — Pre-norm before token mixer and FFN in each block.

### Memory at Inference

| Component | v1 (Attention) | v2 (MLGRU) |
|---|---|---|
| Token mixer state | 80 MB (KV cache, 20 × 516 × 2048 × 2 × 2B) | **80 KB** (20 × 2048 × 2B) |
| Fits in cache | No (RAM-bound) | Yes (L2/L3) |

### References

- [Slender-Mamba (COLING 2025)](https://arxiv.org/abs/2501.11117) — Ternary embeddings and prediction heads.
- [Scalable MatMul-free Language Modeling (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) — MLGRU token mixer.

---

## Pipeline

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
│  (CPU1InferenceEngine)      │  → INT8 packing, torch.compile,
│                             │    zero-alloc generation loop
└─────────────────────────────┘
```

### Step 1 — Extract Teacher Logprobs

```bash
python extract_logprobs.py \
  --backend transformers \
  --model Qwen/Qwen2.5-14B-Instruct \
  --input corpus/ \
  --output ./logprobs \
  --top-k-transformers 100
```

Two backends available:
- **`transformers`** (recommended) — Full forward pass, complete vocabulary, 4-bit NF4 quantization (~8GB VRAM for 14B).
- **`ollama`** — Local Ollama API, incremental prefixes, up to 20 logprobs per position.

Output Parquet columns: `doc_id`, `text_bytes`, `positions` (top-k logprobs per token), `n_tokens`.

### Step 2 — Byte Marginalization

```bash
python byte_marginalization.py \
  --logprobs-dir ./logprobs \
  --output-dir ./dataset \
  --seq-len 2048
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
# Dry-run (no GPU required)
python train.py --dry-run --total-steps 10 --log-every 1

# Single GPU with real data
python train.py --data-path ./distillation_dataset --total-steps 100000

# Distributed (e.g. Kaggle 2×T4)
torchrun --nproc_per_node=2 train.py --data-path ./distillation_dataset
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

```python
from inference_v2 import CPU1InferenceEngine
from model_v2 import CPU1ModelV2

model = CPU1ModelV2()
engine = CPU1InferenceEngine(model)
output = engine.generate(b"Hello, world", max_bytes=200)
```

Or from a checkpoint:
```python
engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
output = engine.generate(b"Hello, world", max_bytes=200)
```

The engine applies all CPU optimizations automatically:
- Ternary weights packed to INT8 (`torch._int_mm`)
- Weight matrices transposed for sequential memory access
- Cache-line alignment (64 bytes)
- `torch.compile` with `max-autotune` backend

---

## Project Structure

```
1.58bits/
├── model_v2.py               # Model architecture (v2: MLGRU, fully ternary)
├── train.py                  # Training loop with knowledge distillation
├── inference_v2.py           # CPU-optimized inference engine
├── extract_logprobs.py       # Teacher logprob extraction
├── byte_marginalization.py   # BPE→byte alignment & dataset construction
├── eval/
│   └── bpe_vs_bytebatch_benchmark.py  # Throughput comparison
└── papers/                   # Reference papers
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
No tokenizer required. The model can process any UTF-8 text, code, or binary data without vocabulary limitations. Byte patches (4 bytes each) keep sequence lengths manageable while preserving full fidelity. The small vocabulary (256) also makes prediction heads and embeddings negligibly small.

**Why MLGRU instead of attention?**
On a CPU, attention's O(window_size × d_model) cost per token is memory-bandwidth-bound: a 512-patch KV cache of 4MB/layer × 20 layers = 80MB does not fit in L3 and causes constant cache misses. MLGRU's recurrent state is just 80KB total (all 20 layers), fits in L2, and costs O(d_model) per token with only Hadamard products — no Q@K^T needed.

**Why fully ternary weights?**
{-1, 0, 1} weights (BitLinear, BitEmbedding, BitHead) reduce model size and enable INT8 MAC operations. FPResidual low-rank branches (rank-8) provide precision recovery without giving up the benefits of quantization. Making embeddings and heads ternary (Slender-Mamba) extends this to every layer.

**Why distill from BPE models?**
Strong pretrained BPE models (e.g. Qwen-14B) encode rich language knowledge. Distillation transfers this knowledge to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch.

**The alignment problem**
BPE tokens have variable byte lengths (1–15 bytes); student patches are fixed at 4 bytes. Boundaries rarely align. The marginalization step handles all three alignment cases and extracts a valid byte-level distribution wherever possible (~20–30% of positions), using pure cross-entropy elsewhere.

---

## Ablation Variants

Four architectures are tracked for comparison on the same consumer CPU hardware:

| Variant | Token mixer | Tokenizer | Quantization | Note |
|---|---|---|---|---|
| `v1_slender_mamba_bpe` | Mamba2 | BPE | Full ternary | Baseline (Slender-Mamba style) |
| `v2_cpu1_attention` | Sliding window attention | Byte | Partial ternary | CPU-1 original |
| `v3_cpu1_mlgru` | MLGRU (MatMul-free) | Byte | Partial ternary | MLGRU replaces attention |
| `v4_cpu1_mlgru_full` | MLGRU (MatMul-free) | Byte | **Full ternary** | **Proposed — current implementation** |

---

## Status

This is an active research project. The end-to-end pipeline (v2) is complete. Ongoing work:
- Evaluation on standard language modeling benchmarks (perplexity on FineWeb)
- CPU throughput measurements vs BPE baseline
- Scaling experiments (layer count, model width)

---

## License

MIT
