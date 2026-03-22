# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from a larger teacher (e.g. Qwen-14B), using fully ternary weights {-1, 0, 1}, MatMul-free token mixing, and PFNet (Prefetch-Aware Nonlinear Residual).

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
| `d_model` | 1024 |
| `n_layers` | 14 |
| `d_ff` | 2730 (SwiGLU, ≈8/3 × d_model) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| `pfnet_hidden` | 64 (PFNet hidden dim, 224 KB total across all layers) |
| `mlgru_state` | 28 KB (14 × 1024 × 2 bytes — fixed, fits in L2) |
| **Total parameters** | **~170M** |

### Key Components

- **BitEmbedding** — Ternary embedding table {-1, 0, 1} with per-row FP16 scale. Lookup: `embed = ternary_row * scale` (Slender-Mamba, COLING 2025).
- **BytePatchEmbeddingV2** — Converts 4-byte patches to `d_model` dims: BitEmbedding (8 dims/byte) → concat (32 dims) → BitLinear → d_model.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE) for gradient flow.
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with BitLinear layers, initialized to zero.
- **PFNet (Prefetch-Aware Nonlinear Residual)** — A 2-layer ternary MLP (hidden=64) added as a residual branch to each block. Addresses the linear interpolation bottleneck in MLGRU mixing: while MLGRU gates are nonlinear, the information pathway itself (`h_t = f*h + (1-f)*c`) is a linear combination. PFNet applies direct SiLU nonlinearity to the information path. Sized so all PFNet weights across all 14 layers (224 KB) fit permanently in L2 cache alongside the recurrent state (28 KB), remaining cache-resident while main model weights stream from RAM. This mirrors the CPU hardware prefetch pattern: PFNet runs on data already in L2 while the memory controller fetches the next layer's main weights.
  ```
  h_pf = W2 @ SiLU(W1 @ RMSNorm(x))   # W1, W2 ternary 1024×64
  block_out = mlgru_out + ffn_out + h_pf
  ```
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

| Component            | v1 (Attention)                          | v2 (MLGRU + PFNet)       |
|---|---|---|
| Token mixer state    | 80 MB (KV cache)                        | **28 KB** (14 × 1024 × 2B) — L2 |
| PFNet weights        | —                                       | **224 KB** (all layers) — L2 |
| Active layer weights | —                                       | ~3.1 MB (1 layer) — L3   |
| Full model weights   | —                                       | ~44 MB — RAM streaming   |
| Fits in cache        | No (RAM-bound)                          | State + PFNet: yes (L2)  |

> The MLGRU recurrent state (28 KB) and all PFNet weights (224 KB)
> together fit in L2 cache (~256–512 KB per core on modern CPUs),
> remaining resident across the full generation loop. Main model
> weights stream from RAM one layer at a time (3.1 MB active
> footprint), fitting within a typical 32 MB L3. The full model
> (44 MB) does not fit in L3 and is not expected to — only the
> active layer needs to be in L3 at any time.

### References

- [Slender-Mamba (COLING 2025)](https://arxiv.org/abs/2501.11117) — Ternary embeddings and prediction heads.
- [Scalable MatMul-free Language Modeling (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) — MLGRU token mixer.
- [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) — LUT-based inference for low-bit LLMs, replacing MatMul with table lookup.

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

#### Step 4b — Inference with T-MAC LUT (zero multiplications)

```python
from inference_tmac import CPU1InferenceEngineLUT

model = CPU1ModelV2()
engine = CPU1InferenceEngineLUT(model, config, group_size=4, use_lut=True)
output = engine.generate(b"Hello, world", max_bytes=200)
```

Or from a checkpoint:
```python
engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
    'checkpoint.pt', config, group_size=4
)
output = engine.generate(b"Hello, world", max_bytes=200)
```

The T-MAC LUT engine replaces all BitLinear matrix multiplications with precomputed Lookup Tables, based on [T-MAC (Wei et al., EuroSys 2025)](https://arxiv.org/abs/2407.00088):
- Ternary weights packed to uint8 (4 weights per byte, 2-bit encoding)
- LUT precomputed per activation group: 256 entries (1KB, fits in L1)
- Zero multiplications at inference — only lookup + accumulation
- ~4x fewer arithmetic operations vs standard MatMul
- Fallback to INT8 mode with `use_lut=False`

**Benchmark comparison:**
```python
from inference_tmac import benchmark_lut_vs_int8_vs_fp16
results = benchmark_lut_vs_int8_vs_fp16(config, checkpoint_path='checkpoint.pt')
```

---

## Project Structure

```
1.58bits/
├── model_v2.py               # Model architecture (v2: MLGRU, fully ternary)
├── train.py                  # Training loop with knowledge distillation
├── inference_v2.py           # CPU-optimized inference engine (INT8)
├── inference_tmac.py         # T-MAC LUT inference (zero multiplications)
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

**Why PFNet?**
The MLGRU mixing operation `h_t = f_t * h_{t-1} + (1-f_t) * c_t` is a convex interpolation — linear in both h and c. All nonlinearity in the MLGRU lives in the gates (sigmoid, SiLU), not in the information transformation itself. This limits the model's ability to learn high-frequency feature interactions and multiplicative dependencies. PFNet adds a direct nonlinear pathway (2-layer ternary MLP with SiLU) as a residual correction. The hidden dimension (64) is chosen so that all PFNet weights across all layers occupy 224 KB — fitting in L2 cache alongside the recurrent state. This means PFNet computation uses data already in L2, effectively running during cycles that would otherwise be idle while the hardware prefetch unit loads the next layer's main weights from RAM.

**Why fully ternary weights?**
{-1, 0, 1} weights (BitLinear, BitEmbedding, BitHead) reduce model size and enable INT8 MAC operations. FPResidual low-rank branches (rank-8) provide precision recovery without giving up the benefits of quantization. Making embeddings and heads ternary (Slender-Mamba) extends this to every layer.

**Why T-MAC LUT for BitLinear projections?**
Even with ternary weights packed to INT8, the CPU still performs multiply-accumulate operations. T-MAC (Wei et al., EuroSys 2025) replaces these with precomputed Lookup Tables. For groups of 4 ternary weights, a 256-entry LUT (1KB, always in L1) maps each possible weight combination to its dot product with the activations. Inference becomes pure lookup + accumulation — zero multiplications. Combined with MLGRU (Hadamard-only token mixing), CPU-1 v2 eliminates ALL matrix multiplications at inference:
1. Token mixing: MLGRU Hadamard products (no Q@K^T)
2. BitLinear projections: T-MAC LUT lookup (no MatMul)
3. Weights: ternary {-1,0,1} throughout

**Why distill from BPE models?**
Strong pretrained BPE models (e.g. Qwen-14B) encode rich language knowledge. Distillation transfers this knowledge to the byte-level student more efficiently than training from scratch, despite the tokenization mismatch.

**The alignment problem**
BPE tokens have variable byte lengths (1–15 bytes); student patches are fixed at 4 bytes. Boundaries rarely align. The marginalization step handles all three alignment cases and extracts a valid byte-level distribution wherever possible (~20–30% of positions), using pure cross-entropy elsewhere.

---

## Ablation Variants

Four architectures are tracked for comparison on the same consumer CPU hardware:

| Variant | Token mixer | Tokenizer | Quantization | Note |
|---|---|---|---|---|
| `v1_slender_mamba_bpe` | Mamba2 | BPE | Full ternary | Baseline |
| `v2_cpu1_attention` | Sliding window attention | Byte | Partial ternary | CPU-1 original |
| `v3_cpu1_mlgru` | MLGRU | Byte | Partial ternary | MLGRU replaces attention |
| `v4_cpu1_mlgru_full` | MLGRU | Byte | Full ternary | Full ternary |
| `v5_cpu1_170m_pfnet` | MLGRU + PFNet | Byte | Full ternary | **Proposed — 170M, PFNet** |

---

## Status

This is an active research project. Current status:
- Architecture scaled to 170M (d_model=1024, 14 layers)
- PFNet implemented and integrated into CPU1Block
- T-MAC LUT inference engine implemented (zero MatMul at inference)
- Evaluation on standard benchmarks (perplexity on FineWeb) — pending
- CPU throughput measurements: INT8 vs T-MAC LUT — pending
- Ablation: v4 (no PFNet) vs v5 (PFNet) to isolate contribution
- T-MAC C++ kernel benchmark (bitnet.cpp / llama.cpp) — future work

---

## License

MIT
