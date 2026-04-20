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

### Size Comparison at 350M Parameters

Theoretical comparison if CPU-1 were scaled to 350M parameters, vs LFM2.5-350M (LiquidAI).

| Model | Format | Size | Ratio |
|---|---|---|---|
| LFM2.5-350M | BF16 (baseline) | ~700 MB | 1.0× |
| LFM2.5-350M | Q4_K_M (GGUF) | ~219 MB | 3.2× smaller |
| **CPU-1 350M** | **2-bit ternary packed** | **~92 MB** | **7.6× smaller than BF16** |
| **CPU-1 350M** | **vs Q4** | **~92 MB** | **2.4× smaller than Q4** |

CPU-1's ternary 2-bit packing (2 bits/weight) is **half the density of Q4** (4 bits/weight), with FPResidual branches in FP16 accounting for the remaining ~4MB.

Beyond size, CPU-1 has **zero matrix multiplications at inference** (T-MAC LUT + Hadamard products only), while LFM2.5 requires standard Q4 matmul for both its convolution and GQA attention blocks.

### Theoretical CPU Speed Comparison at 350M (i7-8750H, ~38 GB/s RAM)

> All numbers below are **theoretical estimates**. CPU-1 at 350M has not been trained; LFM2.5 measured at ~41 tok/s on same hardware (llama.cpp Q4, from benchmark section below).

**Bottleneck: RAM bandwidth.** At decode, both models stream all weights from RAM each token step:

```
Theoretical peak = bandwidth ÷ model_size
LFM2.5-350M Q4:   38 GB/s ÷ 219 MB ≈ 174 tok/s
CPU-1 350M 2-bit: 38 GB/s ÷  92 MB ≈ 413 tok/s
```

| Factor | LFM2.5-350M Q4 | CPU-1 350M 2-bit |
|---|---|---|
| Compute per weight | Q4 matmul (dequantize + multiply) | T-MAC LUT (lookup only, no multiply) — ~4× faster per byte |
| KV cache | Grows with context (6 GQA blocks) | **Zero** — fixed MLGRU state ~57 KB |
| Cache behavior | All weights from RAM | PFNet + MLGRU state in **L2** (~700 KB) — never touch RAM |
| DeleteGate | — | Eliminates 40% of states in layers N/2..N → ~20% extra speedup |

| Scenario | LFM2.5-350M Q4 | CPU-1 350M 2-bit (estimated) |
|---|---|---|
| Short context (<512 tokens) | ~41 tok/s *(measured)* | **~180–220 tok/s** |
| Long context (>2K tokens) | Degrades (KV cache pressure) | **Unchanged** — state is always 57 KB |
| Theoretical bandwidth peak | ~174 tok/s | ~413 tok/s |
| DDR5 hardware (128 GB/s, e.g. Ryzen AI Max) | ~550 tok/s | **~1 400 tok/s** |

CPU-1's advantage compounds at long context: LFM2.5's KV cache grows linearly (6 attention blocks × context length), while CPU-1's recurrent state stays fixed at 57 KB regardless of context length.

### Inference Strategy

Two execution modes, selected via `USE_C_KERNELS` in `inference/inference.py`:

| Mode | Flag | When to use |
|---|---|---|
| **PyTorch** (default) | `USE_C_KERNELS = False` | All ablation runs — fair comparison, same framework for every architecture |
| **C++ T-MAC LUT** | `USE_C_KERNELS = True` | Production benchmark with the final chosen model |

Ablation BPB/perplexity numbers in the paper are measured under PyTorch mode. A separate production benchmark with `USE_C_KERNELS = True` reports absolute throughput: **~235 patches/sec** on the final ternary MLGRU model (17.4× faster than the INT8 PyTorch baseline).

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

| Paper | What it contributes to CPU-1 | Implemented |
|---|---|---|
| [Scalable MatMul-free LM (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) | MLGRU token mixer — Hadamard-only, no Q@K^T | ✅ `MatMulFreeTokenMixer` |
| [Slender-Mamba (COLING 2025)](https://aclanthology.org/2025.coling-main.316/) | Ternary embedding and prediction heads | ✅ `BitEmbedding`, `BitLinear` |
| [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) | LUT-based zero-multiply inference on CPU | ✅ `CPU1InferenceEngineLUT` |
| [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) | 1.58-bit weight quantization with STE | ✅ `BitLinear` |
| [Bolmo (arXiv 2410.20771)](https://arxiv.org/abs/2410.20771) | Dynamic byte patches via boundary prediction | ✅ `BoundaryPredictor` |
| [MrT5 (arXiv 2512.15586)](https://arxiv.org/abs/2512.15586) | Token deletion gate — prunes 40% of states | ✅ `DeleteGate` |
| [Distilling Byte-Level Models (arXiv 2602.01007)](https://arxiv.org/abs/2602.01007) | Embedding alignment distillation loss | ✅ `EmbeddingAligner` (CE-only for ablation) |
| [LFM2 (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404) | Hybrid CNN + GQA architecture (LiquidAI) | ⚠️ **Not implemented** — external benchmark only (LFM2-350M GGUF) |
| [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531) | KL + CE distillation loss, T=4.0 | ✅ `train.py` distillation mode |

### Papers directory (`papers/`)

| File | Paper |
|---|---|
| `slender-mamba-fully-quantized-1.58bits.pdf` | Slender-Mamba: Fully Quantized Mamba in 1.58 Bits From Head to Toe |
| `efficient-llm-inference-on-cpus.pdf` | Efficient LLM Inference on CPUs |
| `hgrn2-gated-linear-rnns-state-expansion.pdf` | HGRN2: Gated Linear RNNs with State Expansion |
| `scalable-matmul-free-language-modeling.pdf` | Scalable MatMul-free Language Modeling |
| `tmac-cpu-table-lookup-low-bit-llm.pdf` | T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge |
| `stuffed-mamba-oversized-states-inability-to-forget.pdf` | Stuffed Mamba: Oversized States Lead to the Inability to Forget |
| `mrt5-dynamic-token-merging-byte-level-lm.pdf` | MrT5: Dynamic Token Merging for Efficient Byte-Level Language Models |
| `gemma3-technical-report.pdf` | Gemma 3 Technical Report |
| `challenging-gpu-dominance-cpu-on-device-llm.pdf` | Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference |
| `lfm2-liquid-foundation-models-technical-report.pdf` | LFM2 Technical Report |
| `bolmo-byteifying-next-gen-language-models.pdf` | Bolmo: Byteifying the Next Generation of Language Models |
| `distilling-token-models-into-byte-level.pdf` | Distilling Token-Trained Models into Byte-Level Models |
| `tide-token-informed-depth-execution-early-exit.pdf` | TIDE: Token-Informed Depth Execution for Per-Token Early Exit in LLM Inference |

---

## Ablation Study — 10 Progressive Runs (50M) + 4 Scale Runs (10M)

The ablation tells a story: starting from a standard FP16 Transformer baseline, we add **one CPU-1 design choice per run** until we reach the full architecture. Each step isolates the contribution of a single component.

All runs: CE-only training from scratch, 50M params, same data/optimizer/steps.

**Architecture constraint:** 1.58-bit ternary weights (BitLinear/BitEmbedding). No activation×activation matrix multiplications at inference (MLGRU uses Hadamard products only; Folded Transformer uses bounded local attention O(N·w)).

### Progressive Design — 50M runs (run_01 to run_12, run_17)

```
run_01: Transformer  + BPE   + FP16                    ← absolute baseline
          ↓ swap tokenizer
run_02: Transformer  + Byte  + FP16                    ← +byte patches
          ↓ swap mixer + keep byte
run_03: MLGRU        + Byte  + FP16                    ← +MatMul-free mixer AND byte
          ↓ quantize
run_04: MLGRU        + Byte  + Ternary                 ← +1.58-bit weights
          ↓ correct
run_05: MLGRU        + Byte  + Ternary + FPRes         ← +FPResidual = CPU-1 core ⭐
          ↓ branch: add optional components
run_06: CPU-1 core   + Bolmo (dynamic byte patches)    ← +boundary predictor
run_07: CPU-1 core   + DeleteGate (state elimination)  ← +MrT5-style compression
run_08: FoldedXfmr   + Byte  + Ternary (w=128)        ← alternative: local attention
run_09: CPU-1 core   + Bolmo + DeleteGate              ← CPU-1 COMPLETE ⭐
          ↓ add nonlinear residual / study memory dissipation / adaptive depth
run_10: CPU-1 COMPLETE + PFNet (pfnet_hidden=32)       ← +cache-resident MLP
run_11: CPU-1 COMPLETE + per-channel decay (RWKV/HGRN2) ← +learned forgetting (~12KB)
run_12: CPU-1 COMPLETE + periodic state reset           ← structural forgetting, 0 params
run_17: CPU-1 COMPLETE + TIDE exit                     ← +per-token early exit (post-train)
```

Key questions answered:
- `run_01 → run_02`: Does byte-level tokenization hurt a FP16 Transformer?
- `run_02 → run_03`: Does MLGRU improve over Transformer at byte level?
- `run_01 → run_03`: Transformer+BPE vs MLGRU+Byte — combined mixer+tokenization effect
- `run_03 → run_04`: Does ternary quantization hurt MLGRU+byte?
- `run_04 → run_05`: Does FPResidual recover ternary quality loss?
- `run_05 → run_06`: Does Bolmo improve over fixed 4-byte patches?
- `run_05 → run_07`: Does DeleteGate preserve quality while pruning 40% of states?
- `run_05 vs run_08`: CPU-1 core (MLGRU) vs best local-attention Transformer?
- `run_05 → run_09`: Full CPU-1 vs core — combined cost of all extras?
- `run_09 → run_10`: Does PFNet improve quality without hurting inference speed?
- `run_09 → run_11`: Does per-channel learnable decay improve over fixed gate?
- `run_09 → run_12`: Does periodic state reset improve long-sequence generalization?
- `run_11 vs run_12`: Learned dissipation vs structural forgetting — which wins?
- `run_09 → run_17`: Does TIDE early exit reduce CPU-1 FLOPs without quality loss?

### TIDE Early Exit (run_17)

Most tokens "converge" well before the final layer — their hidden representation stops changing significantly mid-network. **TIDE** (Token-Informed Depth Execution, [arXiv 2603.21365](https://arxiv.org/abs/2603.21365)) adds a tiny 2-layer MLP classifier per block that is calibrated **post-training** (2K wikitext samples, < 3 min) and predicts *"has this token converged?"* at each layer. Converged tokens skip all remaining layers. Zero base model retraining.

For CPU-1 @ 50M (d_model=512, 12 layers):

| Component | Cost |
|---|---|
| Exit classifiers | 12 × (512 → 64 → 1) ≈ 394K params (~800 KB) |
| Calibration | 2K samples, ~3 min on GPU, post-training only |
| Expected savings | Frequent tokens ("the", ".", ",") exit at layer 4–6 → ~40–50% FLOPs saved |
| Quality target | ≥ 99% of run_09 perplexity (matching TIDE paper results) |

Reference result: on DeepSeek R1 Distill 8B, TIDE achieves 100% prefill exit rate, 7.2% lower latency, with 99% decode quality preserved on A100.

### Justified with Citations (no runs needed)

| Decision | Citation |
|---|---|
| BitEmbedding / BitHead | Slender-Mamba (Yu et al., COLING 2025) |
| LUT vs INT8 matmul | T-MAC (Wei et al., EuroSys 2025) |
| FPResidual parallel branch | MatMul-free LM (Zhu et al., 2024) |
| SwiGLU | PaLM (Chowdhery et al., 2022) |
| RMSNorm | LLaMA (Touvron et al., 2023) |

### Dataset

All ablation runs use CE training on `Cukinator/cpu1-ablation-dataset`. **Final state of the extraction (frozen for the 10M-parameter ablations):**

| Subset | Shards | Sequences | SEQ_LEN | Raw size |
|---|---|---|---|---|
| `byte_marginalized/` | 168 | 8,400 | 5000 patches × 4 bytes | **168 MB / 168M bytes** |
| `bpe_tokenized/` | 157 | 7,850 | 5000 BPE tokens | **~39.25M Qwen2.5 tokens** |

The dataset was built via `create_dataset.py`, which also extracts **Qwen2.5-3B** teacher logprobs (Top-128 per position) and a 128-dim projection of the teacher's last hidden state for each position — both are stored on Hub and available for future distillation experiments but are not used in the current CE-only ablation runs. Extraction was paused at chunk 527/1700 (~263,500 out of the planned 850,000 documents); this scale is sufficient for the 10M-parameter run grid (see *10M Bonus Runs* below). `stream_checkpoint.json` and `stream_carry.pkl` remain in the Hub repo so the run could be resumed if a larger budget is ever needed.

> **Known gap — shards 0060 and 0061 are intentionally missing in both subsets.** During the original extraction the writer crossed a corrupted region of the streaming pipeline at that point, and although `byte_marginalized/` happened to retain those two parquet files, the matching `bpe_tokenized/` shards were lost and could not be recovered from the Hub commit history (search across all 891 historical commits returned no version of the file). Rather than ship two halves of an asymmetric pair — which would silently break any downstream code that expects parallel sequence indexing across the two subsets — the byte counterparts (`byte_marginalized/shard_0060.parquet`, `byte_marginalized/shard_0061.parquet`) were deleted as well, leaving the same gap on both sides. The shard indices are otherwise contiguous; the gap is a stable, well-known property of this dataset version, not a sign of incomplete download.

Future work: scale to `HuggingFaceFW/fineweb` (sample-10BT) for Chinchilla-scale training (~1B tokens for 50M+ params).

> **Known behavior — shard offset on resume:** Each pipeline interruption causes a small gap of ~2 shards per stream (sequences written after the last checkpoint are skipped on resume). Across the full dataset this amounts to roughly **~3% data loss**, considered acceptable. `byte_marginalized` and `bpe_tokenized` track independent shard offsets (v2 checkpoint format) so they never drift relative to each other across restarts. The final dataset only needs a shard renumbering pass to close any filename gaps before training.

> **Expected behavior — shard count difference between streams:** `byte_marginalized/` ends up with slightly more shards than `bpe_tokenized/` (final state: **168 vs 157**, exactly **11 shards = ~7% more byte shards**, after the symmetric removal of shards 0060/0061 documented above). Both use `SEQ_LEN = 5000`, but the two streams pack the corpus at different densities: a byte sequence covers 5000 patches × 4 bytes = **20.0 KB** of source text, while a BPE sequence covers 5000 tokens × ~4.28 bytes ≈ **21.4 KB** of source text (Qwen2.5 tokenizer on FineWeb English averages ~4.28 bytes/token). The same corpus therefore yields ~7% more byte sequences than BPE sequences, which the empirical 168/157 ratio matches almost exactly. The two streams also carry independent residual state across chunk boundaries and are uploaded with independent shard offsets (v2 checkpoint format). This is structural and expected — each stream has the number of shards that its encoding density warrants, and a given `shard_XXXX.parquet` in `byte_marginalized/` is **not** required to align with the same index in `bpe_tokenized/`.

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

### Ablation Pipeline (50M × 10 runs on Lightning.ai T4)

#### Lightning Studio Setup — Files to Upload

Minimum set of files needed to train any ablation run on a Lightning Studio T4 instance:

| File | Purpose |
|---|---|
| `model.py` | CPU-1 architecture — imported by `ablation.py` and `train.py` |
| `train.py` | Training loop with `--mode ablation` |
| `ablation.py` | Ablation configs and configurable model variants |
| `requirements.txt` | Python dependencies |

Everything else is remote or not required:
- **Dataset** — streamed from `Cukinator/cpu1-ablation-dataset` on HF Hub; use `--hf-mount` to avoid downloading the full ~5 GB
- **Credentials** — set `HF_TOKEN` as a Lightning Secret (Settings → Secrets), not via `.env`
- `create_dataset.py` — dataset already built and on HF Hub
- `inference/`, `kaggle/`, `eval/` — not needed at training time

```bash
# On the Lightning VM after uploading the four files above:
pip install -r requirements.txt

# Train any run with lazy dataset streaming (no full download):
python train.py --mode ablation --run run_10 --hf-mount
```

#### 1. Build Shared Dataset (once)

```bash
# Full pipeline: extract teacher logprobs + build byte and BPE datasets
python create_dataset.py --n_docs 50000

# Build only byte-marginalization from existing logprobs
python create_dataset.py --logprobs-dir ./logprobs --output-dir ./dataset
```

Teacher (Qwen2.5-3B in 4-bit NF4 or FP8) runs **once**. Produces:
- `byte_marginalized/` — `patches`, `targets`, `teacher_probs`, `teacher_mask`, `teacher_hidden`, `teacher_dim`
- `bpe_tokenized/` — `input_ids`, `labels`, `teacher_probs_bpe`, `teacher_ids_bpe`, `teacher_mask_bpe`
- Uploaded to `Cukinator/cpu1-ablation-dataset` on HuggingFace Hub (50 sequences per shard, configs declared in YAML so `load_dataset(repo, "byte_marginalized")` and `load_dataset(repo, "bpe_tokenized")` both work)

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

#### Two-Phase Training (runs 07 and 09 — DeleteGate)

For runs that include DeleteGate (MrT5-style), training proceeds in two phases:

- **Phase 1 (CE pretraining):** Standard CE training. DeleteGate disabled (elimination_rate=0).
  `python train.py --mode ablation --run run_07 --phase 1`

- **Phase 2 (Compression):** DeleteGate activated. Elimination rate anneals to 40% over training.
  `python train.py --mode ablation --run run_07 --phase 2 --phase1_ckpt <path>`

#### 3. Checkpoint Formats

Two checkpoint formats are saved automatically:

| Format | When | Contents | Size (50M) | Size (170M) |
|---|---|---|---|---|
| **Resume** (intermediate) | Every N steps | BF16 model + BF16 optimizer state | ~214 MB | ~700 MB |
| **Final** (inference) | End of training | 2-bit packed model, no optimizer | **~14 MB** | **~44 MB** |

The final inference checkpoint uses 2-bit ternary packing: {-1, 0, 1} → {0, 1, 2}, 4 values per byte. Non-ternary weights (norms, gates) are stored as BF16. Load with `unpack_compact_state()` from `model.py`.

#### 4. Analyze Results

```bash
python ablation.py --metrics_dir .
```

Generates a text table for terminal and a LaTeX table for the paper (`ablation_table.tex`).

---

## 10M Scale Runs (run_13 to run_16) — Training Strategy Study

Before committing to long 50M runs, we train **~10M parameter models** to answer one targeted question: **which training data strategy produces the best byte-level model at this scale?** All four 10M runs use the **same CPU-1 COMPLETE architecture** (MLGRU + ternary + FPResidual + Bolmo + DeleteGate). Only the training data strategy varies.

### Architecture (10M — same as CPU-1 COMPLETE)

| Hyperparameter | Value |
|---|---|
| `d_model` | 320 |
| `n_layers` | 8 |
| `d_ff` | 853 (SwiGLU, ~8/3 × d_model) |
| `vocab_size` | 256 (byte-level) or 4096 (compact BPE — see note) |
| **Total parameters** | **~10M** |

### 10M Runs

| Run | Training strategy | Data | Key question |
|---|---|---|---|
| `run_13` | CPU-1 COMPLETE + **compact BPE** (4K vocab), from scratch | `bpe_tokenized/` | Does subword tokenization beat byte-level at 10M? |
| `run_14` | CPU-1 COMPLETE + **byte-level**, from scratch | `byte_marginalized/` | Pure byte baseline (with Qwen teacher probs available) |
| `run_15` | CPU-1 COMPLETE + **complement** — BPE warm-start → byte finetune | BPE then byte | Does BPE pretraining bootstrap byte quality? |
| `run_16` | CPU-1 COMPLETE + **raw bytes**, from scratch, **no teacher** | fineweb raw bytes | Do Qwen teacher logprobs in run_14 actually help? |

**BPE vocab constraint:** Qwen's full 152K BPE vocab → 152K × 320 = 48.7M embedding parameters — larger than the entire 10M model. `run_13` uses a compact 4096-token vocabulary (top-4096 by frequency from `cpu1-ablation-dataset`). Embedding: ~1.3M params.

**Training budget:** 2500 steps each (~1h on T4, ~200M effective tokens).
`run_15` starts from the `run_13` checkpoint, replaces the BPE embedding with a fresh byte-patch embedding, then trains for 2500 byte steps — same byte-step budget as `run_14` for a fair comparison. Inspired by Bolmo's two-stage byteification procedure.

### Chinchilla Scaling Analysis

All four 10M runs are **compute-optimal by Chinchilla law** (tokens ≈ 20 × params = 200M tokens for 10M params).

Dataset structure: `Cukinator/cpu1-ablation-dataset` contains **16,250 total rows** across both splits (8,400 byte sequences + 7,850 BPE sequences):

| Run | Tokenization | Dataset subset | Total tokens | Cycling for 200M | Convergence risk |
|---|---|---|---|---|---|
| `run_13` | BPE (4K vocab) | `bpe_tokenized/` (157 shards) | 7,850 rows × 5,000 BPE = **39.25M BPE** | 200M ÷ 39.25M = **5.10×** | ⚠️ **Moderate** (repeated data) |
| `run_14` | Byte-level | `byte_marginalized/` (168 shards) | 8,400 rows × 5,000 × 4 bytes = **168M bytes** | 200M ÷ 168M = **1.19×** | ✅ **Excellent** (≈ Chinchilla, near-zero repetition) |
| `run_15` | Byte-level (warm-start) | `byte_marginalized/` (168 shards) | Same as run_14 = **168M bytes** | **1.19×** | ✅ **Excellent** |
| `run_16` | Raw bytes, no teacher | `HuggingFaceFW/fineweb` | ~1B+ tokens available | **1.0×** (uses 200M fresh) | ✅ **Good** |

**Key insight:** Byte-level runs are virtually Chinchilla-perfect (1.19 cycles ≈ no repetition). BPE requires ~5 cycles of the same data — modest overfitting risk but acceptable for an ablation grid (the comparison is against the byte runs, not against an absolute LM-quality bar). The shard count difference between `byte_marginalized/` (168) and `bpe_tokenized/` (157) is **structural and expected** — see *Dataset* section above for the encoding-density explanation, and the *Known gap* note for the symmetric removal of shards 0060/0061.

For the paper: *"10M models trained with Chinchilla-optimal scaling (200M tokens each). BPE subset requires ~5× dataset cycling; byte-level subsets converge with near-zero data repetition."*

### What the results answer

| Comparison | Question |
|---|---|
| `run_13` vs `run_14` | BPE vs byte tokenization — perplexity, convergence speed |
| `run_14` vs `run_15` | Pure byte vs BPE-bootstrapped byte — does warm-start help? |
| `run_14` vs `run_16` | Byte with Qwen teacher probs vs raw bytes CE-only — do logprobs help? |
| `run_13` → `run_15` | Quality gain from adapting a BPE model to byte-level |

```bash
python train.py --mode ablation --run run_13   # BPE baseline (~1h)
python train.py --mode ablation --run run_14   # byte + teacher (~1h)
python train.py --mode ablation --run run_15 --complement_ckpt checkpoints/run_13.pt
python train.py --mode ablation --run run_16   # raw bytes, no teacher (~1h)
```

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

Two engines, both accessible via `inference.py`.

```python
# INT8 — ternary weights as quantized float32 + torch.compile
engine = CPU1InferenceEngine.from_checkpoint('checkpoint.pt', config)
output = engine.generate(b"Buenos dias", max_bytes=200)

# T-MAC C++ LUT Kernels — zero allocations, SIMD unrolled, precomputed lookup table + gather
engine = CPU1InferenceEngineLUT.from_checkpoint_with_lut(
    'checkpoint.pt', config, group_size=4
)
output = engine.generate(b"Buenos dias", max_bytes=200)
```

### CPU-1 350M Performance (Local C++ Engine)

The custom C++ kernels (`inference/kernels.cpp`) are ultra-optimized for the MLGRU architecture. They bypass the PyTorch dispatcher and leverage OpenMP, explicit SIMD instructions, Deep Data Dependency Unrolling (DDU), and `thread_local` memory reuse (Zero-Allocation).

**Benchmark on 350M Ablation Model (6 Threads)**:
| Engine | Throughput (Patches/sec) | Latency p50 (ms) | Bandwidth (Bytes/sec) | Speedup |
|---|---|---|---|---|
| INT8 Baseline (PyTorch) | 13.5 TPS | 18.48 ms | ~54 B/s | 1.0x |
| **C++ T-MAC LUT** | **234.8 TPS** | **1.63 ms** | **~939 B/s** | **17.4x** 🚀 |

Since `patch_size = 4`, the 235 TPS translates to generating **almost 1 KB of text per second** entirely locally on a consumer CPU without GPU VRAM constraints.

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
├── ablation.py           # Ablation system: BASE_CONFIG + 10 progressive RUN_CONFIGS,
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
| MLGRU + Byte + Ternary | run_04–run_07, run_09–run_12, run_14–run_16, CPU-1 170M | All 14 checks |
| MLGRU + Byte + FP16 | run_03 | Skip ternary_weights |
| Transformer + Byte + FP16 | run_02 | Skip ternary + states |
| Transformer + BPE + FP16 | run_01 | Skip ternary + states + generation |
| Folded Transformer + Byte + Ternary | run_08 | Skip state_prop, state_size |

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
- **Ablation system** — 13 progressive runs (run_01–run_12 + run_17, 50M) + 4 scale runs (run_13–run_16, 10M), configurable variants, shared dataset, LaTeX table generation (`ablation.py`)
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

- Run the 13 ablation experiments (run_01–run_12, run_17) + 4 scale runs (run_13–run_16) on Kaggle T4 and collect metrics
- Scale winning ablation config to 170M and train on full dataset
- ~~T-MAC C++ kernel integration (bitnet.cpp / llama.cpp)~~ ✅ **Done** (reached 235 TPS)

---

## License

MIT

