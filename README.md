# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen2.5-3B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

---

## Architecture (v2 — 170M)

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| `n_layers` | 14 |
| `d_ff` | 2730 (PyTorch `model.py::D_FF`, ~8/3 × 1024) / 2688 (C benchmark default, rounded to multiple of 64) |
| `vocab_size` | 256 (byte-level) |
| `patch_size` | 4 bytes |
| `fp_rank` | 8 (FPResidual low-rank correction) |
| `pfnet_hidden` | 64 (PFNet hidden dim, 224 KB total across all layers) |
| `mlgru_state` | 28 KB fp16 / 56 KB fp32 (14 × 1024 × {2, 4} bytes — fixed, fits in L2) |
| **Total parameters** | **~170M** |

### Key Components

- **BitEmbedding** — Ternary embedding table {-1, 0, 1} with per-row FP16 scale (Slender-Mamba, COLING 2025).
- **BytePatchEmbeddingV2** — Converts 4-byte patches to `d_model` dims: BitEmbedding (8 dims/byte) → concat (32 dims) → BitLinear → d_model.
- **BitLinear** — Ternary weight projection with Straight-Through Estimator (STE).
- **FPResidual** — Low-rank FP16 correction branches (rank-8) in parallel with BitLinear, initialized to zero.
- **PFNet** — 2-layer ternary MLP residual per block (hidden=64). All PFNet weights across 14 layers (224 KB) fit in L2 alongside the MLGRU state (28 KB fp16 / 56 KB fp32), staying cache-resident while main weights stream from RAM.
- **MatMulFreeTokenMixer (MLGRU)** — GRU-style recurrence with Hadamard products only, no Q@K^T (arXiv 2406.02528). O(d_model) per token, zero KV cache.
- **SwiGLU FFN** — Gated feed-forward with fully ternary weights.
- **BitHead** — 4 ternary prediction heads (one per byte in next patch).
- **RMSNorm** — Pre-norm before token mixer and FFN.

### Memory at Inference

| Component | v1 (Attention) | v2 (MLGRU + PFNet) |
|---|---|---|
| Token mixer state | 80 MB (KV cache) | **28 KB fp16** / **56 KB fp32** (14 × 1024 × {2,4}B) — L2 |
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

Two execution paths — both live in the current tree, but in different places:

| Path | Where it lives | When to use |
|---|---|---|
| **PyTorch** (ablation / training) | `model.py` forward pass, driven by `train.py` and the `eval/` suite | All ablation runs — fair comparison, same framework for every architecture |
| **C++ LUT** (production) | `inference/kernels.cpp` + `inference/benchmark.c` standalone driver | Production benchmark with the final chosen model |

> **The C++ LUT kernels are NOT used in any ablation run.** All ablation BPB/perplexity numbers are measured in pure PyTorch via `model.py`. This is by design: running the C++ LUT only on ternary MLGRU models (run_05 onward) while the FP16 Transformer baselines (run_01–run_03) stay in PyTorch would conflate kernel speed with architecture quality — making the comparison unfair.

The C++ path (`inference/kernels.cpp`) provides: AVX2 `ternary_lut_matmul_omp`, fused MLGRU gate scan (`mlgru_scan_fusion`), INT8 GEMV (`ternary_i8_fused_gemv_avx2`), RMSNorm + SwiGLU, the `forward_full_mad_*` family (modes 0–4) and `local_decode_step` for the MegaByte local decoder. The standalone `inference/benchmark.c` driver — **not a Python engine** — is the authoritative consumer; current numbers are in *CPU-1 170M Performance — Current C Benchmark Status* below. The legacy `inference.py` Python engine has been removed in favour of the C driver.

### Extended Architecture (v3)

CPU-1 v3 extends the base architecture with three optional modules:

- **BoundaryPredictor** (Bolmo-inspired): Replaces fixed 4-byte patches with dynamic patches that align with natural language boundaries. Adds ~2KB of parameters, runs in Thread 2 during CPU inference with zero added latency.

- **DeleteGate** (MrT5-inspired): Placed after layer N//2, learns to discard uninformative states before the remaining layers. Reduces active states by up to 40%, activated only in training Phase 2.

- **EmbeddingAligner** (Distilling Token-Trained Models into Byte-Level Models, see [papers/distilling-token-models-into-byte-level.pdf](papers/distilling-token-models-into-byte-level.pdf)): Aligns the student's hidden space with the teacher's embedding space via cosine similarity loss. Trained in Phase 1 only, frozen in Phase 2.

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
| [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) | LUT-based zero-multiply inference on CPU | ✅ `inference/kernels.cpp::ternary_lut_matmul_omp` + `forward_full_mad_*` |
| [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) | 1.58-bit weight quantization with STE | ✅ `BitLinear` |
| [Bolmo (arXiv 2410.20771)](https://arxiv.org/abs/2410.20771) | Dynamic byte patches via boundary prediction | ✅ `BoundaryPredictor` |
| [MrT5](papers/mrt5-dynamic-token-merging-byte-level-lm.pdf) | Token deletion gate — prunes 40% of states | ✅ `DeleteGate` |
| [Distilling Token-Trained Models into Byte-Level Models](papers/distilling-token-models-into-byte-level.pdf) | Embedding alignment distillation loss | ✅ `EmbeddingAligner` (CE-only for ablation) |
| [LFM2 (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404) | Hybrid CNN + GQA architecture (LiquidAI) | ⚠️ **Not implemented** — external benchmark only (`models/LFM2.5-350M/` safetensors; a Q4_K_M GGUF is also benchmarked via `eval_external`) |
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
| `MEGABYTE-Predicting-byte-sequences.pdf` | MegaByte: Predicting Million-byte Sequences with Multiscale Transformers (Yu et al., 2023) — drives `LocalByteDecoder` |

---

## Ablation Study — 14 Progressive Runs (50M) + 4 Scale Runs (10M)

The ablation tells a story: starting from a standard FP16 Transformer baseline, we add **one CPU-1 design choice per run** until we reach the full architecture. Each step isolates the contribution of a single component.

All runs: CE-only training from scratch, 50M params, same data/optimizer/steps.

The 50M track holds 14 configs: the linear chain `run_01 … run_12` plus two sibling variants branching off `run_09` (CPU-1 COMPLETE) — `run_17` (TIDE early exit) and `new_1` (Bolmo integrated gating). See `RUN_CONFIGS` in `ablation.py` for the authoritative list.

**Architecture constraint:** 1.58-bit ternary weights (BitLinear/BitEmbedding). No activation×activation matrix multiplications at inference (MLGRU uses Hadamard products only; Folded Transformer uses bounded local attention O(N·w)).

### Byte head — MegaByte Local Decoder (intra-patch autoregression)

All `tokenization='byte'` and `tokenization='byte_dynamic'` runs (run_02 onward and the 10M scale runs) predict the 4 bytes of each patch through a **MegaByte local decoder** ([Yu et al., 2023](papers/MEGABYTE-Predicting-byte-sequences.pdf), Listing 1) rather than four independent heads.

**Why:** with four parallel heads `P(b₀ b₁ b₂ b₃ | ctx) = ∏ P(bᵢ | ctx)`, the bytes inside a patch are conditionally independent — byte 2 cannot see byte 1. That directly breaks UTF-8 continuation bytes (lead + continuation), common bigrams (`th`, `\r\n`, `_la`), and shared word prefixes. The output degrades into locally plausible but globally incoherent noise, hurting any downstream reasoning task.

**How (paper-canonical wiring):**

1. **Per-position global slice.** The global patch state `h_G ∈ ℝ^{d_model}` is projected via a single `BitLinear(d_model, P·d_local)` and reshaped to `[P, d_local]`. Each of the P byte positions receives **its own slice** of the global context — position 0's slice can encode "how the patch starts", position 3's slice can encode "how it closes".
2. **Shift-right byte input.** Position k's input byte is `byte_embed(b_{k-1})` for k>0, or a zero-padded embedding for k=0.
3. **Local transformer block.** Position-specific global slice + shifted byte embedding + learned position embedding feed a 1-layer causal transformer (SwiGLU FFN with **ratio 4×**, RMSNorm, d_local=128, 4 heads) over the length-P sequence.

Training is teacher-forced in parallel over all P positions with a causal mask; inference calls `generate_step` once per byte to sample autoregressively within the patch.

| Aspect | Parallel heads (old) | LocalByteDecoder (current) |
|---|---|---|
| Factorization | `∏ P(bᵢ \| ctx)` — independence | `∏ P(bᵢ \| ctx, b₀..b_{i-1})` — chain rule |
| Global→local signal | none (heads read `h` directly) | `BitLinear(d_model, P·d_local)` — positional slices |
| Byte conditioning | — | shift-right embedding, additive |
| FFN ratio | — | 4× (paper-canonical) |
| Params @ d_local=128 | `4 × 512 × 256 ≈ 524K` | `~700K` (h_proj + attn + 4× FFN + byte_embed + out_proj) |
| Inference cost | 4 parallel logits | 4 sequential `generate_step` calls on a length-P sequence |
| UTF-8 handling | broken (continuation bytes independent) | correct |

**Config knobs** (in `BASE_CONFIG`, all opt-in or default-on):

| Flag | Default | Meaning |
|---|---|---|
| `use_local_decoder` | `True` | Master switch. `False` → fall back to 4 independent heads (decoder ablation). |
| `local_decoder_dim` | `128` | `d_local`. Bigger = more capacity per position. Stays L2-resident up to ~256. |
| `local_decoder_heads` | `4` | Attention heads for the local transformer. |
| `local_decoder_ff_ratio` | `4.0` | SwiGLU FFN width as a multiple of `d_local`. Paper uses 4×. |
| `local_decoder_fp16` | `False` | **Opt-in.** Keep the local decoder in FP16 even when the global model is ternary. Costs ~420 KB extra RAM at d_local=128; removes quantization noise from a narrow, shallow submodule where it hurts most. Useful as a separate ablation when d_local is small. |

Note: the `--legacy-heads` flag (pass `config['legacy_heads']=True`) allows loading old checkpoints with `byte_heads.{0..3}.*` weight keys without `LocalByteDecoder`. See the *Checkpoint compatibility* section below.

#### Paper conformance check

| Check against MegaByte (Yu et al., 2023) | Status |
|---|---|
| Factorization `∏ P(bᵢ \| ctx, b<i)` — chain rule, not independence | ✅ |
| Global→local via `W_proj(h_G) → P·D_L` with per-position reshape | ✅ |
| Shift-right byte input with zero-pad at position 0 | ✅ |
| FFN width 4× `d_local` | ✅ |
| Teacher forcing in training / autoregressive in inference | ✅ |
| Global model ≫ Local model in parameters | ✅ (50M vs ~700K ≈ 70×) |
| Patch size 4 | ✅ |

**Deliberate deviations** (design choices, not bugs):

- **SwiGLU instead of GELU** — universally accepted modern upgrade; used everywhere else in the model for consistency.
- **RMSNorm instead of LayerNorm** — cheaper, standard in modern LLMs; matches the rest of the CPU-1 stack.
- **1 local layer instead of 2** — at `seq_len = P = 4`, additional depth gives near-zero return. A second layer becomes worth trying only when `d_local ≥ 256`.
- **Ternary weights by default** — this is the CPU-1 thesis, not a MegaByte choice. The `local_decoder_fp16` flag exists precisely so the ternary-vs-FP16 tradeoff in this submodule can be measured as a separate ablation.

**Intentional omission**: there is no separate "first-byte head" reading `h` directly. Position 0 of the local decoder already receives `global_slice[0] + embed(zero_pad) + pos_embed[0]`, so its output is `P(b₀ | h)` — exactly what a dedicated head would produce, with no extra parameters.

In short: the architectural skeleton matches the paper. Remaining differences are modern activation/norm choices (SwiGLU, RMSNorm) or project-specific constraints (ternary). A reviewer familiar with MegaByte would recognise this as a faithful implementation.

### Progressive Design — 50M runs (run_01 to run_12, run_17, new_1)

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
          ↓ siblings of run_09 — not on the main chain
run_17: CPU-1 COMPLETE + TIDE exit                     ← +per-token early exit (post-train)
new_1:  CPU-1 COMPLETE + Bolmo integrated gating       ← boundary scorer inside patch embedding
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
- `run_09 → new_1`: Does embedding-level boundary gating beat the external BoundaryPredictor used in run_09?

### TIDE Early Exit (run_17)

Most tokens "converge" well before the final layer — their hidden representation stops changing significantly mid-network. **TIDE** (Token-Informed Depth Execution, see [papers/tide-token-informed-depth-execution-early-exit.pdf](papers/tide-token-informed-depth-execution-early-exit.pdf)) adds a tiny 2-layer MLP classifier per block that is calibrated **post-training** (2K wikitext samples, < 3 min) and predicts *"has this token converged?"* at each layer. Converged tokens skip all remaining layers. Zero base model retraining.

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

## Local Checkpoint Mapping

The checkpoint folders under `models/ablation_models/` hold the **untrained initial weights** (`checkpoint.pt`) generated by `zave_ablation_models.py`, plus the matching `run_XX_*.py` source file (tracked since commit `9a71cdb`). The disk sizes below are for those FP32 initial dumps — final packed sizes are listed in *Projected final checkpoint sizes* further down.

> **Numbering drift — disk folders vs current `ablation.py`.** The `run_XX_*.py` files and folders on disk were frozen at a previous revision where the linear chain was still `run_06 = cpu1_core`, `run_07 = cpu1_bolmo`, `run_08 = cpu1_deletegate`, `run_09 = folded_transformer`, `run_10 = cpu1_complete`, `run_11 = cpu1_pfnet`. The current `ablation.py` `RUN_CONFIGS` collapsed `cpu1_core` into `run_05` and shifted the rest by one. The *Progressive Design* tree and all other tables in this README reflect the **current** `ablation.py`; the disk-folder column below is the archival layout. The column *Current `ablation.py` run ID* shows where each archival folder maps after the shift (or `—` if the config no longer exists as a separate run). Either re-run `zave_ablation_models.py` to regenerate with the new numbering, or keep treating the folder names as frozen archival labels.

> **`small_*` folders** predate the unified numbering and map to `run_13`–`run_16` in the current `ablation.py` (no drift inside that block — see *10M Runs* below).

### 50M Runs (disk folders — archival numbering)

| Disk folder | Current `ablation.py` run ID | Architecture summary | Initial size (FP32) |
|---|---|---|---|
| `run_01_transformer_bpe_fp16/` | `run_01` | Transformer + BPE + FP16 | 738.84 MB |
| `run_02_transformer_byte_fp16/` | `run_02` | Transformer + Byte + FP16 | 146.91 MB |
| `run_03_mlgru_bpe_fp16/` | — (current `run_03` = MLGRU + **Byte** + FP16) | MLGRU + BPE + FP16 (archival) | 726.86 MB |
| `run_04_mlgru_byte_fp16/` | `run_03` | MLGRU + Byte + FP16 | 134.93 MB |
| `run_05_mlgru_byte_ternary/` | `run_04` | MLGRU + Byte + Ternary (no FPRes) | 134.94 MB |
| `run_06_cpu1_core/` | `run_05` | + FPResidual — CPU-1 core ⭐ | 135.49 MB |
| `run_07_cpu1_bolmo/` | `run_06` | + Bolmo (dynamic patches) | 135.50 MB |
| `run_08_cpu1_deletegate/` | `run_07` | + DeleteGate | 135.61 MB |
| `run_09_folded_transformer/` | `run_08` | Folded Transformer + Byte + Ternary | 146.91 MB |
| `run_10_cpu1_complete/` | `run_09` | CPU-1 COMPLETE (Bolmo + DeleteGate) ⭐ | 135.63 MB |
| `run_11_cpu1_pfnet/` | `run_10` | CPU-1 COMPLETE + PFNet | 137.16 MB |
| `run_400M_cpu1_kernel_optimal.py` | — (kernel-optimal scale-up, see *400M* section) | CPU-1 @ 1280 × 24 × 3328 | generator only — no checkpoint |

**Current-only runs with no archival folder yet** (generate via `zave_ablation_models.py` if needed):

| Current run ID | `name` | Architecture summary |
|---|---|---|
| `run_11` | `cpu1_decay_learned` | CPU-1 COMPLETE + per-channel learnable decay (RWKV/HGRN2-style, ~12 KB params) |
| `run_12` | `cpu1_decay_reset` | CPU-1 COMPLETE + periodic MLGRU state reset (Stuffed Mamba, 0 params) |
| `run_17` | `cpu1_tide_exit` | CPU-1 COMPLETE + TIDE per-token early exit (post-training calibration) |
| `new_1`  | `cpu1_bolmo_dynamic` | CPU-1 COMPLETE + Bolmo integrated gating inside `BolmoPatchEmbedding` |

### 10M Runs

| Run ID | Local folder | Architecture summary | Checkpoint size |
|--------|-------------|----------------------|-----------------|
| `run_13` | `small_01_cpu1_bpe/` | CPU-1 COMPLETE @ 10M + BPE (4K vocab) | 44.75 MB |
| `run_14` | `small_02_cpu1_byte/` | CPU-1 COMPLETE @ 10M + Byte ⭐ | 36.05 MB |
| `run_15` | `small_03_cpu1_complement/` | CPU-1 COMPLETE @ 10M + Complement (BPE→byte) | 36.05 MB |
| `run_16` | `small_04_cpu1_raw_bytes/` | CPU-1 COMPLETE @ 10M + Raw bytes (no teacher) | 36.05 MB |

**Why BPE checkpoints are so much larger:**
- `run_01` / `run_03` (BPE @ 50M): the Qwen2.5 vocabulary has 152K tokens → embedding `152K × 512` + head `512 → 152K` dominate the size (~730 MB).
- `run_13` (BPE @ 10M): uses a compact 4K-token vocabulary → embedding `4096 × 320` → only 44 MB.
- All byte-level models share a tiny vocab (256 bytes) → embedding is negligible; size is driven almost entirely by the `d_model²` backbone.

**Why adding features barely changes the size (archival `run_05` → `run_11` = MLGRU+byte+ternary → +PFNet):**
All extras (BoundaryPredictor ~2 KB, DeleteGate ~33 KB, PFNet 393K params ~1.5 MB) are orders of magnitude smaller than the ~50M-param backbone. The table below shows the cumulative overhead (disk-folder numbering):

| Addition | Extra params | Δ disk |
|----------|--------------|--------|
| FPResidual (fp_rank=4) | ~50K | +0.55 MB |
| Bolmo (BoundaryPredictor) | ~2 KB | +0.01 MB |
| DeleteGate scorer | ~33 KB | +0.12 MB |
| PFNet (pfnet_hidden=32, 12 layers) | 393K | +1.53 MB |

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
│  inference/benchmark.c      │  CPU-optimized generation
│  + inference/kernels.cpp    │  T-MAC LUT + layer pipeline (mode 4)
└─────────────────────────────┘
```

### Ablation Pipeline (50M × 13 runs on Lightning.ai T4)

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
python train.py --mode ablation --run run_15 --complement_ckpt models/ablation_models/small_01_cpu1_bpe/checkpoint.pt
python train.py --mode ablation --run run_16   # raw bytes, no teacher (~1h)
```

---

### Kaggle Workflow

The `kaggle/` directory contains scripts ready to run directly in Kaggle notebooks (T4 GPU):

| File | Purpose |
|---|---|
| `kaggle/cpu1-create-dataset.py` | Runs Qwen2.5-3B teacher (batch aligned for ~4GB VRAM free), builds and uploads the shared dataset to HF Hub |
| `kaggle/upload_dataset.py` | One-shot uploader for local shards → `Cukinator/cpu1-ablation-dataset` |
| `kaggle/create-dataset-metadata.json` | Kaggle kernel metadata (used by `kaggle kernels push`) |

All scripts read `HF_TOKEN` from Kaggle Secrets automatically. Training runs are launched from `train.py --mode ablation --run <run_id>` directly on the Kaggle VM after `pip install -r requirements.txt`. To push an updated kernel:

```bash
kaggle kernels push -p kaggle/  # reads create-dataset-metadata.json
```

---

## Inference

Production inference is the standalone C driver `inference/benchmark.c` linked against `inference/kernels.cpp` — there is no Python inference engine in the current tree. The kernels are ultra-optimized for the MLGRU architecture: OpenMP, explicit AVX2 + FMA SIMD, Deep Data Dependency Unrolling (DDU), `thread_local` scratch reuse (zero-allocation steady state), and a layer-pipeline mode that keeps each stage's weight slice warm in L2/L3.

The PyTorch forward pass in `model.py` remains the reference used by `train.py` and the `eval/` suite — it is never benchmarked for throughput, only for numerical correctness and ablation quality.

### CPU-1 170M Performance — Current C Benchmark Status

> Baseline measured on commit `478567a` using `inference/benchmark.c` standalone (`gcc -O3 -march=native -fopenmp`). Hardware: Intel i7-8750H (6C/12T, Coffee Lake, AVX2 + FMA, no AVX-512).

**Model under test:** CPU-1 170M — `d_model=1024`, `n_layers=14`, `d_ff=2688`, weights 2-bit packed (~40.4 MB), recurrent state `h_all` ~57 KB. All measurements from `inference/benchmark.c` standalone driver, AVX2 + FMA, no AVX-512.

**Mode descriptions:**

| Mode | Function | Threading model | What it represents |
|---|---|---|---|
| **0** | `forward_full_mad` (OMP) | 1 OMP `parallel` region per layer; threads split rows of each GEMV | Naive intra-layer tensor parallelism — pays a sync barrier per GEMV |
| **1** | per-layer dispatch | 4 OMP threads | Layer-by-layer call from C, OMP teams per kernel — high overhead |
| **2** | `forward_full_mad_st` | **1 thread** | Pure single-threaded reference. The 14 layers run one after another |
| **3** | `forward_full_mad_batch` | **4 threads × 4 streams**, each thread runs `_st` on its own patch | **Throughput-oriented:** 4 independent users in parallel, no sync. Memory-bound |
| **4** | `forward_full_mad_pipeline` ⭐ | **N threads, 1 stream**, layer-pipeline. N is configurable (default **6 stages** on i7-8750H to use all physical cores). DP-calibrated layer split, atomic ring-buffer handoff, cross-patch L3 prefetch, process-wide affinity pinning | **Latency-oriented:** maximise single-stream throughput for one user |

**Mode comparison — FULL PATCH** (1 global pass + 4× MegaByte LocalByteDecoder steps per patch). Best of 3 trials. Hardware: Intel i7-8750H 6C/12T, ~9 MB L3, dual-channel DDR4.

| Mode | Configuration | Total pps | Per-stream pps | vs Mode 2 (per-stream) |
|---|---|---:|---:|---:|
| 2 — single-thread (ST) | 1 thread | 188 | **188** | 1.00× |
| 3 — batch (4 threads) | 4 threads, batch=4 | 578 | 144 | 0.77× |
| **3 — batch (6 threads)** | 6 threads, batch=6 | **726** | 121 | 0.64× |
| **4 — pipeline (6 stages, calibrated)** ⭐ | 6 threads, chunk=128, pin=1, --calibrate 1 | **521** | **521** | **2.77×** |

> **Single-user latency**: mode 4 wins **4.3× per stream** vs mode 3 (6t) — 521 vs 121 pps.
> **Aggregate multi-user throughput**: mode 3 (6t) wins **1.4× total** vs mode 4 — 726 vs 521 pps.

### What "full patch" includes

Each patch of the byte-level model produces 4 bytes:
1. **Global pass** — 14-layer transformer backbone (forward through all `LayerDataMAD`). 40.4 MB of 2-bit weights.
2. **4× LocalByteDecoder steps** — autoregressive byte generation inside the patch using the MegaByte local decoder (d_local=128, d_ff=512, n_heads=4, ~1 MB weights). Each step does `h_proj → causal-attention → SwiGLU FFN → out_proj[256]`.

The 4 local steps are **sequential within a patch** (each byte depends on the previous), but **parallel across patches**. Mode 4 launches an OMP loop over the 128 patches in the chunk to parallelise the local decoder across all 6 cores.

### Mode 4 evolution (gains from successive optimizations)

| Mode 4 variant | pps single-stream | vs original |
|---|---:|---:|
| Original (4 stages, no calibration, no L3 prefetch) | 392 | 1.00× |
| 4 stages, calibrated split | 470 | 1.20× |
| 6 stages, even split | 408 | 1.04× |
| **6 stages, calibrated split + cross-patch prefetch + process affinity** | **521** | **1.33×** |

**Why mode 4 uses less RAM bandwidth than mode 3 with the same cores:** in mode 3, every thread loads the full 40 MB of weights every patch → 6× redundant traffic. In mode 4, each thread owns only ~7-9 MB (its layer slice), and cross-patch prefetch keeps that slice warm in L2/L3. So mode 4 trades higher single-stream throughput for lower aggregate throughput — a different point on the same Pareto curve.

**Pipeline (mode 4) optimizations applied:**

1. **6-stage pipeline using all 6 physical cores** (was 4 stages). Raises the throughput ceiling because the slowest stage now contains 2-3 layers instead of 4. Cap: `max(stage_cost) ≈ 1.21 ms` → **~830 pps** theoretical, **~555 pps** measured.
2. **Cost-balanced layer split via runtime calibration + DP partition.** A startup phase measures per-layer wall-time (~50 iterations each, on a single thread, warmed cache). A linear-partition DP then assigns layers to stages so that `max(stage_cost)` is minimized. For 14 layers / 6 stages, the calibrator typically picks `[3, 2, 2, 3, 2, 2]` instead of the naive `[3, 2, 2, 2, 2, 3]`. **+33% throughput** vs uniform split.
3. **Cross-patch L3 prefetch.** While each thread processes the *last* layer of patch N, it issues `_mm_prefetch` (T2 hint) for its own *first* layer's `W_forget` weights, so they are warm in L2/L3 when patch N+1 starts on that stage. Mitigates L3 thrashing (working set 40 MB > L3 size 9 MB).
4. **Process-wide affinity pinning.** `SetProcessAffinityMask` restricts the entire process to the chosen physical cores (logical CPUs 0,2,4,6,8,10), preventing Windows from scheduling background threads on the HT siblings of pipeline cores. Reduces p95 latency.
5. **Cached partition.** The calibrated split is computed once and reused across batches via a static cache keyed on `(n_stages, n_layers)`.

**Internal structure:**

- Synchronization: `std::atomic<uint64_t>` produced/consumed counters with `_mm_pause()` busy-spin
- Inter-stage handoff: 4-slot ring buffer (`PipeRing`), each slot 64-byte aligned (no false sharing)
- Per-thread scratch (`PipeScratch`, ~80 KB): all RMSNorm/GEMV/SiLU buffers, separate from TLS-based helpers
- `h_all` recurrent state: each thread owns a disjoint slice of layers → no cross-thread writes

**When to use which mode (full-patch pps including local decoder):**

| Workload | Best mode | Single-user pps | Aggregate pps |
|---|---|---:|---:|
| 1 active user, latency-sensitive (chat) | **Mode 4** (pipeline, 6 stages, calibrated) | **521** | 521 |
| 6+ concurrent users (batch serving) | **Mode 3** (6-thread batch) | 121 | **726** |
| 4 concurrent users | Mode 3 (4-thread batch) | 144 | 578 |
| Reference / debugging | Mode 2 (ST) | 188 | 188 |

```bash
# Compile (Windows MSYS / Linux / WSL):
g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math \
    -std=c++11 -x c++ -o benchmark.exe \
    inference/benchmark.c inference/kernels.cpp -lm

# Mode 3 — max aggregate throughput (6-stream batch, RAM-saturated):
./benchmark.exe --mode 3 --batch 6 --patches 100 --warmup 30 --threads 6 --local 1

# Mode 4 — recommended single-stream pipeline (full patch with local decoder):
./benchmark.exe --mode 4 --patches 80 --warmup 30 \
                --chunk 128 --pin 1 --stages 6 --calibrate 1 --local 1

# Strip the local decoder (global-only — useful for kernel-only profiling):
./benchmark.exe --mode 4 --patches 80 --warmup 30 --stages 6 --calibrate 1 --local 0
```

**CLI flags:**

| Flag | Default | Meaning |
|---|---|---|
| `--stages N` | 6 | Number of pipeline stages (= worker threads) for mode 4 |
| `--chunk N` | 128 | Patches per mode-4 pipeline launch (amortizes thread spawn) |
| `--pin 0\|1` | 1 | Pin pipeline threads to physical cores + set process affinity |
| `--calibrate 0\|1` | 1 | Run DP-partition calibration on first launch (~10 ms one-shot) |
| `--local 0\|1` | 1 | Include MegaByte LocalByteDecoder (4× `local_decode_step` per patch) |
| `--threads N` | auto | Number of OMP threads (mode 0/1/3) |
| `--batch N` | 1 | Batch size for mode 3 |
| `--d_model N` | 1024 | Override `d_model` (≤ 2048, multiple of 64 recommended) |
| `--d_ff N` | 2688 | Override `d_ff` (≤ 5120, multiple of 64 recommended) |
| `--n_layers N` | 14 | Override layer count |
| `--fp_rank N` | 8 | FPResidual rank |
| `--pfnet N` | 64 | PFNet hidden dim |

---

### 400M kernel-optimal config — same kernel, scaled up

The same kernel handles a **~430M parameter** ablation variant
([models/ablation_models/run_400M_cpu1_kernel_optimal.py](models/ablation_models/run_400M_cpu1_kernel_optimal.py))
designed so that all dims are kernel-friendly:

| Hyperparameter | 170M (production) | 400M (kernel-optimal) | Notes |
|---|---:|---:|---|
| `d_model` | 1024 | **1280** | Multiple of 64, ≤ MAX_D_MODEL=2048 |
| `n_layers` | 14 | **24** | 24/6 = 4 layers per pipeline stage exact |
| `d_ff` | 2688 | **3328** | Multiple of 64, ≤ MAX_D_FF=5120 |
| `fp_rank` | 8 | 8 | unchanged |
| `pfnet_hidden` | 64 | 64 | unchanged |
| LocalDecoder dim | 128 | **192** | 3 × 64 head_dim |
| Total params | ~170M | **~430M** | 2.5× scale |
| 2-bit weight bytes | 40 MB | **107 MB** | 2.7× scale |

The DP calibrator confirms a perfect split for 24 layers / 6 stages: `[4, 4, 4, 4, 4, 4]` — every stage carries identical work.

**Benchmark on 400M (full patch, MegaByte LocalDecoder ON, best of 3, same i7-8750H):**

| Mode | Configuration | Total pps | Per-stream pps | vs Mode 2 |
|---|---|---:|---:|---:|
| 2 — single-thread (ST) | 1 thread | 73 | 73 | 1.00× |
| 3 — batch (6 threads) | 6 threads, batch=6 | **270** | 45 | 0.62× |
| **4 — pipeline (6 stages, calibrated)** ⭐ | 6 threads, chunk=64, pin=1 | **193** | **193** | **2.65×** |

> Mode 4 single-stream on 400M (193 pps) ≈ mode 2 single-stream on 170M (188 pps).
> **Scaling is consistent**: at 2.7× the model size, mode 4's pipeline efficiency stays at ~45% of ideal speedup (170M: 521/188 = 2.77×; 400M: 193/73 = 2.65×).
> Per-user advantage of mode 4 vs mode 3: 193 / 45 ≈ **4.3× per stream**, identical ratio to the 170M case.

```bash
# Mode 4 on the 400M kernel-optimal config:
./benchmark.exe --d_model 1280 --n_layers 24 --d_ff 3328 --fp_rank 8 \
                --mode 4 --patches 30 --warmup 10 \
                --chunk 64 --pin 1 --stages 6 --calibrate 1 --local 1

# Mode 3 reference on 400M:
./benchmark.exe --d_model 1280 --n_layers 24 --d_ff 3328 --fp_rank 8 \
                --mode 3 --batch 6 --threads 6 --patches 50 --warmup 15 --local 1
```

---

## Project Structure

```
1.58bits/
│
│  Core
├── model.py                 # Full CPU-1 architecture: MLGRU + PFNet + FPResidual +
│                            #   BoundaryPredictor + DeleteGate + EmbeddingAligner +
│                            #   MegaByte LocalByteDecoder
├── train.py                 # Training loop — two modes:
│                            #   default: 170M production model
│                            #   --mode ablation --run run_XX: ablation run
│                            #   (also hosts the `from hf_mount import mount_bucket`
│                            #    lazy-load path used via --hf-mount on Kaggle VMs)
├── ablation.py              # Ablation system: BASE_CONFIG + RUN_CONFIGS + SMALL_RUN_CONFIGS,
│                            #   configurable model variants, analysis + LaTeX tables
├── create_dataset.py        # Full dataset pipeline: teacher logprob extraction,
│                            #   BPE→byte marginalization, HuggingFace Hub upload
├── main_ligtning_studio.py  # Lightning.ai studio entry point (sets up env + launches train)
│
│  Inference (C / C++ only — no Python engine)
├── inference/
│   ├── benchmark.c          # Standalone driver — allocates random ternary weights and
│   │                        #   times the global pass + MegaByte LocalByteDecoder per patch
│   ├── kernels.cpp          # All AVX2 + FMA kernels: GEMV, RMSNorm, SiLU, MLGRU, PFNet,
│   │                        #   FPResidual, forward_full_mad_{st,batch,pipeline}, local_decode_step
│   ├── benchmark.exe        # Compiled driver (gitignored — regenerate with the command in links.md)
│   └── links.md             # Build instructions + paper references
│
│  Kaggle
├── kaggle/
│   ├── cpu1-create-dataset.py        # Kaggle script: build + upload shared dataset (T4)
│   ├── upload_dataset.py             # Local → Hub uploader for pre-built shards
│   └── create-dataset-metadata.json  # Kaggle kernel metadata
│
│  Evaluation (single-engine, multi-architecture — see "Evaluation Suite" below)
├── eval/
│   ├── run_eval.py          # Unified runner — 14 checks across 5 modules
│   ├── run_all_models.py    # Batch-runs `run_eval.py` over every model in models/
│   ├── helpers.py           # Creates the shared engine + hardware introspection
│   ├── eval_correctness.py  # forward_shapes, probs, determinism, state_prop, ternary_weights, state_size
│   ├── eval_performance.py  # throughput+latency+jitter (fused), context_scaling, ram
│   ├── eval_quality.py      # generation+utf8 (fused), perplexity
│   ├── eval_dataset.py      # teacher_coverage, teacher_entropy (no model needed)
│   └── eval_external.py     # llama-bench native benchmark on GGUFs
│
│  Models
├── models/
│   ├── cpu1-170M/               # CPU-1 170M checkpoint — in training
│   ├── LFM2.5-350M/             # LFM2.5-350M (HF safetensors) — mlGRU reference
│   ├── gpt2-medium/             # GPT-2 medium (HF safetensors) — transformer baseline
│   └── ablation_models/         # One sub-folder + `.py` per run (run_01…run_11 archival,
│                                #   run_400M_cpu1_kernel_optimal.py, small_01…small_04)
│                                #   + `zave_ablation_models.py` regenerator.
│                                #   Current-only runs (run_12, run_17, new_1) have no
│                                #   sub-folder yet — regenerate if needed.
│
│  Standalone scripts
├── benchmark_optimized.exe  # Pre-built C benchmark (6-stage pipeline, calibrated)
├── benchmark_test.exe       # Smaller smoke-test binary
│
│  Results & Reference
├── resultados/              # Benchmark reports (markdown + json) per model/thread config
├── papers/                  # Reference PDFs (see table above)
└── Prompts/                 # Design documents per pipeline stage (00_master_pipeline.md …)
```

---

## Evaluation Suite

### Design

14 checks across 5 modules. All inference goes through a single engine created in `eval/helpers.py::create_engine` — the engine is instantiated **once** and reused across all checks, so the model is loaded exactly once per run. This replaced an older 26-file-per-test layout (`test_01_forward.py` … `test_26_benchmark_llama_cpp.py`) that reloaded the model for every check and duplicated benchmark loops across files.

The suite is **multi-architecture**: it auto-detects the model type via `ModelInfo` and adapts each check. Ternary weight checks are skipped for FP16 models, state propagation is skipped for Transformers, generation quality is skipped for BPE models, etc.

```bash
# Basic (11 checks, no dataset needed)
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt

# With dataset (adds perplexity + dataset quality checks)
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt --dataset path/

# Skip external benchmarks (llama.cpp)
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt --skip-external

# Against an ablation snapshot
python eval/run_eval.py --model models/ablation_models/run_10_cpu1_complete/checkpoint.pt
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

## llama.cpp Optimization Analysis — LFM2.5-350M on i7-8750H

Systematic benchmarking of the **LFM2.5-350M** model running via llama.cpp on consumer CPU hardware (Intel i7-8750H, 6C/12T, Coffee Lake). llama.cpp is consumed externally through the `llama-cpp-python` wheel pinned in `requirements.txt` (AVX2 build) — there is no `llamacpp/` source checkout inside this repository. The `eval/eval_external.py` module invokes `llama-bench` directly on GGUFs.

### Build Optimization

For reference, if you clone llama.cpp separately and want to match the settings used for the numbers below:

```
cmake -G Ninja -B build_optimized \
  -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_C_FLAGS="-march=native -O3" \
  -DCMAKE_CXX_FLAGS="-march=native -O3" \
  -DGGML_NATIVE=ON
```

`-march=native` activates **AVX2 + FMA** on this CPU (Coffee Lake does **not** support AVX-512). The pip wheel (`llama-cpp-python==0.3.19` via `https://abetlen.github.io/llama-cpp-python/whl/cpu-avx2`) is built with equivalent flags, which is why the numbers in the eval reports under `resultados/` match a from-source build.

### Runtime Configurations Tested

| Config | Threads | mmap | flash-attn | KV cache | Priority | Batch |
|---|---|---|---|---|---|---|
| ULTRA | 1 | OFF | ON | q8_0 | REALTIME | 2048 |
| OPTIMAL | 1 | OFF | OFF | q8_0 | HIGH | 2048 |
| BASELINE | 1 | default | OFF | f16 | default | default |

Methodology: 900 tokens prefill + 100 tokens decode × 3 repetitions (matching [LiquidAI's official benchmark](https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF)).

### Benchmark Results

Per-model reports are in `resultados/` (`gpt2-medium_q{4,8}_{1,6}t.md`, `checkpoint_{1,6}t.md`).

| Model | Quant | Decode (tok/s) | Config | Notes |
|---|---|---|---|---|
| **gpt2-medium (355M)** | Q4_K_M | see `resultados/gpt2-medium_q4_6t.md` | llama.cpp, 6 threads | Transformer — parallelizable |
| **LFM2.5-350M** | Q4_K_M | **~41** | llama.cpp, 1 thread, ULTRA | mlGRU — sequential decode |

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

**1. mlGRU sequential dependency.** The LFM2.5 architecture uses mlGRU (MatMul-free GRU) for token mixing. Each decode step depends on the hidden state from the previous step — this is inherently sequential and cannot be parallelized. By contrast, transformer models (like `gpt2-medium`) with KV-cache allow parallel computation within each decode step. This architectural difference explains the bulk of the gap between the GPT-2 baseline and LFM2.5-350M despite the similar parameter count.

**2. Memory bandwidth ceiling.** Decode throughput is memory-bandwidth-bound (streaming weights from RAM each token). This system has **mismatched DIMMs** (2667 MHz + 2400 MHz), forcing the memory controller to operate at the slower speed with potential additional timing penalties. Theoretical peak: ~38.4 GB/s. The AMD Ryzen AI Max 395+ has ~128 GB/s (DDR5) — a **3.4× advantage** in raw bandwidth, compounded by AVX-512 (2× SIMD width) and superior Zen5 IPC.

**Why do Apple chips achieve 300+ tok/s?** They run on **GPU** (Metal) with **Mirai** framework using **bf16** weights — not llama.cpp on CPU. The M5 Max has ~546 GB/s bandwidth (14× more than this system). The sequential nature of mlGRU remains, but each sequential step executes in microseconds instead of milliseconds due to the massive bandwidth advantage.

### Key Finding

> The AVX2-prebuilt pip wheel already runs with equivalent `-march=native` kernels — **AVX2 was never missing**. The performance ceiling is imposed by the mlGRU architecture and memory bandwidth, not by compiler flags or runtime configuration.

### Paths to 100+ tok/s

| Strategy | Expected Impact | Effort |
|---|---|---|
| Re-quantize to Q4_0 | +5–15% (simpler kernel, less compute overhead) | Run `llama-quantize` |
| Use llamafile | +10–30% prefill, marginal decode gain (tinyBLAS kernels) | Download binary |
| Match RAM DIMMs (2×2667) | +10% bandwidth (eliminate timing penalty) | Hardware upgrade |
| **All combined** | **~50–70 tok/s estimate** on this hardware | — |

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

Key packages (see `requirements.txt` for the full list and exact pins):
```
torch==2.6.0+cu124             # CUDA 12.4 build (compatible with CUDA 12.6 driver)
transformers>=4.47.0
datasets>=3.2.0
huggingface_hub>=0.27.0
bitsandbytes>=0.45.0           # 4-bit NF4 for the Qwen2.5-3B teacher
torchao>=0.7.0                 # FP8 inference (Float8DynamicActivationFloat8WeightConfig)
llama-cpp-python==0.3.19       # CPU-AVX2 wheel — consumed by eval/eval_external.py
lightning>=2.4.0
kaggle>=1.6.0
pyarrow>=18.0.0
numpy>=1.26.0
psutil>=5.9.0
tqdm>=4.66.0
```

---

## Status

### Implemented

- **Architecture** — 170M model: d_model=1024, 14 layers, MLGRU + PFNet + fully ternary (`model.py`); MegaByte LocalByteDecoder for intra-patch byte autoregression
- **Extended techniques (v3)** — BoundaryPredictor (Bolmo), DeleteGate (MrT5), EmbeddingAligner (Distilling Token-Trained Models into Byte-Level Models), two-phase training
- **Dataset pipeline** — Teacher logprob extraction + BPE→byte marginalization + HF Hub upload (`create_dataset.py`)
- **HF-mount streaming** — `train.py --hf-mount` lazy-loads shards from the Hub without the ~5 GB full download (module imported on demand; no local `hf_mount.py` yet)
- **Training** — AMP, gradient accumulation, distillation loss, both main and ablation modes in one script (`train.py`)
- **Ablation system** — 14 progressive runs (run_01–run_12, run_17 and new_1 at 50M) + 4 scale runs (run_13–run_16, 10M) + `run_400M` kernel-optimal variant (`ablation.py`, `models/ablation_models/`)
- **C inference kernels** — `inference/kernels.cpp` + `inference/benchmark.c` standalone driver; 6-stage DP-calibrated layer pipeline (mode 4) and 6-stream batch (mode 3)
- **Evaluation suite** — 14 checks in 5 modules, multi-architecture adaptive, single engine load, fused benchmarks (`eval/`)
- **Kaggle scripts** — Ready-to-run notebooks for dataset creation on T4 (`kaggle/`)

### Models

| Model | Size | Architecture | Status |
|---|---|---|---|
| `models/cpu1-170M/` | ~170M params (~44 MB 2-bit packed) | MLGRU + Byte + Ternary + FPResidual + PFNet + MegaByte LocalDecoder | 🔄 Training |
| `models/LFM2.5-350M/` | HF safetensors | mlGRU (LiquidAI) | ✅ Benchmark reference (llama.cpp Q4 via `eval_external`) |
| `models/gpt2-medium/` | HF safetensors | Transformer | ✅ Baseline reference (`resultados/gpt2-medium_q{4,8}_{1,6}t.md`) |
| `models/ablation_models/run_*` | initial checkpoints | Per-run architecture | ✅ Generated by `zave_ablation_models.py` |

### Pending

- Run the 14 ablation experiments (run_01–run_12, run_17, new_1) + 4 scale runs (run_13–run_16) on Kaggle T4 and collect metrics
- Scale winning ablation config to 170M and train on full dataset
- ~~T-MAC C++ kernel integration~~ ✅ **Done** — current numbers in *CPU-1 170M Performance* (mode 4: 521 pps, mode 3 6t: 726 pps)
- ~~C benchmark for 400M kernel-optimal config~~ ✅ **Done** — 193 pps (mode 4), 270 pps (mode 3)

---

## Projected final checkpoint sizes

The initial checkpoint under `models/ablation_models/run_XX/` is the **untrained** model serialized in FP32. After training and the final-packing pass (BF16 for non-ternary weights, 2-bit packed for ternary weights — see *3. Checkpoint Formats*) the on-disk size drops substantially, especially for ternary runs:

| Run ID | Weight type | Initial (FP32) | Final (estimated) |
|---|---|---:|---:|
| run_01 | FP16 (BPE vocab 152K) | 738.84 MB | ~369 MB ¹ |
| run_02 | FP16 (Byte) | 146.91 MB | ~73 MB |
| run_03 | FP16 (BPE vocab 152K) | 726.86 MB | ~363 MB ¹ |
| run_04 | FP16 (Byte) | 134.93 MB | ~67 MB |
| run_05 | Ternary | 134.94 MB | ~14 MB |
| run_06 | Ternary + FPRes | 135.49 MB | ~14 MB |
| run_07 | Ternary + FPRes + Bolmo | 135.50 MB | ~14 MB |
| run_08 | Ternary + FPRes + DeleteGate | 135.61 MB | ~14 MB |
| run_09 | Ternary (Folded Transformer) | 146.91 MB | ~15 MB ² |
| run_10 | Ternary + FPRes + Bolmo + DG | 135.63 MB | ~14 MB |
| run_11 | Ternary + PFNet | 137.16 MB | ~14 MB |
| run_13 | Ternary (BPE 4K) | 44.75 MB | ~4.6 MB |
| run_14 | Ternary (Byte) | 36.05 MB | ~3.7 MB |
| run_15 | Ternary (Complement) | 36.05 MB | ~3.7 MB |
| run_16 | Ternary (Raw bytes) | 36.05 MB | ~3.7 MB |

¹ FP16-to-FP16 halving: the checkpoint is already FP16 on disk but the initial dump uses the FP32 master copy. The final dump drops optimizer state and re-packs weights as raw FP16 — roughly ½ of the initial size.
² Folded Transformer has a slightly larger non-ternary portion (attention norms + bounded-window buffers) than the MLGRU runs, hence ~15 MB rather than ~14 MB.

---

## Checkpoint compatibility

> [!WARNING]
> Checkpoints trained **before** the `LocalByteDecoder` migration have weight keys
> `byte_heads.0.*` … `byte_heads.3.*`.  Loading them into the new `CPU1ModelV2`
> (which has `local_decoder.*`) will raise a `RuntimeError: unexpected key`.

To load an old checkpoint without migrating it, pass `legacy_heads=True` in the
model config:

```python
from model import CPU1ModelV2
model = CPU1ModelV2({'legacy_heads': True})
ckpt  = torch.load('old_checkpoint.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
```

This restores the original 4-head architecture.  Generation still works; the
byte-level factorization reverts to the (independent) old form.

**Weight key mapping** (old → new):

| Old key (legacy_heads) | New key (default) |
|---|---|
| `byte_heads.0.proj.weight` … `byte_heads.3.proj.weight` | `local_decoder.h_proj.weight`, `local_decoder.q_proj.weight`, … |
| `byte_heads.{i}.output_scale` | `local_decoder.out_scale` |

There is no automated migration script because the architectures are genuinely
different (4 scalars → 1 full local transformer).  New checkpoints trained with
the default config will always use `local_decoder.*`.

---

## Paper conformance check

Verifies that `LocalByteDecoder.forward()` (teacher-forced, parallel) and
`LocalByteDecoder.generate_step()` (autoregressive, sequential) produce
**identical logits** for every byte position in every patch.

```bash
python model.py   # runs all self-tests including test_local_decoder_parity()
```

Expected output:

```
[OK] LocalByteDecoder training↔inference parity: diff=0
```

The test constructs random global hidden states `h ∈ ℝ^{B×S×d_model}` and
random target byte tensors, then asserts `max|forward_logit[k] - step_logit[k]| < 1e-4`
for each patch position `k ∈ {0, 1, 2, 3}`.  The tolerance accounts for
float32 reorder differences; the true diff is typically `< 1e-6`.

---

## License

MIT