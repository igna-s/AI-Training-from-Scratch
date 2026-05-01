# CPU-1 — Byte-Level Language Model with Ternary Weights

> A ~170M parameter byte-level language model trained via knowledge distillation from Qwen2.5-3B, using fully ternary weights {-1, 0, 1}, MatMul-free token mixing (MLGRU), and zero matrix multiplications at inference (T-MAC LUT). Optimized for consumer CPU (4 cores, 16GB RAM, 32MB L3).

---

## Architecture (v2 — 170M)

| Hyperparameter | Value |
|---|---|
| `d_model` | 1024 |
| `n_layers` | 14 |
| `d_ff` | 2730 (PyTorch `utilities.D_FF`, ~8/3 × 1024) / 2688 (C benchmark default, rounded to multiple of 64) |
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
| **PyTorch** (ablation / training) | Modules in `utilities.py` (BitLinear, MLGRU, LocalByteDecoder, …) driven by `train_final.py` / `train_ablation.py` and the `eval/` suite | All ablation runs — fair comparison, same framework for every architecture |
| **C++ LUT** (production) | `inference/kernels.cpp` + `inference/benchmark.c` standalone driver | Production benchmark with the final chosen model |

> **The C++ LUT kernels are NOT used in any ablation run.** All ablation BPB/perplexity numbers are measured in pure PyTorch via the modules in `utilities.py`. This is by design: running the C++ LUT only on ternary MLGRU models (run_05 onward) while the FP16 Transformer baselines (run_01–run_03) stay in PyTorch would conflate kernel speed with architecture quality — making the comparison unfair.

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
| [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531) | KL + CE distillation loss, T=4.0 | ✅ `train_final.py` distillation mode |

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

## Ablation Study — 12 Progressive Runs (50M) + 4 Scale Runs (10M)

The ablation tells a story: starting from a standard FP16 Transformer baseline, we add **one CPU-1 design choice per run** until we reach the full architecture. Each step isolates the contribution of a single component.

All runs: CE-only training from scratch under **Chinchilla scaling** (`tokens ≈ 20 × N_params`), same seed (42) as `create_dataset.py`. `total_steps` is recomputed per run from the actual parameter count inside `train_run` — see *Chinchilla & Checkpointing* below.

**Dataset routing** (matches the upstream pipeline in `create_dataset.py`):

| Runs | Dataset | Why |
|---|---|---|
| `run_01` … `run_12` (50M) | `HuggingFaceFW/fineweb` `sample-10BT`, streamed | Chinchilla 50M needs ~1B tokens — the pre-processed `Cukinator/cpu1-ablation-dataset` only holds ~37.5M, sized for the 10M scale. The 50M runs use `FineWebStreamingDataset` to consume FineWeb directly with the **same params `create_dataset.py` uses** (seed=42, seq_len=5000, patch_size=4, max 700 BPE tokens / 2800 bytes per doc, 95/5 train/val split modulo doc index 20). On-the-fly UTF-8 byte packing for byte runs, on-the-fly Qwen tokenization for `run_01` (clamped to its 16K vocab cap). |
| `run_13`, `run_14`, `run_15` (10M) | `Cukinator/cpu1-ablation-dataset` (2 subsets) | The pre-processed dataset has two subsets that match each run: `bpe_tokenized/` for `run_13`, `byte_marginalized/` for `run_14`/`run_15`. 10M × 20 = 200M tokens → ~5× cycling over the 37.5M tokens. |
| `run_16` (10M, "no teacher") | `HuggingFaceFW/fineweb`, streamed | Same path as the 50M runs — the run *is* the "skip the Qwen teacher" ablation. |

The 50M track is the linear chain `run_01 … run_12`. See `RUN_CONFIGS` in `train_ablation.py` for the authoritative list.

> **Parked for follow-up papers** (not in this ablation):
> - **TIDE early exit** — post-training calibration of per-layer exit classifiers on a frozen CPU-1 checkpoint. Inference-only optimization, decoupled from architectural choices, so it belongs in a deployment paper rather than this study.
> - **Bolmo integrated boundary gating** — embedding-level scorer that replaces the external `BoundaryPredictor`. Worth a dedicated tokenizer-paper comparison vs Pagnoni et al., but mutually exclusive with run_06/09's Bolmo formulation.

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

### Progressive Design — 50M runs (run_01 to run_12)

```
run_01: Transformer  + BPE   + FP16                    ← absolute baseline (16K vocab cap)
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
          ↓ add nonlinear residual / study memory dissipation
run_10: CPU-1 COMPLETE + PFNet (pfnet_hidden=32)       ← +cache-resident MLP
run_11: CPU-1 COMPLETE + per-channel decay (RWKV/HGRN2) ← +learned forgetting (~12KB)
run_12: CPU-1 COMPLETE + periodic state reset           ← structural forgetting, 0 params
```

`run_01` caps the BPE vocab at **16 384** (top-K by frequency, dataset IDs above the cap collapse to OOV in `BPEDataset`); the full 152 K Qwen vocab would consume ~78 MB just on the embedding and overflow the 50M budget on its own.

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

### Chinchilla & Checkpointing

Each run trains to its own Chinchilla-optimal horizon: `total_steps = (20 × N_params) ÷ (batch × accum × seq_len)`, recomputed inside `train_run` once `build_ablation_model` reports `N_params`. Both `warmup_steps` (5% of horizon) and `save_every_steps` (5% of horizon) are derived from this rescaled `total_steps`, so every run gets the **same warmup ratio and the same checkpoint cadence** regardless of param count. Override with `--total_steps N` to fall back to a fixed budget (smoke tests) or set `chinchilla=False` in the run config.

Checkpoints upload to `Cukinator/cpu1-ablation-checkpoints` on HF Hub via `save_ablation_checkpoint`, named `ablation/checkpoint_<run>_step<N>.pt`. The local file is rotated (only the latest step on disk) but every step is preserved on the Hub for later learning-curve fits and intermediate eval.

| Setting | Where | Default |
|---|---|---|
| `chinchilla` | `BASE_CONFIG` | `True` |
| `chinchilla_tokens_per_param` | `BASE_CONFIG` | `20` |
| `warmup_ratio` | `BASE_CONFIG` | `0.05` (5% of Chinchilla horizon) |
| `checkpoint_fraction` | `BASE_CONFIG` | `0.05` (≈ 20 checkpoints) |
| `checkpoint_repo` | `BASE_CONFIG` | `Cukinator/cpu1-ablation-checkpoints` |
| `seed` | `BASE_CONFIG` | `42` (matches `create_dataset.SEED`) |

Smoke-test sizes (verified):

| Run | N_params | Chinchilla steps | Warmup (5%) | Save every | Tokens / step |
|---|---|---|---|---|---|
| `run_01` (FP16 BPE 16K) | 54.7 M | 9 122 | 456 | 456 | 120 K |
| `run_05` (CPU-1 core) | 35.6 M | 5 926 | 296 | 296 | 120 K |
| `run_09` (CPU-1 COMPLETE) | 35.6 M | 5 932 | 296 | 296 | 120 K |
| `run_10` (+ PFNet) | 36.0 M | 5 999 | 299 | 299 | 120 K |
| `run_13` (10M BPE 4K) | 11.7 M | 1 953 | 97 | 97 | 120 K |
| `run_14` (10M byte) | 9.6 M | 1 600 | 80 | 80 | 120 K |

### Hyperparameter Audit — what's equal, what isn't, why

All ablation runs share the **same training recipe** so quality differences are attributable to architectural choices, not hyperparameter drift. Audited 2026-04-25 across the 16 active runs:

| Hyperparameter | Value | Equal across all runs? |
|---|---|---|
| `seed` | 42 | ✅ — matches `create_dataset.py:987` `SEED=42` |
| `dataset_repo` | varies by scale (see *Dataset routing* above) | 50M + run_16 → `HuggingFaceFW/fineweb` (streamed); 10M run_13/14/15 → `Cukinator/cpu1-ablation-dataset`. The two paths both follow `create_dataset.py` params (same seed, seq_len, patch_size, max-doc-tokens). |
| `lr` / `min_lr` | 3e-4 / 3e-5 | ✅ |
| `batch_per_gpu` × `accum` × `seq_len` | 24 × 1 × 5000 = 120 K tok/step | ✅ |
| `warmup_ratio` | 0.05 of `total_steps` | ✅ (was a bug — see below) |
| `chinchilla_tokens_per_param` | 20 | ✅ |
| `weight_decay_bitlinear` / `_heads` | 0.1 / 0.01 | ✅ |
| `grad_clip` | 1.0 | ✅ |
| `temperature` / `alpha_kl` / `alpha_ce` | 4.0 / 0.9 / 0.1 | ✅ (unused: `training_signal='ce_only'`) |
| `checkpoint_fraction` | 0.05 | ✅ |

**Two datasets, both anchored to `create_dataset.py`:**

- The **upstream raw corpus** is `HuggingFaceFW/fineweb` `sample-10BT` (10B tokens). `create_dataset.py:1225` streams it with `SEED=42` and runs it through Qwen2.5-3B to build the pre-processed dataset.
- The **pre-processed product** is `Cukinator/cpu1-ablation-dataset` (~37.5M tokens, two subsets: `byte_marginalized/`, `bpe_tokenized/`).
- 50M runs (`run_01` … `run_12`) and `run_16` consume the *raw upstream* directly via `FineWebStreamingDataset` — same SEED, same SEQ_LEN, same PATCH_SIZE, same per-doc cap as the build pipeline. Reading raw is necessary for them: Chinchilla at 50M needs ~1B tokens, and the pre-processed dataset doesn't have that many.
- 10M runs `run_13`/`run_14`/`run_15` consume the pre-processed product, picking the matching subset by tokenization.

**Intentional per-run overrides** (these *are* the ablation, not hyperparameter drift):

| Run | Override | Reason |
|---|---|---|
| `run_01` | `bpe_vocab_size=16384` | full Qwen 152K vocab → 78 MB embedding alone, busts 50M budget. Top-16K by frequency, rest collapses to OOV in `BPEDataset._clamp_ids`. |
| `run_08` | `window_size=128` | the run *is* "best CPU-friendly local-attention baseline" |
| `run_10` | `pfnet_hidden=32` | enables the component being ablated |
| `run_11` | `mlgru_per_channel_decay=True` | enables the component being ablated |
| `run_12` | `state_reset_period=512` | enables the component being ablated |
| `run_13` | `bpe_vocab_size=4096`, `use_boundary_predictor=False` | 10M budget cap; BPE has its own subword boundaries |
| `run_15` | `complement_from='run_13'` | warm-start checkpoint, by design |
| `run_16` | `dataset_repo='HuggingFaceFW/fineweb'`, `raw_bytes_only=True` | this run *is* the "no-teacher, raw bytes" ablation question |

**Bugs found and fixed during audit:**

1. **`run_01` BPE vocab overflow** — was defaulting to 152 064 tokens (Qwen2.5), which makes the embedding alone 78 M params and overflows the 50 M budget. Fixed: `bpe_vocab_size=16384` + clamp in `BPEDataset`.
2. **Warmup ratio drift under Chinchilla** — fixed `warmup_steps=500` against a Chinchilla-rescaled `total_steps` produced 5.5% warmup at `run_01` (9122 steps) but 12.8% at `run_13` (1953 steps). Replaced with `warmup_ratio=0.05` so every run gets the same 5% warmup regardless of horizon.
3. **`run_16` had no implementation backing** — its config declared `dataset_repo='HuggingFaceFW/fineweb'` and `raw_bytes_only=True`, but the original loader only knew how to read pre-processed Cukinator parquet shards. The same gap was about to bite the 50M runs, which can't fit in Cukinator under Chinchilla. Fixed by adding `FineWebStreamingDataset` (mirrors `create_dataset.py` params, now in `utilities.py`) and a routing branch in `load_ablation_dataset` keyed on `dataset_repo`. `run_16` now actually runs end-to-end on FineWeb; the 50M runs (`run_01` … `run_12`) inherit the same path via `BASE_CONFIG`.
4. **`run_16` Cukinator → FineWeb routing assumed by config but unused at training time** — the bug above also meant `run_16` would have silently fallen back to Cukinator's `byte_marginalized/` subset (defeating the whole "no teacher" question). Same fix as above.

**Parked for future papers** (deliberately not in this study, see also `train_ablation.py` header):
- **TIDE per-token early exit** (arXiv 2603.21365) — post-training calibration of per-layer exit classifiers on a frozen `run_09` checkpoint. Inference-FLOPs optimization, not an architectural choice. Belongs in a follow-up deployment paper.
- **Bolmo integrated boundary gating** — `BolmoPatchEmbedding` with embedding-level scorer, mutually exclusive with `run_06`/`run_09`'s external `BoundaryPredictor`. Worth a dedicated tokenizer-paper comparison vs Pagnoni et al.

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

The checkpoint folders under `models/ablation_models/` hold **untrained initial weights** (`checkpoint.pt` + `config.json`) — frozen archival dumps used for benchmarks pre-training. The disk sizes below are for those FP32 initial dumps; final packed sizes are in *Projected final checkpoint sizes* further down.

> **Numbering drift — disk folders vs current `train_ablation.py`.** The on-disk folders were frozen at a previous revision where the linear chain was still `run_06 = cpu1_core`, `run_07 = cpu1_bolmo`, `run_08 = cpu1_deletegate`, `run_09 = folded_transformer`, `run_10 = cpu1_complete`, `run_11 = cpu1_pfnet`. The current `RUN_CONFIGS` in `train_ablation.py` collapsed `cpu1_core` into `run_05` and shifted the rest by one. The *Progressive Design* tree and all other tables in this README reflect the **current** numbering; the disk-folder column below is the archival layout. The *Current run ID* column shows where each archival folder maps after the shift (or `—` if the config no longer exists as a separate run). Treat the folder names as frozen archival labels.

> **`small_*` folders** predate the unified numbering and map to `run_13`–`run_16` (no drift inside that block — see *10M Runs* below).

### 50M Runs (disk folders — archival numbering)

| Disk folder | Current run ID | Architecture summary | Initial size (FP32) |
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

**Current-only runs with no archival folder yet** (build manually with `from models.ablation_models import build_run; build_run("run_11")`):

| Current run ID | `name` | Architecture summary |
|---|---|---|
| `run_11` | `cpu1_decay_learned` | CPU-1 COMPLETE + per-channel learnable decay (RWKV/HGRN2-style, ~12 KB params) |
| `run_12` | `cpu1_decay_reset` | CPU-1 COMPLETE + periodic MLGRU state reset (Stuffed Mamba, 0 params) |

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
│  train_final.py             │  Student training with distillation
│  (AMP, gradient             │  → Checkpoints → HuggingFace Hub
│   accumulation)             │  Imports primitives + datasets from utilities.py
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  inference/benchmark.c      │  CPU-optimized generation
│  + inference/kernels.cpp    │  T-MAC LUT + layer pipeline (mode 4)
└─────────────────────────────┘
```

### Ablation Pipeline (50M × 12 runs on Lightning.ai T4)

#### Lightning Studio Setup — Files to Upload

Minimum set of files needed to train any ablation run on a Lightning Studio T4 instance:

| File | Purpose |
|---|---|
| `utilities.py` | Shared primitives (BitLinear, MLGRU, …) + training infrastructure (datasets, loss, LR, checkpointing) |
| `train_ablation.py` | Ablation configs + `AblationModel` + `train_run` + multi-GPU orchestration (`--orchestrate`) |
| `requirements.txt` | Python dependencies |

Everything else is remote or not required:
- **Dataset** — streamed from `Cukinator/cpu1-ablation-dataset` (or `HuggingFaceFW/fineweb` for the 50M chain) on HF Hub; use `--hf-mount` to avoid downloading
- **Credentials** — set `HF_TOKEN` as a Lightning Secret (Settings → Secrets), not via `.env`
- `train_final.py` — only needed for the production 170M training, not the ablation chain
- `create_dataset.py` — dataset already built and on HF Hub
- `inference/`, `kaggle/`, `eval/` — not needed at training time

```bash
# On the Lightning VM after uploading the files above:
pip install -r requirements.txt

# A) Single ablation run with lazy dataset streaming:
python train_ablation.py --run run_10 --hf-mount

# B) Full ablation in parallel groups (one process per GPU):
python train_ablation.py --orchestrate
```

#### Parallel groups — running 4 ablations simultaneously on a 4×T4 instance

`train_ablation.py` declares `PARALLEL_GROUPS: list[list[str]]`. Every sublist is a *group*: when invoked with `--orchestrate`, its runs launch concurrently as subprocesses of the same script (each spawned with `--run <name>`), one per GPU pinned via `CUDA_VISIBLE_DEVICES`. Groups themselves run sequentially. The launcher detects DeleteGate runs (`run_07`, `run_09`) and chains Phase 1 → Phase 2 inside the same GPU slot, and orders `run_15` (warm-start from `run_13`) into a later group.

Default groups:

| Group | Runs | Notes |
|---|---|---|
| 0 | `run_01 run_02 run_03 run_04` | Mixer / tokenizer / quantization sweep |
| 1 | `run_05 run_06 run_07 run_08` | CPU-1 core + branches (run_07 is two-phase) |
| 2 | `run_09 run_10 run_11 run_12` | CPU-1 COMPLETE + extensions (run_09 is two-phase) |
| 3 | `run_13 run_14 run_16` | 10M scale-consistency |
| 4 | `run_15` | warm-start from `run_13` checkpoint |

Each subprocess streams stdout/stderr to `run_logs/<run_name>.log`, so a 4×T4 box completes a whole group of four ablations in one wall-clock pass without log interleaving.

```bash
python train_ablation.py --orchestrate                       # all groups
python train_ablation.py --orchestrate --groups 0 2          # only groups 0 and 2
python train_ablation.py --orchestrate --runs run_05 run_09  # ad-hoc single group
python train_ablation.py --orchestrate --max-parallel 1      # force fully sequential
python train_ablation.py --orchestrate --dry-run             # 10-step smoke test
```

Sized for **one full T4 (16 GB) per run**: `batch_per_gpu=24`, `gradient_accumulation_steps=1` (was 8 / 2). At `d_model=512`, `seq_len=5000`, 12 layers, AMP + grad-checkpointing, this fills ~10–12 GB and leaves room for the prefetch buffers below.

#### CUDA latency hiding

`train_ablation.py` wraps the DataLoader in `utilities.CUDAPrefetcher`, which issues host→device copies for batch *N+1* on a side `cuda.Stream` while the main stream computes batch *N*. Combined with `pin_memory=True`, `non_blocking=True`, six DataLoader workers and `prefetch_factor=4`, the H2D copy is fully overlapped with compute on a saturated T4 (large batch, AMP fp16). `cudnn.benchmark=True` and `torch.compile(mode="reduce-overhead")` are also enabled for steady-state throughput.

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
python train_ablation.py --run run_10

# Dry run (no GPU, synthetic data)
python train_ablation.py --run run_01 --dry_run --total_steps 10

# With HF-mount (lazy dataset streaming, no full download)
python train_ablation.py --run run_10 --hf-mount
```

#### Two-Phase Training (runs 07 and 09 — DeleteGate)

For runs that include DeleteGate (MrT5-style), training proceeds in two phases:

- **Phase 1 (CE pretraining):** Standard CE training. DeleteGate disabled (elimination_rate=0).
  `python train_ablation.py --run run_07 --phase 1`

- **Phase 2 (Compression):** DeleteGate activated. Elimination rate anneals to 40% over training.
  `python train_ablation.py --run run_07 --phase 2 --phase1_ckpt <path>`

The `--orchestrate` runner detects two-phase configs automatically and chains Phase 1 → Phase 2 inside the same GPU slot.

#### 3. Checkpoint Formats

Two checkpoint formats are saved automatically by `utilities.save_ablation_checkpoint`:

| Format | When | Contents | Size (50M) | Size (170M) |
|---|---|---|---|---|
| **Resume** (intermediate) | Every N steps | BF16 model + BF16 optimizer state | ~214 MB | ~700 MB |
| **Final** (inference) | End of training | 2-bit packed model, no optimizer | **~14 MB** | **~44 MB** |

The final inference checkpoint uses 2-bit ternary packing: {-1, 0, 1} → {0, 1, 2}, 4 values per byte. Non-ternary weights (norms, gates) are stored as BF16. Load with `utilities.unpack_compact_state()`.

#### 4. Analyze Results

End-of-run validation metrics (`perplexity`, `tokens_per_sec_cpu`, `ram_mb`, `val_loss`, `n_params_M`) are written to `metrics_<run>.json` by `utilities.evaluate_run`. The full evaluation suite (correctness, quality, perplexity, external comparisons) lives in `eval/` — see *Evaluation Suite* below.

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
python train_ablation.py --run run_13   # BPE baseline (~1h)
python train_ablation.py --run run_14   # byte + teacher (~1h)
python train_ablation.py --run run_15 --complement_ckpt models/ablation_models/small_01_cpu1_bpe/checkpoint.pt
python train_ablation.py --run run_16   # raw bytes, no teacher (~1h)
```

---

### Kaggle Workflow

The `kaggle/` directory contains scripts ready to run directly in Kaggle notebooks (T4 GPU):

| File | Purpose |
|---|---|
| `kaggle/cpu1-create-dataset.py` | Runs Qwen2.5-3B teacher (batch aligned for ~4GB VRAM free), builds and uploads the shared dataset to HF Hub |
| `kaggle/upload_dataset.py` | One-shot uploader for local shards → `Cukinator/cpu1-ablation-dataset` |
| `kaggle/create-dataset-metadata.json` | Kaggle kernel metadata (used by `kaggle kernels push`) |

All scripts read `HF_TOKEN` from Kaggle Secrets automatically. Training runs are launched from `train_ablation.py --run <run_id>` directly on the Kaggle VM after `pip install -r requirements.txt`. To push an updated kernel:

```bash
kaggle kernels push -p kaggle/  # reads create-dataset-metadata.json
```

---

## Inference

Production inference is the standalone C driver `inference/benchmark.c` linked against `inference/kernels.cpp` — there is no Python inference engine in the current tree. The kernels are ultra-optimized for the MLGRU architecture: OpenMP, explicit AVX2 + FMA SIMD, Deep Data Dependency Unrolling (DDU), `thread_local` scratch reuse (zero-allocation steady state), and a layer-pipeline mode that keeps each stage's weight slice warm in L2/L3.

The PyTorch forward path defined in `utilities.py` (and assembled into `CPU1ModelV2` in `train_final.py`) remains the reference used by training and the `eval/` suite — it is never benchmarked for throughput, only for numerical correctness and ablation quality.

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

The same C kernel handles a **~430M parameter** scale-up driven directly via `benchmark.exe` CLI flags (`--d_model 1280 --n_layers 24 --d_ff 3328 --fp_rank 8`), with all dims chosen to be kernel-friendly:

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
│  Core (training)
├── utilities.py             # Shared primitives + training infrastructure:
│                            #   BitLinear, BitEmbedding, RMSNorm, FPResidual, PFNet,
│                            #   MLGRU mixers, LocalByteDecoder, BoundaryPredictor,
│                            #   DeleteGate, EmbeddingAligner. Plus CUDAPrefetcher,
│                            #   datasets (Distillation, ByteLevel, BPE, FineWebStreaming,
│                            #   DryRun), distillation/CE loss, LR schedules, param groups,
│                            #   checkpoint serialization (2-bit pack/unpack), evaluate_run.
├── train_final.py           # Production CPU-1 training:
│                            #   CPU1ModelV2 + CPU1ModelV2WithGates,
│                            #   train() (Phase 1 distillation) + train_phase2() (DeleteGate)
├── train_ablation.py        # Ablation training + multi-GPU orchestration:
│                            #   BASE_CONFIG + RUN_CONFIGS (12 50M) + SMALL_RUN_CONFIGS (4 10M),
│                            #   AblationModel + build_ablation_model + train_run,
│                            #   PARALLEL_GROUPS launcher (--orchestrate) — spawns one
│                            #   subprocess per GPU (CUDA_VISIBLE_DEVICES) and chains
│                            #   two-phase DeleteGate runs (run_07/run_09) inside the
│                            #   same GPU slot.
├── create_dataset.py        # Full dataset pipeline: teacher logprob extraction,
│                            #   BPE→byte marginalization, HuggingFace Hub upload
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
│   └── ablation_models/         # One sub-folder per run with archival initial weights
│                                #   (`checkpoint.pt` + `config.json`) for benchmarks pre-training.
│                                #   `__init__.py` exposes `build_run("run_XX")` →
│                                #   `train_ablation.build_ablation_model(get_run_config("run_XX"))`.
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

### Deliberate paper deviations — what we don't implement, and why

The Revision-3 architectural audit (`walkthrough.md`) closed every gap between the codebase and the reference papers (BitNet b1.58, MatMul-free LM, MegaByte, BLT/Bolmo, MrT5, HGRN2, Stuffed Mamba) **except** the six items below. Each is missing *on purpose* — a fix that, after measuring its real cost, would either break the C++ kernel, blow the L2 budget on the i7-8750H, or regress quality. Priority order is **(1) inference speed → (2) intelligence → (3) paper fidelity**, and these items are the points where (3) loses to (1) or (2).

**1. `W_o` output projection in production MLGRU (MatMul-free LM, Zhu 2024).**
The paper specifies four matrices per token mixer (forget, candidate, output, **W_o**). `MatMulFreeTokenMixer` and `decode_layer_fused` in `kernels.cpp` only run three GEMVs; `W_o` is omitted. Adding it would require a fourth GEMV per layer with its own ternary scale, a fourth packed-weight pointer in the kernel ABI, and re-tuning the OMP scheduling that currently fans out 3 GEMVs across cores. The kernel is the speed-critical path (mode-3 throughput is ~620 packets/s) and a 33 % matmul increase would knock it out of its current L2 working set. `AblationMLGRU` *does* support `use_wo=True` for research runs, so the fidelity question is answerable in ablation without paying the production cost.

**2. State expansion N>1 (HGRN2).**
HGRN2 recommends running the recurrence in an expanded space `N · d_model` (paper picks N=2). At `d_model=1024` that doubles the per-layer hidden state from 4 KB to 8 KB. Across 14 layers the resident state grows from 28 KB to 56 KB. Combined with the 224 KB PFNet weights and the streaming ternary weights, the working set crosses the i7-8750H's 256 KB-per-core L2 ceiling — every token would miss to L3 instead of staying L2-resident, collapsing throughput. `AblationMLGRU` exposes `state_expansion` so the quality delta is measurable in ablation; production stays at N=1.

**3. Dynamic / variable-length patching (BLT, entropy-patching).**
BLT picks patch boundaries by per-byte entropy and produces variable-length patch sequences. `LocalByteDecoder` hardcodes `P=4` in three places: the `h_proj` weight shape (`d_model → P·d_local`), the position embedding (`[P, d_local]`), and the C++ `local_decode_step` autoregressive loop (`for kb in 0..3`). The dataloader (`bytes_to_patches`) and gradient checkpointing partition also assume fixed `P`. Variable patches would touch every component on the byte→state→byte path. Instead, `BolmoPatchEmbedding` keeps `P=4` and approximates BLT by gating the byte-to-patch fusion with a learned boundary score — captures the supervisory signal at constant shape.

**4. Real token deletion (MrT5, gather + sequence shrinkage).**
MrT5 deletes positions and physically shrinks the sequence (B, S, D) → (B, S', D) with S' < S. That breaks (a) MLGRU's sequential scan, which is JIT-compiled to a fixed `S` for `torch.compile`; (b) `LocalByteDecoder`'s `[B, S, P]` byte-targets contract; (c) gradient checkpointing's partition into equal-sized blocks; and (d) the Phase-1 → Phase-2 curriculum transition, which loads a Phase-1 checkpoint and continues at the same shape. `DeleteGate` instead emits a near-binary mask, zeroes deleted positions, and merges their residual mass into surviving neighbours via a depthwise conv — same shape, same kernel, and the new bimodal-entropy regulariser (Fix 2 in `walkthrough.md`) keeps the gate from collapsing into uniform dimming.

**5. Parallel prefix scan in MLGRU (associative scan over time).**
The MLGRU recurrence `h_t = lerp(c_t, h_{t-1}, f_t)` is associative and admits an `O(log S)` prefix scan that would speed up training at long `S`. We keep the sequential scan because it is *incompatible with state reset* (Stuffed Mamba): a hard reset every K tokens breaks associativity at the reset boundary, so a prefix scan would need chunked reduction with per-boundary fix-ups — significant implementation cost. The win is also small for our setup: training is already memory-bandwidth bound on CPU, and inference is strictly autoregressive (one token, one step) where prefix scan does nothing.

**6. Removing SiLU from FPResidual (Slender-Mamba "linear residual" recipe).**
Slender-Mamba proposes that the FP residual branch be a pure low-rank linear map, no activation. Stripping SiLU collapses `down(silu(up(x)))` into a single rank-`r` linear correction, which is mathematically equivalent to widening the BitLinear's effective rank by `r`. The audit measured this as a quality regression (the nonlinear branch was demonstrably useful) at zero cache or compute saving — the SiLU runs on `[d_ff_local]`, well below the bottleneck. Priority-2 (intelligence) overrides priority-3 (paper fidelity) here.

| Deviation | Paper | Status | Blocked by | Recoverable in ablation? |
|---|---|:---:|---|:---:|
| `W_o` output projection | MatMul-free LM | ❌ in prod, ✅ in `AblationMLGRU` | Kernel ABI + L2 budget | ✅ (`use_wo=True`) |
| State expansion N>1 | HGRN2 | ❌ in prod, ✅ in `AblationMLGRU` | L2 cache (i7-8750H, 256 KB) | ✅ (`state_expansion=2`) |
| Variable-length patches | BLT | ❌ everywhere | `LocalByteDecoder` + dataloader + kernel C++ all hardcode `P=4` | Partially (Bolmo gate) |
| Real token deletion | MrT5 | ❌ everywhere | Sequential scan + checkpointing + Phase-1↔Phase-2 transition | Partially (DeleteGate + merge + bimodal) |
| Parallel scan | MatMul-free LM | ❌ everywhere | State reset breaks associativity; inference is autoregressive | Not necessary |
| Linear FPResidual (no SiLU) | Slender-Mamba | ❌ everywhere | Empirical quality regression with no cache/compute saving | Not pursued |

The recurring theme: when a paper's recipe was designed for GPU training at scale, blindly porting it to a CPU inference target degrades the very property the paper was optimising. Priority-1 (don't break the kernel, don't blow L2) is non-negotiable on this hardware; everything else is a measured trade-off.

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

- **Architecture** — 170M model: d_model=1024, 14 layers, MLGRU + PFNet + fully ternary (primitives in `utilities.py`, assembled in `train_final.py::CPU1ModelV2`); MegaByte LocalByteDecoder for intra-patch byte autoregression
- **Extended techniques (v3)** — BoundaryPredictor (Bolmo), DeleteGate (MrT5), EmbeddingAligner (Distilling Token-Trained Models into Byte-Level Models), two-phase training
- **Dataset pipeline** — Teacher logprob extraction + BPE→byte marginalization + HF Hub upload (`create_dataset.py`)
- **HF-mount streaming** — `train_ablation.py --hf-mount` lazy-loads shards from the Hub without the ~5 GB full download (module imported on demand; no local `hf_mount.py` yet)
- **Training** — AMP, gradient accumulation, distillation loss; production model in `train_final.py`, ablation runs in `train_ablation.py`, shared infrastructure in `utilities.py`
- **Ablation system** — 12 progressive runs (run_01–run_12 at 50M) + 4 scale runs (run_13–run_16, 10M), all in `train_ablation.py`; archival initial weights in `models/ablation_models/`. Chinchilla-scaled `total_steps`, 5%-of-horizon HF-Hub checkpointing, parallel multi-GPU launcher integrated as `train_ablation.py --orchestrate`. TIDE early-exit and Bolmo integrated gating are parked for follow-up papers.
- **C inference kernels** — `inference/kernels.cpp` + `inference/benchmark.c` standalone driver; 6-stage DP-calibrated layer pipeline (mode 4) and 6-stream batch (mode 3)
- **Evaluation suite** — 14 checks in 5 modules, multi-architecture adaptive, single engine load, fused benchmarks (`eval/`)
- **Kaggle scripts** — Ready-to-run notebooks for dataset creation on T4 (`kaggle/`)

### Models

| Model | Size | Architecture | Status |
|---|---|---|---|
| `models/cpu1-170M/` | ~170M params (~44 MB 2-bit packed) | MLGRU + Byte + Ternary + FPResidual + PFNet + MegaByte LocalDecoder | 🔄 Training |
| `models/LFM2.5-350M/` | HF safetensors | mlGRU (LiquidAI) | ✅ Benchmark reference (llama.cpp Q4 via `eval_external`) |
| `models/gpt2-medium/` | HF safetensors | Transformer | ✅ Baseline reference (`resultados/gpt2-medium_q{4,8}_{1,6}t.md`) |
| `models/ablation_models/run_*` | initial checkpoints | Per-run architecture | ✅ Archival initial weights (FP32, untrained) |

### Pending

- Run the 12 ablation experiments (run_01–run_12) + 4 scale runs (run_13–run_16) on Lightning.ai 4×T4 (via `train_ablation.py --orchestrate`) and collect metrics
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
from train_final import CPU1ModelV2
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

The MegaByte teacher-forced ↔ autoregressive parity check (`LocalByteDecoder.forward()` vs a sequential `generate_step()` loop) used to live as a self-test in the old `model.py`. With the refactor it moves to the dedicated test in `eval/eval_correctness.py` (`check_state_propagation` covers the analogous invariants on the global model). Run it via the unified evaluation suite:

```bash
python eval/run_eval.py --model models/cpu1-170M/checkpoint.pt
```

The expected behaviour is unchanged: `max|forward_logit[k] - step_logit[k]| < 1e-4` for each patch position `k ∈ {0, 1, 2, 3}`, with the true diff typically `< 1e-6`.

---

## Análisis de fidelidad de cada ablation respecto al paper original

Análisis de fidelidad de cada ablation respecto al paper citado, basado en lo que efectivamente implementan `train_ablation.py` + `utilities.py`.

### run_01 — Transformer + BPE + FP16
**Paper:** ninguno (baseline absoluto)
**Componentes:** `SlidingWindowAttention` FP16 + `AblationSwiGLU` FP16 + BPE 16K.
**Fidelidad:** N/A (no replica paper). Nota: usa sliding-window con `window=512`; un baseline Transformer estricto sería full-attention. Se desvía del "transformer puro" pero se etiqueta como baseline interno consistente con el resto del stack.

---

### run_02a — Transformer + Byte + 4 heads independientes
**Paper:** MEGABYTE (Yu et al., 2023)
**Componente:** `AblationBytePatchEmbedding` + 4 `AblationByteHead` FP16 sin LocalByteDecoder.
**Fidelidad:** **Parcial / aislamiento intencional**. MEGABYTE define dos componentes: (a) patch global model y (b) local model autoregresivo intra-patch. Esta corrida **omite el local model** y predice los 4 bytes con cabezas independientes — el propio paper menciona esa variante como ablation (no es la versión completa). Sirve para aislar el costo de tokenización vs. la cadena AR.

---

### run_02 — Transformer + Byte + LocalByteDecoder
**Paper:** MEGABYTE (Yu et al., 2023)
**Componente:** `LocalByteDecoder` — global→`P × d_local` projection, byte_embed shifted, pos_embed por byte, attention causal por patch.
**Fidelidad:** **Alta para la versión 1-layer.** Replica fielmente Listing 1 del paper (h_proj, shift right, pos_embed, attn causal sobre P bytes). **Desviación documentada:** el paper especifica 2–4 capas locales por defecto; la implementación usa `n_layers=1` por defecto (configurable vía `local_decoder_layers > 1` con `extra_layers`), justificado por el constraint del kernel C++.

---

### run_03 — MLGRU + Byte + FP16
**Papers:** Scalable MatMul-Free LM (Zhu et al., 2024) + MEGABYTE
**Componente:** `AblationMLGRU` en FP16 (sin ternario).
**Fidelidad MatMul-Free:** **Alta.** Implementa f, c, g (forget/cand/output gates) con `f·h + (1-f)·c`, scan paralelo Heinsen log-domain (`_parallel_scan`) que es exactamente la forma "parallel scan" prescrita por el paper, y W_o opcional (4ª matriz del paper). **Desviación:** el paper propone también un "ternary GLU channel mixer"; en run_03 el FFN es FP16 (necesario porque la corrida es FP16, pero rompe la propiedad matmul-free completa).

---

### run_04 — MLGRU + Byte + Ternario
**Papers:** BitNet b1.58 (Wang et al., 2024) + MatMul-Free LM
**Componente:** `BitLinear` en gates + heads + FFN.
**Fidelidad BitNet b1.58:** **Alta.** `w_scale = mean(|w|)`, `w_ternary = round(w/scale).clamp(-1,1)`, activaciones INT8 per-token con `amax/127`, STE — coincide con Algorithm 1 del paper. **Desviación menor:** el paper recomienda RMSNorm pre-BitLinear (Fused-BitLinear); la implementación lo aplica solo en `down_norm`, `o_norm`, `wo_norm` (no antes de cada BitLinear individual — los gates de MLGRU reciben input ya normalizado por norm1 del bloque, lo cual cumple).

---

### run_05 — CPU-1 core (FPResidual añadido)
**Papers:** Slender-Mamba (Yu et al., 2025) + BitNet b1.58
**Componente:** `FPResidual` — low-rank FP16 (rank=4): `silu(A·x) → B`.
**Fidelidad Slender-Mamba:** **Baja / interpretación libre.** Slender-Mamba propone explícitamente mantener **componentes específicos** (matriz A de Mamba) en mayor precisión por sensibilidad recurrente; el resto va a 1.58-bit. La implementación aquí es un **residual de baja rango paralelo** (técnica de BitNet/LoRA-like), no una promoción selectiva de componentes sensibles. La cita "Slender-Mamba" describe correctamente la motivación (precisión mixta) pero el mecanismo concreto **no replica** la receta de sensibilidad por componente del paper.

---

### run_05b_kernel_strict — MLGRU sin W_o
**Papers:** MatMul-Free LM + BitNet b1.58 + Slender-Mamba
**Componente:** Igual a run_05 pero `mlgru_use_wo=False`.
**Fidelidad:** **Intencionalmente menos fiel al paper, más fiel al kernel.** El propio comentario lo declara: el paper de Zhu 2024 **incluye W_o**; esta corrida lo elimina por restricciones de scheduling OMP del kernel C++. Mide costo de la divergencia paper↔producción.

---

### run_06 — + Bolmo (cross-attention + boundary scorer)
**Paper citado:** BLT (Pagnoni et al., 2024) — y por extensión Bolmo
**Componente:** `BolmoPatchEmbedding` — Conv1d k=3 boundary scorer + cross-attention de cada patch a window de bytes con boundary score como bias aditivo.
**Fidelidad BLT:** **Baja-Media.** BLT real usa: (1) patches **de longitud variable** determinados por entropía de un LM byte-level pequeño; (2) encoder/decoder local; (3) global model sobre patches variables. La implementación: (1) **patches de tamaño fijo P=4** con una cross-attention que ve ±1 patch (12 bytes) — aproxima la sensibilidad a frontera pero **no produce patches variables**; (2) supervisión heurística (`get_natural_boundaries` por puntuación/UTF-8/runs ≤8 bytes), **no entropía de LM** como BLT. El propio código admite la divergencia ("approximating BLT variable-length patches via wider fixed attention", "audit §Conflicto 11"). Más cercano a "Bolmo-style boundary gating" que a BLT estricto.

---

### run_07 — + DeleteGate (gather real)
**Papers:** MrT5 (Pagliardini et al., 2024) + MatMul-Free + BitNet + BLT + Slender-Mamba
**Componente:** `DeleteGate` con `forward_gather` (TopK + sort temporal + scatter_back).
**Fidelidad MrT5:** **Media-Alta.**
- ✅ Replica el espíritu: gate produce score, eliminación al ~40%, gather físico que reduce la longitud para capas posteriores, scatter de vuelta.
- ✅ Gumbel-sigmoid + STE para hard mask {0,1}.
- ✅ Penalización bimodal (entropía del soft mask) y constraint de tasa.
- ⚠️ **Desviación:** MrT5 original usa una soft-mask con un **schedule continuo** que evoluciona durante entrenamiento (no TopK estricto + STE). Aquí se hace TopK con K fijo por batch — más simple, menos diferenciable que la formulación MrT5 pura.
- ⚠️ El `merge_conv` depthwise de "merge deleted into neighbours" es propio (MrT5 usa un esquema de fusión basado en suma ponderada por scores; el conv depthwise es una aproximación parametrizada).

---

### run_08 — Folded Transformer (sliding window)
**Papers:** Longformer (Beltagy et al., 2020) + BitNet b1.58
**Componente:** `SlidingWindowAttention` con `window_size=128` ternaria.
**Fidelidad Longformer:** **Parcial.** Longformer combina tres patrones: (a) sliding window local; (b) **dilated sliding window**; (c) **global attention tokens** en posiciones especiales. La implementación cubre solo (a). Es una "sliding-window attention causal" estándar, no Longformer completo. Más justa la atribución sería "local-window causal attn" genérico.

---

### run_09 — + PFNet
**Papers:** MatMul-Free + BitNet + BLT + MrT5 + Slender-Mamba
**Componente:** `PFNet` — RMSNorm → BitLinear(d→32) → SiLU → BitLinear(32→d), residual.
**Fidelidad:** **N/A** — PFNet **no corresponde a ningún paper citado**; es un módulo propio descrito como "cache-resident nonlinear residual" justificado por footprint L2. Las citas son del stack heredado, no del PFNet en sí.

---

### run_10 — + per-channel decay
**Papers:** RWKV (Peng et al., 2023) + HGRN2 (Qin et al., 2024) + MatMul-Free + BitNet + BLT + MrT5
**Componente:** `log_decay = nn.Parameter(state_dim)` con `decay = exp(-softplus(log_decay))`, init jerárquico (capas profundas = decay más alto), aplicado como `f = decay * sigmoid(forget_proj(x))`. Más `cummax` entre capas en `_apply_layer_decay_cummax`.
**Fidelidad HGRN2:** **Alta para el "decay" pieza.** HGRN2 paper especifica: forget-gate ceiling per-channel + init jerárquico monotónico con la profundidad — ambos están. El **cummax inter-layer** que enforce monotonicidad es exactamente la prescripción HGRN2.
**Fidelidad RWKV:** **Media.** RWKV usa decay per-channel (time_decay) pero con una formulación distinta (token shift, time-mixing con receptance). Aquí solo se toma la idea del decay per-channel; no hay token-shift ni receptance. La cita es justa solo en sentido amplio.

---

### run_10b_state_expansion — + state expansion N=2
**Papers:** HGRN2 + MatMul-Free + BitNet + MrT5
**Componente:** `state_dim = d_model · N`; gates proyectan `d_model → state_dim`; W_o comprime de vuelta a `d_model`.
**Fidelidad HGRN2:** **Alta.** HGRN2 define exactamente esto: estado en `D_state >> D_model`, up-projection antes de la celda, down-projection después. El ratio 2x está dentro del rango típico del paper (2–4x). Combina además decay per-channel jerárquico → cubre "HGRN2 paper-faithful spec" completo. **Es la corrida con mayor fidelidad a HGRN2** del set.

---

### run_11 — + state reset periódico
**Papers:** Stuffed Mamba (anonymous, 2024) + MatMul-Free + BitNet + BLT
**Componente:** `state_reset_period=512` ⇒ en el loop secuencial: `if t % 512 == 0: h = zeros_like(h)`.
**Fidelidad Stuffed Mamba:** **Baja-Media.** Stuffed Mamba propone tres soluciones: (1) **state reset periódico con sliding-window recurrence**; (2) **state compression** aprendida; (3) **adaptive forgetting** (entrenamiento con longitudes variables). La implementación cubre solo (1) y en su forma más cruda — **hard reset a cero**, no la "sliding-window recurrence" que el paper detalla (donde el estado se reinicia con interpolación o se mantiene una ventana móvil). Es la idea más simple del paper, no la propuesta principal.

---

### run_13/14/16 (small @ 10M)
**Papers citados:** MatMul-Free + Slender-Mamba + Bolmo + MrT5
**Fidelidad:** Heredan el stack "CPU-1 COMPLETE" — la fidelidad a cada paper es la misma que en run_07/run_06 con sus respectivas desviaciones. Lo distintivo es la **tokenización** (BPE 4K vs byte_dynamic vs raw bytes) y el **signal de entrenamiento** (distillation Qwen vs CE-only); ninguno de estos tiene fidelidad-a-paper que medir más allá del baseline.

---

### Resumen tabular de fidelidad

| Run | Paper principal | Fidelidad | Desviación clave |
|---|---|---|---|
| run_01 | — | N/A | sliding-window en lugar de full-attn |
| run_02a | MEGABYTE | Parcial intencional | sin LocalByteDecoder |
| run_02 | MEGABYTE | **Alta** | 1 capa local vs 2-4 del paper |
| run_03 | MatMul-Free LM | **Alta** | FFN no-ternario (limitación FP16) |
| run_04 | BitNet b1.58 | **Alta** | RMSNorm parcial vs Fused-BitLinear total |
| run_05 | Slender-Mamba | **Baja** | FPResidual ≠ promoción selectiva por sensibilidad |
| run_05b | MatMul-Free LM | Anti-fiel intencional | sin W_o (kernel-strict) |
| run_06 | BLT | **Baja-Media** | patches fijos + boundary heurístico vs entropía LM |
| run_07 | MrT5 | **Media-Alta** | TopK+STE en lugar de soft-mask schedule |
| run_08 | Longformer | **Parcial** | solo local window, sin dilated/global |
| run_09 | (PFNet propio) | N/A | módulo no corresponde a paper citado |
| run_10 | HGRN2 (decay) | **Alta** | cummax + init jerárquico fieles |
| run_10b | HGRN2 (full) | **Alta** ⭐ | state expansion + decay = spec completa |
| run_11 | Stuffed Mamba | **Baja-Media** | solo hard-reset, no sliding-window recurrence |

**Patrón general:** las ablations más fieles al paper original son las de la familia recurrente cuantizada (BitNet, MatMul-Free, MEGABYTE-local, HGRN2). Las menos fieles son **BLT/Bolmo** (boundary heurístico vs entropía-LM, patches fijos vs variables), **Slender-Mamba** (mecanismo distinto al del paper), **Longformer** (solo 1 de 3 patrones) y **Stuffed Mamba** (solo la solución más simple del paper). MrT5 está en zona intermedia: captura el espíritu pero simplifica la formulación.

---

### Detalle de por qué cada uno cayó a Media/Baja

#### run_05 — Slender-Mamba → **Baja**

**Lo que dice el paper Slender-Mamba:**
El hallazgo central es un **análisis de sensibilidad por componente**. La matriz A de Mamba (decay del estado SSM) no se puede cuantizar a 1.58-bit porque los errores se acumulan recurrentemente y colapsa el aprendizaje. Por eso prescribe una tabla concreta:

| Componente | Precisión recomendada |
|---|---|
| in_proj, gate, D, out_proj | ternario (1.58-bit) |
| **Matriz A (decay)** | **FP8/FP16** |
| B, C | INT8 |

El protocolo es: **mantener componentes específicos en mayor precisión** según su sensibilidad recurrente, no agregar un residual aparte.

**Lo que hace `FPResidual`:**
```python
self.fp_A = nn.Linear(in_dim, rank, bias=False)   # rank=4
self.fp_B = nn.Linear(rank, out_dim, bias=False)
return self.fp_B(F.silu(self.fp_A(x)))
```
Es una **rama LoRA-like paralela**: low-rank FP16 sumada al output del BitLinear ternario. Toda la matriz original sigue ternaria; el "fix" es un parche aditivo de baja capacidad.

**Por qué es Baja:**
1. **Mecanismo distinto:** correction aditiva ≠ promoción de precisión del peso original.
2. **No respeta sensibilidad por componente:** el FPResidual se aplica uniformemente a `down_proj` del FFN y a la salida del MLGRU, no específicamente al análogo de la matriz A (que sería `forget_proj` / `log_decay`). El equivalente al hallazgo del paper sería mantener `forget_proj` en INT8/FP16 y dejar el resto ternario — eso no se hace.
3. La cita está justificada en intención (precisión mixta) pero el código está más cerca de **BitNet b1.58 + LoRA** que de Slender-Mamba.

---

#### run_06 — BLT → **Baja-Media**

**Lo que define BLT (Pagnoni et al., 2024):**
1. **Patches de longitud variable** determinados dinámicamente.
2. La señal de boundary viene de la **entropía de un LM byte-level pequeño preentrenado** corriendo en paralelo: cuando la entropía supera un threshold → ahí termina un patch.
3. Encoder local procesa los bytes dentro de cada patch variable; global model procesa la secuencia (más corta) de patches.
4. Resultado: patches densos en regiones predecibles, cortos en regiones informativas.

**Lo que hace `BolmoPatchEmbedding`:**
1. **Patches fijos P=4 bytes**, definidos al cargar el dataset, no por el modelo.
2. Boundary signal = `Conv1d(k=3)` aprendido sobre embeddings de bytes, supervisado contra la heurística `get_natural_boundaries`:
   ```python
   is_boundary = (b == 0x20 or b in PUNCT or b >= 0xC0 or prev_is_space)
   ```
   — espacios, puntuación, lead-bytes UTF-8, runs ≤8.
3. El boundary score se usa como **bias aditivo en la cross-attention** dentro del patch (y opcionalmente sobre ±1 patch vecino). El patch sigue siendo de 4 bytes.

**Por qué es Baja-Media:**
1. **Patches fijos vs variables** — la propiedad central de BLT (compresión adaptativa de la secuencia global) **no existe**: el global model siempre ve `S/4` patches, no menos en regiones predecibles. Esto se debe a constraints aguas abajo (LocalByteDecoder + DataLoader + kernel C++ esperan stride fijo), reconocido en el comentario "approximating BLT variable-length patches via wider fixed attention".
2. **Boundary heurístico vs entropía-LM** — la supervisión es una regla determinista por bytes, no la salida de un LM. Boundaries en una palabra rara o un número (regiones informativas según BLT) no se detectan; sí se detecta cualquier espacio (regiones predecibles).
3. Lo que **sí** se rescata: la idea de "boundary-aware patch embedding" (la cross-attention pondera bytes cerca de fronteras), que aproxima el comportamiento sin la mecánica.

Está más cerca del paper "**Bolmo**" (boundary gating sobre embedding) que de BLT estricto — la cita es ambigua porque el código mezcla ambas tradiciones.

---

#### run_07 — MrT5 → **Media-Alta** (la única "alta" parcial de las medias)

**Lo que define MrT5:**
- Token Deletion Module (TDM) con **soft mask continua** entrenada con un schedule progresivo (la tasa de eliminación crece durante entrenamiento).
- Diferenciable end-to-end vía Gumbel-Softmax sobre la decisión.
- Up-sampling al final para restaurar la resolución original con interpolación de las representaciones eliminadas.

**Lo que hace `DeleteGate.forward_gather`:**
- ✅ Gumbel-sigmoid + STE para `{0,1}` decisivo.
- ✅ Constraint de tasa `((1-soft_mask).mean() - target_rate)²` + bimodal entropy regulariser.
- ✅ Gather físico: TopK survivors → `[B, K, D]` post-gate.
- ⚠️ **TopK con K fijo** (≠ MrT5 que decide token-por-token con soft mask continua). Aquí siempre sobreviven exactamente `S·(1-rate)` tokens por batch.
- ⚠️ El "up-sampling" de MrT5 (interpolar representaciones eliminadas) se reemplaza por `scatter_back` que **deja las posiciones eliminadas con su representación pre-gate**, sin refinamiento. Más simple, conserva gradiente, pero no es el up-sampling del paper.
- ⚠️ El "merge" (redistribuir masa eliminada en vecinos) se hace con un **depthwise Conv1d k=3** inicializado a promedio uniforme. MrT5 hace una suma ponderada por scores aprendidos, no un conv depthwise.

**Por qué Media-Alta y no Alta:** captura toda la lógica de eliminación-y-merge, pero las tres aproximaciones (TopK fijo, scatter sin upsampling, merge=conv) son simplificaciones operacionales del mecanismo continuo del paper.

---

#### run_08 — Longformer → **Parcial**

**Lo que define Longformer:**
Tres patrones de atención **combinados**:
1. **Sliding window local** (cada token atiende a w/2 vecinos a cada lado).
2. **Dilated sliding window** en capas superiores (saltos de tamaño d para mayor receptive field).
3. **Global attention** en tokens especiales (CLS, SEP, query tokens) que atienden a todo y son atendidos por todo.

El paper enfatiza que la combinación es lo que iguala a full-attention en tareas como QA.

**Lo que hace `SlidingWindowAttention`:**
Solo el patrón (1): banda causal de tamaño `window_size=128`. Sin dilatación, sin tokens globales.

**Por qué es Parcial:** falta 2/3 del mecanismo. Es "sliding-window causal attention" estándar, no Longformer. La cita es generosa.

---

#### run_11 — Stuffed Mamba → **Baja-Media**

**Lo que propone Stuffed Mamba:**
El paper diagnostica el problema (state saturation) y propone **tres soluciones, en orden de sofisticación**:
1. **Sliding-window recurrence con state reset** — el reset no es a cero, sino que se mantiene una ventana móvil del estado.
2. **State compression aprendida** — un módulo aprende a comprimir el estado antiguo antes de incorporar nuevo contexto.
3. **Adaptive forgetting** — entrenar con secuencias de longitud variable para que el gate aprenda activamente a olvidar.

La propuesta principal del paper son (2) y (3); (1) es la baseline más simple.

**Lo que hace el código:**
```python
if reset > 0 and t > 0 and t % reset == 0:
    h = torch.zeros_like(h)
```
— **hard reset a cero cada 512 pasos** dentro del loop secuencial. Cero parámetros adicionales.

**Por qué es Baja-Media:**
1. Implementa solo la versión **más cruda** de la solución (1): hard reset, no sliding window.
2. **No implementa** state compression aprendida (2) ni adaptive forgetting (3) — que son las contribuciones principales del paper.
3. Bonus negativo: el hard reset desactiva el `parallel_scan` y fuerza el loop secuencial — pierde la ganancia algorítmica del MatMul-Free LM. El paper no propone este trade-off.

Es esencialmente "la idea más obvia del paper, sin las contribuciones reales".

---

### Patrón común de las "Bajas"

Las cuatro evaluaciones bajas/medias comparten **el mismo modo de falla**: el código toma la **idea conceptual** del paper pero implementa una versión **simplificada operacionalmente** porque:

- **run_05 (Slender-Mamba):** la promoción selectiva por componente requeriría refactor de la BitLinear para aceptar mask de precisión por matriz → se reemplaza por un residual genérico.
- **run_06 (BLT):** patches variables romperían LocalByteDecoder + DataLoader + el kernel C++ que asumen stride fijo → se aproxima con cross-attention.
- **run_08 (Longformer):** dilated + global tokens requerirían tres tipos de máscara y ruteo de tokens especiales → se queda en el patrón más barato.
- **run_11 (Stuffed Mamba):** state compression aprendida añadiría parámetros y rompería la propiedad "zero extra params" → se queda en el reset trivial.

En todos los casos las desviaciones son **trade-offs explícitos** entre fidelidad al paper y compatibilidad con el resto del stack (kernel C++, ternario, Chinchilla budget). Los comentarios del código las reconocen casi siempre — no son omisiones, son simplificaciones deliberadas.

---

## License

MIT