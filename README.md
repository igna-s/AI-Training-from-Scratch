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
| [Scalable MatMul-free LM (arXiv 2406.02528)](https://arxiv.org/abs/2406.02528) | MLGRU token mixer â Hadamard-only, no Q@K^T | â
 `MatMulFreeTokenMixer` |
| [Slender-Mamba (COLING 2025)](https://aclanthology.org/2025.coling-main.316/) | Ternary embedding and prediction heads | â
 `BitEmbedding`, `BitLinear` |
| [T-MAC (EuroSys 2025, arXiv 2407.00088)](https://arxiv.org/abs/2407.00088) | LUT-based zero-multiply inference on CPU | â
 `inference/kernels.cpp::ternary_lut_matmul_omp` + `forward_full_mad_*` |
| [BitNet b1.58 (arXiv 2402.17764)](https://arxiv.org/abs/2402.17764) | 1.58-bit weight quantization with STE | â
 `BitLinear` |
| [Bolmo (arXiv 2410.20771)](https://arxiv.org/abs/2410.20771) | Dynamic byte patches via boundary prediction | â
 `BoundaryPredictor` |
| [MrT5](papers/mrt5-dynamic-token-merging-byte-level-lm.pdf) | Token deletion gate â prunes 40% of states | â
 `DeleteGate` |
| [Distilling Token-Trained Models into Byte-Level Models](papers/distilling-token-models-into-byte-level.pdf) | Embedding alignment distillation loss | â
 `EmbeddingAligner` (CE-only for ablation) |
| [LFM2 (arXiv 2511.23404)](https://arxiv.org/abs/2511.23404) | Hybrid CNN + GQA architecture (LiquidAI) | ⚠️ **Not implemented** — external benchmark only (`models/LFM2.5-350M/` safetensors; a Q4_K_M GGUF is also benchmarked via `eval_external`) |
| [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531) | KL + CE distillation loss, T=4.0 | â
 `train_final.py` distillation mode |

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

## Ablation Study — 50M Progressive Chain + 10M Scale Runs

> **2026-05 status.** All 50M and 10M runs have been trained and uploaded to HF Hub. A round-2 re-run at 15 tok/param confirmed the cold-start ternary plateau is structural, not a budget issue. **For the actual measured results, mechanistic analysis of why ternary cold-start fails at this scale, and the throughput-regime crossover, see [§ Ablation Audit — 2026-05 Findings](#ablation-audit--2026-05-findings) below.** The architectural description in this section is the *design intent*; the audit section is the *empirical outcome*.

The ablation tells a story: starting from a standard FP16 Transformer baseline, we add **one CPU-1 design choice per run** until we reach the full architecture. Each step isolates the contribution of a single component.

All runs: CE-only training from scratch under **ablation-scaled Chinchilla** (`tokens ≈ 2 × N_params`), same seed (42) as `create_dataset.py`. `total_steps` is recomputed per run from the actual parameter count inside `train_run` — see *Chinchilla & Checkpointing* below.

**Dataset routing** (matches the upstream pipeline in `create_dataset.py`):

| Runs | Dataset | Why |
|---|---|---|
| `run_01` … `run_10` (50M) + branches `run_05b_kernel_strict`, `run_08` | `HuggingFaceFW/fineweb` `sample-10BT`, streamed | Chinchilla 50M needs ~1B tokens — the pre-processed `Cukinator/cpu1-ablation-dataset` only holds ~37.5M, sized for the 10M scale. The 50M runs use `FineWebStreamingDataset` to consume FineWeb directly with the **same params `create_dataset.py` uses** (seed=42, seq_len=5000, patch_size=4, max 700 BPE tokens / 2800 bytes per doc, 95/5 train/val split modulo doc index 20). On-the-fly UTF-8 byte packing for byte runs, on-the-fly Qwen tokenization for `run_01` (clamped to its 16K vocab cap). |
| `run_13`, `run_14`, `run_15` (10M) | `Cukinator/cpu1-ablation-dataset` (2 subsets) | The pre-processed dataset has two subsets that match each run: `bpe_tokenized/` for `run_13`, `byte_marginalized/` for `run_14`/`run_15`. At 2 tok/param, 10M models train for ~200 steps (floored by `chinchilla_total_steps`). |
| `run_16` (10M, "no teacher") | `HuggingFaceFW/fineweb`, streamed | Same path as the 50M runs — the run *is* the "skip the Qwen teacher" ablation. |

The 50M track is the linear chain `run_01 → run_02a → run_02 → run_03 → run_04 → run_05 → run_06 → run_07 → run_09 → run_10`, with branches `run_05b_kernel_strict` and `run_08`. See `RUN_CONFIGS` in `train_ablation.py` for the authoritative list.

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
| Factorization `â P(báµ¢ \| ctx, b<i)` â chain rule, not independence | â
 |
| Globalâlocal via `W_proj(h_G) â PÂ·D_L` with per-position reshape | â
 |
| Shift-right byte input with zero-pad at position 0 | â
 |
| FFN width 4Ã `d_local` | â
 |
| Teacher forcing in training / autoregressive in inference | â
 |
| Global model â« Local model in parameters | â
 (50M vs ~700K ≈ 70×) |
| Patch size 4 | â
 |

**Deliberate deviations** (design choices, not bugs):

- **SwiGLU instead of GELU** — universally accepted modern upgrade; used everywhere else in the model for consistency.
- **RMSNorm instead of LayerNorm** — cheaper, standard in modern LLMs; matches the rest of the CPU-1 stack.
- **1 local layer instead of 2** — at `seq_len = P = 4`, additional depth gives near-zero return. A second layer becomes worth trying only when `d_local ≥ 256`.
- **Ternary weights by default** — this is the CPU-1 thesis, not a MegaByte choice. The `local_decoder_fp16` flag exists precisely so the ternary-vs-FP16 tradeoff in this submodule can be measured as a separate ablation.

**Intentional omission**: there is no separate "first-byte head" reading `h` directly. Position 0 of the local decoder already receives `global_slice[0] + embed(zero_pad) + pos_embed[0]`, so its output is `P(b₀ | h)` — exactly what a dedicated head would produce, with no extra parameters.

In short: the architectural skeleton matches the paper. Remaining differences are modern activation/norm choices (SwiGLU, RMSNorm) or project-specific constraints (ternary). A reviewer familiar with MegaByte would recognise this as a faithful implementation.

### Progressive Design — 50M runs

```
LINEAR CHAIN (single Δ per arrow):
run_01:              Transformer  + BPE   + FP16        ← absolute baseline (16K vocab cap)
          ↓ +byte tokenization (no LocalByteDecoder)
run_02a_byte_only_heads: Transformer + Byte + FP16 (4 indep. heads)  ← +byte patches only
          ↓ +LocalByteDecoder
run_02:              Transformer  + Byte  + FP16        ← +MegaByte intra-patch chain
          ↓ swap mixer
run_03:              MLGRU        + Byte  + FP16        ← +MatMul-free mixer
          ↓ quantize
run_04:              MLGRU        + Byte  + Ternary     ← +1.58-bit weights
          ↓ +FPResidual
run_05:              MLGRU        + Byte  + Ternary + FPRes  ← CPU-1 core ⭐
          ↓ +Bolmo (byte_dynamic patches)
run_06:              + Bolmo                            ← +boundary-aware patch embedding
          ↓ +DeleteGate (real gather)
run_07:              CPU-1 COMPLETE ⭐                  ← +MrT5-style state elimination
          ↓ +PFNet
run_09:              + PFNet (pfnet_hidden=32)          ← +cache-resident nonlinear residual
          ↓ +per-channel decay
run_10:              + decay (RWKV/HGRN2-style)         ← chain terminus

BRANCHES (off the linear chain):
run_05b_kernel_strict  ← from run_05  (no W_o — kernel-layout faithful, zero extra cost)
run_08                 ← from run_05  (Folded Transformer + Byte + Ternary, w=128)
```

`run_01` caps the BPE vocab at **16 384** (top-K by frequency, dataset IDs above the cap collapse to OOV in `BPEDataset`); the full 152 K Qwen vocab would consume ~78 MB just on the embedding and overflow the 50M budget on its own.

Key questions answered:
- `run_01 → run_02a`: Does byte tokenization alone hurt a FP16 Transformer (vs BPE)?
- `run_02a → run_02`: Does MegaByte LocalByteDecoder recover the independent-head penalty?
- `run_02 → run_03`: Does MLGRU improve over Transformer at byte level?
- `run_01 → run_03`: Transformer+BPE vs MLGRU+Byte — combined mixer+tokenization effect
- `run_03 → run_04`: Does ternary quantization hurt MLGRU+byte?
- `run_04 → run_05`: Does FPResidual recover ternary quality loss?
- `run_05 vs run_05b_kernel_strict`: Cost of dropping W_o to match production kernel layout?
- `run_05 vs run_08`: CPU-1 core (MLGRU) vs best local-attention Transformer?
- `run_05 → run_06`: Does Bolmo improve over fixed 4-byte patches?
- `run_06 → run_07`: Does DeleteGate preserve quality while pruning 40% of states?
- `run_07 → run_09`: Does PFNet improve quality without hurting inference speed?
- `run_09 → run_10`: Does per-channel learnable decay improve over a fixed forget gate?

### Chinchilla & Checkpointing

Each run trains to its own ablation-scaled Chinchilla horizon: `total_steps = (2 × N_params) ÷ (batch × accum × seq_len)`, recomputed inside `train_run` once `build_ablation_model` reports `N_params`. **2 tokens/param** is the standard academic budget for architecture comparison ablations — enough signal to rank architectures reliably without reaching Chinchilla-optimal loss (which would require 20 tok/param and ~10× longer runs). Both `warmup_steps` (5% of horizon) and `save_every_steps` (5% of horizon) are derived from this rescaled `total_steps`, so every run gets the **same warmup ratio and the same checkpoint cadence** regardless of param count. Override with `--total_steps N` to fall back to a fixed budget (smoke tests).

Checkpoints upload to `Cukinator/cpu1-ablation-checkpoints` on HF Hub via `save_ablation_checkpoint`, named `ablation/checkpoint_<run>_step<N>.pt`. The local file is rotated (only the latest step on disk) but every step is preserved on the Hub for later learning-curve fits and intermediate eval.

| Setting | Where | Default |
|---|---|---|
| `chinchilla` | `BASE_CONFIG` | `True` |
| `chinchilla_tokens_per_param` | `BASE_CONFIG` | `2` (ablation budget) |
| `warmup_ratio` | `BASE_CONFIG` | `0.05` (5% of Chinchilla horizon) |
| `checkpoint_fraction` | `BASE_CONFIG` | `0.05` (≈ 20 checkpoints) |
| `checkpoint_repo` | `BASE_CONFIG` | `Cukinator/cpu1-ablation-checkpoints` |
| `seed` | `BASE_CONFIG` | `42` (matches `create_dataset.SEED`) |

Step counts at 2 tok/param (approximate — exact value depends on param count at build time):

| Run | N_params | Steps (2 tok/param) | Warmup (5%) | Est. time (T4) | Tokens / step |
|---|---|---|---|---|---|
| `run_01` (FP16 BPE 16K) | 54.7 M | ~911 | ~46 | ~1.2 h | 120 K |
| `run_05` (CPU-1 core) | 35.6 M | ~593 | ~30 | ~0.8 h | 120 K |
| `run_07` (CPU-1 COMPLETE) | 35.6 M | ~593 | ~30 | ~0.8 h | 120 K |
| `run_09` (+ PFNet) | 36.0 M | ~600 | ~30 | ~0.8 h | 120 K |
| `run_10` (+ decay) | 36.0 M | ~600 | ~30 | ~0.8 h | 120 K |
| `run_13` (10M BPE 4K) | 11.7 M | ~200 (floor) | ~50 (floor) | ~15 min | 120 K |
| `run_14` (10M byte) | 9.6 M | ~200 (floor) | ~50 (floor) | ~15 min | 120 K |

### Hyperparameter Audit — what's equal, what isn't, why

All ablation runs share the **same training recipe** so quality differences are attributable to architectural choices, not hyperparameter drift. Audited 2026-04-25 across the 16 active runs:

| Hyperparameter | Value | Equal across all runs? |
|---|---|---|
| `seed` | 42 | â
 — matches `create_dataset.py:987` `SEED=42` |
| `dataset_repo` | varies by scale (see *Dataset routing* above) | 50M + run_16 → `HuggingFaceFW/fineweb` (streamed); 10M run_13/14/15 → `Cukinator/cpu1-ablation-dataset`. The two paths both follow `create_dataset.py` params (same seed, seq_len, patch_size, max-doc-tokens). |
| `lr` / `min_lr` | 3e-4 / 3e-5 | â
 |
| `batch_per_gpu` Ã `accum` Ã `seq_len` | 24 Ã 1 Ã 5000 = 120 K tok/step | â
 |
| `warmup_ratio` | 0.05 of `total_steps` | â
 (was a bug — see below) |
| `chinchilla_tokens_per_param` | 2 (ablation budget) | â
 |
| `weight_decay_bitlinear` / `_heads` | 0.1 / 0.01 | â
 |
| `grad_clip` | 1.0 | â
 |
| `temperature` / `alpha_kl` / `alpha_ce` | 4.0 / 0.9 / 0.1 | â
 (unused: `training_signal='ce_only'`) |
| `checkpoint_fraction` | 0.05 | â
 |

**Two datasets, both anchored to `create_dataset.py`:**

- The **upstream raw corpus** is `HuggingFaceFW/fineweb` `sample-10BT` (10B tokens). `create_dataset.py:1225` streams it with `SEED=42` and runs it through Qwen2.5-3B to build the pre-processed dataset.
- The **pre-processed product** is `Cukinator/cpu1-ablation-dataset` (~37.5M tokens, two subsets: `byte_marginalized/`, `bpe_tokenized/`).
- 50M runs (`run_01` … `run_10`, plus branches) and `run_16` consume the *raw upstream* directly via `FineWebStreamingDataset` — same SEED, same SEQ_LEN, same PATCH_SIZE, same per-doc cap as the build pipeline. Reading raw is necessary for them: Chinchilla at 50M needs ~1B tokens, and the pre-processed dataset doesn't have that many.
- 10M runs `run_13`/`run_14`/`run_15` consume the pre-processed product, picking the matching subset by tokenization.

**Intentional per-run overrides** (these *are* the ablation, not hyperparameter drift):

| Run | Override | Reason |
|---|---|---|
| `run_01` | `bpe_vocab_size=16384` | full Qwen 152K vocab → 78 MB embedding alone, busts 50M budget. Top-16K by frequency, rest collapses to OOV in `BPEDataset._clamp_ids`. |
| `run_05b_kernel_strict` | `mlgru_use_wo=False` | the run *is* "kernel-layout faithful MLGRU without W_o" |
| `run_08` | `window_size=128` | the run *is* "best CPU-friendly local-attention baseline" |
| `run_09` | `pfnet_hidden=32` | enables the PFNet component being ablated |
| `run_10` | `mlgru_per_channel_decay=True` | enables per-channel decay being ablated |
| `run_13` | `bpe_vocab_size=4096`, `use_boundary_predictor=False` | 10M budget cap; BPE has its own subword boundaries |
| `run_15` | `use_embedding_alignment=True` | enables hidden-state distillation via EmbeddingAligner |
| `run_16` | `dataset_repo='HuggingFaceFW/fineweb'`, `training_signal='ce_only'` | this run *is* the "no-teacher, raw bytes" ablation question |

**Bugs found and fixed during audit:**

1. **`run_01` BPE vocab overflow** — was defaulting to 152 064 tokens (Qwen2.5), which makes the embedding alone 78 M params and overflows the 50 M budget. Fixed: `bpe_vocab_size=16384` + clamp in `BPEDataset`.
2. **Warmup ratio drift under Chinchilla** — fixed `warmup_steps=500` against a Chinchilla-rescaled `total_steps` produced variable warmup fractions. Replaced with `warmup_ratio=0.05` so every run gets the same 5% warmup regardless of horizon.
3. **`run_16` had no implementation backing** — its config declared `dataset_repo='HuggingFaceFW/fineweb'` and `raw_bytes_only=True`, but the original loader only knew how to read pre-processed Cukinator parquet shards. The same gap was about to bite the 50M runs, which can't fit in Cukinator under Chinchilla. Fixed by adding `FineWebStreamingDataset` (mirrors `create_dataset.py` params, now in `utilities.py`) and a routing branch in `load_ablation_dataset` keyed on `dataset_repo`. `run_16` now actually runs end-to-end on FineWeb; the 50M runs (`run_01` … `run_10`, plus branches) inherit the same path via `BASE_CONFIG`.
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

After the 2026-05 audit the on-disk layout was harmonised with the run names defined in `train_ablation.py`. The previous "archival" naming scheme (`run_06_cpu1_core/`, `run_07_cpu1_bolmo/`, etc.) is no longer used.

### Where checkpoints live now

| Path | Contents | Format |
|------|----------|--------|
| `models/ablation_models/<run>/checkpoint_<run>_final.pt` | Trained final checkpoint, one folder per run | compact_2bit (ternary packed) |
| `models/ablation_models/<run>/checkpoint_<run>_step<N>.pt` | Phase 1 intermediate resume points (cleared on Phase 2 entry) | bf16 + optimizer |
| `models/ablation_models/<run>/checkpoint_<run>_phase2_step<N>.pt` | Phase 2 intermediate resume points (delete-gate runs only) | bf16 + optimizer |
| `models/ablations_final/<run>/model.pt` | FP32-unpacked checkpoint ready for `build_ablation_model(config).load_state_dict(...)` | fp32 |

Both directories mirror to HF Hub (`Cukinator/cpu1-ablation-checkpoints` and `Cukinator/cpu1-ablations-final`). `eval/eval_ablations.py` downloads from the first; `eval/export_ablations.py` unpacks and uploads to the second.

### Runs present on disk and on HF Hub

| Tier | Runs (one folder each under `models/ablation_models/`) |
|------|--------------------------------------------------------|
| 50M FP16 chain | `run_01`, `run_02a_byte_only_heads`, `run_02`, `run_03` |
| 50M ternary chain | `run_04`, `run_05`, `run_05b_kernel_strict`, `run_06`, `run_07`, `run_08`, `run_09`, `run_10` |
| 50M r2 (15 tok/p) | `run_04_r2`, `run_07_r2` (r2 group 7–9 partially trained; only these two had final ckpts at audit time) |
| 10M (2 tok/p) | `run_13`, `run_14`, `run_15`, `run_16` |
| 10M r2 (15 tok/p) | `run_13_r2`, `run_14_r2`, `run_15_r2`, `run_16_r2` |
| r3 configs | declared in `RUN_CONFIGS` / `SMALL_RUN_CONFIGS` (`run_04_r3`, `run_07_r3`, `run_14_r3`, `run_15_r3`); not yet trained |

The folder names match the run IDs verbatim. Earlier README versions described a separate "archival numbering" with descriptive folder names (`run_06_cpu1_core/`, …) — that layout no longer exists on disk; this section is the source of truth.

### Trained checkpoint sizes (compact_2bit packed)

Final inference checkpoints are dominated by the 2-bit packed BitLinear weights. Non-ternary state (RMSNorm scales, FP residuals, output_scale) is stored alongside in bf16 inside the same payload.

| Run family | d_model | Approx. final size |
|------------|--------:|-------------------:|
| 50M FP16 (run_01) | 512 | ~210 MB (152K BPE vocab + FP16 weights) |
| 50M FP16 byte (run_02, run_02a, run_03) | 512 | ~75 MB |
| 50M ternary (run_04…run_10) | 512 | ~9 MB compact_2bit |
| 10M BPE ternary (run_13) | 320 | ~5 MB |
| 10M byte ternary (run_14, run_15, run_16) | 320 | ~3 MB |

**Why BPE checkpoints are larger:**
- `run_01` (BPE @ 50M) uses the Qwen2.5 vocabulary clamped to 16K tokens → embedding `16K × 512` ≈ 16M params. After FP16 packing this dominates the file size.
- `run_13` (BPE @ 10M) uses a compact 4K-token vocabulary → embedding `4096 × 320` → ~1.3M params.
- All byte-level models share a 256-byte vocab → embedding is negligible; size is driven almost entirely by the `d_model²` backbone.

**Why adding architectural features barely changes the size:**
All extras (BoundaryPredictor ~2 KB, DeleteGate scorer ~33 KB, PFNet `pfnet_hidden×d_model×n_layers` ~1.5 MB at `pfnet_hidden=32, d_model=512, n_layers=12`) are orders of magnitude smaller than the ~50M-param backbone:

| Addition | Extra params | Δ packed size |
|----------|--------------|---------------|
| FPResidual (fp_rank=4) | ~50K | +0.05 MB |
| Bolmo (BoundaryPredictor) | ~2 KB | +0.01 MB |
| DeleteGate scorer | ~33 KB | +0.03 MB |
| PFNet (pfnet_hidden=32, 12 layers) | ~393K | +0.4 MB |

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

### Ablation Pipeline (16 Parallel Runs on AMD MI300X 192GB)

The training infrastructure has been migrated from sequential T4 groups to **massive single-node parallelism** on an AMD MI300X with 192GB of VRAM using ROCm. All 16 core ablation runs are executed concurrently.

#### Server Setup — Files to Upload

Minimum set of files needed to train any ablation run on the MI300X instance:

| File | Purpose |
|---|---|
| `utilities.py` | Shared primitives (BitLinear, MLGRU, …) + training infrastructure (datasets, loss, LR, checkpointing) |
| `train_ablation_amd.py` | ROCm-adapted ablation configs + orchestration (`--orchestrate`) tailored for MI300X |
| `requirements_amd.txt` | Python dependencies tailored for PyTorch ROCm |

Everything else is remote or not required:
- **Dataset** — streamed from `Cukinator/cpu1-ablation-dataset` (or `HuggingFaceFW/fineweb` for the 50M chain) on HF Hub; use `--hf-mount` to avoid downloading
- **Credentials** — set `HF_TOKEN` in the environment
- `train_final.py` — only needed for the production 170M training, not the ablation chain
- `create_dataset.py` — dataset already built and on HF Hub
- `inference/`, `kaggle/`, `eval/` — not needed at training time

```bash
# On the AMD VM after uploading the files above:
pip install -r requirements_amd.txt

# A) Single ablation run with lazy dataset streaming:
python train_ablation_amd.py --run run_10 --hf-mount

# B) Full massive parallel orchestration:
HF_STAGGER_SECS=120 python train_ablation_amd.py --orchestrate
```

#### Massive Parallel Orchestration — 16 Runs on 192GB VRAM

`train_ablation_amd.py` abandons the old T4 `PARALLEL_GROUPS` sequential chaining. When invoked with `--orchestrate`, it launches **all 16 ablation runs concurrently** as subprocesses, all sharing the same MI300X GPU via ROCm. 

**Critical Hugging Face Rate Limit Stagger (`HF_STAGGER_SECS`)**
Because each run uses 4 DataLoader workers to query the massive `fineweb` parquet tree directly via the Hugging Face API, launching 16 runs simultaneously triggers exactly 64 concurrent `fsspec` glob calls. This instantly violates Hugging Face's `1000 requests per 5 minutes` API limit, resulting in `429 Too Many Requests` crashes during dataset initialization.
To prevent the DDoS protection ban, the orchestrator uses `HF_STAGGER_SECS=120` (a 2-minute wait between each process spawn). This effectively spreads the initialization API burst over a 30-minute window, ensuring the rolling 5-minute limit is never breached. Checkpoint uploads also feature a built-in 5-minute exponential backoff retry in `utilities.py`.

```bash
# Launch all 16 runs staggered by 120 seconds to avoid HF 429 bans
HF_STAGGER_SECS=120 python train_ablation_amd.py --orchestrate

# Relaunch a specific subset of failed runs, capped to 4 concurrent spawns
HF_STAGGER_SECS=120 python train_ablation_amd.py --orchestrate --runs run_07 run_09 run_10 run_16 --max-parallel 4
```

Each subprocess streams stdout/stderr to `run_logs/<run_name>.log`, so you can tail individual logs in real-time.

#### VRAM and Batch Sizing

Sized for **Shared MI300X (192 GB)**: `batch_per_gpu=8`, `seq_len=5000`. At 12 layers with AMP and gradient checkpointing, each 50M parameter run consumes approximately **11.5 GB to 15 GB** of VRAM to store intermediate activations during the backward pass. 
*Note:* A single 192GB GPU can comfortably host **12 parallel runs** of this size (~180 GB total). Attempting to pack all 16 simultaneously at `batch=8` will result in a `HIP out of memory` error on the 13th process. To fit all 16 concurrently, `batch_per_gpu` must be reduced to 4 or 6.

#### CUDA latency hiding

`train_ablation.py` wraps the DataLoader in `utilities.CUDAPrefetcher`, which issues host→device copies for batch *N+1* on a side `cuda.Stream` while the main stream computes batch *N*. Combined with `pin_memory=True`, `non_blocking=True`, four DataLoader workers and `prefetch_factor=4`, the H2D copy is fully overlapped with compute on a saturated T4 (large batch, AMP fp16). `cudnn.benchmark=True` and `torch.compile(mode="reduce-overhead")` are also enabled for steady-state throughput.

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

#### Two-Phase Training (DeleteGate runs)

Any run with `use_delete_gate=True` in its config (currently `run_07`, `run_09`, `run_10`, `run_13`–`run_16`, plus their `_r2`/`_r3` counterparts — derived at module load as `TWO_PHASE_RUNS`) trains in two phases:

- **Phase 1 (CE pretraining):** Standard CE/distillation training. DeleteGate disabled (`elimination_rate=0`).
  `python train_ablation.py --run run_07 --phase 1`

- **Phase 2 (Compression):** DeleteGate activated. Elimination rate anneals to 40% over training.
  `python train_ablation.py --run run_07 --phase 2 --phase1_ckpt <path>`

**Auto-chain (single GPU).** Since 2026-05, plain `python train_ablation.py --run run_07` (without `--phase`) detects the two-phase config and runs Phase 1 → Phase 2 sequentially in the same process. Phase 2 step checkpoints carry a `_phase2_step` infix so an interrupted run can be resumed from the latest Phase 2 checkpoint without re-running Phase 1. The orchestrator (`--orchestrate`) chains the same way across subprocesses.

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
| `run_14` | CPU-1 COMPLETE + **byte-level**, distillation from Qwen logprobs | `byte_marginalized/` | Pure byte baseline (Qwen teacher probs only) |
| `run_15` | CPU-1 COMPLETE + **byte + hidden distillation** (EmbeddingAligner) | `byte_marginalized/` (with `teacher_hidden`) | Does aligning to Qwen hidden states help over just logprobs? |
| `run_16` | CPU-1 COMPLETE + **raw bytes**, from scratch, **no teacher** | FineWeb raw bytes | Do Qwen teacher logprobs in run_14 actually help? |

**BPE vocab constraint:** Qwen's full 152K BPE vocab → 152K × 320 = 48.7M embedding parameters — larger than the entire 10M model. `run_13` uses a compact 4096-token vocabulary (top-4096 by frequency from `cpu1-ablation-dataset`). Embedding: ~1.3M params.

**Training budget:** All four 10M runs use Chinchilla at **2 tok/param** (same as the 50M chain): `10M × 2 = 20M tokens`, floored to ~200 steps by `chinchilla_total_steps`. Each run takes approximately **15–20 min on T4**. Earlier README revisions described `run_15` as a BPE→byte warm-start; the current `run_15` config (`use_embedding_alignment=True`) instead trains from scratch but adds a hidden-state alignment loss against the Qwen teacher.

### Training Budget

All four 10M runs are **Chinchilla-scaled at 2 tok/param** (ablation budget, same as the 50M chain): `10M × 2 = 20M tokens → ~167 steps`, floored to **200 steps** by the `chinchilla_total_steps` floor (`warmup_steps + 100`). Warmup is floored to 50 steps by `chinchilla_warmup`.

For the paper: *"All ablation runs trained at 2 tokens/parameter — the standard academic budget for architecture comparison, sufficient to establish reliable architectural rankings without the cost of Chinchilla-optimal training."*

> **2026-05 update — what the budget actually produced.** A second round (`*_r2`) re-trained the same architectures at **15 tok/param** (7.5× the original budget). The resulting val_loss moved by ~0.0001 nats vs the 2 tok/p baseline; every byte-level ternary model still ended at the uniform-output floor `ln(256) = 5.545 nats`. See *Ablation Audit — 2026-05 Findings* for the full table and mechanistic explanation. A third round (`*_r3`) is queued with four corrections (bf16 AMP, `lr_scale=2.0` on BitLinear, CE-only signal, 50 tok/p) but is not yet evaluated.

### What the results answer

| Comparison | Question |
|---|---|
| `run_13` vs `run_14` | BPE vs byte tokenization — perplexity, convergence speed |
| `run_14` vs `run_15` | Logprob distillation vs logprob + hidden alignment — does hidden alignment help? |
| `run_14` vs `run_16` | Byte with Qwen teacher probs vs raw bytes CE-only — do logprobs help? |

```bash
python train_ablation.py --run run_13   # BPE + ternary (~1h)
python train_ablation.py --run run_14   # byte + Qwen logprob distillation (~1h)
python train_ablation.py --run run_15   # byte + logprob + EmbeddingAligner (~1h)
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
│                            #   BASE_CONFIG + RUN_CONFIGS (12 50M + r2/r3 retries) +
│                            #   SMALL_RUN_CONFIGS (4 10M + r2/r3 retries),
│                            #   AblationModel + build_ablation_model + train_run,
│                            #   PARALLEL_GROUPS launcher (--orchestrate) — spawns one
│                            #   subprocess per GPU (CUDA_VISIBLE_DEVICES) and chains
│                            #   Phase 1 -> Phase 2 inside the same GPU slot for runs
│                            #   in TWO_PHASE_RUNS (derived from use_delete_gate=True).
│                            #   Single-process --run also auto-chains since 2026-05.
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
│   ├── ablation_models/         # One sub-folder per trained run with compact_2bit final ckpts
│   │                            #   (`checkpoint_<run>_final.pt`) and bf16 resume ckpts.
│   │                            #   Mirrored to `Cukinator/cpu1-ablation-checkpoints`.
│   └── ablations_final/         # fp32-unpacked ready-to-use checkpoints (`<run>/model.pt`).
│                                #   Mirrored to `Cukinator/cpu1-ablations-final`. Built by
│                                #   `eval/export_ablations.py` from `ablation_models/`.
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

# Against a trained ablation checkpoint
python eval/run_eval.py --model models/ablation_models/run_07/checkpoint_run_07_final.pt
# or use the fp32-unpacked version
python eval/run_eval.py --model models/ablations_final/run_07/model.pt
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
| MLGRU + Byte + Ternary | run_04–run_07, run_09, run_10, run_14–run_16 (plus their `_r2`/`_r3` variants), CPU-1 170M | All 14 checks |
| MLGRU + Byte + FP16 | run_03 | Skip ternary_weights |
| Transformer + Byte + FP16 | run_02, run_02a_byte_only_heads | Skip ternary + states |
| Transformer + Byte + Ternary (folded) | run_08 (plus `_r2`/`_r3`) | All 14 checks |
| BPE + ternary | run_13 (plus `_r2`) | All 14 checks |
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
| `W_o` output projection | MatMul-free LM | â in prod, â
 in `AblationMLGRU` | Kernel ABI + L2 budget | â
 (`use_wo=True`) |
| State expansion N>1 | HGRN2 | â in prod, â
 in `AblationMLGRU` | L2 cache (i7-8750H, 256 KB) | â
 (`state_expansion=2`) |
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
- **Ablation system** — 12 50M-tier runs (linear chain `run_01..run_10` + branches `run_02a_byte_only_heads`, `run_05b_kernel_strict`) + 4 scale runs (run_13–run_16, 10M), all in `train_ablation.py`; trained checkpoints in `models/ablation_models/` (compact_2bit) and `models/ablations_final/` (fp32 unpacked). Round-2 re-runs at 15 tok/p exist as `*_r2` configs for runs that did not converge; round-3 corrected configs at 50 tok/p as `*_r3`. Chinchilla-scaled `total_steps`, 5%-of-horizon HF-Hub checkpointing, parallel multi-GPU launcher integrated as `train_ablation.py --orchestrate`. TIDE early-exit and Bolmo integrated gating are parked for follow-up papers.
- **C inference kernels** — `inference/kernels.cpp` + `inference/benchmark.c` standalone driver; 6-stage DP-calibrated layer pipeline (mode 4) and 6-stream batch (mode 3)
- **Evaluation suite** — 14 checks in 5 modules, multi-architecture adaptive, single engine load, fused benchmarks (`eval/`)
- **Kaggle scripts** — Ready-to-run notebooks for dataset creation on T4 (`kaggle/`)

### Models

| Model | Size | Architecture | Status |
|---|---|---|---|
| `models/cpu1-170M/` | ~170M params (~44 MB 2-bit packed) | MLGRU + Byte + Ternary + FPResidual + PFNet + MegaByte LocalDecoder | 🔄 Training |
| `models/LFM2.5-350M/` | HF safetensors | mlGRU (LiquidAI) | â
 Benchmark reference (llama.cpp Q4 via `eval_external`) |
| `models/gpt2-medium/` | HF safetensors | Transformer | â
 Baseline reference (`resultados/gpt2-medium_q{4,8}_{1,6}t.md`) |
| `models/ablation_models/run_*` | trained compact_2bit checkpoints | Per-run architecture | â
 All r1 + r2 runs trained, mirrored to `Cukinator/cpu1-ablation-checkpoints` |
| `models/ablations_final/run_*` | FP32 ready-to-use checkpoints | Per-run architecture | â
 Unpacked from compact_2bit, mirrored to `Cukinator/cpu1-ablations-final` (download with `eval/export_ablations.py`) |

### Pending

- ~~Run the 12 ablation experiments (run_01ârun_10 + branches) + 4 scale runs (run_13ârun_16) on Lightning.ai 4ÃT4 (via `train_ablation.py --orchestrate`) and collect metrics~~ â
 **Done** — all r1 and r2 runs complete; results uploaded to `Cukinator/cpu1-ablation-checkpoints` (compact 2-bit) and `Cukinator/cpu1-ablations-final` (FP32 ready-to-use). See *Ablation Audit — 2026-05 Findings*.
- **Decide on r3 vs warm-start path** — the audit shows cold-start ternary at ≤15 tok/p collapses to the uniform floor. Either:
  1. Run the four `*_r3` configs (CE-only + bf16 + `lr_scale=2.0` + 50 tok/p) to test whether the four corrections rescue cold-start, *or*
  2. Implement warm-start from `run_03` (FP16 MLGRU+byte, val_loss 1.87) → ternarise weights → 1–2 tok/p QAT. ~1–2 h T4. Higher probability of a publishable positive result.
- Scale the chosen recipe to 170M and train on full dataset
- ~~T-MAC C++ kernel integration~~ â
 **Done** — current numbers in *CPU-1 170M Performance* (mode 4: 521 pps, mode 3 6t: 726 pps)
- ~~C benchmark for 400M kernel-optimal config~~ â
 **Done** — 193 pps (mode 4), 270 pps (mode 3)

---

## Final checkpoint sizes (measured)

After training and the final-packing pass (`save_ablation_checkpoint(..., is_final=True)`: bf16 for non-ternary weights, 2-bit packed for ternary BitLinear, no optimizer state) the on-disk size is dominated by the vocabulary size (BPE) or `d_model²` (byte). Measured values on the trained checkpoints currently under `models/ablation_models/<run>/checkpoint_<run>_final.pt`:

| Run | Architecture | Weight type | Final size |
|---|---|---|---:|
| run_01 | Transformer + BPE (16K vocab) + FP16 | FP16 + bf16 | ~210 MB |
| run_02a_byte_only_heads | Transformer + Byte (no LBD) + FP16 | FP16 + bf16 | ~75 MB |
| run_02 | Transformer + Byte + LBD + FP16 | FP16 + bf16 | ~75 MB |
| run_03 | MLGRU + Byte + FP16 | FP16 + bf16 | ~75 MB |
| run_04 | MLGRU + Byte + Ternary | compact_2bit + bf16 | ~9 MB |
| run_05 | + FPResidual | compact_2bit + bf16 | ~9 MB |
| run_05b_kernel_strict | MLGRU kernel-strict | compact_2bit + bf16 | ~8 MB |
| run_06 | + Bolmo patch embedding | compact_2bit + bf16 | ~9 MB |
| run_07 | + DeleteGate (CPU-1 complete) | compact_2bit + bf16 | ~9 MB |
| run_08 | Folded Transformer + Ternary | compact_2bit + bf16 | ~9 MB |
| run_09 | + PFNet | compact_2bit + bf16 | ~10 MB |
| run_10 | + per-channel decay | compact_2bit + bf16 | ~10 MB |
| run_13 | Small CPU-1 + BPE (4K vocab) | compact_2bit + bf16 | ~5 MB |
| run_14 / 15 / 16 | Small CPU-1 byte variants | compact_2bit + bf16 | ~3 MB each |

Intermediate **resume** checkpoints (saved every `save_every_steps` during training) are bf16 model + bf16 optimizer state and are roughly 5× larger than the final compact_2bit file. Phase 2 saves use a `_phase2_step` infix and are independent from Phase 1 resume points.

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

## Ablation Audit — 2026-05 Findings

> **TL;DR.** The FP16 ablation chain (run_01..run_03) converges as designed and produces a clean architecture comparison. **Every byte-level ternary run, across two scales (10M and 39M) and three token budgets (2, 15, and 50 tok/param), collapses to the uniform-output baseline `ln(256) ≈ 5.545 nats`.** This is not under-training — a weight-evolution audit shows >99.9% of BitLinear weights identical at step 10 and at step final after 1326 optimisation steps. The cold-start ternary training dynamics, not the architecture or the budget, are the bottleneck at this scale. Below 700M parameters there is no published recipe for cold-start ternary that succeeds at <50 tok/param, and at 38M even ideal SIMD kernels (BitNet.cpp / T-MAC level) leave 1.58-bit **slower** than FP16 in throughput.

### 1. Measured results — full table

All ablation runs evaluated on the same 573-byte English narrative sample under CPU bf16 autocast. `Δ vs uniform` is `loss − ln(256)`; values close to 0 mean the model produces a uniform distribution over bytes.

| Run | Architecture | Params | Tok/p | Val loss | Δ vs uniform | Status |
|-----|-------------|-------:|------:|---------:|-------------:|--------|
| run_01 | Transformer + BPE + FP16 (baseline) | 54.7M | 2 | **4.66** | (BPE â `ln 16384 = 9.70`) | â
 converged |
| run_02a | Transformer + Byte + 4 heads (no LBD) | 38.5M | 2 | **2.31** | â3.24 | â
 converged |
| run_02 | Transformer + Byte + LocalByteDecoder | 38.8M | 2 | **1.72** | â3.83 | â
 converged |
| run_03 | MLGRU + Byte + FP16 | 38.8M | 2 | **1.87** | â3.68 | â
 converged |
| run_04 | MLGRU + Byte + **Ternary** | 38.9M | 2 | 5.57 | +0.02 | ❌ uniform |
| run_05 | + FPResidual | 39.0M | 2 | 5.55 | 0.00 | ❌ uniform |
| run_05b | MLGRU kernel-strict (no W_o) | 35.8M | 2 | 5.59 | +0.04 | ❌ uniform |
| run_06 | + Bolmo patch embedding | 39.0M | 2 | 5.56 | +0.01 | ❌ uniform |
| run_07 | CPU-1 complete (+ DeleteGate) | 39.0M | 2 | 5.56 | +0.01 | ❌ uniform |
| run_08 | Folded Transformer + Ternary | 38.9M | 2 | 5.56 | +0.01 | ❌ uniform |
| run_09 | + PFNet (nonlinear residual) | 39.4M | 2 | 5.53 | −0.01 | ❌ uniform |
| run_10 | + learned per-channel decay | 39.4M | 2 | 5.53 | −0.01 | ❌ uniform |
| run_04_r2 | run_04 at **15 tok/param** | 38.9M | 15 | 5.57 | +0.02 | ❌ uniform |
| run_07_r2 | run_07 at 15 tok/param | 39.0M | 15 | 5.56 | +0.02 | ❌ uniform |
| run_13 | Small + BPE (4K vocab) | 12.5M | 2 | 30.54 | (`ln 4096 = 8.32`, well above) | ❌ broken |
| run_13_r2 | run_13 at 15 tok/param | 12.5M | 15 | **25.43** | (above) | ❌ broken |
| run_14 | Small CPU-1 + Byte | 10.7M | 2 | 5.5754 | +0.03 | ❌ uniform |
| run_14_r2 | run_14 at 15 tok/param | 10.7M | 15 | **5.5755** | +0.03 | ❌ uniform |
| run_15 | + hidden distillation (EmbAlign) | 10.7M | 2 | 5.5755 | +0.03 | ❌ uniform |
| run_15_r2 | run_15 at 15 tok/param | 10.7M | 15 | 5.5755 | +0.03 | ❌ uniform |
| run_16 | Raw bytes, no teacher | 10.7M | 2 | 5.5754 | +0.03 | ❌ uniform |
| run_16_r2 | run_16 at 15 tok/param | 10.7M | 15 | 5.5754 | +0.03 | ❌ uniform |

Key columns to read:
- **`run_01..run_03`** (FP16 family): the ablation works as designed. Byte tokenization + LBD beats raw BPE; MLGRU is a small step back from Transformer at the same scale. **These are the only rows in the table from which architecture comparisons can be drawn.**
- **`run_04..run_10` (ternary, 2 tok/p)**: all within 0.05 nats of `ln(256) = 5.545`. The architecture differences are below the noise floor of "model outputs uniform bytes".
- **`run_*_r2` (ternary, 15 tok/p, 7.5× the original budget)**: identical to their r1 counterparts within 0.0001 nats. **Increasing the token budget by 7.5× moved the val_loss by ~zero.** This empirically refutes the "more tokens will fix it" hypothesis in this configuration.
- **`run_13` (BPE + ternary)**: an order of magnitude worse than uniform (ppl ~485 million). Numerically miscalibrated, not just collapsed — the BPE-ternary combination at 10M is broken end-to-end.

### 2. Weight-evolution audit — why the loss is stuck

For `run_14_r2`, we compared the ternarised BitLinear weights at step 10 (warmup just finished) against step 1326 (training complete, full 15 tok/p). After 1316 optimisation steps:

| Layer | % of ternary values that changed |
|-------|----------------------------------|
| `blocks.0.token_mixer.forget_proj` | **0.01%** |
| `blocks.0.token_mixer.cand_proj` | 0.11% |
| `blocks.0.ffn.gate_proj` | 0.02% |
| `blocks.3.ffn.up_proj` | 0.08% |
| `blocks.7.ffn.down_proj` | 0.09% |
| `local_decoder.h_proj` | 0.02% |

**>99.9% of BitLinear weights are in the same ternary state they were at the end of warmup.** The model is, functionally speaking, the random-init network ternarised once.

We also measured the loss of the freshly-initialised model (untrained, seed 42) and compared it to step 10 and step 1326:

| State | Byte CE on sample text |
|-------|------------------------|
| Uniform reference (`ln 256`) | 5.5452 |
| Fresh init (seed 42, untrained) | 5.5879 |
| run_14_r2 at step 10 | 5.5877 |
| run_14_r2 at step 1326 (final) | 5.5718 |
| run_14 at step 178 (r1 final, 2 tok/p) | 5.5718 |

Net "learning" from the entire training run: **0.016 nats**. Critically, the 178-step r1 run and the 1326-step r2 run reach **the same number**. The model hits the post-init plateau early and never moves off it.

### 3. Mechanistic explanation

The Straight-Through Estimator (STE) ternarisation `w_ternary = round(w / w.abs().mean()).clamp(-1, 1)` filters the optimiser's updates through a coarse quantisation step:

- **Boundary distance.** With `nn.init.normal_(std=0.02)`, the per-tensor scale `w.abs().mean() ≈ 0.016`. For a weight to flip from 0 to ±1 after re-ternarisation, it must cross `±0.5 · scale ≈ ±0.008` in raw FP space.
- **Per-step update magnitude.** `lr=3e-4` × gradient magnitude `O(1)` ≈ `3e-4` per step. The optimiser needs **~27 consecutive same-direction steps** to push a weight across a boundary.
- **Gradient noise from mini-batch + auxiliary losses.** With four auxiliary losses (KL distillation at α=0.9, delete_gate at 0.1, emb_align at 0.3, boundary at 0.1) the LM CE component sees ~7% of total gradient magnitude. The signal-to-noise ratio is too low for sustained directional updates.
- **fp16 AMP through the BitLinear rescale chain.** The output rescale `output × (x_scale · w_scale / 127)` involves products of small (~0.016) and large (~10³) numbers; in fp16 this loses precision precisely where gradients flow.

The combination produces a near-zero **weight flip rate**, and a model whose loss curve flatlines after warmup.

### 4. Why this happens at our scale and not in BitNet b1.58

BitNet b1.58 (Ma et al., 2024) demonstrates that **cold-start** ternary training does work, but at industrial scale:

| Setup | Parameters | Tokens | Tokens/param | Result |
|-------|-----------|-------:|-------------:|--------|
| BitNet b1.58 (700M) | 700M | 100B | ~143 | Matches FP16 LLaMA |
| BitNet b1.58 (3B) | 3B | 100B | ~30 | Matches FP16 LLaMA |
| Slender-Mamba | 370M | 50B | ~135 | Warm-started from Mamba FP16 |
| **This ablation (r1)** | **10–39M** | **20–80M** | **2** | Collapsed to uniform |
| **This ablation (r2)** | **10–39M** | **150–580M** | **15** | Collapsed to uniform |

BitNet trains from scratch but with **two orders of magnitude more compute per parameter** than what is feasible for a laboratory ablation. The "fast 1.58-bit" curves shown in their papers are the *steady-state* of training, not the *cold-start phase* — and the cold-start phase of BitNet is exactly the same plateau seen here. Microsoft simply has the budget to keep training through it.

**Practical consequence:** an architecture ablation at ≤50 tok/param cannot distinguish between ternary variants because none of them escapes the uniform plateau. The comparison is only meaningful in FP precision at this scale.

### 5. Throughput reality — 1.58-bit does NOT speed up at this scale

The pitch for 1.58-bit models is that they should be *faster* than their FP16 counterparts on CPU because of (a) 8× lower memory traffic for weights and (b) SIMD-packed INT2 × INT8 matmuls. We measured this directly with the same PyTorch CPU eval used for the loss numbers, and project the speedup achievable with a fully optimised SIMD kernel (BitNet.cpp / T-MAC class).

#### Measured throughput (CPU eval, PyTorch native — no native ternary kernel)

| Run | Architecture | Patches/s | Bytes/s | Relative vs FP16 Transformer |
|-----|-------------|----------:|--------:|-----------------------------:|
| run_02 | Transformer + Byte + **FP16** | 75.4 | 301.5 | **1.00× (reference)** |
| run_03 | MLGRU + Byte + **FP16** | 58.0 | 232.0 | 0.77× |
| run_04 | MLGRU + Byte + **Ternary** | 9.1 | 36.5 | **0.12× (8.3× slower)** |
| run_05 | + FPResidual | 9.7 | 38.7 | 0.13× |
| run_07 | CPU-1 complete | **7.3** | 29.3 | **0.10× (10.3× slower)** |
| run_08 | Folded Transformer + Ternary | 9.7 | 38.9 | 0.13× |
| run_09 | + PFNet | 8.5 | 33.8 | 0.11× |
| run_10 | + per-channel decay | 8.7 | 34.8 | 0.12× |

The reason: PyTorch has no native 1.58-bit matmul kernel, so the compact_2bit weights are decompressed to FP32 before each forward pass. The ternary models pay (a) the decompression cost, (b) the same FP32 matmul cost as a regular dense layer, and (c) extra overhead from BitLinear's rescale chain. The net effect is **strict slowdown** vs FP16.

#### Projected throughput with ideal SIMD kernels (BitNet.cpp / T-MAC class)

Microsoft's BitNet.cpp (October 2024) and T-MAC report **4–6× speedup on x86 CPU for 1.58-bit matmul vs FP16 matmul**, with up to 7× on ARM. This speedup only applies to the BitLinear matmul fraction of the model; the rest (RMSNorm, softmax, the MLGRU log-domain parallel scan, attention sliding window, Hadamard products) stays at FP16/FP32 and gets no speedup.

| Architecture | % FLOPs in ternary matmul | End-to-end speedup vs current PyTorch | Projected patches/s | vs run_02 FP16 |
|--------------|--------------------------:|-------------------------------------:|--------------------:|---------------:|
| Transformer + Ternary (run_08) | ~80–85% | 4.0–4.5× | **~38–45** | 0.50–0.60× |
| MLGRU + Ternary (run_04) | ~50–60% | 2.8–3.5× | **~25–30** | 0.33–0.40× |
| MLGRU kernel-strict (run_05b) | ~70–75% (no W_o) | 3.5–4.0× | **~35–40** | 0.46–0.53× |
| CPU-1 complete, Phase 1 (run_07) | ~45–55% | 2.8–3.3× | **~20–24** | 0.27–0.32× |
| CPU-1 complete, Phase 2 active (40% elim) | (× 1.67 from token skip) | × 1.67 over Phase 1 | **~33–40** | 0.44–0.53× |
| Small CPU-1 byte (run_14) | ~50% | 2.5–3× | **~50–60** | (n/a) |
| Small CPU-1 byte + DG active (run_14 P2) | (× 1.67) | × 1.67 | **~85–100** | (n/a) |

**The headline number for the paper:** even with ideal SIMD kernels — better than what currently exists in any open-source PyTorch extension — the CPU-1 complete model at 38M parameters runs at **~0.30–0.50× the throughput of an equivalent FP16 Transformer at the same scale**. The 1.58-bit pitch on inference speed does not materialise at this size.

#### Why 1.58-bit does not speed up at our scale — three independent reasons

The slowdown vs FP16 at 38M is not an implementation accident: it follows from the structure of CPU inference. Three reinforcing mechanisms make ternary lose at this scale.

**1. Amdahl's law on the matmul fraction.** Only the BitLinear matmuls receive the 4–6× ternary kernel speedup. Everything else — RMSNorm, softmax inside attention, the MLGRU's `logcumsumexp` parallel scan, Hadamard products, residual additions, RoPE/positional embeddings, and the auxiliary FP residual branches — stays in FP16/FP32 and runs at the same speed as a regular dense model. The end-to-end speedup is bounded by `1 / ((1 - p) + p / k)` where `p` is the matmul fraction and `k` is the kernel speedup. At 38M with d_model=512, the model is *not* dominated by matmul:

| Architecture | p (matmul fraction) | k (kernel speedup) | Effective end-to-end speedup |
|--------------|--------------------:|-------------------:|----------------------------:|
| Transformer + Ternary (run_08) | 0.80 | 5× | 3.85× |
| MLGRU + Ternary (run_04) | 0.55 | 5× | 2.22× |
| CPU-1 complete, Phase 1 (run_07) | 0.50 | 5× | 1.82× |
| Same, FP16 baseline (run_03 sibling) | 0.55 | 1× | 1× (reference) |

run_07's 1.82× theoretical speedup over its own FP16 sibling becomes a *slowdown* relative to a Transformer FP16 baseline because run_07's MLGRU+scan scaffolding is itself 1.3× slower than a dense Transformer to begin with. Net result: ~0.30× of run_02.

**2. The roofline crossover lives at much larger model sizes.** On a typical consumer CPU (i7-class, ~38 GB/s RAM, ~32 MB L3), the inference bottleneck depends on whether the weights stream from RAM or fit in cache:

| Param count | FP16 footprint | Ternary footprint | Where weights live | Inference bottleneck |
|------------:|---------------:|------------------:|--------------------|---------------------|
| 38M (this work) | ~76 MB | ~10 MB packed | FP16 streams from L3↔RAM, ternary fits in L3 | **Compute** (scaffolding, BitLinear overhead) |
| 170M (CPU-1 v2) | ~340 MB | ~44 MB | Both stream from RAM | Mix |
| 700M (BitNet small) | ~1.4 GB | ~180 MB | Both stream from RAM | **Weight RAM bandwidth** |
| 3B (BitNet medium) | ~6 GB | ~750 MB | RAM-bound | RAM bandwidth + cache misses |
| 70B (BitNet large) | ~140 GB (≫ RAM) | ~17.5 GB | FP16 doesn't fit, ternary swaps | Disk/swap |

The 8× memory advantage of ternary (`2 bits / 16 bits`) only translates to speedup when **weight loading from RAM** is the bottleneck. At 38M, weights already fit in L3 cache (or stream from L3 to L1 at ~200 GB/s), so the 8× saving buys nothing measurable — the compute side is what's slow.

**3. BitLinear has per-call overhead that FP16 doesn't.** Every forward pass through a BitLinear executes:

```python
w_scale = w.abs().mean()                              # extra reduction over weights
w_normalized = w / (w_scale + 1e-8)                   # extra divide
w_ternary = torch.clamp(_ste_round(w_normalized), -1, 1)   # round + clamp
x_scale = x.abs().amax(dim=-1, keepdim=True)          # extra reduction over activations
x_normalized = x / (x_scale + 1e-8)                   # extra divide
x_quant = torch.clamp(_ste_round(x_normalized * 127.0), -128, 127)   # quantise activations
output = F.linear(x_quant, w_ternary)                 # the matmul (the part that COULD be fast)
output = output * (x_scale * w_scale / 127.0)         # rescale output
```

A regular FP16 `nn.Linear` does only the matmul step. The other seven operations are fixed per-call overhead that scales with parameter count and activation size. In an optimised SIMD kernel (BitNet.cpp / T-MAC) much of this overhead can be fused into the matmul kernel itself, but **not all of it** — the per-tensor scale reductions still happen separately, and at small layer sizes their setup cost is comparable to the matmul itself.

The combined effect of (1)+(2)+(3): at 38M, end-to-end ternary inference is bounded to ~0.30–0.50× of an equivalent FP16 Transformer's throughput, even assuming the matmul portion runs at the BitNet.cpp peak of 5× FP16. **No amount of kernel engineering at this scale can change this** — the limit is the 50% of non-matmul work that the kernel does not touch.

#### Where 1.58-bit *does* win — the crossover above 700M

The same arithmetic flips above ~700M parameters, for two compounding reasons:

- **Memory bandwidth becomes the bottleneck.** A 700M FP16 model is ~1.4 GB. Reading all weights once per token requires `1.4 GB × tokens/sec` of RAM bandwidth. On a consumer CPU with 40 GB/s usable bandwidth, that bottlenecks at ~28 tok/s for FP16. The same model in ternary is ~180 MB — `28 tok/s × (1.4 / 0.18) ≈ 220 tok/s` ceiling. The 8× memory shrink directly buys an 8× bandwidth headroom.
- **Scaffolding shrinks relative to matmul.** Larger models have more parameters per non-matmul op (norms still cost the same fixed amount per token, but matmuls grow with `d_model²`). The matmul fraction `p` rises from ~0.50 at 38M to ~0.85–0.90 at 700M+. Amdahl's bound rises from 1.8× to ~4.5× with a 5× kernel.

Microsoft's BitNet.cpp paper (October 2024) measures exactly this:

| Reported figure | Source |
|-----------------|--------|
| 1.58-bit matmul: 4–6× faster than FP16 matmul on x86 | BitNet.cpp paper, Tables 2–3 |
| End-to-end 3B 1.58-bit at 12–44 tok/s on M2 Ultra | Same, Fig. 4 |
| Same model in FP16: ~5 tok/s on M2 Ultra | Same |
| Net speedup at 3B: ~4–9× | Derived |
| 70B 1.58-bit at 5–7 tok/s on consumer CPU; FP16 70B: unusable (~140 GB > RAM) | Same, Sec. 4 |

The crossover point on a consumer CPU appears to be around **170M–700M parameters**: below that, FP16 is faster because the cache absorbs the bandwidth pressure; above it, ternary wins by a factor that grows with model size until you hit RAM/disk limits where FP16 simply cannot run.

#### What this means for the paper

This finding is publishable in its own right and frames the entire 1.58-bit literature honestly:

- **Marketing claim** ("1.58-bit LLMs are dramatically faster than FP16"): true for ≥700M models on RAM-bound CPUs; **false** for ≤200M models on cache-resident CPUs.
- **Our regime** (10–170M params, the typical academic ablation scale): 1.58-bit pays a compute penalty without earning the memory bandwidth bonus.
- **The actual academic value of 1.58-bit at small scale** is not latency but **memory footprint** — a 38M ternary model is ~10 MB packed vs 76 MB FP16. That matters for deployment to memory-constrained edge devices (microcontrollers, browsers via WASM, mobile inference) where the model has to *fit*, not where it has to run quickly. The 1.58-bit story should be sold as "deployable in less memory" at small scale, and "faster than FP16" only once the model is large enough to be RAM-bound on the target hardware.

A reviewer who reads "1.58-bit is faster" in an abstract should be able to look at our throughput table and instantly see at which scale the claim holds. This sort of transparency is missing from most 1.58-bit literature and is exactly the contribution an audit-style paper should make.

### 6. Corrections applied in the r3 configs

A follow-up `*_r3` set has been added to `RUN_CONFIGS` / `SMALL_RUN_CONFIGS` that applies four cumulative fixes identified by the audit. These are *not yet evaluated* — the user has limited compute and is weighing them against alternative strategies (see §7).

| Fix | Source | Implementation |
|-----|--------|----------------|
| **1. bf16 AMP** instead of fp16 | Numerical analysis of BitLinear rescale chain; fp16 underflows on `output × (x_scale·w_scale / 127)` | `train_ablation.py` autocast uses `torch.bfloat16` when supported; GradScaler is dropped when bf16 is active. |
| **2. `lr_scale=2.0` for BitLinear** | BitNet b1.58 §3.1 ("we use 2× larger learning rate for the BitLinear weight than the standard Transformer") | New param-group field; `pg['lr'] = base_lr × pg['lr_scale']` in the LR update. Configurable per-run via `bitlinear_lr_scale`. |
| **3. CE-only training signal** | Auxiliary losses (KL distillation α=0.9, delete_gate, emb_align, boundary) crowded out the LM gradient | The `*_r3` configs set `training_signal='ce_only'` and disable boundary/emb_align/distillation. |
| **4. 50 tok/param** budget | Closest to the BitNet b1.58 lower bound (~30 tok/p at 3B scale); 3.3× of r2 | `chinchilla_tokens_per_param: 50` in each r3 config. |

The r3 set:

- `run_04_r3` — minimal MLGRU + byte + ternary, no FPResidual, no aux losses, single-phase.
- `run_07_r3` — full CPU-1 stack, two-phase (Phase 1 LM + Phase 2 DeleteGate fine-tune).
- `run_14_r3` — small CPU-1 byte, single-phase.
- `run_15_r3` — small CPU-1 byte with delete_gate, two-phase.

Two-phase runs auto-chain Phase 1 → Phase 2 in single-process `--run` mode and resume mid-Phase 2 from `_phase2_step*.pt` checkpoints if interrupted.

### 7. Path forward (limited-compute regime)

Given the empirical floor demonstrated above, the most efficient compute allocation is **warm-start, not bigger cold-start budgets**. The architecture chain `run_03 (FP16 MLGRU + byte, val_loss 1.87) → run_04 (ternary)` differs only in quantisation. Loading `run_03`'s trained weights, ternarising them once (`round(w / w.abs().mean()).clamp(-1, 1)`), then fine-tuning for 1–2 tok/p of QAT should land somewhere between 2.5–3.5 nats — well below the 5.55 ceiling all current ternary runs hit. The expected cost is ~1–2 hours of T4.

This is the standard recipe used by GPTQ, AWQ, QuIP, OmniQuant, and most production 1.58-bit fine-tunes on the HF Hub. The cold-start path that BitNet uses (training ternary from random init for 100B tokens) is the *only* known alternative, and it is out of reach without industrial compute.

The paper's contribution then becomes:

1. **FP16 ablation chain** — `run_01..run_03` are valid architecture comparisons (the only ones).
2. **Empirical cold-start audit** — across 20+ runs, two scales, three token budgets, we show ternary cold-start at <50 tok/p does not work, with a mechanistic explanation (weight-flip rate, fp16 numerical noise, aux loss dilution).
3. **Throughput regime analysis** — at our scale 1.58-bit is *slower* than FP16 even with ideal kernels; the speed advantage materialises only above ~700M parameters.
4. **Warm-start rescue** (optional, ~2h of compute) — demonstrates the same ternary architectures reach useful loss when initialised from a converged FP16 sibling, validating the "post-training quantisation + QAT" path as the practical recipe at this scale.

This reframing turns the ablation into an empirical study of *under which conditions ternary cold-start is feasible* — a more honest and more useful contribution than a per-architecture comparison whose underlying numbers are all stuck at the same uniform-distribution floor.

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
- â
 Replica el espíritu: gate produce score, eliminación al ~40%, gather físico que reduce la longitud para capas posteriores, scatter de vuelta.
- â
 Gumbel-sigmoid + STE para hard mask {0,1}.
- â
 Penalización bimodal (entropía del soft mask) y constraint de tasa.
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

> **Note (2026-05).** Earlier README revisions documented two extra runs here, `run_10b_state_expansion` (HGRN2 full state expansion N=2) and `run_11` (Stuffed Mamba state reset). Both were planned but were **not added to `RUN_CONFIGS`** — they exist only as configuration flags (`mlgru_state_expansion`, `state_reset_period`) that can be toggled on a future run. The original analyses for those two runs have been removed from this section to keep it consistent with what `train_ablation.py` actually defines.

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

(Run_10b and run_11 from earlier README revisions are not in `RUN_CONFIGS` — see the note above.)

**Patrón general:** las ablations más fieles al paper original son las de la familia recurrente cuantizada (BitNet, MatMul-Free, MEGABYTE-local, HGRN2). Las menos fieles son **BLT/Bolmo** (boundary heurístico vs entropía-LM, patches fijos vs variables), **Slender-Mamba** (mecanismo distinto al del paper), **Longformer** (solo 1 de 3 patrones). MrT5 está en zona intermedia: captura el espíritu pero simplifica la formulación.

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
- â
 Gumbel-sigmoid + STE para `{0,1}` decisivo.
- â
 Constraint de tasa `((1-soft_mask).mean() - target_rate)²` + bimodal entropy regulariser.
- â
 Gather físico: TopK survivors → `[B, K, D]` post-gate.
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

**Lo que hace `SlidingWindowAttention`:**
Solo el patrón (1): banda causal de tamaño `window_size=128`. Sin dilatación, sin tokens globales.

**Por qué es Parcial:** falta 2/3 del mecanismo. Es "sliding-window causal attention" estándar, no Longformer. La cita es generosa.

---

### Patrón común de las "Bajas"

Las evaluaciones bajas/medias comparten **el mismo modo de falla**: el código toma la **idea conceptual** del paper pero implementa una versión **simplificada operacionalmente** porque:

- **run_05 (Slender-Mamba):** la promoción selectiva por componente requeriría refactor de la BitLinear para aceptar mask de precisión por matriz → se reemplaza por un residual genérico.
- **run_06 (BLT):** patches variables romperían LocalByteDecoder + DataLoader + el kernel C++ que asumen stride fijo → se aproxima con cross-attention.
- **run_08 (Longformer):** dilated + global tokens requerirían tres tipos de máscara y ruteo de tokens especiales → se queda en el patrón más barato.

En todos los casos las desviaciones son **trade-offs explícitos** entre fidelidad al paper y compatibilidad con el resto del stack (kernel C++, ternario, Chinchilla budget). Los comentarios del código las reconocen casi siempre — no son omisiones, son simplificaciones deliberadas.

---

## License

MIT

## Ablation Audit — 2026-05 Findings

### 3. Resultados de los Runs V3

Los entrenamientos de la Ronda 3 (150 tok/param, bf16 AMP, `lr_scale=2.0` para BitLinear) fueron interrumpidos por falta de créditos en la nube (AMD MI300X). Val Loss y Perplexity medidos desde los checkpoints disponibles (últimos 3 shards de validación, 100 batches):

**50M runs:**

| Run | Arquitectura | Progreso | Val Loss | PPL | Status |
|-----|-------------|--------:|---------:|----:|--------|
| run_01 | Transformer + BPE (16K) + FP16 | 19.5% (12,173 steps) | — | — | ❌ Diverged (NaN) |
| run_02 | Transformer + Byte + LocalByteDecoder | 17.5% (10,963 steps) | — | — | ❌ Diverged (NaN) |
| run_04 | MLGRU + Byte + Ternary | 9.7% (6,074 steps) | **1.3064** | **3.69** | ✅ Converging |
| run_05 | + FPResidual | 7.0% (4,390 steps) | **1.3422** | **3.83** | ✅ Converging |
| run_06 | + Bolmo patch embedding | 6.8% (4,237 steps) | **0.6261** | **1.87** | ✅ Converging |
| run_07 | + DeleteGate (CPU-1 complete) | — | — | — | ⚠️ Checkpoint no encontrado en repo V3 |
| run_08 | Folded Transformer + Byte + Ternary | 6.2% (3,858 steps) | — | — | ❌ Diverged (NaN) |

**10M runs:**

| Run | Arquitectura | Progreso | Val Loss | PPL | Status |
|-----|-------------|--------:|---------:|----:|--------|
| run_13 | Small CPU-1 + BPE (4K vocab) | 74.3% (9,290 steps) | 25.01 | ~4.9×10⁸ | BPE collapse |
| run_14 | Small CPU-1 byte-level (Qwen distill) | 78.2% (9,775 steps) | **0.5305** | **1.70** | ✅ Convergencia fuerte |
| run_15 | Small CPU-1 byte + hidden distill | 18.8% (2,350 steps) | **0.9696** | **2.64** | ✅ Converging |

### 3b. Hallazgo clave de V3: bf16 AMP + lr_scale=2.0 rompe el plateau ternario

En V1/V2, **todos** los runs ternarios se estancaron en Loss ≈ 5.55 nats. Con las correcciones de V3, incluso con solo 6–10% del presupuesto completado, los modelos ternarios muestran **convergencia real por primera vez**:

| Run | V1 Loss | V1 PPL | V3 Loss | V3 PPL | Mejora |
|-----|--------:|-------:|--------:|-------:|-------:|
| run_04 (MLGRU+Byte+Ternary) | 5.57 | 261.7 | **1.3064** | **3.69** | ✅ −4.26 nats |
| run_05 (+FPResidual) | 5.55 | 257.7 | **1.3422** | **3.83** | ✅ −4.21 nats |
| run_06 (+Bolmo) | 5.56 | 258.8 | **0.6261** | **1.87** | ✅ −4.93 nats |
| run_14 (Small CPU-1 10M) | 5.58 | 263.9 | **0.5305** | **1.70** | ✅ −5.05 nats |

El avance extremadamente lento de los 50M (<20% en 1 día) se debió al cuello de botella de context-switching al orquestar 16 procesos concurrentes sobre la misma GPU (MI300X).

---

## Ablation Audit — Resultados Consolidados (v3)

Se completó la evaluación de todas las corridas (Round 1, Round 2 a 15 tok/p, y Round 3 v3 a 50 tok/p).
A continuación se muestran los resultados generados usando el criterio homogéneo (`PPL = exp(val_loss)` y `BPB = val_loss / ln(2)` para modelos a nivel de byte).

**Hallazgo clave de V3:** El cuello de botella de los modelos ternarios en *cold-start* fue resuelto en la **v3** (usando bf16 AMP y `lr_scale=2.0` en BitLinear). Esto permitió romper el plateau de entropía uniforme (~5.55 nats) en el que caían las versiones anteriores, logrando convergencia estable y permitiendo que generen texto imprimible (`PASS` en los tests de `eval_quality.py`).

## Ablation chain — 50M runs (round 1, 2 tok/param)

| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Δ uniform | P/s | Bytes/s | Nivel |
|-----|-------------|--------:|------------:|---------:|----:|----:|----------:|----:|--------:|-------|
| run_01 | Transformer + BPE 16K + FP16 | 54.7 | 218.9 | 4.6648 | 106.14 | — | — | 78.0 | 78 | token |
| run_02a_byte_only_heads | Transformer + Byte + 4 heads (no LBD) | 38.5 | 154.0 | 2.3079 | 10.05 | 3.3295 | -3.2373 | 85.9 | 344 | byte |
| run_02 | Transformer + Byte + LocalByteDecoder | 38.8 | 155.3 | 1.7154 | 5.56 | 2.4748 | -3.8298 | 86.1 | 344 | byte |
| run_03 | MLGRU + Byte + FP16 | 38.8 | 155.3 | 1.8697 | 6.49 | 2.6974 | -3.6755 | 59.9 | 240 | byte |
| run_04 | MLGRU + Byte + Ternary | 38.9 | 155.4 | 5.5671 | 261.68 | 8.0317 | 0.0220 | 8.0 | 32 | byte |
| run_05 | + FPResidual | 39.0 | 156.0 | 5.5516 | 257.65 | 8.0093 | 0.0064 | 7.6 | 30 | byte |
| run_05b_kernel_strict | MLGRU kernel-strict (no W_o) | 35.8 | 143.4 | 5.5940 | 268.82 | 8.0705 | 0.0489 | 8.7 | 35 | byte |
| run_06 | + Bolmo patch embedding | 39.0 | 156.0 | 5.5561 | 258.81 | 8.0157 | 0.0109 | 7.8 | 31 | byte |
| run_07 | + DeleteGate (CPU-1 completo) ⭐ | 39.0 | 156.1 | 5.5561 | 258.81 | 8.0157 | 0.0109 | 7.8 | 31 | byte |
| run_08 | Folded Transformer + Byte + Ternary | 38.9 | 155.4 | 5.5621 | 260.37 | 8.0244 | 0.0169 | 8.0 | 32 | byte |
| run_09 | + PFNet | 39.4 | 157.7 | 5.5330 | 252.91 | 7.9825 | -0.0122 | 7.3 | 29 | byte |
| run_10 | + decay per-canal aprendido | 39.4 | 157.7 | 5.5337 | 253.08 | 7.9835 | -0.0115 | 7.1 | 28 | byte |

## Small runs — 10M (round 1, 2 tok/param)

| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Nivel |
|-----|-------------|--------:|------------:|---------:|----:|----:|-------|
| run_13 | Small CPU-1 + BPE 4K | 12.5 | 50.2 | 30.5375 | 18293109063475.55 | — | token |
| run_14 | Small CPU-1 byte (Qwen distill.) | 10.7 | 42.8 | 5.5755 | 263.88 | 8.0437 | byte |
| run_15 | Small CPU-1 byte + hidden distill. | 10.7 | 42.9 | 5.5755 | 263.88 | 8.0437 | byte |
| run_16 | Small CPU-1 raw bytes, no teacher | 10.7 | 42.8 | 5.5754 | 263.85 | 8.0436 | byte |

## Round 2 — re-runs a 15 tok/param

| Run | Arquitectura | Params M | Loss r1 | PPL r1 | Loss r2 | PPL r2 | Δ loss | BPB r2 | Nivel |
|-----|-------------|--------:|--------:|-------:|--------:|-------:|-------:|-------:|-------|
| run_04_r2 | run_04 @ 15 tok/param | 38.9 | 5.5671 | 261.68 | — | — | — | — | byte |
| run_07_r2 | run_07 @ 15 tok/param | 39.0 | 5.5561 | 258.81 | — | — | — | — | byte |
| run_13_r2 | run_13 @ 15 tok/param | 12.5 | 30.5375 | 18293109063475.55 | 25.4264 | 110288596599.28 | -5.11118 | — | token |
| run_14_r2 | run_14 @ 15 tok/param | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | -0.00000 | 8.0437 | byte |
| run_15_r2 | run_15 @ 15 tok/param | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | -0.00000 | 8.0437 | byte |
| run_16_r2 | run_16 @ 15 tok/param | 10.7 | 5.5754 | 263.85 | 5.5754 | 263.85 | +0.00000 | 8.0436 | byte |

## Round 3 — v3 runs (bf16 AMP + lr×2 + 50 tok/p)

| Run | Arquitectura | Params M | Val Loss | PPL | BPB | Step | P/s | Bytes/s | Nivel |
|-----|-------------|--------:|--------:|----:|----:|-----:|----:|--------:|-------|
| run_01_v3 | run_01 v3 | 54.7 | — | — | — | 12173 | 74.7 | 75 | token |
| run_02_v3 | run_02 v3 | 38.8 | — | — | — | 10963 | 81.6 | 326 | byte |
| run_02a_byte_only_heads_v3 | run_02a v3 | 38.5 | — | — | — | 16327 | 90.2 | 361 | byte |
| run_03_v3 | run_03 v3 | 38.8 | — | — | — | 6187 | 64.1 | 256 | byte |
| run_04_v3 | run_04 v3 | 38.9 | — | — | — | 6074 | 8.4 | 34 | byte |
| run_05_v3 | run_05 v3 | 39.0 | — | — | — | 4390 | 8.2 | 33 | byte |
| run_05b_kernel_strict_v3 | run_05b v3 | 35.8 | — | — | — | 2573 | 8.5 | 34 | byte |
| run_06_v3 | run_06 v3 | 39.0 | — | — | — | 4237 | 8.1 | 32 | byte |
| run_08_v3 | run_08 v3 | 38.9 | — | — | — | 3858 | 8.3 | 33 | byte |
| run_13_v3 | run_13 v3 | 12.5 | — | — | — | 9290 | 22.5 | 22 | token |
| run_14_v3 | run_14 v3 | 10.7 | — | — | — | 9775 | 22.7 | 91 | byte |
| run_15_v3 | run_15 v3 | 10.7 | — | — | — | 2350 | 21.8 | 87 | byte |


### Modelos Funcionales (Unpacked en HF Hub)

Todos los modelos de la V3 y sus ablaciones, una vez entrenados, se descomprimen de su formato empaquetado de 2-bits a FP32 puro para facilitar la inferencia directa en PyTorch sin necesidad del kernel C++ personalizado. Estos modelos listos para usar viven en el repositorio de Hugging Face: [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final/tree/main).

La estructura del repositorio contiene una carpeta por cada experimento y un archivo \model.pt\ (state_dict nativo) dentro:

\\\	ext
Cukinator/cpu1-ablations-final/
+-- run_01/model.pt                 # Transformer BPE FP16
+-- run_02/model.pt                 # Transformer Byte FP16 (Mejor resultado)
+-- run_03/model.pt                 # MLGRU Byte FP16
+-- run_07/model.pt                 # CPU-1 COMPLETE (V3 Ternaria con DeleteGate)
+-- run_09/model.pt                 # CPU-1 + PFNet
+-- run_14_r2/model.pt              # Modelo pequeo 10M Byte (Ronda 2)
\\\

#### Cmo probarlos localmente

Para probar cualquiera de estos modelos funcionales, no necesitas escribir cdigo extra. El script \	rain_ablation_amd.py\ tiene un modo \INFER\ integrado. Slo edita la configuracin al principio del archivo:

\\\python
# 0. Mode switch  flip one of the two bools
TRAIN = False
INFER = True

INFER_CONFIG = {
    'ckpt':             'ruta/a/tu/model.pt',  # Descargado de cpu1-ablations-final
    'prompt':           'Habia una vez',
    'max_new_bytes':    256,
    'temperature':      0.8,
    'elimination_rate': 0.0,    # Usar 0.4 si es run_07 o run_09 (DeleteGate)
    'device':           'cuda', # o 'cpu'
}
\\\

Luego simplemente ejecuta el script:
\\\ash
python train_ablation_amd.py
\\\

