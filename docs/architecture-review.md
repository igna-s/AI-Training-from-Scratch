# Architecture and README review — 2026-10-10

This review checks what the repository implements, what its published artifacts demonstrate, and which architectural names need qualification. It covers every architecture family named in the root README, all 34 published configuration records, all 30 current presets in each ablation entry point, and the separate larger-model path.

The reviewed training/component source is at commit `43902814b8d2f2e5459367b8f90a34d2f03bcbab`. The [machine-readable inventory](../experiments/evaluation/architecture-review-2026-10-10.json) includes source hashes, parameter counts, state-shape fingerprints, instantiated module counts, and effective feature flags. The documentation update does not change those model implementations or checkpoint files.

The inventory's source-file hashes cover the raw bytes of the local Windows checkout, including CRLF line endings. GitHub stores those source files with LF endings, so their Git blob bytes have different SHA-256 values even though the source text is unchanged.

## Method and boundaries

1. Read model constructors, forwards, dataset adapters, training/configuration routing, checkpoint serialization, and native drivers.
2. Construct each of the 34 recorded models in **both** ablation backends with meta tensors: **68 constructions**. Verify recorded configuration hashes and parameter counts, then compare inventories and state-shape fingerprints across backends.
3. Construct all **30 current presets per backend**: **60 constructions**. Record actual modules and flags instead of trusting descriptions or run names.
4. Construct **five larger-model variants**: base/gated classes with local or legacy independent heads, plus the gated class with all optional gates. These constructions validate structure without allocating complete parameter tensors.
5. Execute small synthetic CPU checks of all current presets in both backends, including phase 2 where a gate exists: **92 forward checks**. Also exercise three extra options and six larger-model forward cases. These use width 32, four layers, eight input steps, and random weights; they do not establish trained-model correctness, quality, GPU compatibility, or full-scale performance.
6. Reuse the completed offline recheck of **all 34 actual FP32 exports**: SHA-256, sizes, strict loading, parameter counts, saved steps, configuration agreement, exact frozen-corpus losses, prefix causality, and recurrent chunk checks. Every result matched the original CPU audit within the stated loss tolerance. Full generation and timing were not rerun for all 34 models.

No new model, teacher, or dataset weights were downloaded for these architecture checks. The source review and constructor tests are separate from the historical training jobs: exact job-to-source bindings and deterministic retraining remain unverified. The [validation record](../experiments/evaluation/readme-validation-2026-10-10.json) preserves the recheck results and synthetic-check scope.

Repeat the structural inventory from the repository root:

```powershell
python -m evaluation.architecture_review --output outputs/architecture-review.json
```

The AMD module's GPU availability selector is disabled during this CPU/meta inspection. The command neither tests nor installs ROCm/CUDA. Importing that existing module can load local `.env` settings, but the report contains architecture fields and hashes rather than environment credentials.

## Architecture claims and verdicts

| Name or claim | What the code actually does | Supported wording / limit |
|---|---|---|
| Training from scratch | Constructors initialize student tensors; training can subsequently resume or warm-start from sibling checkpoints | The project investigates student training from scratch. Do not claim that every saved artifact started fresh in the exact published source version |
| Transformer | `AblationBlock` combines `SlidingWindowAttention` (causal PyTorch SDPA, Q/K/V/O projections, optional sliding mask) with RMSNorm and a separate SwiGLU FFN | Custom causal-attention baselines; full-model causality still depends on the input embedding and numerical validity |
| MatMul-free / MLGRU | `AblationMLGRU` projects forget/candidate/output gates and updates a recurrent state; `MatMulFreeTokenMixer` is a separate serial implementation | MLGRU-inspired recurrent models. The PyTorch path still uses floating-point linear operations, and its local decoder uses attention |
| BitNet / 1.58-bit | `BitLinear` and `BitEmbedding` simulate a scaled ternary alphabet with straight-through gradients; linear activations also simulate quantization | Ternary quantization experiments. Neither every tensor nor every operation is integer/ternary; FP32 exports are not 1.58-bit stored models |
| MEGABYTE | Fixed global patches plus a small causal decoder over the next patch's byte positions | MEGABYTE-inspired global/local byte modeling; no demonstrated million-byte context or complete paper reproduction |
| Bolmo / BLT | `BolmoPatchEmbedding` uses a convolutional boundary scorer, multiplicative gates, and small cross-attention over a fixed patch window | A local boundary-aware embedding adaptation. It does not convert a pretrained token model into the original Bolmo architecture or implement BLT entropy-driven variable patches |
| `byte_dynamic` | A configuration label accepted by byte data/model routes; the recorded tensors and model embeddings retain patch size 4 | A label, not proof of variable-length patch allocation. `run_04_r2` even uses the ordinary fixed-patch embedding |
| Boundary prediction | `BoundaryPredictor` can provide heuristic byte-boundary scores and auxiliary BCE in ablations | An optional training auxiliary. It is absent from all 34 exports and inactive in the larger gated model's forward |
| MrT5 / deletion | Custom learned gate, hard/soft masks, neighbor merge, optional top-k gather and later scatter | MrT5-inspired state deletion, not the MrT5 encoder-decoder architecture. Presence of a gate or a configured rate does not demonstrate compression quality or speedup |
| FPResidual | Two floating-point linear projections with a narrow intermediate dimension and SiLU between them contribute an additive correction | A nonlinear floating-point bottleneck residual; no universal accuracy recovery or cache-residency guarantee |
| PFNet | Two **ternary** `BitLinear` layers with SiLU and RMSNorm form a small nonlinear branch | A separate nonlinear residual branch; calling it a floating-point MLP would be inaccurate |
| HGRN2 / learned decay | Optional per-channel decay with layer-dependent initialization, and a vector state widened through learned projections | HGRN2-inspired options, not a faithful HGRN2 reproduction. The initialization does not enforce the paper's hierarchical forget-gate bounds, and its parameter-free outer-product state expansion differs from this implementation |
| Stuffed-Mamba / periodic reset | A serial MLGRU option clears state every configured number of steps | A periodic state reset; not a demonstrated Mamba/selective-SSM architecture. All recorded exports have reset period 0 |
| Teacher distillation | CE/KL routes read Qwen teacher probabilities; optional cosine alignment uses projected teacher hidden states | Teacher-assisted student training. The teacher itself was pretrained elsewhere; full dataset-wide teacher correctness is not established |
| T-MAC / native CPU inference | Custom AVX2/FMA/OpenMP ternary kernels and synthetic drivers | LUT-inspired kernel experiments. No verified published-checkpoint loader/parity or end-to-end trained-model speed claim |
| Production model | `CPU1ModelV2` exists as a larger development path and is called “production” in source comments | Source comments are design intent. No matching evaluated large checkpoint, complete real-data training validation, or production deployment is established |
| Budget / Chinchilla labels | A configured tokens-per-parameter calculation sets training steps | A planned budget, not measured unique training exposure or verified compute-optimality |

Source anchors: [components](../cpu1/components.py), [CUDA-oriented ablations](../training/ablation.py), [ROCm-oriented ablations](../training/ablation_amd.py), [larger models](../training/final.py), [teacher pipeline](../data_pipeline/prepare.py), and [native kernels](../inference/kernels.cpp).

## Published model inventory

The manifest and actual constructed modules agree for **all 34 exports**:

| Feature | Published exports |
|---|---:|
| Sliding-window Transformer mixer | 8 |
| Ablation MLGRU mixer | 26 |
| BPE representation | 5 |
| Fixed four-byte representation | 29 |
| Two-layer local byte decoder | 27 |
| Independent byte heads | 2 |
| Ternary projection/embedding modules | 26 |
| Floating-point low-rank residuals | 21 |
| Boundary-aware `BolmoPatchEmbedding` | 14 |
| DeleteGate module | 15 |
| Hidden-state alignment module | 3 |
| PFNet branch | 2 |
| Learned per-channel decay | 1 (`run_10`) |
| Standalone BoundaryPredictor | 0 |
| State expansion factor greater than 1 | 0 |
| Periodic state reset enabled | 0 |

Every recorded configuration has `_current_phase=1`. Gate-enabled records also configure real gather, but the common audit explicitly uses phase 1 and elimination 0. This establishes module/configuration presence, not that phase-2 benefits were trained or measured. BPE and byte scores below use different units.

`Boundary` means the actual embedding contains its own boundary scorer/cross-attention. `FP` means low-rank FPResidual, `DG` DeleteGate, `PF` PFNet, `Decay` learned recurrent decay, and `Align` teacher-hidden alignment. `Ternary` describes simulated modules; checkpoint storage is FP32 in every row. `Finite` comes from the offline numerical recheck; the causal and chunk outcomes are the common audit's checks.

| Export / configuration | Mixer / input | Parameters | Projections | Head | Extra modules | Saved step | Finite | Causal | Chunks |
|---|---|---:|---|---|---|---:|---|---|---|
| [run_01](../experiments/published/configs/run_01.json) | Transformer / `bpe` | 54,735,360 | Float | BPE | — | 912 | Yes | PASS | N/A |
| [run_01_v3](../experiments/published/configs/run_01_v3.json) | Transformer / `bpe` | 54,735,360 | Float | BPE | — | 12,173 | No | ERROR | N/A |
| [run_02](../experiments/published/configs/run_02.json) | Transformer / `byte` | 38,829,569 | Float | Local 2L | — | 647 | Yes | PASS | N/A |
| [run_02_v3](../experiments/published/configs/run_02_v3.json) | Transformer / `byte` | 38,829,569 | Float | Local 2L | — | 10,963 | No | ERROR | N/A |
| [run_02a_byte_only_heads](../experiments/published/configs/run_02a_byte_only_heads.json) | Transformer / `byte` | 38,500,868 | Float | 4 independent | — | 641 | Yes | PASS | N/A |
| [run_02a_byte_only_heads_v3](../experiments/published/configs/run_02a_byte_only_heads_v3.json) | Transformer / `byte` | 38,500,868 | Float | 4 independent | — | 16,327 | No | ERROR | N/A |
| [run_03](../experiments/published/configs/run_03.json) | MLGRU / `byte` | 38,829,569 | Float | Local 2L | — | 647 | Yes | PASS | FAIL |
| [run_03_v3](../experiments/published/configs/run_03_v3.json) | MLGRU / `byte` | 38,829,569 | Float | Local 2L | — | 6,187 | Yes | PASS | FAIL |
| [run_04](../experiments/published/configs/run_04.json) | MLGRU / `byte` | 38,854,145 | Ternary | Local 2L | — | 647 | Yes | PASS | FAIL |
| [run_04_r2](../experiments/published/configs/run_04_r2.json) | MLGRU / `byte_dynamic` | 38,854,145 | Ternary | Local 2L | — | 4,840 | Yes | PASS | FAIL |
| [run_04_v3](../experiments/published/configs/run_04_v3.json) | MLGRU / `byte` | 38,854,145 | Ternary | Local 2L | — | 6,074 | Yes | PASS | FAIL |
| [run_05](../experiments/published/configs/run_05.json) | MLGRU / `byte` | 38,993,921 | Ternary | Local 2L | FP | 649 | Yes | PASS | FAIL |
| [run_05_v3](../experiments/published/configs/run_05_v3.json) | MLGRU / `byte` | 38,993,921 | Ternary | Local 2L | FP | 4,390 | Yes | PASS | FAIL |
| [run_05b_kernel_strict](../experiments/published/configs/run_05b_kernel_strict.json) | MLGRU / `byte` | 35,842,049 | Ternary | Local 2L | FP | 600 | Yes | PASS | FAIL |
| [run_05b_kernel_strict_v3](../experiments/published/configs/run_05b_kernel_strict_v3.json) | MLGRU / `byte` | 35,842,049 | Ternary | Local 2L | FP | 2,573 | Yes | PASS | FAIL |
| [run_06](../experiments/published/configs/run_06.json) | MLGRU / `byte_dynamic` | 38,994,146 | Ternary | Local 2L | Boundary, FP | 649 | Yes | FAIL | FAIL |
| [run_06_v3](../experiments/published/configs/run_06_v3.json) | MLGRU / `byte_dynamic` | 38,994,146 | Ternary | Local 2L | Boundary, FP | 4,237 | Yes | FAIL | FAIL |
| [run_07](../experiments/published/configs/run_07.json) | MLGRU / `byte_dynamic` | 39,028,579 | Ternary | Local 2L | Boundary, FP, DG | 650 | Yes | FAIL | FAIL |
| [run_07_r2](../experiments/published/configs/run_07_r2.json) | MLGRU / `byte_dynamic` | 39,028,579 | Ternary | Local 2L | Boundary, FP, DG | 4,860 | Yes | FAIL | FAIL |
| [run_08](../experiments/published/configs/run_08.json) | Transformer / `byte` | 38,854,145 | Ternary | Local 2L | — | 647 | Yes | PASS | N/A |
| [run_08_v3](../experiments/published/configs/run_08_v3.json) | Transformer / `byte` | 38,854,145 | Ternary | Local 2L | — | 3,858 | No | ERROR | N/A |
| [run_09](../experiments/published/configs/run_09.json) | MLGRU / `byte_dynamic` | 39,427,939 | Ternary | Local 2L | Boundary, FP, DG, PF | 657 | Yes | FAIL | FAIL |
| [run_10](../experiments/published/configs/run_10.json) | MLGRU / `byte_dynamic` | 39,434,083 | Ternary | Local 2L | Boundary, FP, DG, PF, Decay | 657 | Yes | FAIL | FAIL |
| [run_13](../experiments/published/configs/run_13.json) | MLGRU / `bpe` | 12,547,785 | Ternary | BPE | FP, DG | 600 | Yes | PASS | FAIL |
| [run_13_r2](../experiments/published/configs/run_13_r2.json) | MLGRU / `bpe` | 12,547,785 | Ternary | BPE | FP, DG | 1,568 | Yes | PASS | FAIL |
| [run_13_v3](../experiments/published/configs/run_13_v3.json) | MLGRU / `bpe` | 12,547,785 | Ternary | BPE | FP, DG | 9,290 | Yes | PASS | FAIL |
| [run_14](../experiments/published/configs/run_14.json) | MLGRU / `byte_dynamic` | 10,691,371 | Ternary | Local 2L | Boundary, FP, DG | 600 | Yes | FAIL | FAIL |
| [run_14_r2](../experiments/published/configs/run_14_r2.json) | MLGRU / `byte_dynamic` | 10,691,371 | Ternary | Local 2L | Boundary, FP, DG | 1,336 | Yes | FAIL | FAIL |
| [run_14_v3](../experiments/published/configs/run_14_v3.json) | MLGRU / `byte_dynamic` | 10,691,371 | Ternary | Local 2L | Boundary, FP, DG | 9,775 | Yes | FAIL | FAIL |
| [run_15](../experiments/published/configs/run_15.json) | MLGRU / `byte_dynamic` | 10,732,331 | Ternary | Local 2L | Boundary, FP, DG, Align | 600 | Yes | FAIL | FAIL |
| [run_15_r2](../experiments/published/configs/run_15_r2.json) | MLGRU / `byte_dynamic` | 10,732,331 | Ternary | Local 2L | Boundary, FP, DG, Align | 1,341 | Yes | FAIL | FAIL |
| [run_15_v3](../experiments/published/configs/run_15_v3.json) | MLGRU / `byte_dynamic` | 10,732,331 | Ternary | Local 2L | Boundary, FP, DG, Align | 2,350 | Yes | FAIL | FAIL |
| [run_16](../experiments/published/configs/run_16.json) | MLGRU / `byte_dynamic` | 10,691,371 | Ternary | Local 2L | Boundary, FP, DG | 600 | Yes | FAIL | FAIL |
| [run_16_r2](../experiments/published/configs/run_16_r2.json) | MLGRU / `byte_dynamic` | 10,691,371 | Ternary | Local 2L | Boundary, FP, DG | 1,336 | Yes | FAIL | FAIL |

Both published byte family widths are recorded explicitly: the main family uses width 512, 12 layers, and FFN width 1,376; the small family uses width 320, eight layers, and FFN width 853. Actual counts differ by embedding/head choices and optional branches. These rows do not prove that successive run numbers form a one-change controlled ablation.

## Causality and recurrent execution

All **14** published boundary-aware embeddings have `bolmo_causal_boundary=false`, `bolmo_causal_cross_patch=false`, and cross-patch window 3. The convolution and symmetric neighbor window can consume future input. All 14 corresponding exports fail the frozen prefix-causality probe. The scorer/cross-attention implementation is not causal merely because the global mixer is recurrent.

The shared `AblationMLGRU._parallel_scan` implementation disagrees with its defining serial recurrence. The frozen diagnostic's maximum differences are **19.1005** against serial and **11.4713** against chunked execution, with tolerance `1e-4`. All 26 recurrent exports fail the chunk test; 12 of them still pass the separate prefix-causality probe. The eight Transformer exports are not recurrent and have no recurrent chunk result.

The scan's cumulative maximum/sum construction fails to rescale accumulated terms correctly when its stabilizing maximum changes. Turning off parallel scan is a source option, not proof that a checkpoint trained with the defective scan retains its measured loss under a different numerical implementation. The separate larger model uses the serial mixer and needs its own full-scale numerical evaluation.

Transformer blocks do not implement a persistent KV cache in these wrappers. The generation helper recomputes retained context. An isolated single-input forward benchmark, local decoder evaluation, multi-stream kernel throughput, and actual generated bytes/second measure different work.

## Byte decoders and patching

The published decoder configurations omit an explicit `local_decoder_layers` key; their ablation constructor's effective default is **two layers**. The component class and the larger-model constructor default to **one layer**. Describing every path as the same one-layer architecture would be wrong.

The byte decoder receives global patch context, learned position embeddings, and shifted preceding target bytes, then applies causal intra-patch attention. Global patch `i` predicts patch `i+1`; a missing `byte_targets` argument creates zero placeholders. Correct shape output alone is not a likelihood test. Independent-head baselines lack this within-target-patch autoregressive conditioning.

The standalone `bytes_to_dynamic_patches` helper can create padded variable-boundary groups from supplied boundary flags. That helper is not proof that the recorded model/dataset forward uses it to allocate variable-length patches. All 29 byte exports use patch size 4.

## Gates, state options, and preset names

Mask deletion leaves sequence length unchanged. Gather deletion selects `K=max(floor(S*(1-rate)),1)` survivors, runs later global layers on the shorter sequence, and scatters representations back before output decoding; deleted positions retain their pre-gate representations. In the small synthetic gate tests, gather reduces the final global layer from eight to four steps at rate 0.5. Mask mode retains eight. Neither observation is a trained speedup measurement, and phase-2 causal behavior is not covered by the phase-1 audit.

Hidden alignment is a cosine loss on masked projected states. The three published alignment models use teacher dimension 128. Field existence does not prove that stored masks/hidden rows reach the correct target positions; the dataset review found alignment problems.

State expansion widens the vector from `d_model` to `d_model*factor` and adds learned projection dimensions. A synthetic factor-2 check returns expanded recurrent state of width 64 for a width-32 model. Periodic reset uses the serial path. Every published and current named preset has factor 1 and reset period 0, so these code options are not demonstrated by a trained expanded-state or reset checkpoint.

The learned-decay option also needs qualification: its layer-dependent initialization is not guaranteed to be monotonic. In a fresh 12-layer synthetic initialization, the effective decay falls from approximately 0.572727 to 0.385475 between adjacent layers. The source has no constraint maintaining a monotonic hierarchy during optimization; this is a source/initialization finding, not a measurement of a trained checkpoint's decay values.

Current preset inventories were checked in both entry points. Their structures agree, while configured budgets can differ: the main CUDA-oriented base uses two tokens/parameter and the AMD-oriented base uses 150, before per-run overrides. The 34 published configuration records remain a separate source of truth.

Names and descriptions require care:

- `run_04_r2` says `byte_dynamic` but instantiates `AblationBytePatchEmbedding`, without a boundary scorer.
- Current `run_14_r3` removes DeleteGate and trains CE on FineWeb; current `run_15_r3` has DeleteGate but no hidden aligner and also selects CE. They are not identical to the base KD/alignment runs.
- `_r3` descriptions refer to corrections, but those names do not establish that the recurrent defect or boundary causality is fixed. Current boundary-aware presets still inherit both causal flags as false.
- Current `run_14_r3` is marked proposed. Constructor/synthetic success does not make it a published trained result.
- `_v3` export records require `--config`; those suffixes are not current named presets.

## Larger model path

[training/final.py](../training/final.py) has two classes sharing `CPU1BlockV2`, `BytePatchEmbeddingV2`, and the **serial** `MatMulFreeTokenMixer`. They do not use `AblationMLGRU._parallel_scan`.

| Constructor | Default-size parameters | Head / optional modules |
|---|---:|---|
| `CPU1ModelV2` | 164,869,729 | One-layer local decoder |
| `CPU1ModelV2`, `legacy_heads=true` | 165,064,548 | Four independent byte heads |
| `CPU1ModelV2WithGates`, gates off | 164,869,729 | One-layer local decoder |
| `CPU1ModelV2WithGates`, gates off, legacy heads | 165,064,548 | Four independent byte heads |
| `CPU1ModelV2WithGates`, all three optional gates on | 168,611,875 | One-layer decoder, boundary module, deletion, alignment |

Defaults are width 1,024, 14 layers, FFN width 2,730, FPResidual rank 8, and PFNet hidden width 64. The ablation architecture's audited weights and results do not validate these separate defaults.

The gated class **instantiates `BoundaryPredictor` but never calls it in `forward`**. Small forward-hook checks confirmed zero calls in both phases and with either head choice. Its deletion path uses the mask route, not the ablation gather route. Its default alignment teacher dimension is **3,584**, whereas the recorded ablation aligners use 128; compatible data requires an explicit matching dimension.

The default real FineWeb training route builds an `IterableDataset` and passes `shuffle=true` to its DataLoader, causing a **ValueError before training** in the reviewed PyTorch environment. That failure was reproduced without downloading data. A tiny synthetic training-loop smoke succeeded during the README review; this does not validate full-size real-data training, memory limits, or throughput.

## Native kernels, storage, and reproducibility

`inference/kernels.cpp` and `inference/benchmark.c` explore native ternary operations, local decoding, and execution modes. Their drivers create synthetic weights. No trained-checkpoint loading/parity check is established, and a compatible GCC/MinGW build was not executed in this environment. Do not label their rates as published-model generation performance.

FP32 exports occupy four bytes per stored parameter plus serialization metadata. The compact codec stores ternary values using two bits, plus scales and floating-point residual/normalization tensors. `log2(3)` is alphabet capacity, not measured whole-model storage. The PyTorch inference path unpacks/simulates weights; it is not a packed integer backend.

The dataset check covers three pinned Parquet samples and synthetic reader/tokenizer cases. Dense teacher arrays and short boolean masks can be interpreted incorrectly by the adapter; a partial UTF-8 case also fails the first-byte mapping. No full audit of 917 shards or every intermediate training checkpoint is implied.

The precomputed loader's metadata-based 5 GiB shard selection can fall back to an entire subset. Post-load cache cleanup is an attempted 10 GiB policy, not a hard internet/disk limit. Missing split metadata makes its fallback validation slice overlap training. Exact dataset exposure and an independent training-time validation split cannot be reconstructed from configured budgets alone.

The checkpoint/configuration records do not identify every job's exact source, environment, resolved teacher/tokenizer revision, extraction version, document list, or random projection. Numerical reproduction of a small frozen inference audit is distinct from deterministic end-to-end retraining. The recovered-history map is dated, and later commit-subject changes need their own scope.

## References used to qualify names

Original papers document their own architectures and results; the local source and evidence above determine this project's claims.

- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528)
- [MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers](https://arxiv.org/abs/2305.07185)
- [Bolmo: Byteifying the Next Generation of Language Models](https://arxiv.org/abs/2512.15586)
- [Byte Latent Transformer: Patches Scale Better Than Tokens](https://arxiv.org/abs/2412.09871)
- [MrT5: Dynamic Token Merging for Efficient Byte-level Language Models](https://arxiv.org/abs/2410.20771)
- [HGRN2: Gated Linear RNNs with State Expansion](https://arxiv.org/abs/2404.07904)
- [T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge](https://arxiv.org/abs/2407.00088)
- [Distilling Token-Trained Models into Byte-Level Models](https://arxiv.org/abs/2602.01007)
