# Model architectures and implementation limits

This guide describes the **34 published ablation exports, with 10,691,371–54,735,360 parameters**, their architecture families, and known numerical limitations. It also distinguishes their recorded configurations from the 30 current presets in each ablation entry point. Architectural names identify local adaptations; they do not establish reproduction of the cited papers.

The implementation analysis uses training/component source at commit `43902814b8d2f2e5459367b8f90a34d2f03bcbab`. The [October 10, 2026 inventory](../experiments/evaluation/architecture-review-2026-10-10.json) includes source hashes, parameter counts, state-shape fingerprints, instantiated module counts, and effective feature flags.

The inventory's source-file hashes cover the raw bytes of the local Windows checkout, including CRLF line endings. GitHub stores those source files with LF endings, so their Git blob bytes have different SHA-256 values even though the source text is unchanged.

## Analysis methods

1. Source analysis covers constructors, forwards, dataset adapters, training/configuration routing, checkpoint serialization, and native drivers.
2. The 34 recorded models are constructed in **both** ablation backends with meta tensors: **68 constructions**. Recorded configuration hashes and parameter counts are checked, and module inventories and state-shape fingerprints are compared across backends.
3. All **30 current presets per backend** are constructed: **60 constructions**, recording effective modules and flags.
4. Small synthetic CPU tests cover all current presets in both backends, including phase 2 where a gate exists: **92 forward checks**. Three extra ablation options are also exercised. These use width 32, four layers, eight input steps, and random weights; they do not establish trained-model correctness, quality, GPU compatibility, or full-scale performance.
5. The offline recheck covers **all 34 actual FP32 exports**: SHA-256, sizes, strict loading, parameter counts, saved steps, configuration agreement, exact frozen-corpus losses, prefix causality, and recurrent chunk checks. Every result matched the original CPU evaluation within the stated loss tolerance. Full generation and timing were not rerun for all 34 models.

Meta-tensor construction checks structure without loading trained weights; the synthetic probes use small random initializations. These checks do not identify the source used by each historical training job or establish deterministic retraining. The [validation record](../experiments/evaluation/readme-validation-2026-10-10.json) contains the checkpoint recheck results and synthetic-check scope.

Repeat the structural inventory from the repository root:

```powershell
python -m evaluation.architecture_review --output outputs/architecture-review.json
```

The structural inventory uses CPU/meta tensors with GPU selection disabled. It does not validate CUDA or ROCm execution.

## Implemented components

| Architecture or feature | Implementation | Scope and limitations |
|---|---|---|
| Training from scratch | Constructors initialize student tensors; training can subsequently resume or warm-start from sibling checkpoints | Fresh initialization, resume, and transfer are supported; each artifact's original initialization and training source are not fully attributed |
| Transformer | `AblationBlock` combines `SlidingWindowAttention` (causal PyTorch SDPA, Q/K/V/O projections, optional sliding mask) with RMSNorm and a separate SwiGLU FFN | Custom causal-attention baselines; full-model causality still depends on the input embedding and numerical validity |
| MatMul-free / MLGRU | `AblationMLGRU` projects forget/candidate/output gates and updates a recurrent state | MLGRU-inspired recurrent models. The PyTorch path still uses floating-point linear operations, and its local decoder uses attention |
| BitNet / 1.58-bit | `BitLinear` and `BitEmbedding` simulate a scaled ternary alphabet with straight-through gradients; linear activations also simulate quantization | Ternary quantization experiments. Neither every tensor nor every operation is integer/ternary; FP32 exports are not 1.58-bit stored models |
| MEGABYTE | Fixed global patches plus a small causal decoder over the next patch's byte positions | MEGABYTE-inspired global/local byte modeling; no demonstrated million-byte context or complete paper reproduction |
| Bolmo / BLT | `BolmoPatchEmbedding` uses a convolutional boundary scorer, multiplicative gates, and small cross-attention over a fixed patch window | A local boundary-aware embedding adaptation. It does not convert a pretrained token model into the original Bolmo architecture or implement BLT entropy-driven variable patches |
| `byte_dynamic` | A configuration label accepted by byte data/model routes; the recorded tensors and model embeddings retain patch size 4 | A label, not proof of variable-length patch allocation. `run_04_r2` even uses the ordinary fixed-patch embedding |
| Boundary prediction | `BoundaryPredictor` can provide heuristic byte-boundary scores and auxiliary BCE in ablations | An optional training auxiliary. It is absent from all 34 exports; the integrated boundary-aware embedding is a different module |
| MrT5 / deletion | Custom learned gate, hard/soft masks, neighbor merge, optional top-k gather and later scatter | MrT5-inspired state deletion, not the MrT5 encoder-decoder architecture. Presence of a gate or a configured rate does not demonstrate compression quality or speedup |
| FPResidual | Two floating-point linear projections with a narrow intermediate dimension and SiLU between them contribute an additive correction | A nonlinear floating-point bottleneck residual; no universal accuracy recovery or cache-residency guarantee |
| PFNet | Two **ternary** `BitLinear` layers with SiLU and RMSNorm form a small nonlinear branch | A separate nonlinear residual branch; calling it a floating-point MLP would be inaccurate |
| HGRN2 / learned decay | Optional per-channel decay with layer-dependent initialization; the model clamps decay factors across layers before mixing. A separate option widens vector state through learned projections | HGRN2-inspired options. Ordered decay ceilings differ from the paper's hierarchical forget-gate lower bounds; learned vector projections differ from its parameter-free outer-product state expansion |
| Stuffed-Mamba / periodic reset | A serial MLGRU option clears state every configured number of steps | A periodic state reset; not a demonstrated Mamba/selective-SSM architecture. All recorded exports have reset period 0 |
| Teacher distillation | CE/KL routes read Qwen teacher probabilities; optional cosine alignment uses projected teacher hidden states | Teacher-assisted student training. The teacher itself was pretrained elsewhere; full dataset-wide teacher correctness is not established |
| T-MAC / native CPU inference | Custom AVX2/FMA/OpenMP ternary kernels and synthetic drivers | LUT-inspired kernel experiments. No verified published-checkpoint loader/parity or end-to-end trained-model speed claim |
| Budget / Chinchilla labels | A configured tokens-per-parameter calculation sets training steps | A planned budget, not measured unique training exposure or verified compute-optimality |

Source anchors: [components](../cpu1/components.py), [CUDA-oriented ablations](../training/ablation.py), [ROCm-oriented ablations](../training/ablation_amd.py), [teacher pipeline](../data_pipeline/prepare.py), and [native kernels](../inference/kernels.cpp).

## Reading an architecture

A byte model reads UTF-8 as values from 0 to 255 and groups them into fixed **four-byte patches**. An embedding turns each patch into a vector. The global layers process the patch sequence using either attention or recurrent state. The output head predicts the next patch: a local decoder predicts its four bytes sequentially, while independent heads predict each byte position from the same global context. A patch is four bytes, not necessarily four characters or a word.

A BPE model instead reads mapped Qwen token IDs and predicts the next mapped token through one vocabulary head. The published vocabulary caps are 16,384 for `run_01` and 4,096 for `run_13`. Out-of-range IDs become 0 in the reader/evaluator, so these are vocabulary-reduced token experiments.

The main family uses global width **512**, **12 layers**, and a SwiGLU FFN width of **1,376**. The small `run_13`–`run_16` family uses width **320**, **8 layers**, and FFN width **853**. Optional modules, vocabulary size, and output heads account for the different parameter counts within each family. Every published local byte decoder has width **128**, four attention heads, and **two layers**.

### Component glossary

| Term in the tables | What it does | Interpretation in this experiment |
|---|---|---|
| Transformer | Causal attention mixes information from the available earlier input positions; SwiGLU transforms features within each position | Custom attention baseline. A causal mixer cannot make a noncausal embedding safe |
| MLGRU | Learned forget, candidate, and output projections update a recurrent state | Global recurrence replaces global attention. Published exports use the parallel scan with the documented chunk defect |
| Float / `fp16` label | Ordinary floating-point projections without ternary weight simulation | `fp16` is the saved configuration label; the evaluated checkpoint tensors are FP32 |
| Ternary | `BitLinear`/`BitEmbedding` simulate scaled weights in the alphabet −1, 0, +1 | Quantization is simulated with floating-point operations. This does not mean every tensor is ternary or that exports use packed inference |
| Local 2L | A two-layer causal attention decoder predicts bytes within the next patch, conditioned on global context and preceding target bytes | Teacher forcing supplies shifted true bytes for scoring; generation samples the bytes sequentially |
| 4 independent | Four separate 256-way output heads, one for each target byte position | They do not condition a later target byte on an earlier target byte within that patch |
| BPE | One output head over the saved token-vocabulary cap | Per-token scores describe the mapped token stream and use different units from byte scores |
| Boundary | The embedding uses learned byte-boundary scores and cross-attention over neighboring patches | Patch size stays four. All 14 published uses allow future input and fail the causal probe |
| FP | `FPResidual`: two floating-point projections with a rank-4 bottleneck and SiLU add a correction to ternary mixer/FFN output | A nonlinear residual branch; its presence does not establish that it recovered accuracy |
| DG | `DeleteGate`: a learned module can mask or gather selected intermediate sequence states | Present in 15 exports, disabled in the common phase-1 evaluation. Compression gains are unmeasured |
| PF | `PFNet`: two ternary linear layers, SiLU, and RMSNorm add another nonlinear branch | Hidden width 32 in `run_09` and `run_10`; distinct from the floating-point FPResidual |
| Decay | Learned per-channel factors multiply the recurrent forget gates | Present only in `run_10`; factors are ordered across layers before mixing, but are not HGRN2's forget-gate lower bounds |
| Align | A projection maps teacher hidden features into student space for a masked cosine alignment loss | Present in the three `run_15` exports, with teacher dimension 128; useful alignment depends on correct data positions and masks |

## Run-by-run explanations

The table groups **all 34 exports into 16 named experiment families**. Each family name links to its base configuration; every individual variant has its own link in the [exact inventory](#published-model-inventory). The final column describes the intended investigation, not a benefit established by these results.

**CE** means next-symbol cross-entropy on raw FineWeb. **KD** means the recorded distillation route using the precomputed Qwen teacher dataset. Both describe configured supervision, rather than verified signal coverage in every historical training batch.

| Published family and exports | Architecture and components | Configured signal | Intended investigation |
|---|---|---|---|
| [run_01](../experiments/published/configs/run_01.json), `run_01_v3` | 54.74M; Float Transformer, 16,384-cap BPE input/output, 512 width / 12 layers | CE | Token-based attention reference for the byte experiments; vocabulary reduction limits its interpretation |
| [run_02](../experiments/published/configs/run_02.json), `run_02_v3` | 38.83M; Float Transformer, fixed four-byte patches, Local 2L | CE | Byte attention reference with global patch context and sequential local prediction |
| [run_02a_byte_only_heads](../experiments/published/configs/run_02a_byte_only_heads.json), `run_02a_byte_only_heads_v3` | 38.50M; Float Transformer, fixed bytes, four independent heads | CE | Explore byte prediction without the within-patch local decoder used by `run_02` |
| [run_03](../experiments/published/configs/run_03.json), `run_03_v3` | 38.83M; Float MLGRU, fixed bytes, Local 2L | CE | Explore recurrent global mixing in place of the Transformer mixer in `run_02` |
| [run_04](../experiments/published/configs/run_04.json), `run_04_r2`, `run_04_v3` | 38.85M; ternary MLGRU, fixed bytes, Local 2L; no FP residual | CE | Explore ternary projections in the recurrent byte path; `_r2` has a `byte_dynamic` label but the ordinary fixed-patch embedding |
| [run_05](../experiments/published/configs/run_05.json), `run_05_v3` | 38.99M; ternary MLGRU, fixed bytes, Local 2L + FP | CE | Explore a narrow floating-point correction branch in the ternary model |
| [run_05b_kernel_strict](../experiments/published/configs/run_05b_kernel_strict.json), `run_05b_kernel_strict_v3` | 35.84M; `run_05` structure without the MLGRU `W_o` output projection and its normalization | CE | Explore a reduced mixer structure. The name does not establish native-kernel compatibility or parity |
| [run_06](../experiments/published/configs/run_06.json), `run_06_v3` | 38.99M; ternary MLGRU, fixed bytes, Local 2L + Boundary + FP | CE | Explore boundary-aware input weighting and patch context; recorded settings are noncausal |
| [run_07](../experiments/published/configs/run_07.json), `run_07_r2` | 39.03M; ternary MLGRU, fixed bytes, Local 2L + Boundary + FP + DG | CE | Add a deletion module to the boundary-aware path; its phase-2 effect is outside the shared evaluation |
| [run_08](../experiments/published/configs/run_08.json), `run_08_v3` | 38.85M; ternary Transformer, fixed bytes, Local 2L; no FP residual | CE | Explore ternary attention alongside the floating-point `run_02` and recurrent `run_04` paths |
| [run_09](../experiments/published/configs/run_09.json) | 39.43M; `run_07` structure + PF, hidden width 32 | CE | Explore an additional nonlinear ternary branch |
| [run_10](../experiments/published/configs/run_10.json) | 39.43M; `run_09` structure + learned per-channel Decay | CE | Explore another control over recurrent state retention |
| [run_13](../experiments/published/configs/run_13.json), `run_13_r2`, `run_13_v3` | 12.55M; ternary MLGRU, 4,096-cap BPE input/output + FP + DG; 320 width / 8 layers | KD | Explore a small teacher-assisted token student; it has no local byte decoder or boundary-aware embedding |
| [run_14](../experiments/published/configs/run_14.json), `run_14_r2`, `run_14_v3` | 10.69M; ternary MLGRU, fixed bytes, Local 2L + Boundary + FP + DG; 320 width / 8 layers | KD | Explore a small teacher-assisted byte student |
| [run_15](../experiments/published/configs/run_15.json), `run_15_r2`, `run_15_v3` | 10.73M; `run_14` structure + Align, teacher feature dimension 128 | KD + hidden alignment | Explore teacher hidden-feature supervision in addition to output distributions |
| [run_16](../experiments/published/configs/run_16.json), `run_16_r2` | 10.69M; same model structure as `run_14`, without an aligner | CE | Explore the small byte architecture with no teacher signal; the data route also changes to raw FineWeb |

These families do not form a fully controlled ranking. Training budgets, saved steps, data routes, and sometimes several settings differ. In particular, `run_16` is a supervision/data comparison with `run_14`, not a new global mixer. The number sequence is an experiment identifier; there are no `run_11` or `run_12` exports in this 34-model collection.

### Run suffixes and training signals

The published records configure **2 tokens/parameter** for unsuffixed runs, **15** for `_r2`, and **150** for `_v3`. A larger planned budget does not prove more unique data, a completed run, convergence, numerical stability, or a corrected architecture. The saved step in the inventory is checkpoint metadata. Current presets can use different budgets and are separate from these recorded configurations.

The suffix alone is not an architecture specification. For example, `run_04_r2` changes the input label to `byte_dynamic` while retaining fixed patches and no boundary scorer. Explicit JSON configurations are required to select `_v3` exports; those identifiers are not current named presets.

The nine `run_13`/`run_14`/`run_15` exports configure KD with teacher signals from `Cukinator/cpu1-ablation-dataset`. The byte loss mixes teacher KL and CE on the first target-byte head where a teacher mask is true, and uses CE for the other three heads and uncovered positions. The BPE loss combines KL and CE where teacher IDs, probabilities, and masks are available. Recorded weights are `alpha_kl=0.9` and `alpha_ce=0.1` for covered positions; this is not a 90/10 mixture over every byte or batch. `run_15` also configures the hidden-alignment loss. Reader defects and incomplete teacher coverage prevent treating these flags as proof of successful distillation.

The other **25 exports** configure CE-only training on raw FineWeb. Their recorded `alpha_kl` fields do not enable KD when `training_signal=ce_only`. Qwen is a pretrained teacher/tokenizer reference; the student constructors initialize custom models and can subsequently resume or transfer checkpoints.

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

Module abbreviations are defined in the [component glossary](#component-glossary). `Ternary` describes simulated modules; checkpoint storage is FP32 in every row.

**Parameters** counts all constructed model parameters, including the output head and optional modules. **Saved step** is the checkpoint's recorded step, not a convergence statement. **Finite** reports finite loss on the frozen sample in the offline recheck. **Causal** reports the prefix probe; PASS applies to that check, FAIL means future input changed earlier predictions, and ERROR means the numerical check could not produce a valid result. **Chunks** compares recurrent full-sequence and chunked forwards; N/A identifies a Transformer without recurrent state. Finite loss and a causal PASS do not resolve a chunk FAIL or establish good generated text.

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

The scan's cumulative maximum/sum construction fails to rescale accumulated terms correctly when its stabilizing maximum changes. Turning off parallel scan is a source option, not proof that a checkpoint trained with the defective scan retains its measured loss under a different numerical implementation.

Transformer blocks do not implement a persistent KV cache in these wrappers. The generation helper recomputes retained context. An isolated single-input forward benchmark, local decoder evaluation, multi-stream kernel throughput, and actual generated bytes/second measure different work.

## Byte decoders and patching

The published decoder configurations omit an explicit `local_decoder_layers` key; their ablation constructor's effective default is **two layers**, verified in the inventories for all 27 local-decoder exports. Reading an isolated component's default is insufficient to identify the decoder used by a recorded run.

The byte decoder receives global patch context, learned position embeddings, and shifted preceding target bytes, then applies causal intra-patch attention. Global patch `i` predicts patch `i+1`; a missing `byte_targets` argument creates zero placeholders. Correct shape output alone is not a likelihood test. Independent-head baselines lack this within-target-patch autoregressive conditioning.

The standalone `bytes_to_dynamic_patches` helper can create padded variable-boundary groups from supplied boundary flags. That helper is not proof that the recorded model/dataset forward uses it to allocate variable-length patches. All 29 byte exports use patch size 4.

## Gates, state options, and preset names

Mask deletion leaves sequence length unchanged. Gather deletion selects `K=max(floor(S*(1-rate)),1)` survivors, runs later global layers on the shorter sequence, and scatters representations back before output decoding; deleted positions retain their pre-gate representations. In the small synthetic gate tests, gather reduces the final global layer from eight to four steps at rate 0.5. Mask mode retains eight. Neither observation is a trained speedup measurement, and phase-2 causal behavior is not covered by the phase-1 audit.

Hidden alignment is a cosine loss on masked projected states. The three published alignment models use teacher dimension 128. Field existence does not prove that stored masks/hidden rows reach the correct target positions; the dataset review found alignment problems.

State expansion widens the vector from `d_model` to `d_model*factor` and adds learned projection dimensions. A synthetic factor-2 check returns expanded recurrent state of width 64 for a width-32 model. Periodic reset uses the serial path. Every published and current named preset has factor 1 and reset period 0, so these code options are not demonstrated by a trained expanded-state or reset checkpoint.

The learned-decay initialization is not itself monotonic: a fresh 12-layer synthetic model has adjacent factors approximately 0.572727 and 0.385475. However, **both ablation models call `_enforce_decay_cummax` in `forward`**, applying a running minimum to `log_decay`. Since the factor is `exp(-softplus(log_decay))`, this makes the factors nondecreasing across layers before mixing. A small forward check confirmed that the initial decrease becomes approximately 0.572727 / 0.572727. The operation changes parameter values in place, including during evaluation.

This clamp orders multiplicative decay **ceilings**. The actual forget gate is `decay * sigmoid(forget_projection(x))`, so independently learned input gates need not be ordered. HGRN2 instead constructs increasing **lower bounds** for forget gates. The local clamp therefore does not reproduce that paper's mechanism or establish a trained hierarchy benefit. The small check concerns source behavior, not trained-checkpoint decay statistics.

Current preset inventories were checked in both entry points. Their structures agree, while configured budgets can differ: the main CUDA-oriented base uses two tokens/parameter and the AMD-oriented base uses 150, before per-run overrides. The 34 published configuration records remain a separate source of truth.

Configuration identifiers and effective modules:

- `run_04_r2` says `byte_dynamic` but instantiates `AblationBytePatchEmbedding`, without a boundary scorer.
- Current `run_14_r3` removes DeleteGate and trains CE on FineWeb; current `run_15_r3` has DeleteGate but no hidden aligner and also selects CE. They are not identical to the base KD/alignment runs.
- `_r3` descriptions refer to corrections, but those names do not establish that the recurrent defect or boundary causality is fixed. Current boundary-aware presets still inherit both causal flags as false.
- Current `run_14_r3` is marked proposed. Constructor/synthetic success does not make it a published trained result.
- `_v3` export records require `--config`; those suffixes are not current named presets.

## Native kernels, storage, and reproducibility

`inference/kernels.cpp` and `inference/benchmark.c` explore native ternary operations, local decoding, and execution modes. Their drivers create synthetic weights. No trained-checkpoint loading/parity check or compatible GCC/MinGW build is validated in the recorded environment. Their rates describe synthetic kernel work; published-model generation performance is not established.

FP32 exports occupy four bytes per stored parameter plus serialization metadata. The compact codec stores ternary values using two bits, plus scales and floating-point residual/normalization tensors. `log2(3)` is alphabet capacity, not measured whole-model storage. The PyTorch inference path unpacks/simulates weights; it is not a packed integer backend.

The dataset check covers three pinned Parquet samples and synthetic reader/tokenizer cases. Dense teacher arrays and short boolean masks can be interpreted incorrectly by the adapter; a partial UTF-8 case also fails the first-byte mapping. No full audit of 917 shards or every intermediate training checkpoint is implied.

The precomputed loader's metadata-based 5 GiB shard selection can fall back to an entire subset. Post-load cache cleanup is an attempted 10 GiB policy, not a hard internet/disk limit. Missing split metadata makes its fallback validation slice overlap training. Exact dataset exposure and an independent training-time validation split cannot be reconstructed from configured budgets alone.

The checkpoint/configuration records do not identify every job's exact source, environment, resolved teacher/tokenizer revision, extraction version, document list, or random projection. Numerical reproduction of a small frozen inference evaluation is distinct from deterministic end-to-end retraining. See [research scope and reproducibility](reproducibility.md) for the evaluation protocol and remaining gaps.

## Architectural references

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
