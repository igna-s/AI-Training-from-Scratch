# AI Training from Scratch

Personal investigations I ran with AMD cloud credits: training small language models from scratch, exploring low-bit and byte-level architectures, and running ablation studies.

The main recorded experiment family is **CPU-1**. It combines custom Transformer and recurrent models, byte representations, ternary weight experiments, and optional teacher-assisted training with pretrained **Qwen2.5-3B**. The student code initializes custom models from scratch and also supports resume and transfer between experiment checkpoints.

**Research status:** the checkpoints are loadable, but several correctness checks fail. The results below include those failures alongside the scores and samples.

[Model exports](https://huggingface.co/Cukinator/cpu1-ablations-final) · [Training checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints) · [Teacher dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) · [Setup and usage](docs/usage.md)

## What I explored

| Experiment | Scope |
|---|---|
| Transformer and MLGRU | 8 Transformer and 26 recurrent exports, from 10.69M to 54.74M parameters |
| Bytes and BPE | 29 fixed four-byte models and 5 models using a truncated Qwen token vocabulary |
| Global/local byte prediction | 27 exports use a two-layer local byte decoder; 2 use four independent byte heads |
| Low-bit weights | 26 exports simulate ternary weights in PyTorch; the evaluated files store FP32 tensors |
| Additional components | Boundary-aware embeddings, low-rank residuals, state deletion, learned decay, and teacher-hidden alignment |

These are local adaptations of ideas from BitNet, MatMul-free LM, MEGABYTE, Bolmo/BLT, MrT5, and HGRN2. The published research covers **34 exports with 10.69M–54.74M parameters**. The [architecture guide](docs/architecture-review.md) describes their implemented components and limitations.

## Run names and architectures

Each run identifies a configuration, rather than a paper architecture. For example, **`run_02` is a 38.83M-parameter byte Transformer** with fixed four-byte patches and a two-layer local byte decoder. **`run_04` is a 38.85M-parameter recurrent MLGRU byte model** with the same patch size and decoder depth, using simulated ternary projections. **`run_15` is a 10.73M-parameter recurrent byte model** with boundary-aware embedding, floating-point residuals, a deletion module, and a teacher-hidden alignment objective.

The [run-by-run explanations](docs/architecture-review.md#run-by-run-explanations) map all 34 exports to 16 experiment families, explain their training signals, and describe the question each configuration explores. The [component glossary](docs/architecture-review.md#component-glossary) explains the abbreviations; the [full inventory](docs/architecture-review.md#published-model-inventory) gives exact parameter counts, checkpoint steps, and correctness outcomes.

In the published records, unsuffixed runs configure a budget of 2 tokens/parameter, `_r2` runs 15, and `_v3` runs 150. These are planned budgets, not proof of completed training or improved models. Saved steps differ, and some variants also change configuration labels; see the [variant notes](docs/architecture-review.md#run-suffixes-and-training-signals) before comparing results.

## Results

The tables reproduce the **2026-10-06 CPU evaluation** published in the Hugging Face model cards. All 34 export hashes, strict loading, losses, and causal/chunk outcomes were subsequently rechecked locally and matched the recorded evidence. Timing and the full generation set remain the original measurements.

| Check | Outcome |
|---|---|
| Strict checkpoint loading | 34/34 loaded |
| Finite loss | 30/34; four exports produced nonfinite loss |
| Prefix causality | 16 PASS, 14 FAIL, 4 ERROR |
| Recurrent chunk equivalence | All 26 recurrent exports FAIL; 8 Transformer exports N/A |

### How to read the results

- **NLL** is average negative log-likelihood; **PPL** is perplexity; **bits/byte** expresses byte loss in bits. A uniform byte predictor has 8 bits/byte and perplexity 256.
- **Causal** checks whether changing later input alters earlier predictions. A FAIL can make a low loss benefit from future information.
- **Chunks** checks whether processing a sequence in pieces matches processing it all at once. A causality PASS does not resolve a chunk failure.
- Scores use **32 short WikiText-2 test passages**, batch size 1, teacher forcing, and deletion disabled. They are small-sample diagnostics; different steps, model sizes, and training signals prevent a controlled component ranking.

### Byte models — all 29 exports

| Run | Params M | Saved step | NLL/byte | Bits/byte | Byte PPL | Causal | Chunks |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| [`run_02`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02/model.pt) | 38.830 | 647 | 1.9702 | 2.8424 | 7.172 | PASS | N/A |
| [`run_02_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02_v3/model.pt) | 38.830 | 10963 | ERROR | — | — | ERROR | N/A |
| [`run_02a_byte_only_heads`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02a_byte_only_heads/model.pt) | 38.501 | 641 | 2.4755 | 3.5714 | 11.888 | PASS | N/A |
| [`run_02a_byte_only_heads_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02a_byte_only_heads_v3/model.pt) | 38.501 | 16327 | ERROR | — | — | ERROR | N/A |
| [`run_03`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_03/model.pt) | 38.830 | 647 | 2.1622 | 3.1194 | 8.690 | PASS | FAIL |
| [`run_03_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_03_v3/model.pt) | 38.830 | 6187 | 1.7640 | 2.5449 | 5.836 | PASS | FAIL |
| [`run_04`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04/model.pt) | 38.854 | 647 | 5.5551 | 8.0144 | 258.565 | PASS | FAIL |
| [`run_04_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04_r2/model.pt) | 38.854 | 4840 | 5.5676 | 8.0323 | 261.803 | PASS | FAIL |
| [`run_04_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04_v3/model.pt) | 38.854 | 6074 | 1.5515 | 2.2384 | 4.719 | PASS | FAIL |
| [`run_05`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05/model.pt) | 38.994 | 649 | 5.5485 | 8.0048 | 256.854 | PASS | FAIL |
| [`run_05_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05_v3/model.pt) | 38.994 | 4390 | 1.5921 | 2.2969 | 4.914 | PASS | FAIL |
| [`run_05b_kernel_strict`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05b_kernel_strict/model.pt) | 35.842 | 600 | 5.5789 | 8.0486 | 264.778 | PASS | FAIL |
| [`run_05b_kernel_strict_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05b_kernel_strict_v3/model.pt) | 35.842 | 2573 | 1.8552 | 2.6764 | 6.393 | PASS | FAIL |
| [`run_06`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_06/model.pt) | 38.994 | 649 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_06_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_06_v3/model.pt) | 38.994 | 4237 | 1.0572 | 1.5253 | 2.878 | FAIL | FAIL |
| [`run_07`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_07/model.pt) | 39.029 | 650 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_07_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_07_r2/model.pt) | 39.029 | 4860 | 5.5569 | 8.0169 | 259.010 | FAIL | FAIL |
| [`run_08`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_08/model.pt) | 38.854 | 647 | 5.5535 | 8.0121 | 258.150 | PASS | N/A |
| [`run_08_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_08_v3/model.pt) | 38.854 | 3858 | ERROR | — | — | ERROR | N/A |
| [`run_09`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_09/model.pt) | 39.428 | 657 | 5.5334 | 7.9830 | 253.007 | FAIL | FAIL |
| [`run_10`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_10/model.pt) | 39.434 | 657 | 5.5337 | 7.9834 | 253.068 | FAIL | FAIL |
| [`run_14`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14/model.pt) | 10.691 | 600 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14_r2/model.pt) | 10.691 | 1336 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14_v3/model.pt) | 10.691 | 9775 | 0.8811 | 1.2712 | 2.414 | FAIL | FAIL |
| [`run_15`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15/model.pt) | 10.732 | 600 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15_r2/model.pt) | 10.732 | 1341 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15_v3/model.pt) | 10.732 | 2350 | 1.3248 | 1.9113 | 3.761 | FAIL | FAIL |
| [`run_16`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_16/model.pt) | 10.691 | 600 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_16_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_16_r2/model.pt) | 10.691 | 1336 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |

Many original ternary byte exports score around the uniform-byte baseline. Some `_v3` exports have much lower diagnostic loss, but all recurrent exports still fail chunk equivalence, and boundary-aware variants also fail causality. These scores alone do not establish reliable language generation.

### BPE models — all 5 exports

| Run | Params M | Saved step | NLL/token | Token PPL | Causal | Chunks |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| [`run_01`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_01/model.pt) | 54.735 | 912 | 5.3742 | 215.773 | PASS | N/A |
| [`run_01_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_01_v3/model.pt) | 54.735 | 12173 | ERROR | — | ERROR | N/A |
| [`run_13`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13/model.pt) | 12.548 | 600 | 6.3429 | 568.445 | PASS | FAIL |
| [`run_13_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13_r2/model.pt) | 12.548 | 1568 | 6.4205 | 614.313 | PASS | FAIL |
| [`run_13_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13_v3/model.pt) | 12.548 | 9290 | 6.7925 | 891.155 | PASS | FAIL |

BPE NLL/PPL measure a **mapped token stream**, so they are not directly comparable with byte scores. Qwen IDs outside the saved vocabulary are mapped to 0: **14.26%** of sample IDs for the 16K vocabulary and **31.77%** for the 4K vocabulary. The original audit did not record the resolved tokenizer revision, which limits exact BPE reproduction.

### CPU timing and byte generation

Measured on an **Intel Core i7-8750H**, Windows 11, one PyTorch thread, FP32 parameters, and the PyTorch implementation. Ternary quantization is simulated; the custom C/C++ kernels are not used.

Forward timing uses 10 warm-up and 100 measured single-input steps. Generated bytes/s divides the total bytes by the total elapsed time across three continuations, including prompt prefill and sampling. **Forward steps/s and generated bytes/s measure different work.** Parameter MB covers stored parameter tensors, not peak process RAM.

<details>
<summary><strong>CPU measurements for all 34 exports</strong></summary>

| Run | Param MB (FP32) | Forward steps/s | p50 ms | p95 ms | Generated bytes/s | Valid UTF-8 samples |
|---|---:|---:|---:|---:|---:|---:|
| `run_01` | 218.94 | 54.64 | 18.21 | 20.63 | N/A | N/A |
| `run_01_v3` | 218.94 | ERROR | — | — | N/A | N/A |
| `run_02` | 155.32 | 54.58 | 16.69 | 30.12 | 88.75 | 3/3 |
| `run_02_v3` | 155.32 | ERROR | — | — | N/A | N/A |
| `run_02a_byte_only_heads` | 154.00 | 67.85 | 14.24 | 18.56 | 99.13 | 2/3 |
| `run_02a_byte_only_heads_v3` | 154.00 | ERROR | — | — | N/A | N/A |
| `run_03` | 155.32 | 52.42 | 17.68 | 30.45 | 154.92 | 3/3 |
| `run_03_v3` | 155.32 | 51.10 | 18.68 | 27.36 | 173.00 | 3/3 |
| `run_04` | 155.42 | 6.66 | 146.44 | 189.02 | 22.32 | 0/3 |
| `run_04_r2` | 155.42 | 5.92 | 164.12 | 231.57 | 18.60 | 0/3 |
| `run_04_v3` | 155.42 | 3.49 | 294.80 | 378.66 | 23.96 | 3/3 |
| `run_05` | 155.98 | 6.11 | 156.23 | 224.46 | 17.37 | 0/3 |
| `run_05_v3` | 155.98 | 5.71 | 157.44 | 225.37 | 20.31 | 3/3 |
| `run_05b_kernel_strict` | 143.37 | 5.41 | 171.47 | 288.26 | 18.87 | 0/3 |
| `run_05b_kernel_strict_v3` | 143.37 | 6.69 | 148.31 | 181.31 | 20.92 | 3/3 |
| `run_06` | 155.98 | 6.19 | 158.21 | 206.27 | 23.91 | 0/3 |
| `run_06_v3` | 155.98 | 6.73 | 135.85 | 223.51 | 20.98 | 2/3 |
| `run_07` | 156.11 | 6.78 | 140.98 | 193.73 | 19.25 | 0/3 |
| `run_07_r2` | 156.11 | 6.24 | 151.17 | 223.57 | 21.20 | 0/3 |
| `run_08` | 155.42 | 7.65 | 124.09 | 175.60 | 23.24 | 0/3 |
| `run_08_v3` | 155.42 | ERROR | — | — | N/A | N/A |
| `run_09` | 157.71 | 6.39 | 144.30 | 208.93 | 19.73 | 0/3 |
| `run_10` | 157.74 | 7.14 | 135.05 | 166.90 | 23.57 | 0/3 |
| `run_13` | 50.19 | 23.66 | 41.46 | 47.41 | N/A | N/A |
| `run_13_r2` | 50.19 | 24.46 | 40.26 | 44.37 | N/A | N/A |
| `run_13_v3` | 50.19 | 21.69 | 43.21 | 61.74 | N/A | N/A |
| `run_14` | 42.77 | 28.00 | 35.19 | 38.19 | 80.37 | 0/3 |
| `run_14_r2` | 42.77 | 25.51 | 37.39 | 49.82 | 73.29 | 0/3 |
| `run_14_v3` | 42.77 | 24.80 | 37.64 | 57.30 | 76.37 | 3/3 |
| `run_15` | 42.93 | 28.36 | 35.05 | 36.72 | 81.61 | 0/3 |
| `run_15_r2` | 42.93 | 27.10 | 36.58 | 38.68 | 73.25 | 0/3 |
| `run_15_v3` | 42.93 | 14.17 | 60.10 | 124.86 | 53.99 | 3/3 |
| `run_16` | 42.77 | 20.26 | 44.73 | 73.85 | 71.64 | 0/3 |
| `run_16_r2` | 42.77 | 21.75 | 43.29 | 59.96 | 55.22 | 0/3 |

</details>

Each byte run uses three prompts, 128 new bytes per prompt, seeds 42–44, temperature 1.0, top-k 40, and top-p 0.9. Transformers recompute retained context; recurrent models carry state through the implementation with the known scan defect. Valid UTF-8 checks byte encoding, not text quality. BPE generation is N/A because its lossy token mapping is not comparable.

### What the generated text looks like

These are the first 120 characters of the **first recorded continuation** for three byte variants, all with prompt **“The history of artificial intelligence”** and seed 42. Escapes preserve newlines. Complete 128-byte outputs and all three prompts are in the [per-run JSON records](experiments/evaluation/model-audit-2026-10-06/).

**run_02**

```text
"icadian virmmerred- dastot Blays including teat gomam franging a design that pretirely or brangible. The chat patt by a "
```

**run_04_v3**

```text
"atite white waterfalper diangiets opespucica continues apoliomin for natinal envical grapable tha rust of pria a treatin"
```

**run_14_v3**

```text
"al C APLEOSTrorog's NC. Overness I Co eforgram.\nAt 2odromist - Careforrs, Milz Drology in M Ontooyoum , one volume late "
```

The outputs contain fragments resembling words, but remain weak and often incoherent. Low diagnostic loss or valid UTF-8 does not establish useful text generation.

### Known failures

The four numerical failures are `run_01_v3`, `run_02_v3`, `run_02a_byte_only_heads_v3`, and `run_08_v3`. They loaded successfully but produced nonfinite loss; missing scores are shown as ERROR.

The shared recurrent parallel scan disagrees with its serial recurrence and chunked execution. Boundary-aware embeddings in the published configurations can see future patches. The common evaluation uses phase 1 with deletion disabled, so it does not establish phase-2 compression gains or speedups. See the [architecture guide](docs/architecture-review.md) for implementation details and the [full model card](docs/model_cards/cpu1-ablations-final.md#correctness-checks) for individual correctness checks.

## Models and data

These totals describe the reviewed artifact revisions; the model evaluation covers the 34 FP32 exports, rather than every intermediate training checkpoint.

| Resource | Published contents |
|---|---|
| [Final exports](https://huggingface.co/Cukinator/cpu1-ablations-final/tree/65951289352c1a036b12ba6d888af037e7c90b13) | 34 custom PyTorch model files, 4.178 GB total |
| [Training archive](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/tree/fb402d025e843ed00fb990966bf2430c32eb958c) | 1,346 checkpoint files in 34 run folders, including 21 finals; 215.178 GB total |
| [Teacher dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset/tree/bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf) | 917 Parquet shards: 480 byte and 437 BPE; 68.019 GB total |

The [34-run manifest](experiments/published/manifest.json) records configurations, checkpoint hashes, saved steps, and revisions. Saved steps and configured budgets do not prove completed training or convergence. “1.58-bit” describes the ternary value alphabet; compact finals use two-bit packing plus scales and floating-point tensors, while these inference exports are FP32.

The [dataset review](docs/model_cards/cpu1-ablation-dataset.md) inspected three fixed shards and found teacher-reader alignment, mask, and UTF-8 limitations. Dataset-wide teacher correctness is not established. Training download limits are best-effort; a fallback can request an entire subset. The [usage guide](docs/usage.md) explains these behaviors before starting a run.

## Using the project

Start with the [setup and usage guide](docs/usage.md) for installation, loading one checkpoint, generating bytes, and training recipes. Hugging Face hosts the model and dataset artifacts; this repository contains their source and evaluation records.

Code lives in [cpu1/](cpu1/), [training/](training/), [data_pipeline/](data_pipeline/), [evaluation/](evaluation/), and [runners/](runners/). Experimental native kernels live in [inference/](inference/); their synthetic benchmarks are not verified generation speed for a published checkpoint.

The [evaluation records](experiments/evaluation/README.md) include the exact sample, environment, results, and checkpoint checks. [Experiment provenance](docs/provenance/README.md) describes recorded configurations and platform recipes. The [research scope and reproducibility guide](docs/reproducibility.md) explains the evaluation protocol and the limits of the available evidence. Deterministic end-to-end retraining remains unverified.

## References and licensing

Architectural inspirations: [BitNet b1.58](https://arxiv.org/abs/2402.17764), [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528), [MEGABYTE](https://arxiv.org/abs/2305.07185), [Bolmo](https://arxiv.org/abs/2512.15586), [BLT](https://arxiv.org/abs/2412.09871), [MrT5](https://arxiv.org/abs/2410.20771), [HGRN2](https://arxiv.org/abs/2404.07904), [T-MAC](https://arxiv.org/abs/2407.00088), and [Distilling Token-Trained Models into Byte-Level Models](https://arxiv.org/abs/2602.01007).

The companion models declare Apache-2.0 and the dataset declares MIT. This source tree has no project license file; artifact licenses do not assign a source-code license here. WikiText excerpts retain their [upstream attribution and terms](experiments/evaluation/model-audit-2026-10-06/CORPUS_LICENSE.txt).
