---
license: apache-2.0
language:
- en
tags:
- cpu-1
- ablation
- ternary
- research
---

# CPU-1 ablations: training checkpoint archive

1,346 checkpoint files across 34 run folders, totaling **215.178 GB** (200.400 GiB). The inventory includes **21 `_final.pt` files** and **1325 step files**. Counts and sizes are derived from the pinned Hub inventory, not estimates from nominal architecture sizes.

For evaluated FP32 exports, see [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final). Source code: [igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08).

## Evaluation summary

The evaluation covers 34 FP32 exports on 32 short WikiText test passages, with CPU execution, one thread, and a fixed protocol. It does not evaluate every intermediate archive checkpoint.

| Check | Outcome |
|---|---|
| Strict checkpoint loading | 34/34 loaded |
| Numerical evaluation | 4 models produced an evaluation error |
| Prefix causality | 14/30 finite-output models failed |
| Recurrent chunk equivalence | 26/26 recurrent models failed |

Full results, generation examples, and the recurrent implementation diagnostic follow below. Loss units depend on tokenization. Loss from a model that fails causality cannot establish autoregressive prediction quality; printable output alone does not establish useful language generation.

## What is stored

- `checkpoint_<run>_final.pt`: inference-oriented final checkpoints. The source supports `compact_2bit` for ternary exports and floating-point storage for baselines; inspect the `format` field in the payload when loading a file.
- `checkpoint_<run>_step<N>.pt`: intermediate training state; includes saved configuration and model state, and may include optimizer state. `_phase2_step<N>.pt` names distinguish phase-2 files where present.
- The 2-bit codec packs a ternary alphabet into two bits per stored value plus scales and floating-point tensors. It is not a guarantee of 1.58 effective bits per total model parameter. Resume step files and packed finals have different purposes.
- A folder without `_final.pt` is an intermediate-checkpoint archive. The highest filename step establishes the highest stored step; it does not establish convergence, completed training budget, or an early-stop reason.

## Archive inventory

| Folder | Step files | Highest filename step | Final file MB | Total GB |
|---|---:|---:|---:|---:|
| `run_01` | 21 | 900 | 109.51 | 7.009 |
| `run_01_v3` | 72 | 12173 | — | 23.656 |
| `run_02` | 21 | 640 | 77.71 | 4.974 |
| `run_02_v3` | 73 | 10963 | — | 17.020 |
| `run_02a_byte_only_heads` | 21 | 640 | 77.04 | 4.931 |
| `run_02a_byte_only_heads_v3` | 75 | 16327 | — | 17.337 |
| `run_03` | 21 | 640 | 77.72 | 4.974 |
| `run_03_v3` | 71 | 6187 | — | 16.555 |
| `run_04` | 21 | 640 | 9.89 | 1.643 |
| `run_04_r2` | 21 | 4840 | 9.90 | 1.643 |
| `run_04_v3` | 71 | 6074 | — | 16.568 |
| `run_05` | 21 | 640 | 10.19 | 1.650 |
| `run_05_v3` | 71 | 4390 | — | 16.632 |
| `run_05b_kernel_strict` | 21 | 600 | 9.38 | 1.517 |
| `run_05b_kernel_strict_v3` | 71 | 2573 | — | 15.288 |
| `run_06` | 21 | 640 | 10.19 | 1.650 |
| `run_06_v3` | 71 | 4237 | — | 16.632 |
| `run_07` | 21 | 640 | 10.26 | 1.652 |
| `run_07_r2` | 21 | 4860 | — | 1.641 |
| `run_08` | 21 | 640 | 9.88 | 1.643 |
| `run_08_v3` | 71 | 3858 | — | 16.567 |
| `run_09` | 21 | 640 | 10.39 | 1.669 |
| `run_10` | 21 | 640 | 10.41 | 1.669 |
| `run_13` | 21 | 600 | 3.38 | 1.586 |
| `run_13_r2` | 21 | 1560 | 3.39 | 1.586 |
| `run_13_v3` | 72 | 9290 | — | 5.426 |
| `run_14` | 21 | 600 | 2.94 | 0.453 |
| `run_14_r2` | 21 | 1320 | 2.94 | 0.453 |
| `run_14_v3` | 75 | 9775 | — | 4.820 |
| `run_15` | 21 | 600 | 3.02 | 0.455 |
| `run_15_r2` | 21 | 1340 | 3.02 | 0.455 |
| `run_15_v3` | 70 | 2350 | — | 4.516 |
| `run_16` | 21 | 600 | 2.94 | 0.453 |
| `run_16_r2` | 21 | 1320 | 2.94 | 0.453 |

Sizes use decimal MB and GB. Step counts include phase-2 step files. Highest step is parsed from filenames and does not merge training phases into a continuous training counter.

## Load one checkpoint

Clone the [source repository](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08) and run this from its root. The source repository is currently private and requires authorized GitHub access. These are custom PyTorch checkpoints; `AutoModelForCausalLM.from_pretrained()` does not load them. The examples use a trusted checkpoint source and strict state loading.

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from huggingface_hub import hf_hub_download
torch.set_num_threads(1)
path = hf_hub_download("Cukinator/cpu1-ablation-checkpoints", "run_04/checkpoint_run_04_final.pt",
    revision="fb402d025e843ed00fb990966bf2430c32eb958c", local_dir="D:/cpu1/active", cache_dir="D:/cpu1/hf/hub")
from training.ablation import build_ablation_model, load_ablation_checkpoint
state, config = load_ablation_checkpoint(path)
config["use_gradient_checkpointing"] = False
model = build_ablation_model(config)
model.load_state_dict(state, strict=True)
model.eval()
x = torch.tensor(list(b"The history of AI.  ")).reshape(1, -1, 4)
with torch.inference_mode():
    logits, states, extra = model(x, training_phase=1, target_elimination=0.0)
print([tuple(head.shape) for head in logits])
```

This example supplies zero target bytes to the local decoder to demonstrate output shapes. For likelihood scoring with LocalByteDecoder, supply the ground-truth `byte_targets`; the decoder shifts them internally. Byte generation requires sampling within each patch and retaining the appropriate recurrent state or Transformer context. See the GitHub evaluation implementation and the recorded continuation examples.

## Training configurations

Recorded configurations, saved steps and checkpoint hashes are indexed in `experiments/published/manifest.json` in the GitHub source. Use `--config` to select the intended experiment. Configuration compatibility is required when loading or resuming a checkpoint.

## Resume training

Use an intermediate checkpoint, not an inference-only packed final. The current source CLI names the argument `--resume_ckpt`:

```powershell
python -m training.ablation --run run_04 --config experiments/published/configs/run_04.json --resume_ckpt D:/cpu1/active/run_04/checkpoint_run_04_step640.pt --skip-install
```

Download the exact step file from its run folder first. Resume requires matching source/configuration, dependencies, and suitable training hardware. End-to-end training resume has not been verified for this release.

## Scope of the measured results

The results below were measured on the 34 published exports in [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final), **not by evaluating every intermediate archive checkpoint**. Matching run names do not prove tensor identity between a packed final, an intermediate step, and an export. No packed/FP32 numerical-parity claim is made by this card.

## Evaluation methodology

Results were measured on **2026-10-06** for the 34 FP32 exports. Raw results, checkpoint hashes, the text sample, package versions, corpus attribution and evaluator metadata are available in the [GitHub evaluation files](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08/experiments/evaluation).

- Hardware: Intel Core i7-8750H, Windows 11, CPU execution, one PyTorch thread, batch size 1 and FP32 parameters. PyTorch 2.6.0+cu124. The measurements use the PyTorch implementation without custom C/C++ kernels.
- Text: the first 32 non-heading paragraphs of the WikiText-2 raw test split with at least 260 UTF-8 bytes, truncated to 260 bytes at a complete character boundary. This sample supports small-scale comparisons; it is not the full benchmark. Training/test overlap has not been exhaustively checked.
- Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), revision `b08601e04326c79dfdd32d625aee71d232d685c3`, file `wikitext-2-raw-v1/test-00000-of-00001.parquet`. Parquet SHA-256: `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`. Sample JSON SHA-256: `11d143c06478ff7bbe364c541a58e6a2e76504c4cb0595b294f46e98f3cfe798`.
- Corpus license: WikiText text derives from English Wikipedia contributions. The dataset metadata lists CC-BY-SA-3.0 and GFDL; CORPUS_LICENSE.txt in the evaluation files preserves the attribution and terms separately from the model license.
- Byte scoring predicts the next four-byte patch. The first patch supplies context and incomplete final patches are discarded. LocalByteDecoder receives ground-truth target bytes, shifted internally for causal teacher forcing within the patch. Cross-entropy is summed over targets and divided by their total count.
- Byte NLL is in nats/byte; BPB = NLL / ln(2), and byte PPL = exp(NLL). A uniform byte predictor has BPB=8 and PPL=256. BPE scoring uses Qwen2.5-3B token IDs, mapping IDs outside the saved vocabulary cap to 0. BPE PPL describes this lossy mapped stream; it is not full-vocabulary text likelihood and is not comparable to byte PPL.
- The resolved Qwen tokenizer revision was not recorded. This limits exact reproduction of the BPE measurements; the environment and evaluator hashes are preserved in the result files.
- Prefix causality changes the last 16 of 32 input steps and compares earlier logits at tolerance 1e-5. State chunk equivalence compares one 128-step forward with four 32-step chunks across all output heads at tolerance 1e-4.
- DeleteGate uses phase 1 with elimination rate 0. Results do not measure phase-2 quality or the effect of token elimination.
- Forward speed uses 10 warm-up steps and 100 measured single-step forwards. A step consumes four bytes for byte models or one mapped token for BPE models. Local decoder inputs are zeros; Transformer forwards have no persistent attention context. This microbenchmark does not measure growing-context throughput. Parameter MB reports FP32 parameter storage, not peak process RAM.
- Byte continuations use three prompts, 128 new bytes each, seeds 42/43/44, temperature 1.0, top-k 40 and top-p 0.9. Recurrent models carry state; Transformers reprocess the growing context. Generated bytes/s includes prompt prefill and sampling. UTF-8 validity is checked on each complete continuation. BPE generation is not included because of the lossy vocabulary mapping.

Evaluated export revision: `65951289352c1a036b12ba6d888af037e7c90b13`. Archive inventory revision: `fb402d025e843ed00fb990966bf2430c32eb958c`.

### Code and result files

The [GitHub source](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08) contains the model implementation, training recipes and evaluation tools. The [evaluation files](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08/experiments/evaluation) include all 34 per-run JSON results, exact corpus, dependency versions, checkpoint hashes and the reference evaluator location. Use these records when comparing or reproducing the measurements.

## Evaluation results

**Do not rank models across tokenizations or treat a failed-causality loss as autoregressive language-model perplexity.** Saved training metrics are retained separately in the JSONs; the numbers below were measured on the shared test sample.

### Byte models

| Run | Params M | Saved step | File MB | NLL | BPB | PPL | Causal | State chunks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [`run_02`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02/model.pt) | 38.830 | 647 | 155.35 | 1.9702 | 2.8424 | 7.172 | PASS | N/A |
| [`run_02_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02_v3/model.pt) | 38.830 | 10963 | 155.35 | ERROR | — | — | ERROR | N/A |
| [`run_02a_byte_only_heads`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02a_byte_only_heads/model.pt) | 38.501 | 641 | 154.04 | 2.4755 | 3.5714 | 11.888 | PASS | N/A |
| [`run_02a_byte_only_heads_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_02a_byte_only_heads_v3/model.pt) | 38.501 | 16327 | 154.03 | ERROR | — | — | ERROR | N/A |
| [`run_03`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_03/model.pt) | 38.830 | 647 | 155.38 | 2.1622 | 3.1194 | 8.690 | PASS | FAIL |
| [`run_03_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_03_v3/model.pt) | 38.830 | 6187 | 155.38 | 1.7640 | 2.5449 | 5.836 | PASS | FAIL |
| [`run_04`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04/model.pt) | 38.854 | 647 | 155.49 | 5.5551 | 8.0144 | 258.565 | PASS | FAIL |
| [`run_04_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04_r2/model.pt) | 38.854 | 4840 | 155.49 | 5.5676 | 8.0323 | 261.803 | PASS | FAIL |
| [`run_04_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_04_v3/model.pt) | 38.854 | 6074 | 155.49 | 1.5515 | 2.2384 | 4.719 | PASS | FAIL |
| [`run_05`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05/model.pt) | 38.994 | 649 | 156.06 | 5.5485 | 8.0048 | 256.854 | PASS | FAIL |
| [`run_05_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05_v3/model.pt) | 38.994 | 4390 | 156.06 | 1.5921 | 2.2969 | 4.914 | PASS | FAIL |
| [`run_05b_kernel_strict`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05b_kernel_strict/model.pt) | 35.842 | 600 | 143.44 | 5.5789 | 8.0486 | 264.778 | PASS | FAIL |
| [`run_05b_kernel_strict_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_05b_kernel_strict_v3/model.pt) | 35.842 | 2573 | 143.44 | 1.8552 | 2.6764 | 6.393 | PASS | FAIL |
| [`run_06`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_06/model.pt) | 38.994 | 649 | 156.06 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_06_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_06_v3/model.pt) | 38.994 | 4237 | 156.06 | 1.0572 | 1.5253 | 2.878 | FAIL | FAIL |
| [`run_07`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_07/model.pt) | 39.029 | 650 | 156.20 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_07_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_07_r2/model.pt) | 39.029 | 4860 | 156.20 | 5.5569 | 8.0169 | 259.010 | FAIL | FAIL |
| [`run_08`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_08/model.pt) | 38.854 | 647 | 155.46 | 5.5535 | 8.0121 | 258.150 | PASS | N/A |
| [`run_08_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_08_v3/model.pt) | 38.854 | 3858 | 155.46 | ERROR | — | — | ERROR | N/A |
| [`run_09`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_09/model.pt) | 39.428 | 657 | 157.81 | 5.5334 | 7.9830 | 253.007 | FAIL | FAIL |
| [`run_10`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_10/model.pt) | 39.434 | 657 | 157.83 | 5.5337 | 7.9834 | 253.068 | FAIL | FAIL |
| [`run_14`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14/model.pt) | 10.691 | 600 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14_r2/model.pt) | 10.691 | 1336 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_14_v3/model.pt) | 10.691 | 9775 | 42.82 | 0.8811 | 1.2712 | 2.414 | FAIL | FAIL |
| [`run_15`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15/model.pt) | 10.732 | 600 | 42.99 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15_r2/model.pt) | 10.732 | 1341 | 42.99 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_15_v3/model.pt) | 10.732 | 2350 | 42.99 | 1.3248 | 1.9113 | 3.761 | FAIL | FAIL |
| [`run_16`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_16/model.pt) | 10.691 | 600 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_16_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_16_r2/model.pt) | 10.691 | 1336 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |

### BPE models (mapped Qwen token stream)

| Run | Params M | Saved step | File MB | NLL | BPB | PPL | Causal | State chunks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [`run_01`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_01/model.pt) | 54.735 | 912 | 218.97 | 5.3742 | — | 215.773 | PASS | N/A |
| [`run_01_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_01_v3/model.pt) | 54.735 | 12173 | 218.97 | ERROR | — | — | ERROR | N/A |
| [`run_13`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13/model.pt) | 12.548 | 600 | 50.24 | 6.3429 | — | 568.445 | PASS | FAIL |
| [`run_13_r2`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13_r2/model.pt) | 12.548 | 1568 | 50.24 | 6.4205 | — | 614.313 | PASS | FAIL |
| [`run_13_v3`](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/65951289352c1a036b12ba6d888af037e7c90b13/run_13_v3/model.pt) | 12.548 | 9290 | 50.24 | 6.7925 | — | 891.155 | PASS | FAIL |

| BPE run | IDs mapped to 0 | Vocabulary cap |
|---|---:|---:|
| `run_01` | 14.26% | 16,384 |
| `run_13` | 31.77% | 4,096 |
| `run_13_r2` | 31.77% | 4,096 |
| `run_13_v3` | 31.77% | 4,096 |

### CPU timing and byte generation

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

### Example continuations

The following fixed examples use the first evaluation prompt, `The history of artificial intelligence`, seed 42, and the sampling settings above. They illustrate why low diagnostic loss or valid UTF-8 alone does not establish useful text generation. Escapes preserve the recorded characters; complete byte sequences and all three prompts are in the JSON reports.

`run_04_v3`:

```text
"atite white waterfalper diangiets opespucica continues apoliomin for natinal envical grapable tha rust of pria a treating papora"
```

`run_14_v3`:

```text
"al C APLEOSTrorog's NC. Overness I Co eforgram.\nAt 2odromist - Careforrs, Milz Drology in M Ontooyoum , one volume late from obs"
```


### Correctness checks

| Run | Shapes | Probabilities | Determinism | State chunks | Ternary projection | State size | Causality |
|---|---|---|---|---|---|---|---|
| `run_01` | PASS | PASS | PASS | N/A | N/A | N/A | PASS |
| `run_01_v3` | FAIL | FAIL | FAIL | N/A | N/A | N/A | ERROR |
| `run_02` | PASS | PASS | PASS | N/A | N/A | N/A | PASS |
| `run_02_v3` | FAIL | FAIL | FAIL | N/A | N/A | N/A | ERROR |
| `run_02a_byte_only_heads` | PASS | PASS | PASS | N/A | N/A | N/A | PASS |
| `run_02a_byte_only_heads_v3` | FAIL | FAIL | FAIL | N/A | N/A | N/A | ERROR |
| `run_03` | PASS | PASS | PASS | FAIL | N/A | PASS | PASS |
| `run_03_v3` | PASS | PASS | PASS | FAIL | N/A | PASS | PASS |
| `run_04` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_04_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_04_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_05` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_05_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_05b_kernel_strict` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_05b_kernel_strict_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_06` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_06_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_07` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_07_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_08` | PASS | PASS | PASS | N/A | PASS | N/A | PASS |
| `run_08_v3` | FAIL | FAIL | FAIL | N/A | FAIL | N/A | ERROR |
| `run_09` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_10` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_13` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_13_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_13_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | PASS |
| `run_14` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_14_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_14_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_15` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_15_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_15_v3` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_16` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |
| `run_16_r2` | PASS | PASS | PASS | FAIL | PASS | PASS | FAIL |

**Causality failures: 14/30 finite-output models.** `run_06`, `run_06_v3`, `run_07`, `run_07_r2`, `run_09`, `run_10`, `run_14`, `run_14_r2`, `run_14_v3`, `run_15`, `run_15_r2`, `run_15_v3`, `run_16`, `run_16_r2`

Changing future inputs changed prefix logits in these models. Some saved configurations enable noncausal Bolmo cross-patch operations; the test detects the effect without isolating its individual source. Their full-sequence losses are descriptive reconstruction scores and cannot establish causal prediction quality.

**State chunk failures: 26.** `run_03`, `run_03_v3`, `run_04`, `run_04_r2`, `run_04_v3`, `run_05`, `run_05_v3`, `run_05b_kernel_strict`, `run_05b_kernel_strict_v3`, `run_06`, `run_06_v3`, `run_07`, `run_07_r2`, `run_09`, `run_10`, `run_13`, `run_13_r2`, `run_13_v3`, `run_14`, `run_14_r2`, `run_14_v3`, `run_15`, `run_15_r2`, `run_15_v3`, `run_16`, `run_16_r2`

These configurations did not reproduce a full-sequence forward when carrying state between chunks at the stated tolerance. An O(1) state tensor alone does not establish correct streaming equivalence. These results describe the published implementation and checkpoints.

### Recurrent scan implementation

The source’s `AblationMLGRU._parallel_scan` was tested independently against its defining serial recurrence with random candidate/forget tensors, seed 19, shape `[1,32,8]`, FP32. The maximum difference was **19.100506**; splitting the same scan into two 16-step chunks differed by **11.471270** (tolerance `1e-4`). See the [recorded implementation results](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08/experiments/evaluation) and [evaluation/diagnostics.py](https://github.com/igna-s/AI-Training-from-Scratch/blob/cpu1-experiment-2026-10-08/evaluation/diagnostics.py).

The implementation replaces log-cumulative-sum-exp with a cumulative-maximum normalization followed by a cumulative sum; accumulated terms are not rescaled when that maximum changes. The observed outputs therefore do not satisfy the intended recurrence. This is a source implementation defect; disabling or repairing the scan would create a different inference protocol and was not silently applied to these results. The measurements use the published implementation.

### Evaluation errors

Scores and speed are unavailable when model loading or numerical evaluation fails. Nonfinite values are represented as JSON null; they are never substituted with valid-looking numbers.

| Run | Error |
|---|---|
| `run_01_v3` | ValueError: nonfinite loss |
| `run_02_v3` | ValueError: nonfinite loss |
| `run_02a_byte_only_heads_v3` | ValueError: nonfinite loss |
| `run_08_v3` | ValueError: nonfinite loss |

## Training and architecture

Training, dataset preparation and Kaggle/Lightning recipes are maintained in the [GitHub source](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08). The source provides an AMD/ROCm-oriented entry point; these CPU results do not validate a fresh GPU training job. The [saved configurations](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08/experiments/published) record the 34 exported runs, checkpoint hashes, parameter counts and saved steps. Current presets can differ from the published runs; select a recorded configuration explicitly.

Configured token budgets, including the v3 150 tokens/parameter setting, are not measurements of completed training exposure. The exact source commit and full dependency lock for each training job were not recorded.

The [training dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) contains teacher signals with documented short/empty arrays, dense-signal alignment limitations in the reader and UTF-8 conversion limitations. Their effect on each trained checkpoint has not been isolated.

### Saved architecture settings

Dimensions and features below come from the recorded configurations. The quantization column is the training label; all evaluated exports contain FP32 parameters. Parameter counts are measured from the loaded models.

| Run | Mixer | Tokens | Quant. label | Width / layers / FFN | Local decoder | FP residual | Bolmo | DeleteGate | PFNet hidden | Channel decay |
|---|---|---|---|---|---|---|---|---|---:|---|
| `run_01` | transformer | bpe | fp16 | 512 / 12 / 1376 | N/A | no | no | no | 0 | no |
| `run_01_v3` | transformer | bpe | fp16 | 512 / 12 / 1376 | N/A | no | no | no | 0 | no |
| `run_02` | transformer | byte | fp16 | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_02_v3` | transformer | byte | fp16 | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_02a_byte_only_heads` | transformer | byte | fp16 | 512 / 12 / 1376 | no | no | no | no | 0 | no |
| `run_02a_byte_only_heads_v3` | transformer | byte | fp16 | 512 / 12 / 1376 | no | no | no | no | 0 | no |
| `run_03` | mlgru | byte | fp16 | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_03_v3` | mlgru | byte | fp16 | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_04` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_04_r2` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_04_v3` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_05` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | yes | no | no | 0 | no |
| `run_05_v3` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | yes | no | no | 0 | no |
| `run_05b_kernel_strict` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | yes | no | no | 0 | no |
| `run_05b_kernel_strict_v3` | mlgru | byte | ternary | 512 / 12 / 1376 | yes | yes | no | no | 0 | no |
| `run_06` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | no | 0 | no |
| `run_06_v3` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | no | 0 | no |
| `run_07` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | yes | 0 | no |
| `run_07_r2` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | yes | 0 | no |
| `run_08` | transformer | byte | ternary | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_08_v3` | transformer | byte | ternary | 512 / 12 / 1376 | yes | no | no | no | 0 | no |
| `run_09` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | yes | 32 | no |
| `run_10` | mlgru | byte_dynamic | ternary | 512 / 12 / 1376 | yes | yes | yes | yes | 32 | yes |
| `run_13` | mlgru | bpe | ternary | 320 / 8 / 853 | N/A | yes | no | yes | 0 | no |
| `run_13_r2` | mlgru | bpe | ternary | 320 / 8 / 853 | N/A | yes | no | yes | 0 | no |
| `run_13_v3` | mlgru | bpe | ternary | 320 / 8 / 853 | N/A | yes | no | yes | 0 | no |
| `run_14` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_14_r2` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_14_v3` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_15` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_15_r2` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_15_v3` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_16` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |
| `run_16_r2` | mlgru | byte_dynamic | ternary | 320 / 8 / 853 | yes | yes | yes | yes | 0 | no |

These experiments change parameter counts, representations, training signal, budgets, saved steps, and sometimes multiple architectural settings. These differences prevent isolating the benefit of an individual component from these results alone. `chinchilla_tokens_per_param` is a configured budget; it does not prove that a run completed that budget. Actual saved steps are reported separately.

## License

The model repository declares Apache-2.0. Dataset, corpus and tokenizer licenses are separate.

## Related resources

- [GitHub: igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch): extraction pipeline, training and evaluation code.
- [Hugging Face: Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final): FP32 model exports.
- [Hugging Face: Cukinator/cpu1-ablation-dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset): byte and BPE training data.
