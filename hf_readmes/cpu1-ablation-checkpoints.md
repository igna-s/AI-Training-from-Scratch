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

For evaluated FP32 exports, see [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final). Source code: [Cukinator/1.58bits](https://github.com/igna-s/New-1.58).

## What the audit found

The shared audit covers 34 FP32 exports on 32 short WikiText test passages, with CPU execution, one thread, and a fixed protocol. It does not evaluate every intermediate archive checkpoint.

| Check | Outcome |
|---|---|
| Strict checkpoint loading | 34/34 loaded |
| Numerical evaluation | 4 models produced an evaluation error |
| Prefix causality | 14/30 finite-output models failed |
| Recurrent chunk equivalence | 26/26 recurrent models failed |

Full results, generation examples, and the recurrent implementation diagnostic follow below. Loss units depend on tokenization. Loss from a model that fails causality cannot establish autoregressive prediction quality; printable output alone does not establish useful language generation.

## What is stored

- `checkpoint_<run>_final.pt`: inference-oriented final checkpoints. Ternary runs use `compact_2bit`; floating-point baselines use floating-point checkpoint storage. File names alone do not identify the serialization format: inspect `format` inside the payload.
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

Clone the [source repository](https://github.com/igna-s/New-1.58) and run this from its root. These are custom PyTorch checkpoints; `AutoModelForCausalLM.from_pretrained()` does not load them. The examples use a trusted checkpoint source and strict state loading.

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from huggingface_hub import hf_hub_download
torch.set_num_threads(1)
path = hf_hub_download("Cukinator/cpu1-ablation-checkpoints", "run_04/checkpoint_run_04_final.pt",
    revision="fb402d025e843ed00fb990966bf2430c32eb958c", local_dir="D:/cpu1/active", cache_dir="D:/cpu1/hf/hub")
from train_ablation import build_ablation_model, load_ablation_checkpoint
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

This forward uses zero target bytes in the local decoder; it is a shape smoke test. For scored likelihood, pass `byte_targets` as shown in the audit. For byte generation with retained Transformer context and one recurrent prefill, use `generation()` in `eval/audit_readmes.py`. Generation remains experimental; inspect the saved examples before choosing a checkpoint.

## Resume training

Use an intermediate checkpoint, not an inference-only packed final. The current source CLI names the argument `--resume_ckpt`:

```powershell
python train_ablation.py --run run_04 --resume_ckpt D:/cpu1/active/run_04/checkpoint_run_04_step640.pt
```

Download the exact step file from its run folder first. Resume requires matching source/configuration, dependencies, and suitable training hardware. This audit did not run a training-resume experiment.

## Scope of the measured results

The results below were measured on the 34 published exports in [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final), **not by evaluating every intermediate archive checkpoint**. Matching run names do not prove tensor identity between a packed final, an intermediate step, and an export. No packed/FP32 numerical-parity claim is made by this card.

## Evaluation protocol

This card was rebuilt from a fresh CPU audit on **2026-10-06**. All 34 published FP32 exports were attempted; 34 loaded with zero missing or unexpected state keys. Numerical and loading errors remain visible. Raw JSON, the exact text sample, and the repository inventories are included in [`evaluation/`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/tree/main/evaluation). Previous `resultados/ablation_*.json` reports are historical and were not mixed into these tables.

- Hardware: Intel Core i7-8750H (6 cores / 12 logical processors), Windows 11; CPU execution with **1 PyTorch thread**, batch size 1, FP32 parameters. PyTorch 2.6.0+cu124. No custom C/C++ kernels were used.
- Test text: [`Salesforce/wikitext`, `wikitext-2-raw-v1`, test split](https://huggingface.co/datasets/Salesforce/wikitext). First 32 non-heading paragraphs with at least 260 UTF-8 bytes, each truncated to 260 bytes without an incomplete trailing character. This is a small diagnostic sample, not the complete benchmark. No exhaustive training-overlap audit was performed.
- Corpus attribution: WikiText by its dataset authors, based on English Wikipedia contributions. The dataset metadata lists CC-BY-SA-3.0 and GFDL; the included corpus excerpt retains those upstream terms, separately from the model repository license. See `evaluation/CORPUS_LICENSE.txt`.
- Dataset revision: `b08601e04326c79dfdd32d625aee71d232d685c3`; exact file: `wikitext-2-raw-v1/test-00000-of-00001.parquet`; parquet SHA-256: `5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91`.
- Shared sample JSON SHA-256: `11d143c06478ff7bbe364c541a58e6a2e76504c4cb0595b294f46e98f3cfe798`.
- Byte targets are the next patch; incomplete final patches are discarded. The first patch is context only. LocalByteDecoder receives ground-truth target bytes and shifts them internally for causal teacher forcing within a patch. Cross-entropy is summed over all targets and divided by their total count.
- Byte loss is nats/byte; `BPB = NLL / ln(2)` and byte PPL is `exp(NLL)`. A uniform byte predictor gives BPB=8 and PPL=256. BPE loss uses Qwen2.5-3B token IDs with IDs outside the configured vocabulary mapped to 0, matching the dataset adapter. Its PPL is over that lossy mapped token stream and cannot be compared with byte PPL or interpreted as the likelihood of the original text.
- All BPE models shared one resolved Qwen tokenizer in this run. Its resolved revision is recorded in `run_01.json` and `environment.json`; reuse that tokenizer revision/cache for strict reproduction. The current evaluator resolves the tokenizer Hub name at runtime rather than hard-coding that revision.
- Causality: change the last 16 of 32 input steps and compare all prefix logits, tolerance `1e-5`. Recurrent chunk equivalence: compare a 128-step forward against four 32-step chunks across all output heads, tolerance `1e-4`. Failed checks remain visible.
- DeleteGate is disabled consistently: phase 1, elimination rate 0. The audit does not measure token elimination, phase-2 quality, or the benefit of DeleteGate.
- Forward timing: 10 warm-up steps, then 100 measured single-input forwards. `steps/s = 100 / total measured seconds`; p50 and p95 are per-step latency. One byte-model step consumes a patch of 4 bytes; one BPE step consumes one token. Local decoder inputs are zeros in this microbenchmark. Transformer timing has one input step and no persistent attention context. This is not a context-scaling test or a generation benchmark.
- Byte generation: three prompts, 128 new bytes per prompt, seeds 42/43/44, temperature 1.0, top-k 40, top-p 0.9. Recurrent models carry state; Transformers reprocess the growing context. Generated bytes/s includes prompt prefill and local sampling. UTF-8 validity is tested on the complete continuation. Samples are retained as hex and decoded text. BPE generation is N/A because vocabulary truncation prevents an equivalent byte-output comparison.
- Printable ASCII fraction, UTF-8 validity, and repetition are diagnostics; they do not measure factual accuracy, instruction following, or useful language quality. RAM columns report parameter storage, not a measured peak process footprint. Ternary models still hold FP32 parameters and simulate quantization in PyTorch.

Export repository revision audited: `65951289352c1a036b12ba6d888af037e7c90b13`. Archive inventory revision: `fb402d025e843ed00fb990966bf2430c32eb958c`. Each result records its checkpoint SHA-256, actual parameter count, saved step, configuration, thread count, time, sample hash, and evaluator hash.

### Reproduce

From the source checkout, install PyTorch plus `huggingface_hub`, `transformers`, `tokenizers`, `pyarrow`, `psutil`, `numpy`, and `datasets`. Pin the versions from `evaluation/environment.json` when reproducing this run.

The audit additions are bundled in `evaluation/source/`. Copy those Python files into the checkout’s `eval/` directory when using a source revision that predates this audit. The base Git commit is recorded in `evaluation/environment.json`.

```powershell
python eval/audit_readmes.py --storage D:/cpu1-readme-audit --threads 1
python eval/audit_readmes.py --storage D:/cpu1-readme-audit --cards
```

Model downloads, tokenizer files, Hugging Face caches, and temporary files go to D:. Only one model checkpoint is kept in the task storage at a time; it is removed after its result is saved. Interrupted runs resume from matching evaluator hashes. Raw reports remain in `resultados/readme_audit/`.

## Fresh diagnostic results

**Do not rank models across tokenizations or treat a failed-causality loss as autoregressive language-model perplexity.** Saved training metrics are retained separately in the JSONs; the numbers below are recomputed on the shared test sample.

### Byte models

| Run | Params M | Saved step | File MB | NLL | BPB | PPL | Causal | State chunks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [`run_02`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_02.json) | 38.830 | 647 | 155.35 | 1.9702 | 2.8424 | 7.172 | PASS | N/A |
| [`run_02_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_02_v3.json) | 38.830 | 10963 | 155.35 | ERROR | — | — | ERROR | N/A |
| [`run_02a_byte_only_heads`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_02a_byte_only_heads.json) | 38.501 | 641 | 154.04 | 2.4755 | 3.5714 | 11.888 | PASS | N/A |
| [`run_02a_byte_only_heads_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_02a_byte_only_heads_v3.json) | 38.501 | 16327 | 154.03 | ERROR | — | — | ERROR | N/A |
| [`run_03`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_03.json) | 38.830 | 647 | 155.38 | 2.1622 | 3.1194 | 8.690 | PASS | FAIL |
| [`run_03_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_03_v3.json) | 38.830 | 6187 | 155.38 | 1.7640 | 2.5449 | 5.836 | PASS | FAIL |
| [`run_04`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_04.json) | 38.854 | 647 | 155.49 | 5.5551 | 8.0144 | 258.565 | PASS | FAIL |
| [`run_04_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_04_r2.json) | 38.854 | 4840 | 155.49 | 5.5676 | 8.0323 | 261.803 | PASS | FAIL |
| [`run_04_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_04_v3.json) | 38.854 | 6074 | 155.49 | 1.5515 | 2.2384 | 4.719 | PASS | FAIL |
| [`run_05`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_05.json) | 38.994 | 649 | 156.06 | 5.5485 | 8.0048 | 256.854 | PASS | FAIL |
| [`run_05_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_05_v3.json) | 38.994 | 4390 | 156.06 | 1.5921 | 2.2969 | 4.914 | PASS | FAIL |
| [`run_05b_kernel_strict`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_05b_kernel_strict.json) | 35.842 | 600 | 143.44 | 5.5789 | 8.0486 | 264.778 | PASS | FAIL |
| [`run_05b_kernel_strict_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_05b_kernel_strict_v3.json) | 35.842 | 2573 | 143.44 | 1.8552 | 2.6764 | 6.393 | PASS | FAIL |
| [`run_06`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_06.json) | 38.994 | 649 | 156.06 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_06_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_06_v3.json) | 38.994 | 4237 | 156.06 | 1.0572 | 1.5253 | 2.878 | FAIL | FAIL |
| [`run_07`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_07.json) | 39.029 | 650 | 156.20 | 5.5535 | 8.0120 | 258.144 | FAIL | FAIL |
| [`run_07_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_07_r2.json) | 39.029 | 4860 | 156.20 | 5.5569 | 8.0169 | 259.010 | FAIL | FAIL |
| [`run_08`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_08.json) | 38.854 | 647 | 155.46 | 5.5535 | 8.0121 | 258.150 | PASS | N/A |
| [`run_08_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_08_v3.json) | 38.854 | 3858 | 155.46 | ERROR | — | — | ERROR | N/A |
| [`run_09`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_09.json) | 39.428 | 657 | 157.81 | 5.5334 | 7.9830 | 253.007 | FAIL | FAIL |
| [`run_10`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_10.json) | 39.434 | 657 | 157.83 | 5.5337 | 7.9834 | 253.068 | FAIL | FAIL |
| [`run_14`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_14.json) | 10.691 | 600 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_14_r2.json) | 10.691 | 1336 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_14_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_14_v3.json) | 10.691 | 9775 | 42.82 | 0.8811 | 1.2712 | 2.414 | FAIL | FAIL |
| [`run_15`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_15.json) | 10.732 | 600 | 42.99 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_15_r2.json) | 10.732 | 1341 | 42.99 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_15_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_15_v3.json) | 10.732 | 2350 | 42.99 | 1.3248 | 1.9113 | 3.761 | FAIL | FAIL |
| [`run_16`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_16.json) | 10.691 | 600 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |
| [`run_16_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_16_r2.json) | 10.691 | 1336 | 42.82 | 5.5700 | 8.0358 | 262.432 | FAIL | FAIL |

### BPE models (mapped Qwen token stream)

| Run | Params M | Saved step | File MB | NLL | BPB | PPL | Causal | State chunks |
|---|---:|---:|---:|---:|---:|---:|---|---|
| [`run_01`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_01.json) | 54.735 | 912 | 218.97 | 5.3742 | — | 215.773 | PASS | N/A |
| [`run_01_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_01_v3.json) | 54.735 | 12173 | 218.97 | ERROR | — | — | ERROR | N/A |
| [`run_13`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_13.json) | 12.548 | 600 | 50.24 | 6.3429 | — | 568.445 | PASS | FAIL |
| [`run_13_r2`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_13_r2.json) | 12.548 | 1568 | 50.24 | 6.4205 | — | 614.313 | PASS | FAIL |
| [`run_13_v3`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/run_13_v3.json) | 12.548 | 9290 | 50.24 | 6.7925 | — | 891.155 | PASS | FAIL |

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

The following fixed examples use the first audit prompt, `The history of artificial intelligence`, seed 42, and the sampling settings above. They illustrate why low diagnostic loss or valid UTF-8 alone does not establish useful text generation. Escapes preserve the recorded characters; complete byte sequences and all three prompts are in the JSON reports.

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

**Causality failures: 14/34.** `run_06`, `run_06_v3`, `run_07`, `run_07_r2`, `run_09`, `run_10`, `run_14`, `run_14_r2`, `run_14_v3`, `run_15`, `run_15_r2`, `run_15_v3`, `run_16`, `run_16_r2`

Changing future inputs changed prefix logits in these models. Some saved configurations enable noncausal Bolmo cross-patch operations; the test detects the effect without isolating its individual source. Their full-sequence losses are descriptive reconstruction scores and cannot establish causal prediction quality.

**State chunk failures: 26.** `run_03`, `run_03_v3`, `run_04`, `run_04_r2`, `run_04_v3`, `run_05`, `run_05_v3`, `run_05b_kernel_strict`, `run_05b_kernel_strict_v3`, `run_06`, `run_06_v3`, `run_07`, `run_07_r2`, `run_09`, `run_10`, `run_13`, `run_13_r2`, `run_13_v3`, `run_14`, `run_14_r2`, `run_14_v3`, `run_15`, `run_15_r2`, `run_15_v3`, `run_16`, `run_16_r2`

These configurations did not reproduce a full-sequence forward when carrying state between chunks at the stated tolerance. An O(1) state tensor alone does not establish correct streaming equivalence. No fix to the trained architectures or weights was made for this audit.

### Recurrent implementation diagnostic

The source’s `AblationMLGRU._parallel_scan` was tested independently against its defining serial recurrence with random candidate/forget tensors, seed 19, shape `[1,32,8]`, FP32. The maximum difference was **19.100506**; splitting the same scan into two 16-step chunks differed by **11.471270** (tolerance `1e-4`). See [`implementation_diagnostics.json`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints/blob/main/evaluation/implementation_diagnostics.json) and the bundled `audit_primitives.py`.

The implementation replaces log-cumulative-sum-exp with a cumulative-maximum normalization followed by a cumulative sum; accumulated terms are not rescaled when that maximum changes. The observed outputs therefore do not satisfy the intended recurrence. This is a source implementation defect; disabling or repairing the scan would create a different inference protocol and was not silently applied to these results. The trained-checkpoint audit uses the source as supplied.

### Evaluation errors

Scores and speed are unavailable when model loading or numerical evaluation fails. Nonfinite values are represented as JSON null; they are never substituted with valid-looking numbers.

| Run | Error |
|---|---|
| `run_01_v3` | ValueError: nonfinite loss |
| `run_02_v3` | ValueError: nonfinite loss |
| `run_02a_byte_only_heads_v3` | ValueError: nonfinite loss |
| `run_08_v3` | ValueError: nonfinite loss |

## Architecture recorded in each checkpoint

Run names identify experiments; dimensions and features below come from the saved configuration. FP16 is the training quantization label, while this audit executes the loaded tensors in FP32. “50M” and “10M” in older cards were design families, not exact model sizes.

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

These experiments change parameter counts, representations, training signal, budgets, saved steps, and sometimes multiple architectural settings. The audit is not a controlled causal estimate of the benefit of an individual component. `chinchilla_tokens_per_param` is a configured budget; it does not prove that a run completed that budget. Actual saved steps are reported separately.

## License

Hub metadata declares Apache-2.0. Dataset and tokenizer licenses are separate. The source checkout does not include a standalone LICENSE file.
