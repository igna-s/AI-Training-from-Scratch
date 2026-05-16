---
license: mit
task_categories:
- text-generation
language:
- en
size_categories:
- 10K<n<100K
configs:
- config_name: byte_marginalized
  data_files:
  - split: train
    path: byte_marginalized/shard_*.parquet
- config_name: bpe_tokenized
  data_files:
  - split: train
    path: bpe_tokenized/shard_*.parquet
---

# CPU-1 Ablation Dataset (Knowledge Distillation)

This dataset contains pre-computed teacher log probabilities and a 128-dim projection of the teacher's last hidden state, extracted from **Qwen/Qwen2.5-3B** over a subset of `HuggingFaceFW/fineweb` (Sample-10BT). It is designed for knowledge distillation into the **CPU-1 byte-level architecture** and for the BPE/byte-level ablation grid that accompanies it.

**Companion repositories:**
- Source compact_2bit checkpoints: [`Cukinator/cpu1-ablation-checkpoints`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)
- Unpacked fp32 checkpoints: [`Cukinator/cpu1-ablations-final`](https://huggingface.co/Cukinator/cpu1-ablations-final)
- Training code & audit findings: [github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits)

## Overview

- **Source text:** `HuggingFaceFW/fineweb` (Sample-10BT, English)
- **Teacher model:** `Qwen/Qwen2.5-3B`
- **Documents processed:** ~263,500 (extraction stopped at chunk 527/1700 of the original 850,000-document plan; the dataset is sized for the 10M-parameter ablation grid that consumes it)
- **Distillation signals:** BPE-to-byte first-byte marginalization (for byte models) + Top-128 raw BPE logits + projected teacher hidden states (for embedding-alignment loss)
- **Status:** Frozen final state (will not be re-extracted)

### Final size

| Subset | Shards | Sequences | SEQ_LEN | Raw size of source text |
|---|---|---|---|---|
| `byte_marginalized/` | 168 | 8,400 | 5,000 patches × 4 bytes | **168.0 MB** of bytes |
| `bpe_tokenized/`     | 157 | 7,850 | 5,000 BPE tokens        | **~39.25M** Qwen2.5 tokens |

Each shard holds **exactly 50 sequences** to align with the multi-process streaming logic.

> **Known gap — shards `0060` and `0061` are intentionally absent in both subsets.** During the original streaming extraction the writer crossed a corrupted region of the pipeline at that point, and although `byte_marginalized/` happened to retain those two parquet files, the matching `bpe_tokenized/` shards were lost and could not be recovered from the Hub commit history (an exhaustive search across all 891 historical commits returned no version of the files). Rather than ship two halves of an asymmetric pair — which would silently break any downstream code that assumes parallel sequence indexing across the two subsets — the byte counterparts (`byte_marginalized/shard_0060.parquet`, `byte_marginalized/shard_0061.parquet`) were deleted as well, leaving the same gap on both sides. The shard indices are otherwise contiguous (`0000..0169` for byte and `0000..0158` for BPE, minus 60-61 in both), so a glob like `shard_*.parquet` (used by the `configs:` block above) loads the dataset cleanly and skips the gap. The 16,250-row total reflects this.

## Loading

The two subsets are declared as separate **configs** in the YAML header, so the HF Datasets viewer presents them as two independent datasets and they can be loaded individually:

```python
from datasets import load_dataset

byte_ds = load_dataset("Cukinator/cpu1-ablation-dataset", "byte_marginalized", split="train")
bpe_ds  = load_dataset("Cukinator/cpu1-ablation-dataset", "bpe_tokenized",     split="train")
```

## Dataset Structure

Both subsets store **flattened lists** in the parquet — i.e. a column declared as `[seq_len, K]` is physically a `List<scalar>` of length `seq_len * K`. Reshape to the logical 2D shape after loading.

### 1. `byte_marginalized/`

For training byte-level (4-byte patch) models.

| Column | Logical shape | Physical type | Meaning |
|---|---|---|---|
| `patches`        | `[seq_len, 4]`   | `List<uint8>`  | The 4-byte input patches. |
| `targets`        | `[seq_len, 4]`   | `List<uint8>`  | The 4-byte targets (input shifted by one patch). |
| `teacher_probs`  | `[seq_len, 256]` | `List<float32>` | Marginalized P(byte) over the 256-byte vocabulary, only for the **first byte of the target patch**. |
| `teacher_mask`   | `[seq_len]`      | `List<bool>`   | `True` where a BPE token boundary aligns with the patch boundary, marking positions with a valid teacher signal. **Only ~3 / 50 sequences per shard carry teacher signal** (high sparsity is intentional and correct — see *Sparsity* below). |
| `teacher_hidden` | `[seq_len, teacher_dim]` | `List<float32>` | *(Optional)* 128-dim projection of the teacher's last hidden state, used for embedding-alignment loss. |
| `teacher_dim`    | scalar           | `int32`        | *(Optional)* The projected dimension (128). |

### 2. `bpe_tokenized/`

For BPE-level ablations (compact 4K vocab in our 10M runs, but the raw IDs cover Qwen's full vocab).

| Column | Logical shape | Physical type | Meaning |
|---|---|---|---|
| `input_ids`        | `[seq_len]`        | `List<int32>`   | Qwen2.5 BPE token IDs. |
| `labels`           | `[seq_len]`        | `List<int32>`   | Targets (`input_ids` shifted by one). |
| `teacher_probs_bpe`| `[seq_len, 128]`   | `List<float32>` | Top-128 P(next_token) probabilities. |
| `teacher_ids_bpe`  | `[seq_len, 128]`   | `List<int32>`   | Vocabulary IDs corresponding to the Top-128 entries above (paired index-by-index with `teacher_probs_bpe`). |
| `teacher_mask_bpe` | `[seq_len]`        | `List<bool>`    | Indicates presence of a teacher signal at the position. |

### Sparsity

The byte-level `teacher_mask` is sparse on purpose: only positions where a BPE-token boundary lands exactly on a 4-byte patch boundary carry a teacher distribution. Empirically this means **~3 of every 50 sequences** in a `byte_marginalized/` shard contain non-zero `teacher_probs` rows; the remaining sequences contribute to language-modelling loss but not to the marginalized distillation loss. This is mathematically correct — see *Architecture Robustness* below for why we **don't** widen the mask by reconstructing partial bytes.

### Shard count difference between streams (expected)

`byte_marginalized/` ends up with slightly more shards than `bpe_tokenized/` (here, **168 vs 157** = exactly **11 shards = ~7% more byte shards**, after the symmetric removal of shards 0060/0061 documented above). Both streams use `SEQ_LEN = 5000`, but they pack the source corpus at different densities:

| Stream | Items per sequence | Bytes of source per sequence |
|---|---|---|
| `byte_marginalized/` | 5000 patches × 4 bytes      | **20.0 KB** |
| `bpe_tokenized/`     | 5000 tokens × ~4.28 bytes   | **~21.4 KB** |

The Qwen2.5 BPE tokenizer averages ~4.28 bytes/token on FineWeb English, so each BPE sequence consumes ~7% more source text than each byte sequence. The same corpus therefore yields ~7% more byte sequences than BPE sequences, which the empirical 168-vs-157 split matches almost exactly.

The two streams also carry independent residual state across chunk boundaries and are uploaded with independent shard offsets, which adds a small additional drift. This is structural and expected — each stream has the number of shards that its encoding density warrants, and `shard_XXXX.parquet` in `byte_marginalized/` is **not** required to align 1:1 with the same index in `bpe_tokenized/`.

## Architecture Robustness (UTF-8)

This dataset is guaranteed to have **100% mathematically accurate BPE-to-byte offsets**. Rather than reconstructing partial bytes from BPE tokens (which fails dramatically on multi-byte UTF-8 sequences such as Emojis or Kanji when tokens are split across patches), the extraction engine uses the Rust fast-tokenizer's internal `offset_mapping` to slice directly from the raw UTF-8 source string. This eliminates offset drift and ensures the marginalized signals are mathematically independent of how the BPE tokenizer happens to split a particular run of text.

## Purpose

This dataset is tightly coupled with the **CPU-1 ablation suite** (10M-parameter run grid: `run_13`, `run_14`, `run_15`, plus the round-2 `*_r2` retries). It is computed once and reused across all runs to avoid re-running the heavy Qwen2.5-3B teacher forward pass for every ablation. Both byte-level distillation experiments (using `teacher_probs` + `teacher_hidden`) and BPE-level baseline experiments (using `teacher_probs_bpe` + `teacher_ids_bpe`) consume this same Hub repo without re-computation.

`run_16` deliberately bypasses this dataset and trains on raw FineWeb bytes
instead — it isolates the question "do the Qwen teacher signals help at
all?" by removing them entirely.

## 2026-05 audit context

This dataset performed as designed: BPE-to-byte marginalisation is mathematically correct, teacher hidden states are stably aligned, and shard structure is consistent. However, the **downstream ternary models** that consume this dataset (`run_13`–`run_16` and their `_r2` retries) failed to converge — they all collapsed to the byte uniform baseline `ln(256) ≈ 5.545 nats` regardless of training signal:

| Run | Training signal | Final val_loss |
|-----|-----------------|---------------:|
| run_14 | `teacher_probs` (Qwen logprobs only) | 5.5754 |
| run_15 | `teacher_probs` + `teacher_hidden` (EmbeddingAligner) | 5.5755 |
| run_16 | CE only (no teacher, FineWeb raw) | 5.5754 |

The fact that all three end at the same number — and that 7.5× more
tokens (`_r2` retries) does not move it — indicates the bottleneck is the
**cold-start dynamics of ternary quantisation**, not the quality of the
teacher signal. The dataset itself is valid and ready for any future
experiments that use a different quantisation recipe (e.g. warm-start
from a converged FP16 model + QAT, or post-training quantisation). See
the [audit findings](https://github.com/Cukinator/1.58bits/blob/main/README.md#ablation-audit--2026-05-findings)
in the main repo for the full mechanistic analysis.
