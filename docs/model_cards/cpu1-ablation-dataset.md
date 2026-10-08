---
license: mit
task_categories:
- text-generation
language:
- en
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

# CPU-1 ablation dataset

Precomputed training data for the CPU-1 knowledge-distillation and tokenization experiments. The dataset contains independently packed byte-patch and BPE-token subsets with stored teacher probabilities. The extraction pipeline is configured to use [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) over English text from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT`. Teacher/source revisions and hidden-layer selection are not specified for every historical shard.

**The byte subset contains incomplete teacher arrays, and the companion dataset reader has positional-alignment defects. Resolve these limitations before using the stored signals for distillation.**

## Dataset contents

Data revision: `bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf`.

| Subset | Parquet files | Compressed size, decimal GB | Shard IDs | Missing IDs | Rows, HF dataset service |
|---|---:|---:|---|---|---:|
| `byte_marginalized` | 480 | 0.686841331 | 0000–0481 | 0060, 0061 | 24,101 |
| `bpe_tokenized` | 437 | 67.332026716 | 0000–0438 | 0060, 0061 | 21,928 |

Total: 917 parquet files, approximately 68.02 GB, and 46,029 rows according to the HF dataset service. Compressed file size differs from source-text size and loaded-memory requirements. Only a `train` split is declared.

The two streams pack text at different densities and maintain independent residual state and shard offsets. Matching shard IDs or row indices do not identify matching source spans. Load each subset independently; do not zip them by index. File globs skip the missing shard IDs.

## Stored fields

Lists are physically flattened in parquet. The writer uses four-byte patches, 5,000 steps per packed sequence, top-128 BPE candidates and a 128-dimensional hidden projection. Teacher arrays in the byte subset can be shorter than the input sequence or empty.

### Byte subset: `byte_marginalized`

| Field | Physical type | Nominal logical shape | Meaning |
|---|---|---|---|
| `patches` | list of uint8 | `[5000, 4]` | Input bytes grouped into four-byte patches. |
| `targets` | list of uint8 | `[5000, 4]` | Next-patch target bytes constructed before document packing. |
| `teacher_probs` | list of float32 | `[5000, 256]` | First-byte teacher distributions; short and empty arrays also occur. |
| `teacher_mask` | list of bool | `[5000]` | Teacher-signal coverage mask; short and empty masks also occur. |
| `teacher_hidden` | list of float32 | `[5000, 128]` | Projected teacher hidden values; short and empty arrays also occur. |
| `teacher_dim` | int32 | scalar | Hidden projection width, configured as 128. |

The target construction includes document-local shifts and separators. A global shift of the packed input row need not reproduce every target at document boundaries.

### BPE subset: `bpe_tokenized`

| Field | Physical type | Nominal logical shape | Meaning |
|---|---|---|---|
| `input_ids` | list of int32 | `[5000]` | Qwen BPE token IDs. |
| `labels` | list of int32 | `[5000]` | Next-token targets. |
| `teacher_probs_bpe` | list of float32 | `[5000, 128]` | Retained next-token probabilities. |
| `teacher_ids_bpe` | list of int32 | `[5000, 128]` | Vocabulary IDs paired with the retained probabilities. |
| `teacher_mask_bpe` | list of bool | `[5000]` | Teacher-signal availability mask. |

The stored probability columns contain probabilities. The BPE writer exponentiates retained log probabilities. The byte converter sums candidate probability masses by first byte and renormalizes the retained top-k mass. This approximates the full-vocabulary first-byte distribution; it does not provide a joint distribution over all four target bytes.

The extraction code defaults to a middle-layer hidden projection and uses last-layer states separately to compute logits. A last-layer origin for the stored hidden projection is not guaranteed.

## Teacher-signal layout and reader compatibility

Byte teacher arrays do not consistently span all 5,000 patch positions. Examples of stored layouts:

| Byte shard | Row, zero-based | Input byte count | Mask length | Hidden-value count |
|---|---:|---:|---:|---:|
| `0000` | 0 | 20,000 | 5,000 | 640,000 |
| `0000` | 1 | 20,000 | 42 | 5,376 |
| `0000` | 2 | 20,000 | 0 | 0 |
| `0481` | 13 | 20,000 | 513 | 65,664 |

Check array lengths before reshaping. A short boolean mask does not encode integer position indices, and its length alone does not identify the corresponding positions in the full sequence. Padding missing values does not recover their positional mapping.

The companion [ByteLevelDataset reader](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/702ac8b8616c25454dede9a13b057c4da0a8f97b/evaluation/source/cpu1/components.py) treats probability arrays as compact lists of supervised rows, even when the stored layout is already dense. It scatters the leading rows onto true-mask positions, moving or discarding teacher signals. For short masks, it casts boolean values to integer indices 0 and 1. Hidden arrays are scattered using the resulting mask and can also become misaligned.

A compatible reader must preserve dense positional layouts and handle short arrays only when their full-sequence positions are available. The linked reader requires changes before it can reliably consume these signals.

BPE-token boundaries do not always coincide with four-byte patch boundaries, so some patch positions lack a first-byte teacher distribution. This does not account for all missing or truncated arrays. Coverage varies between shards: `0000` has 10 of 50 rows with at least one true mask entry, while `0481` has 1 of 50. These examples do not define a dataset-wide coverage rate.

## UTF-8 limitations

The extractor uses character offsets to locate source spans, but converts individual candidate tokens to strings before encoding them as UTF-8. A token containing only part of a multibyte character can decode to the replacement character U+FFFD, changing the candidate bytes used for marginalization.

For example, the Qwen tokenizer at revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` splits U+1FAE8 in `A` + U+1FAE8 + `B` into token IDs `9284`, `104` and `101`, sharing character offset `(1, 2)`. Decoding each token independently produces U+FFFD, whose first UTF-8 byte is 239; the emoji begins with byte 240. Source-string offsets alone do not fix candidate-byte conversion. Exact BPE-to-byte alignment is therefore not guaranteed.

## Loading

Streaming avoids an upfront full download. Parquet reads can still require substantial transfer and memory.

```python
from datasets import load_dataset

rows = load_dataset(
    "Cukinator/cpu1-ablation-dataset",
    "byte_marginalized",  # or "bpe_tokenized"
    revision="bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf",
    split="train",
    streaming=True,
)

row = next(iter(rows))
for name, value in row.items():
    print(name, len(value) if isinstance(value, list) else value)
```

This example loads a row and displays its field lengths. Training requires an adapter that handles the actual signal layout and missing values.

The companion BPE reader maps IDs outside the configured student vocabulary to ID 0 and filters teacher IDs accordingly. This mapping is lossy. Loss on the mapped stream cannot be interpreted as likelihood under the full Qwen vocabulary or directly compared with per-byte loss. Establish document-level separation when creating training and evaluation splits.

## Reproducible inspection and source history

The [provenance review](https://github.com/igna-s/New-1.58/tree/main/docs/provenance) links the recovered historical extraction, training and platform code. Kaggle and Lightning launchers configure an 850,000-document target with chunks of 500 or 3,000 respectively. These are recipe settings, not verified document totals. Earlier teacher settings used 14B/7B; the exact extraction commit and teacher revision for each shard remain unestablished.

```powershell
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer
```

The [sample report](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset/blob/main/provenance/dataset-review.json) contains pinned SHA-256 values, schemas and footers for two byte shards (`0000`, `0481`) and one BPE shard (`0000`), all row lengths/masks in the two byte shards, the first BPE row, the dense-reader comparison and the synthetic tokenizer case. The parquet samples total about 158.5 MB and are stored on D:. No teacher or student model is loaded. Append `--offline` to reuse downloaded files.

The inspected dense byte row has 1,333 nonzero stored distributions; the linked reader returns 369 with a maximum absolute difference of 1.0. The first BPE row has zero next-token shift mismatches across 4,999 positions. A global shift comparison across the first packed byte rows observes 368 differences in `0000` and 40 in `0481`, across 19,996 positions each; document-local target construction and separators prevent interpreting that comparison alone as target corruption.

Inventory totals come from the pinned Hub listing; row totals are dataset-service reports, not a scan of every parquet footer. This limited inspection does not validate all teacher distributions, hidden-state alignment, per-shard provenance, document deduplication or train/evaluation independence. The dataset and model weights were not altered.

## Related resources

- [Source code](https://github.com/igna-s/New-1.58): extraction pipeline, dataset readers and training configurations.
- [Packed and intermediate checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints).
- [FP32 model exports](https://huggingface.co/Cukinator/cpu1-ablations-final).

## License

Dataset repository metadata declares MIT. Source text and teacher/tokenizer assets are subject to their respective upstream terms.
