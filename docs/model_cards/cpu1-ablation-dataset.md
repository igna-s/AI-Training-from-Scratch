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

Precomputed training data for the CPU-1 knowledge-distillation and tokenization experiments. The dataset contains independently packed byte-patch and BPE-token subsets with stored teacher probabilities. The [extraction pipeline](https://github.com/igna-s/AI-Training-from-Scratch/blob/cpu1-experiment-2026-10-08/data_pipeline/prepare.py) is configured to use [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) over English text from [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT`. Teacher/source revisions and hidden-layer selection are not specified for every historical shard.

**The byte subset contains incomplete teacher arrays, and the companion dataset reader has positional-alignment defects. Resolve these limitations before using the stored signals for distillation.**

## Dataset contents

Data revision: `bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf`.

| Subset | Parquet files | Compressed size, decimal GB | Shard IDs | Missing IDs | Rows, HF dataset service |
|---|---:|---:|---|---|---:|
| `byte_marginalized` | 480 | 0.686841331 | 0000–0481 | 0060, 0061 | 24,101 |
| `bpe_tokenized` | 437 | 67.332026716 | 0000–0438 | 0060, 0061 | 21,928 |

Total: 917 parquet files, approximately 68.02 GB, and 46,029 rows according to the HF dataset service, checked on 2026-10-10. These row counts are service-reported totals, not a full independent scan of all shards. Compressed file size differs from source-text size and loaded-memory requirements. Only a `train` split is declared.

The two streams pack text at different densities and maintain independent residual state and shard offsets. Matching shard IDs or row indices do not identify matching source spans. Load each subset independently; do not zip them by index. File globs skip the missing shard IDs.

## Stored fields

Lists are physically flattened in parquet. The writer configuration uses four-byte patches, 5,000 steps per packed sequence, top-128 BPE candidates and a 128-dimensional hidden projection. Teacher arrays in the byte subset can be shorter than the input sequence or empty.

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

The companion [ByteLevelDataset reader](https://github.com/igna-s/AI-Training-from-Scratch/blob/cpu1-experiment-2026-10-08/cpu1/components.py) treats probability arrays as compact lists of supervised rows, even when the stored layout is already dense. It scatters the leading rows onto true-mask positions, moving or discarding teacher signals. For short masks, it casts boolean values to integer indices 0 and 1. Hidden arrays are scattered using the resulting mask and can also become misaligned.

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

When `bpe_vocab_size < 152064`, the companion `BPEDataset` reader maps input and target IDs at or above that limit to ID 0. When teacher signals are enabled, it also zeroes retained teacher probabilities outside that limit and replaces their IDs with 0. This mapping loses information when such IDs are present; with `bpe_vocab_size >= 152064`, the reader leaves input, target and teacher IDs unchanged. Loss on a vocabulary-reduced stream cannot be interpreted as likelihood under the full Qwen vocabulary. Per-token and per-byte losses use different units and cannot be compared directly without a common metric. Establish document-level separation when creating training and evaluation splits.

## Source and inspection records

Extraction, dataset readers, training code and configuration records are maintained in the [pinned GitHub source](https://github.com/igna-s/AI-Training-from-Scratch/tree/cpu1-experiment-2026-10-08). The [dataset inspection report](https://github.com/igna-s/AI-Training-from-Scratch/blob/cpu1-experiment-2026-10-08/experiments/evaluation/dataset-review-2026-10-08.json) contains the reviewed shard hashes, schemas, row lengths and tokenizer/reader diagnostics. It is a limited sample review, not dataset-wide validation.

## Related resources

- [GitHub: igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch): extraction pipeline, dataset readers and training configurations.
- [Hugging Face: Cukinator/cpu1-ablation-checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints): packed and intermediate checkpoints.
- [Hugging Face: Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final): FP32 model exports.

## License

Dataset repository metadata declares MIT. Source text and teacher/tokenizer assets are subject to their respective upstream terms.
