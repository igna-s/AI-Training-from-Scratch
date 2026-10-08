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

Experimental training artifact for the CPU-1 knowledge-distillation and tokenization experiments. It contains byte-patch and BPE-token subsets with stored teacher signals. The reviewed extraction configuration names [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) as teacher and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb), `sample-10BT`, as source text. The provenance of every stored row was not independently verified.

**Known signal-formatting and reader-alignment problems are documented below. This dataset has not passed a dataset-wide validation of teacher-signal correctness.**

## Verified inventory: 2026-10-08

Data revision reviewed: `bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf`.

| Subset | Parquet files | File size, decimal GB | Shard IDs | Missing IDs within range | Rows reported by HF viewer |
|---|---:|---:|---|---|---:|
| `byte_marginalized` | 480 | 0.686841331 | 0000-0481 | 0060, 0061 | 24,101 |
| `bpe_tokenized` | 437 | 67.332026716 | 0000-0438 | 0060, 0061 | 21,928 |

File counts, sizes and gaps come from the Hub file inventory at the revision above. The [dataset service](https://datasets-server.huggingface.co/info?dataset=Cukinator%2Fcpu1-ablation-dataset) reported 46,029 rows in total on the review date. Row counts are service-reported; this review did not rescan the metadata of every parquet file. File sizes are compressed parquet sizes, not source-text sizes or loaded-memory requirements.

The byte and BPE streams use independent packing. Equal shard IDs or row indices do not establish matching source spans. Do not zip the subsets by index. The reason for the missing shards was not independently established.

The revision-pinned [stream checkpoint](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset/blob/bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf/stream_checkpoint.json) records `last_completed_chunk = 943`, `byte_shard_offset = 482` and `bpe_shard_offset = 439`. These are extraction bookkeeping counters, not a verified processed-document total or proof that extraction completed. No exact document total or future extraction schedule is asserted here.

## Stored fields

The schemas below were checked in byte shards `0000` and `0481`, and BPE shard `0000`. Lists are physically flattened. The reviewed writer configuration uses four-byte patches, 5,000 steps per sequence, top-128 BPE candidates and a 128-dimensional hidden projection. These settings do not guarantee identical array lengths in every stored row.

| Subset | Field | Physical type | Intended meaning |
|---|---|---|---|
| Byte | `patches` | list of uint8 | Input bytes grouped into four-byte patches. |
| Byte | `targets` | list of uint8 | Next-patch target bytes, as constructed by the writer. |
| Byte | `teacher_probs` | list of float32 | Flattened first-byte teacher distributions, with 256 values per distribution. |
| Byte | `teacher_mask` | list of bool | Intended coverage mask for teacher-supervised positions. Short and empty masks are present. |
| Byte | `teacher_hidden` | list of float32 | Flattened projected hidden values; short and empty arrays are present. |
| Byte | `teacher_dim` | int32 | Stored projection width; 128 in both inspected shards. |
| BPE | `input_ids` | list of int32 | Qwen BPE token IDs. |
| BPE | `labels` | list of int32 | Next-token targets, as constructed by the writer. |
| BPE | `teacher_probs_bpe` | list of float32 | Retained next-token probabilities. |
| BPE | `teacher_ids_bpe` | list of int32 | Vocabulary IDs paired with the retained probabilities. |
| BPE | `teacher_mask_bpe` | list of bool | Intended teacher-signal availability mask. |

The stored probability columns contain probabilities, not raw logits or log probabilities. The reviewed BPE writer exponentiates retained log probabilities. The byte converter sums candidate probability masses by first byte and renormalizes the retained top-k mass. This is an approximation to the full-vocabulary first-byte distribution, and does not provide a joint distribution over all four target bytes.

The reviewed extraction code defaults to a middle-layer hidden projection; it uses last-layer states separately to compute logits. The exact extraction version and hidden-layer selection for every historical shard were not established. A last-layer origin is therefore not guaranteed.

## Observed shape and alignment problems

Two byte shards were inspected, approximately 3.24 MB combined. BPE shard `0000` was also inspected (155.23 MB); its first row has 5,000 IDs and labels, 640,000 teacher values/IDs, and 5,000 mask entries. Its first-row next-token shift comparison has zero mismatches across 4,999 positions. Both have 50 rows and 20,000 input bytes per row. Teacher arrays do not consistently span all 5,000 patch positions:

| Shard | Row, zero-based | Input byte count | Mask length | Hidden-value count |
|---|---:|---:|---:|---:|
| `0000` | 0 | 20,000 | 5,000 | 640,000 |
| `0000` | 1 | 20,000 | 42 | 5,376 |
| `0000` | 2 | 20,000 | 0 | 0 |
| `0481` | 13 | 20,000 | 513 | 65,664 |

Do not blindly reshape every mask or hidden array to a 5,000-step shape. A short boolean mask is not a list of integer position indices, and its length alone does not recover its intended positions in the full sequence.

The reviewed [ByteLevelDataset reader](https://huggingface.co/Cukinator/cpu1-ablations-final/blob/702ac8b8616c25454dede9a13b057c4da0a8f97b/evaluation/source/cpu1/components.py) also has a reproduced alignment defect. In shard `0000`, row 0 stores a dense `[5000, 256]` probability array with 1,333 nonzero distributions and 1,333 true mask entries. The reader scatters the first 1,333 array rows onto true-mask positions, rather than retaining the already dense positional layout. Reproducing those array operations leaves only 369 nonzero distributions and gives a maximum absolute difference of 1.0 from the stored array. For shorter masks, the reader casts the observed booleans to integer indices 0 and 1.

This diagnostic demonstrates a mismatch between stored data and that reader. It does not measure the effect on a training run. Validate and resolve the signal layout and positional mapping before using this reader for distillation; simply padding arrays or interpreting booleans as indices does not establish correctness.

Shard `0000` has 10/50 rows with at least one true mask entry; shard `0481` has 1/50. These two observations are not a dataset-wide coverage estimate. Token/patch boundary alignment can explain sparse positions, but does not by itself establish why entire sequences have missing or short teacher arrays.

## UTF-8 limitations

There is no guarantee of exact BPE-to-byte alignment. The reviewed extraction code uses character offsets and converts individual candidate tokens to strings before encoding them as UTF-8. Partial byte tokens can decode to the replacement character instead of preserving their original bytes.

A test with the Qwen tokenizer at revision `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b` illustrates the issue: in `A` + U+1FAE8 + `B`, the emoji is split into token IDs `9284`, `104`, `101`, all sharing character offset `(1, 2)`. Each token decoded independently becomes U+FFFD, whose first UTF-8 byte is 239; the original emoji begins with byte 240. Correct source-string offsets alone do not fix this candidate-byte conversion. This test identifies an implementation limitation; it does not count affected positions across the dataset.

## Load and inspect a subset

Use the revision below to inspect the data reviewed by this card. Streaming avoids an upfront full download; parquet reads can still require substantial transfer and memory.

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

This is a loading and shape-inspection example, not a validated training adapter. Only a training split is declared. The review did not establish a held-out split, document-level deduplication or training/evaluation independence.

The companion BPE reader maps IDs outside the configured student vocabulary to ID 0 and filters teacher IDs accordingly. This is a lossy token representation. Loss on that mapped stream cannot be compared directly with byte loss or interpreted as likelihood under the full Qwen vocabulary.

## Review scope and related models

The review checked the Hub inventory, viewer-reported counts, extraction bookkeeping, two byte shards, the first BPE row and three file schemas/footers, reader reconstruction and tokenizer byte conversion. It did not load the teacher model, inspect every row, verify every target shift, establish historical extraction provenance, or validate teacher quality or hidden-state alignment across the dataset.

No conclusion about the cause of model-training failures follows from this review. Similar validation losses do not isolate quantization from data, alignment, optimization or implementation defects. The previous `2026-05 audit context` section has been removed.

Related artifacts:

- [Packed and intermediate checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)
- [FP32 exports and model-evaluation evidence](https://huggingface.co/Cukinator/cpu1-ablations-final)

## Reproduce the review and inspect source history

The [source repository](https://github.com/igna-s/New-1.58) includes the extraction pipeline, dataset readers and a [provenance review](https://github.com/igna-s/New-1.58/tree/main/docs/provenance). Historical Kaggle and Lightning extraction launchers, notebooks, earlier 14B/7B teacher configurations and the current 3B configuration are preserved in Git history. Committed launchers request 850,000 documents with chunks of 500 on Kaggle or 3,000 on Lightning; those settings are not verified processed-document totals. The exact teacher/source version for each shard remains unestablished.

```powershell
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer
```

This inspects three pinned parquet samples, verifies their SHA-256, reproduces the dense reader comparison and synthetic UTF-8 tokenizer case, and writes aggregate/schema evidence to `outputs/provenance/dataset-review.json`. It loads no teacher or student model. Use `--offline` to repeat with downloaded files. The published report is [provenance/dataset-review.json](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset/blob/main/provenance/dataset-review.json).

A global next-patch shift comparison across the first packed byte rows observed 368 differences in shard `0000` and 40 in `0481`, across 19,996 bytes each. The writer packs document targets and separators, so this global comparison does not isolate a target corruption. The stored targets were not rewritten. Source/teacher provenance and positional layout require separate validation before new training.

## License metadata

The dataset repository metadata declares MIT. This card reports that label; it does not establish that the label resolves the licensing of every source document or teacher/tokenizer asset. Consult the linked upstream repositories for their terms.
