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

# CPU-1 ablation training dataset

Companion dataset for the CPU-1 distillation experiments. The extraction source config uses Qwen/Qwen2.5-3B as teacher over FineWeb text. The byte and BPE subsets have different schemas and different shard counts; they must not be assumed to be aligned row by row.

## Verified Hub inventory: 2026-10-06

Revision: `bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf`. The following counts and compressed sizes are from the Hub file listing, not from a full scan of the dataset rows.

| Subset | Parquet files | Compressed GB | Shard range | Missing IDs within range |
|---|---:|---:|---|---|
| `byte_marginalized` | 480 | 0.687 | 0000–0481 | 0060, 0061 |
| `bpe_tokenized` | 437 | 67.332 | 0000–0438 | 0060, 0061 |

Sizes use decimal GB. Auxiliary `stream_checkpoint.json` and `stream_carry.pkl` files are extraction bookkeeping, not dataset examples. The file listing does not establish document totals, row totals, teacher coverage, split independence, or whether extraction has finished. Older shard and token totals must not be treated as current.

## Schema expected by the source adapters

| Subset | Fields | Interpretation |
|---|---|---|
| Byte | `patches`, `targets`, `teacher_probs`, `teacher_mask`; optional `teacher_hidden`, `teacher_dim` | Flattened byte patches and next-patch targets; teacher signal and mask. Hidden arrays may be compacted to teacher-covered positions; follow the adapter. |
| BPE | `input_ids`, `labels`, `teacher_probs_bpe`, `teacher_ids_bpe`, `teacher_mask_bpe` | Qwen token IDs and shifted labels; flattened teacher top-k values and their original IDs. |

This table describes `data_pipeline/prepare.py` and the readers in `cpu1/components.py`; the October audit did not validate every row or measure teacher-signal quality. Check actual row lengths, optional fields and `teacher_dim` rather than assuming a fixed hidden width or identical schemas in all shards.

The extraction source currently configures four-byte patches, 5,000-step sequences, and top-128 BPE teacher entries. Those are writer settings, not a claim that every stored row matches them. Byte teacher signals are first-byte marginalizations of BPE candidates at mapped boundaries; they are not full autoregressive distributions for all four positions. Training uses the coverage masks.

The BPE adapter maps IDs outside the configured student vocabulary to ID 0, and filters teacher IDs accordingly. This is a lossy representation. Its token perplexity cannot be compared directly with byte perplexity or with an untruncated Qwen language model.

## Stream one subset

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_DATASETS_CACHE"] = "D:/cpu1/datasets"
from datasets import load_dataset
dataset = load_dataset(
    "Cukinator/cpu1-ablation-dataset",
    "byte_marginalized",
    revision="bd75d7b811ab730cc912aaa671f9eaaf49ba4bdf",
    split="train", streaming=True,
    cache_dir="D:/cpu1/datasets",
)
row = next(iter(dataset))
print(row.keys())
```

Only a training split is declared by the Hub card. If you need held-out evaluation, create and document a deterministic independent split or use a separate test corpus. Never silently substitute the training split for validation. Do not zip the byte and BPE streams by shard index or row index.

## Model evaluation

The new model-card audit uses a separate 32-passage WikiText test sample and loads no teacher model. Its corpus, procedure and raw results are in the [FP32 model repository](https://huggingface.co/Cukinator/cpu1-ablations-final). That diagnostic does not evaluate this dataset’s teacher coverage, entropy, hidden-state alignment, or full training distribution.

## Related artifacts

- [Source and adapters](https://github.com/igna-s/New-1.58)
- [FP32 exports and common evaluation](https://huggingface.co/Cukinator/cpu1-ablations-final)
- [Packed finals and intermediate checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)

## License

The dataset Hub metadata declares MIT. FineWeb source data and Qwen tokenizer/teacher assets retain their own upstream licenses. The repository label does not replace those upstream terms.
