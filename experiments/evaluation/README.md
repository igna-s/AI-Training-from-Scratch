# Evaluation records

This directory contains the evaluation protocol, machine-readable results, and selected implementation checks. Hugging Face hosts the model checkpoints and training datasets. The [research scope and reproducibility guide](../../docs/reproducibility.md) explains what these records establish and what remains unverified.

| Record | Contents |
|---|---|
| [model-audit-2026-10-06/](model-audit-2026-10-06/) | 34 original FP32 results, pinned model/data inventories, exact text sample and attribution, environment metadata, recurrence diagnostic, and packed/FP32 loading example |
| [dataset-review-2026-10-08.json](dataset-review-2026-10-08.json) | Three fixed-shard inspections and synthetic tokenizer/reader diagnostics |
| [architecture-review-2026-10-10.json](architecture-review-2026-10-10.json) | Module inventories for 34 published configurations in both constructors, 30 current presets per backend, and five larger-model variants |
| [readme-validation-2026-10-10.json](readme-validation-2026-10-10.json) | Later recheck of all 34 exports, small synthetic architecture tests, and a CPU loading/scoring/generation example |
| [checksums.json](checksums.json) | SHA-256 values for the 42 original evaluation files |

The October 6 records retain their original serialized bytes and hashes. Their source, environment, and measurement dates identify the original protocol; the October 10 checks are separate records. Full original timing and generation were not repeated for all 34 exports.

The exact October 6 evaluator is available at [eval/audit_readmes.py](https://github.com/igna-s/AI-Training-from-Scratch/blob/bc58efaa7ef00dc24f57ea7479415520fb92705f/eval/audit_readmes.py). Its SHA-256 matches `environment.json` and each per-run `suite_sha256`.

Run `python -m evaluation.audit --storage D:/cpu1-readme-audit --restore-evidence` to copy the original records into `outputs/audit/` without model or dataset downloads. Repeat the structural inventory with `python -m evaluation.architecture_review`; see the [architecture guide](../../docs/architecture-review.md) for module definitions and numerical limitations.

The model and dataset cards retain failed checks alongside scores. Inventory metadata does not validate every archive binary or dataset row. WikiText excerpts keep their [upstream attribution and terms](model-audit-2026-10-06/CORPUS_LICENSE.txt), independently of the model license.
