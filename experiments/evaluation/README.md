# Evaluation evidence

Code, configuration records and selected evaluation evidence are maintained in this GitHub repository. Hugging Face stores the model checkpoints and training datasets with their cards.

- `model-audit-2026-10-06/`: the 34 original FP32 model audit records, pinned model/data inventories, exact text sample and attribution, environment metadata, recurrence diagnostic and packed/FP32 loading example.
- `dataset-review-2026-10-08.json`: the limited three-shard dataset inspection and synthetic tokenizer/reader diagnostics.

The records retain their original serialized bytes and hashes. The `.gitattributes` rule disables newline conversion in this evidence directory. `python -m evaluation.audit --restore-evidence` copies the committed model audit into ignored `outputs/audit/` without network access or model downloads.

The exact October 6 evaluator is preserved at [eval/audit_readmes.py](https://github.com/igna-s/New-1.58/blob/bc58efaa7ef00dc24f57ea7479415520fb92705f/eval/audit_readmes.py). Its SHA-256 matches `environment.json` and every record's `suite_sha256`. The original source/environment fields are retained; they are not replaced by the publication commit. No model evaluation was rerun while moving these files.

Model and dataset limitations remain in the cards and provenance review. These records do not establish dataset-wide teacher correctness or deterministic training reproduction. WikiText excerpts retain the upstream terms in `model-audit-2026-10-06/CORPUS_LICENSE.txt`.
