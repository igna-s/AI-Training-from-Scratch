# Evaluation evidence

Code, configuration records and selected evaluation evidence are maintained in this GitHub repository. Hugging Face stores the model checkpoints and training datasets with their cards.

- `model-audit-2026-10-06/`: the 34 original FP32 model audit records, pinned model/data inventories, exact text sample and attribution, environment metadata, recurrence diagnostic and packed/FP32 loading example.
- `dataset-review-2026-10-08.json`: the limited three-shard dataset inspection and synthetic tokenizer/reader diagnostics.
- `architecture-review-2026-10-10.json`: offline module inventories for all 34 published configuration records in both constructors, all 30 current presets per backend, and five larger-model variants. This checks structure, not trained quality or GPU compatibility.
- `readme-validation-2026-10-10.json`: the later offline recheck of all 34 exports, small synthetic architecture checks, and validation of the revised README example. Previously measured full generation/timing remain historical.
- `documentation-review-2026-10-10.json`: checks across all current guides/cards, preserved records, source links, cached CPU examples, sample datasets, and eighteen regression tests. See the [documentation review](../../docs/documentation-review.md) for findings and limits.

The frozen October 6 records retain their original serialized bytes and hashes. The `.gitattributes` rule disables newline conversion in this evidence directory. `python -m evaluation.audit --storage D:/cpu1-readme-audit --restore-evidence` copies the committed model audit into ignored `outputs/audit/` without network access or model downloads.

The exact October 6 evaluator is preserved at [eval/audit_readmes.py](https://github.com/igna-s/AI-Training-from-Scratch/blob/bc58efaa7ef00dc24f57ea7479415520fb92705f/eval/audit_readmes.py). Its SHA-256 matches `environment.json` and every record's `suite_sha256`. The original source/environment fields are retained; they are not replaced by the publication commit. Moving the frozen records did not rerun their measurements; the October 10 recheck is recorded separately.

Model and dataset limitations remain in the cards and provenance review. These records do not establish dataset-wide teacher correctness or deterministic training reproduction. WikiText excerpts retain the upstream terms in `model-audit-2026-10-06/CORPUS_LICENSE.txt`.

The [architecture review](../../docs/architecture-review.md) distinguishes the code's paper-inspired components from unsupported reproduction and production claims. Structural checks are repeatable with `python -m evaluation.architecture_review`; the saved source hashes identify the reviewed implementations. The frozen October 6 records retain their original bytes.
