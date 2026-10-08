# Experiment provenance review — 2026-10-08

The initial cleanup excluded experimental source too broadly. This correction restores **148 historical paths** while keeping the working tree organized. The 87 original commits and the subsequent security commit retain their authors, dates, messages and parent relationships after remapping. Commit IDs change because their trees or ancestors change.

[restored-files.json](restored-files.json) lists every restored path, its original and remapped source commit, blob, SHA-256 and direct source link. It also lists the 109 paths that remain excluded. [history-map.json](history-map.json) maps the original commits to the public history. The retained file modes and blobs were compared in every original commit; the cleanup tip's tree is unchanged.

## What belongs in the project

| Material | Treatment |
|---|---|
| Earlier architectures, ablation configuration/model/training modules, extraction and byte-marginalization code | Preserved in Git history with original contents |
| Kaggle and Lightning launchers, SDK helpers, three notebooks and platform metadata | Preserved in history; current package-based recipes live in `runners/` |
| Eighteen Python files formerly under `models/ablation_models/` | Preserved in history as experimental source, distinct from downloaded weights |
| Historical evaluation and inference tools, including the earlier 26 checks | Preserved in history; current common evaluator remains in `evaluation/` |
| 34 historical `resultados/ablation_run_*.json` files and their summary | Preserved in history; historical metrics are not substituted for the October CPU audit |
| Model weights, downloaded tokenizer/config caches, parquet data, papers, compiled binaries and generated benchmark outputs | Excluded from Git; model/data artifacts are linked on Hugging Face |
| Personal prompts, reference summaries, private backups, environments and credentials | Remain local and ignored; design proposals are not treated as experimental evidence |

Historical code is an audit trail, not a claim that every old snapshot runs successfully. Six distinct historical `train_ablation.py` blobs have a syntax error. Earlier run numbers refer to different architectures; matching a name alone does not identify the same experiment. Use the current packages and an explicit recorded configuration.

## Training platforms and recorded configurations

The author confirmed using **Kaggle, Lightning and AMD**, with the code committed. That statement identifies the platforms used across the project; it does not assign each checkpoint to a job.

- Kaggle dataset launcher: target 850,000 documents, chunks of 500, initial shard offset 0, streaming upload.
- Lightning dataset launcher: target 850,000 documents, chunks of 3,000, initial shard offset 0, streaming upload.
- These are committed recipe settings, not verified document totals. Resume bookkeeping and changed extraction versions prevent deriving exposure from those settings alone.
- The current AMD preset uses 150 configured tokens/parameter and the historical `cpu1-ablation-checkpoints-v3` upload destination. Published `_v3` configurations are consistent with those settings. This does not prove the exact job platform or source commit.

[experiments/published/manifest.json](../../experiments/published/manifest.json) indexes the **34 published FP32 exports**, checkpoint SHA-256 values, pinned model revision, saved steps, parameter counts and configuration files. Configurations come from the pinned CPU audit records. The evaluation loader disabled `use_gradient_checkpointing` before recording them; they are not byte-for-byte original checkpoint config payloads. The trainer enables gradient checkpointing for actual GPU training.

`--config` loads the record without merging current presets. Explicit configurations run their selected phase instead of silently chaining another phase. The records include historical upload destinations; copy a record and set your own destination when starting a new experiment. Saved steps are actual checkpoint metadata; configured token budgets are not measured completion.

## Current platform recipes

Run from a writable checkout, such as `/kaggle/working/New-1.58` or a Lightning Studio project. Install the appropriate PyTorch build and `requirements/data.txt` or `requirements/training.txt` first. The recipes run inside the selected platform; they do not submit remote jobs through the platform SDK.

```bash
# Prints the committed dataset recipe without extracting or uploading.
python -m runners.kaggle dataset
python -m runners.lightning dataset

# Executes extraction and streaming upload; configure your own HF_TOKEN first.
python -m runners.kaggle dataset --execute
python -m runners.lightning dataset --execute

# Preview a recorded training configuration.
python -m runners.lightning ablation --backend amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json
# Append --execute to run, with a compatible ROCm installation and GPU.
```

Kaggle execution can read `HF_TOKEN` from Kaggle Secrets; Lightning/AMD use the environment. Tokens are never printed by these recipes. Optional HF mounts use `cpu1/hub_storage.py`; install hf-mount separately on Linux/macOS. Training mounts a dataset repository for reading. The helper passes credentials in the child environment and does not run a remote installer.

## Evidence and unresolved provenance

The [dataset card](../model_cards/cpu1-ablation-dataset.md) documents the 917 parquet files, revision, sample scope, teacher formatting, observed reader alignment defect and UTF-8 limitations. The inspection is repeatable:

```powershell
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer
# Reuse the same three downloaded shards/tokenizer without network access:
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer --offline
```

This downloads about 158.5 MB of parquet samples to D:, not a teacher or student model. Results are written under ignored `outputs/provenance/`. Each sample is checked against its reviewed SHA-256. To publish the reviewed dataset card with owner authentication, append `--publish-card`; code and review evidence remain on GitHub under `experiments/evaluation/`; all parquet paths, sizes and LFS hashes must still match the reviewed data revision. All 34 recorded configurations were checked against their audited parameter counts using meta tensors; no model weights were downloaded for that check. Fourteen offline tests passed.

The available artifacts do not bind each training job to its exact source commit, dependency lock, resolved teacher/tokenizer revision, extraction version, source documents or projection matrix. Restoring committed code makes its evolution inspectable; it does not fill those missing records. A deterministic end-to-end retraining or dataset reconstruction is therefore not established. The stored model/data artifacts and the known numerical implementations were left unchanged; existing audit results retain their original date and limitations.
