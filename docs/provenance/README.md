# Experiment provenance and platform recipes

The published artifacts document a family of small language-model experiments with Transformer and MLGRU-inspired mixers, byte and BPE inputs, ternary weights, and optional Qwen teacher supervision. Experiments used Kaggle, Lightning, and AMD resources across the project. The available records do not identify the platform and exact source of every checkpoint's training job.

## Checkpoints and configurations

The [experiment manifest](../../experiments/published/manifest.json) indexes **34 FP32 exports** at Hugging Face revision `65951289352c1a036b12ba6d888af037e7c90b13`. Each entry links its checkpoint SHA-256, saved step, parameter count, configuration, and evaluation result. The [model card](../model_cards/cpu1-ablations-final.md) reports their common CPU evaluation.

Configuration records describe the loaded models. The evaluation loader disabled `use_gradient_checkpointing` before recording them, so they are not byte-for-byte copies of the checkpoint configuration payloads. GPU training enables gradient checkpointing. Saved steps come from checkpoint metadata; token budgets specify planned work and do not establish completed exposure or convergence.

Run names are identifiers, not complete architecture specifications. Earlier source versions and current presets can assign different settings to the same name. Use an explicit recorded configuration when inspecting a published export.

The [historical source index](source-versions.json) identifies 40 distinct versions of the dataset and ablation entry points by Git commit, path, and blob. Six historical `train_ablation.py` versions have a syntax error. This index documents source evolution; it does not assign a version to each training job. Current entry points live in [training/](../../training/) and [data_pipeline/](../../data_pipeline/).

## Recorded platform settings

| Recipe | Recorded settings | Interpretation |
|---|---|---|
| Kaggle dataset extraction | 850,000 target documents, chunks of 500, initial shard offset 0, streaming upload | A configured target, not a verified processed total |
| Lightning dataset extraction | 850,000 target documents, chunks of 3,000, initial shard offset 0, streaming upload | A configured target, not a verified processed total |
| AMD ablation base | 150 configured tokens/parameter; historical `cpu1-ablation-checkpoints-v3` destination | Consistent with published `_v3` records, without proving their job platform or source |

Changed extraction versions and resume bookkeeping prevent reconstructing exact document exposure from these settings alone.

## Preview and run a recipe

Use a writable checkout inside the selected platform, such as `/kaggle/working/AI-Training-from-Scratch` or a Lightning Studio project. Install the appropriate PyTorch build and [data requirements](../../requirements/data.txt) or [training requirements](../../requirements/training.txt) first. These runners execute locally inside the platform; they do not provision or submit remote jobs.

```bash
# Preview dataset extraction settings.
python -m runners.kaggle dataset
python -m runners.lightning dataset

# Preview a recorded AMD training configuration.
python -m runners.lightning ablation --backend amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json
```

The commands print plans until `--execute` is supplied. Dataset execution extracts teacher signals and uploads shards. Kaggle can read `HF_TOKEN` from Kaggle Secrets; Lightning and AMD use the environment. For a new training experiment, copy the configuration, set `checkpoint_repo` to your own destination or `null`, and choose a fresh run name. With an explicit configuration, the selected phase runs without automatically chaining another phase. See the [usage guide](../usage.md#training-and-teacher-data) for resume behavior and hardware requirements.

## Dataset inspection

The [dataset card](../model_cards/cpu1-ablation-dataset.md) records the 917-file inventory and observations from three fixed Parquet shards. It includes reader alignment, mask-format, and UTF-8 limitations. Repeat that sample inspection with:

```powershell
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer
# Reuse cached shards and tokenizer files.
python -m evaluation.dataset --storage D:/cpu1-readme-audit --tokenizer --offline
```

The three shards total approximately **158.5 MB**. An uncached `--tokenizer` also downloads tokenizer/configuration files. Teacher and student weights are not required. The inspector verifies the recorded shard SHA-256 values and writes results to `outputs/provenance/`. This inspection does not validate all dataset rows or reconstruct the original extraction jobs.

## Reproducibility limits

The artifacts do not bind every training job to its exact source commit, dependency lock, resolved teacher/tokenizer revision, extraction version, source documents, or projection matrix. A historical source index and loadable checkpoint do not fill those gaps. Deterministic end-to-end retraining and dataset reconstruction remain unverified. The [research scope and reproducibility guide](../reproducibility.md) distinguishes measured inference outcomes from these open questions.
