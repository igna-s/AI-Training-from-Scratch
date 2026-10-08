# Published experiment records

`published/manifest.json` links 34 FP32 exports to configuration JSON, model SHA-256, saved step, parameter count and the pinned audit result. See [the provenance review](../docs/provenance/README.md) for source history and platform recipes.

These configuration records preserve published architecture and budget settings. They were recorded by the evaluation loader, which disabled gradient checkpointing; GPU training enables it. They do not pin the exact original training source or environment.

```powershell
python -m training.ablation --run run_04_v3 --config experiments/published/configs/run_04_v3.json --skip-install
python -m training.ablation_amd --run run_04_v3 --config experiments/published/configs/run_04_v3.json --skip-install
```

These commands start real training. They need suitable hardware, dependencies and dataset access. Use the backend appropriate to the machine, and configure your own upload destination in a copied config before a new run. The historical `checkpoint_repo` value is preserved as provenance.

For an intermediate training checkpoint, append `--resume_ckpt PATH`. The recorded configuration must match the checkpoint and source. The recovery review did not perform a training-resume test or retrain models. An inference-only packed final is not equivalent to a full optimizer checkpoint.

Without `--config`, the CLI uses current presets and rejects unknown names such as `run_04_v3`. With an explicit configuration, phase defaults to its recorded `_current_phase` (or 1); another phase requires an explicit selection and appropriate checkpoint. Historical reports in Git and the October CPU evaluation use different protocols and remain distinct.
