# CPU-1: language-model ablation research

CPU-1 explores byte-patch language models, MLGRU recurrent mixers, and ternary projections. This repository contains training code, custom checkpoint codecs, native kernel experiments, and a reproducible CPU audit of the published ablation exports. The published checkpoints are small research models; the larger production configurations in the source are development targets.

## Published artifacts

| Repository | Contents |
|---|---|
| [Cukinator/cpu1-ablations-final](https://huggingface.co/Cukinator/cpu1-ablations-final) | 34 custom PyTorch FP32 exports, each with saved config and step |
| [Cukinator/cpu1-ablation-checkpoints](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints) | Packed finals and intermediate training checkpoints; exact inventory in the model card |
| [Cukinator/cpu1-ablation-dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) | Precomputed teacher signals for the dataset experiments |

## Verified status: 2026-10-06

The fresh audit attempted all 34 published exports with strict state compatibility, the same 32 WikiText test passages, and CPU forward timing and byte generation under one protocol. **14 models failed the prefix causality check; 26 failed recurrent chunk equivalence; 4 had a loading or numerical evaluation error.** Detailed failures and all raw evidence are retained, including samples that look printable but are not coherent language.

The audit does not establish a production-ready streaming language model. Some boundary-aware embedding configurations allow future inputs to affect earlier outputs. A separate primitive test also finds that the implemented parallel scan does not satisfy the intended serial recurrence; see [implementation_diagnostics.json](resultados/readme_audit/implementation_diagnostics.json). A small full-sequence loss from a model that fails causality is a reconstruction diagnostic, not evidence of causal next-byte prediction quality. Byte PPL and mapped BPE PPL are separate units. We do not infer component improvements from experiments with different parameter counts, training signals, saved steps, or budgets.

Read the complete [export model card](hf_readmes/cpu1-ablations-final.md) for per-model results, configurations, protocol and limitations. Read the [archive model card](hf_readmes/cpu1-ablation-checkpoints.md) for the pinned file inventory and storage formats. Historical files in `resultados/ablation_*.json` use older protocols and should not be combined with `resultados/readme_audit/`.

## Architecture and source

- `utilities.py`: BitLinear and BitEmbedding, RMSNorm, MLGRU and attention variants, SwiGLU, FPResidual, PFNet, byte patch embeddings, LocalByteDecoder, DeleteGate, dataset adapters, and checkpoint serialization.
- `train_ablation.py`: constructs an AblationModel from its saved configuration; trains individual experiments and orchestrates run groups. `train_ablation_amd.py` is the ROCm-oriented variant.
- `train_final.py`: larger CPU1ModelV2 training configurations. These are separate from the exported ablation collection.
- `create_dataset.py` and `kaggle/`: teacher signal extraction and dataset preparation.
- `inference/kernels.cpp`, `inference/benchmark.c`, and `inference/links.md`: native kernel experiments and build notes. The Python audit does not use these kernels. Synthetic native-kernel throughput must not be presented as measured trained-model generation speed.
- `eval/audit_readmes.py`: sequential downloads, scoring, correctness checks, generation samples, and timing. `eval/audit_model_cards.py`: cards and publication.

The exact model depends on `config` stored in the checkpoint. Experiments mix BPE and four-byte patches, Transformer and MLGRU mixers, floating-point and ternary projections, optional low-rank residuals, boundary-aware embedding, DeleteGate, PFNet, and learned decay. Check the configuration table instead of assuming all variants form a one-change chain.

## Setup and reproduce the audit

Use Python 3.12 with PyTorch installed for your hardware. For the CPU audit:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install torch transformers tokenizers huggingface_hub datasets numpy pyarrow psutil
.venv/Scripts/python eval/audit_readmes.py --storage D:/cpu1-readme-audit --threads 1
.venv/Scripts/python eval/audit_readmes.py --storage D:/cpu1-readme-audit --cards
```

Exact package versions are recorded in [environment.json](resultados/readme_audit/environment.json). Use them to reproduce the recorded run. The audit pins both Hub inventories and the dataset revision, saves the text sample, hashes each model, and records errors instead of silently filling missing weights. All model downloads, tokenizer caches, and temporaries go to D:. One checkpoint is kept at a time and removed after its JSON is saved. Existing results are reused only when the evaluator hash matches. Use `--force` to repeat or `--runs run_04 run_14_v3` to inspect a subset.

The scoring sample is diagnostic: 32 short passages, batch 1, no teacher model, DeleteGate disabled. Byte models use shifted next-patch targets with proper intra-patch teacher forcing. BPE uses the truncated Qwen vocabulary mapping used during training and reports the fraction mapped to ID 0. Forward steps/s and generated bytes/s are measured separately. The full protocol is in both cards.

Legacy `eval/run_eval.py` expects a production inference engine that is absent from this checkout; use the audit above for the published ablation models. `eval/eval_ablations.py` and `eval/compute_missing_val_loss.py` are legacy entry points and do not reproduce the new common corpus or cards.

## Load one checkpoint

Clone the [source repository](https://github.com/igna-s/New-1.58) and run this from its root. These are custom PyTorch checkpoints; `AutoModelForCausalLM.from_pretrained()` does not load them. The examples use a trusted checkpoint source and strict state loading.

```python
import os
os.environ["HF_HOME"] = "D:/cpu1/hf"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
from huggingface_hub import hf_hub_download
torch.set_num_threads(1)
path = hf_hub_download("Cukinator/cpu1-ablations-final", "run_04/model.pt",
    revision="65951289352c1a036b12ba6d888af037e7c90b13", local_dir="D:/cpu1/active", cache_dir="D:/cpu1/hf/hub")
from train_ablation import build_ablation_model
payload = torch.load(path, map_location="cpu", weights_only=False)
config, state = payload["config"], payload["state_dict"]
config["use_gradient_checkpointing"] = False
model = build_ablation_model(config)
model.load_state_dict(state, strict=True)
model.eval()
x = torch.tensor(list(b"The history of AI.  ")).reshape(1, -1, 4)
with torch.inference_mode():
    logits, states, extra = model(x, training_phase=1, target_elimination=0.0)
print([tuple(head.shape) for head in logits])
```

This forward uses zero target bytes in the local decoder; it is a shape smoke test. For scored likelihood, pass `byte_targets` as shown in the audit. For byte generation with retained Transformer context and one recurrent prefill, use `generation()` in `eval/audit_readmes.py`. Generation remains experimental; inspect the saved examples before choosing a checkpoint.

## Training

```powershell
python train_ablation.py --run run_04 --dry_run --total_steps 2
python train_ablation.py --run run_04
python train_ablation.py --run run_04 --resume_ckpt D:/cpu1/active/run_04/checkpoint_run_04_step640.pt
```

Normal training uses CUDA/ROCm-capable PyTorch and dataset access. The dry run uses synthetic CPU data and checks plumbing rather than model quality. Intermediate checkpoints may include optimizer state for resumption; packed finals are inference artifacts. `chinchilla_tokens_per_param` is a configured target, not proof of actual training exposure. All reported saved steps come from the evaluated payloads.

## Checkpoints and storage

- Export: `format="fp32"`, `state_dict`, `config`, `metrics`, `step`, `run_name`. Build from config and load strictly.
- Packed ternary final: `format="compact_2bit"`; use `load_ablation_checkpoint()` to unpack. Floating-point baselines have floating-point payloads.
- Training step: model/configuration and training state, typically BF16 storage. Inspect the payload rather than relying solely on its filename.

A ternary alphabet has log2(3)≈1.585 bits of information per value. The codec stores two bits per packed value plus scales and other floating-point tensors. Unpacked exports hold FP32 tensors; neither their disk footprint nor their Python resident parameter storage is 1.58 bits per parameter. Peak runtime RAM and native packed-inference speed are separate measurements.

## License

The Hugging Face model repositories declare Apache-2.0. This checkout has no standalone LICENSE file. Dataset and tokenizer assets have separate upstream licenses.
