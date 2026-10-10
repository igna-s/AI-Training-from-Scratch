"""Generate final Hub model cards from recorded evaluation results."""
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import re
import sys

from evaluation.audit import ROOT, OUT, FINAL, ARCHIVE, DATA, DATA_REV, DATA_FILE, PROMPTS, sha
from evaluation.publication import SOURCE_TREE, SOURCE_BLOB, card_operations

RESULTS_TREE = f"{SOURCE_TREE}/experiments/evaluation"


def link(repo, filename, revision='main'):
    return f'https://huggingface.co/{repo}/blob/{revision}/{filename}'


def manifest_rows(inventory):
    folders = defaultdict(list)
    for f in inventory[ARCHIVE]['siblings']:
        if f['rfilename'].endswith('.pt'):
            folders[f['rfilename'].split('/')[0]].append(f)
    rows = ['| Folder | Step files | Highest filename step | Final file MB | Total GB |', '|---|---:|---:|---:|---:|']
    for run, files in sorted(folders.items()):
        steps = [int(m.group(1)) for f in files if (m := re.search(r'_step(\d+)\.pt$', f['rfilename']))]
        finals = [f for f in files if f['rfilename'].endswith('_final.pt')]
        final = f'{sum(f["size"] for f in finals)/1e6:.2f}' if finals else '—'
        rows.append(f'| `{run}` | {len(steps)} | {max(steps) if steps else "—"} | {final} | {sum(f["size"] for f in files)/1e9:.3f} |')
    return rows


def evaluation_summary(records):
    checks = [r.get('correctness', {}) for r in records]
    causal = [c.get('causality', {}) for c in checks]
    recurrent = [c['state_prop'] for c in checks if c.get('state_prop', {}).get('status') != 'N/A']
    return [
        '## Evaluation summary', '',
        f'The evaluation covers {len(records)} FP32 exports on 32 short WikiText test passages, with CPU execution, one thread, and a fixed protocol. It does not evaluate every intermediate archive checkpoint.', '',
        '| Check | Outcome |', '|---|---|',
        f'| Strict checkpoint loading | {sum(r.get("strict_load", False) for r in records)}/{len(records)} loaded |',
        f'| Numerical evaluation | {sum(bool(r.get("error")) for r in records)} models produced an evaluation error |',
        f'| Prefix causality | {sum(c.get("status") == "FAIL" for c in causal)}/{sum(c.get("status") in ("PASS", "FAIL") for c in causal)} finite-output models failed |',
        f'| Recurrent chunk equivalence | {sum(c.get("status") == "FAIL" for c in recurrent)}/{len(recurrent)} recurrent models failed |', '',
        'Full results, generation examples, and the recurrent implementation diagnostic follow below. Loss units depend on tokenization. Loss from a model that fails causality cannot establish autoregressive prediction quality; printable output alone does not establish useful language generation.', '',
    ]


def protocol(inventory, corpus, records):
    return [
        '## Evaluation methodology', '',
        f'Results were measured on **2026-10-06** for the {len(records)} FP32 exports. Raw results, checkpoint hashes, the text sample, package versions, corpus attribution and evaluator metadata are available in the [GitHub evaluation files]({RESULTS_TREE}).', '',
        '- Hardware: Intel Core i7-8750H, Windows 11, CPU execution, one PyTorch thread, batch size 1 and FP32 parameters. PyTorch ' + records[0]['torch'] + '. The measurements use the PyTorch implementation without custom C/C++ kernels.',
        '- Text: the first 32 non-heading paragraphs of the WikiText-2 raw test split with at least 260 UTF-8 bytes, truncated to 260 bytes at a complete character boundary. This sample supports small-scale comparisons; it is not the full benchmark. Training/test overlap has not been exhaustively checked.',
        f'- Dataset: [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), revision `{DATA_REV}`, file `{DATA_FILE}`. Parquet SHA-256: `{corpus["parquet_sha256"]}`. Sample JSON SHA-256: `{sha(OUT/"corpus.json")}`.',
        '- Corpus license: WikiText text derives from English Wikipedia contributions. The dataset metadata lists CC-BY-SA-3.0 and GFDL; CORPUS_LICENSE.txt in the evaluation files preserves the attribution and terms separately from the model license.',
        '- Byte scoring predicts the next four-byte patch. The first patch supplies context and incomplete final patches are discarded. LocalByteDecoder receives ground-truth target bytes, shifted internally for causal teacher forcing within the patch. Cross-entropy is summed over targets and divided by their total count.',
        '- Byte NLL is in nats/byte; BPB = NLL / ln(2), and byte PPL = exp(NLL). A uniform byte predictor has BPB=8 and PPL=256. BPE scoring uses Qwen2.5-3B token IDs, mapping IDs outside the saved vocabulary cap to 0. BPE PPL describes this lossy mapped stream; it is not full-vocabulary text likelihood and is not comparable to byte PPL.',
        '- The resolved Qwen tokenizer revision was not recorded. This limits exact reproduction of the BPE measurements; the environment and evaluator hashes are preserved in the result files.',
        '- Prefix causality changes the last 16 of 32 input steps and compares earlier logits at tolerance 1e-5. State chunk equivalence compares one 128-step forward with four 32-step chunks across all output heads at tolerance 1e-4.',
        '- DeleteGate uses phase 1 with elimination rate 0. Results do not measure phase-2 quality or the effect of token elimination.',
        '- Forward speed uses 10 warm-up steps and 100 measured single-step forwards. A step consumes four bytes for byte models or one mapped token for BPE models. Local decoder inputs are zeros; Transformer forwards have no persistent attention context. This microbenchmark does not measure growing-context throughput. Parameter MB reports FP32 parameter storage, not peak process RAM.',
        '- Byte continuations use three prompts, 128 new bytes each, seeds 42/43/44, temperature 1.0, top-k 40 and top-p 0.9. Recurrent models carry state; Transformers reprocess the growing context. Generated bytes/s includes prompt prefill and sampling. UTF-8 validity is checked on each complete continuation. BPE generation is not included because of the lossy vocabulary mapping.', '',
        f'Evaluated export revision: `{inventory[FINAL]["sha"]}`. Archive inventory revision: `{inventory[ARCHIVE]["sha"]}`.', '',
        '### Code and result files', '',
        f'The [GitHub source]({SOURCE_TREE}) contains the model implementation, training recipes and evaluation tools. The [evaluation files]({RESULTS_TREE}) include all 34 per-run JSON results, exact corpus, dependency versions, checkpoint hashes and the reference evaluator location. Use these records when comparing or reproducing the measurements.', '',
    ]


def tables(records):
    lines = ['## Evaluation results', '',
             '**Do not rank models across tokenizations or treat a failed-causality loss as autoregressive language-model perplexity.** Saved training metrics are retained separately in the JSONs; the numbers below were measured on the shared test sample.', '']
    for byte in (True, False):
        lines += ['### Byte models' if byte else '### BPE models (mapped Qwen token stream)', '',
                  '| Run | Params M | Saved step | File MB | NLL | BPB | PPL | Causal | State chunks |',
                  '|---|---:|---:|---:|---:|---:|---:|---|---|']
        for r in records:
            if 'config' not in r or (r['config']['tokenization'] in ('byte','byte_dynamic')) != byte:
                continue
            lm = r.get('loss')
            cc = r['correctness']
            if lm is None:
                lines.append(f'| [`{r["run"]}`]({link(FINAL, r["file"], r["revision"])}) | {r["parameters"]/1e6:.3f} | {r["step"]} | {r["file_bytes"]/1e6:.2f} | ERROR | — | — | {cc["causality"]["status"]} | {cc["state_prop"]["status"]} |')
                continue
            bpb = f'{lm["bits_per_byte"]:.4f}' if byte else '—'
            lines.append(f'| [`{r["run"]}`]({link(FINAL, r["file"], r["revision"])}) | {r["parameters"]/1e6:.3f} | {r["step"]} | {r["file_bytes"]/1e6:.2f} | {lm["mean_nll"]:.4f} | {bpb} | {lm["perplexity"]:.3f} | {cc["causality"]["status"]} | {cc["state_prop"]["status"]} |')
        if not byte:
            lines += ['', '| BPE run | IDs mapped to 0 | Vocabulary cap |', '|---|---:|---:|']
            for r in records:
                if r.get('config',{}).get('tokenization') == 'bpe' and 'loss' in r:
                    lines.append(f'| `{r["run"]}` | {r["loss"]["mapped_ids_fraction"]*100:.2f}% | {r["config"]["bpe_vocab_size"]:,} |')
        lines += ['']
    lines += ['### CPU timing and byte generation', '', '| Run | Param MB (FP32) | Forward steps/s | p50 ms | p95 ms | Generated bytes/s | Valid UTF-8 samples |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        bm = r.get('benchmark')
        gen = r.get('generation')
        if bm is None:
            lines.append(f'| `{r["run"]}` | {r.get("parameter_bytes",0)/1e6:.2f} | ERROR | — | — | N/A | N/A |')
            continue
        if isinstance(gen, list):
            speed = f'{sum(s["bytes"] for s in gen)/sum(s["seconds"] for s in gen):.2f}'
            utf = f'{sum(s["utf8_valid"] for s in gen)}/{len(gen)}'
        else:
            speed = utf = 'N/A'
        lines.append(f'| `{r["run"]}` | {r["parameter_bytes"]/1e6:.2f} | {bm["forward_steps_per_second"]:.2f} | {bm["p50_ms"]:.2f} | {bm["p95_ms"]:.2f} | {speed} | {utf} |')
    lines += ['', '### Example continuations', '',
        'The following fixed examples use the first evaluation prompt, `The history of artificial intelligence`, seed 42, and the sampling settings above. They illustrate why low diagnostic loss or valid UTF-8 alone does not establish useful text generation. Escapes preserve the recorded characters; complete byte sequences and all three prompts are in the JSON reports.', '']
    for name in ('run_04_v3','run_14_v3'):
        r = next((r for r in records if r['run']==name),None)
        if r and isinstance(r.get('generation'),list):
            lines += [f'`{name}`:', '', '```text', json.dumps(r['generation'][0]['text_replacement_decode'],ensure_ascii=True), '```', '']
    lines += ['', '### Correctness checks', '', '| Run | Shapes | Probabilities | Determinism | State chunks | Ternary projection | State size | Causality |', '|---|---|---|---|---|---|---|---|']
    keys = ['forward_shapes','probs','determinism','state_prop','ternary_weights','state_size','causality']
    for r in records:
        lines.append('| `' + r['run'] + '` | ' + ' | '.join(r.get('correctness',{}).get(k,{}).get('status','ERROR') for k in keys) + ' |')
    failed_causal = [r['run'] for r in records if r.get('correctness',{}).get('causality',{}).get('status')=='FAIL']
    failed_state = [r['run'] for r in records if r.get('correctness',{}).get('state_prop',{}).get('status')=='FAIL']
    lines += ['', f'**Causality failures: {len(failed_causal)}/{sum(r.get("correctness",{}).get("causality",{}).get("status") in ("PASS", "FAIL") for r in records)} finite-output models.** ' + (', '.join(f'`{r}`' for r in failed_causal) or 'None.'), '',
              'Changing future inputs changed prefix logits in these models. Some saved configurations enable noncausal Bolmo cross-patch operations; the test detects the effect without isolating its individual source. Their full-sequence losses are descriptive reconstruction scores and cannot establish causal prediction quality.', '',
              f'**State chunk failures: {len(failed_state)}.** ' + (', '.join(f'`{r}`' for r in failed_state) or 'None.'), '',
              'These configurations did not reproduce a full-sequence forward when carrying state between chunks at the stated tolerance. An O(1) state tensor alone does not establish correct streaming equivalence. These results describe the published implementation and checkpoints.', '']
    diagnostic = OUT/'implementation_diagnostics.json'
    if diagnostic.exists():
        d = json.loads(diagnostic.read_text(encoding='utf-8'))
        lines += ['### Recurrent scan implementation', '',
            f'The source’s `AblationMLGRU._parallel_scan` was tested independently against its defining serial recurrence with random candidate/forget tensors, seed 19, shape `[1,32,8]`, FP32. The maximum difference was **{d["serial_max_diff"]:.6f}**; splitting the same scan into two 16-step chunks differed by **{d["chunks_max_diff"]:.6f}** (tolerance `1e-4`). See the [recorded implementation results]({RESULTS_TREE}) and [evaluation/diagnostics.py]({SOURCE_BLOB}/evaluation/diagnostics.py).', '',
            'The implementation replaces log-cumulative-sum-exp with a cumulative-maximum normalization followed by a cumulative sum; accumulated terms are not rescaled when that maximum changes. The observed outputs therefore do not satisfy the intended recurrence. This is a source implementation defect; disabling or repairing the scan would create a different inference protocol and was not silently applied to these results. The measurements use the published implementation.', '']
    errors = [r for r in records if r.get('error')]
    if errors:
        lines += ['### Evaluation errors', '', 'Scores and speed are unavailable when model loading or numerical evaluation fails. Nonfinite values are represented as JSON null; they are never substituted with valid-looking numbers.', '', '| Run | Error |', '|---|---|']
        lines += [f'| `{r["run"]}` | {r["error"].replace("|","/")} |' for r in errors]
        lines += ['']
    return lines


def architecture(records):
    lines = ['## Training and architecture', '',
             f'Training, dataset preparation and Kaggle/Lightning recipes are maintained in the [GitHub source]({SOURCE_TREE}). The source provides an AMD/ROCm-oriented entry point; these CPU results do not validate a fresh GPU training job. The [saved configurations]({SOURCE_TREE}/experiments/published) record the 34 exported runs, checkpoint hashes, parameter counts and saved steps. Current presets can differ from the published runs; select a recorded configuration explicitly.', '',
             'Configured token budgets, including the v3 150 tokens/parameter setting, are not measurements of completed training exposure. The exact source commit and full dependency lock for each training job were not recorded.', '',
             'The [training dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset) contains teacher signals with documented short/empty arrays, dense-signal alignment limitations in the reader and UTF-8 conversion limitations. Their effect on each trained checkpoint has not been isolated.', '',
             '### Saved architecture settings', '',
             'Dimensions and features below come from the recorded configurations. The quantization column is the training label; all evaluated exports contain FP32 parameters. Parameter counts are measured from the loaded models.', '',
             '| Run | Mixer | Tokens | Quant. label | Width / layers / FFN | Local decoder | FP residual | Bolmo | DeleteGate | PFNet hidden | Channel decay |',
             '|---|---|---|---|---|---|---|---|---|---:|---|']
    for r in records:
        if 'config' not in r:
            continue
        c = r['config']
        yes = lambda k: 'yes' if c.get(k, False) else 'no'
        lines.append(f'| `{r["run"]}` | {c["token_mixer"]} | {c["tokenization"]} | {c["quantization"]} | {c["d_model"]} / {c["n_layers"]} / {c["d_ff"]} | {yes("use_local_decoder") if c["tokenization"]!="bpe" else "N/A"} | {yes("fp_residual")} | {yes("use_bolmo_gating")} | {yes("use_delete_gate")} | {c.get("pfnet_hidden",0)} | {yes("mlgru_per_channel_decay")} |')
    lines += ['', 'These experiments change parameter counts, representations, training signal, budgets, saved steps, and sometimes multiple architectural settings. These differences prevent isolating the benefit of an individual component from these results alone. `chinchilla_tokens_per_param` is a configured budget; it does not prove that a run completed that budget. Actual saved steps are reported separately.', '']
    return lines


def quickstart(repo, revision, compact=False):
    filename = 'run_04/checkpoint_run_04_final.pt' if compact else 'run_04/model.pt'
    loader = ['from training.ablation import build_ablation_model, load_ablation_checkpoint', f'state, config = load_ablation_checkpoint(path)'] if compact else ['from training.ablation import build_ablation_model', 'payload = torch.load(path, map_location="cpu", weights_only=False)', 'config, state = payload["config"], payload["state_dict"]']
    return ['## Load one checkpoint', '', 'Clone the [source repository](https://github.com/igna-s/AI-Training-from-Scratch) and run this from its root. The source repository is currently private and requires authorized GitHub access. These are custom PyTorch checkpoints; `AutoModelForCausalLM.from_pretrained()` does not load them. The examples use a trusted checkpoint source and strict state loading.', '', '```python', 'import os', 'os.environ["HF_HOME"] = "D:/cpu1/hf"', 'os.environ["HF_HUB_DISABLE_XET"] = "1"', 'import torch', 'from huggingface_hub import hf_hub_download', 'torch.set_num_threads(1)', f'path = hf_hub_download("{repo}", "{filename}",', f'    revision="{revision}", local_dir="D:/cpu1/active", cache_dir="D:/cpu1/hf/hub")', *loader, 'config["use_gradient_checkpointing"] = False', 'model = build_ablation_model(config)', 'model.load_state_dict(state, strict=True)', 'model.eval()', 'x = torch.tensor(list(b"The history of AI.  ")).reshape(1, -1, 4)', 'with torch.inference_mode():', '    logits, states, extra = model(x, training_phase=1, target_elimination=0.0)', 'print([tuple(head.shape) for head in logits])', '```', '', 'This example supplies zero target bytes to the local decoder to demonstrate output shapes. For likelihood scoring with LocalByteDecoder, supply the ground-truth `byte_targets`; the decoder shifts them internally. Byte generation requires sampling within each patch and retaining the appropriate recurrent state or Transformer context. See the GitHub evaluation implementation and the recorded continuation examples.', '']


def related_resources(repo):
    other = ARCHIVE if repo == FINAL else FINAL
    description = 'packed and intermediate checkpoints' if other == ARCHIVE else 'FP32 model exports'
    return ['', '## Related resources', '',
            '- [GitHub: igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch): extraction pipeline, training and evaluation code.',
            f'- [Hugging Face: {other}](https://huggingface.co/{other}): {description}.',
            '- [Hugging Face: Cukinator/cpu1-ablation-dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset): byte and BPE training data.', '']


def build_cards(inventory, corpus):
    names = sorted(f['rfilename'].split('/')[0] for f in inventory[FINAL]['siblings'] if f['rfilename'].endswith('/model.pt'))
    records = [json.loads((OUT/f'{r}.json').read_text(encoding='utf-8')) for r in names]
    expected = {f['rfilename']: f for f in inventory[FINAL]['siblings'] if f['rfilename'].endswith('/model.pt')}
    corpus_hash = sha(OUT/'corpus.json')
    for r in records:
        if not r.get('evaluation_complete') or r['revision'] != inventory[FINAL]['sha']:
            raise ValueError(f'Incomplete evaluation for {r["run"]}')
        file = expected.get(r.get('file'))
        if (not file or r['file'] != f'{r["run"]}/model.pt' or
                r['sha256'] != file['lfs']['sha256'] or r['file_bytes'] != file['size'] or
                r['corpus_sha256'] != corpus_hash):
            raise ValueError(f'Evaluation metadata does not match checkpoint/corpus for {r["run"]}')
    if len({r['suite_sha256'] for r in records}) != 1 or len({r['corpus_sha256'] for r in records}) != 1:
        raise ValueError('Mixed evaluation protocols')
    header = ['---', 'license: apache-2.0', 'language:', '- en', 'tags:', '- cpu-1', '- ablation', '- ternary', '- research', '---', '']
    shared = protocol(inventory, corpus, records) + tables(records) + architecture(records)
    final_files = [f for f in inventory[FINAL]['siblings'] if f['rfilename'].endswith('/model.pt')]
    final_card = header + ['# CPU-1 ablations: FP32 checkpoint exports', '',
        f'{len(final_files)} custom PyTorch model exports, {sum(f["size"] for f in final_files)/1e9:.3f} GB of checkpoint files (decimal units). These checkpoints are intended for research on the CPU-1 ablation configurations. It includes Transformer and MLGRU variants with byte patches or truncated BPE vocabularies, and floating-point or simulated ternary projections.', '',
        f'Source: [igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch). Training archive: [{ARCHIVE}](https://huggingface.co/{ARCHIVE}). Training dataset companion: [Cukinator/cpu1-ablation-dataset](https://huggingface.co/datasets/Cukinator/cpu1-ablation-dataset).', '',
        *evaluation_summary(records),
        '## Payload format', '', '`<run>/model.pt` contains `format="fp32"`, `state_dict`, `config`, `step`, `run_name`, `metrics`, and source metadata. The exported parameters occupy four bytes each. “1.58-bit” describes the ternary value alphabet; these files are not stored as 1.58-bit tensors and this PyTorch path does not use packed-weight inference.', '',
        'Historical `metrics` may be empty or come from different training validation protocols. The evaluation tables below use the shared test sample; stored training metrics remain separate.', '',
        *quickstart(FINAL, inventory[FINAL]['sha']), *shared,
        '## License', '', 'The model repository declares Apache-2.0. Upstream datasets, corpus excerpts and tokenizer assets retain their own licenses.', '']
    archived = [f for f in inventory[ARCHIVE]['siblings'] if f['rfilename'].endswith('.pt')]
    finals = [f for f in archived if f['rfilename'].endswith('_final.pt')]
    archive_card = header + ['# CPU-1 ablations: training checkpoint archive', '',
        f'{len(archived):,} checkpoint files across {len({f["rfilename"].split("/")[0] for f in archived})} run folders, totaling **{sum(f["size"] for f in archived)/1e9:.3f} GB** ({sum(f["size"] for f in archived)/2**30:.3f} GiB). The inventory includes **{len(finals)} `_final.pt` files** and **{len(archived)-len(finals)} step files**. Counts and sizes are derived from the pinned Hub inventory, not estimates from nominal architecture sizes.', '',
        f'For evaluated FP32 exports, see [{FINAL}](https://huggingface.co/{FINAL}). Source code: [igna-s/AI-Training-from-Scratch](https://github.com/igna-s/AI-Training-from-Scratch).', '',
        *evaluation_summary(records),
        '## What is stored', '',
        '- `checkpoint_<run>_final.pt`: inference-oriented final checkpoints. The source supports `compact_2bit` for ternary exports and floating-point storage for baselines; inspect the `format` field in the payload when loading a file.',
        '- `checkpoint_<run>_step<N>.pt`: intermediate training state; includes saved configuration and model state, and may include optimizer state. `_phase2_step<N>.pt` names distinguish phase-2 files where present.',
        '- The 2-bit codec packs a ternary alphabet into two bits per stored value plus scales and floating-point tensors. It is not a guarantee of 1.58 effective bits per total model parameter. Resume step files and packed finals have different purposes.',
        '- A folder without `_final.pt` is an intermediate-checkpoint archive. The highest filename step establishes the highest stored step; it does not establish convergence, completed training budget, or an early-stop reason.', '',
        '## Archive inventory', '', *manifest_rows(inventory), '',
        'Sizes use decimal MB and GB. Step counts include phase-2 step files. Highest step is parsed from filenames and does not merge training phases into a continuous training counter.', '',
        *quickstart(ARCHIVE, inventory[ARCHIVE]['sha'], compact=True),
        '## Training configurations', '',
        'Recorded configurations, saved steps and checkpoint hashes are indexed in `experiments/published/manifest.json` in the GitHub source. Use `--config` to select the intended experiment. Configuration compatibility is required when loading or resuming a checkpoint.', '',
        '## Resume training', '', 'Use an intermediate checkpoint, not an inference-only packed final. The current source CLI names the argument `--resume_ckpt`:', '',
        '```powershell', 'python -m training.ablation --run run_04 --config experiments/published/configs/run_04.json --resume_ckpt D:/cpu1/active/run_04/checkpoint_run_04_step640.pt --skip-install', '```', '',
        'Download the exact step file from its run folder first. Resume requires matching source/configuration, dependencies, and suitable training hardware. End-to-end training resume has not been verified for this release.', '',
        '## Scope of the measured results', '',
        f'The results below were measured on the {len(records)} published exports in [{FINAL}](https://huggingface.co/{FINAL}), **not by evaluating every intermediate archive checkpoint**. Matching run names do not prove tensor identity between a packed final, an intermediate step, and an export. No packed/FP32 numerical-parity claim is made by this card.', '',
        *shared, '## License', '', 'The model repository declares Apache-2.0. Dataset, corpus and tokenizer licenses are separate.', '']
    directory = ROOT / 'docs' / 'model_cards'
    for stem, card in [('cpu1-ablations-final', final_card), ('cpu1-ablation-checkpoints', archive_card)]:
        content = '\n'.join(card)
        content = content.replace('](https://github.com/igna-s/AI-Training-from-Scratch)', f']({SOURCE_TREE})')
        repo = FINAL if stem == 'cpu1-ablations-final' else ARCHIVE
        content += '\n'.join(related_resources(repo))
        if 'audit' in content.lower():
            raise ValueError('Model cards must use final documentation wording')
        (directory/(stem+'.md')).write_text(content, encoding='utf-8')
    save_environment(records)
    (OUT/'CORPUS_LICENSE.txt').write_text('The text excerpts in corpus.json originate from WikiText (Salesforce/wikitext), based on English Wikipedia.\nDataset authors and contributors: see https://huggingface.co/datasets/Salesforce/wikitext .\nUpstream dataset metadata declares CC-BY-SA-3.0 and GFDL.\nCC-BY-SA-3.0: https://creativecommons.org/licenses/by-sa/3.0/\nGFDL: https://www.gnu.org/licenses/fdl-1.3.html\nChanges: selection of 32 test paragraphs and truncation to 260 UTF-8 bytes.\nThe corpus remains under its upstream terms; the model repository Apache-2.0 metadata does not relicense these excerpts.\n',encoding='utf-8')
    dataset_card()
    print('Generated both model cards and the dataset template; source README preserved.', flush=True)


def dataset_card():
    """Preserve the reviewed canonical dataset card; reject inventory drift."""
    path = OUT/'dataset_inventory.json'
    if not path.exists():
        return
    inventory = json.loads(path.read_text(encoding='utf-8'))
    card = (ROOT/'docs/model_cards/cpu1-ablation-dataset.md').read_text(encoding='utf-8')
    if inventory['sha'] not in card:
        raise ValueError('Dataset card revision differs from the restored evidence')
    for subset in ('byte_marginalized', 'bpe_tokenized'):
        count = sum(f['rfilename'].startswith(subset+'/') and
                    f['rfilename'].endswith('.parquet') for f in inventory['siblings'])
        if f'| `{subset}` | {count} |' not in card:
            raise ValueError('Review the dataset card before publishing changed shard counts')


def save_environment(records):
    existing = OUT/'environment.json'
    if existing.exists():
        metadata = json.loads(existing.read_text(encoding='utf-8'))
        if metadata.get('audit_script_sha256') == records[0]['suite_sha256']:
            return  # Rendering historical evidence must preserve its provenance.
    from evaluation.audit import save
    from importlib.metadata import version
    packages = {name: version(name) for name in ('torch','transformers','tokenizers','huggingface_hub','datasets','numpy','pyarrow','psutil')}
    tokenizer_revision = next((r['tokenizer_revision'] for r in records if r.get('tokenizer_revision')), None)
    save(OUT/'environment.json', {'python': sys.version, 'packages': packages, 'hardware': records[0].get('processor', 'CPU'), 'platform': records[0]['platform'], 'threads': records[0]['threads'], 'source_commit': records[0].get('source_commit'), 'source_changes': records[0].get('source_changes'), 'audit_script_sha256': records[0]['suite_sha256'], 'qwen_tokenizer_resolved_revision': tokenizer_revision})


def upload_cards(inventory):
    from huggingface_hub import HfApi
    from evaluation.audit import save
    import os
    token = os.environ.get('HF_TOKEN')
    if not token:
        from getpass import getpass
        token = getpass('Hugging Face token (hidden input): ').strip()
    api = HfApi(token=token)
    identity = api.whoami()
    if identity['name'].lower() != 'cukinator':
        raise ValueError('Token does not belong to the authorized repository owner')
    commits = {}
    for repo, stem in [(FINAL,'cpu1-ablations-final'), (ARCHIVE,'cpu1-ablation-checkpoints')]:
        current = api.model_info(repo, files_metadata=True)
        parent = current.sha
        # A README-only publication changes the tip without changing audited weights.
        # Pin the new commit to that tip, but require the exact audited model inventory.
        expected = {f['rfilename']: (f['size'], f['lfs']['sha256'])
                    for f in inventory[repo]['siblings'] if f['rfilename'].endswith('.pt')}
        actual = {f.rfilename: (f.size, f.lfs.sha256 if f.lfs else None)
                  for f in current.siblings if f.rfilename.endswith('.pt')}
        if actual != expected:
            raise ValueError(f'{repo} model files changed since the audit; rerun evaluation before publishing')
        card = ROOT/'docs/model_cards'/f'{stem}.md'
        operations = card_operations(card, current.siblings, 'model')
        commit = api.create_commit(repo_id=repo, repo_type='model', parent_commit=parent,
            commit_message='Publish final model documentation with measured results and limitations',
            operations=operations)
        commits[repo] = {'oid': commit.oid, 'url': commit.commit_url}
        # Verify the actual published README bytes against the local card.
        from huggingface_hub import hf_hub_download
        readme = hf_hub_download(repo, 'README.md', revision=commit.oid, cache_dir=os.environ['HF_HUB_CACHE'], token=token)
        if Path(readme).read_bytes() != card.read_bytes():
            raise RuntimeError('Published card verification failed')
        print(f'Published and verified {repo}: {commit.oid}', flush=True)
        save(OUT/'publication.json', commits)
