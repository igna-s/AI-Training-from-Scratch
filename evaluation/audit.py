"""Reproducible Hub audit. Download and evaluate exactly one model at a time.

python -m evaluation.audit --storage D:/cpu1-readme-audit --threads 1
python -m evaluation.audit --storage D:/cpu1-readme-audit --cards
python -m evaluation.audit --storage D:/cpu1-readme-audit --upload
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs' / 'audit'
FINAL = 'Cukinator/cpu1-ablations-final'
ARCHIVE = 'Cukinator/cpu1-ablation-checkpoints'
FINAL_REV = '65951289352c1a036b12ba6d888af037e7c90b13'
ARCHIVE_REV = 'fb402d025e843ed00fb990966bf2430c32eb958c'
DATA = 'Salesforce/wikitext'
DATA_REV = 'b08601e04326c79dfdd32d625aee71d232d685c3'
DATA_FILE = 'wikitext-2-raw-v1/test-00000-of-00001.parquet'
PROMPTS = ['The history of artificial intelligence', 'Once upon a time', 'The purpose of science is']


def save(path, value):
    def finite_json(obj):
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        if isinstance(obj, dict):
            return {k: finite_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [finite_json(v) for v in obj]
        return obj
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(finite_json(value), indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def restore_evidence(storage):
    """Copy committed GitHub evidence locally without downloading models."""
    import shutil
    from evaluation.publication import AUDIT_PATH
    folder = ROOT/AUDIT_PATH
    if not folder.is_dir():
        raise FileNotFoundError('The checkout is missing committed audit evidence')
    OUT.mkdir(parents=True, exist_ok=True)
    for source in sorted(folder.glob('*')):
        if source.suffix == '.json' or source.name == 'CORPUS_LICENSE.txt':
            shutil.copyfile(source, OUT/source.name)
    print(f'Restored GitHub evidence to {OUT}; no model weights downloaded.', flush=True)


def prepare(storage):
    OUT.mkdir(parents=True, exist_ok=True)
    invpath = OUT / 'inventory.json'
    if not invpath.exists():
        inventory = {r: json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{r}/revision/{revision}?blobs=true')) for r, revision in ((FINAL, FINAL_REV), (ARCHIVE, ARCHIVE_REV))}
        save(invpath, inventory)
    corpuspath = OUT / 'corpus.json'
    if not corpuspath.exists():
        import pyarrow.parquet as pq
        dest = storage / 'data' / 'test.parquet'
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f'https://huggingface.co/datasets/{DATA}/resolve/{DATA_REV}/{DATA_FILE}', dest)
        texts = pq.read_table(dest, columns=['text']).column('text').to_pylist()
        # First 32 non-heading paragraphs with at least 260 UTF-8 bytes.
        samples = [t.encode('utf-8')[:260].decode('utf-8', errors='ignore') for t in texts if len(t.encode('utf-8')) >= 260 and not t.strip().startswith('=')][:32]
        assert len(samples) == 32
        save(corpuspath, {'dataset': DATA, 'revision': DATA_REV, 'file': DATA_FILE, 'parquet_sha256': sha(dest), 'selection': 'first 32 non-heading paragraphs >=260 UTF-8 bytes; truncate to 260 bytes, discard incomplete trailing UTF-8', 'texts': samples})
    return json.loads(invpath.read_text(encoding='utf-8')), json.loads(corpuspath.read_text(encoding='utf-8'))


def loss_eval(model, config, corpus, tokenizer):
    import torch
    import torch.nn.functional as F
    byte = config['tokenization'] in ('byte', 'byte_dynamic')
    p = config.get('patch_size', 4)
    total, count, mapped, all_ids = 0., 0, 0, 0
    with torch.inference_mode():
        for text in corpus['texts']:
            if byte:
                raw = list(text.encode('utf-8'))
                data = torch.tensor(raw[:len(raw)//p*p]).reshape(1, -1, p)
                x, y = data[:, :-1], data[:, 1:]
                logits = model(x, byte_targets=y, training_phase=1, target_elimination=0.)[0]
                if isinstance(logits, (tuple, list)):
                    logits = torch.stack(logits, dim=2)
            else:
                ids = torch.tensor(tokenizer.encode(text, add_special_tokens=False)).reshape(1, -1)
                cap = config['bpe_vocab_size']
                mapped += int((ids >= cap).sum())
                all_ids += ids.numel()
                ids = torch.where(ids < cap, ids, 0)
                x, y = ids[:, :-1], ids[:, 1:]
                logits = model(x, training_phase=1, target_elimination=0.)[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction='sum')
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            total += loss.item()
            count += y.numel()
    avg = total / count
    return {'mean_nll': avg, 'targets': count, 'unit': 'byte' if byte else 'mapped_qwen_token', 'perplexity': math.exp(avg), 'bits_per_byte': avg/math.log(2) if byte else None, 'mapped_ids_fraction': mapped/all_ids if all_ids else None, 'source': 'shared_wikitext_test_sample', 'teacher_forcing': True, 'delete_gate': 'disabled (phase 1, elimination 0)'}


def generation(model, config, prompt, length=128):
    """Keep Transformer context; recurrent models carry state exactly once."""
    import torch
    from training.ablation import _global_pass, _sample_logits
    p = config.get('patch_size', 4)
    recurrent = config['token_mixer'] == 'mlgru'
    raw = prompt.encode('utf-8')
    raw += b'\x00' * ((-len(raw)) % p)
    context = torch.tensor(list(raw)).reshape(1, -1, p)
    states = [None] * len(model.blocks)
    output = bytearray()
    with torch.inference_mode():
        h, states = _global_pass(model, context, states, 0.)
        while len(output) < length:
            new = torch.zeros(1, 1, p, dtype=torch.long)
            for k in range(p):
                logits = model.local_decoder(h[:, -1:], new)[k][0, 0] if model.use_local_decoder else model.heads[k](h[:, -1:])[0, 0]
                value = _sample_logits(logits, 1., 40, .9)
                new[0, 0, k] = value
                output.append(value)
            if len(output) >= length:
                break
            if recurrent:
                h, states = _global_pass(model, new, states, 0.)
            else:
                context = torch.cat([context, new], dim=1)
                window = config.get('window_size', 512)
                h, _ = _global_pass(model, context[:, -window:], [None]*len(model.blocks), 0.)
    return bytes(output[:length])


def benchmark(engine):
    """Isolated single-input forward, not generated bytes/s or context scaling."""
    import torch
    dummy = torch.zeros((1, 1, engine.info.patch_size) if engine.info.is_byte_level else (1, 1), dtype=torch.long)
    times = []
    with torch.inference_mode():
        for _ in range(10):
            _, engine.hidden_states = engine._ablation_step(engine._model, dummy)
        engine.reset_state()
        for _ in range(100):
            t = time.perf_counter()
            _, engine.hidden_states = engine._ablation_step(engine._model, dummy)
            times.append(time.perf_counter()-t)
    engine.reset_state()
    values = sorted(times)
    return {'forward_steps_per_second': len(times)/sum(times), 'p50_ms': statistics.median(times)*1000, 'p95_ms': values[94]*1000, 'jitter_cv': statistics.stdev(times)/statistics.mean(times), 'warmup_steps': 10, 'measured_steps': 100, 'context_steps': 1, 'backend': 'PyTorch FP32; quantization simulated for ternary modules; no C++ kernels', 'meaning': 'one input patch/token forward, zero local decoder targets; excludes autoregressive local sampling; Transformer has no persistent context'}


def causality(engine):
    import torch
    from evaluation.correctness import _get_logits_list, _make_dummy_input
    torch.manual_seed(101)
    x = _make_dummy_input(engine.info, 1, 32)
    changed = x.clone()
    changed[:, 16:] = (changed[:, 16:] + 71) % engine.info.vocab_size
    with torch.inference_mode():
        a = _get_logits_list(engine._model(x)[0], engine.info)
        b = _get_logits_list(engine._model(changed)[0], engine.info)
    diff = max((u[:, :16]-v[:, :16]).abs().max().item() for u,v in zip(a,b))
    return {'status': ('PASS' if diff < 1e-5 else 'FAIL') if math.isfinite(diff) else 'ERROR', 'max_prefix_logit_diff': diff, 'tolerance': 1e-5, 'protocol': '32 input steps; change steps 16:32; compare all heads at steps 0:16 with identical zero local targets'}


def evaluate(storage, inventory, corpus, args):
    import torch
    import psutil
    from huggingface_hub import hf_hub_download
    from evaluation.checkpoints import load_run
    from evaluation.correctness import run_correctness
    from transformers import AutoTokenizer
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    revision = inventory[FINAL]['sha']
    runs = [f['rfilename'].split('/')[0] for f in inventory[FINAL]['siblings'] if f['rfilename'].endswith('/model.pt')]
    if args.runs:
        runs = [r for r in runs if r in args.runs]
    import subprocess
    source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    source_commit = source.stdout.strip() if source.returncode == 0 else None
    tokenizer = None
    for index, run in enumerate(runs, 1):
        engine = model = raw = None
        result_path = OUT / f'{run}.json'
        if result_path.exists() and not args.force and json.loads(result_path.read_text(encoding='utf-8')).get('suite_sha256') == sha(Path(__file__)):
            print(f'[{index}/{len(runs)}] existing {run}', flush=True)
            continue
        # Only this task-owned file may be removed. No cache of model blobs.
        model_path = storage / 'active' / run / 'model.pt'
        for stale in (storage / 'active').glob('*/model.pt'):
            target = stale.resolve()
            if not target.is_relative_to((storage / 'active').resolve()):
                raise ValueError('unsafe cleanup path')
            target.unlink()
        print(f'[{index}/{len(runs)}] downloading {run} to {model_path}', flush=True)
        started = time.perf_counter()
        hf_hub_download(FINAL, f'{run}/model.pt', revision=revision, local_dir=str(storage/'active'), cache_dir=str(storage/'hf'/'hub'))
        record = {'run': run, 'revision': revision, 'file': f'{run}/model.pt', 'sha256': sha(model_path), 'file_bytes': model_path.stat().st_size, 'date_utc': datetime.now(timezone.utc).isoformat(), 'threads': args.threads, 'torch': torch.__version__, 'platform': platform.platform(), 'processor': platform.processor(), 'corpus_sha256': sha(OUT/'corpus.json'), 'suite_sha256': sha(Path(__file__)), 'seed': 42, 'source_commit': source_commit}
        try:
            torch.manual_seed(42)
            engine, raw, config, _, _ = load_run(model_path, run)
            model = engine._model
            record.update({'config': config, 'step': raw.get('step'), 'stored_training_metrics': raw.get('metrics', {}), 'parameters': sum(p.numel() for p in model.parameters()), 'parameter_bytes': sum(p.numel()*p.element_size() for p in model.parameters()), 'format': raw.get('format'), 'strict_load': True})
            record['correctness'] = run_correctness(engine)
            record['correctness']['causality'] = causality(engine)
            if not engine.info.is_byte_level and tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B', cache_dir=str(storage/'hf'/'hub'), trust_remote_code=False)
                record['tokenizer_revision'] = tokenizer.init_kwargs.get('_commit_hash')
            record['loss'] = loss_eval(model, config, corpus, tokenizer)
            record['benchmark'] = benchmark(engine)
            if engine.info.is_byte_level:
                samples = []
                for i, prompt in enumerate(PROMPTS):
                    torch.manual_seed(42+i)
                    t = time.perf_counter()
                    generated = generation(model, config, prompt)
                    elapsed = time.perf_counter()-t
                    try:
                        generated.decode('utf-8', errors='strict')
                        valid = True
                    except UnicodeDecodeError:
                        valid = False
                    samples.append({'prompt': prompt, 'seed': 42+i, 'hex': generated.hex(), 'text_replacement_decode': generated.decode('utf-8', errors='replace'), 'bytes': len(generated), 'utf8_valid': valid, 'ascii_printable_fraction': sum(32<=b<127 or b in (9,10,13) for b in generated)/len(generated), 'repeated_8byte_fraction': sum(generated[i:i+8]==generated[i-8:i] for i in range(8,len(generated)-7))/max(1,len(generated)-15), 'seconds': elapsed, 'generated_bytes_per_second': len(generated)/elapsed})
                record['generation'] = samples
            else:
                record['generation'] = {'status': 'N/A', 'reason': 'truncated Qwen vocabulary; token mapping is lossy, no comparable byte generation test'}
            record['process_rss_bytes'] = psutil.Process().memory_info().rss
        except Exception as exc:
            record['error'] = f'{type(exc).__name__}: {exc}'
            print(record['error'], flush=True)
        record['total_seconds'] = time.perf_counter()-started
        record['evaluation_complete'] = True
        save(result_path, record)
        print(f'{run}: loss={record.get("loss",{}).get("mean_nll")} elapsed={record["total_seconds"]:.1f}s', flush=True)
        model_path.unlink()
        engine = model = raw = None
        import gc
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--storage', type=Path, default=Path('D:/cpu1-readme-audit'))
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--runs', nargs='+')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--cards', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--restore-evidence', action='store_true', help='Copy the committed audit JSONs and corpus license to local outputs')
    args = parser.parse_args()
    if args.storage.resolve().drive.upper() != 'D:':
        raise ValueError('This audit requires storage on D:')
    args.storage.mkdir(parents=True, exist_ok=True)
    for name in ('HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TRANSFORMERS_CACHE','TEMP','TMP'):
        dest = args.storage / ('temp' if name in ('TEMP','TMP') else name.lower())
        dest.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(dest)
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    if args.restore_evidence:
        restore_evidence(args.storage)
        return
    inventory, corpus = prepare(args.storage)
    if args.cards or args.upload:
        from evaluation.cards import build_cards, upload_cards
        build_cards(inventory, corpus)
        if args.upload:
            upload_cards(inventory)
    else:
        evaluate(args.storage, inventory, corpus, args)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
