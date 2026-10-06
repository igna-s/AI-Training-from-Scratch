"""Verify the two published loading examples, sequentially, in D: storage."""
import gc
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str((value.dtype, tuple(value.shape))).encode())
        result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    import json
    from evaluation.audit import FINAL, ARCHIVE, OUT, save, sha
    storage = Path('D:/cpu1-readme-audit')
    assert not list((storage/'active').glob('*/model.pt')), 'Finish the model audit first'
    os.environ['HF_HOME'] = str(storage/'hf_home')
    os.environ['HF_HUB_CACHE'] = str(storage/'hf_hub_cache')
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
    import torch
    from huggingface_hub import hf_hub_download
    from training.ablation import build_ablation_model, load_ablation_checkpoint
    torch.set_num_threads(1)
    inventory = json.loads((OUT/'inventory.json').read_text(encoding='utf-8'))
    records = []
    for repo, filename in [(ARCHIVE,'run_04/checkpoint_run_04_final.pt'),(FINAL,'run_04/model.pt')]:
        path = Path(hf_hub_download(repo,filename,revision=inventory[repo]['sha'],local_dir=str(storage/'smoke'),cache_dir=str(storage/'hf_hub_cache')))
        raw = torch.load(path,map_location='cpu',weights_only=False)
        fmt = raw.get('format')
        if repo == FINAL:
            state, config = raw['state_dict'], raw['config']
        else:
            state, config = load_ablation_checkpoint(str(path))
        fingerprint = digest(state)
        config['use_gradient_checkpointing'] = False
        model = build_ablation_model(config)
        model.load_state_dict(state,strict=True)
        model.eval()
        x = torch.tensor(list(b'The history of AI.  ')).reshape(1,-1,4)
        with torch.inference_mode():
            logits, states, extra = model(x,training_phase=1,target_elimination=0.)
        assert all(torch.isfinite(head).all() for head in logits)
        shapes = [list(head.shape) for head in logits]
        assert shapes == [[1,5,256]]*4
        records.append({'repo':repo,'revision':inventory[repo]['sha'],'file':filename,'format':fmt,'sha256':sha(path),'state_sha256':fingerprint,'strict_load':True,'finite_logits':True,'shapes':shapes})
        print(f'Example verified: {repo} ({fmt})',flush=True)
        del raw, state, model, logits, states, extra
        gc.collect()
        path.unlink()
    save(OUT/'loading_examples.json',{'examples':records,'run_04_state_dicts_identical':records[0]['state_sha256']==records[1]['state_sha256'],'scope':'only the run_04 pair; does not establish parity across other models or intermediate steps'})


if __name__ == '__main__':
    main()
