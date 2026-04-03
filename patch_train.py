import sys

with open('train.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. args.seq_len default 2048 -> 5000
text = text.replace(
    "p.add_argument('--seq-len', type=int, default=2048,",
    "p.add_argument('--seq-len', type=int, default=5000,"
)

# 2. _get_dry_run_item return dict
old_dry_run_dict = """        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
        }"""

new_dry_run_dict = """        teacher_hidden = torch.zeros(S, 128, dtype=torch.float32)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
            'teacher_hidden': teacher_hidden,
        }"""
if old_dry_run_dict in text:
    text = text.replace(old_dry_run_dict, new_dry_run_dict, 1)

# 3. _get_hf_item 
old_hf_item = """        if 'teacher_mask' in item and item['teacher_mask'] is not None:
            teacher_mask = _to_tensor(item['teacher_mask'], torch.bool)
        else:
            teacher_mask = torch.zeros(seq_len, dtype=torch.bool)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
        }"""

new_hf_item = """        if 'teacher_mask' in item and item['teacher_mask'] is not None:
            teacher_mask = _to_tensor(item['teacher_mask'], torch.bool)
        else:
            teacher_mask = torch.zeros(seq_len, dtype=torch.bool)

        if 'teacher_hidden' in item and item['teacher_hidden'] is not None:
            t_dim = int(item.get('teacher_dim', 128))
            teacher_hidden = _to_tensor(item['teacher_hidden'], torch.float32)
            teacher_hidden = teacher_hidden.view(seq_len, t_dim)
        else:
            teacher_hidden = torch.zeros(seq_len, 128, dtype=torch.float32)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
            'teacher_hidden': teacher_hidden,
        }"""
if old_hf_item in text:
    text = text.replace(old_hf_item, new_hf_item)

# 4. _get_hf_item docstring
text = text.replace(
    "        teacher_mask:  [seq_len] bool — True donde hay señal teacher",
    "        teacher_mask:  [seq_len] bool — True donde hay señal teacher\n        teacher_hidden:[seq_len, dim] float32 — estados ocultos del teacher"
)

# 5. Add teacher_hidden support to train parameter mapping
# Locate:
#             patches = batch['patches'].to(device, non_blocking=True)
#             targets = batch['targets'].to(device, non_blocking=True)
#             teacher_probs = batch['teacher_probs'].to(device, non_blocking=True)
#             teacher_mask = batch['teacher_mask'].to(device, non_blocking=True)
# And add teacher_hidden below it
batch_loading = """            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            teacher_probs = batch['teacher_probs'].to(device, non_blocking=True)
            teacher_mask = batch['teacher_mask'].to(device, non_blocking=True)"""
            
batch_loading_new = """            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            teacher_probs = batch['teacher_probs'].to(device, non_blocking=True)
            teacher_mask = batch['teacher_mask'].to(device, non_blocking=True)
            teacher_hidden = batch.get('teacher_hidden')
            if teacher_hidden is not None:
                teacher_hidden = teacher_hidden.to(device, non_blocking=True)"""

text = text.replace(batch_loading, batch_loading_new)

# Locate:
#             with amp_ctx:
#                 logits, _ = model(patches)
#                 loss, metrics = distillation_loss(
# Replace with passing teacher_hidden and teacher_mask (some variant models use them)
model_forward = """            with amp_ctx:
                logits, _ = model(patches)"""
model_forward_new = """            with amp_ctx:
                # If model is AblationModel, it might accept teacher_hidden
                import inspect
                if 'teacher_hidden' in inspect.signature(model.forward).parameters if not hasattr(model, 'module') else inspect.signature(model.module.forward).parameters:
                    logits, _, _ = model(patches, teacher_hidden=teacher_hidden, teacher_mask=teacher_mask)
                else:
                    logits, _ = model(patches)"""
text = text.replace(model_forward, model_forward_new)

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('train.py patched successfully.')
