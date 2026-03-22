"""
Unified training loop for the CPU-1 ablation study.

One script runs any of the 11 ablation runs based on config.
All runs share the same dataset (Cukinator/cpu1-ablation-dataset).

Usage:
    # Single GPU
    python ablation_train.py --run run_10

    # Kaggle 2×T4
    torchrun --nproc_per_node=2 ablation_train.py --run run_10

    # Dry run (no GPU, random data)
    python ablation_train.py --run run_01 --dry_run --total_steps 10
"""

import os
import math
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch import Tensor

from ablation_config import BASE_CONFIG, RUN_CONFIGS, get_run_config
from ablation_models import build_ablation_model


# ============================================================
# Dataset loaders
# ============================================================

class DryRunDataset(Dataset):
    """Random data for testing the training loop without real data."""

    def __init__(self, config: dict, n_samples: int = 1000):
        self.config = config
        self.n_samples = n_samples
        self.seq_len = config['seq_len']
        self.tok = config['tokenization']
        self.sig = config['training_signal']

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.tok == 'byte':
            patches = torch.randint(0, 256, (self.seq_len, 4))
            targets = torch.randint(0, 256, (self.seq_len, 4))
            sample = {'patches': patches, 'targets': targets}
            if self.sig == 'distillation':
                sample['teacher_probs'] = torch.zeros(self.seq_len, 256)
                sample['teacher_mask'] = torch.zeros(self.seq_len, dtype=torch.bool)
        else:
            vocab = self.config.get('bpe_vocab_size', 152064)
            input_ids = torch.randint(0, vocab, (self.seq_len,))
            labels = torch.randint(0, vocab, (self.seq_len,))
            sample = {'input_ids': input_ids, 'labels': labels}
            if self.sig == 'distillation':
                sample['teacher_probs_bpe'] = torch.zeros(self.seq_len, 64)
                sample['teacher_mask_bpe'] = torch.zeros(
                    self.seq_len, dtype=torch.bool
                )
        return sample


def load_ablation_dataset(config: dict, split: str = 'train',
                          dry_run: bool = False):
    """Load the correct dataset subset from HuggingFace Hub."""
    if dry_run:
        return DryRunDataset(config)

    from datasets import load_dataset

    repo = config['dataset_repo']
    tok = config['tokenization']
    sig = config['training_signal']

    subset = 'bpe_tokenized' if tok == 'bpe' else 'byte_marginalized'
    ds = load_dataset(repo, subset, split=split)

    if sig == 'ce_only':
        teacher_cols = (
            ['teacher_probs_bpe', 'teacher_mask_bpe'] if tok == 'bpe'
            else ['teacher_probs', 'teacher_mask']
        )
        for col in teacher_cols:
            if col in ds.column_names:
                ds = ds.remove_columns([col])

    return ds


# ============================================================
# Loss computation
# ============================================================

def compute_loss(
    student_logits,
    targets,
    teacher_probs,
    teacher_mask,
    config: dict,
) -> Tensor:
    """
    Compute loss based on run configuration.

    For byte-level (4 heads):
        Head 0: KL(student, teacher) where teacher exists + CE elsewhere
        Heads 1-3: CE only
    For BPE (1 head):
        KL + CE where teacher exists, CE elsewhere
    """
    tok = config['tokenization']
    sig = config['training_signal']
    T = config['temperature']
    alpha_kl = config['alpha_kl']
    alpha_ce = config['alpha_ce']

    if tok == 'bpe':
        # Single head over BPE vocab
        logits = student_logits
        vocab_size = logits.shape[-1]

        ce = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction='none',
        ).reshape(targets.shape)

        if sig == 'distillation' and teacher_probs is not None and teacher_mask is not None:
            student_log_T = F.log_softmax(logits / T, dim=-1)
            teacher_T = (teacher_probs / T).softmax(dim=-1) if teacher_probs.shape[-1] == vocab_size else teacher_probs
            kl = F.kl_div(
                student_log_T, teacher_T,
                reduction='none', log_target=False,
            ).sum(dim=-1) * T ** 2

            mask = teacher_mask.float()
            loss = mask * (alpha_kl * kl + alpha_ce * ce) + (1 - mask) * ce
            return loss.mean()

        return ce.mean()

    else:
        # Byte-level: 4 heads
        # Import distillation_loss from the existing train.py
        from train import distillation_loss

        if sig == 'distillation' and teacher_probs is not None:
            loss, _ = distillation_loss(
                student_logits, targets,
                teacher_probs, teacher_mask,
                temperature=T,
                alpha_kl=alpha_kl,
                alpha_ce=alpha_ce,
            )
        else:
            no_mask = torch.zeros(
                targets.shape[0], targets.shape[1],
                dtype=torch.bool, device=targets.device,
            )
            loss, _ = distillation_loss(
                student_logits, targets,
                None, no_mask,
                temperature=T,
                alpha_kl=alpha_kl,
                alpha_ce=alpha_ce,
            )
        return loss


# ============================================================
# LR schedule
# ============================================================

def get_lr(step: int, config: dict) -> float:
    """Cosine annealing with linear warmup."""
    ws = config['warmup_steps']
    ts = config['total_steps']
    lr = config['lr']
    min_lr = config['min_lr']
    if step < ws:
        return lr * step / max(ws, 1)
    p = (step - ws) / max(ts - ws, 1)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * p))


# ============================================================
# CPU throughput measurement
# ============================================================

def measure_cpu_throughput(model: nn.Module, config: dict,
                           n_tokens: int = 100) -> float:
    """Measure tokens/sec on CPU for this model variant."""
    model_cpu = model.cpu()
    model_cpu.eval()
    tok = config['tokenization']

    if tok == 'byte':
        dummy = torch.zeros(1, 1, 4, dtype=torch.long)
    else:
        dummy = torch.zeros(1, 1, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model_cpu(dummy)

    # Benchmark
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_tokens):
            model_cpu(dummy)
    t1 = time.perf_counter()

    return n_tokens / (t1 - t0)


# ============================================================
# Evaluation
# ============================================================

def evaluate_run(model: nn.Module, val_ds, config: dict,
                 device) -> dict:
    """Evaluate on validation set. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    tok = config['tokenization']

    with torch.no_grad():
        for batch in val_loader:
            if n_batches >= 100:
                break

            if tok == 'byte':
                x = batch['patches'].to(device)
                targets = batch['targets'].to(device)
            else:
                x = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            if t_probs is not None:
                t_probs = t_probs.to(device)
            if t_mask is not None:
                t_mask = t_mask.to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                logits, _ = model(x)
                loss = compute_loss(logits, targets, t_probs, t_mask, config)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

    # CPU throughput
    raw_model = model.module if hasattr(model, 'module') else model
    tps = measure_cpu_throughput(raw_model, config)
    ram_mb = sum(p.numel() * p.element_size() for p in raw_model.parameters()) / 1e6

    metrics = {
        'run_name': config.get('name', 'unknown'),
        'perplexity': perplexity,
        'tokens_per_sec_cpu': tps,
        'ram_mb': ram_mb,
        'val_loss': avg_loss,
        'n_params_M': sum(p.numel() for p in raw_model.parameters()) / 1e6,
    }

    print(f"\n[Val] {config.get('name', '?')}")
    print(f"  Perplexity:     {perplexity:.2f}")
    print(f"  Tokens/sec CPU: {tps:.1f}")
    print(f"  RAM:            {ram_mb:.1f} MB")

    return metrics


# ============================================================
# Checkpoint save
# ============================================================

def save_checkpoint(model: nn.Module, optimizer, run_name: str,
                    metrics: dict, config: dict, step: int) -> str:
    """Save checkpoint locally and optionally to HuggingFace Hub."""
    path = f"checkpoint_{run_name}_step{step}.pt"
    raw_model = model.module if hasattr(model, 'module') else model

    torch.save({
        'model_state_dict': raw_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'step': step,
        'run_name': run_name,
        'config': config,
    }, path)

    # Upload to Hub if configured
    hub_repo = config.get('checkpoint_repo')
    if hub_repo:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=f"ablation/{path}",
                repo_id=hub_repo,
            )
            print(f"[Checkpoint] Uploaded: {path}")
        except Exception as e:
            print(f"[Checkpoint] Hub upload failed: {e}")

    return path


# ============================================================
# Main training loop
# ============================================================

def train_run(run_name: str, dry_run: bool = False,
              total_steps_override: int | None = None) -> dict:
    """Train one ablation run. Returns final metrics."""
    config = get_run_config(run_name)

    if total_steps_override is not None:
        config['total_steps'] = total_steps_override

    assert config['seed'] == BASE_CONFIG['seed'], "Seed mismatch"

    # DDP setup
    use_ddp = 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1
    if use_ddp:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        if torch.cuda.is_available() and not dry_run:
            torch.cuda.set_device(0)

    torch.manual_seed(config['seed'])
    device = torch.device('cpu') if dry_run else torch.device(f'cuda:{local_rank}')

    if local_rank == 0:
        print(f"\n{'=' * 60}")
        print(f"Run: {run_name}")
        print(f"Description: {config['description']}")
        print(f"Device: {device}")
        print(f"{'=' * 60}\n")

    # Build model
    model = build_ablation_model(config).to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # Dataset
    train_ds = load_ablation_dataset(config, split='train', dry_run=dry_run)
    val_ds = load_ablation_dataset(config, split='val' if not dry_run else 'train',
                                    dry_run=dry_run)

    train_sampler = (
        DistributedSampler(train_ds, shuffle=True, seed=config['seed'])
        if use_ddp else None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_per_gpu'],
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=0 if dry_run else 2,
        pin_memory=not dry_run,
    )

    # Optimizer with per-group weight decay
    bitlinear_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'weight' in name and any(
            k in name for k in ['forget_proj', 'cand_proj', 'output_proj',
                                 'gate_proj', 'up_proj', 'down_proj',
                                 'q_proj', 'k_proj', 'v_proj', 'o_proj']
        ):
            bitlinear_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': bitlinear_params,
         'weight_decay': config['weight_decay_bitlinear']},
        {'params': other_params,
         'weight_decay': 0.0},
    ], lr=config['lr'], betas=(0.9, 0.95))

    use_amp = not dry_run and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None

    # Training loop
    model.train()
    step = 0
    losses = []
    tok = config['tokenization']
    accum_steps = config['gradient_accumulation_steps']
    total_steps = config['total_steps']
    t_start = time.time()

    while step < total_steps:
        for batch in train_loader:
            if step >= total_steps:
                break

            # Unpack batch
            if tok == 'byte':
                x = batch['patches'].to(device)
                targets = batch['targets'].to(device)
            else:
                x = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            if t_probs is not None:
                t_probs = t_probs.to(device)
            if t_mask is not None:
                t_mask = t_mask.to(device)

            # Forward
            ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                   if use_amp else torch.no_grad().__class__())
            if use_amp:
                ctx = torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                with ctx:
                    logits, _ = model(x)
                    loss = compute_loss(logits, targets, t_probs, t_mask, config)
                    loss = loss / accum_steps
                scaler.scale(loss).backward()
            else:
                logits, _ = model(x)
                loss = compute_loss(logits, targets, t_probs, t_mask, config)
                loss = loss / accum_steps
                loss.backward()

            # Optimizer step at accumulation boundary
            if (step + 1) % accum_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['grad_clip']
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config['grad_clip']
                    )
                    optimizer.step()
                optimizer.zero_grad()

                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(step, config)

            losses.append(loss.item() * accum_steps)

            if step % 500 == 0 and local_rank == 0:
                avg_loss = sum(losses[-100:]) / min(100, len(losses))
                elapsed = time.time() - t_start
                print(f"Step {step:5d}/{total_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {get_lr(step, config):.2e} | "
                      f"Time: {elapsed:.0f}s")

            # Save periodic checkpoint
            if (step > 0 and step % config['save_every_steps'] == 0
                    and local_rank == 0 and not dry_run):
                save_checkpoint(model, optimizer, run_name, {}, config, step)

            step += 1

    # Final evaluation
    if local_rank == 0:
        val_metrics = evaluate_run(model, val_ds, config, device)

        if not dry_run:
            save_checkpoint(model, optimizer, run_name, val_metrics,
                            config, step)

        # Save metrics JSON
        metrics_path = f"metrics_{run_name}.json"
        with open(metrics_path, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        print(f"\n[Done] Metrics saved to {metrics_path}")
    else:
        val_metrics = {}

    if use_ddp:
        dist.destroy_process_group()

    return val_metrics


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train one ablation run for CPU-1"
    )
    parser.add_argument('--run', required=True,
                        choices=list(RUN_CONFIGS.keys()),
                        help="Which run to train")
    parser.add_argument('--dry_run', action='store_true',
                        help="Use random data on CPU (for testing)")
    parser.add_argument('--total_steps', type=int, default=None,
                        help="Override total training steps")
    args = parser.parse_args()

    metrics = train_run(args.run, dry_run=args.dry_run,
                        total_steps_override=args.total_steps)
    print(f"\nFinished: {args.run}")
    print(json.dumps(metrics, indent=2))
