"""
Unified training loop for the CPU-1 ablation study.

One script runs any of the 15 ablation runs based on config.
All runs share the same dataset (Cukinator/cpu1-ablation-dataset).

Usage:
    # Single T4 (Kaggle)
    python ablation_train.py --run run_10

    # Dry run (no GPU, random data)
    python ablation_train.py --run run_01 --dry_run --total_steps 10
"""

import os
import math
import json
import time
import argparse
import contextlib
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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
        if self.tok in ('byte', 'byte_dynamic'):
            patches = torch.randint(0, 256, (self.seq_len, 4))
            targets = torch.randint(0, 256, (self.seq_len, 4))
            sample = {'patches': patches, 'targets': targets}
            if self.sig in ('distillation', 'distillation_then_ce'):
                sample['teacher_probs'] = torch.zeros(self.seq_len, 256)
                sample['teacher_mask'] = torch.zeros(self.seq_len, dtype=torch.bool)
            if self.config.get('use_embedding_alignment'):
                teacher_dim = self.config.get('teacher_dim', 5120)
                sample['teacher_hidden'] = torch.randn(self.seq_len, teacher_dim)
        else:
            vocab = self.config.get('bpe_vocab_size', 152064)
            input_ids = torch.randint(0, vocab, (self.seq_len,))
            labels = torch.randint(0, vocab, (self.seq_len,))
            sample = {'input_ids': input_ids, 'labels': labels}
            if self.sig == 'distillation':
                sample['teacher_probs_bpe'] = torch.zeros(self.seq_len, 128)
                sample['teacher_ids_bpe'] = torch.zeros(
                    self.seq_len, 128, dtype=torch.long
                )
                sample['teacher_mask_bpe'] = torch.zeros(
                    self.seq_len, dtype=torch.bool
                )
        return sample


class ByteLevelDataset(Dataset):
    """Wraps a HuggingFace byte-level dataset.

    Each Parquet row stores a full sequence as flat arrays:
      patches/targets  → [seq_len * patch_size] uint8, reshaped to [seq_len, patch_size]
      teacher_probs    → [seq_len * 256] float32, reshaped to [seq_len, 256]
      teacher_mask     → [seq_len] bool
    """

    def __init__(self, hf_dataset, config: dict):
        self.ds = hf_dataset
        self.seq_len = config['seq_len']
        self.patch_size = config.get('patch_size', 4)
        self.has_teacher = config['training_signal'] in ('distillation', 'distillation_then_ce')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        S, P = self.seq_len, self.patch_size
        patches = torch.tensor(row['patches'], dtype=torch.long).reshape(S, P)
        targets = torch.tensor(row['targets'], dtype=torch.long).reshape(S, P)
        sample = {'patches': patches, 'targets': targets}
        if self.has_teacher and 'teacher_probs' in row:
            sample['teacher_probs'] = (
                torch.tensor(row['teacher_probs'], dtype=torch.float32)
                .reshape(S, 256)
            )
            sample['teacher_mask'] = torch.tensor(
                row['teacher_mask'], dtype=torch.bool
            )
        if 'teacher_hidden' in row and row['teacher_hidden'] is not None:
            teacher_dim = row.get('teacher_dim', 0)
            if teacher_dim > 0:
                sample['teacher_hidden'] = (
                    torch.tensor(row['teacher_hidden'], dtype=torch.float32)
                    .reshape(S, teacher_dim)
                )
        return sample


class BPEDataset(Dataset):
    """Wraps a HuggingFace BPE dataset.

    Each Parquet row stores a full sequence as flat arrays:
      input_ids/labels      → [seq_len] int32
      teacher_probs_bpe     → [seq_len * TOP_K] float32, reshaped to [seq_len, TOP_K]
      teacher_ids_bpe       → [seq_len * TOP_K] int32,   reshaped to [seq_len, TOP_K]
      teacher_mask_bpe      → [seq_len] bool
    """
    TOP_K = 128

    def __init__(self, hf_dataset, config: dict):
        self.ds = hf_dataset
        self.seq_len = config['seq_len']
        self.has_teacher = config['training_signal'] in ('distillation', 'distillation_then_ce')

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        S, K = self.seq_len, self.TOP_K
        sample = {
            'input_ids': torch.tensor(row['input_ids'], dtype=torch.long),
            'labels':    torch.tensor(row['labels'],    dtype=torch.long),
        }
        if self.has_teacher and 'teacher_probs_bpe' in row:
            sample['teacher_probs_bpe'] = (
                torch.tensor(row['teacher_probs_bpe'], dtype=torch.float32)
                .reshape(S, K)
            )
            sample['teacher_ids_bpe'] = (
                torch.tensor(row['teacher_ids_bpe'], dtype=torch.long)
                .reshape(S, K)
            )
            sample['teacher_mask_bpe'] = torch.tensor(
                row['teacher_mask_bpe'], dtype=torch.bool
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
    subset = 'bpe_tokenized' if tok == 'bpe' else 'byte_marginalized'
    ds = load_dataset(repo, subset, split=split)

    if tok in ('byte', 'byte_dynamic'):
        return ByteLevelDataset(ds, config)
    else:
        return BPEDataset(ds, config)


# ============================================================
# Loss computation
# ============================================================

def compute_loss(
    student_logits,
    targets,
    teacher_probs,
    teacher_mask,
    config: dict,
    teacher_ids: Tensor | None = None,
) -> Tensor:
    """
    Compute loss based on run configuration.

    For byte-level (4 heads):
        Head 0: KL(student, teacher) where teacher exists + CE elsewhere
        Heads 1-3: CE only
    For BPE (1 head):
        KL + CE where teacher exists, CE elsewhere

    Args:
        teacher_ids: [B, S, K] long — token IDs for the top-K BPE teacher probs.
                     Required when tok='bpe' and sig='distillation'.
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

        if (sig in ('distillation', 'distillation_then_ce')
                and teacher_probs is not None
                and teacher_mask is not None
                and teacher_ids is not None):
            # Expand sparse top-K teacher distribution to full vocab.
            # teacher_probs: [B, S, K], teacher_ids: [B, S, K]
            teacher_full = torch.zeros_like(logits)  # [B, S, vocab_size]
            teacher_full.scatter_(2, teacher_ids, teacher_probs.to(logits.dtype))
            # Normalize — top-K probs don't necessarily sum to 1
            teacher_full = teacher_full / teacher_full.sum(-1, keepdim=True).clamp(min=1e-8)

            student_log_T = F.log_softmax(logits / T, dim=-1)
            teacher_T = (teacher_full / T).softmax(dim=-1)
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

        if sig in ('distillation', 'distillation_then_ce') and teacher_probs is not None:
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
    original_device = next(model.parameters()).device
    model_cpu = model.cpu()
    model_cpu.eval()
    tok = config['tokenization']

    if tok in ('byte', 'byte_dynamic'):
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

    model.to(original_device)
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

            if tok in ('byte', 'byte_dynamic'):
                x = batch['patches'].to(device)
                targets = batch['targets'].to(device)
            else:
                x = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            t_ids = batch.get('teacher_ids_bpe')
            if t_probs is not None:
                t_probs = t_probs.to(device)
            if t_mask is not None:
                t_mask = t_mask.to(device)
            if t_ids is not None:
                t_ids = t_ids.to(device)

            amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if device.type == 'cuda' else nullcontext())
            with amp_ctx:
                result = model(x)
                if len(result) == 3:
                    logits, _, _ = result
                else:
                    logits, _ = result
                loss = compute_loss(logits, targets, t_probs, t_mask, config,
                                    teacher_ids=t_ids)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow

    # CPU throughput
    tps = measure_cpu_throughput(model, config)
    ram_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    metrics = {
        'run_name': config.get('name', 'unknown'),
        'perplexity': perplexity,
        'tokens_per_sec_cpu': tps,
        'ram_mb': ram_mb,
        'val_loss': avg_loss,
        'n_params_M': sum(p.numel() for p in model.parameters()) / 1e6,
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

    torch.save({
        'model_state_dict': model.state_dict(),
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
              total_steps_override: int | None = None,
              phase: int = 1,
              phase1_ckpt: str | None = None) -> dict:
    """Train one ablation run. Returns final metrics."""
    config = get_run_config(run_name)
    config['_current_phase'] = phase

    if total_steps_override is not None:
        config['total_steps'] = total_steps_override

    assert config['seed'] == BASE_CONFIG['seed'], "Seed mismatch"

    # Device setup (single T4)
    if torch.cuda.is_available() and not dry_run:
        torch.cuda.set_device(0)

    torch.manual_seed(config['seed'])
    device = torch.device('cpu') if dry_run else torch.device('cuda:0')

    # Optimizaciones globales GPU (T4)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')  # TF32: ~2x throughput matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\n{'=' * 60}")
    print(f"Run: {run_name}")
    print(f"Description: {config['description']}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    # Build model — activar gradient checkpointing para ahorrar VRAM en T4
    if not dry_run:
        config['use_gradient_checkpointing'] = True
    model = build_ablation_model(config).to(device)

    # Phase 2: load Phase 1 checkpoint, activate DeleteGate, freeze EmbAligner
    if phase == 2 and phase1_ckpt:
        print(f"[Phase 2] Loading Phase 1 checkpoint: {phase1_ckpt}")
        ckpt = torch.load(phase1_ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        if hasattr(model, 'delete_gate'):
            model.delete_gate.set_elimination_rate(
                config.get('target_elimination', 0.4)
            )
        if hasattr(model, 'embedding_aligner'):
            for p in model.embedding_aligner.parameters():
                p.requires_grad = False
        print(f"[Phase 2] DeleteGate active, EmbAligner frozen")

    # Dataset
    train_ds = load_ablation_dataset(config, split='train', dry_run=dry_run)
    val_ds = load_ablation_dataset(config, split='val' if not dry_run else 'train',
                                    dry_run=dry_run)

    n_workers = 0 if dry_run else 4  # 4 workers: paralelismo de datos para saturar GPU
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_per_gpu'],
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == 'cuda'),  # DMA transfer GPU sin copia extra
        persistent_workers=(n_workers > 0),  # Evitar re-fork cada epoch
        prefetch_factor=2 if n_workers > 0 else None,  # Prefetch 2 batches/worker
        drop_last=False,
    )

    # Optimizer with per-group weight decay
    bitlinear_params = []
    head_params = []
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
        elif 'heads.' in name and 'proj.weight' in name:
            head_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': bitlinear_params,
         'weight_decay': config['weight_decay_bitlinear']},
        {'params': head_params,
         'weight_decay': config['weight_decay_heads']},
        {'params': other_params,
         'weight_decay': 0.0},
    ], lr=config['lr'], betas=(0.9, 0.95))

    use_amp = not dry_run and device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None

    # torch.compile: fusiona kernels, reduce overhead de lanzamiento CUDA
    # En T4 con modelos ~50M params, ganancia tipica 10-20% throughput.
    if device.type == 'cuda' and not dry_run:
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[Opt] torch.compile enabled for training")
        except Exception as e:
            print(f"[Opt] torch.compile failed ({e}), using eager mode")

    # Training loop
    model.train()
    optim_step = 0
    micro_step = 0
    losses = []
    tok = config['tokenization']
    accum_steps = config['gradient_accumulation_steps']
    total_steps = config['total_steps']
    t_start = time.time()

    while optim_step < total_steps:
        for batch in train_loader:
            if optim_step >= total_steps:
                break

            # Unpack batch con non_blocking para solapar transfer CPU->GPU
            if tok in ('byte', 'byte_dynamic'):
                x = batch['patches'].to(device, non_blocking=True)
                targets = batch['targets'].to(device, non_blocking=True)
            else:
                x = batch['input_ids'].to(device, non_blocking=True)
                targets = batch['labels'].to(device, non_blocking=True)

            t_probs = batch.get('teacher_probs', batch.get('teacher_probs_bpe'))
            t_mask = batch.get('teacher_mask', batch.get('teacher_mask_bpe'))
            t_ids = batch.get('teacher_ids_bpe')
            t_hidden = batch.get('teacher_hidden')
            if t_probs is not None:
                t_probs = t_probs.to(device, non_blocking=True)
            if t_mask is not None:
                t_mask = t_mask.to(device, non_blocking=True)
            if t_ids is not None:
                t_ids = t_ids.to(device, non_blocking=True)
            if t_hidden is not None:
                t_hidden = t_hidden.to(device, non_blocking=True)

            is_last_micro = (micro_step % accum_steps == accum_steps - 1)
            amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                # Support both 2-tuple (runs 01-11) and 3-tuple (runs 12-15)
                training_phase = config.get('_current_phase', 1)
                target_elim = config.get('target_elimination', 0.0)
                result = model(
                    x, training_phase=training_phase,
                    target_elimination=target_elim,
                    teacher_hidden=t_hidden, teacher_mask=t_mask,
                )
                if len(result) == 3:
                    logits, _, extra_losses = result
                else:
                    logits, _ = result
                    extra_losses = {}
                loss = compute_loss(logits, targets, t_probs, t_mask, config,
                                    teacher_ids=t_ids)
                # Add extra losses (runs 12-15)
                if 'delete_gate' in extra_losses:
                    loss = loss + 0.1 * extra_losses['delete_gate']
                if 'emb_align' in extra_losses and extra_losses['emb_align'] is not None:
                    loss = loss + config.get('lambda_align', 0.3) * extra_losses['emb_align']
                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(loss.item() * accum_steps)
            micro_step += 1

            # Optimizer step at accumulation boundary
            if micro_step % accum_steps == 0:
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
                optimizer.zero_grad(set_to_none=True)

                optim_step += 1
                for pg in optimizer.param_groups:
                    pg['lr'] = get_lr(optim_step, config)

                if optim_step % 500 == 0:
                    avg_loss = sum(losses[-100:]) / min(100, len(losses))
                    elapsed = time.time() - t_start
                    print(f"Step {optim_step:5d}/{total_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {get_lr(optim_step, config):.2e} | "
                          f"Time: {elapsed:.0f}s")

                # Save periodic checkpoint
                if (optim_step > 0
                        and optim_step % config['save_every_steps'] == 0
                        and not dry_run):
                    save_checkpoint(model, optimizer, run_name, {},
                                    config, optim_step)

    # Final evaluation
    val_metrics = evaluate_run(model, val_ds, config, device)

    if not dry_run:
        save_checkpoint(model, optimizer, run_name, val_metrics,
                        config, optim_step)

    # Save metrics JSON
    metrics_path = f"metrics_{run_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\n[Done] Metrics saved to {metrics_path}")

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
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help="Training phase (1=distillation, 2=delete gate)")
    parser.add_argument('--phase1_ckpt', type=str, default=None,
                        help="Path to Phase 1 checkpoint (required for --phase 2)")
    args = parser.parse_args()

    if args.phase == 2 and not args.phase1_ckpt:
        parser.error("--phase1_ckpt is required when --phase 2")

    metrics = train_run(args.run, dry_run=args.dry_run,
                        total_steps_override=args.total_steps,
                        phase=args.phase,
                        phase1_ckpt=args.phase1_ckpt)
    print(f"\nFinished: {args.run}")
    print(json.dumps(metrics, indent=2))
