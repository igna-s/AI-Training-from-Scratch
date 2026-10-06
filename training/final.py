"""
Training entry point for the production CPU-1 model (CPU1ModelV2).

Two modes:
  1. Phase 1 — distillation training of CPU1ModelV2 (Qwen teacher → CPU-1).
        python -m training.final --dry-run --total-steps 10
        python -m training.final --data-path Cukinator/cpu1-dataset

  2. Phase 2 — DeleteGate training of CPU1ModelV2WithGates from a Phase-1 ckpt.
        python -m training.final --phase2 --phase1-ckpt checkpoint_final.pt

Inference is implemented in `inference/` (C/C++ kernels).
The full evaluation suite lives in `evaluation/`.
"""

import os
import math
import time
import argparse
import contextlib
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from cpu1.components import (
    # Constants
    D_MODEL, N_LAYERS, D_FF, FP_RANK, PFNET_HIDDEN, VOCAB_SIZE, PATCH_SIZE,
    # Primitives / blocks
    RMSNorm, BitLinear, BitEmbedding, FPResidual, PFNet,
    BytePatchEmbeddingV2, BitHead, LocalByteDecoder, CPU1BlockV2,
    BoundaryPredictor, DeleteGate, EmbeddingAligner,
    # Training infrastructure
    DistillationDataset, FineWebStreamingDataset, distillation_loss,
    get_lr, build_param_groups, save_checkpoint, load_checkpoint,
)

_FINEWEB_REPO = 'HuggingFaceFW/fineweb'
_FINEWEB_CONFIG = {
    'tokenization': 'byte',
    'seq_len': 5000,
    'patch_size': PATCH_SIZE,
    'seed': 42,
    'training_signal': 'ce_only',
}


# =============================================================================
# 1. CPU1ModelV2 — production architecture
# =============================================================================

class CPU1ModelV2(nn.Module):
    """CPU-1 v2: ternary weights, MatMul-free token mixing, MegaByte byte head.

    Inference state per-layer:
      - MLGRU h: d_model × 2 bytes (28 KB total at 14 layers × d=1024) → L2
      - PFNet weights: 14 × 16 KB = 224 KB → L2
    """

    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        d_model = config.get('d_model', D_MODEL)
        n_layers = config.get('n_layers', N_LAYERS)
        d_ff = config.get('d_ff', D_FF)
        fp_rank = config.get('fp_rank', FP_RANK)
        pfnet_hidden = config.get('pfnet_hidden', PFNET_HIDDEN)

        self.d_model = d_model
        self.n_layers = n_layers

        self.patch_embedding = BytePatchEmbeddingV2(d_model)
        self.blocks = nn.ModuleList([
            CPU1BlockV2(d_model, d_ff, fp_rank, pfnet_hidden) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # Default head: MegaByte LocalByteDecoder (autoregressive intra-patch
        # chain). Pass legacy_heads=True to load older 4-independent-BitHead
        # checkpoints (keys byte_heads.{0..3}.*).
        if config.get('legacy_heads', False):
            self.byte_heads = nn.ModuleList([
                BitHead(d_model, VOCAB_SIZE) for _ in range(4)
            ])
            self.local_decoder = None
        else:
            self.local_decoder = LocalByteDecoder(
                d_model=d_model,
                patch_size=PATCH_SIZE,
                d_local=config.get('local_decoder_dim', 128),
                n_heads=config.get('local_decoder_heads', 4),
                ff_ratio=config.get('local_decoder_ff_ratio', 4.0),
            )
            self.byte_heads = None

        self.use_gradient_checkpointing = config.get(
            'use_gradient_checkpointing', False
        )

    def _forward_global(
        self,
        patches: Tensor,
        hidden_states: list | None = None,
    ) -> tuple[Tensor, list]:
        """Embedding → all MLGRU blocks → final norm. Returns (post-norm, states)."""
        x = self.patch_embedding(patches)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states: list = []

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _block_fn(x_in, h_in, _block=block):
                    return _block(x_in, h_in)
                x, h_new = checkpoint(_block_fn, x, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)
        else:
            for i, block in enumerate(self.blocks):
                # Touch a layer-i+2 weight pointer to hint at memory prefetch.
                if i + 2 < len(self.blocks):
                    _ = self.blocks[i + 2].token_mixer.forget_proj.weight.data_ptr()
                x, h_new = block(x, hidden_states[i])
                new_states.append(h_new)

        return self.norm(x), new_states

    def forward(
        self,
        patches: Tensor,
        hidden_states: list | None = None,
        byte_targets: Tensor | None = None,
    ) -> tuple[tuple, list]:
        """Returns (logits_tuple, new_hidden_states).

        byte_targets is required when using LocalByteDecoder (teacher forcing).
        Falls back to zeros for benchmarks / dry runs.
        """
        x, new_states = self._forward_global(patches, hidden_states)

        if self.local_decoder is not None:
            if byte_targets is None:
                B, S = x.shape[0], x.shape[1]
                byte_targets = torch.zeros(
                    B, S, PATCH_SIZE, dtype=torch.long, device=x.device
                )
            logits = self.local_decoder(x, byte_targets)
        else:
            logits = tuple(head(x) for head in self.byte_heads)

        return logits, new_states


# =============================================================================
# 2. CPU1ModelV2WithGates — Phase-2 architecture (DeleteGate + optional aux)
# =============================================================================

class CPU1ModelV2WithGates(nn.Module):
    """CPU-1 v2 + BoundaryPredictor / DeleteGate / EmbeddingAligner gates.

    Identical to CPU1ModelV2 when all use_* flags are False — i.e. it's a
    superset that adds the auxiliary modules used in Phase 2 training.
    """

    def __init__(self, config: dict):
        super().__init__()
        d_model = config.get('d_model', D_MODEL)
        n_layers = config.get('n_layers', N_LAYERS)
        d_ff = config.get('d_ff', D_FF)
        fp_rank = config.get('fp_rank', FP_RANK)
        pfnet_hidden = config.get('pfnet_hidden', PFNET_HIDDEN)

        self.use_boundary = config.get('use_boundary_predictor', False)
        self.use_delete_gate = config.get('use_delete_gate', False)
        self.use_emb_align = config.get('use_embedding_alignment', False)
        self.delete_gate_pos = n_layers // 2

        self.patch_embedding = BytePatchEmbeddingV2(d_model)
        self.blocks = nn.ModuleList([
            CPU1BlockV2(d_model, d_ff, fp_rank, pfnet_hidden)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        if config.get('legacy_heads', False):
            self.byte_heads = nn.ModuleList([
                BitHead(d_model, VOCAB_SIZE) for _ in range(4)
            ])
            self.local_decoder = None
        else:
            self.local_decoder = LocalByteDecoder(
                d_model=d_model,
                patch_size=PATCH_SIZE,
                d_local=config.get('local_decoder_dim', 128),
                n_heads=config.get('local_decoder_heads', 4),
                ff_ratio=config.get('local_decoder_ff_ratio', 4.0),
            )
            self.byte_heads = None

        if self.use_boundary:
            self.boundary_predictor = BoundaryPredictor(embed_dim=8, hidden_dim=32)

        if self.use_delete_gate:
            self.delete_gate = DeleteGate(d_model=d_model, elimination_rate=0.0)

        if self.use_emb_align:
            self.embedding_aligner = EmbeddingAligner(
                student_dim=d_model,
                teacher_dim=config.get('teacher_dim', 3584),
            )

        self.use_gradient_checkpointing = config.get(
            'use_gradient_checkpointing', False
        )

    def forward(
        self,
        patches: Tensor,
        hidden_states: list | None = None,
        training_phase: int = 1,
        target_elimination: float = 0.0,
        teacher_hidden: Tensor | None = None,
        teacher_mask: Tensor | None = None,
        byte_targets: Tensor | None = None,
        **kwargs,
    ) -> tuple:
        """Returns (logits_tuple, new_hidden_states, extra_losses_dict)."""
        x = self.patch_embedding(patches)

        if hidden_states is None:
            hidden_states = [None] * len(self.blocks)

        new_states: list = []
        extra_losses: dict = {}

        if self.use_gradient_checkpointing and self.training:
            from torch.utils.checkpoint import checkpoint
            for i, block in enumerate(self.blocks):
                def _block_fn(x_in, h_in, _block=block):
                    return _block(x_in, h_in)
                x, h_new = checkpoint(_block_fn, x, hidden_states[i],
                                      use_reentrant=False)
                new_states.append(h_new)

                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    x, _ = self.delete_gate.forward_train(
                        x, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss
        else:
            for i, block in enumerate(self.blocks):
                if i + 2 < len(self.blocks):
                    _ = self.blocks[i + 2].token_mixer.forget_proj.weight.data_ptr()
                x, h_new = block(x, hidden_states[i])
                new_states.append(h_new)

                if (self.use_delete_gate
                        and i == self.delete_gate_pos
                        and training_phase == 2):
                    x, _ = self.delete_gate.forward_train(
                        x, target_rate=target_elimination
                    )
                    extra_losses['delete_gate'] = self.delete_gate.last_elimination_loss

        x_pre_norm = x
        x = self.norm(x)

        if self.local_decoder is not None:
            if byte_targets is None:
                B, S = x.shape[0], x.shape[1]
                byte_targets = torch.zeros(
                    B, S, PATCH_SIZE, dtype=torch.long, device=x.device
                )
            logits = self.local_decoder(x, byte_targets)
        else:
            logits = tuple(head(x) for head in self.byte_heads)

        if (self.use_emb_align and training_phase == 1
                and teacher_hidden is not None and teacher_mask is not None):
            extra_losses['emb_align'] = self.embedding_aligner.alignment_loss(
                x_pre_norm, teacher_hidden, teacher_mask,
            )

        return logits, new_states, extra_losses


# =============================================================================
# 3. Phase 1 training — distillation against a first-byte teacher
# =============================================================================

def train(args: argparse.Namespace):
    """Phase 1 training loop with multi-GPU DataParallel."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device} ({n_gpus} GPU{'s' if n_gpus != 1 else ''})")

    model = CPU1ModelV2({
        'use_gradient_checkpointing': args.gradient_checkpointing,
    }).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,}")

    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Multi-GPU] DataParallel on {n_gpus} GPUs")

    param_groups = build_param_groups(model)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if args.dry_run:
        dataset = DistillationDataset(
            dry_run_samples=args.dry_run_samples,
            seq_len=args.seq_len,
        )
    elif args.data_path is None or args.data_path == _FINEWEB_REPO:
        fw_cfg = {**_FINEWEB_CONFIG, 'seq_len': args.seq_len}
        print(f"[fineweb] Streaming HuggingFaceFW/fineweb sample-10BT (seq_len={args.seq_len})")
        dataset = FineWebStreamingDataset(fw_cfg, split='train')
    else:
        dataset = DistillationDataset(
            data_path=args.data_path,
            seq_len=args.seq_len,
        )

    use_workers = args.num_workers > 0 and not args.dry_run
    total_batch = args.batch_per_gpu * max(n_gpus, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=total_batch,
        shuffle=True,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=(device.type == 'cuda'),
        drop_last=False,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )

    start_step = 0
    if args.resume_from:
        model_for_load = model.module if hasattr(model, 'module') else model
        start_step = load_checkpoint(args.resume_from, model_for_load, optimizer, scaler)
        print(f"Resumiendo desde step {start_step}")

    grad_accum = args.gradient_accumulation_steps
    total_steps = args.total_steps
    effective_batch = total_batch * grad_accum

    print(f"Batch efectivo: {effective_batch} "
          f"({args.batch_per_gpu}/gpu × {max(n_gpus, 1)} GPUs × {grad_accum} accum)")
    print(f"Steps: {start_step} → {total_steps}")
    print(f"Seq len: {args.seq_len} patches = {args.seq_len * PATCH_SIZE} bytes")

    model.train()
    data_iter = iter(dataloader)
    epoch = 0
    log_timer = time.time()
    metrics: dict[str, float] = {}
    last_loss = 0.0

    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup_steps, total_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)

        optimizer.zero_grad(set_to_none=True)
        should_log = (step % args.log_every == 0)

        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(dataloader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            B, S = patches.shape[0], patches.shape[1]
            _tp = batch.get('teacher_probs')
            _tm = batch.get('teacher_mask')
            teacher_probs = (
                _tp.to(device, non_blocking=True) if _tp is not None
                else torch.zeros(B, S, VOCAB_SIZE, device=device)
            )
            teacher_mask = (
                _tm.to(device, non_blocking=True) if _tm is not None
                else torch.zeros(B, S, dtype=torch.bool, device=device)
            )
            teacher_hidden = batch.get('teacher_hidden')
            if teacher_hidden is not None:
                teacher_hidden = teacher_hidden.to(device, non_blocking=True)

            is_last_micro = (micro_step == grad_accum - 1)
            amp_ctx = (torch.amp.autocast('cuda', dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                # Detect WithGates models that take teacher_hidden in their forward
                fwd = model.module.forward if hasattr(model, 'module') else model.forward
                sig_params = inspect.signature(fwd).parameters
                if 'teacher_hidden' in sig_params:
                    logits, _, _ = model(
                        patches, byte_targets=targets,
                        teacher_hidden=teacher_hidden, teacher_mask=teacher_mask,
                    )
                else:
                    logits, _ = model(patches, byte_targets=targets)
                loss, metrics = distillation_loss(
                    logits, targets, teacher_probs, teacher_mask,
                    temperature=args.temperature,
                    compute_metrics=(should_log and is_last_micro),
                )
                if is_last_micro:
                    last_loss = loss.item()
                loss = loss / grad_accum

            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if should_log:
            elapsed = time.time() - log_timer
            steps_since_log = max(args.log_every, 1)
            patches_per_sec = (
                total_batch * args.seq_len * grad_accum * steps_since_log
            ) / max(elapsed, 1e-6)

            parts = [
                f"Step {step:6d}/{total_steps}",
                f"Loss: {metrics.get('loss_total', 0):.4f}",
                f"H0-KL: {metrics.get('loss_head0_kl', 0):.4f}",
                f"H0-CE: {metrics.get('loss_head0_ce', 0):.4f}",
                f"H123: {metrics.get('loss_heads_123_ce', 0):.4f}",
                f"Tcov: {metrics.get('teacher_coverage', 0):.1%}",
                f"‖g‖: {grad_norm:.2f}",
                f"LR: {lr:.2e}",
                f"P/s: {patches_per_sec:.0f}",
            ]
            if use_amp:
                gpu_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                parts.append(f"VRAM: {gpu_gb:.1f}G")

            print(" | ".join(parts))
            log_timer = time.time()

        if step > 0 and step % args.save_every == 0:
            ckpt_path = str(Path(args.output_dir) / f"checkpoint_{step}.pt")
            save_checkpoint(
                model, optimizer, scaler, step,
                last_loss,
                ckpt_path,
                upload_to_hf=args.upload_hf,
            )
            print(f"  Checkpoint: {ckpt_path}")

    ckpt_path = str(Path(args.output_dir) / "checkpoint_final.pt")
    save_checkpoint(
        model, optimizer, scaler, total_steps,
        last_loss,
        ckpt_path,
        upload_to_hf=args.upload_hf,
    )
    print(f"Training completo. Final: {ckpt_path}")


# =============================================================================
# 4. Phase 2 training — DeleteGate elimination on top of a Phase-1 checkpoint
# =============================================================================

def train_phase2(
    model,
    dataset,
    config: dict,
    checkpoint_phase1: str,
    target_elimination: float = 0.4,
    total_steps: int = 5000,
    lr: float = 1e-4,
    device: torch.device | None = None,
) -> dict:
    """Phase 2: load Phase-1 weights, activate DeleteGate, freeze EmbAligner,
    and train with CE + elimination regularization (no KL — sequence shortens).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(checkpoint_phase1, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)

    if hasattr(model, 'delete_gate'):
        model.delete_gate.set_elimination_rate(target_elimination)

    if hasattr(model, 'embedding_aligner'):
        for p in model.embedding_aligner.parameters():
            p.requires_grad = False

    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Phase2 Multi-GPU] DataParallel on {n_gpus} GPUs")

    def phase2_loss(logits, targets, extra_losses):
        ce = sum(
            F.cross_entropy(
                logits[i].reshape(-1, 256),
                targets[:, :, i].long().reshape(-1)
            ) for i in range(4)
        ) / 4

        elim_loss = extra_losses.get(
            'delete_gate', torch.tensor(0.0, device=device)
        )
        if elim_loss.dim() > 0:
            elim_loss = elim_loss.mean()

        return ce + 0.1 * elim_loss

    model_params = model.module if hasattr(model, 'module') else model
    optimizer = torch.optim.AdamW(
        [p for p in model_params.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95),
    )

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler() if use_amp else None

    total_batch = 4 * max(n_gpus, 1)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=total_batch, shuffle=True, drop_last=True,
    )

    model.train()
    step = 0
    t_start = time.time()

    while step < total_steps:
        for batch in dataloader:
            if step >= total_steps:
                break

            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)

            amp_ctx = (torch.amp.autocast(device_type='cuda', dtype=torch.float16)
                       if use_amp else contextlib.nullcontext())
            with amp_ctx:
                logits, _, extra_losses = model(
                    patches, training_phase=2,
                    target_elimination=target_elimination,
                )
                loss = phase2_loss(logits, targets, extra_losses)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            step += 1
            if step % 500 == 0:
                elapsed = time.time() - t_start
                print(f"[Phase2] Step {step}/{total_steps} | "
                      f"Loss: {loss.item():.4f} | Time: {elapsed:.0f}s")

    return {'phase': 2, 'target_elimination': target_elimination, 'steps': step}


# =============================================================================
# 5. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training CPU-1 con distillation loss")

    p.add_argument('--data-path', type=str, default=None,
                   help="Ruta a HF dataset, directorio, o .parquet. "
                        "Por defecto usa HuggingFaceFW/fineweb (CE-only, sin teacher).")
    p.add_argument('--dry-run', action='store_true',
                   help="Usar datos aleatorios (testing)")
    p.add_argument('--dry-run-samples', type=int, default=1000)
    p.add_argument('--seq-len', type=int, default=5000,
                   help="Patches por secuencia")

    p.add_argument('--batch-per-gpu', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=2)
    p.add_argument('--total-steps', type=int, default=100_000)
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--max-lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=4.0)
    p.add_argument('--gradient-checkpointing', action='store_true')

    p.add_argument('--output-dir', type=str, default='./checkpoints')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--upload-hf', action='store_true',
                   help="Subir checkpoints a Cukinator/cpu1-checkpoints")

    p.add_argument('--resume-from', type=str, default=None,
                   help="Ruta a checkpoint .pt para resumir")
    p.add_argument('--num-workers', type=int, default=4)

    # Phase 2 controls
    p.add_argument('--phase2', action='store_true',
                   help="Run DeleteGate Phase 2 starting from --phase1-ckpt")
    p.add_argument('--phase1-ckpt', type=str, default=None,
                   help="Phase 1 checkpoint to start Phase 2 from")
    p.add_argument('--target-elimination', type=float, default=0.4,
                   help="Phase 2 fraction of states to eliminate")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.phase2:
        if not args.phase1_ckpt:
            raise SystemExit("--phase2 requires --phase1-ckpt")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cfg = {
            'use_delete_gate': True,
            'use_gradient_checkpointing': args.gradient_checkpointing,
        }
        model = CPU1ModelV2WithGates(cfg)

        if args.dry_run:
            dataset = DistillationDataset(
                dry_run_samples=args.dry_run_samples,
                seq_len=args.seq_len,
            )
        else:
            dataset = DistillationDataset(
                data_path=args.data_path,
                seq_len=args.seq_len,
            )

        result = train_phase2(
            model=model,
            dataset=dataset,
            config=cfg,
            checkpoint_phase1=args.phase1_ckpt,
            target_elimination=args.target_elimination,
            total_steps=args.total_steps,
            lr=args.max_lr,
            device=device,
        )
        print(f"[Phase2] Done: {result}")
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
