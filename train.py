"""
Training loops for CPU-1.

Two modes:
  1. Main training: distillation loss on the production model (CPU1ModelV2).
     python train.py --dry-run --total-steps 10

  2. Ablation training: unified loop for all 15 ablation runs.
     python train.py --mode ablation --run run_09 --dry_run
"""

import os
import gc
import math
import json
import time
import contextlib
import argparse
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

from model import CPU1ModelV2, D_MODEL, VOCAB_SIZE, PATCH_SIZE, pack_model_compact


# =============================================================================
# 1. Distillation Loss
# =============================================================================

def distillation_loss(
    student_logits: tuple[Tensor, Tensor, Tensor, Tensor],  # 4 × [B, S, 256]
    target_bytes: Tensor,          # [B, S, 4] — ground truth next patch
    teacher_probs: Tensor | None,  # [B, S, 256] — distribucion marginalizada byte 0
    teacher_mask: Tensor,          # [B, S] bool — True donde hay senal teacher
    temperature: float = 4.0,
    alpha_kl: float = 0.9,
    alpha_ce: float = 0.1,
    compute_metrics: bool = False,
) -> tuple[Tensor, dict[str, float]]:
    """
    Loss combinada para distillation byte-level con teacher BPE.

    Args:
        student_logits: tupla de 4 tensores [B, S, 256], uno por byte del patch.
                        Son los logits crudos (pre-softmax) del student.
        target_bytes:   [B, S, 4] — bytes reales del proximo patch (ground truth)
        teacher_probs:  [B, S, 256] — distribucion de probabilidad del teacher
                        marginalizada al primer byte (viene de first-byte marginalization).
                        Sparse: la mayoria de los 256 valores son 0, solo ~5-20 tienen prob > 0.
        teacher_mask:   [B, S] bool — True en posiciones donde el teacher tiene senal
                        (donde un BPE boundary coincide con un patch boundary, ~20-30% del tiempo)
        temperature:    temperatura para suavizar distribuciones en KL (T=4.0 por default).
                        Mas alto = distribuciones mas planas = el student aprende mas de
                        las probabilidades "no top-1" del teacher (dark knowledge).
        alpha_kl:       peso del KL loss donde hay teacher (0.9 = 90% del loss viene del teacher)
        alpha_ce:       peso del CE loss donde hay teacher (0.1 = 10% hard target)
        compute_metrics: si True, calcula metricas detalladas. Fuerza GPU->CPU sync
                         (lento), usar solo cada N steps de logging.

    Returns:
        (loss, metrics): loss escalar diferenciable + dict de metricas (vacio si compute_metrics=False)

    Logica:
        - Head 0 (primer byte): KL(student/T, teacher/T) * T^2 donde hay senal + CE.
          El factor T^2 compensa que softmax(logits/T) produce gradientes T veces menores.
        - Heads 1-3 (bytes internos): solo CE contra ground truth.
          El teacher BPE no da info util sobre bytes internos a un token.
        - Loss final = promedio de las 4 cabezas.
    """
    metrics: dict[str, float] = {}

    # ===== Head 0: primer byte con posible señal teacher =====
    logits_0 = student_logits[0]  # [B, S, 256]
    targets_0 = target_bytes[:, :, 0].long()  # [B, S]

    ce_loss_0 = F.cross_entropy(
        logits_0.reshape(-1, 256),
        targets_0.reshape(-1),
        reduction='none',
    ).reshape_as(targets_0)  # [B, S]

    if teacher_probs is not None and teacher_mask is not None and teacher_mask.any():
        # Softmax con temperatura sobre logits del student: log(softmax(logits/T))
        student_log_probs_T = F.log_softmax(logits_0 / temperature, dim=-1)

        # Aplicar temperatura al teacher: como no tenemos los logits originales del teacher
        # (solo las probabilidades), usamos p^(1/T) que es equivalente a softmax(logits/T)
        # cuando p = softmax(logits). Despues renormalizamos.
        teacher_probs_T = teacher_probs ** (1.0 / temperature)
        teacher_probs_T = teacher_probs_T / (teacher_probs_T.sum(dim=-1, keepdim=True) + 1e-8)

        # KL divergence: mide cuanto difiere la distribucion del student de la del teacher.
        # sum(dim=-1) suma sobre los 256 bytes para obtener KL por posicion.
        kl = F.kl_div(
            student_log_probs_T,  # log-probs del student (log_target=False -> teacher en probs)
            teacher_probs_T,       # probs del teacher (no log)
            reduction='none',
            log_target=False,
        ).sum(dim=-1)  # [B, S] — un valor KL por posicion

        # Factor T^2: compensa que al dividir logits por T, los gradientes se escalan por 1/T.
        # Sin esto, el KL loss seria T veces mas debil de lo que deberia.
        kl_scaled = kl * (temperature ** 2)

        # Combinar con mascara: posiciones con teacher usan KL+CE mixta,
        # posiciones sin teacher usan solo CE contra ground truth
        mask_float = teacher_mask.float()
        loss_with_teacher = alpha_kl * kl_scaled + alpha_ce * ce_loss_0
        head0_loss = mask_float * loss_with_teacher + (1.0 - mask_float) * ce_loss_0

        if compute_metrics:
            n_teacher = mask_float.sum().clamp(min=1)
            metrics['loss_head0_kl'] = (kl_scaled * mask_float).sum().item() / n_teacher.item()
            metrics['teacher_coverage'] = mask_float.mean().item()
    else:
        head0_loss = ce_loss_0
        if compute_metrics:
            metrics['loss_head0_kl'] = 0.0
            metrics['teacher_coverage'] = 0.0

    total_loss = head0_loss.mean()

    if compute_metrics:
        metrics['loss_head0_ce'] = ce_loss_0.mean().item()

    # ===== Heads 1-3: solo CE, sin teacher =====
    heads_ce_sum = 0.0
    for i in range(1, 4):
        logits_i = student_logits[i]
        targets_i = target_bytes[:, :, i].long()
        ce_i = F.cross_entropy(
            logits_i.reshape(-1, 256),
            targets_i.reshape(-1),
            reduction='mean',
        )
        total_loss = total_loss + ce_i
        if compute_metrics:
            heads_ce_sum += ce_i.item()

    total_loss = total_loss / 4.0  # promedio sobre las 4 cabezas

    if compute_metrics:
        metrics['loss_heads_123_ce'] = heads_ce_sum / 3.0
        metrics['loss_total'] = total_loss.item()

    return total_loss, metrics


# =============================================================================
# 2. Dataset
# =============================================================================

class DistillationDataset(Dataset):
    """
    Carga patches pre-computados + teacher signals.

    Soporta:
    - HuggingFace Dataset / Parquet con document packing pre-hecho
    - Modo dry-run con datos aleatorios (para testing del loop)

    Cada sample:
        patches:       [seq_len, 4] long — input patches
        targets:       [seq_len, 4] long — next patches (shifted by 1)
        teacher_probs: [seq_len, 256] float32 — distribución marginalizada byte 0
        teacher_mask:  [seq_len] bool — True donde hay señal teacher
        teacher_hidden:[seq_len, dim] float32 — estados ocultos del teacher

    Document packing: documentos concatenados con patch separador [0,0,0,0].
    El packing se hace en el script de byte_marginalization, no acá.
    """

    def __init__(
        self,
        data_path: str | None = None,
        seq_len: int = 2048,
        dry_run_samples: int = 0,
        teacher_coverage: float = 0.25,
    ):
        """
        Args:
            data_path:        ruta a HF dataset, directorio, .parquet, o repo HF Hub.
                              Mutuamente excluyente con dry_run_samples.
            seq_len:          patches por secuencia (2048 patches = 8192 bytes de contexto)
            dry_run_samples:  si > 0, genera datos aleatorios en vez de cargar datos reales.
                              Util para testear el training loop sin necesitar el dataset.
            teacher_coverage: fraccion de posiciones con senal teacher en dry_run (~25%)
        """
        self.seq_len = seq_len

        if dry_run_samples > 0:
            self.mode = 'dry_run'  # Datos sinteticos para testing
            self.n_samples = dry_run_samples
            self.teacher_coverage = teacher_coverage
        elif data_path is not None:
            self.mode = 'hf'  # Datos reales desde HuggingFace/Parquet
            self._load_hf(data_path)
        else:
            raise ValueError("Necesitas data_path o dry_run_samples > 0")

    def _load_hf(self, data_path: str):
        """Carga desde directorio HF o Parquet.

        Soporta:
          - Directorio local con dataset_info.json (load_from_disk)
          - Archivo .parquet individual
          - Directorio local con shard_*.parquet
          - Repo HF Hub con subdir byte_marginalized/ (formato streaming pipeline)
          - Repo HF Hub estándar
        """
        from datasets import load_from_disk, load_dataset
        p = Path(data_path)
        if p.is_dir() and (p / 'dataset_info.json').exists():
            self.dataset = load_from_disk(data_path)
        elif p.suffix == '.parquet':
            self.dataset = load_dataset('parquet', data_files=str(p), split='train')
        elif p.is_dir():
            # Directorio local con shards (byte_marginalized/ o similar)
            shards = sorted(p.glob('shard_*.parquet'))
            if not shards:
                shards = sorted(p.glob('**/*.parquet'))
            self.dataset = load_dataset(
                'parquet', data_files=[str(s) for s in shards], split='train',
            )
        else:
            # Repo HF Hub — tiene byte_marginalized/shard_*.parquet
            from huggingface_hub import list_repo_files
            try:
                files = [
                    f"hf://datasets/{data_path}/{f}"
                    for f in list_repo_files(data_path, repo_type='dataset')
                    if f.startswith('byte_marginalized/') and f.endswith('.parquet')
                ]
                if files:
                    self.dataset = load_dataset('parquet', data_files=files, split='train')
                else:
                    self.dataset = load_dataset(data_path, split='train')
            except Exception:
                self.dataset = load_dataset(data_path, split='train')

    def __len__(self) -> int:
        if self.mode == 'dry_run':
            return self.n_samples
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if self.mode == 'dry_run':
            return self._get_dry_run_item()
        return self._get_hf_item(idx)

    def _get_dry_run_item(self) -> dict[str, Tensor]:
        """Sample aleatorio para testing del loop."""
        S = self.seq_len
        patches = torch.randint(0, 256, (S, PATCH_SIZE), dtype=torch.long)
        targets = torch.randint(0, 256, (S, PATCH_SIZE), dtype=torch.long)

        # Teacher: distribución sparse
        teacher_probs = torch.zeros(S, VOCAB_SIZE)
        teacher_mask = torch.rand(S) < self.teacher_coverage

        # Llenar posiciones con señal teacher
        n_with_teacher = teacher_mask.sum().item()
        if n_with_teacher > 0:
            # ~8 bytes activos por posición, softmax random
            for i in teacher_mask.nonzero(as_tuple=True)[0]:
                n_active = torch.randint(3, 12, (1,)).item()
                active = torch.randperm(256)[:n_active]
                teacher_probs[i, active] = torch.softmax(torch.randn(n_active), dim=0)

        teacher_hidden = torch.zeros(S, 128, dtype=torch.float32)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
            'teacher_hidden': teacher_hidden,
        }

    def _get_hf_item(self, idx: int) -> dict[str, Tensor]:
        """Carga sample desde HuggingFace dataset.

        Soporta tanto arrays nativos (HF Dataset con features tipadas) como
        JSON strings (formato de byte_marginalization.py en Parquet crudo).
        """
        item = self.dataset[idx]

        def _to_tensor(raw, dtype):
            if isinstance(raw, str):
                raw = json.loads(raw)
            return torch.as_tensor(raw, dtype=dtype)

        patches = _to_tensor(item['patches'], torch.long)
        targets = _to_tensor(item['targets'], torch.long)

        # Dataset stores flat arrays: [seq_len*4] and [seq_len*256]
        # Reshape to [seq_len, PATCH_SIZE] and [seq_len, VOCAB_SIZE]
        seq_len = patches.numel() // PATCH_SIZE
        patches = patches.view(seq_len, PATCH_SIZE)
        targets = targets.view(seq_len, PATCH_SIZE)

        if 'teacher_probs' in item and item['teacher_probs'] is not None:
            teacher_probs = _to_tensor(item['teacher_probs'], torch.float32)
            teacher_probs = teacher_probs.view(seq_len, VOCAB_SIZE)
        else:
            teacher_probs = torch.zeros(seq_len, VOCAB_SIZE)

        if 'teacher_mask' in item and item['teacher_mask'] is not None:
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
        }


# =============================================================================
# 3. LR Schedule
# =============================================================================

def get_lr(
    step: int,
    warmup_steps: int = 2000,
    total_steps: int = 100_000,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
) -> float:
    """
    Cosine annealing con warmup lineal.

    Args:
        step:         step actual del training (0-indexed)
        warmup_steps: pasos de warmup lineal (0 -> max_lr)
        total_steps:  pasos totales de training
        max_lr:       learning rate maximo alcanzado al final del warmup
        min_lr:       learning rate minimo al final del coseno (10% del max)
    Returns:
        learning rate para este step

    Curva:
        [0, warmup)       -> rampa lineal de 0 a max_lr
        [warmup, total)   -> coseno de max_lr a min_lr
        [total, ...)      -> min_lr constante
    """
    if step < warmup_steps:
        return max_lr * step / max(warmup_steps, 1)  # Rampa lineal
    if step >= total_steps:
        return min_lr  # Piso
    # Cosine decay: empieza en max_lr, termina en min_lr
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


# =============================================================================
# 4. Param Groups
# =============================================================================

def build_param_groups(model: nn.Module) -> list[dict]:
    """
    Separa parámetros en 4 grupos con diferente weight decay y LR.

    - BitLinear weights + embedding projection: wd=0.1
    - FPResidual, RMSNorm, biases, byte_embedding, BitEmbedding scale: wd=0.0
    - Head weights (nn.Linear o BitHead): wd=0.01
    - PFNet weights: wd=0.05, lr=base_lr*1.5 (higher LR for faster convergence)

    Compatible con CPU1ModelV2.
    """
    decay_params = []       # BitLinear weights, embedding proj
    no_decay_params = []    # FPResidual, norms, biases, byte embedding
    head_params = []        # prediction heads
    pfnet_params = []       # PFNet weights (separate LR)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # PFNet parameters get their own group
        if '.pfnet.' in name:
            pfnet_params.append(param)
        # v1 usa head_0..head_3, v2 usa byte_heads.0..byte_heads.3, new: local_decoder
        elif ('head_' in name or 'byte_heads' in name) and 'weight' in name:
            head_params.append(param)
        # pos_embed is a learned position embedding — no weight decay
        elif any(k in name for k in ('fp_A', 'fp_B', 'norm', 'bias',
                                      'byte_embedding', 'byte_embed',
                                      'output_scale', '.scale', 'pos_embed')):
            no_decay_params.append(param)
        else:
            # BitLinear weights + local_decoder projections
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': head_params, 'weight_decay': 0.01},
        {'params': pfnet_params, 'weight_decay': 0.05, 'lr_scale': 1.5},
    ]


# =============================================================================
# 5. Checkpoint
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    loss: float,
    path: str,
    upload_to_hf: bool = False,
    hf_repo: str = "Cukinator/cpu1-checkpoints",
):
    """
    Guarda checkpoint a disco y opcionalmente sube a HuggingFace Hub.

    Args:
        model:        modelo
        optimizer:    estado del optimizador (momentum, etc.)
        scaler:       estado del GradScaler (escala FP16 actual)
        step:         step de training actual
        loss:         ultimo loss registrado
        path:         ruta local donde guardar el .pt
        upload_to_hf: si True, sube el .pt al repo de HuggingFace
        hf_repo:      repo HF destino (default: Cukinator/cpu1-checkpoints)
    """
    # Unwrap DataParallel si es necesario
    model_to_save = model.module if hasattr(model, 'module') else model
    state = model_to_save.state_dict()
    checkpoint = {
        'step': step,
        'model_state_dict': state,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, path)

    if upload_to_hf:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=Path(path).name,
                repo_id=hf_repo,
                repo_type="model",
            )
        except Exception as e:
            print(f"[WARN] No se pudo subir checkpoint a HF Hub: {e}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> int:
    """
    Carga checkpoint y retorna el step desde donde resumir.

    Args:
        path:      ruta al archivo .pt
        model:     modelo destino
        optimizer: si se pasa, restaura el estado del optimizador (momentum, etc.)
        scaler:    si se pasa, restaura el estado del GradScaler
    Returns:
        step (int) desde el cual resumir el training
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('step', 0)


# =============================================================================
# 6. Training Loop
# =============================================================================

def train(args: argparse.Namespace):
    """Training loop principal (multi-GPU con DataParallel si hay 2+ GPUs)."""
    # --- Setup dispositivo ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0

    # --- Optimizaciones globales GPU ---
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  # Autotuner de kernels convolucionales
        torch.set_float32_matmul_precision('high')  # TF32 en T4: ~2x throughput en matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Device: {device} ({n_gpus} GPU{'s' if n_gpus != 1 else ''})")

    # --- Modelo ---
    model = CPU1ModelV2({
        'use_gradient_checkpointing': args.gradient_checkpointing,
    }).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,}")

    # --- Multi-GPU ---
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Multi-GPU] DataParallel on {n_gpus} GPUs")

    # --- Optimizer ---
    param_groups = build_param_groups(model)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=args.max_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # --- Mixed precision (FP16 en T4, deshabilitado en CPU) ---
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # --- Dataset ---
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

    use_workers = args.num_workers > 0 and not args.dry_run
    total_batch = args.batch_per_gpu * max(n_gpus, 1)
    dataloader = DataLoader(
        dataset,
        batch_size=total_batch,
        shuffle=True,
        num_workers=args.num_workers if not args.dry_run else 0,
        pin_memory=(device.type == 'cuda'),  # DMA transfer directo sin copia extra
        drop_last=False,
        persistent_workers=use_workers,  # Evita re-fork de workers cada epoch
        prefetch_factor=2 if use_workers else None,  # Prefetch 2 batches por worker
    )

    # --- Resume ---
    start_step = 0
    if args.resume_from:
        # Unwrap DataParallel para cargar state_dict sin prefix 'module.'
        model_for_load = model.module if hasattr(model, 'module') else model
        start_step = load_checkpoint(args.resume_from, model_for_load, optimizer, scaler)
        print(f"Resumiendo desde step {start_step}")

    # --- Config ---
    grad_accum = args.gradient_accumulation_steps
    total_steps = args.total_steps
    effective_batch = total_batch * grad_accum

    print(f"Batch efectivo: {effective_batch} "
          f"({args.batch_per_gpu}/gpu × {max(n_gpus, 1)} GPUs × {grad_accum} accum)")
    print(f"Steps: {start_step} → {total_steps}")
    print(f"Seq len: {args.seq_len} patches = {args.seq_len * PATCH_SIZE} bytes")

    # --- Training loop ---
    model.train()
    data_iter = iter(dataloader)
    epoch = 0
    log_timer = time.time()
    metrics: dict[str, float] = {}
    last_loss = 0.0

    for step in range(start_step, total_steps):
        # LR schedule
        lr = get_lr(step, args.warmup_steps, total_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr * pg.get('lr_scale', 1.0)

        optimizer.zero_grad(set_to_none=True)  # Más rápido que fill(0): libera memoria
        should_log = (step % args.log_every == 0)

        for micro_step in range(grad_accum):
            # Obtener batch (reciclar dataloader al agotarse)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                data_iter = iter(dataloader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            teacher_probs = batch['teacher_probs'].to(device, non_blocking=True)
            teacher_mask = batch['teacher_mask'].to(device, non_blocking=True)
            teacher_hidden = batch.get('teacher_hidden')
            if teacher_hidden is not None:
                teacher_hidden = teacher_hidden.to(device, non_blocking=True)

            is_last_micro = (micro_step == grad_accum - 1)
            amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with amp_ctx:
                # If model is AblationModel, it might accept teacher_hidden
                import inspect
                sig_params = (
                    inspect.signature(model.forward).parameters
                    if not hasattr(model, 'module')
                    else inspect.signature(model.module.forward).parameters
                )
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

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if should_log:
            elapsed = time.time() - log_timer
            steps_since_log = max(args.log_every, 1)
            patches_per_sec = (total_batch * args.seq_len * grad_accum * steps_since_log) / max(elapsed, 1e-6)

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

        # Checkpoint
        if step > 0 and step % args.save_every == 0:
            ckpt_path = str(Path(args.output_dir) / f"checkpoint_{step}.pt")
            save_checkpoint(
                model, optimizer, scaler, step,
                last_loss,
                ckpt_path,
                upload_to_hf=args.upload_hf,
            )
            print(f"  Checkpoint: {ckpt_path}")

    # --- Final ---
    ckpt_path = str(Path(args.output_dir) / f"checkpoint_final.pt")
    save_checkpoint(
        model, optimizer, scaler, total_steps,
        last_loss,
        ckpt_path,
        upload_to_hf=args.upload_hf,
    )
    print(f"Training completo. Final: {ckpt_path}")


# =============================================================================
# 8. CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training CPU-1 con distillation loss")

    # Data
    p.add_argument('--data-path', type=str, default=None,
                   help="Ruta a HF dataset, directorio, o archivo .parquet")
    p.add_argument('--dry-run', action='store_true',
                   help="Usar datos aleatorios (testing)")
    p.add_argument('--dry-run-samples', type=int, default=1000)
    p.add_argument('--seq-len', type=int, default=5000,
                   help="Patches por secuencia")

    # Training
    p.add_argument('--batch-per-gpu', type=int, default=4)
    p.add_argument('--gradient-accumulation-steps', type=int, default=2)
    p.add_argument('--total-steps', type=int, default=100_000)
    p.add_argument('--warmup-steps', type=int, default=2000)
    p.add_argument('--max-lr', type=float, default=3e-4)
    p.add_argument('--min-lr', type=float, default=3e-5)
    p.add_argument('--temperature', type=float, default=4.0)
    p.add_argument('--gradient-checkpointing', action='store_true')

    # Output
    p.add_argument('--output-dir', type=str, default='./checkpoints')
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--upload-hf', action='store_true',
                   help="Subir checkpoints a Cukinator/cpu1-checkpoints")

    # Resume
    p.add_argument('--resume-from', type=str, default=None,
                   help="Ruta a checkpoint .pt para resumir")

    # Workers: 4 para saturar el pipeline de datos en T4
    p.add_argument('--num-workers', type=int, default=4)

    return p.parse_args()


# =============================================================================
# 9. Phase 2 Training — DeleteGate (MrT5)
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
    """
    Fase 2 de training: activar DeleteGate y entrenar para velocidad.

    Cambios respecto a Fase 1:
    1. Cargar checkpoint de Fase 1
    2. Activar DeleteGate (elimination_rate > 0)
    3. Congelar EmbeddingAligner (si existe)
    4. Cambiar loss: solo CE, sin KL distillation (porque la secuencia se acorta)
    5. Agregar elimination_loss para regularizar el DeleteGate

    POR QUÉ cambiar la loss en Fase 2:
        La distillation loss requiere alineación posición a posición.
        Con DeleteGate activo, algunas posiciones se eliminan.
        No hay garantía de que el teacher_prob[i] corresponda a student_output[i].
        → Usar solo CE sobre los estados que sobreviven.

    POR QUÉ no usar nueva distilación en Fase 2:
        El modelo de Fase 1 ya aprendió el conocimiento del teacher.
        Fase 2 solo aprende a ser eficiente sin degradar calidad.
        El "nuevo teacher" para Fase 2 es el propio modelo de Fase 1.

    CONCURRENCIA GPU en Fase 2:
        - Single T4: gradient accumulation para batch efectivo grande
        - Los gradientes fluyen a través del soft mask del DeleteGate
    """
    from model import CPU1ModelV2WithGates

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Cargar Fase 1
    ckpt = torch.load(checkpoint_phase1, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.to(device)

    # Activar DeleteGate
    if hasattr(model, 'delete_gate'):
        model.delete_gate.set_elimination_rate(target_elimination)

    # Congelar EmbeddingAligner
    if hasattr(model, 'embedding_aligner'):
        for p in model.embedding_aligner.parameters():
            p.requires_grad = False

    # Multi-GPU
    n_gpus = torch.cuda.device_count() if device.type == 'cuda' else 0
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Phase2 Multi-GPU] DataParallel on {n_gpus} GPUs")

    # Loss de Fase 2: CE + elimination_regularization
    def phase2_loss(logits, targets, extra_losses):
        # CE puro sobre tokens que sobreviven
        ce = sum(
            F.cross_entropy(
                logits[i].reshape(-1, 256),
                targets[:, :, i].long().reshape(-1)
            ) for i in range(4)
        ) / 4

        # Regularización del DeleteGate (con DataParallel, escalares → dim>0)
        elim_loss = extra_losses.get('delete_gate', torch.tensor(0.0, device=device))
        if elim_loss.dim() > 0:
            elim_loss = elim_loss.mean()

        return ce + 0.1 * elim_loss

    # Optimizer
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

    # Training loop
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
# 10. Tests
# =============================================================================

def test_distillation_loss():
    """Verifica que la loss computa correctamente y es diferenciable."""
    B, S = 2, 8
    logits = tuple(torch.randn(B, S, 256, requires_grad=True) for _ in range(4))
    targets = torch.randint(0, 256, (B, S, 4))

    # Teacher: distribución sparse sobre ~10 bytes
    teacher_probs = torch.zeros(B, S, 256)
    for b in range(B):
        for s in range(S):
            active = torch.randint(0, 256, (8,))
            teacher_probs[b, s, active] = torch.softmax(torch.randn(8), dim=0)
    teacher_mask = torch.rand(B, S) > 0.5

    loss, metrics = distillation_loss(
        logits, targets, teacher_probs, teacher_mask,
        compute_metrics=True,
    )

    assert loss.shape == (), f"Loss debería ser escalar, got {loss.shape}"
    assert loss.requires_grad, "Loss debería requerir gradientes"
    assert not math.isnan(loss.item()), "Loss es NaN"
    assert 'loss_total' in metrics
    assert 'loss_head0_kl' in metrics
    assert 'teacher_coverage' in metrics

    # Verificar backward
    loss.backward()
    for i, l in enumerate(logits):
        assert l.grad is not None, f"Head {i} no recibió gradiente"

    print(f"[OK] distillation_loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"     {k}: {v:.4f}")


def test_param_groups():
    """Verifica que todos los parámetros se asignan a exactamente un grupo (v1 y v2)."""
    for label, model in [
        ("v2", CPU1ModelV2({'d_model': 2048, 'n_layers': 2, 'd_ff': 5504, 'fp_rank': 8})),
    ]:
        groups = build_param_groups(model)

        grouped = sum(p.numel() for g in groups for p in g['params'])
        total = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert grouped == total, (
            f"[{label}] Parámetros en grupos ({grouped:,}) != total ({total:,}). "
            f"Hay parámetros sin asignar."
        )

        # Verificar que no hay duplicados
        ids = []
        for g in groups:
            for p in g['params']:
                assert id(p) not in ids, f"[{label}] Parámetro duplicado en param groups"
                ids.append(id(p))

        print(f"[OK] Param groups {label} (2 capas):")
        for i, g in enumerate(groups):
            n_params = sum(p.numel() for p in g['params'])
            print(f"     Grupo {i} (wd={g['weight_decay']}): {n_params:,} params, "
                  f"{len(g['params'])} tensores")


def test_dataset_dry_run():
    """Verifica que el dataset dry-run produce shapes correctas."""
    ds = DistillationDataset(dry_run_samples=10, seq_len=16)
    assert len(ds) == 10

    sample = ds[0]
    assert sample['patches'].shape == (16, 4), f"patches: {sample['patches'].shape}"
    assert sample['targets'].shape == (16, 4), f"targets: {sample['targets'].shape}"
    assert sample['teacher_probs'].shape == (16, 256), f"teacher_probs: {sample['teacher_probs'].shape}"
    assert sample['teacher_mask'].shape == (16,), f"teacher_mask: {sample['teacher_mask'].shape}"

    # Verificar que teacher_probs suma ~1 donde hay señal
    mask = sample['teacher_mask']
    if mask.any():
        probs_sum = sample['teacher_probs'][mask].sum(dim=-1)
        assert (probs_sum > 0.99).all(), f"Teacher probs no suman 1: {probs_sum}"

    print(f"[OK] Dataset dry-run: {len(ds)} samples, shapes correctas")


def test_lr_schedule():
    """Verifica el LR schedule."""
    # Warmup
    assert get_lr(0) == 0.0
    assert abs(get_lr(1000, warmup_steps=2000) - 1.5e-4) < 1e-8
    assert abs(get_lr(2000, warmup_steps=2000) - 3e-4) < 1e-8

    # Cosine decay
    lr_mid = get_lr(51000, warmup_steps=2000, total_steps=100_000)
    assert 3e-5 < lr_mid < 3e-4, f"LR mid: {lr_mid}"

    # Final
    lr_end = get_lr(100_000, warmup_steps=2000, total_steps=100_000)
    assert abs(lr_end - 3e-5) < 1e-8, f"LR final: {lr_end}"

    print(f"[OK] LR schedule: warmup ->{get_lr(2000):.2e}, mid ->{lr_mid:.2e}, end ->{lr_end:.2e}")



# =============================================================================
# Ablation: Dataset loaders
# =============================================================================

from ablation import BASE_CONFIG, RUN_CONFIGS, ALL_RUN_CONFIGS, get_run_config, build_ablation_model

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
                teacher_dim = self.config.get('teacher_dim', 128)
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
        self.config = config
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
            teacher_dim = self.config.get('teacher_dim', 128)
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
                          dry_run: bool = False,
                          mount_path: Path | None = None):
    """Load dataset from hf-mount (preferred) or HuggingFace Hub download.

    Con mount_path: lee parquet files directamente del mount (lazy, sin cache).
    Sin mount_path: descarga completa via load_dataset (usa ~5 GB de disco).
    """
    if dry_run:
        return DryRunDataset(config)

    from datasets import load_dataset

    repo = config['dataset_repo']
    tok = config['tokenization']
    subset = 'bpe_tokenized' if tok == 'bpe' else 'byte_marginalized'

    from huggingface_hub import list_repo_files, hf_hub_download

    # Cargar todos los shards del subset
    if mount_path is not None:
        parquet_dir = Path(mount_path) / subset
        all_files = sorted(str(f) for f in parquet_dir.glob('shard_*.parquet'))
        if not all_files:
            raise FileNotFoundError(f"No shards in {parquet_dir}")
        ds = load_dataset('parquet', data_files=all_files, split='train')
        print(f"[mount] Loaded {len(ds)} rows from {len(all_files)} shards ({subset})")
    else:
        try:
            repo_files = list(list_repo_files(repo, repo_type='dataset'))
            hub_files = [
                f"hf://datasets/{repo}/{f}"
                for f in repo_files
                if f.startswith(f'{subset}/') and f.endswith('.parquet')
            ]
            if not hub_files:
                raise FileNotFoundError(f"No shards for {subset}/ in {repo}")
            ds = load_dataset('parquet', data_files=hub_files, split='train')
            print(f"[hub] Loaded {len(ds)} rows from {len(hub_files)} shards ({subset})")
        except Exception as e:
            print(f"[hub] Fallback to load_dataset: {e}")
            ds = load_dataset(repo, subset, split=split)

    # Aplicar split (train/val/test) desde split_info.json
    if split != 'all':
        split_info = None
        # Buscar split_info.json local o en HF Hub
        local_split = Path(mount_path) / subset / 'split_info.json' if mount_path else None
        if local_split and local_split.exists():
            import json as _json
            with open(local_split) as f:
                split_info = _json.load(f)
        else:
            try:
                split_path = hf_hub_download(
                    repo_id=repo, filename=f'{subset}/split_info.json', repo_type='dataset',
                )
                import json as _json
                with open(split_path) as f:
                    split_info = _json.load(f)
            except Exception:
                pass  # sin split_info: usar todo para train, nada para val

        if split_info and split in split_info:
            indices = split_info[split]
            # Filtrar índices que existen en el dataset
            indices = [i for i in indices if i < len(ds)]
            ds = ds.select(indices)
            print(f"[split] {split}: {len(ds)} sequences")
        elif split == 'train':
            pass  # sin split_info: todo es train
        else:
            # val/test sin split_info: tomar último 5%
            n = max(1, len(ds) // 20)
            ds = ds.select(range(len(ds) - n, len(ds)))
            print(f"[split] {split} (fallback last 5%): {len(ds)} sequences")

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

def get_lr_from_config(step: int, config: dict) -> float:
    """Cosine annealing with linear warmup (ablation variant using config dict)."""
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
                # byte_targets is what the LocalByteDecoder teacher-forces on.
                # Ignored by AblationModel when use_local_decoder is False or BPE.
                fwd_kwargs = {'byte_targets': targets} if tok in ('byte', 'byte_dynamic') else {}
                result = model(x, **fwd_kwargs)
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

def save_ablation_checkpoint(model: nn.Module, optimizer, run_name: str,
                             metrics: dict, config: dict, step: int,
                             is_final: bool = False) -> str:
    """
    Save ablation checkpoint locally and optionally to HuggingFace Hub.

    Two formats:
    - Intermediate (is_final=False): full resume checkpoint — model in BF16,
      optimizer in BF16, needed to continue training after instance preemption.
      ~2× smaller than FP32 (~300 MB for 50M vs ~600 MB).
    - Final (is_final=True): inference-only compact checkpoint — ternary weights
      packed to 2-bit (4 values/byte), rest in BF16, NO optimizer state.
      ~14 MB for 50M, ~44 MB for 170M (16× smaller than FP32).
    """
    path = (
        f"checkpoint_{run_name}_final.pt"
        if is_final
        else f"checkpoint_{run_name}_step{step}.pt"
    )

    model_to_save = model.module if hasattr(model, 'module') else model

    if is_final:
        # Inference checkpoint: 2-bit packed weights, no optimizer state
        torch.save({
            'compact': pack_model_compact(model_to_save),
            'metrics': metrics,
            'step': step,
            'run_name': run_name,
            'config': config,
            'format': 'compact_2bit',
        }, path)
        print(f"[Checkpoint] Final compact saved: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")
    else:
        # Resume checkpoint: BF16 model + BF16 optimizer (2× smaller than FP32)
        model_bf16 = {
            k: v.to(torch.bfloat16) if v.is_floating_point() else v
            for k, v in model_to_save.state_dict().items()
        }
        opt_state = optimizer.state_dict()
        # Optimizer momentum tensors to BF16 to halve their size
        for group in opt_state.get('state', {}).values():
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in group and group[key].is_floating_point():
                    group[key] = group[key].to(torch.bfloat16)

        torch.save({
            'model_state_dict': model_bf16,
            'optimizer_state_dict': opt_state,
            'metrics': metrics,
            'step': step,
            'run_name': run_name,
            'config': config,
            'format': 'bf16',
        }, path)
        print(f"[Checkpoint] Resume checkpoint saved: {path} "
              f"({os.path.getsize(path) / 1e6:.1f} MB)")

    # Upload to Hub
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
            print(f"[Checkpoint] Uploaded to HF: {path}")
        except Exception as e:
            print(f"[Checkpoint] Hub upload failed: {e}")

    # Disk cleanup: keep only the latest resume checkpoint per run
    if not is_final:
        import glob as _glob
        for old in _glob.glob(f"checkpoint_{run_name}_step*.pt"):
            if old != path:
                try:
                    os.remove(old)
                    print(f"[Checkpoint] Removed old: {old}")
                except OSError:
                    pass

    return path


# ============================================================
# Main training loop
# ============================================================

def train_run(run_name: str, dry_run: bool = False,
              total_steps_override: int | None = None,
              phase: int = 1,
              phase1_ckpt: str | None = None,
              use_hf_mount: bool = False) -> dict:
    """Train one ablation run. Returns final metrics."""
    config = get_run_config(run_name)
    config['_current_phase'] = phase

    if total_steps_override is not None:
        config['total_steps'] = total_steps_override

    assert config['seed'] == BASE_CONFIG['seed'], "Seed mismatch"

    # Device setup (multi-GPU si disponible)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() and not dry_run else 0
    if n_gpus > 0:
        torch.cuda.set_device(0)

    # Full seeding — same as create_dataset.py (SEED=42)
    import random as _random
    import numpy as _np
    _random.seed(config['seed'])
    _np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
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
    print(f"Device: {device} ({n_gpus} GPU{'s' if n_gpus != 1 else ''})")
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

    # Multi-GPU: wrap con DataParallel (despues de cargar checkpoint)
    if n_gpus > 1:
        model = torch.nn.DataParallel(model)
        print(f"[Multi-GPU] DataParallel on {n_gpus} GPUs: "
              f"{[torch.cuda.get_device_name(i) for i in range(n_gpus)]}")

    # Dataset — usar hf-mount si disponible (lazy load, ahorra disco en Kaggle)
    mount_path = None
    if use_hf_mount and not dry_run:
        try:
            from hf_mount import mount_bucket
            mount_path = mount_bucket(config['dataset_repo'])
        except Exception as e:
            print(f"[hf-mount] Mount failed, falling back to download: {e}")

    train_ds = load_ablation_dataset(config, split='train', dry_run=dry_run,
                                      mount_path=mount_path)
    val_ds = load_ablation_dataset(config, split='val' if not dry_run else 'train',
                                    dry_run=dry_run, mount_path=mount_path)

    n_workers = 0 if dry_run else 4  # 4 workers: paralelismo de datos para saturar GPU
    total_batch = config['batch_per_gpu'] * max(n_gpus, 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=total_batch,
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
    # No compatible con DataParallel — skip si multi-GPU.
    if device.type == 'cuda' and not dry_run and n_gpus <= 1:
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
    last_ckpt_time = t_start

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
                training_phase = config.get('_current_phase', 1)
                target_elim = config.get('target_elimination', 0.0)
                fwd_kwargs = {'byte_targets': targets} if tok in ('byte', 'byte_dynamic') else {}
                result = model(
                    x, training_phase=training_phase,
                    target_elimination=target_elim,
                    teacher_hidden=t_hidden, teacher_mask=t_mask,
                    **fwd_kwargs,
                )
                if len(result) == 3:
                    logits, _, extra_losses = result
                else:
                    logits, _ = result
                    extra_losses = {}
                loss = compute_loss(logits, targets, t_probs, t_mask, config,
                                    teacher_ids=t_ids)
                # Add extra losses (runs 12-15)
                # Con DataParallel, escalares se concatenan en dim 0 → tomar mean
                if 'delete_gate' in extra_losses:
                    dg = extra_losses['delete_gate']
                    loss = loss + 0.1 * (dg.mean() if dg.dim() > 0 else dg)
                if 'emb_align' in extra_losses and extra_losses['emb_align'] is not None:
                    ea = extra_losses['emb_align']
                    loss = loss + config.get('lambda_align', 0.3) * (ea.mean() if ea.dim() > 0 else ea)
                if 'boundary' in extra_losses:
                    bp = extra_losses['boundary']
                    loss = loss + 0.1 * (bp.mean() if bp.dim() > 0 else bp)
                loss = loss / accum_steps

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            losses.append(loss.item() * accum_steps)
            del logits, result, extra_losses
            if t_probs is not None:
                del t_probs, t_mask
            if t_hidden is not None:
                del t_hidden
            del loss
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
                    pg['lr'] = get_lr_from_config(optim_step, config)

                if optim_step % 500 == 0:
                    avg_loss = sum(losses[-100:]) / min(100, len(losses))
                    elapsed = time.time() - t_start
                    if device.type == 'cuda':
                        vram_mb = torch.cuda.memory_allocated() / (1024**2)
                        vram_res_mb = torch.cuda.memory_reserved() / (1024**2)
                        print(f"Step {optim_step:5d}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {get_lr_from_config(optim_step, config):.2e} | "
                              f"VRAM: {vram_mb:.0f}/{vram_res_mb:.0f} MB | "
                              f"Time: {elapsed:.0f}s")
                        gc.collect()
                        torch.cuda.empty_cache()
                    else:
                        print(f"Step {optim_step:5d}/{total_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {get_lr_from_config(optim_step, config):.2e} | "
                              f"Time: {elapsed:.0f}s")

                # Save periodic checkpoint: step 10 (smoke test), then every 25 min by wall time
                now = time.time()
                should_save = (
                    (optim_step == 10)
                    or (not dry_run and (now - last_ckpt_time) >= 25 * 60)
                )
                if should_save and not dry_run:
                    save_ablation_checkpoint(model, optimizer, run_name, {},
                                             config, optim_step)
                    last_ckpt_time = now

    # Final evaluation (unwrap DataParallel)
    model_eval = model.module if hasattr(model, 'module') else model
    val_metrics = evaluate_run(model_eval, val_ds, config, device)

    if not dry_run:
        # Final checkpoint: compact 2-bit inference format (no optimizer state, ~14MB for 50M)
        save_ablation_checkpoint(model, optimizer, run_name, val_metrics,
                                 config, optim_step, is_final=True)

    # Save metrics JSON
    metrics_path = f"metrics_{run_name}.json"
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\n[Done] Metrics saved to {metrics_path}")

    # Cleanup hf-mount
    if mount_path is not None:
        from hf_mount import unmount
        unmount(mount_path)

    return val_metrics


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import sys

    # If first arg is --mode ablation or --run, use ablation entry point
    if len(sys.argv) > 1 and ('--mode' in sys.argv or '--run' in sys.argv):
        from ablation import ALL_RUN_CONFIGS
        parser = argparse.ArgumentParser(
            description="Train one ablation run for CPU-1"
        )
        parser.add_argument('--mode', type=str, default='ablation',
                            choices=['train', 'ablation'])
        parser.add_argument('--run', required=True,
                            choices=list(ALL_RUN_CONFIGS.keys()),
                            help="Which run to train")
        parser.add_argument('--dry_run', action='store_true',
                            help="Use random data on CPU (for testing)")
        parser.add_argument('--total_steps', type=int, default=None,
                            help="Override total training steps")
        parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                            help="Training phase (1=distillation, 2=delete gate)")
        parser.add_argument('--phase1_ckpt', type=str, default=None,
                            help="Path to Phase 1 checkpoint (required for --phase 2)")
        parser.add_argument('--hf-mount', action='store_true',
                            help="Use hf-mount for lazy dataset loading (saves disk on Kaggle)")
        args = parser.parse_args()

        if args.phase == 2 and not args.phase1_ckpt:
            parser.error("--phase1_ckpt is required when --phase 2")

        metrics = train_run(args.run, dry_run=args.dry_run,
                            total_steps_override=args.total_steps,
                            phase=args.phase,
                            phase1_ckpt=args.phase1_ckpt,
                            use_hf_mount=args.hf_mount)
        print(f"\nFinished: {args.run}")
        print(json.dumps(metrics, indent=2))

    elif len(sys.argv) == 1:
        # No arguments -> run tests
        print("Sin argumentos -> corriendo tests\n")
        print("=" * 60)
        test_distillation_loss()
        print()
        test_param_groups()
        print()
        test_dataset_dry_run()
        print()
        test_lr_schedule()
        print("\n" + "=" * 60)
        print("Todos los tests pasaron.")
        print("\nPara entrenar: python train.py --dry-run --total-steps 10 --log-every 1")
    else:
        args = parse_args()
        os.makedirs(args.output_dir, exist_ok=True)
        train(args)
