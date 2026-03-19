"""
Training loop para CPU-1 con distillation loss.
Diseñado para Kaggle 2×T4 con DDP, compatible con single-GPU y CPU (dry-run).

Uso:
    # Dry-run (testing sin GPU)
    python train.py --dry-run --total-steps 10 --log-every 1

    # Single GPU con datos reales
    python train.py --data-path ./distillation_dataset --total-steps 100000

    # 2×T4 con DDP (Kaggle)
    torchrun --nproc_per_node=2 train.py --data-path ./distillation_dataset
"""

import os
import math
import json
import time
import contextlib
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor

from model import CPU1Model, D_MODEL, VOCAB_SIZE, PATCH_SIZE
from model_v2 import CPU1ModelV2


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

    if teacher_probs is not None and teacher_mask.any():
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
        """Carga desde directorio HF o Parquet."""
        from datasets import load_from_disk, load_dataset
        p = Path(data_path)
        if p.is_dir() and (p / 'dataset_info.json').exists():
            self.dataset = load_from_disk(data_path)
        elif p.suffix == '.parquet':
            self.dataset = load_dataset('parquet', data_files=str(p), split='train')
        else:
            # Intentar como repo de HuggingFace Hub
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

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
        }

    def _get_hf_item(self, idx: int) -> dict[str, Tensor]:
        """Carga sample desde HuggingFace dataset.

        Soporta tanto arrays nativos (HF Dataset con features tipadas) como
        JSON strings (formato de byte_marginalization.py en Parquet crudo).
        """
        item = self.dataset[idx]

        raw_patches = item['patches']
        if isinstance(raw_patches, str):
            raw_patches = json.loads(raw_patches)
        patches = torch.as_tensor(raw_patches, dtype=torch.long)

        raw_targets = item['targets']
        if isinstance(raw_targets, str):
            raw_targets = json.loads(raw_targets)
        targets = torch.as_tensor(raw_targets, dtype=torch.long)

        if 'teacher_probs' in item and item['teacher_probs'] is not None:
            raw_tp = item['teacher_probs']
            if isinstance(raw_tp, str):
                raw_tp = json.loads(raw_tp)
            teacher_probs = torch.as_tensor(raw_tp, dtype=torch.float32)
        else:
            teacher_probs = torch.zeros(patches.shape[0], VOCAB_SIZE)

        if 'teacher_mask' in item and item['teacher_mask'] is not None:
            raw_tm = item['teacher_mask']
            if isinstance(raw_tm, str):
                raw_tm = json.loads(raw_tm)
            teacher_mask = torch.as_tensor(raw_tm, dtype=torch.bool)
        else:
            teacher_mask = torch.zeros(patches.shape[0], dtype=torch.bool)

        return {
            'patches': patches,
            'targets': targets,
            'teacher_probs': teacher_probs,
            'teacher_mask': teacher_mask,
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
    Separa parámetros en 3 grupos con diferente weight decay.

    - BitLinear weights + embedding projection: wd=0.1
    - FPResidual, RMSNorm, biases, byte_embedding, BitEmbedding scale: wd=0.0
    - Head weights (nn.Linear o BitHead): wd=0.01

    Compatible con v1 (CPU1Model) y v2 (CPU1ModelV2).
    """
    decay_params = []       # BitLinear weights, embedding proj
    no_decay_params = []    # FPResidual, norms, biases, byte embedding
    head_params = []        # prediction heads

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # v1 usa head_0..head_3, v2 usa byte_heads.0..byte_heads.3
        if ('head_' in name or 'byte_heads' in name) and 'weight' in name:
            head_params.append(param)
        elif any(k in name for k in ('fp_A', 'fp_B', 'norm', 'bias',
                                      'byte_embedding', 'byte_embed',
                                      'output_scale', '.scale')):
            # FPResidual, RMSNorm, biases, lookup/BitEmbedding, scales — sin decay
            no_decay_params.append(param)
        else:
            # BitLinear weights (q/k/v/o_proj, gate/up/down_proj, forget/cand/output_proj)
            # + embedding.proj / patch_proj
            decay_params.append(param)

    return [
        {'params': decay_params, 'weight_decay': 0.1},
        {'params': no_decay_params, 'weight_decay': 0.0},
        {'params': head_params, 'weight_decay': 0.01},
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
        model:        modelo (puede ser DDP wrapper o modelo directo)
        optimizer:    estado del optimizador (momentum, etc.)
        scaler:       estado del GradScaler (escala FP16 actual)
        step:         step de training actual
        loss:         ultimo loss registrado
        path:         ruta local donde guardar el .pt
        upload_to_hf: si True, sube el .pt al repo de HuggingFace
        hf_repo:      repo HF destino (default: Cukinator/cpu1-checkpoints)

    Nota: guarda model/optimizer/scaler state_dicts para poder resumir training.
    Si el modelo esta wrapeado en DDP, extrae model.module para guardar sin el wrapper.
    """
    # .module unwraps DDP para obtener el modelo real
    state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
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
        model:     modelo destino (puede ser DDP wrapper)
        optimizer: si se pasa, restaura el estado del optimizador (momentum, etc.)
        scaler:    si se pasa, restaura el estado del GradScaler
    Returns:
        step (int) desde el cual resumir el training
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scaler and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint.get('step', 0)


# =============================================================================
# 6. DDP Setup
# =============================================================================

def setup_ddp() -> tuple[int, int]:
    """
    Inicializa DDP para 2xT4 en Kaggle (PCIe only, sin NVLink).

    Returns:
        (local_rank, world_size): indice de GPU local y cantidad total de GPUs

    Las variables de entorno NCCL desactivan P2P e InfiniBand porque las T4
    de Kaggle estan conectadas por PCIe, no por NVLink. Sin esto, NCCL intenta
    usar P2P que falla silenciosamente y mata el rendimiento.
    """
    os.environ.setdefault("NCCL_P2P_DISABLE", "1")    # Sin peer-to-peer GPU directo
    os.environ.setdefault("NCCL_IB_DISABLE", "1")     # Sin InfiniBand
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo") # Usar loopback para comunicacion

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])  # Seteado por torchrun
    torch.cuda.set_device(local_rank)

    return local_rank, dist.get_world_size()


# =============================================================================
# 7. Training Loop
# =============================================================================

def train(args: argparse.Namespace):
    """Training loop principal."""
    # --- Setup dispositivo ---
    is_ddp = 'LOCAL_RANK' in os.environ
    if is_ddp:
        local_rank, world_size = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        world_size = 1
        is_main = True

    if is_main:
        print(f"Device: {device} | World size: {world_size}")

    # --- Modelo ---
    if args.model_version == 'v2':
        model = CPU1ModelV2({
            'use_gradient_checkpointing': args.gradient_checkpointing,
        }).to(device)
        model_is_v2 = True
    else:
        model = CPU1Model(
            use_gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
        model_is_v2 = False

    if is_ddp:
        model = DDP(model, device_ids=[local_rank])

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parámetros: {n_params:,}")

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
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

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

    sampler = DistributedSampler(dataset, shuffle=True) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=use_amp,
        drop_last=True,
    )

    # --- Resume ---
    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, scaler)
        if is_main:
            print(f"Resumiendo desde step {start_step}")

    # --- Config ---
    grad_accum = args.gradient_accumulation_steps
    total_steps = args.total_steps
    effective_batch = args.batch_per_gpu * grad_accum * world_size

    if is_main:
        print(f"Batch efectivo: {effective_batch} "
              f"({args.batch_per_gpu} × {grad_accum} accum × {world_size} GPUs)")
        print(f"Steps: {start_step} ->{total_steps}")
        print(f"Seq len: {args.seq_len} patches = {args.seq_len * PATCH_SIZE} bytes")

    # --- Training loop ---
    model.train()
    data_iter = iter(dataloader)
    epoch = 0
    log_timer = time.time()

    for step in range(start_step, total_steps):
        # LR schedule
        lr = get_lr(step, args.warmup_steps, total_steps, args.max_lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad()
        should_log = (step % args.log_every == 0) and is_main

        for micro_step in range(grad_accum):
            # Obtener batch (reciclar dataloader al agotarse)
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if is_ddp:
                    sampler.set_epoch(epoch)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            patches = batch['patches'].to(device, non_blocking=True)
            targets = batch['targets'].to(device, non_blocking=True)
            teacher_probs = batch['teacher_probs'].to(device, non_blocking=True)
            teacher_mask = batch['teacher_mask'].to(device, non_blocking=True)

            # no_sync() evita sincronizar gradientes entre GPUs en micro-steps intermedios.
            # Solo en el ultimo micro-step se hace all-reduce (suma gradientes de ambas GPUs).
            # Esto ahorra ancho de banda PCIe, critico sin NVLink.
            is_last_micro = (micro_step == grad_accum - 1)
            sync_ctx = contextlib.nullcontext() if (is_last_micro or not is_ddp) else model.no_sync()

            with sync_ctx:
                amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    if model_is_v2:
                        logits, _ = model(patches)  # v2: descartar hidden_states
                    else:
                        logits = model(patches)  # v1: tuple de 4 logits
                    loss, metrics = distillation_loss(
                        logits, targets, teacher_probs, teacher_mask,
                        temperature=args.temperature,
                        compute_metrics=(should_log and is_last_micro),
                    )
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
            patches_per_sec = (args.batch_per_gpu * args.seq_len * grad_accum * steps_since_log) / max(elapsed, 1e-6)

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
        if step > 0 and step % args.save_every == 0 and is_main:
            ckpt_path = str(Path(args.output_dir) / f"checkpoint_{step}.pt")
            save_checkpoint(
                model, optimizer, scaler, step,
                metrics.get('loss_total', 0.0),
                ckpt_path,
                upload_to_hf=args.upload_hf,
            )
            print(f"  Checkpoint: {ckpt_path}")

    # --- Final ---
    if is_main:
        ckpt_path = str(Path(args.output_dir) / f"checkpoint_final.pt")
        save_checkpoint(
            model, optimizer, scaler, total_steps,
            metrics.get('loss_total', 0.0),
            ckpt_path,
            upload_to_hf=args.upload_hf,
        )
        print(f"Training completo. Final: {ckpt_path}")

    if is_ddp:
        dist.destroy_process_group()


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
    p.add_argument('--seq-len', type=int, default=2048,
                   help="Patches por secuencia")

    # Model
    p.add_argument('--model-version', type=str, default='v1',
                   choices=['v1', 'v2'],
                   help="v1=CPU1Model (attention), v2=CPU1ModelV2 (MLGRU)")

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

    # Workers
    p.add_argument('--num-workers', type=int, default=2)

    return p.parse_args()


# =============================================================================
# 9. Tests
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
        ("v1", CPU1Model(n_layers=2)),
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


if __name__ == "__main__":
    # Si no hay argumentos de CLI, correr tests
    import sys
    if len(sys.argv) == 1:
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
