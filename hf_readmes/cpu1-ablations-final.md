---
language:
  - en
license: apache-2.0
tags:
  - pytorch
  - text-generation
  - 1.58bit
  - ternary
  - byte-level
  - mlgru
  - ablation
library_name: pytorch
pipeline_tag: text-generation
model_type: custom
---

# CPU-1 Ablation Study — Ready-to-use Checkpoints (fp32 unpacked)

Repo:    `Cukinator/cpu1-ablations-final`
Source:  [`Cukinator/cpu1-ablation-checkpoints`](https://huggingface.co/Cukinator/cpu1-ablation-checkpoints)
Code:    [github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits)

> **Tablas recalculadas 2026-06-13** — criterio homogéneo:
> `perplexity = exp(val_loss)` para todos los runs sin excepción.
> `BPB = val_loss / ln(2)` — solo byte-level.
> Referencia uniforme byte: **ln(256) = 5.545 nats → PPL=256 → BPB=8.000**
> BPE runs (run\_01, run\_13, \*\_r2): PPL es **token-level**.

## Ablation chain — 50M runs (round 1, 2 tok/param)

| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Δ uniform | P/s | Bytes/s | Nivel |
|-----|-------------|--------:|------------:|---------:|----:|----:|----------:|----:|--------:|-------|
| run_01 | Transformer + BPE 16K + FP16 | 54.7 | 218.9 | 4.6648 | 106.14 | — | — | 78.0 | 78 | token |
| run_02a_byte_only_heads | Transformer + Byte + 4 heads (no LBD) | 38.5 | 154.0 | 2.3079 | 10.05 | 3.3295 | -3.2373 | 85.9 | 344 | byte |
| run_02 | Transformer + Byte + LocalByteDecoder | 38.8 | 155.3 | 1.7154 | 5.56 | 2.4748 | -3.8298 | 86.1 | 344 | byte |
| run_03 | MLGRU + Byte + FP16 | 38.8 | 155.3 | 1.8697 | 6.49 | 2.6974 | -3.6755 | 59.9 | 240 | byte |
| run_04 | MLGRU + Byte + Ternary | 38.9 | 155.4 | 5.5671 | 261.68 | 8.0317 | 0.0220 | 8.0 | 32 | byte |
| run_05 | + FPResidual | 39.0 | 156.0 | 5.5516 | 257.65 | 8.0093 | 0.0064 | 7.6 | 30 | byte |
| run_05b_kernel_strict | MLGRU kernel-strict (no W_o) | 35.8 | 143.4 | 5.5940 | 268.82 | 8.0705 | 0.0489 | 8.7 | 35 | byte |
| run_06 | + Bolmo patch embedding | 39.0 | 156.0 | 5.5561 | 258.81 | 8.0157 | 0.0109 | 7.8 | 31 | byte |
| run_07 | + DeleteGate (CPU-1 completo) ⭐ | 39.0 | 156.1 | 5.5561 | 258.81 | 8.0157 | 0.0109 | 7.8 | 31 | byte |
| run_08 | Folded Transformer + Byte + Ternary | 38.9 | 155.4 | 5.5621 | 260.37 | 8.0244 | 0.0169 | 8.0 | 32 | byte |
| run_09 | + PFNet | 39.4 | 157.7 | 5.5330 | 252.91 | 7.9825 | -0.0122 | 7.3 | 29 | byte |
| run_10 | + decay per-canal aprendido | 39.4 | 157.7 | 5.5337 | 253.08 | 7.9835 | -0.0115 | 7.1 | 28 | byte |

## Small runs — 10M (round 1, 2 tok/param)

| Run | Arquitectura | Params M | RAM fp32 MB | Val Loss | PPL | BPB | Nivel |
|-----|-------------|--------:|------------:|---------:|----:|----:|-------|
| run_13 | Small CPU-1 + BPE 4K | 12.5 | 50.2 | 30.5375 | 18293109063475.55 | — | token |
| run_14 | Small CPU-1 byte (Qwen distill.) | 10.7 | 42.8 | 5.5755 | 263.88 | 8.0437 | byte |
| run_15 | Small CPU-1 byte + hidden distill. | 10.7 | 42.9 | 5.5755 | 263.88 | 8.0437 | byte |
| run_16 | Small CPU-1 raw bytes, no teacher | 10.7 | 42.8 | 5.5754 | 263.85 | 8.0436 | byte |

## Round 2 — re-runs a 15 tok/param (y r3 budget extendido)

| Run | Arquitectura | Params M | Loss r1 | PPL r1 | Loss r2 | PPL r2 | Δ loss | BPB r2 | Nivel |
|-----|-------------|--------:|--------:|-------:|--------:|-------:|-------:|-------:|-------|
| run_04_r2 | run_04 @ 15 tok/param | 38.9 | 5.5671 | 261.68 | 5.5894 | 267.57 | +0.02226 | 8.0638 | byte |
| run_07_r2 | run_07 @ 15 tok/param | 39.0 | 5.5561 | 258.81 | 5.5434 | 255.54 | -0.01271 | 7.9974 | byte |
| run_13_r2 | run_13 @ 15 tok/param | 12.5 | 30.5375 | 18293109063475.55 | 25.4264 | 110288596599.28 | -5.11118 | — | token |
| run_14_r2 | run_14 @ 15 tok/param | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | -0.00000 | 8.0437 | byte |
| run_15_r2 | run_15 @ 15 tok/param | 10.7 | 5.5755 | 263.88 | 5.5755 | 263.88 | -0.00000 | 8.0437 | byte |
| run_16_r2 | run_16 @ 15 tok/param | 10.7 | 5.5754 | 263.85 | 5.5754 | 263.85 | +0.00000 | 8.0436 | byte |

## Round 3 — v3 runs (bf16 AMP + lr×2 + 50 tok/p)

| Run | Arquitectura | Params M | Val Loss | PPL | BPB | Step | P/s | Bytes/s | Nivel |
|-----|-------------|--------:|--------:|----:|----:|-----:|----:|--------:|-------|
| run_01_v3 | run_01 v3 | 54.7 | nan | nan | — | 12173 | 196.0 | 196 | token |
| run_02_v3 | run_02 v3 | 38.8 | nan | nan | nan | 10963 | 91.4 | 366 | byte |
| run_02a_byte_only_heads_v3 | run_02a v3 | 38.5 | nan | nan | nan | 16327 | 96.0 | 384 | byte |
| run_03_v3 | run_03 v3 | 38.8 | 3.7841 | 43.99 | 5.4592 | 6187 | 65.5 | 262 | byte |
| run_04_v3 | run_04 v3 | 38.9 | 2.5988 | 13.45 | 3.7493 | 6074 | 9.5 | 38 | byte |
| run_05_v3 | run_05 v3 | 39.0 | 2.7034 | 14.93 | 3.9002 | 4390 | 7.0 | 28 | byte |
| run_05b_kernel_strict_v3 | run_05b v3 | 35.8 | 3.2144 | 24.89 | 4.6374 | 2573 | 8.3 | 33 | byte |
| run_06_v3 | run_06 v3 | 39.0 | 2.1493 | 8.58 | 3.1008 | 4237 | 7.0 | 28 | byte |
| run_08_v3 | run_08 v3 | 38.9 | nan | nan | nan | 3858 | 7.5 | 30 | byte |
| run_13_v3 | run_13 v3 | 12.5 | 5.4113 | 223.92 | — | 9290 | 19.6 | 20 | token |
| run_14_v3 | run_14 v3 | 10.7 | 2.2977 | 9.95 | 3.3149 | 9775 | 19.2 | 77 | byte |
| run_15_v3 | run_15 v3 | 10.7 | 2.8020 | 16.48 | 4.0424 | 2350 | 18.2 | 73 | byte |

**Hallazgo clave de V3:** El cuello de botella de los modelos ternarios en *cold-start* fue resuelto en la **v3** (usando bf16 AMP y `lr_scale=2.0` en BitLinear). Esto permitió romper el plateau de entropía uniforme (~5.55 nats) en el que caían las versiones anteriores, logrando convergencia estable y permitiendo que generen texto imprimible (`PASS` en los tests de generación).

## Licencia

Apache-2.0. Same as the source code at [github.com/Cukinator/1.58bits](https://github.com/Cukinator/1.58bits).
