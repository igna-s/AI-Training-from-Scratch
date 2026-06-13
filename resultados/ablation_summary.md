# Ablation Study — Evaluación completa

Generado: 2026-06-12 21:25
Criterio: PPL = exp(val_loss) | BPB = val_loss / ln(2) (byte-only)
Referencia uniforme byte: ln(256) = 5.545 nats → BPB = 8.000
Performance: PyTorch puro, sin kernels customizados (ablation mode).

## Correctness

| Run | fwd_shapes | probs | determinism | state_prop | ternary | state_size |
|-----|:----------:|:-----:|:-----------:|:----------:|:-------:|:----------:|
| run_01_v3 | FAIL | FAIL | FAIL | — | PASS | PASS |
| run_02_v3 | FAIL | FAIL | FAIL | — | PASS | PASS |
| run_02a_byte_only_heads_v3 | FAIL | FAIL | FAIL | — | PASS | PASS |
| run_03_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_04_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_05_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_05b_kernel_strict_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_06_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_08_v3 | FAIL | FAIL | FAIL | — | FAIL | PASS |
| run_13_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_14_v3 | PASS | PASS | PASS | — | PASS | PASS |
| run_15_v3 | PASS | PASS | PASS | — | PASS | PASS |

## Performance (PyTorch puro)

| Run | P/s | Bytes/s | p50 ms | p95 ms | CV | ctx_scaling | RAM Δ MB |
|-----|----:|-------:|-------:|-------:|---:|:-----------:|---------:|
| run_01_v3 | 74.7 | 75 | 13.4 | 17.4 | 0.1767 | PASS | 0.0 |
| run_02_v3 | 81.6 | 326 | 12.3 | 14.3 | 0.0938 | PASS | 0.0 |
| run_02a_byte_only_heads_v3 | 90.2 | 361 | 11.1 | 13.1 | 0.1075 | PASS | 0.0 |
| run_03_v3 | 64.1 | 256 | 15.6 | 19.4 | 0.1164 | PASS | 0.0 |
| run_04_v3 | 8.4 | 34 | 119.0 | 137.6 | 0.0776 | PASS | 0.0 |
| run_05_v3 | 8.2 | 33 | 121.7 | 133.8 | 0.0682 | PASS | 0.0 |
| run_05b_kernel_strict_v3 | 8.5 | 34 | 117.7 | 131.0 | 0.0709 | PASS | 0.0 |
| run_06_v3 | 8.1 | 32 | 123.6 | 145.0 | 0.0831 | PASS | 0.0 |
| run_08_v3 | 8.3 | 33 | 120.7 | 139.8 | 0.0758 | PASS | 0.0 |
| run_13_v3 | 22.5 | 22 | 44.4 | 58.3 | 0.1952 | PASS | -0.1 |
| run_14_v3 | 22.7 | 91 | 44.0 | 51.7 | 0.0859 | PASS | 0.0 |
| run_15_v3 | 21.8 | 87 | 45.9 | 54.9 | 0.1141 | PASS | 0.0 |

## Quality

| Run | Val Loss | PPL | BPB | Δ uniforme | gen% print | utf8% | gen status |
|-----|--------:|----:|----:|----------:|-----------:|------:|:----------:|
| run_01_v3 | — | — | — | — | — | — | N/A |
| run_02_v3 | — | — | — | — | — | — | ERROR |
| run_02a_byte_only_heads_v3 | — | — | — | — | — | — | ERROR |
| run_03_v3 | — | — | — | — | 0.997 | 0.933 | PASS |
| run_04_v3 | — | — | — | — | 1.000 | 1.000 | PASS |
| run_05_v3 | — | — | — | — | 0.996 | 1.000 | PASS |
| run_05b_kernel_strict_v3 | — | — | — | — | 0.996 | 1.000 | PASS |
| run_06_v3 | — | — | — | — | 0.959 | 0.833 | PASS |
| run_08_v3 | — | — | — | — | — | — | ERROR |
| run_13_v3 | — | — | — | — | — | — | N/A |
| run_14_v3 | — | — | — | — | 0.979 | 0.833 | PASS |
| run_15_v3 | — | — | — | — | 0.989 | 1.000 | PASS |
