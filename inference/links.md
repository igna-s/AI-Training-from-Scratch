# CPU-1 Inference — Links & References

## Core Papers
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) — Ternary weight quantization
- [T-MAC: CPU-Based LUT-Based Inference](https://arxiv.org/abs/2407.00088) — LUT approach for ternary GEMV
- [minLSTM / MLGRU](https://arxiv.org/abs/2410.01201) — Minimal recurrent architecture used in CPU-1
- [MegaByte (Yu et al., 2023)](https://arxiv.org/abs/2305.07185) — Two-tier byte-level architecture used by `local_decode_step`

## Implementation References
- [T-MAC GitHub](https://github.com/microsoft/T-MAC) — LUT kernel reference implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGML quantized inference reference
- [BitNet.cpp](https://github.com/microsoft/BitNet) — Microsoft's official BitNet inference

## Source files in this folder

- `kernels.cpp` — All AVX2 + FMA inference kernels (GEMV, RMSNorm, SiLU, MLGRU,
  PFNet, FPResidual, the `forward_full_mad_*` family, and `local_decode_step`).
  Compiled inline via `g++ -x c++` from the benchmark.
- `benchmark.c` — Standalone driver. Allocates random ternary weights and times
  the global pass + MegaByte LocalByteDecoder per patch across modes 0–4.
- `links.md` — this file.

Build artifacts (`benchmark.exe`, `*.dll`, `*.o`) are gitignored.

## Build & run

```bash
# Compile (Windows MSYS / Linux / WSL):
g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math \
    -std=c++11 -x c++ -o benchmark.exe \
    inference/benchmark.c inference/kernels.cpp -lm

# Defaults: CPU-1 170M (d_model=1024, n_layers=14, d_ff=2688), full patch
# (1 global pass + 4 LocalByteDecoder steps per patch).

# Single-stream pipeline (mode 4, recommended for 1 active user):
./benchmark.exe --mode 4 --patches 80 --warmup 30 \
                --chunk 128 --pin 1 --stages 6 --calibrate 1 --local 1

# Multi-stream batch (mode 3, max aggregate throughput):
./benchmark.exe --mode 3 --batch 6 --threads 6 --patches 100 --warmup 30 --local 1

# 400M kernel-optimal config (24 layers, perfect 4-per-stage split):
./benchmark.exe --d_model 1280 --n_layers 24 --d_ff 3328 --fp_rank 8 \
                --mode 4 --patches 30 --warmup 10 \
                --chunk 64 --pin 1 --stages 6 --calibrate 1 --local 1
```

See the main `README.md` (`CPU-1 170M Performance — Current C Benchmark Status`)
for full mode descriptions and benchmark tables.

## Optional: build as a shared library

If you want to call the kernels from Python via `ctypes`:

```bash
g++ -O3 -shared -fPIC -mavx2 -mfma -fopenmp -ffast-math \
    -o kernels.dll inference/kernels.cpp
```

Currently no Python code in this repo loads the DLL; the C benchmark is the
authoritative consumer.
