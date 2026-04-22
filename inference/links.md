# CPU-1 Inference — Links & References

## Core Papers
- [The Era of 1-bit LLMs (BitNet b1.58)](https://arxiv.org/abs/2402.17764) — Ternary weight quantization
- [T-MAC: CPU-Based LUT-Based Inference](https://arxiv.org/abs/2407.00088) — LUT approach for ternary GEMV
- [minLSTM / MLGRU](https://arxiv.org/abs/2410.01201) — Minimal recurrent architecture used in CPU-1

## Implementation References
- [T-MAC GitHub](https://github.com/microsoft/T-MAC) — LUT kernel reference implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) — GGML quantized inference reference
- [BitNet.cpp](https://github.com/microsoft/BitNet) — Microsoft's official BitNet inference

## Architecture
- **Model**: CPU-1 170M (d_model=1024, n_layers=14, d_ff=2688)
- **Weights**: 2-bit MAD (Mixed Activation Data) column-interleaved packing
- **Kernel**: AVX2+FMA GEMV with vpshufb LUT decode
- **Forward**: Single C call per patch via `forward_full_mad`

## Build

```bash
# Compile shared library (for external use)
g++ -O3 -shared -fPIC -mavx2 -mfma -fopenmp -ffast-math -o kernels.dll kernels.cpp

# Compile standalone benchmark
g++ -O3 -mavx2 -mfma -fopenmp -ffast-math -x c++ -o benchmark.exe benchmark.c kernels.cpp -lm

# Run benchmark
./benchmark.exe --patches 200 --warmup 20 --mode 0
```
