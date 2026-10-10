# Native CPU kernel experiments

This folder contains custom AVX2/FMA/OpenMP kernels and a synthetic benchmark driver. The driver creates random weights and input vectors. It does not load the published checkpoints, and trained-model parity or end-to-end generation performance has not been established.

The local-decoder benchmark computes logits but feeds back bytes from a fixed formula, `(byte_position * 17) % 256`, rather than selecting or sampling those logits. Its rates measure synthetic kernel work, not meaningful text generation. See the [architecture guide](../docs/architecture-review.md#native-kernels-storage-and-reproducibility) and the separate [PyTorch model results](../README.md#results).

## References

- [Scalable MatMul-free Language Modeling](https://arxiv.org/abs/2406.02528): the MLGRU inspiration used by the token mixers.
- [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764): ternary weight quantization.
- [T-MAC](https://arxiv.org/abs/2407.00088): lookup-table computation for low-bit inference. These custom kernels are not a verified reproduction of T-MAC.
- [MEGABYTE](https://arxiv.org/abs/2305.07185): inspiration for the global/local byte decoder.
- [Were RNNs All We Needed?](https://arxiv.org/abs/2410.01201): related minLSTM/minGRU work; it is a different paper from the MLGRU reference above.

Reference implementations: [T-MAC](https://github.com/microsoft/T-MAC), [llama.cpp](https://github.com/ggml-org/llama.cpp), and [BitNet](https://github.com/microsoft/BitNet).

## Source and execution modes

- [kernels.cpp](kernels.cpp) implements ternary GEMV, normalization, activation, recurrent mixing, residual branches, global execution modes, and local decoding.
- [benchmark.c](benchmark.c) allocates synthetic tensors and calls those kernels. Despite its extension, the build recipe compiles it as C++ together with `kernels.cpp`.

| Mode | Driver call |
|---|---|
| 0 | Fused global pass using OpenMP |
| 1 | Per-layer calls |
| 2 | Single-threaded global pass |
| 3 | Multiple independent streams |
| 4 | Pipeline over a chunk of synthetic inputs |

`--local 1` additionally runs four local-decoder steps per patch. The examples select synthetic dimensions explicitly; they do not load a trained ablation checkpoint or establish native/PyTorch parity. No mode is established as universally fastest or optimal for a particular number of users.

## Build recipes

These recipes require an x86 CPU with AVX2/FMA and a compatible GCC/OpenMP toolchain. Compilation and runtime correctness were not validated in the reviewed environment. Run from the repository root in Bash, such as MSYS2 or WSL.

```bash
g++ -O3 -mavx2 -mfma -fopenmp -falign-loops=32 -funroll-loops -ffast-math \
    -std=c++11 -x c++ -o benchmark.exe \
    inference/benchmark.c inference/kernels.cpp -lm

# Pipeline example with explicit synthetic dimensions.
./benchmark.exe --d_model 512 --n_layers 12 --d_ff 1376 --fp_rank 4 --pfnet 32 \
    --mode 4 --patches 80 --warmup 30 \
    --chunk 128 --pin 1 --stages 6 --calibrate 1 --local 1

# Independent-stream example; its aggregate rate measures different work.
./benchmark.exe --d_model 512 --n_layers 12 --d_ff 1376 --fp_rank 4 --pfnet 32 \
    --mode 3 --batch 6 --threads 6 --patches 100 --warmup 30 --local 1
```

For a shared-library build with a compatible toolchain:

```bash
g++ -O3 -shared -fPIC -mavx2 -mfma -fopenmp -ffast-math \
    -o kernels.dll inference/kernels.cpp
```

The current Python model/evaluation packages do not load this library. Build artifacts are ignored by Git. SIMD floating-point approximations and `-ffast-math` make numerical parity a separate validation task.
