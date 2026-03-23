"""
Eval de rendimiento CPU — arquitectura 170M params, inferencia real.

Pre-congela los pesos ternarios una sola vez antes del loop para simular
inferencia real (sin STE overhead de entrenamiento).

Mide throughput procesando 40000 bytes (10000 patches de 4) con pesos aleatorios.
No importa que la salida sea basura: el objetivo es medir tiempo.

Uso:
    python eval_cpu.py
"""

import time
import torch
import torch.nn.functional as F
from model_v2 import CPU1ModelV2, BitLinear, PATCH_SIZE, count_parameters


N_PATCHES = 10_000   # 10000 patches × 4 bytes = 40000 bytes
WARMUP    = 20


def freeze_bitlinear_for_inference(model):
    """
    Pre-computa w_ternary y w_scale para cada BitLinear y reemplaza
    forward() con una version que usa los valores pre-computados.
    Elimina el overhead STE que existe en el forward de entrenamiento.
    """
    for module in model.modules():
        if not isinstance(module, BitLinear):
            continue
        with torch.no_grad():
            w_scale = module.weight.data.abs().mean()
            w_ternary = (module.weight.data / (w_scale + 1e-8)).round().clamp(-1.0, 1.0)
        # Guardar como buffers (no parametros, no gradientes)
        module.register_buffer('w_ternary_frozen', w_ternary)
        module.register_buffer('w_scale_frozen', w_scale.unsqueeze(0))

        # Reemplazar forward con version sin recomputacion
        def _inference_forward(x, mod=module):
            x_scale = x.abs().amax(dim=-1, keepdim=True)
            x_quant = (x / (x_scale + 1e-8) * 127.0).round().clamp(-128.0, 127.0)
            out = F.linear(x_quant, mod.w_ternary_frozen)
            return out * (x_scale * mod.w_scale_frozen / 127.0)

        module.forward = _inference_forward


def main():
    print("=" * 60)
    print("CPU-1 v2 — Eval inferencia real (pesos ternarios pre-congelados)")
    print("=" * 60)

    print("\nCargando modelo 170M...", flush=True)
    t0 = time.perf_counter()
    model = CPU1ModelV2()
    model.eval()
    t_load = time.perf_counter() - t0
    print(f"  Parametros: {count_parameters(model)/1e6:.1f}M  |  Init: {t_load:.2f}s", flush=True)

    print("Congelando pesos ternarios para inferencia...", flush=True)
    freeze_bitlinear_for_inference(model)
    print("  OK", flush=True)

    patch = torch.randint(0, 256, (1, 1, PATCH_SIZE), dtype=torch.long)
    hs = [None] * len(model.blocks)

    print(f"\nConfig:", flush=True)
    print(f"  Total: {N_PATCHES} patches = {N_PATCHES * PATCH_SIZE:,} bytes", flush=True)
    print(f"  Warmup: {WARMUP} patches", flush=True)

    print(f"\nWarmup...", flush=True)
    with torch.no_grad():
        for _ in range(WARMUP):
            _, hs = model(patch, hs)

    hs = [None] * len(model.blocks)
    latencies = []

    print(f"Midiendo {N_PATCHES} patches...", flush=True)
    with torch.no_grad():
        for i in range(N_PATCHES):
            t0 = time.perf_counter()
            _, hs = model(patch, hs)
            latencies.append(time.perf_counter() - t0)
            if (i + 1) % 1000 == 0:
                print(f"  {i+1}/{N_PATCHES}  ({latencies[-1]*1000:.1f} ms/patch)", flush=True)

    lat_s = sorted(latencies)
    n = len(lat_s)
    total_s = sum(latencies)
    patches_sec = N_PATCHES / total_s
    bytes_sec = N_PATCHES * PATCH_SIZE / total_s

    print("\n" + "=" * 60)
    print("RESULTADOS")
    print("=" * 60)
    print(f"  Total procesado:  {N_PATCHES * PATCH_SIZE:,} bytes ({N_PATCHES:,} patches)")
    print(f"  Tiempo total:     {total_s:.1f}s ({total_s/60:.1f} min)")
    print(f"  Throughput:       {patches_sec:.2f} patches/s  |  {bytes_sec:.1f} bytes/s")
    print(f"  Latencia media:   {total_s/N_PATCHES*1000:.1f} ms/patch")
    print(f"  Latencia p50:     {lat_s[n//2]*1000:.1f} ms")
    print(f"  Latencia p95:     {lat_s[int(n*0.95)]*1000:.1f} ms")
    print(f"  Latencia p99:     {lat_s[int(n*0.99)]*1000:.1f} ms")
    print("=" * 60)


if __name__ == "__main__":
    main()
