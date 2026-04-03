"""
Test 22 — Eval CPU Raw (Ternary Weights Frozen)

Mide throughput de inferencia del modelo con pesos ternarios pre-congelados.
Simula inferencia real sin overhead STE del entrenamiento.

Metrica: patches/seg, bytes/seg
Criterio: > 50 patches/seg (mismo umbral que test_06 en modo optimizado)
"""
import time
import torch
import torch.nn.functional as F


def _freeze_bitlinear_for_inference(model):
    """Pre-computa w_ternary y w_scale para cada BitLinear."""
    try:
        from model import BitLinear
    except ImportError:
        return  # Si no hay BitLinear, no hacer nada

    for module in model.modules():
        if not isinstance(module, BitLinear):
            continue
        with torch.no_grad():
            w_scale = module.weight.data.abs().mean()
            w_ternary = (module.weight.data / (w_scale + 1e-8)).round().clamp(-1.0, 1.0)
        module.register_buffer('w_ternary_frozen', w_ternary)
        module.register_buffer('w_scale_frozen', w_scale.unsqueeze(0))

        def _inference_forward(x, mod=module):
            x_scale = x.abs().amax(dim=-1, keepdim=True)
            x_quant = (x / (x_scale + 1e-8) * 127.0).round().clamp(-128.0, 127.0)
            out = F.linear(x_quant, mod.w_ternary_frozen)
            return out * (x_scale * mod.w_scale_frozen / 127.0)

        module.forward = _inference_forward


def test_eval_cpu_raw(model, n_patches: int = 500, warmup: int = 20):
    """
    Mide throughput con pesos ternarios congelados.
    n_patches=500 por defecto para ser rápido en el test suite.
    """
    _freeze_bitlinear_for_inference(model)
    model.eval()
    model.cpu()

    try:
        from model import PATCH_SIZE
    except ImportError:
        PATCH_SIZE = 4

    patch = torch.randint(0, 256, (1, 1, PATCH_SIZE), dtype=torch.long)
    hs = [None] * len(model.blocks) if hasattr(model, 'blocks') else None

    with torch.no_grad():
        for _ in range(warmup):
            _, hs = model(patch, hs)

    # Reset state
    hs = [None] * len(model.blocks) if hasattr(model, 'blocks') else None
    latencies = []

    t_total = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_patches):
            t0 = time.perf_counter()
            _, hs = model(patch, hs)
            latencies.append(time.perf_counter() - t0)
    total_s = time.perf_counter() - t_total

    patches_sec = n_patches / total_s
    bytes_sec = n_patches * PATCH_SIZE / total_s
    lat_s = sorted(latencies)
    n = len(lat_s)
    p50 = lat_s[n // 2] * 1000

    print(f"[Test 22] CPU raw (ternary frozen): {patches_sec:.1f} patches/s | p50={p50:.1f}ms")
    assert patches_sec > 50, f"Throughput muy bajo: {patches_sec:.1f} patches/s"

    return {
        'patches_sec': round(patches_sec, 1),
        'bytes_sec': round(bytes_sec, 1),
        'p50_ms': round(p50, 2),
    }


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--n-patches', type=int, default=2000)
    args = parser.parse_args()
    model = load_model(args.model)
    test_eval_cpu_raw(model, args.n_patches)
