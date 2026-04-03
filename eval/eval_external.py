"""
Eval — External: benchmark con llama.cpp (proceso externo).

No usa el modelo Python. Ejecuta llama-bench.exe con modelos GGUF.
Mantiene la comparativa con benchmarks oficiales de LiquidAI.
"""
import json
import os
import re
import subprocess
from pathlib import Path


N_PREFILL_TOKENS = 900
N_DECODE_TOKENS = 100
N_RUNS = 3
N_CTX = 1024


def _find_gguf_files(root: Path, pattern: str = "Q4_K_M") -> list[Path]:
    found = []
    models_dir = root / "models"
    if not models_dir.exists():
        return found
    for gguf in models_dir.rglob("*.gguf"):
        if pattern.upper() in gguf.name.upper():
            if not any(x in gguf.name.lower() for x in ("hip", "cuda", "rocm", "vulkan")):
                found.append(gguf)
    return found


def _find_llama_bench(root: Path) -> Path | None:
    candidates = [
        root / "llamacpp" / "build_optimized" / "bin" / "llama-bench.exe",
        root / "llamacpp" / "build_ninja" / "bin" / "llama-bench.exe",
        root / "llamacpp" / "build" / "bin" / "Release" / "llama-bench.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _run_bench(bench_exe: Path, model_path: Path, n_prompt: int, n_gen: int,
               n_threads: int, n_ctx: int, repetitions: int,
               flash_attn: bool = False, use_mmap: bool = True,
               batch_size: int | None = None,
               cache_type_k: str | None = None,
               cache_type_v: str | None = None,
               prio: int | None = None) -> dict | None:
    cmd = [
        str(bench_exe), "-m", str(model_path),
        "-p", str(n_prompt), "-n", str(n_gen),
        "-t", str(n_threads), "-r", str(repetitions), "-o", "json",
    ]
    if flash_attn:
        cmd.extend(["-fa", "1"])
    if not use_mmap:
        cmd.extend(["-mmp", "0"])
    if batch_size:
        cmd.extend(["-b", str(batch_size)])
    if cache_type_k:
        cmd.extend(["-ctk", cache_type_k])
    if cache_type_v:
        cmd.extend(["-ctv", cache_type_v])
    if prio is not None:
        cmd.extend(["--prio", str(prio)])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(bench_exe.parent),
        )
    except (subprocess.TimeoutExpired, Exception) as e:
        print(f"    [ERROR] {e}")
        return None

    if result.returncode != 0:
        return None

    output = result.stdout.strip()
    if not output:
        return None

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('[') or line.startswith('{'):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            return None

    if isinstance(data, list):
        prefill_tps = decode_tps = None
        for entry in data:
            if isinstance(entry, dict):
                if entry.get("n_prompt", 0) > 0 and entry.get("n_gen", 0) == 0:
                    prefill_tps = entry.get("avg_ts", 0)
                elif entry.get("n_gen", 0) > 0 and entry.get("n_prompt", 0) == 0:
                    decode_tps = entry.get("avg_ts", 0)
        return {'prefill_tps': prefill_tps or 0, 'decode_tps': decode_tps or 0}
    return None


def _print_comparison(label: str, prefill: float, decode: float):
    table = [
        ("AMD Ryzen AI Max 395+ (llama.cpp Q4)", 2900, 313),
        ("Apple M1 Max GPU (Mirai bf16)", 10400, 328),
        ("Raspberry Pi5 CPU (Cactus int8)", 200, 30),
    ]
    print(f"\n  {'Device':<42} {'Prefill':>8} {'Decode':>8}")
    print(f"  {'-' * 42} {'-' * 8} {'-' * 8}")
    for name, p, d in table:
        print(f"  {name:<42} {p:>7,} tok/s  {d:>6} tok/s")
    print(f"  {'-' * 42} {'-' * 8} {'-' * 8}")
    print(f"  {label:<42} {prefill:>7,.0f} tok/s  {decode:>6.0f} tok/s  << TU CPU")


def check_llama_cpp(gguf_path: str | None = None,
                    n_threads: int | None = None) -> dict:
    """Test 26 — Benchmark llama.cpp con GGUF."""
    root = Path(__file__).resolve().parent.parent
    bench_exe = _find_llama_bench(root)

    if bench_exe is None:
        raise AssertionError("llama-bench.exe no encontrado")

    if gguf_path:
        gguf_files = [Path(gguf_path)]
    else:
        gguf_files = _find_gguf_files(root)

    if not gguf_files:
        raise AssertionError("No se encontraron archivos GGUF en models/")

    configs = [
        {'name': 'ULTRA', 'threads': 1, 'use_mmap': False, 'flash_attn': True,
         'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0', 'prio': 3, 'batch_size': 2048},
        {'name': 'BASELINE', 'threads': 1, 'use_mmap': True, 'flash_attn': False,
         'cache_type_k': None, 'cache_type_v': None, 'prio': None, 'batch_size': None},
    ]

    if n_threads is not None:
        configs = [{'name': f'CUSTOM-{n_threads}T', 'threads': n_threads,
                    'use_mmap': False, 'flash_attn': False,
                    'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0',
                    'prio': 2, 'batch_size': 2048}]

    results = []
    for gf in gguf_files:
        label = re.sub(r'\.gguf$', '', gf.name)
        best_decode = 0
        best_result = None

        for cfg in configs:
            r = _run_bench(
                bench_exe, gf,
                n_prompt=N_PREFILL_TOKENS, n_gen=N_DECODE_TOKENS,
                n_threads=cfg['threads'], n_ctx=N_CTX, repetitions=N_RUNS,
                flash_attn=cfg.get('flash_attn', False),
                use_mmap=cfg.get('use_mmap', True),
                batch_size=cfg.get('batch_size'),
                cache_type_k=cfg.get('cache_type_k'),
                cache_type_v=cfg.get('cache_type_v'),
                prio=cfg.get('prio'),
            )
            if r and r['decode_tps'] > best_decode:
                best_decode = r['decode_tps']
                best_result = r

        if best_result:
            _print_comparison(label[:42], best_result['prefill_tps'], best_result['decode_tps'])
            results.append({
                'model': label,
                'prefill_tps': best_result['prefill_tps'],
                'decode_tps': best_result['decode_tps'],
            })

    if not results:
        raise AssertionError("Todas las configs fallaron")

    best = max(r['decode_tps'] for r in results)
    assert best > 20, f"Decode TPS muy bajo: {best:.1f}"
    return {'models': results, 'best_decode_tps': round(best, 1)}


# =========================================================================
# Entry point
# =========================================================================

def run_external(gguf_path: str | None = None,
                 n_threads: int | None = None) -> dict:
    """Ejecutar benchmarks externos."""
    from helpers import run_check

    print("\n--- External ---")
    results = {}
    r = run_check('llama_cpp', check_llama_cpp, gguf_path, n_threads)
    results['llama_cpp'] = r
    print(f"  {r['status']:5s}  llama_cpp")
    return results
