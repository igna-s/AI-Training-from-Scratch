"""
Test 26 — Benchmark llama.cpp (GGUF Q4_K_M) — OPTIMIZED

Usa el binario nativo llama-bench.exe directamente (sin wrapper Python).
Aplica TODAS las optimizaciones de CPU:
  - mmap OFF:    carga directa en RAM (evita page-faults lazy loading)
  - flash-attn:  flash attention para CPU (reduce memoria KV)
  - threads=1:   1 thread para decode (memory-bandwidth-bound)
  - prio=3:      prioridad REALTIME del proceso
  - KV cache q8:  reduce tamaño del KV cache

Metodologia identica al benchmark oficial de LiquidAI:
  - Prefill: 900 tokens  (modelo tiene ctx_train=1024)
  - Decode:  100 tokens
  - Reporta: prefill tok/s y decode tok/s por separado

Referencia oficial (CPU llama.cpp Q4):
  AMD Ryzen AI Max 395+  → prefill 2.9K tok/s | decode 313 tok/s
"""
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# --- Parametros identicos a la tabla oficial ---
N_PREFILL_TOKENS = 900   # LFM2-350M tiene ctx_train=1024; max prefill = 1024-100-24
N_DECODE_TOKENS  = 100
N_RUNS           = 3
N_CTX            = 1024   # Respetar ctx_train del modelo


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
    """Busca llama-bench.exe en los directorios de build."""
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
    """
    Ejecuta llama-bench.exe y parsea los resultados JSON.
    Devuelve dict con prefill_tps, decode_tps, o None si falla.
    """
    cmd = [
        str(bench_exe),
        "-m", str(model_path),
        "-p", str(n_prompt),
        "-n", str(n_gen),
        "-t", str(n_threads),
        "-r", str(repetitions),
        "-o", "json",
    ]

    # Flash attention
    if flash_attn:
        cmd.extend(["-fa", "1"])

    # mmap OFF = carga directa
    if not use_mmap:
        cmd.extend(["-mmp", "0"])

    # Batch size
    if batch_size:
        cmd.extend(["-b", str(batch_size)])

    # KV cache types (q8_0 is smaller, faster)
    if cache_type_k:
        cmd.extend(["-ctk", cache_type_k])
    if cache_type_v:
        cmd.extend(["-ctv", cache_type_v])

    # Process priority (3 = realtime)
    if prio is not None:
        cmd.extend(["--prio", str(prio)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(bench_exe.parent),
        )
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] Benchmark timed out after 300s")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    if result.returncode != 0:
        stderr = result.stderr or ""
        print(f"  [WARN] llama-bench exit code {result.returncode}")
        if stderr:
            first_line = stderr.strip().split('\n')[0]
            print(f"  [STDERR] {first_line[:120]}")
        return None

    # Parsear JSON output
    output = result.stdout.strip()
    if not output:
        print("  [WARN] No output from llama-bench")
        return None

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        # Intentar linea por linea
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('[') or line.startswith('{'):
                try:
                    data = json.loads(line)
                    break
                except json.JSONDecodeError:
                    continue
        else:
            print(f"  [WARN] Could not parse JSON output")
            return None

    # Extraer resultados
    if isinstance(data, list):
        prefill_tps = None
        decode_tps = None
        for entry in data:
            if isinstance(entry, dict):
                n_prompt_val = entry.get("n_prompt", 0)
                n_gen_val = entry.get("n_gen", 0)
                avg_ts = entry.get("avg_ts", 0)
                if n_prompt_val > 0 and n_gen_val == 0:
                    prefill_tps = avg_ts
                elif n_gen_val > 0 and n_prompt_val == 0:
                    decode_tps = avg_ts
        return {
            'prefill_tps': prefill_tps or 0,
            'decode_tps': decode_tps or 0,
        }
    return None


def _print_comparison(label: str, prefill: float, decode: float):
    table = [
        ("AMD Ryzen AI Max 395+ (llama.cpp Q4)", 2900, 313),
        ("Apple M1 Max GPU (Mirai bf16)",        10400, 328),
        ("Apple M5 Max GPU (Mirai bf16)",        44800, 564),
        ("Apple A18 Pro GPU (Mirai bf16)",        1953,  73),
        ("Qualcomm SD8 Elite NPU (RunAnywhere)",  2800,  15),
        ("Google Pixel 6a CPU (Cactus int8)",      208,  42),
        ("Raspberry Pi5 CPU (Cactus int8)",        200,  30),
    ]

    print()
    print("=" * 70)
    print("  COMPARACION vs benchmark oficial LiquidAI  (1K prefill / 100 decode)")
    print("=" * 70)
    print(f"  {'Device':<42} {'Prefill':>8} {'Decode':>8}")
    print(f"  {'-'*42} {'-'*8} {'-'*8}")
    for name, p, d in table:
        print(f"  {name:<42} {p:>7,} {'tok/s':>4}  {d:>6} tok/s")
    print(f"  {'-'*42} {'-'*8} {'-'*8}")
    print(f"  {label:<42} {prefill:>7,.0f} {'tok/s':>4}  {decode:>6.0f} tok/s  << TU i7")
    print("=" * 70)

    vs_rpi_decode = decode / 30
    print(f"\n  vs Raspberry Pi5 decode: {vs_rpi_decode:.1f}x {'mas rapido' if vs_rpi_decode >= 1 else 'mas lento'}")
    vs_ryzen_decode = decode / 313
    print(f"  vs Ryzen AI Max 395+:    {vs_ryzen_decode:.2f}x ({decode:.0f} / 313)")
    print()


def test_benchmark_llama_cpp(gguf_path: str | None = None,
                               n_prefill: int = N_PREFILL_TOKENS,
                               n_decode: int = N_DECODE_TOKENS,
                               n_runs: int = N_RUNS,
                               n_threads: int | None = None):
    import psutil
    n_physical = psutil.cpu_count(logical=False) or 4

    root = Path(__file__).resolve().parent.parent

    # Buscar binario nativo
    bench_exe = _find_llama_bench(root)
    if bench_exe is None:
        raise AssertionError(
            "No se encontro llama-bench.exe. "
            "Compilar con: cmake -G Ninja -B build_optimized ..."
        )
    print(f"\n[Test 26] Usando binario nativo: {bench_exe}")

    # Buscar GGUF
    if gguf_path:
        gguf_files = [Path(gguf_path)]
    else:
        gguf_files = _find_gguf_files(root, "Q4_K_M")

    if not gguf_files:
        raise AssertionError(
            "No se encontraron archivos Q4_K_M GGUF en models/."
        )

    # ==========================================================
    # CONFIGURACIONES DE TEST — de más agresiva a conservadora
    # ==========================================================
    configs = [
        {
            'name': 'ULTRA (1T, no-mmap, flash-attn, q8_0 KV, REALTIME)',
            'threads': 1,
            'use_mmap': False,
            'flash_attn': True,
            'cache_type_k': 'q8_0',
            'cache_type_v': 'q8_0',
            'prio': 3,
            'batch_size': 2048,
        },
        {
            'name': 'OPTIMAL (1T, no-mmap, q8_0 KV, HIGH prio)',
            'threads': 1,
            'use_mmap': False,
            'flash_attn': False,
            'cache_type_k': 'q8_0',
            'cache_type_v': 'q8_0',
            'prio': 2,
            'batch_size': 2048,
        },
        {
            'name': 'BASELINE (1T, defaults)',
            'threads': 1,
            'use_mmap': True,
            'flash_attn': False,
            'cache_type_k': None,
            'cache_type_v': None,
            'prio': None,
            'batch_size': None,
        },
    ]

    if n_threads is not None:
        configs = [{
            'name': f'CUSTOM ({n_threads}T)',
            'threads': n_threads,
            'use_mmap': False,
            'flash_attn': False,
            'cache_type_k': 'q8_0',
            'cache_type_v': 'q8_0',
            'prio': 2,
            'batch_size': 2048,
        }]

    results = []
    for gf in gguf_files:
        label = re.sub(r'\.gguf$', '', gf.name)
        print(f"\n{'='*60}")
        print(f"[Test 26] Modelo: {gf.name}")
        print(f"{'='*60}")

        best_decode = 0
        best_config = None
        best_result = None

        for cfg in configs:
            print(f"\n--- Config: {cfg['name']} ---")
            r = _run_bench(
                bench_exe, gf,
                n_prompt=n_prefill,
                n_gen=n_decode,
                n_threads=cfg['threads'],
                n_ctx=N_CTX,
                repetitions=n_runs,
                flash_attn=cfg.get('flash_attn', False),
                use_mmap=cfg.get('use_mmap', True),
                batch_size=cfg.get('batch_size'),
                cache_type_k=cfg.get('cache_type_k'),
                cache_type_v=cfg.get('cache_type_v'),
                prio=cfg.get('prio'),
            )

            if r is None and cfg.get('flash_attn'):
                # Reintentar sin flash-attn
                print("  Reintentando sin flash-attn...")
                r = _run_bench(
                    bench_exe, gf,
                    n_prompt=n_prefill,
                    n_gen=n_decode,
                    n_threads=cfg['threads'],
                    n_ctx=N_CTX,
                    repetitions=n_runs,
                    flash_attn=False,
                    use_mmap=cfg.get('use_mmap', True),
                    batch_size=cfg.get('batch_size'),
                    cache_type_k=cfg.get('cache_type_k'),
                    cache_type_v=cfg.get('cache_type_v'),
                    prio=cfg.get('prio'),
                )

            if r is None:
                # Modo minimo
                print("  Reintentando modo basico...")
                r = _run_bench(
                    bench_exe, gf,
                    n_prompt=n_prefill,
                    n_gen=n_decode,
                    n_threads=cfg['threads'],
                    n_ctx=N_CTX,
                    repetitions=n_runs,
                )

            if r:
                print(f"  Prefill: {r['prefill_tps']:,.1f} tok/s")
                print(f"  Decode:  {r['decode_tps']:.1f} tok/s")
                if r['decode_tps'] > best_decode:
                    best_decode = r['decode_tps']
                    best_config = cfg['name']
                    best_result = r
            else:
                print(f"  [SKIP] Config no soportada")

        if best_result:
            print(f"\n[Test 26] MEJOR CONFIG: {best_config}")
            print(f"          Prefill: {best_result['prefill_tps']:,.1f} tok/s")
            print(f"          Decode:  {best_result['decode_tps']:.1f} tok/s")
            _print_comparison(label[:42], best_result['prefill_tps'], best_result['decode_tps'])
            results.append({
                'model': label,
                'prefill_tps': best_result['prefill_tps'],
                'decode_tps': best_result['decode_tps'],
                'config': best_config,
            })
        else:
            raise AssertionError(f"Todas las configs fallaron para {gf.name}")

    best_decode = max(r['decode_tps'] for r in results)
    assert best_decode > 20, f"Decode TPS muy bajo: {best_decode:.1f} (minimo 20 tok/s)"

    if len(results) == 1:
        return {
            'prefill_tps': round(results[0]['prefill_tps'], 1),
            'decode_tps':  round(results[0]['decode_tps'], 1),
            'model':       results[0]['model'],
            'config':      results[0]['config'],
        }
    return {
        'models':          results,
        'best_decode_tps': round(best_decode, 1),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark llama.cpp — 900 prefill / 100 decode (binario nativo)")
    parser.add_argument('--gguf-path', default=None, help="Ruta directa al .gguf")
    parser.add_argument('--n-prefill', type=int, default=N_PREFILL_TOKENS)
    parser.add_argument('--n-decode',  type=int, default=N_DECODE_TOKENS)
    parser.add_argument('--n-runs',    type=int, default=N_RUNS)
    parser.add_argument('--threads',   type=int, default=None, help="Forzar numero de threads")
    args = parser.parse_args()
    test_benchmark_llama_cpp(args.gguf_path, args.n_prefill, args.n_decode, args.n_runs, args.threads)
