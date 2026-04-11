"""
bench_quick.py — Mide tokens/segundo del kernel ternary_lut_matmul_omp.
Uso: .venv/Scripts/python.exe inference/bench_quick.py
"""
import ctypes, sys, os

dll_path = os.path.join(os.path.dirname(__file__), "kernels.dll")
lib = ctypes.CDLL(dll_path)
lib.benchmark_lut_matmul.restype  = ctypes.c_double
lib.benchmark_lut_matmul.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

# Configuraciones tipicas del modelo MLGRU d_model=1024
configs = [
    ("q/k/v/o  d_in=1024 d_out=1024", 1024, 1024),
    ("ff_gate  d_in=1024 d_out=2730", 1024, 2730),
    ("ff_up    d_in=1024 d_out=4096", 1024, 4096),
]

N = 300  # iteraciones por config (~10s total)

print("=" * 68)
print("  BENCHMARK ternary_lut_matmul_omp  (M=1 decode, TILE_G=16)")
print("=" * 68)
print(f"  {'Config':<30}  {'Gather (AVX2)':>12}  {'Scalar':>12}  Winner")
print("-" * 68)

for name, d_in, d_out in configs:
    tps_g = lib.benchmark_lut_matmul(d_in, d_out, N, 1)
    tps_s = lib.benchmark_lut_matmul(d_in, d_out, N, 0)
    winner = "GATHER" if tps_g > tps_s else "SCALAR"
    pct = abs(tps_g - tps_s) / min(tps_g, tps_s) * 100
    print(f"  {name:<30}  {tps_g:>9,.0f} TPS  {tps_s:>9,.0f} TPS  {winner} (+{pct:.0f}%)")

print("=" * 68)
print(f"  Linea base pre-fix reportada: ~10 TPS")
print("=" * 68)
