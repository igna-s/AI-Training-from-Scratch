"""Check the recurrent scan against its defining equation without model files."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from cpu1.components import AblationMLGRU
    from evaluation.audit import OUT, save, sha
    torch.set_num_threads(1)
    torch.manual_seed(19)
    c = torch.randn(1, 32, 8)
    f = torch.sigmoid(torch.randn(1, 32, 8))
    initial = torch.randn(1, 8)
    actual, _ = AblationMLGRU._parallel_scan(c, f, initial)
    h = initial.clone()
    steps = []
    for i in range(32):
        h = f[:, i]*h + (1-f[:, i])*c[:, i]
        steps.append(h)
    reference = torch.stack(steps, dim=1)
    first, carry = AblationMLGRU._parallel_scan(c[:, :16], f[:, :16], initial)
    second, _ = AblationMLGRU._parallel_scan(c[:, 16:], f[:, 16:], carry)
    serial_diff = (actual-reference).abs().max().item()
    chunk_diff = (actual-torch.cat([first, second], dim=1)).abs().max().item()
    result = {'seed':19, 'shape':[1,32,8], 'torch':torch.__version__, 'dtype':'float32', 'reference':'h[t] = f[t] * h[t-1] + (1-f[t]) * c[t]', 'serial_max_diff':serial_diff, 'chunks_max_diff':chunk_diff, 'tolerance':1e-4, 'status':'PASS' if max(serial_diff,chunk_diff)<1e-4 else 'FAIL', 'components_sha256':sha(ROOT/'cpu1'/'components.py'), 'script_sha256':sha(Path(__file__))}
    save(OUT/'implementation_diagnostics.json', result)
    print(result)


if __name__ == '__main__':
    main()
