"""
Utilidades compartidas para la suite de validacion CPU-1 v2.

Optimizaciones multicore para Intel i7 8th gen:
- KMP_AFFINITY: pin threads a cores consecutivos (menos cache thrashing)
- KMP_BLOCKTIME=0: sin sleep de threads idle (menor latencia de wakeup)
- Inter-op parallelism: pipeline de ops independientes entre capas
- torch.compile: fusion de kernels para eliminar overhead de Python
"""
import sys
import os
import platform

# Intel MKL/OpenMP: configurar ANTES de importar torch
os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
os.environ.setdefault('KMP_BLOCKTIME', '0')

import torch

# Agregar raiz del proyecto al path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model import CPU1ModelV2, BitLinear, bytes_to_patches  # noqa: E402


def configure_cpu_threads(n_threads: int | None = None) -> int:
    """
    Configurar threads para maximo throughput en CPU multicore.
    Usa cores fisicos para intra-op (optimo para GEMM) y
    2 threads inter-op para pipeline de ops independientes.
    """
    if n_threads is None:
        total = os.cpu_count() or 4
        if platform.machine() in ('arm64', 'aarch64'):
            n_threads = total
        else:
            n_threads = max(1, total // 2)

    # Sincronizar env vars con el runtime de OpenMP/MKL
    os.environ['OMP_NUM_THREADS'] = str(n_threads)
    os.environ['MKL_NUM_THREADS'] = str(n_threads)

    torch.set_num_threads(n_threads)

    # Inter-op: pipeline de ops independientes (e.g., 4 output heads)
    interop = min(2, max(1, n_threads // 3))
    try:
        torch.set_num_interop_threads(interop)
    except RuntimeError:
        interop = 1

    print(f"[CPU opt] Threads: {n_threads} intra-op, {interop} inter-op")
    return n_threads


def load_model(checkpoint_path: str, config: dict | None = None,
               device: str = 'cpu') -> CPU1ModelV2:
    """Carga modelo desde checkpoint."""
    model = CPU1ModelV2(config)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def generate_bytes_from_model(model: CPU1ModelV2, prompt: bytes,
                              max_bytes: int = 500) -> bytes:
    """Genera bytes usando el metodo del modelo."""
    return model.generate_bytes(prompt, max_tokens=max_bytes, temperature=1.0)


def get_model_parallel_workers() -> int:
    """Workers para forward passes concurrentes en CPU.
    Calcula logical_cores / intra_op_threads para evitar sobresuscripcion
    del thread pool ATen/OpenMP. Para i7-8xxx: 12 / 6 = 2 workers."""
    logical = os.cpu_count() or 4
    intra = torch.get_num_threads() or max(1, logical // 2)
    return max(1, logical // intra)


def get_dataloader_workers() -> int:
    """Workers para DataLoader (carga de datos I/O en paralelo).
    Conservador en Windows (spawn overhead). Retorna 0-2 workers."""
    return max(0, min(2, (os.cpu_count() or 4) // 4))


# ---------------------------------------------------------------------------
# Optimizacion de modelo para benchmarks de throughput
# ---------------------------------------------------------------------------
_compile_cache = {}


def _setup_msvc_env() -> bool:
    """Configura INCLUDE/LIB/CXX para cl.exe en Windows si no estan seteados.

    Busca VS Build Tools en D:\\Programas\\BuildTools y el Windows SDK en
    C:\\Program Files (x86)\\Windows Kits\\10.  Solo actua si INCLUDE no
    esta ya configurado (e.g. vcvarsall.bat ya fue llamado externamente).
    Retorna True si el entorno quedo listo, False si no se encontraron los paths.
    """
    if not sys.platform.startswith('win'):
        return False
    if os.environ.get('INCLUDE'):
        return True  # ya configurado

    import glob as _glob

    # Localizar MSVC
    msvc_base = r'D:\Programas\BuildTools\VC\Tools\MSVC'
    if not os.path.isdir(msvc_base):
        return False
    versions = sorted(os.listdir(msvc_base), reverse=True)
    if not versions:
        return False
    msvc_root = os.path.join(msvc_base, versions[0])

    # Localizar Windows SDK
    sdk_include_base = r'C:\Program Files (x86)\Windows Kits\10\Include'
    if not os.path.isdir(sdk_include_base):
        return False
    sdk_versions = sorted(os.listdir(sdk_include_base), reverse=True)
    if not sdk_versions:
        return False
    sdk_ver = sdk_versions[0]
    sdk_include = os.path.join(sdk_include_base, sdk_ver)
    sdk_lib = os.path.join(r'C:\Program Files (x86)\Windows Kits\10\Lib', sdk_ver)

    include = ';'.join([
        os.path.join(msvc_root, 'include'),
        os.path.join(sdk_include, 'ucrt'),
        os.path.join(sdk_include, 'um'),
        os.path.join(sdk_include, 'shared'),
    ])
    lib = ';'.join([
        os.path.join(msvc_root, 'lib', 'x64'),
        os.path.join(sdk_lib, 'ucrt', 'x64'),
        os.path.join(sdk_lib, 'um', 'x64'),
    ])
    cl_exe = os.path.join(msvc_root, 'bin', 'Hostx64', 'x64', 'cl.exe')

    if not os.path.isfile(cl_exe):
        return False

    os.environ['INCLUDE'] = include
    os.environ['LIB'] = lib
    os.environ.setdefault('CXX', cl_exe)
    return True


def optimize_model_for_inference(model):
    """Optimiza modelo para maximo throughput en CPU multicore.

    - torch.compile con inductor + cl.exe: genera codigo AVX2/AVX-512
      vectorizado y fusiona kernels (elimina overhead Python/autograd)
    - Configura entorno MSVC automaticamente si no esta activado
    - Cachea compilacion por modelo (id) para evitar recompilar entre tests
    - Compatible con inference_mode() para eliminar dispatch overhead

    Fallback a eager si torch.compile no esta disponible o falla.
    """
    model_id = id(model)
    if model_id in _compile_cache:
        return _compile_cache[model_id]

    model.eval()
    result = model

    if hasattr(torch, 'compile'):
        try:
            _setup_msvc_env()
            if hasattr(torch, '_dynamo'):
                torch._dynamo.config.suppress_errors = True
            result = torch.compile(model)
            print("[CPU opt] torch.compile activado (inductor + cl.exe)")
        except Exception as e:
            print(f"[CPU opt] torch.compile no disponible: {e}")

    _compile_cache[model_id] = result
    return result
