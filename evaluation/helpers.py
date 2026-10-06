"""Shared access and result handling for the active checkpoint audit."""
import os
import time

os.environ.setdefault('KMP_AFFINITY', 'granularity=fine,compact,1,0')
os.environ.setdefault('KMP_BLOCKTIME', '0')

from cpu1.components import BitLinear


def get_model(engine):
    return engine._get_model()


def run_check(name: str, fn, *args, **kwargs) -> dict:
    """Ejecutar un check individual con manejo de errores y timing."""
    try:
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        status = 'N/A' if isinstance(result, dict) and result.get('skipped') else 'PASS'
        return {'status': status, 'time_s': round(elapsed, 2), 'result': result}
    except AssertionError as e:
        return {'status': 'FAIL', 'error': str(e)}
    except Exception as e:
        return {'status': 'ERROR', 'error': f'{type(e).__name__}: {e}'}
