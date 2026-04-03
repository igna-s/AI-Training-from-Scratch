"""
Eval — Quality: generacion, UTF-8, perplexity.

Adaptivo segun arquitectura:
  - Byte-level: usa engine.generate() para generar bytes
  - BPE: genera tokens (futuro) — por ahora solo perplexity
  - Perplexity: funciona con ambos formatos

Tests 12 y 13 fusionados: generan UNA VEZ y evaluan sobre el mismo output.
"""
import math
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from helpers import get_model


def _generate_text(engine, prompt: bytes = b"The history of artificial intelligence",
                   max_bytes: int = 1000) -> bytes:
    """Generar texto usando engine.generate() — unica via."""
    return engine.generate(prompt, max_bytes=max_bytes,
                           temperature=1.0, top_p=0.9, reset=True)


def check_generation(generated: bytes) -> dict:
    """Output no degenerado."""
    if len(generated) < 8:
        raise AssertionError(f"Output demasiado corto: {len(generated)} bytes")

    printable = sum(32 <= b < 127 for b in generated)
    pct_print = printable / len(generated)

    repeats = sum(
        generated[i:i + 8] == generated[i - 8:i]
        for i in range(8, len(generated))
    ) / max(len(generated) - 8, 1)

    print(f"    Printable: {pct_print * 100:.1f}%  Repeticion: {repeats * 100:.1f}%")
    print(f"    Muestra: {generated[:80]}")

    assert pct_print > 0.5, \
        f"Solo {pct_print * 100:.1f}% bytes printables — modelo degenerado"
    assert repeats < 0.3, \
        f"Repeticion alta: {repeats * 100:.1f}% — modelo en loop"

    return {'printable_pct': round(pct_print, 3), 'repeat_ratio': round(repeats, 3)}


def check_utf8(generated: bytes) -> dict:
    """UTF-8 valido en output."""
    windows = [generated[i:i + 64] for i in range(0, len(generated) - 64, 32)]
    if not windows:
        return {'utf8_pct': 0.0}

    valid = sum(1 for w in windows if _is_valid_utf8(w))
    pct = valid / len(windows)

    print(f"    UTF-8 valido: {pct * 100:.1f}%")
    assert pct > 0.7, f"Demasiados bytes UTF-8 invalidos: {pct * 100:.1f}%"
    return {'utf8_pct': round(pct, 3)}


def _is_valid_utf8(b: bytes) -> bool:
    try:
        b.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


def check_perplexity(engine, dataset, max_batches: int = 200) -> dict:
    """Perplexity en held-out dataset. Funciona con byte y BPE."""
    info = engine.info
    model = get_model(engine)
    total_loss = 0.0
    n_batches = 0

    n_workers = max(0, min(2, (os.cpu_count() or 4) // 4))
    loader_kwargs = {'batch_size': 4}
    if n_workers > 0:
        loader_kwargs['num_workers'] = n_workers
        loader_kwargs['prefetch_factor'] = 2

    for batch in DataLoader(dataset, **loader_kwargs):
        if n_batches >= max_batches:
            break

        patches = batch['patches']
        targets = batch['targets']

        with torch.inference_mode():
            result = model(patches)
            logits = result[0]

            if isinstance(logits, (tuple, list)):
                # Byte-level: 4 heads × 256
                n_heads = len(logits)
                all_logits = torch.cat(
                    [logits[i].reshape(-1, info.vocab_size) for i in range(n_heads)], dim=0
                )
                all_targets = torch.cat(
                    [targets[:, :, i].long().reshape(-1) for i in range(n_heads)], dim=0
                )
            else:
                # BPE: single head
                all_logits = logits.reshape(-1, info.vocab_size)
                all_targets = targets.long().reshape(-1)

            loss = F.cross_entropy(all_logits, all_targets)

        total_loss += loss.item()
        n_batches += 1

    if n_batches == 0:
        raise AssertionError("Dataset vacio — no se pudo calcular perplexity")

    avg_loss = total_loss / n_batches
    ppl = math.exp(avg_loss)

    print(f"    Perplexity: {ppl:.2f} ({n_batches} batches)")
    assert ppl < 200, f"Perplexity inaceptable: {ppl:.2f}"
    return {'perplexity': round(ppl, 2), 'avg_loss': round(avg_loss, 4)}


# =========================================================================
# Entry point
# =========================================================================

def run_quality(engine, dataset=None) -> dict:
    """Ejecutar checks de calidad. Adaptivo por arquitectura."""
    from helpers import run_check

    info = engine.info
    print(f"\n--- Quality ({info.model_class}) ---")
    results = {}

    # Generacion solo para modelos byte-level
    if info.is_byte_level:
        try:
            generated = _generate_text(engine, max_bytes=1000)
        except Exception as e:
            print(f"  ERROR  generation failed: {e}")
            generated = b""

        if generated:
            results['generation'] = run_check('generation', check_generation, generated)
            print(f"  {results['generation']['status']:5s}  generation")

            results['utf8'] = run_check('utf8', check_utf8, generated)
            print(f"  {results['utf8']['status']:5s}  utf8")
        else:
            results['generation'] = {'status': 'ERROR', 'error': 'No output generated'}
            results['utf8'] = {'status': 'ERROR', 'error': 'No output generated'}
    else:
        print("  N/A    generation (BPE — generate_tokens() no implementado aun)")
        print("  N/A    utf8 (BPE — no aplica)")
        results['generation'] = {'status': 'N/A', 'reason': 'BPE model'}
        results['utf8'] = {'status': 'N/A', 'reason': 'BPE model'}

    # Perplexity (funciona con ambos formatos)
    if dataset is not None:
        results['perplexity'] = run_check('perplexity', check_perplexity, engine, dataset)
        print(f"  {results['perplexity']['status']:5s}  perplexity")

    return results
