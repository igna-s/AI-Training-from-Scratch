"""
Eval — Correctness: verifica propiedades del modelo.

Checks adaptivos segun arquitectura (engine.info):
  forward_shapes  — Forward pass sin crash para varias shapes
  probs           — Distribuciones de probabilidad validas
  determinism     — Mismo input → mismo output
  state_prop      — Estado MLGRU se propaga correctamente (solo recurrente)
  ternary_weights — Pesos BitLinear son {-1, 0, 1} (solo ternary)
  state_size      — Estado MLGRU < 1MB (solo recurrente)

Todos usan el modelo cargado por CPU1InferenceEngine.
"""
import torch

from helpers import get_model, BitLinear


def _make_dummy_input(info, batch: int = 1, seq_len: int = 64):
    """Crear input dummy segun tipo de modelo."""
    if info.is_byte_level:
        return torch.randint(0, 256, (batch, seq_len, info.patch_size), dtype=torch.long)
    else:
        return torch.randint(0, info.vocab_size, (batch, seq_len), dtype=torch.long)


def _unpack_forward(result):
    """Extraer logits y states de cualquier modelo.
    CPU1ModelV2: (logits_tuple, states)
    AblationModel: (logits_or_tuple, states, extra_losses)
    """
    if len(result) == 2:
        return result[0], result[1]
    else:
        return result[0], result[1]


def _get_logits_list(logits, info):
    """Normalizar logits a lista de tensores [B, S, V] siempre.

    Casos:
      - tuple/list de heads (byte sin LocalDecoder): devuelve lista tal cual.
      - LocalByteDecoder: [B, S, P, 256] → lista de P tensores [B, S, 256].
      - tensor 3D (BPE o produccion): devuelve lista de 1 elemento.
    """
    if isinstance(logits, (tuple, list)):
        return list(logits)
    if logits.dim() == 4:
        # LocalByteDecoder: [B, S, P, 256] → P × [B, S, 256]
        return [logits[:, :, i, :] for i in range(logits.shape[2])]
    return [logits]


def check_forward_shapes(engine) -> dict:
    """Forward sin crash para varias combinaciones batch × seq."""
    info = engine.info
    model = engine._get_model()
    configs = [(b, s) for b in [1, 2] for s in [1, 64, 256]]
    errors = []

    for batch, seq_len in configs:
        x = _make_dummy_input(info, batch, seq_len)
        with torch.inference_mode():
            result = model(x)
            logits, _ = _unpack_forward(result)

        logits_list = _get_logits_list(logits, info)
        expected_heads = info.n_heads
        expected_vocab = info.vocab_size

        if len(logits_list) != expected_heads:
            errors.append(f"Esperado {expected_heads} heads, obtenido {len(logits_list)}")
            continue

        for i, l in enumerate(logits_list):
            expected_shape = (batch, seq_len, expected_vocab)
            if l.shape != expected_shape:
                errors.append(f"Head {i}: shape {l.shape} != {expected_shape}")
            if torch.isnan(l).any():
                errors.append(f"NaN en logits batch={batch} seq={seq_len} head={i}")
            if torch.isinf(l).any():
                errors.append(f"Inf en logits batch={batch} seq={seq_len} head={i}")

    assert not errors, "\n".join(errors)
    return {'configs_tested': len(configs), 'heads': info.n_heads}


def check_probs(engine) -> dict:
    """Las heads producen distribuciones validas (suman a 1)."""
    info = engine.info
    model = engine._get_model()
    x = _make_dummy_input(info, batch=1, seq_len=128)

    with torch.inference_mode():
        logits, _ = _unpack_forward(model(x))

    logits_list = _get_logits_list(logits, info)
    max_err = 0.0
    for i, head_logits in enumerate(logits_list):
        probs = torch.softmax(head_logits, dim=-1)
        sums = probs.sum(dim=-1)
        err = (sums - 1.0).abs().max().item()
        max_err = max(max_err, err)
        assert err < 1e-4, f"Head {i}: suma de probs error={err:.6f}"

    return {'max_error': max_err}


def check_determinism(engine) -> dict:
    """Mismo input con mismo estado da output identico."""
    info = engine.info
    model = engine._get_model()
    torch.manual_seed(42)
    x = _make_dummy_input(info, batch=1, seq_len=64)

    with torch.inference_mode():
        logits1, _ = _unpack_forward(model(x, hidden_states=None))
        logits2, _ = _unpack_forward(model(x, hidden_states=None))

    list1 = _get_logits_list(logits1, info)
    list2 = _get_logits_list(logits2, info)

    for i in range(len(list1)):
        diff = (list1[i] - list2[i]).abs().max().item()
        assert diff == 0.0, f"Head {i}: output no deterministico, diff={diff}"

    return {'deterministic': True}


def check_state_propagation(engine) -> dict:
    """Procesar en chunks da el mismo resultado que todo junto (solo recurrente)."""
    info = engine.info
    if not info.is_recurrent:
        return {'skipped': True, 'reason': 'modelo no recurrente (Transformer)'}

    model = engine._get_model()
    torch.manual_seed(0)
    x = _make_dummy_input(info, batch=1, seq_len=128)

    with torch.inference_mode():
        logits_full, _ = _unpack_forward(model(x))

    states = None
    chunk_outputs = []
    with torch.inference_mode():
        for i in range(0, 128, 32):
            chunk = x[:, i:i + 32] if x.dim() == 2 else x[:, i:i + 32, :]
            result = model(chunk, states)
            logits_chunk, states = _unpack_forward(result)
            first_head = _get_logits_list(logits_chunk, info)[0]
            chunk_outputs.append(first_head)

    out_chunked = torch.cat(chunk_outputs, dim=1)
    out_full = _get_logits_list(logits_full, info)[0]
    diff = (out_full - out_chunked).abs().max().item()
    assert diff < 1e-4, f"State propagation error: {diff:.6f}"
    return {'max_diff': diff}


def check_ternary_weights(engine) -> dict:
    """Pesos BitLinear quantizados son {-1, 0, 1} (solo ternary)."""
    info = engine.info
    if not info.is_ternary:
        return {'skipped': True, 'reason': 'modelo FP16 (sin BitLinear)'}

    model = engine._get_model()
    n_checked = 0

    for name, module in model.named_modules():
        if not isinstance(module, BitLinear):
            continue

        if hasattr(module, 'weight_T_ternary'):
            w = module.weight_T_ternary
            unique = set(w.unique().tolist())
            invalid = unique - {-1.0, 0.0, 1.0}
            assert not invalid, f"{name}: valores no ternarios en weight_T_ternary: {invalid}"
        elif module.weight.numel() > 0:
            w = module.weight.data
            w_scale = w.abs().mean()
            w_q = (w / (w_scale + 1e-8)).round().clamp(-1, 1)
            unique = set(w_q.unique().tolist())
            invalid = unique - {-1.0, 0.0, 1.0}
            assert not invalid, f"{name}: valores no ternarios: {invalid}"

        n_checked += 1

    assert n_checked > 0, "No se encontraron modulos BitLinear"
    return {'modules_checked': n_checked}


def check_state_size(engine) -> dict:
    """Estado MLGRU < 1MB (solo recurrente)."""
    info = engine.info
    if not info.is_recurrent:
        return {'skipped': True, 'reason': 'modelo no recurrente (Transformer)'}

    model = engine._get_model()
    x = _make_dummy_input(info, batch=1, seq_len=1)

    with torch.inference_mode():
        _, states = _unpack_forward(model(x))

    # Filtrar None states (pueden existir si algun bloque es transformer)
    valid_states = [s for s in states if s is not None]
    state_bytes = sum(s.numel() * s.element_size() for s in valid_states)
    state_kb = state_bytes / 1024

    kv_cache_kb = 20 * 512 * 2048 * 2 * 2 / 1024
    reduction = kv_cache_kb / state_kb if state_kb > 0 else 0

    assert state_kb < 1024, f"Estado MLGRU demasiado grande: {state_kb:.1f} KB"
    return {'state_kb': round(state_kb, 1), 'kv_cache_kb': round(kv_cache_kb, 1),
            'reduction_x': round(reduction, 0)}


# =========================================================================
# Entry point
# =========================================================================

def run_correctness(engine) -> dict:
    """Ejecutar todos los checks de correctness. 1 engine → 6 checks."""
    from helpers import run_check

    print(f"\n--- Correctness ({engine.info.model_class}) ---")
    results = {}
    checks = [
        ('forward_shapes',  check_forward_shapes),
        ('probs',           check_probs),
        ('determinism',     check_determinism),
        ('state_prop',      check_state_propagation),
        ('ternary_weights', check_ternary_weights),
        ('state_size',      check_state_size),
    ]
    for name, fn in checks:
        r = run_check(name, fn, engine)
        results[name] = r
        status = r['status']
        extra = ''
        if r.get('result', {}).get('skipped'):
            extra = f" (skip: {r['result']['reason']})"
        print(f"  {status:5s}  {name}{extra}")

    return results
