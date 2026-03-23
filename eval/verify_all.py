"""
Verificacion rapida: corre todos los tests con un modelo mini
para confirmar que no hay errores de codigo.
"""
import sys
import os
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, EVAL_DIR)

import torch
from model_v2 import CPU1ModelV2

# Modelo minimo para tests rapidos
MINI_CONFIG = {
    'd_model': 64,
    'n_layers': 2,
    'd_ff': 128,
    'fp_rank': 4,
    'pfnet_hidden': 16,
}


def make_mini_model():
    model = CPU1ModelV2(MINI_CONFIG)
    model.eval()
    return model


def make_mini_dataset(n_samples=32, seq_len=64):
    """Dataset falso con la estructura que esperan los tests."""
    from torch.utils.data import TensorDataset

    patches = torch.randint(0, 256, (n_samples, seq_len, 4), dtype=torch.long)
    targets = torch.randint(0, 256, (n_samples, seq_len, 4), dtype=torch.long)
    # Crear distribuciones con entropia razonable (no uniformes)
    # Concentrar probabilidad en unos pocos bytes para simular teacher real
    logits = torch.zeros(n_samples, seq_len, 256)
    for i in range(n_samples):
        # Poner masa en 3-5 posiciones aleatorias por cada posicion
        hot_indices = torch.randint(0, 256, (seq_len, 4))
        for s in range(seq_len):
            logits[i, s, hot_indices[s]] = torch.randn(4).abs() * 5 + 2
    teacher_probs = logits.softmax(dim=-1)
    teacher_mask = torch.rand(n_samples, seq_len) > 0.7  # ~30% coverage

    class FakeDataset:
        def __init__(self):
            self.data = {
                'patches': patches,
                'targets': targets,
                'teacher_probs': teacher_probs,
                'teacher_mask': teacher_mask,
            }
            self._len = n_samples

        def __len__(self):
            return self._len

        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.data.items()}

    return FakeDataset()


def run_test(name, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        print(f"  PASS  {name:<30} ({elapsed:.2f}s)")
        return True
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  FAIL  {name:<30} ({elapsed:.2f}s)")
        traceback.print_exc()
        print()
        return False


def main():
    print("Creando modelo mini y dataset falso...")
    model = make_mini_model()
    dataset = make_mini_dataset()

    from test_01_forward import test_forward_no_crash
    from test_02_probs import test_probability_distributions
    from test_03_determinism import test_determinism
    from test_04_state_propagation import test_state_propagation
    from test_05_ternary_weights import test_ternary_weights
    from test_06_cpu_throughput import test_cpu_throughput
    from test_07_latency import test_latency_distribution
    from test_08_jitter import test_latency_jitter
    from test_09_context_scaling import test_throughput_vs_context
    from test_10_ram_usage import test_ram_usage
    from test_11_perplexity import test_perplexity
    from test_12_generation_quality import test_generation_quality
    from test_13_utf8_validity import test_utf8_validity
    from test_14_training_convergence import test_training_convergence
    from test_17_teacher_coverage import test_teacher_coverage
    from test_18_teacher_entropy import test_teacher_entropy
    from test_21_mlgru_state_size import test_mlgru_state_size

    print(f"\n{'='*50}")
    print("Verificacion de todos los tests")
    print(f"{'='*50}\n")

    results = []

    # Cat 1: Correctitud
    results.append(run_test("01_forward", lambda: test_forward_no_crash(model)))
    results.append(run_test("02_probs", lambda: test_probability_distributions(model)))
    results.append(run_test("03_determinism", lambda: test_determinism(model)))
    results.append(run_test("04_state_propagation", lambda: test_state_propagation(model)))
    results.append(run_test("05_ternary_weights", lambda: test_ternary_weights(model)))

    # Cat 2: Velocidad (con thresholds relajados para modelo mini)
    results.append(run_test("06_throughput", lambda: test_cpu_throughput(model, n_tokens=20, n_runs=2)))
    results.append(run_test("07_latency", lambda: test_latency_distribution(model, n_tokens=50)))
    results.append(run_test("08_jitter", lambda: test_latency_jitter(model, n_tokens=50)))
    results.append(run_test("09_context_scaling", lambda: test_throughput_vs_context(model)))
    results.append(run_test("10_ram_usage", lambda: test_ram_usage(model, n_tokens=20)))

    # Cat 3: Calidad — con modelo sin entrenar solo verificamos que corran sin crash
    # (los asserts de calidad fallan legitimamente con modelo random)
    def test_perplexity_smoke():
        ppl = test_perplexity.__wrapped__(model, dataset, max_batches=5) \
            if hasattr(test_perplexity, '__wrapped__') else None
        # Llamar manualmente sin assert
        model.eval()
        total_loss = 0
        n = 0
        from torch.utils.data import DataLoader
        import math
        for batch in DataLoader(dataset, batch_size=4):
            if n >= 5:
                break
            patches = batch['patches']
            targets = batch['targets']
            with torch.no_grad():
                logits, _ = model(patches)
                loss = sum(
                    torch.nn.functional.cross_entropy(
                        logits[i].reshape(-1, 256),
                        targets[:, :, i].long().reshape(-1)
                    ) for i in range(4)
                ) / 4
            total_loss += loss.item()
            n += 1
        ppl = math.exp(total_loss / n)
        print(f"[Test 11] Perplexity: {ppl:.2f} (sin entrenar, solo smoke test)")
        assert ppl > 0 and not math.isnan(ppl), "Perplexity invalida"

    def test_generation_smoke():
        """Solo verifica que genera sin crash, no calidad."""
        model.eval()
        model.cpu()
        from model_v2 import bytes_to_patches
        prompt = b"The history of artificial intelligence"
        patches = bytes_to_patches(prompt)
        states = None
        with torch.no_grad():
            _, states = model(patches, states)
        last_patch = patches[:, -1:, :]
        generated = []
        with torch.no_grad():
            for _ in range(25):
                logits, states = model(last_patch, states)
                next_bytes = [logits[i][0, 0].argmax().item() for i in range(4)]
                generated.extend(next_bytes)
                last_patch = torch.tensor(next_bytes, dtype=torch.uint8).reshape(1, 1, 4)
        print(f"[Test 12] Generados {len(generated)} bytes (smoke test OK)")

    def test_utf8_smoke():
        """Solo verifica que generate_bytes corre sin crash."""
        gen = model.generate_bytes(b"Hello", max_tokens=50)
        print(f"[Test 13] generate_bytes produjo {len(gen)} bytes (smoke test OK)")
        assert len(gen) > 0, "generate_bytes no produjo nada"

    def test_convergence_smoke():
        """Solo verifica que el training loop corre sin crash."""
        m = make_mini_model()
        m.train()
        optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)
        from torch.utils.data import DataLoader
        losses = []
        for i, batch in enumerate(DataLoader(dataset, batch_size=4)):
            if i >= 5:
                break
            patches = batch['patches']
            targets = batch['targets']
            logits, _ = m(patches)
            loss = sum(
                torch.nn.functional.cross_entropy(
                    logits[j].reshape(-1, 256),
                    targets[:, :, j].long().reshape(-1)
                ) for j in range(4)
            ) / 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"[Test 14] Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (smoke test OK)")
        assert all(not math.isnan(l) for l in losses), "NaN en training"

    import math
    results.append(run_test("11_perplexity", test_perplexity_smoke))
    results.append(run_test("12_generation", test_generation_smoke))
    results.append(run_test("13_utf8", test_utf8_smoke))
    results.append(run_test("14_convergence", test_convergence_smoke))

    # Cat 5: Distilacion
    results.append(run_test("17_teacher_coverage", lambda: test_teacher_coverage(dataset)))
    results.append(run_test("18_teacher_entropy", lambda: test_teacher_entropy(dataset, max_batches=5)))

    # Cat 6: Estado MLGRU
    model.eval()
    results.append(run_test("21_state_size", lambda: test_mlgru_state_size(model)))

    # Resumen
    passed = sum(results)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"Resultado: {passed}/{total} tests pasaron")
    print(f"{'='*50}")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
