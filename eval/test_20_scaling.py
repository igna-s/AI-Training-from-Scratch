"""
Test 20 — Scaling: 50M vs 170M

Verificar que el modelo escala correctamente de 50M a 170M.
La perplexity debe bajar y el throughput debe mantenerse razonable.

Metrica: PPL y TPS para ambos tamanos
Criterio: PPL(170M) < PPL(50M), TPS(170M) > 5 patches/seg
"""
from test_11_perplexity import test_perplexity
from test_06_cpu_throughput import test_cpu_throughput


def test_scaling(model_50m, model_170m, test_dataset):
    print("\n[Test 20] Scaling 50M -> 170M:")

    ppl_50 = test_perplexity(model_50m, test_dataset)
    ppl_170 = test_perplexity(model_170m, test_dataset)
    tps_50 = test_cpu_throughput(model_50m)
    tps_170 = test_cpu_throughput(model_170m)

    print(f"  50M:  PPL={ppl_50:.2f}   TPS={tps_50:.1f}")
    print(f"  170M: PPL={ppl_170:.2f}  TPS={tps_170:.1f}")
    print(f"  Mejora PPL: {ppl_50 - ppl_170:.2f} puntos")
    print(f"  Costo TPS:  {(tps_50 - tps_170) / tps_50 * 100:.1f}% mas lento")

    assert ppl_170 < ppl_50, "170M deberia tener mejor PPL que 50M"
    assert tps_170 > 5, f"170M demasiado lento: {tps_170:.1f} patches/seg"

    return {
        '50m': {'ppl': ppl_50, 'tps': tps_50},
        '170m': {'ppl': ppl_170, 'tps': tps_170},
    }


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-50m', required=True)
    parser.add_argument('--model-170m', required=True)
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    m50 = load_model(args.model_50m)
    m170 = load_model(args.model_170m)
    dataset = load_from_disk(args.dataset)
    test_scaling(m50, m170, dataset['test'])
