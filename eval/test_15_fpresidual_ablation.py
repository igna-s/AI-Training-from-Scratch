"""
Test 15 — Comparativa FPResidual vs Sin FPResidual

Comparar Run 10 (con FPResidual) vs Run 11 (sin FPResidual).
Esta es la tabla mas importante del paper para justificar el diseno.

Metrica: diferencia de perplexity y tokens/seg
Criterio: reportar, no hay pass/fail — el resultado informa la decision

Concurrencia: ThreadPoolExecutor para evaluar perplexity de ambos modelos
en paralelo (2 workers, uno por modelo). La perplexity no mide timing
asi que la contention de CPU es aceptable. El throughput se mide
secuencialmente para precision.
"""
from concurrent.futures import ThreadPoolExecutor

from test_11_perplexity import test_perplexity
from test_06_cpu_throughput import test_cpu_throughput


def test_fpresidual_ablation(model_with_fp, model_without_fp, test_dataset):
    results = {}

    # Perplexity en paralelo (2 workers, uno por modelo)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future_fp = pool.submit(test_perplexity, model_with_fp, test_dataset)
        future_nfp = pool.submit(test_perplexity, model_without_fp, test_dataset)
        ppl_fp = future_fp.result()
        ppl_nfp = future_nfp.result()

    # Throughput secuencial (benchmark de timing preciso)
    tps_fp = test_cpu_throughput(model_with_fp)
    tps_nfp = test_cpu_throughput(model_without_fp)

    for name, m, ppl, tps in [
        ('con_fpresidual', model_with_fp, ppl_fp, tps_fp),
        ('sin_fpresidual', model_without_fp, ppl_nfp, tps_nfp),
    ]:
        ram = sum(p.numel() * 2 for p in m.parameters()) / 1e6
        results[name] = {'perplexity': ppl, 'tps': tps, 'ram_mb': ram}

    delta_ppl = results['sin_fpresidual']['perplexity'] - \
        results['con_fpresidual']['perplexity']
    delta_tps = results['sin_fpresidual']['tps'] - \
        results['con_fpresidual']['tps']

    print(f"\n[Test 15] FPResidual Ablation:")
    print(f"  Con FP:  PPL={results['con_fpresidual']['perplexity']:.2f}"
          f"  TPS={results['con_fpresidual']['tps']:.1f}")
    print(f"  Sin FP:  PPL={results['sin_fpresidual']['perplexity']:.2f}"
          f"  TPS={results['sin_fpresidual']['tps']:.1f}")
    print(f"  DeltaPerplexity: {delta_ppl:+.2f}  DeltaTPS: {delta_tps:+.1f}")

    if delta_ppl < 1.0:
        print("  -> Sin FPResidual es preferible (mismo PPL, mas rapido/simple)")
    else:
        print(f"  -> FPResidual justificado (mejora PPL en {delta_ppl:.2f} puntos)")

    return results


if __name__ == "__main__":
    from helpers import load_model
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-fp', required=True, help='Modelo con FPResidual')
    parser.add_argument('--model-no-fp', required=True, help='Modelo sin FPResidual')
    parser.add_argument('--dataset', required=True)
    args = parser.parse_args()

    from datasets import load_from_disk
    m_fp = load_model(args.model_fp)
    m_no_fp = load_model(args.model_no_fp)
    dataset = load_from_disk(args.dataset)
    test_fpresidual_ablation(m_fp, m_no_fp, dataset['test'])
