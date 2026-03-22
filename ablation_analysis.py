"""
Ablation results analysis — generates paper-ready tables (text + LaTeX).

Usage:
    # After all runs complete, collect metrics and generate tables
    python ablation_analysis.py --metrics_dir .

    # Or load from a single JSON with all results
    python ablation_analysis.py --results_json ablation_results.json
"""

import json
import argparse
from pathlib import Path

from ablation_config import RUN_CONFIGS


def load_all_metrics(metrics_dir: str = ".") -> dict[str, dict]:
    """Load metrics_run_XX.json files from directory."""
    results = {}
    for run_name in RUN_CONFIGS:
        path = Path(metrics_dir) / f"metrics_{run_name}.json"
        if path.exists():
            with open(path) as f:
                results[run_name] = json.load(f)
        else:
            print(f"  [WARN] Missing: {path}")
    return results


def generate_text_table(results: dict[str, dict]) -> str:
    """Generate a text table for terminal display."""
    proposed = results.get('run_10', {})
    proposed_tps = proposed.get('tokens_per_sec_cpu', 1)

    header = (
        f"{'Run':<8} {'Mixer':<12} {'Tok':<6} {'Quant':<10} "
        f"{'Signal':<12} {'PPL':>7} {'TPS':>7} {'RAM':>6} {'vs Prop':>8}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for run_name in sorted(results.keys()):
        metrics = results[run_name]
        cfg = RUN_CONFIGS.get(run_name, {})
        tps = metrics.get('tokens_per_sec_cpu', 0)
        ppl = metrics.get('perplexity', 0)
        ram = metrics.get('ram_mb', 0)
        ratio = tps / proposed_tps * 100 if proposed_tps > 0 else 0
        marker = " <-" if cfg.get('is_proposed') else ""

        row = (
            f"{run_name:<8} "
            f"{cfg.get('token_mixer', '?'):<12} "
            f"{cfg.get('tokenization', '?'):<6} "
            f"{cfg.get('quantization', '?'):<10} "
            f"{cfg.get('training_signal', '?'):<12} "
            f"{ppl:>7.2f} "
            f"{tps:>7.1f} "
            f"{ram:>6.1f} "
            f"{ratio:>7.1f}%"
            f"{marker}"
        )
        rows.append(row)

    return "\n".join(rows)


def generate_latex_table(results: dict[str, dict]) -> str:
    """Generate LaTeX table ready for paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study at 50M parameters, 1B tokens. "
        r"TPS = tokens per second on consumer CPU (4-core). "
        r"$\downarrow$ lower is better, $\uparrow$ higher is better.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lllllrrr}",
        r"\toprule",
        r"Run & Mixer & Tok. & Quant. & Signal "
        r"& PPL$\downarrow$ & TPS$\uparrow$ & RAM$\downarrow$ \\",
        r"\midrule",
    ]

    for run_name in sorted(results.keys()):
        metrics = results[run_name]
        cfg = RUN_CONFIGS.get(run_name, {})
        tps = metrics.get('tokens_per_sec_cpu', 0)
        ppl = metrics.get('perplexity', 0)
        ram = metrics.get('ram_mb', 0)
        bold = cfg.get('is_proposed', False)

        mixer = cfg.get('token_mixer', '?').replace('_', '-')
        tok = cfg.get('tokenization', '?')
        quant = cfg.get('quantization', '?').replace('_', '-')
        signal = cfg.get('training_signal', '?').replace('_', '-')

        if bold:
            row = (
                f"\\textbf{{{run_name}}} & \\textbf{{{mixer}}} & "
                f"\\textbf{{{tok}}} & \\textbf{{{quant}}} & "
                f"\\textbf{{{signal}}} & \\textbf{{{ppl:.2f}}} & "
                f"\\textbf{{{tps:.1f}}} & \\textbf{{{ram:.0f}}} \\\\"
            )
        else:
            row = (
                f"{run_name} & {mixer} & {tok} & {quant} & "
                f"{signal} & {ppl:.2f} & {tps:.1f} & {ram:.0f} \\\\"
            )
        lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    return "\n".join(lines)


def generate_comparison_summary(results: dict[str, dict]) -> str:
    """Generate human-readable comparison of key design decisions."""
    lines = ["=" * 60, "ABLATION ANALYSIS — Key Comparisons", "=" * 60, ""]

    def _compare(label, run_a, run_b):
        if run_a not in results or run_b not in results:
            return f"  {label}: MISSING DATA ({run_a} or {run_b})\n"
        a = results[run_a]
        b = results[run_b]
        ppl_a, ppl_b = a.get('perplexity', 0), b.get('perplexity', 0)
        tps_a, tps_b = a.get('tokens_per_sec_cpu', 0), b.get('tokens_per_sec_cpu', 0)
        ppl_delta = ((ppl_b - ppl_a) / max(ppl_a, 1e-6)) * 100
        tps_delta = ((tps_b - tps_a) / max(tps_a, 1e-6)) * 100
        return (
            f"  {label}:\n"
            f"    {run_a}: PPL={ppl_a:.2f}, TPS={tps_a:.1f}\n"
            f"    {run_b}: PPL={ppl_b:.2f}, TPS={tps_b:.1f}\n"
            f"    Delta: PPL {ppl_delta:+.1f}%, TPS {tps_delta:+.1f}%\n"
        )

    lines.append("1. MLGRU vs Transformer (BPE FP16, CE):")
    lines.append(_compare("  Mixer effect", 'run_01', 'run_05'))

    lines.append("2. Byte vs BPE (Transformer FP16, CE):")
    lines.append(_compare("  Tokenization effect", 'run_01', 'run_02'))

    lines.append("3. Ternary vs FP16 (Transformer BPE, CE):")
    lines.append(_compare("  Quantization effect", 'run_01', 'run_03'))

    lines.append("4. Distillation vs CE (Transformer BPE ternary):")
    lines.append(_compare("  Training signal effect", 'run_03', 'run_04'))

    lines.append("5. Full proposal vs baseline:")
    lines.append(_compare("  CPU-1 vs Transformer baseline", 'run_01', 'run_10'))

    lines.append("6. FPResidual ablation:")
    lines.append(_compare("  With vs without FPResidual", 'run_11', 'run_10'))

    lines.append("7. Distillation effect on CPU-1:")
    lines.append(_compare("  CE vs distillation (MLGRU+Byte+Ternary)",
                           'run_09', 'run_10'))

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze CPU-1 ablation results"
    )
    parser.add_argument('--metrics_dir', type=str, default='.',
                        help="Directory containing metrics_run_XX.json files")
    parser.add_argument('--results_json', type=str, default=None,
                        help="Single JSON file with all results")
    parser.add_argument('--output', type=str, default='ablation_table.tex',
                        help="Output LaTeX file")
    args = parser.parse_args()

    if args.results_json:
        with open(args.results_json) as f:
            results = json.load(f)
    else:
        results = load_all_metrics(args.metrics_dir)

    if not results:
        print("No results found. Run ablation_train.py first.")
        return

    print(f"\nLoaded {len(results)}/{len(RUN_CONFIGS)} runs\n")

    # Text table
    text_table = generate_text_table(results)
    print(text_table)

    # Comparison summary
    summary = generate_comparison_summary(results)
    print(f"\n{summary}")

    # LaTeX table
    latex_table = generate_latex_table(results)
    with open(args.output, 'w') as f:
        f.write(latex_table)
    print(f"\nLaTeX table saved to {args.output}")

    # Save all results to single JSON
    combined_path = Path(args.metrics_dir) / "ablation_results.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Combined results saved to {combined_path}")


if __name__ == "__main__":
    main()
