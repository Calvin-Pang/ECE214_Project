"""Print a summary table of all completed experiment results.

Usage:
    uv run summary.py
    uv run summary.py --out-dir /path/to/results
"""

import json
from pathlib import Path

import typer

VARIANTS = {
    "pncc":          13,
    "pncc_39":       39,
    "pncc_42":       42,
    "mfcc":          13,
    "mfcc_39":       39,
    "pncc_mfcc_78":  78,
    "pncc_n40":     120,
    "pncc_contrast":     60,
    "pncc_n40_contrast": 141,
}

app = typer.Typer(add_completion=False)


@app.command()
def main(out_dir: Path = typer.Option(Path("results"), help="Root output directory")) -> None:
    header = f"{'variant':<18} {'F':>4}  {'clean':>6}  {'10dB':>6}  {'5dB':>6}"
    print(header)
    print("-" * len(header))

    for v, feat_dim in VARIANTS.items():
        metrics_path = out_dir / v / "metrics.json"

        if not metrics_path.exists():
            print(f"{v:<18}  (missing)")
            continue

        d = json.loads(metrics_path.read_text())["final"]
        print(
            f"{v:<18} {feat_dim:>4}  "
            f"{d['clean']:>6.4f}  {d['10db']:>6.4f}  {d['5db']:>6.4f}"
        )


if __name__ == "__main__":
    app()
