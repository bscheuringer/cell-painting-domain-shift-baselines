#!/usr/bin/env python
"""
Aggregate results.json files and produce comparison tables and plots.

Usage
-----
# Aggregate all results in the default directory
python scripts/evaluate_results.py

# Filter to a specific fold strategy
python scripts/evaluate_results.py --fold_strategy k5
python scripts/evaluate_results.py --fold_strategy lobo

# Custom path
python scripts/evaluate_results.py \\
    --results_dir outputs/results/2026-03-22_K5_SEED_42 \\
    --output_dir  outputs/analysis

# Combine results across multiple seed runs
python scripts/evaluate_results.py \\
    --results_dir outputs/results/K5_SEED_42 \\
                  outputs/results/K5_SEED_123 \\
                  outputs/results/K5_SEED_456 \\
    --output_dir  outputs/analysis/combined_K5 \\
    --fold_strategy k5

Outputs saved to outputs/analysis/
  results_table.csv        — per-fold test_acc (method × fold)
  summary_table.csv        — mean / std / min / max / worst_domain / best_domain per method
  summary_table.txt        — human-readable summary + significance tests
  significance_tests.csv   — pairwise Wilcoxon signed-rank p-values
  bar_plot.pdf/.png        — mean ± std test_acc per method
  box_plot.pdf/.png        — distribution of fold test_acc per method
  domain_heatmap.pdf/.png  — per-domain accuracy heatmap (method × domain)

PDF files are vector graphics — use these in Overleaf/LaTeX:
  \\includegraphics[width=\\columnwidth]{bar_plot.pdf}
  \\includegraphics[width=\\textwidth]{domain_heatmap.pdf}

Use --style screen for wider figures suited to notebooks/slides.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_RESULTS_DIR = REPO_ROOT / "outputs" / "results"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "analysis"

METHOD_ORDER = ["erm", "dann", "instance_norm"]
METHOD_LABELS = {
    "erm": "ERM",
    "dann": "DANN",
    "instance_norm": "InstanceNorm",
}
COLORS = {
    "erm": "#4C72B0",
    "dann": "#DD8452",
    "instance_norm": "#55A868",
}

# ---------------------------------------------------------------------------
# Style configuration
# "paper" targets a two-column A4 / IEEE / NeurIPS layout.
# ---------------------------------------------------------------------------
STYLES: dict[str, dict] = {
    "paper": {
        "figsize_single":      (3.3, 2.6),   # one column  (~\columnwidth in most templates)
        "figsize_wide":        (6.8, 3.2),   # full text width (two columns)
        "figsize_heatmap_col": 1.5,          # width per method column in heatmap
        "figsize_heatmap_row": 0.45,         # height per domain row in heatmap
        "dpi":                 300,
        "fontsize":            9,
        "mpl_style":           "seaborn-v0_8-paper",
    },
    "screen": {
        "figsize_single":      (7, 4),
        "figsize_wide":        (10, 4),
        "figsize_heatmap_col": 2.0,
        "figsize_heatmap_row": 0.6,
        "dpi":                 150,
        "fontsize":            10,
        "mpl_style":           "seaborn-v0_8-whitegrid",
    },
}


def _apply_style(style: dict) -> None:
    """Apply matplotlib style and font sizes."""
    try:
        plt.style.use(style["mpl_style"])
    except OSError:
        pass  # style not available — use default
    plt.rcParams.update({
        "font.size":        style["fontsize"],
        "axes.titlesize":   style["fontsize"],
        "axes.labelsize":   style["fontsize"],
        "xtick.labelsize":  style["fontsize"] - 1,
        "ytick.labelsize":  style["fontsize"] - 1,
        "legend.fontsize":  style["fontsize"] - 1,
        "figure.dpi":       style["dpi"],
    })


def _savefig(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> list[str]:
    """Save as both PDF (vector, for LaTeX) and PNG (raster preview)."""
    saved = []
    for ext in ("pdf", "png"):
        p = output_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        saved.append(f"{stem}.{ext}")
    return saved


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_results(results_dirs: Path | list[Path], fold_strategy: str | None) -> pd.DataFrame:
    if isinstance(results_dirs, Path):
        results_dirs = [results_dirs]
    files = []
    for d in results_dirs:
        files.extend(sorted(d.glob("*.json")))
    if not files:
        raise FileNotFoundError(
            f"No result files found in {results_dirs}\n"
            "Run experiments first with scripts/run_all_experiments.py"
        )

    rows = []
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        if fold_strategy and d.get("fold_strategy") != fold_strategy:
            continue
        row = {
            "run_name":      d["run_name"],
            "method":        d["method"],
            "fold_strategy": d.get("fold_strategy", ""),
            "fold_index":    d["fold_index"],
        }
        for metric, val in d.get("test_metrics", {}).items():
            row[metric] = val
        rows.append(row)

    if not rows:
        strategies = {json.load(open(f)).get("fold_strategy") for f in files}
        raise ValueError(
            f"No results match fold_strategy={fold_strategy!r}. "
            f"Available strategies: {strategies}"
        )

    df = pd.DataFrame(rows)
    # Ensure consistent method ordering
    known = [m for m in METHOD_ORDER if m in df["method"].unique()]
    other = [m for m in df["method"].unique() if m not in METHOD_ORDER]
    df["method"] = pd.Categorical(df["method"], categories=known + other, ordered=True)
    return df.sort_values(["method", "fold_index"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _domain_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("test_acc_domain_")]


def _domain_name(col: str) -> str:
    return col.replace("test_acc_domain_", "")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    domain_cols = _domain_cols(df)
    rows = []
    for method, grp in df.groupby("method", observed=True):
        acc = grp["test_acc"].dropna()
        row: dict = {
            "method":    method,
            "n_folds":   len(acc),
            "mean_acc":  acc.mean(),
            "std_acc":   acc.std(ddof=1) if len(acc) > 1 else float("nan"),
            "min_acc":   acc.min(),
            "max_acc":   acc.max(),
        }
        if domain_cols:
            # Per-fold worst / best domain accuracy
            fold_worst, fold_best = [], []
            for _, frow in grp.iterrows():
                vals = [frow[c] for c in domain_cols if pd.notna(frow[c])]
                if vals:
                    fold_worst.append(min(vals))
                    fold_best.append(max(vals))
            row["mean_worst_domain_acc"] = float(np.mean(fold_worst)) if fold_worst else float("nan")
            row["mean_best_domain_acc"]  = float(np.mean(fold_best))  if fold_best  else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("method")


def pairwise_significance(df: pd.DataFrame) -> pd.DataFrame | None:
    """Run Wilcoxon signed-rank tests on all method pairs.

    Pairs observations by (seed, fold_index) so that multi-seed runs
    produce one paired observation per (seed, fold) combination rather
    than averaging across seeds.

    Returns a DataFrame with columns: method_a, method_b, n, mean_diff,
    p_value, significant_005.  Returns None if scipy is unavailable or
    fewer than 6 paired observations exist.
    """
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return None

    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    if len(methods) < 2:
        return None

    # Build a pivot that keeps each (seed, fold_index) as a separate row
    has_seed = "seed" in df.columns or any("seed" in str(c) for c in df.columns)
    if "seed" in df.columns:
        index_cols = ["seed", "fold_index"]
    else:
        # Fallback: use the DataFrame index to distinguish duplicate fold_index values
        df = df.copy()
        df["_obs_id"] = range(len(df))
        index_cols = ["_obs_id"]

    rows = []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            a_vals = df[df["method"] == a][["fold_index", "test_acc"]].reset_index(drop=True)
            b_vals = df[df["method"] == b][["fold_index", "test_acc"]].reset_index(drop=True)

            # Pair by position (same ordering guaranteed by load_results sort)
            n = min(len(a_vals), len(b_vals))
            if n < 6:
                continue
            a_acc = a_vals["test_acc"].values[:n]
            b_acc = b_vals["test_acc"].values[:n]
            diff = a_acc - b_acc
            if (diff == 0).all():
                p = 1.0
            else:
                _, p = wilcoxon(a_acc, b_acc, alternative="two-sided")
            rows.append({
                "method_a": a,
                "method_b": b,
                "n": n,
                "mean_diff": float(diff.mean()),
                "p_value": float(p),
                "significant_005": p < 0.05,
            })
    return pd.DataFrame(rows) if rows else None


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(
        index="fold_index", columns="method", values="test_acc", observed=True
    )


def build_domain_pivot(df: pd.DataFrame) -> pd.DataFrame | None:
    domain_cols = _domain_cols(df)
    if not domain_cols:
        return None
    rows = []
    for method, grp in df.groupby("method", observed=True):
        for col in domain_cols:
            vals = grp[col].dropna()
            if not vals.empty:
                rows.append({
                    "method": method,
                    "domain": _domain_name(col),
                    "mean_acc": vals.mean(),
                })
    if not rows:
        return None
    pivot = pd.DataFrame(rows).pivot(index="domain", columns="method", values="mean_acc")
    # Keep method column order
    present = [m for m in METHOD_ORDER if m in pivot.columns]
    return pivot[present]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_bar(summary: pd.DataFrame, output_dir: Path, style: dict) -> list[str]:
    methods = list(summary.index)
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    means  = summary["mean_acc"].values
    stds   = summary["std_acc"].fillna(0).values
    colors = [COLORS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(figsize=style["figsize_single"])
    bars = ax.bar(labels, means, yerr=stds, capsize=4, color=colors,
                  edgecolor="white", linewidth=0.8, error_kw={"elinewidth": 1.2})
    ax.set_ylabel("Test Accuracy (macro)")
    ax.set_title("Method Comparison — Mean ± Std across Folds")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.set_ylim(max(0, means.min() - 0.1), min(1.0, means.max() + 0.09))
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=style["fontsize"] - 1)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    saved = _savefig(fig, output_dir, "bar_plot", style["dpi"])
    plt.close(fig)
    return saved


def plot_box(df: pd.DataFrame, output_dir: Path, style: dict) -> list[str]:
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    data    = [df[df["method"] == m]["test_acc"].dropna().values for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    colors  = [COLORS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(figsize=style["figsize_single"])
    bp = ax.boxplot(data, patch_artist=True, labels=labels,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    for i, (vals, color) in enumerate(zip(data, colors), 1):
        x = np.random.normal(i, 0.06, size=len(vals))
        ax.scatter(x, vals, color=color, s=20, zorder=3, edgecolors="white", linewidths=0.4)
    ax.set_ylabel("Test Accuracy (macro)")
    ax.set_title("Method Comparison — Fold Distribution")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=1))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    saved = _savefig(fig, output_dir, "box_plot", style["dpi"])
    plt.close(fig)
    return saved


def plot_domain_heatmap(domain_pivot: pd.DataFrame, output_dir: Path, style: dict) -> list[str]:
    col_labels = [METHOD_LABELS.get(c, c) for c in domain_pivot.columns]
    data = domain_pivot.values
    n_cols, n_rows = len(col_labels), len(domain_pivot)

    fw = max(style["figsize_wide"][0], n_cols * style["figsize_heatmap_col"])
    fh = max(2.5, n_rows * style["figsize_heatmap_row"])
    fig, ax = plt.subplots(figsize=(fw, fh))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean Accuracy", shrink=0.8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(domain_pivot.index)
    ax.set_title("Per-Domain Accuracy (mean across folds)")

    cell_fs = max(6, style["fontsize"] - 2)
    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            v = data[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=cell_fs, color="black" if 0.3 < v < 0.8 else "white")
    fig.tight_layout()
    saved = _savefig(fig, output_dir, "domain_heatmap", style["dpi"])
    plt.close(fig)
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate experiment results and generate plots.")
    p.add_argument("--results_dir", nargs="+", default=[str(DEFAULT_RESULTS_DIR)],
                   help="One or more directories containing results.json files. "
                        "Pass multiple dirs to aggregate across seeds, e.g. "
                        "--results_dir outputs/results/K5_SEED_42 outputs/results/K5_SEED_123")
    p.add_argument("--output_dir",  default=str(DEFAULT_OUTPUT_DIR),
                   help=f"Where to save analysis outputs (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--fold_strategy", default=None,
                   help="Filter by fold strategy: 'k5' or 'lobo' (default: all)")
    p.add_argument("--style", default="paper", choices=["paper", "screen"],
                   help="Plot style: 'paper' (3.3\" column-width PDF, 300 DPI) or "
                        "'screen' (wide PNG, 150 DPI). Default: paper")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    results_dirs = [Path(d) for d in args.results_dir]
    output_dir   = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style = STYLES[args.style]
    _apply_style(style)
    print(f"Plot style: {args.style} "
          f"(figsize={style['figsize_single']}, dpi={style['dpi']}, "
          f"outputs: PDF + PNG)\n")

    # ---- Load ----
    dirs_str = ", ".join(str(d) for d in results_dirs)
    print(f"Loading results from: {dirs_str}")
    df = load_results(results_dirs, fold_strategy=args.fold_strategy)
    n_runs    = len(df)
    methods   = df["method"].unique().tolist()
    strategies = df["fold_strategy"].unique().tolist()
    print(f"  {n_runs} runs  |  methods: {methods}  |  strategies: {strategies}\n")

    # ---- Tables ----
    summary = build_summary(df)
    pivot   = build_pivot(df)
    domain_pivot = build_domain_pivot(df)

    # Save CSVs
    pivot.to_csv(output_dir / "results_table.csv")
    summary.to_csv(output_dir / "summary_table.csv")

    # Human-readable summary
    float_cols = summary.select_dtypes("float").columns
    txt_summary = summary.copy()
    for col in float_cols:
        txt_summary[col] = txt_summary[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    # ---- Statistical significance (Wilcoxon signed-rank) ----
    sig_df = pairwise_significance(df)
    sig_txt = ""
    if sig_df is not None and not sig_df.empty:
        sig_df.to_csv(output_dir / "significance_tests.csv", index=False)
        sig_txt = "\n\nPairwise Wilcoxon signed-rank tests:\n\n"
        for _, row in sig_df.iterrows():
            a_label = METHOD_LABELS.get(row["method_a"], row["method_a"])
            b_label = METHOD_LABELS.get(row["method_b"], row["method_b"])
            star = "*" if row["significant_005"] else " "
            sig_txt += (
                f"  {a_label:14s} vs {b_label:14s}  "
                f"n={row['n']:2d}  Δ={row['mean_diff']:+.4f}  "
                f"p={row['p_value']:.4f} {star}\n"
            )
        sig_txt += "\n  * significant at α=0.05\n"

    summary_txt = (
        "Method Comparison Summary\n"
        "=========================\n\n"
        + txt_summary.to_string()
        + "\n\nPer-fold test accuracy:\n\n"
        + pivot.to_string(float_format=lambda x: f"{x:.4f}")
        + sig_txt
        + "\n"
    )
    (output_dir / "summary_table.txt").write_text(summary_txt)

    # Print to stdout
    print(summary_txt)

    # ---- Plots ----
    print("Generating plots...")
    saved = plot_bar(summary, output_dir, style)
    print(f"  {', '.join(saved)}")

    saved = plot_box(df, output_dir, style)
    print(f"  {', '.join(saved)}")

    if domain_pivot is not None:
        saved = plot_domain_heatmap(domain_pivot, output_dir, style)
        print(f"  {', '.join(saved)}")
    else:
        print("  (domain_heatmap skipped — no per-domain metrics found)")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
