#!/usr/bin/env python
"""
Analyze per-batch image statistics of the JUMP-CP dataset (Source 3, positives
only) to understand what visual differences drive the observed domain shift in
the K9 Leave-One-Batch-Out experiments.

For each selected batch a random sample of N images is loaded.  Per image and
per channel the following features are computed:

  Intensity statistics:
    mean, std, median, p5, p95, IQR,
    dynamic_range (p95 − p5),
    sat_low  (fraction of pixels < 0.02 — near-black / unexposed),
    sat_high (fraction of pixels > 0.98 — near-white / saturated)

  Image-quality metrics:
    sharpness   — variance of the discrete Laplacian (focus quality)
    hf_ratio    — fraction of FFT power in high-frequency components

  Cross-channel correlations:
    Pearson r for all 10 channel pairs (AGP/DNA/ER/Mito/RNA)

All features are aggregated per batch (mean ± std across sampled images) and
exported to a CSV.  A suite of plots is saved alongside.  Finally a
statistical comparison (KS test + Wasserstein distance) between the hardest
and easiest batches is printed to stdout and written to a text file.

Usage
-----
  python scripts/analyze_domain_shift.py \\
      --parquet  data/indices/JKU_JUMPCP_Source3_AllPositives.parquet \\
      --img_root /path/to/jumpcp/images \\
      --n_samples 200

  # Restrict to specific batches
  python scripts/analyze_domain_shift.py \\
      --parquet  data/indices/JKU_JUMPCP_Source3_AllPositives.parquet \\
      --img_root /path/to/jumpcp/images \\
      --batches CP59 CP60 CP_25_all_Phenix1 CP_31_all_Phenix1 CP_32_all_Phenix1 \\
      --n_samples 300

  # Load LOBO accuracy from a specific results directory
  python scripts/analyze_domain_shift.py \\
      --parquet     data/indices/JKU_JUMPCP_Source3_AllPositives.parquet \\
      --img_root    /path/to/jumpcp/images \\
      --lobo_dir    outputs/results/2026-03-23_17-43-48_K9_SEED_42 \\
      --output_dir  outputs/domain_shift_analysis

Outputs saved to --output_dir (default: outputs/domain_shift_analysis):
  per_image_features.csv          — raw per-image feature vectors
  batch_aggregate_stats.csv       — mean ± std of every feature per batch
  hard_vs_easy_statistics.csv     — KS test + Wasserstein distance per feature
  domain_shift_summary.txt        — plain-language findings
  intensity_boxplots.{pdf,png}    — per-channel intensity distributions per batch
  mean_intensity_heatmap.{pdf,png}— batch × channel heatmap of mean intensities
  sharpness.{pdf,png}             — Laplacian variance per batch × channel
  dynamic_range.{pdf,png}         — p95−p5 dynamic range per batch × channel
  saturation.{pdf,png}            — near-black / near-white pixel fractions
  intensity_histograms.{pdf,png}  — intensity distribution overlays per channel
  cross_channel_correlation.{pdf,png} — median Pearson r heatmap
  pca.{pdf,png}                   — PCA of per-image feature vectors
  acc_vs_*.{pdf,png}              — LOBO accuracy vs feature scatter plots
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from scipy import ndimage as ndi
from scipy import stats as sp_stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).parent.parent.resolve()

# ── constants ──────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["AGP", "DNA", "ER", "Mito", "RNA"]
N_CHANNELS = 5

# ERM LOBO test accuracy per batch from the 2026-03-23 K9 run (fold 0-8).
# Used as fallback when --lobo_dir is not provided.
_DEFAULT_LOBO_ERM_ACC: dict[str, float] = {
    "CP59":               0.8276,
    "CP60":               0.9385,
    "CP_25_all_Phenix1":  0.9608,
    "CP_26_all_Phenix1":  0.8717,
    "CP_27_all_Phenix1":  0.9189,
    "CP_28_all_Phenix1":  0.8399,
    "CP_29_all_Phenix1":  0.9114,
    "CP_31_all_Phenix1":  0.7754,
    "CP_32_all_Phenix1":  0.7620,
}

# ── image loading ──────────────────────────────────────────────────────────


def _read_tiff(path: str) -> np.ndarray:
    """Load a 5-channel TIFF image (stored with .jpg extension by JUMP-CP).

    Returns a float32 array of shape (H, W, 5) scaled to [0, 1].
    Falls back to a pre-converted .npy file when present (faster).
    Matches the logic in ``src/data/dataset.py::JumpCPDataset._read_image``.
    """
    npy_path = path.replace(".jpg", ".npy")
    try:
        img = np.load(npy_path, allow_pickle=False)
    except FileNotFoundError:
        img = tifffile.imread(str(path), maxworkers=1)
    if img.ndim != 3 or img.shape[-1] != N_CHANNELS:
        raise ValueError(
            f"Expected (H, W, {N_CHANNELS}) image, got {img.shape} at {path}"
        )
    return img.astype(np.float32) / 255.0


# ── feature extraction ─────────────────────────────────────────────────────


def _laplacian_variance(channel: np.ndarray) -> float:
    """Variance of the discrete Laplacian — a proxy for image sharpness.

    Higher values indicate sharper, better-focused images with more
    high-frequency edge detail.
    """
    return float(np.var(ndi.laplace(channel)))


def _high_freq_ratio(channel: np.ndarray) -> float:
    """Fraction of FFT power in frequencies above the median radial frequency.

    High values indicate fine-grained texture or sharp edges; low values
    indicate blurry or low-contrast images dominated by low frequencies.
    """
    f_shift = np.fft.fftshift(np.abs(np.fft.fft2(channel)) ** 2)
    total = f_shift.sum()
    if total == 0.0:
        return 0.0
    cy, cx = np.array(f_shift.shape) // 2
    Y, X = np.ogrid[: f_shift.shape[0], : f_shift.shape[1]]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return float(f_shift[r > float(np.median(r))].sum() / total)


def extract_image_features(img: np.ndarray) -> dict[str, float]:
    """Extract all scalar features from a single (H, W, 5) float32 image."""
    feats: dict[str, float] = {}

    for c, name in enumerate(CHANNEL_NAMES):
        ch = img[:, :, c]
        p5, p25, p50, p75, p95 = np.percentile(ch, [5, 25, 50, 75, 95])
        feats[f"{name}_mean"] = float(ch.mean())
        feats[f"{name}_std"] = float(ch.std())
        feats[f"{name}_median"] = float(p50)
        feats[f"{name}_p5"] = float(p5)
        feats[f"{name}_p95"] = float(p95)
        feats[f"{name}_iqr"] = float(p75 - p25)
        feats[f"{name}_dynamic_range"] = float(p95 - p5)
        feats[f"{name}_sat_low"] = float((ch < 0.02).mean())
        feats[f"{name}_sat_high"] = float((ch > 0.98).mean())
        feats[f"{name}_sharpness"] = _laplacian_variance(ch)
        feats[f"{name}_hf_ratio"] = _high_freq_ratio(ch)

    # Cross-channel Pearson correlations (10 pairs from 5 channels)
    for (i, ni), (j, nj) in combinations(enumerate(CHANNEL_NAMES), 2):
        r, _ = sp_stats.pearsonr(img[:, :, i].ravel(), img[:, :, j].ravel())
        feats[f"corr_{ni}_{nj}"] = float(r)

    return feats


# ── batch-level feature extraction ────────────────────────────────────────


def compute_batch_features(
    df: pd.DataFrame,
    img_root: Path,
    batch_col: str,
    sample_col: str,
    ext: str,
    n_samples: int,
    rng: np.random.Generator,
    nested_by_batch: bool = False,
) -> pd.DataFrame:
    """Sample N images per batch and return a DataFrame of per-image features.

    Columns: all feature names plus ``batch_col`` and ``sample_col``.
    Images that cannot be read (missing file, wrong shape) are silently
    skipped; a warning is printed at the end of each batch.
    """
    records: list[dict] = []
    for batch in sorted(df[batch_col].unique()):
        sub = df[df[batch_col] == batch]
        n = min(n_samples, len(sub))
        sampled = sub.sample(n=n, random_state=int(rng.integers(0, 2**31)))
        print(
            f"  batch={batch:<30s}  sampling {n:4d} / {len(sub):5d} images …",
            flush=True,
        )
        missing = 0
        for _, row in sampled.iterrows():
            sid = row[sample_col]
            if nested_by_batch:
                path = str(img_root / batch / sid[-1] / f"{sid}{ext}")
            else:
                path = str(img_root / f"{sid}{ext}")
            try:
                img = _read_tiff(path)
            except (FileNotFoundError, ValueError):
                missing += 1
                continue
            feats = extract_image_features(img)
            feats[batch_col] = batch
            feats[sample_col] = sid
            records.append(feats)
        if missing:
            print(
                f"    WARNING: {missing} image(s) could not be read and were skipped.",
                file=sys.stderr,
            )
    return pd.DataFrame(records)


# ── colour and label helpers ───────────────────────────────────────────────


def _batch_label(name: str) -> str:
    """Shorten batch name for axis labels (strip common suffix)."""
    return name.replace("_all_Phenix1", "").replace("_", " ")


def _accuracy_palette(batches: list[str], lobo_acc: dict[str, float]) -> list:
    """Map ERM LOBO accuracy to a RdYlGn colour for each batch."""
    accs = [lobo_acc.get(b, float("nan")) for b in batches]
    finite = [a for a in accs if not np.isnan(a)]
    if not finite:
        return ["#cccccc"] * len(batches)
    lo, hi = min(finite), max(finite)
    cmap = plt.cm.RdYlGn
    return [
        cmap((a - lo) / (hi - lo + 1e-9)) if not np.isnan(a) else "#cccccc"
        for a in accs
    ]


def _add_accuracy_colorscale_legend(fig: plt.Figure) -> None:
    """Append a small legend explaining the RdYlGn colour scale."""
    from matplotlib.patches import Patch

    patches = [
        Patch(facecolor=plt.cm.RdYlGn(0.0), label="Low ERM accuracy  (hard)"),
        Patch(facecolor=plt.cm.RdYlGn(0.5), label="Medium ERM accuracy"),
        Patch(facecolor=plt.cm.RdYlGn(1.0), label="High ERM accuracy (easy)"),
    ]
    fig.legend(
        handles=patches,
        loc="upper right",
        fontsize=7,
        framealpha=0.8,
        ncol=1,
    )


def _savefig(fig: plt.Figure, out_dir: Path, stem: str, dpi: int = 150) -> None:
    """Save figure as both PDF (vector) and PNG (raster preview)."""
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")


# ── individual plots ───────────────────────────────────────────────────────


def plot_intensity_boxplots(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """One panel per channel: per-batch distribution of per-image mean intensity."""
    batches = sorted(feat_df[batch_col].unique())
    labels = [_batch_label(b) for b in batches]
    colors = _accuracy_palette(batches, lobo_acc)

    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(18, 5), sharey=False)
    for ax, ch in zip(axes, CHANNEL_NAMES):
        data = [feat_df.loc[feat_df[batch_col] == b, f"{ch}_mean"].values for b in batches]
        bp = ax.boxplot(
            data,
            patch_artist=True,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.0},
            showfliers=False,
            whis=(10, 90),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        if ch == CHANNEL_NAMES[0]:
            ax.set_ylabel("Mean intensity (0–1)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    _add_accuracy_colorscale_legend(fig)
    fig.suptitle(
        "Per-channel mean intensity distribution per batch\n"
        "(box = Q1–Q3, whiskers = 10th–90th percentile)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, out_dir, "intensity_boxplots")
    plt.close(fig)


def plot_mean_intensity_heatmap(
    feat_df: pd.DataFrame, batch_col: str, out_dir: Path
) -> None:
    """Heatmap of mean intensity: rows = batches, columns = channels."""
    agg = feat_df.groupby(batch_col)[[f"{c}_mean" for c in CHANNEL_NAMES]].mean()
    agg.index = [_batch_label(b) for b in agg.index]
    agg.columns = CHANNEL_NAMES

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        agg,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Mean intensity"},
    )
    ax.set_title("Mean channel intensity per batch", fontsize=12)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Batch")
    plt.tight_layout()
    _savefig(fig, out_dir, "mean_intensity_heatmap")
    plt.close(fig)


def plot_sharpness(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """Bar chart: median Laplacian variance (sharpness) per batch and channel."""
    batches = sorted(feat_df[batch_col].unique())
    labels = [_batch_label(b) for b in batches]
    colors = _accuracy_palette(batches, lobo_acc)

    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(18, 4), sharey=False)
    for ax, ch in zip(axes, CHANNEL_NAMES):
        vals = [
            feat_df.loc[feat_df[batch_col] == b, f"{ch}_sharpness"].median()
            for b in batches
        ]
        ax.bar(labels, vals, color=colors, edgecolor="grey", linewidth=0.4)
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        if ch == CHANNEL_NAMES[0]:
            ax.set_ylabel("Laplacian variance (sharpness)")

    _add_accuracy_colorscale_legend(fig)
    fig.suptitle("Image sharpness (Laplacian variance) per batch and channel", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, out_dir, "sharpness")
    plt.close(fig)


def plot_dynamic_range(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """Bar chart: median dynamic range (p95 − p5) per batch and channel."""
    batches = sorted(feat_df[batch_col].unique())
    labels = [_batch_label(b) for b in batches]
    colors = _accuracy_palette(batches, lobo_acc)

    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(18, 4), sharey=False)
    for ax, ch in zip(axes, CHANNEL_NAMES):
        vals = [
            feat_df.loc[feat_df[batch_col] == b, f"{ch}_dynamic_range"].median()
            for b in batches
        ]
        ax.bar(labels, vals, color=colors, edgecolor="grey", linewidth=0.4)
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        if ch == CHANNEL_NAMES[0]:
            ax.set_ylabel("Dynamic range (p95 − p5)")

    _add_accuracy_colorscale_legend(fig)
    fig.suptitle("Dynamic range (p95 − p5) per batch and channel", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fig, out_dir, "dynamic_range")
    plt.close(fig)


def plot_saturation(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """Stacked bar: mean near-black and near-white pixel fractions per batch."""
    batches = sorted(feat_df[batch_col].unique())
    labels = [_batch_label(b) for b in batches]

    low_vals = [
        feat_df.loc[
            feat_df[batch_col] == b, [f"{c}_sat_low" for c in CHANNEL_NAMES]
        ].values.mean()
        for b in batches
    ]
    high_vals = [
        feat_df.loc[
            feat_df[batch_col] == b, [f"{c}_sat_high" for c in CHANNEL_NAMES]
        ].values.mean()
        for b in batches
    ]

    x = np.arange(len(batches))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, low_vals, label="Near-black (< 2 %)", color="#4C72B0")
    ax.bar(x, high_vals, bottom=low_vals, label="Near-white (> 98 %)", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of pixels")
    ax.set_title("Pixel saturation (mean over all channels) per batch", fontsize=12)
    ax.legend()
    plt.tight_layout()
    _savefig(fig, out_dir, "saturation")
    plt.close(fig)


def plot_intensity_histograms(
    feat_df: pd.DataFrame, batch_col: str, out_dir: Path
) -> None:
    """Overlaid histograms of per-image mean intensity per channel."""
    batches = sorted(feat_df[batch_col].unique())
    palette = sns.color_palette("tab10", n_colors=len(batches))

    fig, axes = plt.subplots(1, N_CHANNELS, figsize=(18, 4), sharey=False)
    for ax, ch in zip(axes, CHANNEL_NAMES):
        for b, col in zip(batches, palette):
            vals = feat_df.loc[feat_df[batch_col] == b, f"{ch}_mean"].values
            ax.hist(
                vals,
                bins=30,
                density=True,
                alpha=0.55,
                label=_batch_label(b),
                color=col,
                histtype="stepfilled",
                edgecolor="none",
            )
        ax.set_title(ch, fontsize=11, fontweight="bold")
        ax.set_xlabel("Mean intensity")
        if ch == CHANNEL_NAMES[0]:
            ax.set_ylabel("Density")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=palette[i]) for i in range(len(batches))]
    fig.legend(
        handles,
        [_batch_label(b) for b in batches],
        loc="lower center",
        ncol=min(len(batches), 5),
        fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )
    fig.suptitle(
        "Distribution of per-image mean intensities per batch", fontsize=12
    )
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    _savefig(fig, out_dir, "intensity_histograms")
    plt.close(fig)


def plot_cross_channel_correlation(
    feat_df: pd.DataFrame, batch_col: str, out_dir: Path
) -> None:
    """Heatmap of median cross-channel Pearson r per batch."""
    pairs = list(combinations(CHANNEL_NAMES, 2))
    corr_cols = [f"corr_{ni}_{nj}" for ni, nj in pairs]
    short_labels = [f"{ni[:1]}–{nj[:1]}" for ni, nj in pairs]

    agg = feat_df.groupby(batch_col)[corr_cols].median()
    agg.index = [_batch_label(b) for b in agg.index]
    agg.columns = short_labels

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(
        agg,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("Median cross-channel Pearson correlation per batch", fontsize=12)
    ax.set_xlabel("Channel pair")
    ax.set_ylabel("Batch")
    plt.tight_layout()
    _savefig(fig, out_dir, "cross_channel_correlation")
    plt.close(fig)


def plot_pca(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """PCA of per-image feature vectors, one scatter point per image."""
    exclude = {batch_col, "Metadata_Sample_ID"}
    feature_cols = [
        c
        for c in feat_df.columns
        if c not in exclude and pd.api.types.is_float_dtype(feat_df[c])
    ]
    X = feat_df[feature_cols].fillna(0.0).values
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)

    batches = sorted(feat_df[batch_col].unique())
    palette = sns.color_palette("tab10", n_colors=len(batches))
    color_map = dict(zip(batches, palette))

    fig, ax = plt.subplots(figsize=(9, 7))
    for b in batches:
        mask = feat_df[batch_col].values == b
        acc = lobo_acc.get(b)
        label = _batch_label(b) + (f"  (ERM {acc:.1%})" if acc is not None else "")
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=15,
            alpha=0.6,
            color=color_map[b],
            label=label,
            linewidths=0,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var.)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var.)")
    ax.set_title("PCA of per-image feature vectors — coloured by batch", fontsize=12)
    ax.legend(fontsize=8, loc="best", framealpha=0.7)
    plt.tight_layout()
    _savefig(fig, out_dir, "pca")
    plt.close(fig)


def plot_acc_vs_features(
    feat_df: pd.DataFrame,
    batch_col: str,
    out_dir: Path,
    lobo_acc: dict[str, float],
) -> None:
    """Scatter plots: ERM LOBO accuracy vs mean feature value per batch.

    Produces one figure per feature group (intensity, sharpness, …), each
    with one panel per channel.  A regression line with Pearson r is shown.
    """
    batches = sorted(feat_df[batch_col].unique())
    accs = np.array([lobo_acc.get(b, float("nan")) for b in batches])
    if np.all(np.isnan(accs)):
        return

    groups: dict[str, list[str]] = {
        "Mean intensity per channel": [f"{c}_mean" for c in CHANNEL_NAMES],
        "Std of intensities per channel": [f"{c}_std" for c in CHANNEL_NAMES],
        "Dynamic range (p95−p5) per channel": [
            f"{c}_dynamic_range" for c in CHANNEL_NAMES
        ],
        "Sharpness (Laplacian var.) per channel": [
            f"{c}_sharpness" for c in CHANNEL_NAMES
        ],
        "High-freq. energy ratio per channel": [
            f"{c}_hf_ratio" for c in CHANNEL_NAMES
        ],
    }
    palette = sns.color_palette("tab10", n_colors=N_CHANNELS)

    for group_title, cols in groups.items():
        fig, axes = plt.subplots(1, len(cols), figsize=(16, 4), sharey=True)
        for ax, col, color in zip(axes, cols, palette):
            vals = np.array(
                [feat_df.loc[feat_df[batch_col] == b, col].mean() for b in batches]
            )
            valid = ~(np.isnan(vals) | np.isnan(accs))
            ax.scatter(vals[valid], accs[valid], color=color, s=60, zorder=3)
            for v, a, b in zip(vals, accs, batches):
                if not (np.isnan(v) or np.isnan(a)):
                    ax.annotate(
                        _batch_label(b),
                        (v, a),
                        fontsize=6,
                        textcoords="offset points",
                        xytext=(4, 2),
                    )
            if valid.sum() >= 3 and vals[valid].std() > 0:
                m, intercept, r, p, _ = sp_stats.linregress(
                    vals[valid], accs[valid]
                )
                xl = np.array([vals[valid].min(), vals[valid].max()])
                ax.plot(xl, m * xl + intercept, "k--", lw=1,
                        label=f"r={r:.2f} (p={p:.2f})")
                ax.legend(fontsize=7)
            ax.set_title(col.split("_")[0], fontsize=10, fontweight="bold")
            ax.set_xlabel("Mean feature value")
        axes[0].set_ylabel("ERM LOBO test accuracy")
        axes[0].yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        fig.suptitle(group_title, fontsize=11)
        plt.tight_layout()
        safe_name = (
            group_title.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("−", "-")
            .replace(".", "")
            .replace("/", "_")
        )
        _savefig(fig, out_dir, f"acc_vs_{safe_name}")
        plt.close(fig)


# ── statistical comparison ─────────────────────────────────────────────────


def statistical_comparison(
    feat_df: pd.DataFrame,
    batch_col: str,
    lobo_acc: dict[str, float],
) -> pd.DataFrame:
    """KS test + Wasserstein distance between hard and easy batches per feature.

    Hard batches: lowest tertile of ERM accuracy.
    Easy batches: highest tertile of ERM accuracy.

    Returns a DataFrame sorted by Wasserstein distance (descending).
    """
    avail = sorted(
        [b for b in lobo_acc if b in feat_df[batch_col].values],
        key=lambda b: lobo_acc[b],
    )
    if len(avail) < 2:
        return pd.DataFrame()

    n_group = max(1, len(avail) // 3)
    hard_batches = avail[:n_group]
    easy_batches = avail[-n_group:]

    exclude = {batch_col, "Metadata_Sample_ID"}
    feat_cols = [
        c
        for c in feat_df.columns
        if c not in exclude and pd.api.types.is_float_dtype(feat_df[c])
    ]

    hard_df = feat_df[feat_df[batch_col].isin(hard_batches)]
    easy_df = feat_df[feat_df[batch_col].isin(easy_batches)]

    rows: list[dict] = []
    for col in feat_cols:
        h = hard_df[col].dropna().values
        e = easy_df[col].dropna().values
        if len(h) < 2 or len(e) < 2:
            continue
        ks_stat, ks_p = sp_stats.ks_2samp(h, e)
        w1 = sp_stats.wasserstein_distance(h, e)
        rows.append(
            {
                "feature": col,
                "hard_mean": float(h.mean()),
                "easy_mean": float(e.mean()),
                "mean_diff (easy-hard)": float(e.mean() - h.mean()),
                "hard_std": float(h.std()),
                "easy_std": float(e.std()),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "wasserstein_dist": float(w1),
                "hard_batches": ", ".join(_batch_label(b) for b in hard_batches),
                "easy_batches": ", ".join(_batch_label(b) for b in easy_batches),
            }
        )
    return pd.DataFrame(rows).sort_values("wasserstein_dist", ascending=False)


# ── aggregate stats ────────────────────────────────────────────────────────


def compute_aggregate_stats(feat_df: pd.DataFrame, batch_col: str) -> pd.DataFrame:
    """Mean and std of every feature, one row per batch."""
    exclude = {batch_col, "Metadata_Sample_ID"}
    feat_cols = [
        c
        for c in feat_df.columns
        if c not in exclude and pd.api.types.is_float_dtype(feat_df[c])
    ]
    agg = feat_df.groupby(batch_col)[feat_cols].agg(["mean", "std"])
    agg.columns = ["_".join(col) for col in agg.columns]
    return agg


# ── plain-language summary ─────────────────────────────────────────────────


def build_summary_text(
    feat_df: pd.DataFrame,
    batch_col: str,
    stat_df: pd.DataFrame,
    lobo_acc: dict[str, float],
) -> str:
    """Return a plain-language summary string of the key findings."""
    lines: list[str] = []
    sep = "=" * 70

    lines += [f"\n{sep}", "DOMAIN SHIFT ANALYSIS SUMMARY", sep]

    if lobo_acc:
        lines.append("\nK9 LOBO ERM test accuracy per batch (easiest → hardest):")
        for b, acc in sorted(lobo_acc.items(), key=lambda kv: -kv[1]):
            marker = "  ← easiest" if acc >= 0.93 else (
                "  ← hardest" if acc < 0.80 else ""
            )
            lines.append(f"  {b:<32s}  {acc:.3%}{marker}")

    n_images = len(feat_df)
    n_batches = feat_df[batch_col].nunique()
    lines.append(
        f"\nImages analysed: {n_images:,} from {n_batches} batches "
        f"({n_images // n_batches} on average per batch)."
    )

    if stat_df.empty:
        lines.append(
            "\n(No statistical comparison possible — fewer than 2 batches with "
            "known LOBO accuracy.)"
        )
    else:
        hard = stat_df["hard_batches"].iloc[0]
        easy = stat_df["easy_batches"].iloc[0]
        lines.append(f"\nHARD batches: {hard}")
        lines.append(f"EASY batches: {easy}")
        lines.append(
            "\nTop 20 most diverging features (ranked by Wasserstein distance):\n"
        )
        header = (
            f"  {'Feature':<42} {'Hard μ':>9} {'Easy μ':>9} "
            f"{'Δ (e-h)':>9} {'KS':>7} {'p':>7} {'W₁':>8}"
        )
        lines.append(header)
        lines.append("  " + "-" * 102)
        for _, row in stat_df.head(20).iterrows():
            lines.append(
                f"  {row['feature']:<42} "
                f"{row['hard_mean']:>9.4f} "
                f"{row['easy_mean']:>9.4f} "
                f"{row['mean_diff (easy-hard)']:>+9.4f} "
                f"{row['ks_stat']:>7.3f} "
                f"{row['ks_pvalue']:>7.3f} "
                f"{row['wasserstein_dist']:>8.4f}"
            )

        lines.append("\n--- Grouped insights ---")
        groups = [
            ("_mean",          "Mean intensity"),
            ("_std",           "Intensity std"),
            ("_dynamic_range", "Dynamic range (p95−p5)"),
            ("_sharpness",     "Sharpness (Laplacian var.)"),
            ("_sat_low",       "Near-black fraction"),
            ("_sat_high",      "Near-white fraction"),
            ("_hf_ratio",      "High-frequency energy ratio"),
            ("corr_",          "Cross-channel correlation"),
        ]
        for suffix, label in groups:
            grp = stat_df[stat_df["feature"].str.contains(suffix, regex=False)]
            if grp.empty:
                continue
            avg_w1 = grp["wasserstein_dist"].mean()
            avg_diff = grp["mean_diff (easy-hard)"].mean()
            direction = "higher in easy" if avg_diff > 0 else "higher in hard"
            lines.append(
                f"  {label:<38}  avg W₁={avg_w1:.4f}  "
                f"avg Δ={avg_diff:+.4f}  ({direction})"
            )

        lines.append(
            "\nInterpretation guide:\n"
            "  • Large W₁ (Wasserstein) + low p-value → strong distributional shift.\n"
            "  • 'Δ > 0' means the easy batches have higher values of that feature.\n"
            "  • 'Δ < 0' means the hard batches have higher values of that feature.\n"
            "  • Sharpness ↑ in easy batches → hard batches may be blurrier.\n"
            "  • Dynamic range ↑ in easy batches → hard batches have weaker signal.\n"
            "  • sat_low ↑ in hard batches → hard batches have more dark/unexposed areas.\n"
            "  • Cross-channel correlations changing → staining protocol differences."
        )

    lines.append(f"\n{sep}\n")
    return "\n".join(lines)


# ── LOBO accuracy loading ──────────────────────────────────────────────────


def load_lobo_accuracy(lobo_dir: str | None) -> dict[str, float]:
    """Return a dict mapping batch name → ERM LOBO test accuracy.

    If *lobo_dir* is a directory it is scanned for
    ``jumpcp-erm-lobo-fold*.json`` files.  If it is ``None`` or the directory
    contains no matching files, the hard-coded default values from the
    2026-03-23 run are used as a fallback.
    """
    if lobo_dir is None:
        return _DEFAULT_LOBO_ERM_ACC.copy()

    results_path = Path(lobo_dir)
    if not results_path.is_dir():
        print(
            f"WARNING: --lobo_dir '{lobo_dir}' is not a directory; "
            "using built-in accuracy values.",
            file=sys.stderr,
        )
        return _DEFAULT_LOBO_ERM_ACC.copy()

    acc: dict[str, float] = {}
    for fn in sorted(results_path.glob("jumpcp-erm-lobo-fold*.json")):
        with open(fn) as f:
            data = json.load(f)
        metrics = data.get("test_metrics", {})
        for key, val in metrics.items():
            if key.startswith("test_acc_domain_") and isinstance(val, float):
                batch = key.replace("test_acc_domain_", "")
                acc[batch] = val

    if not acc:
        print(
            f"WARNING: No ERM LOBO results found in '{lobo_dir}'; "
            "using built-in accuracy values.",
            file=sys.stderr,
        )
        return _DEFAULT_LOBO_ERM_ACC.copy()

    return acc


# ── CLI ────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyse per-batch image statistics to identify JUMP-CP domain shift.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--parquet",
        default=str(REPO_ROOT / "data" / "indices" / "JKU_JUMPCP_Source3_AllPositives.parquet"),
        help="Path to the parquet metadata file.",
    )
    p.add_argument(
        "--img_root",
        required=True,
        help="Root directory that contains the TIFF/jpg images.",
    )
    p.add_argument(
        "--output_dir",
        default=str(REPO_ROOT / "outputs" / "domain_shift_analysis"),
        help="Directory to write all analysis outputs.",
    )
    p.add_argument(
        "--batches",
        nargs="*",
        default=None,
        help=(
            "Batch names to include.  If omitted, all batches in the parquet "
            "are used."
        ),
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=200,
        help="Number of images to sample per batch.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    p.add_argument(
        "--nested_by_batch",
        action="store_true",
        help=(
            "Images are stored as <img_root>/<batch>/<site>/<sample>.jpg "
            "instead of the flat <img_root>/<sample>.jpg layout."
        ),
    )
    p.add_argument(
        "--ext",
        default=".jpg",
        help="File extension of the image files.",
    )
    p.add_argument(
        "--lobo_dir",
        default=None,
        help=(
            "Path to a K9 LOBO results directory (containing "
            "jumpcp-erm-lobo-fold*.json) for batch accuracy colour-coding.  "
            "If omitted, the built-in 2026-03-23 ERM accuracy values are used."
        ),
    )
    return p.parse_args()


# ── main ───────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load LOBO accuracy for colour coding ──────────────────────────────
    lobo_acc = load_lobo_accuracy(args.lobo_dir)
    print("ERM LOBO accuracy per batch (used for colour coding):")
    for b, acc in sorted(lobo_acc.items(), key=lambda kv: -kv[1]):
        print(f"  {b:<32s}  {acc:.3%}")

    # ── Load metadata ──────────────────────────────────────────────────────
    print(f"\nLoading metadata from {args.parquet} …")
    df = pd.read_parquet(args.parquet)
    batch_col = "Metadata_Batch"
    sample_col = "Metadata_Sample_ID"

    if args.batches:
        unknown = set(args.batches) - set(df[batch_col].unique())
        if unknown:
            print(
                f"WARNING: the following batches were not found in the parquet "
                f"and will be skipped: {sorted(unknown)}",
                file=sys.stderr,
            )
        df = df[df[batch_col].isin(args.batches)]

    batches_present = sorted(df[batch_col].unique())
    print(f"Batches to analyse ({len(batches_present)}): {batches_present}")
    print(f"Total images available: {len(df):,}")
    print(f"Sampling up to {args.n_samples} images per batch …\n")

    # ── Extract features ───────────────────────────────────────────────────
    rng = np.random.default_rng(args.seed)
    feat_df = compute_batch_features(
        df=df,
        img_root=Path(args.img_root),
        batch_col=batch_col,
        sample_col=sample_col,
        ext=args.ext,
        n_samples=args.n_samples,
        rng=rng,
        nested_by_batch=args.nested_by_batch,
    )

    if feat_df.empty:
        print(
            "ERROR: No images could be read.  "
            "Check --img_root and --ext.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"\nSuccessfully extracted features for {len(feat_df):,} images.")

    # ── Save raw features ──────────────────────────────────────────────────
    raw_path = out_dir / "per_image_features.csv"
    feat_df.to_csv(raw_path, index=False)
    print(f"Per-image features  → {raw_path}")

    # ── Save aggregate stats ───────────────────────────────────────────────
    agg_df = compute_aggregate_stats(feat_df, batch_col)
    agg_path = out_dir / "batch_aggregate_stats.csv"
    agg_df.to_csv(agg_path)
    print(f"Batch aggregate stats → {agg_path}")

    # ── Statistical comparison: hard vs easy ───────────────────────────────
    stat_df = statistical_comparison(feat_df, batch_col, lobo_acc)
    if not stat_df.empty:
        stat_path = out_dir / "hard_vs_easy_statistics.csv"
        stat_df.to_csv(stat_path, index=False)
        print(f"Hard-vs-easy stats  → {stat_path}")

    # ── Generate plots ─────────────────────────────────────────────────────
    print("\nGenerating plots …")
    warnings.filterwarnings("ignore", category=UserWarning)

    plot_intensity_boxplots(feat_df, batch_col, out_dir, lobo_acc)
    print("  intensity_boxplots.{pdf,png}")

    plot_mean_intensity_heatmap(feat_df, batch_col, out_dir)
    print("  mean_intensity_heatmap.{pdf,png}")

    plot_sharpness(feat_df, batch_col, out_dir, lobo_acc)
    print("  sharpness.{pdf,png}")

    plot_dynamic_range(feat_df, batch_col, out_dir, lobo_acc)
    print("  dynamic_range.{pdf,png}")

    plot_saturation(feat_df, batch_col, out_dir, lobo_acc)
    print("  saturation.{pdf,png}")

    plot_intensity_histograms(feat_df, batch_col, out_dir)
    print("  intensity_histograms.{pdf,png}")

    plot_cross_channel_correlation(feat_df, batch_col, out_dir)
    print("  cross_channel_correlation.{pdf,png}")

    plot_pca(feat_df, batch_col, out_dir, lobo_acc)
    print("  pca.{pdf,png}")

    plot_acc_vs_features(feat_df, batch_col, out_dir, lobo_acc)
    print("  acc_vs_*.{pdf,png}")

    print(f"\nAll plots saved to {out_dir}")

    # ── Plain-language summary ─────────────────────────────────────────────
    summary = build_summary_text(feat_df, batch_col, stat_df, lobo_acc)
    print(summary)
    summary_path = out_dir / "domain_shift_summary.txt"
    summary_path.write_text(summary)
    print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()