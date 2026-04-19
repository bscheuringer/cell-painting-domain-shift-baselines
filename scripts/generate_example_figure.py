#!/usr/bin/env python
"""
Generate a publication-ready Cell Painting example figure for the report.

Shows two rows: one "easy" batch (bright, high contrast) and one "hard"
batch (dark, low contrast) to visually illustrate the domain shift.
Each row displays the 5 fluorescence channels + a false-color composite.

Usage
-----
python scripts/generate_example_figure.py
python scripts/generate_example_figure.py --easy CP_25_all_Phenix1 --hard CP_32_all_Phenix1
python scripts/generate_example_figure.py --output_dir outputs/figures

Output: cell_painting_example.pdf + .png in the output directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile as tiff

REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_PROBES_DIR = REPO_ROOT / "sample_data" / "batch_probes"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"

CHANNEL_NAMES = ["DNA", "ER", "RNA", "AGP", "Mito"]
# Channel order in the TIFF files (alphabetical: AGP=0, DNA=1, ER=2, Mito=3, RNA=4)
CHANNEL_INDEX = {"AGP": 0, "DNA": 1, "ER": 2, "Mito": 3, "RNA": 4}

# False-color mapping (standard Cell Painting conventions)
CHANNEL_COLORS = {
    "DNA":  (0.0, 0.4, 1.0),   # blue
    "ER":   (0.0, 1.0, 0.0),   # green
    "RNA":  (1.0, 0.0, 1.0),   # magenta
    "AGP":  (1.0, 1.0, 0.0),   # yellow
    "Mito": (1.0, 0.0, 0.0),   # red
}


def load_image(path: Path) -> np.ndarray:
    """Load a 5-channel TIFF (stored with .jpg extension) as float32 [0,1]."""
    img = tiff.imread(str(path), maxworkers=1).astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def pick_sample(probes_dir: Path, batch: str, seed: int = 42) -> Path:
    """Pick a representative sample from a batch (deterministic)."""
    batch_dir = probes_dir / batch
    files = sorted(batch_dir.glob("*.jpg"))
    if not files:
        raise FileNotFoundError(f"No images in {batch_dir}")
    rng = np.random.default_rng(seed)
    return files[rng.integers(len(files))]


def normalize_channel(ch: np.ndarray, percentile_low: float = 1, percentile_high: float = 99.5) -> np.ndarray:
    """Percentile-based contrast stretch for a single channel."""
    lo = np.percentile(ch, percentile_low)
    hi = np.percentile(ch, percentile_high)
    if hi <= lo:
        return np.zeros_like(ch)
    return np.clip((ch - lo) / (hi - lo), 0, 1)


def make_figure(
    easy_img: np.ndarray,
    hard_img: np.ndarray,
    easy_label: str,
    hard_label: str,
    output_dir: Path,
    dpi: int = 300,
) -> list[str]:
    """Create the two-row figure: easy batch (top) vs hard batch (bottom)."""

    n_cols = len(CHANNEL_NAMES)
    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 2.0, 2 * 2.0 + 0.6))

    for row_idx, (img, label) in enumerate([(easy_img, easy_label), (hard_img, hard_label)]):
        for col_idx, ch_name in enumerate(CHANNEL_NAMES):
            ch_idx = CHANNEL_INDEX[ch_name]
            raw = img[:, :, ch_idx]
            normed = normalize_channel(raw)
            axes[row_idx][col_idx].imshow(normed, cmap="gray", vmin=0, vmax=1)
            axes[row_idx][col_idx].axis("off")

    # Column titles
    for col_idx, ch_name in enumerate(CHANNEL_NAMES):
        axes[0][col_idx].set_title(ch_name, fontsize=10, fontweight="bold")

    fig.tight_layout(pad=0.3, h_pad=0.5, w_pad=0.3)
    fig.subplots_adjust(left=0.11)

    # Row labels via fig.text (avoids tight_layout clipping)
    for row_idx, label in enumerate([easy_label, hard_label]):
        bbox = axes[row_idx][0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(0.01, y_center, label, va="center", ha="left",
                 fontsize=8, fontweight="bold", color="black")

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in ("pdf", "png"):
        p = output_dir / f"cell_painting_example.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(str(p))
    plt.close(fig)
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Cell Painting example figure for report.")
    p.add_argument("--probes_dir", default=str(DEFAULT_PROBES_DIR),
                   help="Directory with downloaded batch probes")
    p.add_argument("--easy", default="CP_25_all_Phenix1",
                   help="Batch name for the 'easy' (bright) row")
    p.add_argument("--hard", default="CP_32_all_Phenix1",
                   help="Batch name for the 'hard' (dark) row")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    probes_dir = Path(args.probes_dir)
    output_dir = Path(args.output_dir)

    easy_path = pick_sample(probes_dir, args.easy, args.seed)
    hard_path = pick_sample(probes_dir, args.hard, args.seed)
    print(f"Easy batch: {args.easy} -> {easy_path.name}")
    print(f"Hard batch: {args.hard} -> {hard_path.name}")

    easy_img = load_image(easy_path)
    hard_img = load_image(hard_path)
    print(f"Image shape: {easy_img.shape}")

    saved = make_figure(
        easy_img, hard_img,
        easy_label=args.easy.replace("_all_Phenix1", "\nall_Phenix1"),
        hard_label=args.hard.replace("_all_Phenix1", "\nall_Phenix1"),
        output_dir=output_dir,
    )
    for s in saved:
        print(f"Saved: {s}")


if __name__ == "__main__":
    main()
