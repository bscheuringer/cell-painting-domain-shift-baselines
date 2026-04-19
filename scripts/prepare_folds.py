"""prepare_folds.py

Generates fixed, versioned fold definitions for all K-fold CV experiments
on the JUMP-CP Source_3 dataset. Two JSON files are produced:

  data/splits/folds_k5_seed42.json
      K=5 batch-level cross-validation. Batches are shuffled with seed=42
      and grouped by sklearn KFold into 5 folds (sizes: 2+2+2+2+1).
      For each fold:
        test  = the held-out batch group (1-2 batches)
        val   = the next batch in the sorted remaining list (1 batch)
        train = all other remaining batches

  data/splits/folds_k9_lobo.json
      Leave-one-batch-out (LOBO). One batch is held out per fold,
      rotating through all 9 batches. Val rotates with a +1 offset.
      For each fold:
        test  = 1 batch
        val   = next batch in sorted order (wraps around)
        train = remaining 7 batches

Both files share the same JSON schema so that the DataModule can load
either interchangeably.

Usage (run from repo root):
    python scripts/prepare_folds.py
    python scripts/prepare_folds.py --parquet data/indices/my_file.parquet
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_batches(parquet_path: str, batch_col: str = "Metadata_Batch") -> list:
    df = pd.read_parquet(parquet_path, columns=[batch_col])
    batches = sorted(df[batch_col].unique().tolist())
    counts = df[batch_col].value_counts().sort_index().to_dict()
    return batches, counts


def _sample_counts(batch_list: list, counts: dict) -> int:
    return sum(counts.get(b, 0) for b in batch_list)


# ---------------------------------------------------------------------------
# Fold builders
# ---------------------------------------------------------------------------

def make_k5_folds(batches: list, counts: dict, k: int = 5, seed: int = 42) -> list:
    """K-fold over batch list.

    Val selection uses a round-robin offset across folds so that no single
    batch dominates the val role. Specifically, for fold i the val batch is
    the batch at position (i % len(train_val_pool)) in the sorted non-test
    pool, ensuring each fold picks a different val batch wherever possible.
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    arr = np.array(batches)
    folds = []
    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(arr)):
        test_batches = sorted(arr[test_idx].tolist())
        train_val_batches = sorted(arr[train_val_idx].tolist())
        # Round-robin val: rotate the pick index by fold number
        val_pick = fold_idx % len(train_val_batches)
        val_batches = [train_val_batches[val_pick]]
        train_batches = [b for b in train_val_batches if b not in val_batches]
        folds.append({
            "fold": fold_idx,
            "test_batches": test_batches,
            "val_batches": val_batches,
            "train_batches": train_batches,
            "n_test": _sample_counts(test_batches, counts),
            "n_val": _sample_counts(val_batches, counts),
            "n_train": _sample_counts(train_batches, counts),
        })
    return folds


def make_lobo_folds(batches: list, counts: dict) -> list:
    """Leave-one-batch-out. Val rotates with a +1 offset so every batch
    also serves as validation domain exactly once across the 9 folds."""
    folds = []
    n = len(batches)
    for fold_idx, test_batch in enumerate(batches):
        remaining = [b for b in batches if b != test_batch]
        # Rotate val: use the batch that comes after test_batch in the
        # original sorted list (wraps around), skipping the test batch.
        val_candidate_idx = (fold_idx + 1) % n
        val_batch = batches[val_candidate_idx]
        if val_batch == test_batch:          # safety (shouldn't happen for n>1)
            val_batch = batches[(fold_idx + 2) % n]
        val_batches = [val_batch]
        train_batches = [b for b in remaining if b != val_batch]
        folds.append({
            "fold": fold_idx,
            "test_batches": [test_batch],
            "val_batches": val_batches,
            "train_batches": train_batches,
            "n_test": _sample_counts([test_batch], counts),
            "n_val": _sample_counts(val_batches, counts),
            "n_train": _sample_counts(train_batches, counts),
        })
    return folds


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def save_folds(folds: list, batches: list, out_path: Path,
               description: str, extra_meta: dict):
    doc = {
        "description": description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_folds": len(folds),
        "num_batches": len(batches),
        "all_batches": batches,
        **extra_meta,
        "folds": folds,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"  Saved → {out_path}")


def print_summary(label: str, folds: list):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    for f in folds:
        print(
            f"  Fold {f['fold']:>2d} | "
            f"test={f['test_batches']} (n={f['n_test']:>5d}) | "
            f"val={f['val_batches']} (n={f['n_val']:>5d}) | "
            f"train n={f['n_train']:>6d}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed fold definitions for JUMP-CP Source_3 experiments."
    )
    parser.add_argument(
        "--parquet",
        default="data/indices/JKU_JUMPCP_Source3_AllPositives.parquet",
        help="Path to the source_3 parquet file (default: data/indices/JKU_JUMPCP_Source3_AllPositives.parquet)",
    )
    parser.add_argument(
        "--out_dir", default="data/splits",
        help="Output directory for fold JSON files (default: data/splits)",
    )
    parser.add_argument("--batch_col", default="Metadata_Batch")
    parser.add_argument("--k", type=int, default=5, help="K for the KFold variant (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffle (default: 42)")
    args = parser.parse_args()

    print(f"Loading batches from: {args.parquet}")
    batches, counts = load_batches(args.parquet, args.batch_col)
    print(f"Found {len(batches)} batches, {sum(counts.values())} total samples")
    for b in batches:
        print(f"  {b}: {counts[b]:>6d} samples")

    out_dir = Path(args.out_dir)

    # --- K=5 ---
    k5_folds = make_k5_folds(batches, counts, k=args.k, seed=args.seed)
    k5_path = out_dir / f"folds_k{args.k}_seed{args.seed}.json"
    save_folds(
        k5_folds, batches, k5_path,
        description=(
            f"K={args.k} batch-level cross-validation on JUMP-CP Source_3 "
            f"(sklearn KFold, shuffle=True, random_state={args.seed}). "
            "Test = held-out batch group; val = round-robin selected batch "
            "from the remaining non-test pool (rotates by fold index); train = rest."
        ),
        extra_meta={"strategy": "kfold", "k": args.k, "seed": args.seed},
    )
    print_summary(f"K={args.k} folds  [{k5_path}]", k5_folds)

    # --- K=9 LOBO ---
    lobo_folds = make_lobo_folds(batches, counts)
    lobo_path = out_dir / "folds_k9_lobo.json"
    save_folds(
        lobo_folds, batches, lobo_path,
        description=(
            "Leave-one-batch-out (LOBO) on JUMP-CP Source_3. "
            "Each fold holds out exactly one batch as test domain; "
            "val = next batch in sorted order (wraps around); train = remaining 7."
        ),
        extra_meta={"strategy": "lobo", "k": len(batches), "seed": None},
    )
    print_summary(f"K=9 LOBO  [{lobo_path}]", lobo_folds)

    print(f"\nDone. Fold files written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
