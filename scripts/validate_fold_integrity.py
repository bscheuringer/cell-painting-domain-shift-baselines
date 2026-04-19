"""validate_fold_integrity.py

Validates fold JSON files against the actual parquet data. Checks:
  1. Sample counts in JSON match actual parquet counts
  2. All classes present in every split (train / val / test) per fold
  3. No severely imbalanced class distribution within any split
  4. Per-batch class distribution overview

Run from repo root:
    python scripts/validate_fold_integrity.py
"""

import json
from pathlib import Path

import pandas as pd

PARQUET    = "data/indices/JKU_JUMPCP_Source3_AllPositives.parquet"
SPLITS_DIR = Path("data/splits")
BATCH_COL  = "Metadata_Batch"
CLASS_COL  = "Metadata_JCP2022"


def main():
    df = pd.read_parquet(PARQUET, columns=[BATCH_COL, CLASS_COL])
    all_classes = sorted(df[CLASS_COL].unique())
    all_batches = sorted(df[BATCH_COL].unique())

    print(f"Dataset: {len(df):,} rows | {len(all_batches)} batches | {len(all_classes)} classes")

    # Per-batch class distribution
    print("\n--- Class counts per batch (rows=batch, cols=class) ---")
    pivot = df.groupby([BATCH_COL, CLASS_COL]).size().unstack(fill_value=0)
    print(pivot.to_string())

    issues = []

    for fname in sorted(SPLITS_DIR.glob("*.json")):
        data = json.loads(fname.read_text())
        strategy = data["strategy"]
        k        = data["k"]
        print(f"\n{'='*70}")
        print(f"  {fname.name}  (strategy={strategy}, k={k})")
        print(f"{'='*70}")
        print(f"  {'fold':>4}  {'train':>7}  {'val':>6}  {'test':>7}  {'test classes':>12}  {'val classes':>11}  {'train classes':>13}")

        for fold in data["folds"]:
            fi = fold["fold"]

            train_sub = df[df[BATCH_COL].isin(fold["train_batches"])]
            val_sub   = df[df[BATCH_COL].isin(fold["val_batches"])]
            test_sub  = df[df[BATCH_COL].isin(fold["test_batches"])]

            for split, sub in (("train", train_sub), ("val", val_sub), ("test", test_sub)):
                # 1. Count consistency
                n_json = fold[f"n_{split}"]
                if len(sub) != n_json:
                    issues.append(
                        f"{fname.name} fold {fi} {split}: "
                        f"count {len(sub):,} != JSON {n_json:,}"
                    )

                # 2. Class coverage
                present     = set(sub[CLASS_COL].unique())
                missing_cls = set(all_classes) - present
                if missing_cls:
                    issues.append(
                        f"{fname.name} fold {fi} {split}: "
                        f"MISSING classes {missing_cls}"
                    )

                # 3. Severe class imbalance (< 10 % of expected average count)
                counts  = sub[CLASS_COL].value_counts()
                avg     = len(sub) / len(all_classes)
                very_low = {c: int(v) for c, v in counts.items() if v < avg * 0.10}
                if very_low:
                    issues.append(
                        f"{fname.name} fold {fi} {split}: "
                        f"severely underrepresented classes {very_low}"
                    )

            n_train_cls = len(train_sub[CLASS_COL].unique())
            n_val_cls   = len(val_sub[CLASS_COL].unique())
            n_test_cls  = len(test_sub[CLASS_COL].unique())

            print(
                f"  {fi:>4}  "
                f"{len(train_sub):>7,}  "
                f"{len(val_sub):>6,}  "
                f"{len(test_sub):>7,}  "
                f"{n_test_cls:>12}  "
                f"{n_val_cls:>11}  "
                f"{n_train_cls:>13}"
            )

    # Summary
    print(f"\n{'='*70}")
    if issues:
        print(f"ISSUES FOUND ({len(issues)}):")
        for iss in issues:
            print(f"  ✗  {iss}")
    else:
        print("✓  No issues found — fold setup is valid for domain shift comparison.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
