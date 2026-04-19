#!/usr/bin/env python
"""
Launcher for all (method × fold) experiments.

Runs every (method, fold_index) combination sequentially, one GPU job
at a time.  Each run's output is tee'd to the console and saved to
outputs/launcher_logs/<timestamp>/<run_id>.log

Usage examples
--------------
# All 3 methods × all K=5 folds (default)
python scripts/run_all_experiments.py

# Switch to LOBO fold file
python scripts/run_all_experiments.py \\
    --fold_file data/splits/folds_k9_lobo.json

# Subset: only ERM, only folds 0 and 1
python scripts/run_all_experiments.py \\
    --methods erm --fold_indices 0 1

# Dry run — print commands without executing
python scripts/run_all_experiments.py --dry_run

# Pass extra Hydra overrides to every run
python scripts/run_all_experiments.py \\
    --hydra_overrides logger=wandb train.max_epochs=50
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

METHODS = ["erm", "dann", "instance_norm"]
REPO_ROOT = Path(__file__).parent.parent.resolve()
DEFAULT_FOLD_FILE = "data/splits/folds_k5_seed42.json"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all (method × fold) experiments sequentially.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--fold_file",
        default=DEFAULT_FOLD_FILE,
        help=f"Path to fold config JSON, relative to repo root or absolute "
             f"(default: {DEFAULT_FOLD_FILE})",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=METHODS,
        choices=METHODS,
        metavar="METHOD",
        help=f"Methods to run (default: all — {METHODS})",
    )
    p.add_argument(
        "--fold_indices",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Specific fold indices to run (default: all folds in the file)",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them",
    )
    p.add_argument(
        "--hydra_overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Hydra overrides appended to every command "
             "(e.g. logger=wandb train.max_epochs=50)",
    )
    return p.parse_args()


def _fold_label(fold_cfg: dict) -> str:
    """Derive a short label from fold-file metadata (e.g. 'k5', 'lobo')."""
    strategy = fold_cfg.get("strategy", "")
    k = fold_cfg.get("k")
    return "lobo" if strategy == "lobo" else (f"k{k}" if k else strategy or "fold")


def _run_job(
    cmd: list[str],
    log_path: Path,
    run_timestamp: str,
    tail_on_success: int = 5,
    tail_on_failure: int = 40,
) -> bool:
    """
    Run *cmd*, tee output to *log_path* and the console in real time.
    Returns True on success.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_fh:
        # Run with unbuffered output directly to both log file and console.
        # Using PIPE causes Python/PL progress bars to buffer indefinitely.
        proc = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1", "RUN_TIMESTAMP": run_timestamp},
        )
        proc.wait()

    success = proc.returncode == 0
    if success:
        # Print last few lines (usually the test metrics table)
        lines = log_path.read_text().splitlines()
        for line in lines[-tail_on_success:]:
            print("  │ " + line, flush=True)
    else:
        print(f"\n  ✗ FAILED (exit {proc.returncode})", flush=True)
        lines = log_path.read_text().splitlines()
        for line in lines[-tail_on_failure:]:
            print("  │ " + line, flush=True)
    return success


def main() -> None:
    args = _parse_args()

    # Resolve fold file
    fold_path = Path(args.fold_file)
    if not fold_path.is_absolute():
        fold_path = REPO_ROOT / fold_path
    fold_path = fold_path.resolve()

    if not fold_path.exists():
        print(f"ERROR: fold file not found: {fold_path}", file=sys.stderr)
        sys.exit(1)

    with open(fold_path) as f:
        fold_cfg = json.load(f)

    num_folds = fold_cfg["num_folds"]
    label = _fold_label(fold_cfg)
    fold_indices = args.fold_indices if args.fold_indices is not None else list(range(num_folds))

    # Build job list: iterate methods first so all folds of one method run together
    jobs: list[tuple[str, int]] = [
        (method, fi) for method in args.methods for fi in fold_indices
    ]
    total = len(jobs)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = REPO_ROOT / "outputs" / "launcher_logs" / ts

    # ---- Print summary ----
    print("=" * 60, flush=True)
    print(f"  Fold file  : {fold_path}")
    print(f"  Strategy   : {label}  ({num_folds} folds total)")
    print(f"  Methods    : {args.methods}")
    print(f"  Folds      : {fold_indices}")
    print(f"  Total runs : {total}")
    if args.hydra_overrides:
        print(f"  Overrides  : {args.hydra_overrides}")
    if args.dry_run:
        print("  Mode       : DRY RUN")
    else:
        print(f"  Logs       : {log_dir}")
    print("=" * 60, flush=True)
    print(flush=True)

    failed: list[str] = []

    for idx, (method, fold_idx) in enumerate(jobs, 1):
        run_id = f"jumpcp-{method}-{label}-fold{fold_idx}"
        cmd = [
            sys.executable, "-m", "src.train",
            f"method={method}",
            f"data.fold_index={fold_idx}",
            f"data.fold_config_file={fold_path}",
        ] + list(args.hydra_overrides)

        print(f"[{idx}/{total}] {run_id}")
        print("  " + " ".join(str(c) for c in cmd))

        if args.dry_run:
            print()
            continue

        log_file = log_dir / f"{run_id}.log"
        print(f"  log  → {log_file}")
        print()

        ok = _run_job(cmd, log_file, run_timestamp=ts)

        if ok:
            print(f"\n  ✓ {run_id} — done")
        else:
            failed.append(run_id)
        print()

    # ---- Final summary ----
    print("=" * 60)
    if args.dry_run:
        print(f"DRY RUN complete — {total} commands listed.")
    else:
        n_ok = total - len(failed)
        print(f"Finished: {n_ok}/{total} runs succeeded.")
        if failed:
            print(f"\nFailed ({len(failed)}):")
            for r in failed:
                print(f"  ✗ {r}")
            sys.exit(1)
        else:
            print("All runs completed successfully.")


if __name__ == "__main__":
    main()
