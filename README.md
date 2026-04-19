# JUMP-CP Domain Shift Baseline

Baseline study comparing three methods for domain-shift robustness on the
[JUMP-CP dataset](https://jump-cellpainting.broadinstitute.org/) (Source 3,
positive controls, 5-channel fluorescence microscopy images).

Methods compared:

| Method | Key idea |
|--------|---------|
| **ERM** | Standard cross-entropy, frozen BN stats at test time |
| **DANN** | Gradient-reversal domain adversarial training (Ganin et al. 2016) |
| **InstanceNorm** | Replace all BatchNorm2d with InstanceNorm2d — no domain stats |

---

## Project structure

```
jku-master-basics/
├── configs/
│   ├── config.yaml          # main Hydra config
│   ├── method/              # per-method config group (erm | dann | instance_norm)
│   └── logger/              # logger config group (tensorboard | wandb | none)
├── data/
│   ├── indices/             # parquet file (Source-3 positive controls)
│   └── splits/              # fold JSON files (K=5 and LOBO)
├── scripts/
│   ├── prepare_folds.py             # generate fold JSON files
│   ├── validate_fold_integrity.py   # sanity-check fold splits
│   ├── run_all_experiments.py       # launcher: all (method x fold) runs
│   ├── evaluate_results.py          # aggregate results + generate plots
│   ├── analyze-domain-shift.py      # domain shift analysis
│   └── generate_example_figure.py   # cell painting example figure
├── src/
│   ├── data/
│   │   ├── dataset.py       # JumpCPDataset + transforms
│   │   └── datamodule.py    # JumpCPDataModule (fold-file based splits)
│   ├── models/
│   │   ├── base_classifier.py       # ResNet-50 backbone, 5-channel, metrics
│   │   ├── erm_classifier.py        # ERM baseline
│   │   ├── dann_classifier.py       # DANN (GRL + domain discriminator)
│   │   └── in_classifier.py         # InstanceNorm
│   ├── utils/
│   │   └── logger.py        # build_logger (TensorBoard / W&B)
│   └── train.py             # training entry point (Hydra)
├── tests/
│   └── test_folds.py        # fold correctness tests
├── env.yml                  # conda environment
└── README.md
```

---

## Environment setup

```bash
conda env create -f env.yml
conda activate jku_jumpcp_baseline
```

---

## Data

Images which have been used to train our baselines are contained in Source-3 parquet file at:
```
data/indices/JKU_JUMPCP_Source3_AllPositives.parquet
```

Set the image root path in `configs/config.yaml` (key `data.img_root`) to point to your local copy of the JUMP-CP images.

Generate fold files (already committed — only needed if regenerating):
```bash
python scripts/prepare_folds.py
```

---

## Running experiments

### Single run
```bash
# ERM, fold 0, K=5 (defaults)
python -m src.train

# DANN, fold 3
python -m src.train method=dann data.fold_index=3

# InstanceNorm on LOBO folds with W&B logging
python -m src.train \
    method=instance_norm \
    data.fold_config_file='${base_dir}/data/splits/folds_k9_lobo.json' \
    data.fold_index=5 \
    logger=wandb
```

### Full experiment sweep

The launcher script runs all (method × fold) combinations sequentially,
tees output to console and per-run log files, and exits non-zero if any job fails.

```bash
# Dry run first (prints all commands, nothing executes)
python scripts/run_all_experiments.py --dry_run

# K=5 cross-validation — 3 methods × 5 folds = 15 runs
python scripts/run_all_experiments.py

# LOBO — 3 methods × 9 folds = 27 runs
python scripts/run_all_experiments.py \
    --fold_file data/splits/folds_k9_lobo.json

# Subset run (e.g. resume after a failure)
python scripts/run_all_experiments.py \
    --methods dann instance_norm --fold_indices 2 3 4
```

Run logs: `outputs/launcher_logs/<timestamp>/<run_name>.log`
Per-run metrics: `outputs/results/<run_name>.json`

---

## Analysing results

```bash
# Generate summary table + plots (paper-ready PDF + PNG)
python scripts/evaluate_results.py --fold_strategy k5

# LOBO results
python scripts/evaluate_results.py \
    --fold_strategy lobo \
    --output_dir outputs/analysis_lobo

# Combine results across multiple seeds
python scripts/evaluate_results.py \
    --results_dir outputs/results/K5_SEED_42_... \
                  outputs/results/K5_SEED_123_... \
                  outputs/results/K5_SEED_456_... \
    --output_dir outputs/analysis/combined_K5 \
    --fold_strategy k5

# Wider figures for slides
python scripts/evaluate_results.py --style screen
```

Outputs in `outputs/analysis/`:
- `summary_table.txt` — mean/std/worst-domain/best-domain per method
- `bar_plot.pdf`, `box_plot.pdf` — single-column LaTeX figures
- `domain_heatmap.pdf` — full-width heatmap (method × domain)

### Generate Cell Painting example figure
```bash
python scripts/generate_example_figure.py --seed 7
```

Outputs `cell_painting_example.pdf/.png` to `outputs/figures/`.

---

## Running tests

```bash
pytest tests/
```

