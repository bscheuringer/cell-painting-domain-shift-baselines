"""Tests for all fold definition files found in data/splits/.

Automatically discovers every *.json file in the splits directory and
validates it according to its declared `strategy` field:

  "kfold"  — K-fold batch-level CV (e.g. folds_k5_seed42.json)
  "lobo"   — Leave-one-batch-out   (e.g. folds_k9_lobo.json)

Adding a new fold file to data/splits/ is enough to include it in the
test suite — no changes to this file are needed.

Run from repo root:
    pytest tests/test_folds.py -v
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ground-truth constants (from the filtered JUMP-CP Source_3 parquet)
# ---------------------------------------------------------------------------

SPLITS_DIR = Path("data/splits")

EXPECTED_BATCHES = frozenset({
    "CP59",
    "CP60",
    "CP_25_all_Phenix1",
    "CP_26_all_Phenix1",
    "CP_27_all_Phenix1",
    "CP_28_all_Phenix1",
    "CP_29_all_Phenix1",
    "CP_31_all_Phenix1",
    "CP_32_all_Phenix1",
})

EXPECTED_TOTAL_SAMPLES = 49_491
REQUIRED_FOLD_KEYS = ("fold", "test_batches", "val_batches", "train_batches",
                      "n_test", "n_val", "n_train")


# ---------------------------------------------------------------------------
# File discovery (runs at collection time)
# ---------------------------------------------------------------------------

def _load_all_fold_files() -> dict[str, dict]:
    """Return {filename: parsed_json} for every *.json in SPLITS_DIR."""
    if not SPLITS_DIR.exists():
        return {}
    return {
        p.name: json.loads(p.read_text())
        for p in sorted(SPLITS_DIR.glob("*.json"))
    }


# Loaded once at module import so parametrize decorators can use them.
_ALL_FILES: dict[str, dict] = _load_all_fold_files()

# Flat list of (filename, fold_dict) for per-fold parametrized tests.
_ALL_FOLDS: list[tuple[str, dict]] = [
    (fname, fold)
    for fname, data in _ALL_FILES.items()
    for fold in data["folds"]
]


def _fold_id(fname_fold: tuple) -> str:
    fname, fold = fname_fold
    return f"{fname}::fold{fold['fold']}"


# ---------------------------------------------------------------------------
# Shared assertion helpers
# ---------------------------------------------------------------------------

def assert_no_overlap(fname: str, fold: dict) -> None:
    test  = set(fold["test_batches"])
    val   = set(fold["val_batches"])
    train = set(fold["train_batches"])
    i = fold["fold"]
    assert test & val   == set(), f"{fname} fold {i}: test ∩ val   = {test & val}"
    assert test & train == set(), f"{fname} fold {i}: test ∩ train = {test & train}"
    assert val  & train == set(), f"{fname} fold {i}: val  ∩ train = {val & train}"


def assert_full_coverage(fname: str, fold: dict) -> None:
    all_in_fold = (
        set(fold["test_batches"])
        | set(fold["val_batches"])
        | set(fold["train_batches"])
    )
    missing = EXPECTED_BATCHES - all_in_fold
    extra   = all_in_fold - EXPECTED_BATCHES
    assert not missing, f"{fname} fold {fold['fold']}: missing batches {missing}"
    assert not extra,   f"{fname} fold {fold['fold']}: unexpected batches {extra}"


def assert_counts_sum_to_total(fname: str, fold: dict) -> None:
    total = fold["n_test"] + fold["n_val"] + fold["n_train"]
    assert total == EXPECTED_TOTAL_SAMPLES, (
        f"{fname} fold {fold['fold']}: "
        f"{fold['n_test']} + {fold['n_val']} + {fold['n_train']} "
        f"= {total} ≠ {EXPECTED_TOTAL_SAMPLES}"
    )


# ---------------------------------------------------------------------------
# Directory / discovery tests
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_splits_dir_exists(self):
        assert SPLITS_DIR.exists(), f"Splits directory not found: {SPLITS_DIR}"

    def test_at_least_one_fold_file_found(self):
        assert _ALL_FILES, f"No *.json fold files found in {SPLITS_DIR}"

    def test_each_file_has_required_top_level_keys(self):
        for fname, data in _ALL_FILES.items():
            for key in ("strategy", "k", "num_folds", "all_batches", "folds"):
                assert key in data, f"{fname}: missing top-level key '{key}'"

    def test_each_file_declares_known_strategy(self):
        known = {"kfold", "lobo"}
        for fname, data in _ALL_FILES.items():
            assert data["strategy"] in known, (
                f"{fname}: unknown strategy '{data['strategy']}' (expected one of {known})"
            )

    def test_each_file_declares_expected_batches(self):
        for fname, data in _ALL_FILES.items():
            assert set(data["all_batches"]) == EXPECTED_BATCHES, (
                f"{fname}: all_batches mismatch.\n"
                f"  Missing: {EXPECTED_BATCHES - set(data['all_batches'])}\n"
                f"  Extra:   {set(data['all_batches']) - EXPECTED_BATCHES}"
            )

    def test_num_folds_matches_folds_array(self):
        for fname, data in _ALL_FILES.items():
            assert data["num_folds"] == len(data["folds"]), (
                f"{fname}: num_folds={data['num_folds']} but folds array has "
                f"{len(data['folds'])} entries"
            )


# ---------------------------------------------------------------------------
# Per-fold tests — apply to every fold in every file
# ---------------------------------------------------------------------------

class TestAllFolds:
    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_required_keys(self, fname, fold):
        for key in REQUIRED_FOLD_KEYS:
            assert key in fold, f"{fname} fold {fold['fold']}: missing key '{key}'"

    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_no_overlap(self, fname, fold):
        assert_no_overlap(fname, fold)

    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_full_batch_coverage(self, fname, fold):
        assert_full_coverage(fname, fold)

    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_sample_counts_sum_to_total(self, fname, fold):
        assert_counts_sum_to_total(fname, fold)

    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_val_not_in_test(self, fname, fold):
        val  = set(fold["val_batches"])
        test = set(fold["test_batches"])
        assert not val & test, (
            f"{fname} fold {fold['fold']}: val batch(es) also appear in test set: {val & test}"
        )

    @pytest.mark.parametrize("fname,fold", _ALL_FOLDS, ids=list(map(_fold_id, _ALL_FOLDS)))
    def test_fold_index_is_non_negative_int(self, fname, fold):
        assert isinstance(fold["fold"], int) and fold["fold"] >= 0


# ---------------------------------------------------------------------------
# Per-file tests — strategy-agnostic
# ---------------------------------------------------------------------------

class TestAllFiles:
    @pytest.mark.parametrize("fname,data", list(_ALL_FILES.items()))
    def test_fold_indices_are_sequential(self, fname, data):
        indices = [f["fold"] for f in data["folds"]]
        assert indices == list(range(len(data["folds"]))), (
            f"{fname}: fold indices are not sequential starting from 0: {indices}"
        )

    @pytest.mark.parametrize("fname,data", list(_ALL_FILES.items()))
    def test_each_batch_in_test_exactly_once(self, fname, data):
        """Every batch must be a test domain in exactly one fold (for any strategy)."""
        counts: dict[str, int] = {b: 0 for b in EXPECTED_BATCHES}
        for fold in data["folds"]:
            for b in fold["test_batches"]:
                counts[b] += 1
        bad = {b: c for b, c in counts.items() if c != 1}
        assert not bad, (
            f"{fname}: batches with wrong test-occurrence count "
            f"(expected 1 each): {bad}"
        )


# ---------------------------------------------------------------------------
# Strategy-specific tests
# ---------------------------------------------------------------------------

class TestKFoldStrategy:
    """Extra invariants that only apply to strategy='kfold' files."""

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "kfold"],
    )
    def test_val_batch_unique_across_folds(self, fname, data):
        val_batches = [fold["val_batches"][0] for fold in data["folds"]]
        assert len(val_batches) == len(set(val_batches)), (
            f"{fname}: duplicate val batches across folds: {val_batches}"
        )

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "kfold"],
    )
    def test_seed_is_recorded(self, fname, data):
        assert "seed" in data and data["seed"] is not None, (
            f"{fname}: kfold file must record the random seed used"
        )


class TestLoboStrategy:
    """Extra invariants that only apply to strategy='lobo' files."""

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "lobo"],
    )
    def test_k_equals_num_batches(self, fname, data):
        assert data["k"] == len(EXPECTED_BATCHES), (
            f"{fname}: LOBO k={data['k']} should equal number of batches "
            f"({len(EXPECTED_BATCHES)})"
        )

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "lobo"],
    )
    def test_exactly_one_test_batch_per_fold(self, fname, data):
        bad = [f["fold"] for f in data["folds"] if len(f["test_batches"]) != 1]
        assert not bad, f"{fname}: folds with ≠1 test batch: {bad}"

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "lobo"],
    )
    def test_val_covers_all_batches(self, fname, data):
        """Val rotates through every batch exactly once across all LOBO folds."""
        val_batches = {fold["val_batches"][0] for fold in data["folds"]}
        assert val_batches == EXPECTED_BATCHES, (
            f"{fname}: not all batches appear as val domain.\n"
            f"  Missing: {EXPECTED_BATCHES - val_batches}"
        )

    @pytest.mark.parametrize(
        "fname,data",
        [(f, d) for f, d in _ALL_FILES.items() if d["strategy"] == "lobo"],
    )
    def test_val_differs_from_test_in_every_fold(self, fname, data):
        bad = [
            f["fold"] for f in data["folds"]
            if f["val_batches"][0] == f["test_batches"][0]
        ]
        assert not bad, f"{fname}: folds where val == test batch: {bad}"
