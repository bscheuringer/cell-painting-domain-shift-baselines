"""
Tests for JumpCPDataModule — fold-file mode and domain label support.

These tests do NOT load any images; they only verify:
  - The correct rows are assigned to train / val / test splits
  - Domain label encoding is consistent
  - Edge-case validation (bad fold_index)

Run from the repo root so that relative paths resolve:
    pytest tests/test_datamodule.py -v
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.datamodule import JumpCPDataModule
from src.data.dataset import build_transforms

# ── fixtures ──────────────────────────────────────────────────────────────────

PARQUET = Path("data/indices/JKU_JUMPCP_Source3_AllPositives.parquet")
K5_FILE = Path("data/splits/folds_k5_seed42.json")
K9_FILE = Path("data/splits/folds_k9_lobo.json")

# Skip all tests in this module if the real parquet isn't present
pytestmark = pytest.mark.skipif(
    not PARQUET.exists(),
    reason="Parquet file not found; skipping datamodule tests",
)


def _make_dm(fold_file, fold_index, return_domain=False) -> JumpCPDataModule:
    dm = JumpCPDataModule(
        parquet_path=str(PARQUET),
        img_root="DUMMY",  # no image I/O in these tests
        fold_config_file=str(fold_file),
        fold_index=fold_index,
        return_domain=return_domain,
    )
    dm.setup()
    return dm


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_fold(fold_file, fold_index):
    with open(fold_file) as f:
        data = json.load(f)
    return data["folds"][fold_index]


# ── parametrise over every fold in every file ─────────────────────────────────

def _fold_params():
    params = []
    for fpath in [K5_FILE, K9_FILE]:
        if not fpath.exists():
            continue
        with open(fpath) as f:
            data = json.load(f)
        for fi in range(data["num_folds"]):
            params.append(pytest.param(fpath, fi, id=f"{fpath.stem}_fold{fi}"))
    return params


@pytest.mark.parametrize("fold_file,fold_index", _fold_params())
class TestFoldBasedSplits:
    """Verify that setup() produces splits matching the fold JSON."""

    def test_train_batches_correct(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        actual = set(dm.train_df["Metadata_Batch"].unique())
        assert actual == set(fold["train_batches"])

    def test_val_batches_correct(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        actual = set(dm.val_df["Metadata_Batch"].unique())
        assert actual == set(fold["val_batches"])

    def test_test_batches_correct(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        actual = set(dm.test_df["Metadata_Batch"].unique())
        assert actual == set(fold["test_batches"])

    def test_train_row_count(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        assert len(dm.train_df) == fold["n_train"]

    def test_val_row_count(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        assert len(dm.val_df) == fold["n_val"]

    def test_test_row_count(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        fold = _load_fold(fold_file, fold_index)
        assert len(dm.test_df) == fold["n_test"]

    def test_no_overlap_between_splits(self, fold_file, fold_index):
        """Train / val / test batches must be disjoint."""
        dm = _make_dm(fold_file, fold_index)
        train_b = set(dm.train_df["Metadata_Batch"].unique())
        val_b   = set(dm.val_df["Metadata_Batch"].unique())
        test_b  = set(dm.test_df["Metadata_Batch"].unique())
        assert train_b.isdisjoint(val_b),   "train and val share a batch"
        assert train_b.isdisjoint(test_b),  "train and test share a batch"
        assert val_b.isdisjoint(test_b),    "val and test share a batch"

    def test_num_classes(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        assert dm.num_classes == 8

    def test_split_summary_keys(self, fold_file, fold_index):
        dm = _make_dm(fold_file, fold_index)
        s = dm.split_summary
        assert s["fold_index"] == fold_index
        assert "fold_config_file" in s
        assert "train" in s and "val" in s and "test" in s


# ── domain label tests ────────────────────────────────────────────────────────

class TestDomainLabels:
    """Verify domain label encoding when return_domain=True."""

    def test_num_domains_equals_9(self):
        dm = _make_dm(K5_FILE, 0, return_domain=True)
        assert dm.num_domains == 9

    def test_domain_labels_in_train_dataset(self):
        dm = _make_dm(K5_FILE, 0, return_domain=True)
        ds = dm._build_ds(dm.train_df, dm.train_tfms)
        assert ds.domain_labels is not None
        assert len(ds.domain_labels) == len(dm.train_df)

    def test_domain_label_values_are_valid_ints(self):
        dm = _make_dm(K5_FILE, 0, return_domain=True)
        ds = dm._build_ds(dm.train_df, dm.eval_tfms)
        assert ds.domain_labels.dtype == np.int64
        assert ds.domain_labels.min() >= 0
        assert ds.domain_labels.max() < dm.num_domains

    def test_domain_labels_consistent_across_folds(self):
        """Same batch must get the same domain id in different folds."""
        dm0 = _make_dm(K5_FILE, 0, return_domain=True)
        dm1 = _make_dm(K5_FILE, 1, return_domain=True)
        # Both encoders should have the same classes_ in the same order
        assert list(dm0.domain_encoder.classes_) == list(dm1.domain_encoder.classes_)

    def test_no_domain_labels_when_return_domain_false(self):
        dm = _make_dm(K5_FILE, 0, return_domain=False)
        ds = dm._build_ds(dm.train_df, dm.eval_tfms)
        assert ds.domain_labels is None


# ── edge-case / error handling ────────────────────────────────────────────────

class TestEdgeCases:
    def test_invalid_fold_index_raises(self):
        with pytest.raises(ValueError, match="fold_index"):
            dm = JumpCPDataModule(
                parquet_path=str(PARQUET),
                img_root="DUMMY",
                fold_config_file=str(K5_FILE),
                fold_index=99,
            )
            dm.setup()


# ── transform shape tests ─────────────────────────────────────────────────────

class TestBuildTransforms:
    """Output tensor must be (5, image_size, image_size) regardless of input shape."""

    @pytest.mark.parametrize("h,w", [(224, 224), (300, 200), (100, 400), (512, 512)])
    def test_train_output_shape(self, h, w):
        tfm = build_transforms(image_size=224, train=True)
        img = np.zeros((h, w, 5), dtype=np.float32)
        out = tfm(image=img)["image"]
        assert out.shape == (5, 224, 224), f"Expected (5,224,224), got {out.shape}"

    @pytest.mark.parametrize("h,w", [(224, 224), (300, 200), (100, 400), (512, 512)])
    def test_eval_output_shape(self, h, w):
        tfm = build_transforms(image_size=224, train=False)
        img = np.zeros((h, w, 5), dtype=np.float32)
        out = tfm(image=img)["image"]
        assert out.shape == (5, 224, 224), f"Expected (5,224,224), got {out.shape}"

    def test_output_is_tensor(self):
        tfm = build_transforms(image_size=224, train=False)
        img = np.zeros((224, 224, 5), dtype=np.float32)
        out = tfm(image=img)["image"]
        assert isinstance(out, torch.Tensor)
