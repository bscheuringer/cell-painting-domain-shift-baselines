from typing import Optional, Tuple, Dict, Any
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import JumpCPDataset, build_transforms


class JumpCPDataModule(pl.LightningDataModule):
    """
    DataModule for JUMP-CP using fold-file based splits.

    Reads train/val/test batch assignments from a JSON file produced by
    ``scripts/prepare_folds.py`` and filters the parquet accordingly.

    Domain labels:
        Set ``return_domain=True`` to get a third integer in every batch:
        ``(images, class_label, domain_label)``.  Domain labels encode
        ``Metadata_Batch`` as an integer, fitted on **all** batches in the
        parquet so the mapping is stable across folds.  Required by DANN;
        ERM / IN / BN leave this at the default ``False``.
    """

    def __init__(
        self,
        parquet_path: str,
        img_root: str,
        fold_config_file: str,
        fold_index: int = 0,
        return_domain: bool = False,
        batch_size: int = 16,
        num_workers: int = 8,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 4,
        label_col: str = "Metadata_JCP2022",
        batch_col: str = "Metadata_Batch",
        sample_col: str = "Metadata_Sample_ID",
        ext: str = ".jpg",
        nested_by_batch: bool = False,
        image_size: int = 224,
    ):
        super().__init__()
        self.parquet_path = parquet_path
        self.img_root = img_root
        self.fold_config_file = fold_config_file
        self.fold_index = fold_index
        self.return_domain = return_domain
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.label_col = label_col
        self.batch_col = batch_col
        self.sample_col = sample_col
        self.ext = ext
        self.nested_by_batch = nested_by_batch
        self.image_size = image_size

        self.encoder = LabelEncoder()
        self.domain_encoder = LabelEncoder()
        self.train_df = self.val_df = self.test_df = None
        self.split_info: Dict[str, Any] = {}

    def setup(self, stage: Optional[str] = None):
        if self.train_df is not None:  # already set up — PL may call twice (fit + test)
            return
        df = pd.read_parquet(self.parquet_path)

        # Fit domain encoder on ALL batches before any filtering (stable mapping).
        if self.return_domain:
            self.domain_encoder.fit(sorted(df[self.batch_col].unique()))

        with open(self.fold_config_file) as f:
            fold_data = json.load(f)

        folds = fold_data["folds"]
        if self.fold_index >= len(folds):
            raise ValueError(
                f"fold_index={self.fold_index} out of range "
                f"(num_folds={len(folds)} in {self.fold_config_file})"
            )

        fold = folds[self.fold_index]
        train_batches = set(fold["train_batches"])
        val_batches   = set(fold["val_batches"])
        test_batches  = set(fold["test_batches"])

        self.train_df = df[df[self.batch_col].isin(train_batches)].reset_index(drop=True)
        self.val_df   = df[df[self.batch_col].isin(val_batches)].reset_index(drop=True)
        self.test_df  = df[df[self.batch_col].isin(test_batches)].reset_index(drop=True)

        self.split_info = {
            "fold_config_file": str(self.fold_config_file),
            "fold_index": self.fold_index,
            "train": {"batches": sorted(train_batches), "n": len(self.train_df)},
            "val":   {"batches": sorted(val_batches),   "n": len(self.val_df)},
            "test":  {"batches": sorted(test_batches),  "n": len(self.test_df)},
        }

        # Fit class encoder on training labels only.
        self.encoder.fit(self.train_df[self.label_col])

        self.train_tfms = build_transforms(self.image_size, train=True)
        self.eval_tfms  = build_transforms(self.image_size, train=False)

    def _build_ds(self, df: pd.DataFrame, tfms) -> JumpCPDataset:
        domain_labels = None
        if self.return_domain:
            domain_labels = self.domain_encoder.transform(
                df[self.batch_col]
            ).astype(np.int64)
        return JumpCPDataset(
            df=df,
            img_root=self.img_root,
            label_col=self.label_col,
            sample_col=self.sample_col,
            batch_col=self.batch_col,
            ext=self.ext,
            nested_by_batch=self.nested_by_batch,
            transforms=tfms,
            label_encoder=self.encoder,
            domain_labels=domain_labels,
        )

    def train_dataloader(self) -> DataLoader:
        ds = self._build_ds(self.train_df, self.train_tfms)
        pw = self.persistent_workers and self.num_workers > 0
        pf = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=pw, prefetch_factor=pf)

    def val_dataloader(self) -> DataLoader:
        ds = self._build_ds(self.val_df, self.eval_tfms)
        pw = self.persistent_workers and self.num_workers > 0
        pf = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=pw, prefetch_factor=pf)

    def test_dataloader(self) -> DataLoader:
        # Sort by domain so consecutive mini-batches belong to a single domain.
        # This ensures TTT-BN computes per-domain batch statistics rather than
        # blended statistics from a shuffled mix of multiple test domains.
        # All other methods are unaffected by this ordering.
        test_df_sorted = self.test_df.sort_values(self.batch_col).reset_index(drop=True)
        ds = self._build_ds(test_df_sorted, self.eval_tfms)
        pw = self.persistent_workers and self.num_workers > 0
        pf = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=pw, prefetch_factor=pf)

    @property
    def num_classes(self) -> int:
        return len(self.encoder.classes_)

    @property
    def num_domains(self) -> int:
        """Number of distinct domains (batches). Only valid after setup()."""
        return len(self.domain_encoder.classes_)

    @property
    def split_summary(self) -> Dict[str, Any]:
        return self.split_info.copy()

