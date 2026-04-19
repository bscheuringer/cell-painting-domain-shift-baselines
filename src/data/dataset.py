from pathlib import Path
from typing import Optional

import numpy as np
import tifffile as tiff
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class JumpCPDataset(Dataset):
    """
    Dataset for JUMP-CP source-3:

    - Single file per sample: <img_root>/<Metadata_Sample_ID>.jpg
    - Files are TIFF on disk (even though having .jpg suffix)
    - Each file is (H, W, 5) uint8
    - Scales to [0, 1] float32 and returns a (C, H, W) tensor with C=5
    """

    def __init__(
        self,
        df,
        img_root: str,
        label_col: str = "Metadata_JCP2022",
        sample_col: str = "Metadata_Sample_ID",
        batch_col: str = "Metadata_Batch",
        ext: str = ".jpg",
        nested_by_batch: bool = False,
        transforms: Optional[A.Compose] = None,
        label_encoder=None,
        domain_labels: Optional[np.ndarray] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.label_col = label_col
        self.sample_col = sample_col
        self.batch_col = batch_col
        self.ext = ext if ext.startswith(".") else f".{ext}"
        self.nested_by_batch = nested_by_batch
        self.transforms = transforms
        self.label_encoder = label_encoder
        self.domain_labels = domain_labels  # int64 array, one entry per row; None means no domain

        # Pre-build paths and encode labels once at init to avoid per-sample
        # pandas/sklearn overhead inside __getitem__ (which runs in DataLoader workers).
        if self.nested_by_batch:
            # <img_root>/<batch>/<site>/<sample>.jpg
            # site = last character of sample ID (site number 1-9)
            self._paths: list[str] = [
                str(self.img_root / batch / sid[-1] / f"{sid}{self.ext}")
                for sid, batch in zip(self.df[sample_col], self.df[batch_col])
            ]
        else:
            self._paths: list[str] = [
                str(self.img_root / f"{sid}{self.ext}")
                for sid in self.df[sample_col]
            ]
        if label_encoder is not None:
            self._labels: np.ndarray = label_encoder.transform(
                self.df[label_col]
            ).astype(np.int64)
        else:
            self._labels = np.full(len(self.df), -1, dtype=np.int64)

    def _row_to_path(self, idx) -> str:
        return self._paths[idx]

    def _read_image(self, path: str) -> np.ndarray:
        # Prefer .npy (fast memmap load) over TIFF decode when available
        npy_path = path.replace(".jpg", ".npy")
        try:
            img = np.load(npy_path)
        except FileNotFoundError:
            img = tiff.imread(str(path), maxworkers=1)
        if img.ndim != 3 or img.shape[-1] != 5:
            raise ValueError(f"Expected (H, W, 5) image, got shape {img.shape} at {path}")
        return img.astype(np.float32) / 255.0

    def __getitem__(self, idx):
        path = self._row_to_path(idx)
        img = self._read_image(path)  # (H,W,5)

        if self.transforms:
            # Albumentations expects HWC; ToTensorV2 will convert to CHW
            img = self.transforms(image=img)["image"]  # torch.Tensor (5,H,W)

        y = self._labels[idx]

        if self.domain_labels is not None:
            return img, y, int(self.domain_labels[idx])
        return img, y

    def __len__(self):
        return len(self.df)


def build_transforms(image_size: int = 224, train: bool = True) -> A.Compose:
    resize = [
        A.LongestMaxSize(max_size=image_size),
        A.PadIfNeeded(image_size, image_size, border_mode=0, fill=0),
    ]
    if train:
        return A.Compose([
            *resize,
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            ToTensorV2(),
        ])
    return A.Compose([
        *resize,
        ToTensorV2(),
    ])
