"""
Smoke tests — verify the full forward pass of every method without real data.

Uses synthetic (5, 224, 224) float32 tensors so these tests run on any
machine that has the conda env, with no parquet or image files required.

Run from repo root:
    pytest tests/test_smoke.py -v
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import pytest
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from src.models.erm_classifier import ERMClassifier
from src.models.dann_classifier import DANNClassifier
from src.models.in_classifier import InstanceNormClassifier
from src.models.ttt_bn_classifier import TTTBatchNormClassifier
from src.utils.logger import build_logger

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
NUM_DOMAINS = 4
BATCH_SIZE  = 4
N_BATCHES   = 3      # total synthetic batches per split
IMG_SIZE    = 224    # keep the same spatial size as production


def _common_kwargs() -> dict:
    return dict(
        backbone="resnet50",
        in_channels=5,
        num_classes=NUM_CLASSES,
        lr=1e-3,
        weight_decay=0.0,
        optimizer="adamw",
        label_smoothing=0.0,
        pretrained=False,   # skip downloading weights — not needed for shape checks
        scheduler=None,
        domain_names=[f"batch{i}" for i in range(NUM_DOMAINS)],
    )


# ---------------------------------------------------------------------------
# Synthetic DataLoaders
# ---------------------------------------------------------------------------

def _make_loader(with_domain: bool = False) -> DataLoader:
    """
    Returns a DataLoader yielding random (image, class_label [, domain_label]) tuples.

    with_domain=True  → 3-tuple (required by DANN)
    with_domain=False → 2-tuple
    """
    n = BATCH_SIZE * N_BATCHES
    images  = torch.randn(n, 5, IMG_SIZE, IMG_SIZE)
    labels  = torch.randint(0, NUM_CLASSES, (n,))
    if with_domain:
        domains = torch.randint(0, NUM_DOMAINS, (n,))
        ds = TensorDataset(images, labels, domains)
    else:
        ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


def _make_trainer() -> pl.Trainer:
    """Minimal Trainer: 1 epoch, CPU, no logging, no checkpointing."""
    return pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


# ---------------------------------------------------------------------------
# 1. Logger factory
# ---------------------------------------------------------------------------

def test_logger_none():
    cfg = OmegaConf.create({"type": "none"})
    result = build_logger(cfg)
    assert result is False, "logger=none should return False (disables PL logger)"


def test_logger_unknown_type_raises():
    cfg = OmegaConf.create({"type": "tensorboard"})
    with pytest.raises(ValueError, match="Unknown logger type"):
        build_logger(cfg)


# ---------------------------------------------------------------------------
# 2. ERM — standard cross-entropy, no domain adaptation
# ---------------------------------------------------------------------------

def test_erm_forward():
    model = ERMClassifier(**_common_kwargs())
    loader = _make_loader(with_domain=False)
    trainer = _make_trainer()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)


def test_erm_with_domain_labels():
    """ERM should silently ignore the third element of the batch tuple."""
    model = ERMClassifier(**_common_kwargs())
    loader = _make_loader(with_domain=True)
    trainer = _make_trainer()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)


# ---------------------------------------------------------------------------
# 3. InstanceNorm — all BN2d replaced by IN2d
# ---------------------------------------------------------------------------

def test_instance_norm_forward():
    model = InstanceNormClassifier(**_common_kwargs())
    # Assert that no BatchNorm2d survived the replacement
    bn_layers = [m for m in model.feature_extractor.modules()
                 if isinstance(m, nn.BatchNorm2d)]
    assert len(bn_layers) == 0, "InstanceNormClassifier must have zero BN2d layers"

    loader = _make_loader(with_domain=False)
    trainer = _make_trainer()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)

# ---------------------------------------------------------------------------
# 5. DANN — adversarial domain head, GRL, 3-tuple batches
# ---------------------------------------------------------------------------

def test_dann_forward():
    kwargs = _common_kwargs()
    kwargs.pop("domain_names")   # re-add with correct num_domains
    model = DANNClassifier(
        **kwargs,
        num_domains=NUM_DOMAINS,
        penalty_weight=1.0,
        domain_names=[f"batch{i}" for i in range(NUM_DOMAINS)],
    )
    loader = _make_loader(with_domain=True)
    trainer = _make_trainer()
    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)
    trainer.test(model, dataloaders=loader)


def test_dann_domain_head_shape():
    """Domain head output must have shape (batch, num_domains)."""
    kwargs = _common_kwargs()
    kwargs.pop("domain_names")
    model = DANNClassifier(
        **kwargs,
        num_domains=NUM_DOMAINS,
        domain_names=None,
    )
    model.eval()
    x = torch.randn(BATCH_SIZE, 5, IMG_SIZE, IMG_SIZE)
    features = model.extract_features(x)
    domain_logits = model.domain_head(model.grl(features, 1.0))
    assert domain_logits.shape == (BATCH_SIZE, NUM_DOMAINS), (
        f"Expected ({BATCH_SIZE}, {NUM_DOMAINS}), got {domain_logits.shape}"
    )
