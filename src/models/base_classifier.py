"""Base classifier for all JUMP-CP domain-shift experiments.

Provides:
- ResNet backbone loading with 5-channel first-conv adaptation
- Feature extractor / classifier head split (DANN adds a domain head on top)
- Stage-separated metrics (train / val / test) to avoid state leakage
- Per-class and per-domain (batch) accuracy logged at test time
- Confusion matrix logged at test time
- Shared configure_optimizers (AdamW + optional cosine LR)

Subclasses override _build_backbone() and/or training_step() as needed.

Per-domain accuracy:
    When the DataModule is configured with ``return_domain=True`` the batch
    is a 3-tuple ``(images, class_labels, domain_labels)``.  BaseClassifier
    accumulates predictions grouped by domain during test and logs
    ``test_acc_domain_{name}`` for every domain that appears in the test
    split.  This works for both K=5 (possibly 2 domains per fold) and LOBO
    (exactly 1 domain per fold).

    Pass ``domain_names`` (list from ``DataModule.domain_encoder.classes_``)
    to get human-readable batch names instead of integer indices.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as tv_models
from torchmetrics import Accuracy, ConfusionMatrix


class BaseClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 5,
        num_classes: int = 8,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        label_smoothing: float = 0.0,
        pretrained: bool = True,
        scheduler: Optional[str] = None,       # "cosine" | None
        domain_names: Optional[List[str]] = None,  # from DataModule.domain_encoder.classes_
    ):
        super().__init__()
        self.save_hyperparameters()
        # domain_names is metadata, not a model weight — keep separately
        self.domain_names = domain_names

        self.feature_extractor, feature_dim = self._build_backbone()
        self.classifier_head = nn.Linear(feature_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Stage-separated metrics — avoids running-state leakage across stages
        # average="macro": equal weight per class, robust to per-domain class imbalance
        mk = dict(task="multiclass", num_classes=num_classes, average="macro")
        self.train_acc          = Accuracy(**mk)
        self.val_acc            = Accuracy(**mk)
        self.test_acc           = Accuracy(**mk)
        self.test_acc_per_class = Accuracy(task="multiclass", num_classes=num_classes,
                                           average="none")
        self.test_conf_mat      = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        # Buffers for per-domain accuracy (populated only when domain labels present)
        self._test_preds:      List[torch.Tensor] = []
        self._test_targets:    List[torch.Tensor] = []
        self._test_domain_ids: List[torch.Tensor] = []

    # ------------------------------------------------------------------
    # Backbone construction (override in subclasses, e.g. InstanceNorm)
    # ------------------------------------------------------------------

    def _build_backbone(self) -> tuple[nn.Module, int]:
        """Return (feature_extractor, feature_dim).

        The feature extractor is a pretrained ResNet with:
          - first conv adapted to hparams.in_channels
          - final fc replaced by nn.Identity so forward() outputs features
        """
        hp = self.hparams
        ctor = getattr(tv_models, hp.backbone, None)
        if ctor is None:
            raise ValueError(f"Unknown torchvision backbone: {hp.backbone}")

        # Support both new (weights=) and old (pretrained=) torchvision APIs
        try:
            weights_enum = getattr(tv_models, f"{hp.backbone}_Weights", None)
            weights = weights_enum.DEFAULT if (hp.pretrained and weights_enum) else None
            model = ctor(weights=weights)
        except TypeError:
            model = ctor(pretrained=hp.pretrained)

        # Adapt first conv to in_channels (e.g. 5 for JUMP-CP)
        if hp.in_channels != 3:
            old = model.conv1
            new_conv = nn.Conv2d(
                hp.in_channels, old.out_channels,
                old.kernel_size, old.stride, old.padding,
                bias=(old.bias is not None),
            )
            with torch.no_grad():
                if old.weight.shape[1] == 3:
                    w = old.weight                           # [out_c, 3, k, k]
                    w_mean = w.mean(dim=1, keepdim=True)    # [out_c, 1, k, k]
                    if hp.in_channels >= 3:
                        new_conv.weight[:, :3] = w
                        if hp.in_channels > 3:
                            new_conv.weight[:, 3:] = w_mean.expand(
                                -1, hp.in_channels - 3, -1, -1
                            )
                    else:
                        new_conv.weight[:, :hp.in_channels] = w[:, :hp.in_channels]
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out",
                                            nonlinearity="relu")
                if new_conv.bias is not None:
                    nn.init.zeros_(new_conv.bias)
            model.conv1 = new_conv

        feature_dim = model.fc.in_features
        model.fc = nn.Identity() # replace final fc with identity - we do not want to have the 1000-dim output of ImageNet pretraining, but rather the feature vector before it
        return model, feature_dim # for Res-net-50, feature_dim=2048

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier_head(self.extract_features(x))

    # ------------------------------------------------------------------
    # Shared step (handles 2- and 3-tuple batches; DANN overrides)
    # ------------------------------------------------------------------

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch[0], batch[1]  # ignore domain label (batch[2]) here
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        if stage == "train":
            self.train_acc(preds, y)
            self.log("train_acc", self.train_acc, prog_bar=True,
                     on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_acc(preds, y)
            self.log("val_acc", self.val_acc, prog_bar=True,
                     on_step=False, on_epoch=True)
        else:  # test
            self.test_acc(preds, y)
            self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
            self.test_acc_per_class.update(preds, y)
            self.test_conf_mat.update(preds, y)
            # Accumulate for per-domain accuracy (only when domain labels present)
            if len(batch) == 3:
                self._test_preds.append(preds.cpu())
                self._test_targets.append(y.cpu())
                self._test_domain_ids.append(batch[2].cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ------------------------------------------------------------------
    # Epoch-end hooks for per-class acc + confusion matrix
    # ------------------------------------------------------------------

    def on_test_epoch_end(self):
        # Per-class accuracy
        per_class = self.test_acc_per_class.compute()
        for i, acc in enumerate(per_class):
            self.log(f"test_acc_class_{i}", acc)
        self.test_acc_per_class.reset()

        # Per-domain (batch) accuracy — works for both K=5 and LOBO
        if self._test_domain_ids:
            all_preds   = torch.cat(self._test_preds)
            all_targets = torch.cat(self._test_targets)
            all_domains = torch.cat(self._test_domain_ids)
            for d in all_domains.unique():
                mask = all_domains == d
                # macro accuracy over classes within this domain
                acc = Accuracy(task="multiclass",
                               num_classes=self.hparams.num_classes,
                               average="macro")(all_preds[mask], all_targets[mask])
                name = (self.domain_names[d.item()]
                        if self.domain_names is not None else str(d.item()))
                self.log(f"test_acc_domain_{name}", acc)
            self._test_preds.clear()
            self._test_targets.clear()
            self._test_domain_ids.clear()

        # Confusion matrix
        cm = self.test_conf_mat.compute()
        self.test_conf_mat.reset()
        self._log_confusion_matrix(cm)

    def _log_confusion_matrix(self, cm: torch.Tensor):
        """Log confusion matrix to W&B (if available) as a heatmap image."""
        try:
            from pytorch_lightning.loggers import WandbLogger
            if not isinstance(self.logger, WandbLogger):
                return
            import wandb
            import numpy as np
            import matplotlib.pyplot as plt

            n = self.hparams.num_classes
            fig, ax = plt.subplots(figsize=(n, n))
            mat = cm.cpu().numpy().astype(int)
            im = ax.imshow(mat, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            for r in range(n):
                for c in range(n):
                    ax.text(c, r, str(mat[r, c]), ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            self.logger.experiment.log(
                {"test_confusion_matrix": wandb.Image(fig)}, commit=False,
            )
            plt.close(fig)
        except Exception:
            pass  # non-W&B loggers: skip

    # ------------------------------------------------------------------
    # Optimiser + optional cosine LR scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
        if hp.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(self.parameters(), lr=hp.lr,
                                    weight_decay=hp.weight_decay)
        else:
            opt = torch.optim.Adam(self.parameters(), lr=hp.lr,
                                   weight_decay=hp.weight_decay)

        if hp.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt
