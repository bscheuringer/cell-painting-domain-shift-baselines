"""DANN — Domain-Adversarial Neural Network (Ganin et al. 2016).

Architecture
------------
feature_extractor (ResNet-50, 5ch)
    │
    ├──► classifier_head  → class logits   (cross-entropy loss, normal grads)
    │
    └──► GRL(λ) ──► domain_head → domain logits  (cross-entropy loss, reversed grads)

Training objective
------------------
    loss = class_loss + domain_loss

The Gradient Reversal Layer (GRL) multiplies the gradient flowing from the
domain loss into the feature extractor by -λ, so the extractor learns to be
*domain-invariant* while the domain classifier is trained normally.

Lambda schedule (Ganin et al. 2016 §3.2)
-----------------------------------------
    λ(p) = 2 / (1 + exp(-γ · p)) − 1,    p = step / total_steps ∈ [0, 1]

λ ramps smoothly from 0 (no adversarial signal at start) to ≈1 (full signal
at the end).  γ=10 is the standard value from the paper.

Logging
-------
    train_loss          total loss
    train_class_loss    cross-entropy on class labels
    train_domain_loss   cross-entropy on domain labels
    train_acc           macro class accuracy
    train_domain_acc    domain classifier accuracy (should ↓ toward chance)
    grl_lambda          current λ value (logged per step)

Val / test stages inherit from BaseClassifier (class loss + metrics only).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
from torchmetrics import Accuracy

from src.models.base_classifier import BaseClassifier


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GRLFunction(torch.autograd.Function):
    """Identity forward; scaled gradient negation backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()  # new tensor — avoids in-place mutation issues

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_ * grad_output, None  # reverse & scale; no grad for lambda_


class GradientReversal(nn.Module):
    """Stateless wrapper so GRL can be used like any nn.Module."""

    def forward(self, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        return _GRLFunction.apply(x, lambda_)


# ---------------------------------------------------------------------------
# DANN classifier
# ---------------------------------------------------------------------------

class DANNClassifier(BaseClassifier):
    """Domain-Adversarial Neural Network built on top of BaseClassifier."""

    def __init__(
        self,
        backbone: str = "resnet50",
        in_channels: int = 5,
        num_classes: int = 8,
        num_domains: int = 9,
        domain_hidden_dim: int = 1024,
        grl_gamma: float = 10.0,
        penalty_weight: float = 1.0,      # scalar weight on domain loss (WILDS-style tuning knob)
        discriminator_lr: float = None,   # if set, domain head uses a separate (usually higher) LR
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        optimizer: str = "adamw",
        label_smoothing: float = 0.0,
        pretrained: bool = True,
        scheduler: Optional[str] = None,
        domain_names: Optional[List[str]] = None,
    ):
        super().__init__(
            backbone=backbone,
            in_channels=in_channels,
            num_classes=num_classes,
            lr=lr,
            weight_decay=weight_decay,
            optimizer=optimizer,
            label_smoothing=label_smoothing,
            pretrained=pretrained,
            scheduler=scheduler,
            domain_names=domain_names,
        )
        # Override hparams to include DANN-specific params
        self.save_hyperparameters()

        feature_dim = self.classifier_head.in_features

        self.grl = GradientReversal()

        # 3-layer domain discriminator with BatchNorm (matches WILDS reference architecture)
        h = domain_hidden_dim
        self.domain_head = nn.Sequential(
            nn.Linear(feature_dim, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.BatchNorm1d(h),
            nn.ReLU(),
            nn.Linear(h, num_domains),
        )
        self.domain_criterion = nn.CrossEntropyLoss()
        self.train_domain_acc = Accuracy(task="multiclass", num_classes=num_domains)

    # ------------------------------------------------------------------
    # Lambda schedule
    # ------------------------------------------------------------------

    def _compute_lambda(self) -> float:
        """Ganin et al. 2016 schedule: ramps λ from 0 → ~1 over training."""
        try:
            total = self.trainer.estimated_stepping_batches
            p = self.global_step / total if total and total > 0 else 0.0
        except Exception:
            p = 0.0
        p = min(max(p, 0.0), 1.0)
        return float(2.0 / (1.0 + math.exp(-self.hparams.grl_gamma * p)) - 1.0)

    # ------------------------------------------------------------------
    # Training step (overrides BaseClassifier; val/test inherited)
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        x, y_class, y_domain = batch
        lambda_ = self._compute_lambda()

        features = self.extract_features(x)

        # --- Classification branch (normal gradient flow) ---
        class_logits = self.classifier_head(features)
        class_loss = self.criterion(class_logits, y_class)

        # --- Domain branch (gradient reversed through GRL) ---
        domain_logits = self.domain_head(self.grl(features, lambda_))
        domain_loss = self.domain_criterion(domain_logits, y_domain)

        loss = class_loss + self.hparams.penalty_weight * domain_loss

        # Metrics
        class_preds  = class_logits.argmax(dim=1)
        domain_preds = domain_logits.argmax(dim=1)
        self.train_acc(class_preds, y_class)
        self.train_domain_acc(domain_preds, y_domain)

        self.log("train_loss",        loss,                  prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_class_loss",  class_loss,            prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_domain_loss", domain_loss,           prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_acc",         self.train_acc,        prog_bar=True,  on_step=False, on_epoch=True)
        self.log("train_domain_acc",  self.train_domain_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("grl_lambda",        lambda_,               prog_bar=False, on_step=True,  on_epoch=False)

        return loss

    # ------------------------------------------------------------------
    # Optimiser — optional separate LR for domain discriminator
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        hp = self.hparams
        disc_lr = hp.discriminator_lr if hp.discriminator_lr is not None else hp.lr

        backbone_params   = list(self.feature_extractor.parameters())
        classifier_params = list(self.classifier_head.parameters())
        domain_params     = list(self.domain_head.parameters())

        param_groups = [
            {"params": backbone_params + classifier_params, "lr": hp.lr},
            {"params": domain_params,                       "lr": disc_lr},
        ]

        if hp.optimizer.lower() == "adamw":
            opt = torch.optim.AdamW(param_groups, weight_decay=hp.weight_decay)
        else:
            opt = torch.optim.Adam(param_groups, weight_decay=hp.weight_decay)

        if hp.scheduler == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.trainer.max_epochs
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt
