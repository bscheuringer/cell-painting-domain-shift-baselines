"""InstanceNorm classifier for JUMP-CP domain-shift experiments.

Replaces every BatchNorm2d in the ResNet backbone with InstanceNorm2d.

Motivation
----------
BatchNorm normalises over the batch dimension and accumulates running
statistics.  In a multi-domain setting those statistics mix information
across domains and can harm generalisation when the test domain is unseen.

InstanceNorm normalises each sample independently over its spatial
dimensions (H×W) — there are no batch-level or cross-domain statistics.
This makes the feature representations more domain-agnostic, at the cost
of slightly weaker normalisation signal.

Implementation
--------------
``_build_backbone()`` calls the parent implementation (which handles the
pretrained weights and 5-channel first-conv adaptation) and then walks the
network recursively, replacing every ``nn.BatchNorm2d`` with
``nn.InstanceNorm2d(affine=True)``.  All other layers are unchanged.

Everything else (training loop, metrics, optimiser) is identical to ERM
and is inherited from BaseClassifier without modification.
"""

from __future__ import annotations

import torch.nn as nn

from src.models.base_classifier import BaseClassifier


def _replace_bn_with_in(module: nn.Module) -> None:
    """Recursively replace every BatchNorm2d with InstanceNorm2d in-place.

    Args:
        module: any nn.Module (modified in-place, nothing returned).

    Notes:
        - ``affine=True``: keeps learnable scale/shift parameters (same as BN).
        - ``track_running_stats=False``: IN has no running mean/var by design.
        - Affine params are left at their default init (weight=1, bias=0);
          copying BN affine weights would be misleading because BN and IN use
          completely different normalisation statistics.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name,
                    nn.InstanceNorm2d(child.num_features,
                                      affine=True,
                                      track_running_stats=False))
        else:
            _replace_bn_with_in(child)


class InstanceNormClassifier(BaseClassifier):
    """ResNet-50 with all BatchNorm2d layers replaced by InstanceNorm2d."""

    def _build_backbone(self):
        # Let BaseClassifier handle pretrained loading + 5-channel first-conv
        feature_extractor, feature_dim = super()._build_backbone()
        # Then swap every BN layer for IN
        _replace_bn_with_in(feature_extractor)
        return feature_extractor, feature_dim
