"""ERM (Empirical Risk Minimization) classifier for JUMP-CP.

Standard supervised training with cross-entropy loss.  No domain
adaptation — serves as the baseline against which DANN, InstanceNorm
and BatchNorm are compared.

Inherits all behaviour from BaseClassifier:
- ResNet-50 backbone (5-channel first-conv)
- Macro-averaged accuracy metrics (train / val / test)
- Per-class accuracy and confusion matrix at test time
- Per-domain accuracy at test time (when return_domain=True)
- AdamW optimiser + optional cosine LR scheduler
"""

from src.models.base_classifier import BaseClassifier


class ERMClassifier(BaseClassifier):
    """Empirical Risk Minimization — plain cross-entropy, no domain adaptation."""
    pass
