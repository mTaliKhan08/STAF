"""
Class-Balanced Focal Loss for STAF Framework.

Implements the loss function from Thesis-1 Equation 10:
    L_balanced = -alpha_c * (1 - p_t)^gamma * log(p_t)

Addresses the extreme long-tail class distribution in autonomous driving
datasets where safety-critical classes (pedestrians, cyclists) account
for < 1% of raw samples, while dominant classes (road, building) dominate
the gradient signal.

Uses Effective Number of Samples weighting (Cui et al., CVPR 2019) to
compute per-class weights alpha_c.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ClassBalancedFocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss.

    Combines:
    1. Focal modulation: (1 - p_t)^gamma to down-weight easy, well-classified examples.
    2. Class-balanced weighting: alpha_c based on effective number of samples to
       up-weight rare classes (pedestrians, cyclists, small obstacles).

    Args:
        num_classes: Number of semantic classes.
        class_counts: Per-class sample counts in the training set, shape (C,).
        gamma: Focusing parameter (default 2.0). Higher = more focus on hard examples.
        beta: Effective number hyperparameter (default 0.999).
            Controls the rate of diminishing returns for additional samples.
        reduction: 'mean', 'sum', or 'none'.
    """

    def __init__(
        self,
        num_classes: int,
        class_counts: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        beta: float = 0.999,
        reduction: str = "mean",
    ):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction

        # Compute effective number of samples weights
        if class_counts is not None:
            self.register_buffer("alpha", self._compute_weights(class_counts, beta))
        else:
            self.register_buffer("alpha", torch.ones(num_classes))

    @staticmethod
    def _compute_weights(class_counts: torch.Tensor, beta: float) -> torch.Tensor:
        """
        Compute class-balanced weights using Effective Number of Samples.

        E_n = (1 - beta^n) / (1 - beta)
        alpha_c = 1 / E_n (normalized)

        Reference: Cui et al., "Class-Balanced Loss Based on Effective Number
        of Samples", CVPR 2019.
        """
        effective_num = 1.0 - torch.pow(beta, class_counts.float())
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # Normalize so that sum of weights = num_classes
        weights = weights / weights.sum() * len(weights)
        return weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the class-balanced focal loss.

        Args:
            logits: Raw model predictions of shape (B, C, ...) where C = num_classes.
            targets: Ground truth class indices of shape (B, ...).

        Returns:
            Scalar loss value (if reduction='mean' or 'sum') or per-element loss.
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather the probability of the true class: p_t
        # targets shape: (B, ...) -> (B, 1, ...) for gather
        targets_one_hot = targets.unsqueeze(1)  # (B, 1, ...)
        p_t = probs.gather(1, targets_one_hot).squeeze(1)  # (B, ...)

        # Focal modulation: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Class-balanced weight: alpha_c for each sample
        alpha_t = self.alpha[targets]  # (B, ...)

        # Loss: -alpha_c * (1 - p_t)^gamma * log(p_t)
        loss = -alpha_t * focal_weight * torch.log(p_t + 1e-8)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def update_class_counts(self, class_counts: torch.Tensor):
        """Update the class-balanced weights with new class counts."""
        self.alpha = self._compute_weights(class_counts, self.beta).to(self.alpha.device)
