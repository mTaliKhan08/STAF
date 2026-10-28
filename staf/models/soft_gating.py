"""
IMU-Guided Soft-Gating Module for STAF Framework.

Implements the learned soft-gating mechanism that takes the 9-DoF kinematic
delta state vector from the IMU pre-integrator and outputs real-time
reliability weights for the camera and LiDAR feature streams.

Architecture (Fix #8 - fully specified):
    Input: X_kinematic in R^9
    -> Linear(9, 64) -> LayerNorm(64) -> ReLU -> Dropout(0.1)
    -> Linear(64, 1) -> Sigmoid
    Output: w in [0, 1]

Two independent heads: one for w_cam, one for w_lidar.
Total additional parameters: ~1,350 (negligible overhead).
"""

import torch
import torch.nn as nn


class SoftGatingHead(nn.Module):
    """
    Single soft-gating head for one sensor modality.

    Maps the 9-DoF kinematic vector to a scalar reliability weight in [0, 1].
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_kinematic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_kinematic: Kinematic vector of shape (B, 9) or (9,).

        Returns:
            Reliability weight of shape (B, 1) or (1,).
        """
        return self.mlp(x_kinematic)


class IMUGuidedSoftGating(nn.Module):
    """
    IMU-Guided Soft-Gating Module.

    Produces per-modality reliability weights from the pre-integrated
    kinematic state vector. The weights are used to scale the feature
    tokens before cross-attention fusion:

        f_hat_cam   = w_cam   * f_cam
        f_hat_lidar = w_lidar * f_lidar

    The two heads are independent (no shared weights) so that each
    modality's trust can be estimated independently based on the
    current motion dynamics.

    Thesis-1 Equations 6-9.
    """

    def __init__(self, input_dim: int = 9, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.cam_head = SoftGatingHead(input_dim, hidden_dim, dropout)
        self.lidar_head = SoftGatingHead(input_dim, hidden_dim, dropout)

    def forward(self, x_kinematic: torch.Tensor):
        """
        Args:
            x_kinematic: Kinematic vector of shape (B, 9).

        Returns:
            w_cam: Camera reliability weight, shape (B, 1).
            w_lidar: LiDAR reliability weight, shape (B, 1).
        """
        w_cam = self.cam_head(x_kinematic)
        w_lidar = self.lidar_head(x_kinematic)
        return w_cam, w_lidar

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Canonical alias for compatibility
DynamicSoftGating = IMUGuidedSoftGating
