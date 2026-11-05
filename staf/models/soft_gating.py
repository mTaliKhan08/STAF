"""
IMU-Guided Soft-Gating Module for STAF Framework.

Implements the learned soft-gating mechanism that maps the kinematic
state vector (causal 3-d [v/10, a, omega] or 9-d pre-integrated state)
to continuous reliability weights w_cam, w_lidar in [0, 1].

Supports two gate modes:
    - "learned": Evaluates the learned MLP heads from kinematic inputs.
    - "constant": Sets fixed equal weights w_cam = w_lidar = 0.5.
"""

import torch
import torch.nn as nn


class SoftGatingHead(nn.Module):
    """
    Single soft-gating head for one sensor modality.

    Maps the kinematic vector to a scalar reliability weight in [0, 1].
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
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
            x_kinematic: Kinematic vector of shape (B, D) or (D,), with D in {3, 9}.

        Returns:
            Reliability weight of shape (B, 1) or (1,).
        """
        if x_kinematic.dim() == 1:
            x_kinematic = x_kinematic.unsqueeze(0)

        if x_kinematic.shape[-1] != self.input_dim:
            if x_kinematic.shape[-1] == 3 and self.input_dim == 9:
                x_kinematic = torch.cat(
                    [x_kinematic, torch.zeros(x_kinematic.size(0), 6, device=x_kinematic.device, dtype=x_kinematic.dtype)],
                    dim=-1,
                )
            elif x_kinematic.shape[-1] == 9 and self.input_dim == 3:
                x_kinematic = x_kinematic[..., :3]
            else:
                raise ValueError(f"Expected kinematic dimension {self.input_dim}, got {x_kinematic.shape[-1]}.")

        return self.mlp(x_kinematic)


class IMUGuidedSoftGating(nn.Module):
    """
    IMU-Guided Soft-Gating Module.

    Produces per-modality reliability weights for camera and LiDAR.
    Supports 'learned' mode (trained MLP) and 'constant' mode (fixed 0.5/0.5).
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        gate_mode: str = "learned",
    ):
        super().__init__()
        if gate_mode not in ["learned", "constant"]:
            raise ValueError(f"gate_mode must be 'learned' or 'constant', got {gate_mode}")
        self.input_dim = input_dim
        self.gate_mode = gate_mode
        self.cam_head = SoftGatingHead(input_dim, hidden_dim, dropout)
        self.lidar_head = SoftGatingHead(input_dim, hidden_dim, dropout)

    def forward(self, x_kinematic: torch.Tensor):
        """
        Args:
            x_kinematic: Kinematic vector of shape (B, D) or (D,), with D in {3, 9}.

        Returns:
            w_cam: Camera reliability weight, shape (B, 1).
            w_lidar: LiDAR reliability weight, shape (B, 1).
        """
        if x_kinematic is None:
            raise ValueError("x_kinematic must be provided for the full-fusion modality.")

        if x_kinematic.dim() == 1:
            x_kinematic = x_kinematic.unsqueeze(0)

        bs = x_kinematic.size(0)
        device = x_kinematic.device
        dtype = x_kinematic.dtype

        if self.gate_mode == "constant":
            w_cam = torch.full((bs, 1), 0.5, device=device, dtype=dtype)
            w_lidar = torch.full((bs, 1), 0.5, device=device, dtype=dtype)
            return w_cam, w_lidar

        w_cam = self.cam_head(x_kinematic)
        w_lidar = self.lidar_head(x_kinematic)
        return w_cam, w_lidar

    def get_param_count(self) -> int:
        """Return total trainable parameter count."""
        if self.gate_mode == "constant":
            return 0
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Canonical alias for compatibility
DynamicSoftGating = IMUGuidedSoftGating
