"""
IMU Pre-integration Module for STAF Framework.

Implements on-manifold IMU pre-integration following:
- Forster et al., "On-Manifold Preintegration" (TRO 2017)
- Thesis-1 Report, Equations 3-5 (Section 5.2)

Converts raw high-frequency IMU measurements into compact 9-DoF
kinematic delta state vectors for the soft-gating MLP.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    """Compute the skew-symmetric matrix [v]_x from a 3D vector v.
    
    Handles both batched (..., 3) and unbatched (3,) inputs.
    """
    v0, v1, v2 = v[..., 0], v[..., 1], v[..., 2]
    zero = torch.zeros_like(v0)
    # Stack into (..., 9) then reshape to (..., 3, 3)
    mat = torch.stack([
        zero, -v2,  v1,
        v2,   zero, -v0,
        -v1,  v0,   zero
    ], dim=-1)
    out_shape = v.shape[:-1] + (3, 3)
    return mat.reshape(out_shape)


def so3_exp(omega: torch.Tensor) -> torch.Tensor:
    """Exponential map from so(3) to SO(3) using Rodrigues' formula.
    
    Handles both batched (..., 3) and unbatched (3,) inputs.
    """
    # Ensure at least 2D for uniform processing
    squeeze = omega.dim() == 1
    if squeeze:
        omega = omega.unsqueeze(0)

    theta_sq = (omega * omega).sum(dim=-1, keepdim=True)  # (..., 1)
    theta = torch.sqrt(theta_sq + 1e-12)  # (..., 1)

    omega_hat = omega / (theta + 1e-12)  # (..., 3)
    K = skew_symmetric(omega_hat)  # (..., 3, 3)

    sin_theta = torch.sin(theta).unsqueeze(-1)   # (..., 1, 1)
    cos_theta = torch.cos(theta).unsqueeze(-1)   # (..., 1, 1)

    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand_as(K)
    R = I + sin_theta * K + (1 - cos_theta) * (K @ K)

    # Small angle correction: R ≈ I + [ω]_x
    small_angle = (theta.squeeze(-1) < 1e-6)  # (...)
    if small_angle.any():
        K_small = skew_symmetric(omega)
        R_small = I + K_small
        R = torch.where(small_angle.unsqueeze(-1).unsqueeze(-1), R_small, R)

    if squeeze:
        R = R.squeeze(0)
    return R


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Logarithmic map from SO(3) to so(3).
    
    Handles both batched (..., 3, 3) and unbatched (3, 3) inputs.
    """
    squeeze = R.dim() == 2
    if squeeze:
        R = R.unsqueeze(0)

    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_theta = torch.clamp((trace - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    small = (theta.abs() < 1e-6)

    skew_part = (R - R.transpose(-2, -1)) / 2.0
    omega = torch.stack([
        skew_part[..., 2, 1],
        skew_part[..., 0, 2],
        skew_part[..., 1, 0]
    ], dim=-1)

    sin_theta = torch.sin(theta)
    scale = theta / (sin_theta + 1e-12)
    scale = torch.where(small, torch.ones_like(scale), scale)
    omega = omega * scale.unsqueeze(-1)

    if squeeze:
        omega = omega.squeeze(0)
    return omega


class IMUPreintegrator:
    """
    On-manifold IMU pre-integration between two keyframes.

    Integrates raw accelerometer and gyroscope measurements over the temporal
    window [t_i, t_j] between successive camera/LiDAR frames to produce the
    9-DoF kinematic delta state vector X_kinematic.

    Implements Thesis-1 Equations 3-5.
    """

    def __init__(
        self,
        bias_gyro: Optional[torch.Tensor] = None,
        bias_accel: Optional[torch.Tensor] = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.delta_R = torch.eye(3, device=device, dtype=dtype)
        self.delta_v = torch.zeros(3, device=device, dtype=dtype)
        self.delta_p = torch.zeros(3, device=device, dtype=dtype)
        self.bias_gyro = bias_gyro if bias_gyro is not None else torch.zeros(3, device=device, dtype=dtype)
        self.bias_accel = bias_accel if bias_accel is not None else torch.zeros(3, device=device, dtype=dtype)
        # Jacobians for first-order bias correction
        self.J_R_bg = torch.zeros(3, 3, device=device, dtype=dtype)
        self.J_v_bg = torch.zeros(3, 3, device=device, dtype=dtype)
        self.J_v_ba = torch.zeros(3, 3, device=device, dtype=dtype)
        self.J_p_bg = torch.zeros(3, 3, device=device, dtype=dtype)
        self.J_p_ba = torch.zeros(3, 3, device=device, dtype=dtype)
        self.dt_sum = 0.0
        self.num_measurements = 0

    def reset(self):
        """Reset the pre-integrator to identity state."""
        self.delta_R = torch.eye(3, device=self.device, dtype=self.dtype)
        self.delta_v = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.delta_p = torch.zeros(3, device=self.device, dtype=self.dtype)
        self.J_R_bg.zero_()
        self.J_v_bg.zero_()
        self.J_v_ba.zero_()
        self.J_p_bg.zero_()
        self.J_p_ba.zero_()
        self.dt_sum = 0.0
        self.num_measurements = 0

    def integrate(self, gyro: torch.Tensor, accel: torch.Tensor, dt: float):
        """
        Integrate a single IMU measurement.

        Args:
            gyro: Raw gyroscope omega^B_k (3,) in rad/s.
            accel: Raw accelerometer a^B_k (3,) in m/s^2.
            dt: Time step in seconds.
        """
        omega_corrected = gyro - self.bias_gyro
        accel_corrected = accel - self.bias_accel
        dR = so3_exp(omega_corrected * dt)
        acc_rotated = self.delta_R @ accel_corrected

        # Jacobian propagation for bias correction
        Jr = self._right_jacobian(omega_corrected * dt)
        acc_skew = skew_symmetric(acc_rotated.unsqueeze(0)).squeeze(0)
        self.J_R_bg = dR.T @ self.J_R_bg - Jr * dt
        self.J_v_bg = self.J_v_bg - acc_skew @ self.J_R_bg * dt
        self.J_v_ba = self.J_v_ba - self.delta_R * dt
        self.J_p_bg = self.J_p_bg + self.J_v_bg * dt - 0.5 * acc_skew @ self.J_R_bg * dt * dt
        self.J_p_ba = self.J_p_ba + self.J_v_ba * dt - 0.5 * self.delta_R * dt * dt

        # Pre-integration update (Eq. 3-5)
        self.delta_p = self.delta_p + self.delta_v * dt + 0.5 * acc_rotated * dt * dt
        self.delta_v = self.delta_v + acc_rotated * dt
        self.delta_R = self.delta_R @ dR
        self.dt_sum += dt
        self.num_measurements += 1

    def integrate_batch(self, gyro_batch: torch.Tensor, accel_batch: torch.Tensor, dt: float):
        """Integrate a batch of IMU measurements sequentially."""
        for i in range(gyro_batch.shape[0]):
            self.integrate(gyro_batch[i], accel_batch[i], dt)

    def get_kinematic_vector(self) -> torch.Tensor:
        """Return X_kinematic = [Log(dR)^T, dv^T, dp^T]^T in R^9."""
        log_R = so3_log(self.delta_R)  # (3,) - handles unbatched
        return torch.cat([log_R, self.delta_v, self.delta_p], dim=0)

    def correct_bias(self, delta_bg: torch.Tensor, delta_ba: torch.Tensor):
        """First-order bias correction using pre-computed Jacobians."""
        self.delta_R = self.delta_R @ so3_exp(self.J_R_bg @ delta_bg)
        self.delta_v = self.delta_v + self.J_v_bg @ delta_bg + self.J_v_ba @ delta_ba
        self.delta_p = self.delta_p + self.J_p_bg @ delta_bg + self.J_p_ba @ delta_ba
        self.bias_gyro = self.bias_gyro + delta_bg
        self.bias_accel = self.bias_accel + delta_ba

    @staticmethod
    def _right_jacobian(phi: torch.Tensor) -> torch.Tensor:
        """Right Jacobian of SO(3)."""
        theta = torch.norm(phi)
        if theta < 1e-6:
            return torch.eye(3, device=phi.device, dtype=phi.dtype)
        K = skew_symmetric(phi.unsqueeze(0)).squeeze(0) / theta
        I = torch.eye(3, device=phi.device, dtype=phi.dtype)
        return I - ((1 - torch.cos(theta)) / theta) * K + ((theta - torch.sin(theta)) / theta) * (K @ K)


def estimate_initial_bias(
    gyro_stationary: torch.Tensor,
    accel_stationary: torch.Tensor,
    gravity: float = 9.81,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Estimate initial IMU biases from stationary measurements.

    Args:
        gyro_stationary: Gyroscope readings at rest, shape (N, 3).
        accel_stationary: Accelerometer readings at rest, shape (N, 3).
        gravity: Gravity magnitude (default 9.81 m/s^2).

    Returns:
        Tuple of (bias_gyro, bias_accel), each shape (3,).
    """
    bias_gyro = gyro_stationary.mean(dim=0)
    accel_mean = accel_stationary.mean(dim=0)
    gravity_vec = torch.tensor([0.0, 0.0, gravity], device=accel_mean.device, dtype=accel_mean.dtype)
    bias_accel = accel_mean - gravity_vec
    return bias_gyro, bias_accel
