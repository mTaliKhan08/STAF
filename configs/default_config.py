"""
STAF Default Configuration.

Centralizes all hyperparameters and architecture choices for
reproducibility and easy ablation control.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class IMUConfig:
    """IMU pre-integration settings."""
    sample_rate_hz: float = 200.0
    stationary_init_frames: int = 200
    gravity: float = 9.81
    # Noise parameters (can be tuned per sensor)
    gyro_noise_density: float = 1.6968e-4  # rad/s/sqrt(Hz)
    accel_noise_density: float = 2.0e-3    # m/s^2/sqrt(Hz)
    gyro_random_walk: float = 1.9393e-5    # rad/s^2/sqrt(Hz)
    accel_random_walk: float = 3.0e-3      # m/s^3/sqrt(Hz)


@dataclass
class BackboneConfig:
    """Feature extraction backbone settings."""
    visual_backbone: str = "resnet34"
    lidar_backbone: str = "resnet18"
    lidar_in_channels: int = 2  # BEV channels (height + intensity)
    pretrained: bool = True


@dataclass
class SoftGatingConfig:
    """Soft-gating MLP settings (Fix #8 - fully specified)."""
    input_dim: int = 9
    hidden_dim: int = 64
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Gated cross-attention transformer settings."""
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    dropout: float = 0.1


@dataclass
class TemporalConfig:
    """Temporal memory queue settings."""
    queue_length: int = 5  # T frames in FIFO buffer
    num_layers: int = 2
    # Uses same embed_dim, num_heads, ff_dim from FusionConfig


@dataclass
class LossConfig:
    """Loss function settings."""
    gamma: float = 2.0     # Focal loss focusing parameter
    beta: float = 0.999    # Effective number of samples hyperparameter
    label_smoothing: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters (Fix #9 - grid search ranges documented)."""
    # Primary hyperparameters
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 8
    num_epochs: int = 50
    warmup_epochs: int = 5

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    gradient_clip_max_norm: float = 1.0
    use_amp: bool = True  # Mixed precision (AMP)

    # Grid search ranges (for reference)
    lr_search: List[float] = field(default_factory=lambda: [1e-4, 3e-4, 5e-4])
    gamma_search: List[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    queue_length_search: List[int] = field(default_factory=lambda: [3, 5, 8])


@dataclass
class DataConfig:
    """Dataset and data pipeline settings."""
    # Datasets
    nuscenes_root: str = ""
    carla_root: str = ""
    euroc_root: str = ""

    # Sensor rates
    camera_rate_hz: float = 20.0
    lidar_rate_hz: float = 10.0
    imu_rate_hz: float = 200.0

    # Image dimensions
    image_height: int = 256
    image_width: int = 256
    lidar_bev_height: int = 256
    lidar_bev_width: int = 256

    # Data cleaning thresholds (Fix #5)
    min_lidar_points: int = 100
    max_black_pixel_ratio: float = 0.3
    max_imu_gap_ms: float = 5.0

    # Class balancing
    num_classes: int = 16  # nuScenes semantic classes
    num_workers: int = 4


@dataclass
class EvalConfig:
    """Evaluation and B2B testing settings."""
    # B2B testing thresholds (Fix #12)
    b2b_kappa_threshold: float = 0.99
    b2b_max_fp_divergence: float = 1e-4

    # Gazebo scenarios (Fix #11 - 5 prioritized)
    gazebo_scenarios: List[str] = field(default_factory=lambda: [
        "high_speed_cornering",
        "rough_terrain_potholes",
        "tunnel_entry_exit",
        "heavy_rain_fog",
        "nominal_urban_driving",
    ])


@dataclass
class STAFConfig:
    """Top-level STAF configuration."""
    imu: IMUConfig = field(default_factory=IMUConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    soft_gating: SoftGatingConfig = field(default_factory=SoftGatingConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # Experiment tracking
    experiment_name: str = "staf_v1"
    seed: int = 42
    device: str = "cuda"
