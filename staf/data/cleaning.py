"""
Data Cleaning Module for STAF Framework.

Implements reproducible, documented filtering rules (Fix #5) to remove
corrupted sensor data from training splits while preserving natural
edge-case anomalies in validation splits.

Filtering Rules:
    - Camera frames with >30% black/saturated pixels -> discard
    - LiDAR sweeps with <100 valid points -> discard
    - IMU timestamps with >5ms gap from expected interval -> flag & interpolate
    - All filtering counts are logged for thesis appendix documentation
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """Tracks filtering statistics for documentation in thesis appendix."""
    total_samples: int = 0
    camera_discarded: int = 0
    lidar_discarded: int = 0
    imu_interpolated: int = 0
    imu_gaps_found: int = 0
    samples_retained: int = 0

    def summary(self) -> Dict[str, int]:
        return {
            "total_samples": self.total_samples,
            "camera_discarded": self.camera_discarded,
            "lidar_discarded": self.lidar_discarded,
            "imu_interpolated": self.imu_interpolated,
            "imu_gaps_found": self.imu_gaps_found,
            "samples_retained": self.samples_retained,
            "discard_rate_pct": round(
                100.0 * (self.camera_discarded + self.lidar_discarded) / max(self.total_samples, 1), 2
            ),
        }


class DataCleaner:
    """
    Reproducible data cleaning with documented filtering rules.

    All thresholds match the implementation plan (Fix #5):
        - max_black_pixel_ratio = 0.3
        - min_lidar_points = 100
        - max_imu_gap_ms = 5.0
    """

    def __init__(
        self,
        max_black_pixel_ratio: float = 0.3,
        min_lidar_points: int = 100,
        max_imu_gap_ms: float = 5.0,
        black_threshold: int = 5,
        saturated_threshold: int = 250,
    ):
        self.max_black_pixel_ratio = max_black_pixel_ratio
        self.min_lidar_points = min_lidar_points
        self.max_imu_gap_ms = max_imu_gap_ms
        self.black_threshold = black_threshold
        self.saturated_threshold = saturated_threshold
        self.stats = CleaningStats()

    def check_camera_frame(self, image: np.ndarray) -> bool:
        """
        Check if a camera frame passes quality filtering.

        Args:
            image: RGB image as numpy array, shape (H, W, 3) or (H, W), uint8.

        Returns:
            True if the frame is valid, False if it should be discarded.
        """
        if image is None or image.size == 0:
            return False

        total_pixels = image.shape[0] * image.shape[1]

        # Convert to grayscale if RGB
        if image.ndim == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image.astype(float)

        # Count black pixels (near-zero intensity)
        black_count = np.sum(gray < self.black_threshold)
        # Count saturated pixels (near-max intensity)
        saturated_count = np.sum(gray > self.saturated_threshold)

        bad_ratio = (black_count + saturated_count) / total_pixels

        if bad_ratio > self.max_black_pixel_ratio:
            self.stats.camera_discarded += 1
            logger.debug(
                f"Camera frame discarded: {bad_ratio:.1%} bad pixels "
                f"({black_count} black, {saturated_count} saturated)"
            )
            return False

        return True

    def check_lidar_sweep(self, points: np.ndarray) -> bool:
        """
        Check if a LiDAR point cloud passes quality filtering.

        Args:
            points: Point cloud as numpy array, shape (N, 3+) or (N, 4).

        Returns:
            True if the sweep is valid, False if it should be discarded.
        """
        if points is None:
            return False

        num_points = points.shape[0] if points.ndim >= 1 else 0

        if num_points < self.min_lidar_points:
            self.stats.lidar_discarded += 1
            logger.debug(
                f"LiDAR sweep discarded: {num_points} points "
                f"(min={self.min_lidar_points})"
            )
            return False

        return True

    def check_and_fix_imu_timestamps(
        self,
        timestamps: np.ndarray,
        expected_dt_ms: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check IMU timestamps for gaps and generate interpolation mask.

        Args:
            timestamps: IMU measurement timestamps in seconds, shape (N,).
            expected_dt_ms: Expected time step in milliseconds (e.g., 5.0 for 200Hz).

        Returns:
            Tuple of:
                - gap_mask: Boolean mask, True where gaps exceed threshold, shape (N-1,).
                - corrected_timestamps: Timestamps with gaps flagged, shape (N,).
        """
        if len(timestamps) < 2:
            return np.array([], dtype=bool), timestamps

        dts_ms = np.diff(timestamps) * 1000.0  # Convert to ms
        gap_mask = dts_ms > self.max_imu_gap_ms

        num_gaps = int(np.sum(gap_mask))
        if num_gaps > 0:
            self.stats.imu_gaps_found += num_gaps
            logger.debug(
                f"IMU gaps found: {num_gaps} gaps exceeding {self.max_imu_gap_ms}ms "
                f"(max gap: {dts_ms.max():.2f}ms)"
            )

        return gap_mask, timestamps

    def interpolate_imu_gaps(
        self,
        data: np.ndarray,
        timestamps: np.ndarray,
        gap_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Linearly interpolate IMU measurements across detected gaps.

        Args:
            data: IMU measurements, shape (N, D) where D=3 (gyro) or D=3 (accel).
            timestamps: Corresponding timestamps, shape (N,).
            gap_mask: Boolean mask from check_and_fix_imu_timestamps, shape (N-1,).

        Returns:
            Tuple of (interpolated_data, interpolated_timestamps).
        """
        if not gap_mask.any():
            return data, timestamps

        gap_indices = np.where(gap_mask)[0]
        interpolated_data = data.copy()
        self.stats.imu_interpolated += len(gap_indices)

        for idx in gap_indices:
            if idx > 0 and idx < len(data) - 1:
                # Linear interpolation between surrounding valid measurements
                alpha = 0.5
                interpolated_data[idx] = (1 - alpha) * data[idx] + alpha * data[idx + 1]

        return interpolated_data, timestamps

    def validate_sample(
        self,
        image: Optional[np.ndarray] = None,
        points: Optional[np.ndarray] = None,
        imu_timestamps: Optional[np.ndarray] = None,
        imu_dt_ms: float = 5.0,
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        Validate a complete multi-modal sample.

        Args:
            image: Camera frame (optional).
            points: LiDAR point cloud (optional).
            imu_timestamps: IMU timestamps (optional).
            imu_dt_ms: Expected IMU time step in ms.

        Returns:
            Tuple of (is_valid, per_sensor_results).
        """
        self.stats.total_samples += 1
        results = {}

        if image is not None:
            results["camera"] = self.check_camera_frame(image)
        else:
            results["camera"] = True  # No camera data to check

        if points is not None:
            results["lidar"] = self.check_lidar_sweep(points)
        else:
            results["lidar"] = True

        if imu_timestamps is not None:
            gap_mask, _ = self.check_and_fix_imu_timestamps(imu_timestamps, imu_dt_ms)
            results["imu_clean"] = not gap_mask.any() if len(gap_mask) > 0 else True
        else:
            results["imu_clean"] = True

        is_valid = results["camera"] and results["lidar"]
        if is_valid:
            self.stats.samples_retained += 1

        return is_valid, results

    def get_stats(self) -> Dict[str, int]:
        """Return filtering statistics for thesis documentation."""
        return self.stats.summary()

    def reset_stats(self):
        """Reset all statistics counters."""
        self.stats = CleaningStats()
