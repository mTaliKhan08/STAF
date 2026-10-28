"""
Asynchronous Spatio-Temporal Synchronization for STAF Framework.

Synchronizes multi-rate sensor inputs:
    - Camera @ 20 Hz
    - LiDAR  @ 10 Hz
    - IMU    @ 100-200 Hz

Uses nearest-timestamp matching for Camera-LiDAR alignment and
collects all IMU measurements within each inter-frame window for
pre-integration.

This module is the entry point for Phase 1.5 of the implementation plan.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import bisect
import logging

logger = logging.getLogger(__name__)


@dataclass
class SyncedSample:
    """A single time-synchronized multi-modal sample."""
    timestamp: float  # Reference timestamp (camera)
    camera_idx: int  # Index into camera data array
    lidar_idx: int  # Index into LiDAR data array (-1 if no match)
    imu_start_idx: int  # Start index of IMU window (inclusive)
    imu_end_idx: int  # End index of IMU window (exclusive)
    time_offset_lidar_ms: float  # Temporal offset between camera and matched LiDAR
    num_imu_samples: int  # Number of IMU samples in this window


class SensorSynchronizer:
    """
    Multi-rate sensor synchronization engine.

    Aligns camera, LiDAR, and IMU streams using nearest-timestamp matching
    and sliding-window IMU collection for pre-integration.
    """

    def __init__(
        self,
        max_lidar_offset_ms: float = 50.0,
        imu_window_pad_ms: float = 2.0,
    ):
        """
        Args:
            max_lidar_offset_ms: Maximum acceptable temporal offset between
                camera and LiDAR timestamps for a valid match (ms).
            imu_window_pad_ms: Extra padding on each side of the IMU window
                to ensure we capture all measurements in the interval.
        """
        self.max_lidar_offset_ms = max_lidar_offset_ms
        self.imu_window_pad_ms = imu_window_pad_ms

    def synchronize(
        self,
        camera_timestamps: np.ndarray,
        lidar_timestamps: np.ndarray,
        imu_timestamps: np.ndarray,
    ) -> List[SyncedSample]:
        """
        Synchronize all three sensor streams.

        The camera stream is the reference clock. For each camera frame:
        1. Find the nearest LiDAR sweep (within max_lidar_offset_ms).
        2. Collect all IMU measurements between previous and current camera frame.

        Args:
            camera_timestamps: Camera frame timestamps in seconds, shape (N_cam,).
            lidar_timestamps: LiDAR sweep timestamps in seconds, shape (N_lidar,).
            imu_timestamps: IMU measurement timestamps in seconds, shape (N_imu,).

        Returns:
            List of SyncedSample objects, one per camera frame.
        """
        synced_samples = []
        lidar_ts_sorted = np.sort(lidar_timestamps)
        imu_ts_sorted = np.sort(imu_timestamps)

        for i, cam_ts in enumerate(camera_timestamps):
            # --- Match nearest LiDAR sweep ---
            lidar_idx, lidar_offset_ms = self._find_nearest(
                cam_ts, lidar_ts_sorted, lidar_timestamps
            )

            # Reject if offset exceeds threshold
            if lidar_offset_ms > self.max_lidar_offset_ms:
                lidar_idx = -1
                lidar_offset_ms = float("inf")

            # --- Collect IMU window ---
            if i == 0:
                # First frame: collect IMU from start to current camera time
                prev_cam_ts = cam_ts - 0.1  # 100ms before first frame
            else:
                prev_cam_ts = camera_timestamps[i - 1]

            pad_s = self.imu_window_pad_ms / 1000.0
            imu_start = bisect.bisect_left(imu_ts_sorted, prev_cam_ts - pad_s)
            imu_end = bisect.bisect_right(imu_ts_sorted, cam_ts + pad_s)

            sample = SyncedSample(
                timestamp=cam_ts,
                camera_idx=i,
                lidar_idx=lidar_idx,
                imu_start_idx=imu_start,
                imu_end_idx=imu_end,
                time_offset_lidar_ms=lidar_offset_ms,
                num_imu_samples=imu_end - imu_start,
            )
            synced_samples.append(sample)

        # Log statistics
        valid_lidar = sum(1 for s in synced_samples if s.lidar_idx >= 0)
        avg_imu = np.mean([s.num_imu_samples for s in synced_samples]) if synced_samples else 0
        logger.info(
            f"Synchronized {len(synced_samples)} samples: "
            f"{valid_lidar}/{len(synced_samples)} with LiDAR match, "
            f"avg {avg_imu:.1f} IMU samples/window"
        )

        return synced_samples

    @staticmethod
    def _find_nearest(
        target_ts: float,
        sorted_timestamps: np.ndarray,
        original_timestamps: np.ndarray,
    ) -> Tuple[int, float]:
        """
        Find the nearest timestamp in sorted array and return its original index.

        Returns:
            Tuple of (original_index, offset_ms).
        """
        if len(sorted_timestamps) == 0:
            return -1, float("inf")

        idx = bisect.bisect_left(sorted_timestamps, target_ts)

        # Check both neighbors
        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(sorted_timestamps):
            candidates.append(idx)

        best_idx = min(candidates, key=lambda i: abs(sorted_timestamps[i] - target_ts))
        best_ts = sorted_timestamps[best_idx]
        offset_ms = abs(best_ts - target_ts) * 1000.0

        # Map back to original index
        original_idx = int(np.argmin(np.abs(original_timestamps - best_ts)))

        return original_idx, offset_ms


class IMUWindowExtractor:
    """
    Extracts and prepares IMU measurement windows for pre-integration.

    Given a SyncedSample, this class slices the raw IMU data arrays
    and computes per-step time deltas for the IMUPreintegrator.
    """

    def __init__(self, gravity: float = 9.81):
        self.gravity = gravity

    def extract_window(
        self,
        sample: SyncedSample,
        imu_timestamps: np.ndarray,
        imu_gyro: np.ndarray,
        imu_accel: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extract IMU measurements for a synced sample's window.

        Args:
            sample: SyncedSample with imu_start_idx and imu_end_idx.
            imu_timestamps: All IMU timestamps, shape (N_imu,).
            imu_gyro: All gyroscope measurements, shape (N_imu, 3).
            imu_accel: All accelerometer measurements, shape (N_imu, 3).

        Returns:
            Dict with keys:
                'gyro': Gyroscope data for this window, shape (W, 3).
                'accel': Accelerometer data for this window, shape (W, 3).
                'dt': Per-step time deltas, shape (W,).
                'timestamps': Timestamps for this window, shape (W,).
        """
        s, e = sample.imu_start_idx, sample.imu_end_idx

        if s >= e or s >= len(imu_timestamps):
            return {
                "gyro": np.zeros((0, 3)),
                "accel": np.zeros((0, 3)),
                "dt": np.zeros(0),
                "timestamps": np.zeros(0),
            }

        window_ts = imu_timestamps[s:e]
        window_gyro = imu_gyro[s:e]
        window_accel = imu_accel[s:e]

        # Compute per-step dt
        if len(window_ts) > 1:
            dts = np.diff(window_ts)
            # Prepend the first dt (assume uniform for the first step)
            dts = np.concatenate([[dts[0]], dts])
        else:
            dts = np.array([0.005])  # Default 200Hz

        return {
            "gyro": window_gyro,
            "accel": window_accel,
            "dt": dts,
            "timestamps": window_ts,
        }


def compute_class_frequencies(
    labels_list: List[np.ndarray],
    num_classes: int,
) -> np.ndarray:
    """
    Compute per-class sample frequencies across the entire training set.

    Used for Class-Balanced Focal Loss weight computation.

    Args:
        labels_list: List of label arrays (each can be any shape).
        num_classes: Total number of semantic classes.

    Returns:
        Per-class counts, shape (num_classes,).
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for labels in labels_list:
        for c in range(num_classes):
            counts[c] += np.sum(labels == c)
    return counts
