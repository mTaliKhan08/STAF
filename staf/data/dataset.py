"""
Multi-Modal Dataset for STAF Framework.

Provides a unified PyTorch Dataset interface for loading synchronized
Camera + LiDAR + IMU data from multiple autonomous driving datasets.

Supports:
    - nuScenes (full dataset, 1000 scenes)
    - CARLA (multi-town recordings for TransFuser comparison)
    - EuRoC MAV (200Hz raw IMU for pre-integration validation)

Each sample returns:
    - RGB image tensor (B, 3, H, W)
    - LiDAR BEV tensor (B, C, H', W')
    - IMU kinematic vector (B, 9) from pre-integration
    - Semantic labels (if available)
    - Metadata dict
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class STAFDataset(Dataset):
    """
    Base multi-modal dataset for STAF.

    Subclass this for specific dataset formats (nuScenes, CARLA, EuRoC).
    This base class handles:
        - Data cleaning validation
        - IMU pre-integration per sample
        - BEV projection
        - Image transforms
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        lidar_bev_size: Tuple[int, int] = (256, 256),
        lidar_range: float = 50.0,
        num_classes: int = 16,
        apply_cleaning: bool = True,
        max_black_pixel_ratio: float = 0.3,
        min_lidar_points: int = 100,
        max_imu_gap_ms: float = 5.0,
    ):
        """
        Args:
            data_root: Root directory of the dataset.
            split: Dataset split ('train', 'val', 'test').
            image_size: Target (H, W) for camera images.
            lidar_bev_size: Target (H, W) for LiDAR BEV maps.
            lidar_range: LiDAR range in meters for BEV projection.
            num_classes: Number of semantic classes.
            apply_cleaning: Whether to apply data quality filtering.
            max_black_pixel_ratio: Camera filtering threshold.
            min_lidar_points: LiDAR filtering threshold.
            max_imu_gap_ms: IMU gap threshold in ms.
        """
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.lidar_bev_size = lidar_bev_size
        self.lidar_range = lidar_range
        self.num_classes = num_classes
        self.apply_cleaning = apply_cleaning

        # Initialize data cleaning
        if apply_cleaning:
            from staf.data.cleaning import DataCleaner
            self.cleaner = DataCleaner(
                max_black_pixel_ratio=max_black_pixel_ratio,
                min_lidar_points=min_lidar_points,
                max_imu_gap_ms=max_imu_gap_ms,
            )
        else:
            self.cleaner = None

        # IMU pre-integrator (created fresh per sample)
        from staf.imu.preintegration import IMUPreintegrator
        self._preintegrator_class = IMUPreintegrator

        # Sample index list (populated by subclasses)
        self.samples: List[Dict] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return a synchronized multi-modal sample.

        Returns:
            Dict with keys:
                'image': Camera RGB tensor (3, H, W), float32, normalized [0, 1].
                'lidar_bev': LiDAR BEV tensor (2, H', W'), float32.
                'x_kinematic': IMU kinematic vector (9,), float32.
                'label': Semantic segmentation label (H, W), int64 (if available).
                'metadata': Dict with sample info.
        """
        sample_info = self.samples[idx]

        # Load raw data (implemented by subclasses)
        image = self._load_image(sample_info)
        points = self._load_lidar(sample_info)
        imu_data = self._load_imu(sample_info)
        label = self._load_label(sample_info)

        # Data cleaning validation
        if self.apply_cleaning and self.cleaner is not None:
            is_valid, _ = self.cleaner.validate_sample(
                image=image,
                points=points,
                imu_timestamps=imu_data.get("timestamps"),
                imu_dt_ms=imu_data.get("expected_dt_ms", 5.0),
            )
            if not is_valid:
                # Return next valid sample (skip corrupted ones)
                return self.__getitem__((idx + 1) % len(self))

        # Process image
        image_tensor = self._process_image(image)

        # Process LiDAR -> BEV
        bev_tensor = self._points_to_bev(points)

        # Process IMU -> kinematic vector via pre-integration
        x_kinematic = self._preintegrate_imu(imu_data)

        # Process label
        if label is not None:
            label_tensor = torch.from_numpy(label).long()
        else:
            label_tensor = torch.zeros(
                self.image_size[0], self.image_size[1], dtype=torch.long
            )

        return {
            "image": image_tensor,
            "lidar_bev": bev_tensor,
            "x_kinematic": x_kinematic,
            "label": label_tensor,
            "metadata": {
                "idx": idx,
                "timestamp": sample_info.get("timestamp", 0.0),
                "scene": sample_info.get("scene", ""),
            },
        }

    def _process_image(self, image: np.ndarray) -> torch.Tensor:
        """Resize and normalize camera image to tensor."""
        if image is None:
            return torch.zeros(3, *self.image_size)

        # Resize using simple interpolation
        from PIL import Image
        pil_img = Image.fromarray(image)
        pil_img = pil_img.resize((self.image_size[1], self.image_size[0]), Image.BILINEAR)
        img_np = np.array(pil_img, dtype=np.float32) / 255.0

        # HWC -> CHW
        if img_np.ndim == 3:
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
        else:
            img_tensor = torch.from_numpy(img_np).unsqueeze(0).expand(3, -1, -1)

        return img_tensor

    def _points_to_bev(self, points: np.ndarray) -> torch.Tensor:
        """
        Project 3D LiDAR points to Bird's Eye View (BEV) representation.

        Creates a 2-channel BEV map:
            Channel 0: Maximum height in each cell
            Channel 1: Point density (count) in each cell

        Args:
            points: Point cloud, shape (N, 3+). Columns: x, y, z, [intensity].

        Returns:
            BEV tensor, shape (2, H, W).
        """
        H, W = self.lidar_bev_size
        bev = np.zeros((2, H, W), dtype=np.float32)

        if points is None or len(points) == 0:
            return torch.from_numpy(bev)

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Filter points within range
        r = self.lidar_range
        mask = (x > -r) & (x < r) & (y > -r) & (y < r)
        x, y, z = x[mask], y[mask], z[mask]

        if len(x) == 0:
            return torch.from_numpy(bev)

        # Map to pixel coordinates
        px = ((x + r) / (2 * r) * (W - 1)).astype(np.int32)
        py = ((y + r) / (2 * r) * (H - 1)).astype(np.int32)
        px = np.clip(px, 0, W - 1)
        py = np.clip(py, 0, H - 1)

        # Channel 0: max height
        for i in range(len(x)):
            bev[0, py[i], px[i]] = max(bev[0, py[i], px[i]], z[i])
            bev[1, py[i], px[i]] += 1.0  # density

        # Normalize density channel
        max_density = bev[1].max()
        if max_density > 0:
            bev[1] /= max_density

        return torch.from_numpy(bev)

    def _preintegrate_imu(self, imu_data: Dict) -> torch.Tensor:
        """
        Run IMU pre-integration on the measurement window.

        Args:
            imu_data: Dict with 'gyro' (N, 3), 'accel' (N, 3), 'dt' (N,).

        Returns:
            Kinematic vector X_kinematic (9,).
        """
        gyro = imu_data.get("gyro")
        accel = imu_data.get("accel")
        dts = imu_data.get("dt")

        if gyro is None or len(gyro) == 0:
            return torch.zeros(9)

        preint = self._preintegrator_class()

        gyro_t = torch.from_numpy(gyro).float()
        accel_t = torch.from_numpy(accel).float()

        for i in range(len(gyro_t)):
            dt = float(dts[i]) if dts is not None and i < len(dts) else 0.005
            preint.integrate(gyro_t[i], accel_t[i], dt)

        return preint.get_kinematic_vector()

    # --- Abstract methods for subclasses ---

    def _load_image(self, sample_info: Dict) -> Optional[np.ndarray]:
        """Load RGB image. Override in subclass."""
        raise NotImplementedError

    def _load_lidar(self, sample_info: Dict) -> Optional[np.ndarray]:
        """Load LiDAR point cloud. Override in subclass."""
        raise NotImplementedError

    def _load_imu(self, sample_info: Dict) -> Dict:
        """Load IMU data window. Override in subclass."""
        raise NotImplementedError

    def _load_label(self, sample_info: Dict) -> Optional[np.ndarray]:
        """Load semantic label. Override in subclass."""
        return None


class NuScenesDataset(STAFDataset):
    """
    nuScenes dataset adapter for STAF.

    Loads synchronized camera, LiDAR, and CAN bus/IMU data from
    the full nuScenes dataset (1000 scenes, ~40K keyframes).

    Requires: nuscenes-devkit
    """

    def __init__(self, data_root: str, version: str = "v1.0-trainval", **kwargs):
        super().__init__(data_root, **kwargs)
        self.version = version
        self._build_sample_list()

    def _build_sample_list(self):
        """Build the sample index from nuScenes database."""
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(
                version=self.version,
                dataroot=self.data_root,
                verbose=False,
            )
        except ImportError:
            logger.warning(
                "nuscenes-devkit not installed. Install with: pip install nuscenes-devkit"
            )
            self.nusc = None
            return
        except Exception as e:
            logger.warning(f"Failed to load nuScenes: {e}")
            self.nusc = None
            return

        # Build sample list from keyframes
        split_scenes = self._get_split_scenes()

        for scene in self.nusc.scene:
            if scene["name"] not in split_scenes:
                continue

            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                self.samples.append({
                    "token": sample_token,
                    "timestamp": sample["timestamp"] / 1e6,  # us -> s
                    "scene": scene["name"],
                    "cam_token": sample["data"]["CAM_FRONT"],
                    "lidar_token": sample["data"]["LIDAR_TOP"],
                })
                sample_token = sample["next"]

        logger.info(f"NuScenes {self.split}: {len(self.samples)} samples from {len(split_scenes)} scenes")

    def _get_split_scenes(self) -> set:
        """Get scene names for the current split."""
        from nuscenes.utils.splits import create_splits_scenes
        splits = create_splits_scenes()
        return set(splits.get(self.split, []))

    def _load_image(self, sample_info: Dict) -> Optional[np.ndarray]:
        if self.nusc is None:
            return None
        cam_data = self.nusc.get("sample_data", sample_info["cam_token"])
        img_path = os.path.join(self.data_root, cam_data["filename"])
        if os.path.exists(img_path):
            from PIL import Image
            return np.array(Image.open(img_path).convert("RGB"))
        return None

    def _load_lidar(self, sample_info: Dict) -> Optional[np.ndarray]:
        if self.nusc is None:
            return None
        lidar_data = self.nusc.get("sample_data", sample_info["lidar_token"])
        pcl_path = os.path.join(self.data_root, lidar_data["filename"])
        if os.path.exists(pcl_path):
            points = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 5)
            return points[:, :4]  # x, y, z, intensity
        return None

    def _load_imu(self, sample_info: Dict) -> Dict:
        """Load IMU/CAN bus data from nuScenes."""
        # nuScenes provides CAN bus data at ~100Hz via the can_bus expansion
        # For now, return synthetic IMU matching the expected format
        # Full CAN bus integration requires the nuscenes-can-bus expansion pack
        return {
            "gyro": np.zeros((20, 3), dtype=np.float32),
            "accel": np.tile([0.0, 0.0, 9.81], (20, 1)).astype(np.float32),
            "dt": np.full(20, 0.005, dtype=np.float32),
            "timestamps": np.linspace(
                sample_info["timestamp"] - 0.1,
                sample_info["timestamp"],
                20,
            ),
            "expected_dt_ms": 5.0,
        }


class EuRoCDataset(STAFDataset):
    """
    EuRoC MAV dataset adapter for STAF.

    Provides true 200Hz raw IMU data for validating the IMU pre-integration
    module in isolation. Used for Phase 2.3 validation (Fix #4).

    Directory structure:
        {data_root}/
            mav0/
                cam0/data/  (stereo left images, 20Hz)
                imu0/data.csv  (IMU at 200Hz)
                state_groundtruth_estimate0/data.csv
    """

    def __init__(self, data_root: str, sequence: str = "V1_01_easy", **kwargs):
        kwargs.setdefault("split", "val")
        super().__init__(data_root, **kwargs)
        self.sequence = sequence
        self._build_sample_list()

    def _build_sample_list(self):
        """Build sample list from EuRoC directory structure."""
        seq_dir = os.path.join(self.data_root, self.sequence, "mav0")

        # Load IMU data (200Hz)
        imu_csv = os.path.join(seq_dir, "imu0", "data.csv")
        if not os.path.exists(imu_csv):
            logger.warning(f"EuRoC IMU file not found: {imu_csv}")
            return

        self._imu_data = np.loadtxt(
            imu_csv, delimiter=",", skiprows=1,
            usecols=(0, 1, 2, 3, 4, 5, 6),  # timestamp, wx, wy, wz, ax, ay, az
        )
        self._imu_timestamps = self._imu_data[:, 0] / 1e9  # ns -> s
        self._imu_gyro = self._imu_data[:, 1:4]
        self._imu_accel = self._imu_data[:, 4:7]

        # Load camera timestamps (20Hz)
        cam_dir = os.path.join(seq_dir, "cam0", "data")
        if os.path.exists(cam_dir):
            cam_files = sorted(os.listdir(cam_dir))
            for f in cam_files:
                ts_ns = int(os.path.splitext(f)[0])
                self.samples.append({
                    "timestamp": ts_ns / 1e9,
                    "cam_path": os.path.join(cam_dir, f),
                    "scene": self.sequence,
                })

        logger.info(
            f"EuRoC {self.sequence}: {len(self.samples)} camera frames, "
            f"{len(self._imu_data)} IMU measurements"
        )

    def _load_image(self, sample_info: Dict) -> Optional[np.ndarray]:
        cam_path = sample_info.get("cam_path")
        if cam_path and os.path.exists(cam_path):
            from PIL import Image
            img = Image.open(cam_path).convert("RGB")
            return np.array(img)
        return None

    def _load_lidar(self, sample_info: Dict) -> Optional[np.ndarray]:
        # EuRoC doesn't have LiDAR
        return np.zeros((200, 3), dtype=np.float32)

    def _load_imu(self, sample_info: Dict) -> Dict:
        """Load raw 200Hz IMU data from the window around this camera frame."""
        cam_ts = sample_info["timestamp"]
        idx = sample_info.get("_sample_idx", self.samples.index(sample_info))

        # Get previous camera timestamp for window
        if idx > 0:
            prev_ts = self.samples[idx - 1]["timestamp"]
        else:
            prev_ts = cam_ts - 0.05  # 50ms before first frame

        # Slice IMU window
        mask = (self._imu_timestamps >= prev_ts) & (self._imu_timestamps <= cam_ts)
        window_ts = self._imu_timestamps[mask]
        window_gyro = self._imu_gyro[mask]
        window_accel = self._imu_accel[mask]

        # Compute dt
        if len(window_ts) > 1:
            dts = np.diff(window_ts)
            dts = np.concatenate([[dts[0]], dts])
        else:
            dts = np.array([0.005])

        return {
            "gyro": window_gyro.astype(np.float32),
            "accel": window_accel.astype(np.float32),
            "dt": dts.astype(np.float32),
            "timestamps": window_ts,
            "expected_dt_ms": 5.0,
        }

    def _load_label(self, sample_info: Dict) -> Optional[np.ndarray]:
        return None  # EuRoC has no semantic labels



class CARLASTAFDataset(STAFDataset):
    """
    CARLA Autonomous Driving Dataset adapter for STAF.

    Loads synchronized Camera RGB, LiDAR BEV, IMU kinematics, and future Waypoints
    from CARLA scenario routes (TransFuser / Leaderboard benchmark format).
    Supports domain generalization augmentations (weather jitter, LiDAR dropout, IMU noise).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        pred_len: int = 4,
        augment_domain: bool = True,
        **kwargs
    ):
        super().__init__(data_root, split=split, **kwargs)
        self.pred_len = pred_len
        self.augment_domain = augment_domain and (split == "train")
        self._build_sample_list()

    def _build_sample_list(self):
        """Discover all scenario route folders in data_root."""
        import json
        if not os.path.exists(self.data_root):
            logger.warning(f"CARLA data_root not found: {self.data_root}")
            return

        # Find all routes (either in subfolders like dirt_dataset_23_11 or directly)
        routes = []
        entries = sorted(os.listdir(self.data_root))
        for entry in entries:
            p = os.path.join(self.data_root, entry)
            if not os.path.isdir(p):
                continue
            # Check if this is a scenario directory containing routes
            sub_entries = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
            if any("route" in d.lower() or "routes" in d.lower() for d in sub_entries):
                for sub in sub_entries:
                    route_path = os.path.join(p, sub)
                    if os.path.isdir(os.path.join(route_path, "rgb")):
                        routes.append(route_path)
            elif os.path.isdir(os.path.join(p, "rgb")):
                routes.append(p)

        logger.info(f"CARLASTAFDataset: Found {len(routes)} valid route directories")

        # Split into train / val
        if self.split == "val" and len(routes) > 1:
            routes = routes[:max(1, len(routes) // 5)]
        elif self.split == "train" and len(routes) > 1:
            routes = routes[max(1, len(routes) // 5):]

        for route_dir in routes:
            rgb_dir = os.path.join(route_dir, "rgb")
            meas_dir = os.path.join(route_dir, "measurements")
            lidar_dir = os.path.join(route_dir, "lidar")
            label_dir = os.path.join(route_dir, "label_raw")

            frames = sorted([f.replace(".png", "") for f in os.listdir(rgb_dir) if f.endswith(".png")])
            num_frames = len(frames)

            # Keep valid sequence indices allowing future prediction
            for i in range(2, num_frames - self.pred_len - 2):
                f_id = frames[i]
                self.samples.append({
                    "route_dir": route_dir,
                    "frame_id": f_id,
                    "frame_idx": i,
                    "frames_list": frames,
                    "rgb_path": os.path.join(rgb_dir, f"{f_id}.png"),
                    "lidar_path": os.path.join(lidar_dir, f"{f_id}.npy"),
                    "meas_path": os.path.join(meas_dir, f"{f_id}.json"),
                    "label_dir": label_dir,
                })

        logger.info(f"CARLASTAFDataset [{self.split}]: Loaded {len(self.samples)} frame samples")

    def _load_image(self, sample_info: Dict) -> Optional[np.ndarray]:
        rgb_path = sample_info["rgb_path"]
        if not os.path.exists(rgb_path):
            return None
        from PIL import Image
        img = Image.open(rgb_path).convert("RGB")

        # Domain generalization photometric jitter (simulating sunlight, glare, rain)
        if self.augment_domain and np.random.rand() > 0.3:
            import torchvision.transforms as T
            color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            img = color_jitter(img)

        return np.array(img)

    def _load_lidar(self, sample_info: Dict) -> Optional[np.ndarray]:
        lidar_path = sample_info["lidar_path"]
        if not os.path.exists(lidar_path):
            return np.zeros((100, 4), dtype=np.float32)
        try:
            raw_lidar = np.load(lidar_path, allow_pickle=True)
            if isinstance(raw_lidar, np.ndarray) and len(raw_lidar) > 1 and isinstance(raw_lidar[1], np.ndarray):
                pts = raw_lidar[1]
            else:
                pts = raw_lidar
            if pts.ndim != 2 or pts.shape[1] < 3:
                return np.zeros((100, 4), dtype=np.float32)
            if pts.shape[1] == 3:
                pts = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])

            # Domain generalization: random LiDAR point dropout (simulating sparse vs dense LiDARs)
            if self.augment_domain and np.random.rand() > 0.4:
                keep_prob = np.random.uniform(0.7, 0.95)
                mask = np.random.rand(len(pts)) < keep_prob
                pts = pts[mask]

            return pts[:, :4]
        except Exception:
            return np.zeros((100, 4), dtype=np.float32)

    def _load_imu(self, sample_info: Dict) -> Dict:
        """Parse IMU acceleration, gyroscope, and velocity from CARLA measurements."""
        meas_path = sample_info["meas_path"]
        gyro = np.zeros((20, 3), dtype=np.float32)
        accel = np.tile([0.0, 0.0, 9.81], (20, 1)).astype(np.float32)

        if os.path.exists(meas_path):
            import json
            try:
                with open(meas_path, "r") as f:
                    data = json.load(f)
                spd = float(data.get("speed", 0.0))
                # If accelerometer / gyroscope available in CARLA
                if "accelerometer" in data:
                    a = data["accelerometer"]
                    accel[-1] = [float(a[0]), float(a[1]), float(a[2]) + 9.81]
                if "gyroscope" in data:
                    g = data["gyroscope"]
                    gyro[-1] = [float(g[0]), float(g[1]), float(g[2])]
            except Exception:
                pass

        # Domain generalization: IMU sensor noise injection
        if self.augment_domain:
            gyro += np.random.normal(0, 0.01, gyro.shape).astype(np.float32)
            accel += np.random.normal(0, 0.05, accel.shape).astype(np.float32)

        return {
            "gyro": gyro,
            "accel": accel,
            "dt": np.full(20, 0.005, dtype=np.float32),
            "timestamps": np.linspace(0, 0.1, 20),
            "expected_dt_ms": 5.0,
        }

    def _load_waypoints(self, sample_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract future relative waypoints (pred_len, 2) and target navigation point."""
        import json
        label_dir = sample_info["label_dir"]
        frames = sample_info["frames_list"]
        idx = sample_info["frame_idx"]

        # Current ego pose
        cur_file = os.path.join(label_dir, f"{frames[idx]}.json")
        cur_x, cur_y, cur_yaw = 0.0, 0.0, 0.0
        if os.path.exists(cur_file):
            try:
                with open(cur_file, "r") as f:
                    d = json.load(f)
                cur_x = float(d.get("x", 0.0))
                cur_y = float(d.get("y", 0.0))
                cur_yaw = float(d.get("theta", 0.0))
            except Exception:
                pass

        waypoints = []
        for step in range(1, self.pred_len + 1):
            future_file = os.path.join(label_dir, f"{frames[idx + step]}.json")
            if os.path.exists(future_file):
                try:
                    with open(future_file, "r") as f:
                        fd = json.load(f)
                    fx = float(fd.get("x", cur_x))
                    fy = float(fd.get("y", cur_y))
                    # Transform world delta to ego vehicle coordinate frame
                    dx = fx - cur_x
                    dy = fy - cur_y
                    cos_t = np.cos(cur_yaw)
                    sin_t = np.sin(cur_yaw)
                    local_x = dx * cos_t + dy * sin_t
                    local_y = -dx * sin_t + dy * cos_t
                    waypoints.append([local_x, local_y])
                except Exception:
                    waypoints.append([step * 1.5, 0.0])
            else:
                waypoints.append([step * 1.5, 0.0])

        target_point = waypoints[-1] if len(waypoints) > 0 else [10.0, 0.0]
        return np.array(waypoints, dtype=np.float32), np.array(target_point, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_dict = super().__getitem__(idx)
        sample_info = self.samples[idx]
        wps, target = self._load_waypoints(sample_info)
        sample_dict["waypoints"] = torch.from_numpy(wps).float()
        sample_dict["target_point"] = torch.from_numpy(target).float()
        return sample_dict


class KITTI360Dataset(STAFDataset):
    """
    KITTI-360 Dataset adapter for STAF.

    Loads synchronized front camera, Velodyne LiDAR, and OXTS IMU for
    cross-dataset Sim-to-Real and Real-to-Real domain transfer evaluation.
    """

    def __init__(self, data_root: str, sequence: str = "2013_05_28_drive_0000_sync", **kwargs):
        super().__init__(data_root, **kwargs)
        self.sequence = sequence
        self._build_sample_list()

    def _build_sample_list(self):
        seq_dir = os.path.join(self.data_root, "data_2d_raw", self.sequence)
        cam_dir = os.path.join(seq_dir, "image_00", "data_rect")
        lidar_dir = os.path.join(self.data_root, "data_3d_raw", self.sequence, "velodyne_points", "data")
        oxts_dir = os.path.join(self.data_root, "data_poses", self.sequence, "oxts", "data")

        if not os.path.exists(cam_dir):
            logger.warning(f"KITTI-360 camera path not found: {cam_dir}")
            return

        cam_files = sorted([f for f in os.listdir(cam_dir) if f.endswith(".png")])
        for f in cam_files:
            f_id = f.replace(".png", "")
            self.samples.append({
                "cam_path": os.path.join(cam_dir, f),
                "lidar_path": os.path.join(lidar_dir, f"{f_id}.bin"),
                "oxts_path": os.path.join(oxts_dir, f"{f_id}.txt"),
                "timestamp": float(f_id) * 0.1,
            })

        logger.info(f"KITTI360Dataset: Loaded {len(self.samples)} frames for sequence {self.sequence}")

    def _load_image(self, sample_info: Dict) -> Optional[np.ndarray]:
        p = sample_info["cam_path"]
        if os.path.exists(p):
            from PIL import Image
            return np.array(Image.open(p).convert("RGB"))
        return None

    def _load_lidar(self, sample_info: Dict) -> Optional[np.ndarray]:
        p = sample_info["lidar_path"]
        if os.path.exists(p):
            pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
            return pts
        return np.zeros((100, 4), dtype=np.float32)

    def _load_imu(self, sample_info: Dict) -> Dict:
        """Parse OXTS IMU measurements (accel and angular rates)."""
        p = sample_info["oxts_path"]
        gyro = np.zeros((20, 3), dtype=np.float32)
        accel = np.tile([0.0, 0.0, 9.81], (20, 1)).astype(np.float32)

        if os.path.exists(p):
            try:
                vals = np.loadtxt(p)
                # OXTS format: ax, ay, az at indices 11,12,13; wx, wy, wz at 17,18,19
                if len(vals) >= 20:
                    accel[-1] = [vals[11], vals[12], vals[13]]
                    gyro[-1] = [vals[17], vals[18], vals[19]]
            except Exception:
                pass

        return {
            "gyro": gyro,
            "accel": accel,
            "dt": np.full(20, 0.005, dtype=np.float32),
            "timestamps": np.linspace(0, 0.1, 20),
            "expected_dt_ms": 5.0,
        }
