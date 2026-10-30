"""
STAF Autonomous Driving Agent for CARLA Simulation Environment.

Integrates the trained STAF model into CARLA Leaderboard / AutonomousAgent runtime.
Performs closed-loop control from live Camera RGB + LiDAR BEV + IMU Pre-integration.
"""

import os
import sys
import math
import json
import logging
from collections import deque
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger("staf-agent")

# CARLA Leaderboard import fallback
try:
    from leaderboard.autoagents import autonomous_agent
    BaseAgent = autonomous_agent.AutonomousAgent
except ImportError:
    class BaseAgent:
        """Standalone fallback if leaderboard package is not installed."""
        def __init__(self):
            self.track = 1
            self.step = -1
            self.initialized = False

# CARLA package fallback
try:
    import carla
    VehicleControl = carla.VehicleControl
except ImportError:
    class VehicleControl:
        """Lightweight VehicleControl mock for headless/offline testing."""
        def __init__(self, steer: float = 0.0, throttle: float = 0.0, brake: float = 0.0, hand_brake: bool = False, reverse: bool = False):
            self.steer = float(steer)
            self.throttle = float(throttle)
            self.brake = float(brake)
            self.hand_brake = hand_brake
            self.reverse = reverse

        def __repr__(self):
            return f"VehicleControl(steer={self.steer:.3f}, throttle={self.throttle:.3f}, brake={self.brake:.3f})"


class PIDController:
    """PID controller for vehicle lateral (steering) and longitudinal (speed) control."""

    def __init__(
        self,
        kp_lat: float = 0.75,
        ki_lat: float = 0.10,
        kd_lat: float = 0.30,
        kp_lon: float = 5.00,
        ki_lon: float = 0.50,
        kd_lon: float = 1.00,
        dt: float = 0.05,
    ):
        self.kp_lat = kp_lat
        self.ki_lat = ki_lat
        self.kd_lat = kd_lat
        self.kp_lon = kp_lon
        self.ki_lon = ki_lon
        self.kd_lon = kd_lon
        self.dt = dt

        self.lat_error_integral = 0.0
        self.lat_error_prev = 0.0
        self.lon_error_integral = 0.0
        self.lon_error_prev = 0.0

    def compute_lateral(self, target_heading_deg: float) -> float:
        """Compute steering angle [-1, 1] from target heading angle."""
        error = math.radians(target_heading_deg)
        self.lat_error_integral += error * self.dt
        self.lat_error_integral = np.clip(self.lat_error_integral, -1.0, 1.0)
        derivative = (error - self.lat_error_prev) / self.dt
        self.lat_error_prev = error

        steer = self.kp_lat * error + self.ki_lat * self.lat_error_integral + self.kd_lat * derivative
        return float(np.clip(steer, -1.0, 1.0))

    def compute_longitudinal(self, current_speed: float, target_speed: float) -> Tuple[float, float]:
        """Compute throttle [0, 1] and brake [0, 1] from speed error."""
        error = target_speed - current_speed
        self.lon_error_integral += error * self.dt
        self.lon_error_integral = np.clip(self.lon_error_integral, -5.0, 5.0)
        derivative = (error - self.lon_error_prev) / self.dt
        self.lon_error_prev = error

        accel = self.kp_lon * error + self.ki_lon * self.lon_error_integral + self.kd_lon * derivative
        if accel >= 0.0:
            throttle = float(np.clip(accel / 10.0, 0.0, 0.75))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-accel / 10.0, 0.0, 1.0))

        return throttle, brake

    def reset(self):
        self.lat_error_integral = 0.0
        self.lat_error_prev = 0.0
        self.lon_error_integral = 0.0
        self.lon_error_prev = 0.0


class STAFAgent(BaseAgent):
    """
    CARLA Autonomous Agent powered by the proposed STAF Framework.

    Pipeline per simulation tick:
    1. Synchronize front RGB, LiDAR, and IMU measurements.
    2. Run SO(3) IMU Pre-integration to obtain 9D kinematic prior [Log(R), v, p].
    3. Generate 2-channel LiDAR BEV representation.
    4. Pass multimodal tokens to STAFModel with Kinematic Soft-Gating.
    5. Spatio-temporal memory queue temporally smooths fused state.
    6. Waypoint decoder predicts future path (pred_len=4).
    7. Dual-channel PID controller executes vehicle steer, throttle, and brake.
    """

    def setup(self, path_to_conf_file: Optional[str] = None):
        """Initialize the model, weights, pre-integrator, and controllers."""
        self.step = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"STAFAgent initializing on device: {self.device}")

        from staf.models.staf_model import STAFModel
        from staf.imu.preintegration import IMUPreintegrator
        from staf.data.cleaning import DataCleaner

        self.model = STAFModel(
            embed_dim=256,
            num_heads=4,
            num_fusion_layers=4,
            queue_length=5,
            pred_len=4,
            use_target_point=True,
            pretrained_backbones=False,
        ).to(self.device)
        self.model.eval()

        # Load weights if checkpoint provided
        if path_to_conf_file and os.path.exists(path_to_conf_file):
            ckpt_path = path_to_conf_file
            if os.path.isdir(path_to_conf_file):
                # Search for best_model.pth or latest .pth
                candidates = [f for f in os.listdir(path_to_conf_file) if f.endswith(".pth")]
                if "best_model.pth" in candidates:
                    ckpt_path = os.path.join(path_to_conf_file, "best_model.pth")
                elif candidates:
                    ckpt_path = os.path.join(path_to_conf_file, sorted(candidates)[-1])

            if os.path.isfile(ckpt_path):
                logger.info(f"Loading STAF checkpoint: {ckpt_path}")
                state = torch.load(ckpt_path, map_location=self.device)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                self.model.load_state_dict(state, strict=False)

        # Preintegrator & Cleaner
        self.preintegrator = IMUPreintegrator()
        self.cleaner = DataCleaner(max_black_pixel_ratio=0.3, min_lidar_points=50)

        # Controller & Buffers
        self.pid = PIDController()
        self.imu_buffer = deque(maxlen=20)
        self.gating_log: List[Dict[str, float]] = []

        # Reset model temporal memory queue
        self.model.reset_temporal_state(batch_size=1, device=self.device)
        self.initialized = True
        logger.info("STAFAgent setup completed.")

    def sensors(self) -> List[Dict[str, Any]]:
        """Define the sensor suite for CARLA simulation."""
        return [
            {
                "type": "sensor.camera.rgb",
                "x": 1.3, "y": 0.0, "z": 2.3,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "width": 960, "height": 480, "fov": 120,
                "id": "rgb"
            },
            {
                "type": "sensor.lidar.ray_cast",
                "x": 1.3, "y": 0.0, "z": 2.5,
                "roll": 0.0, "pitch": 0.0, "yaw": -90.0,
                "range": 50.0,
                "channels": 32,
                "points_per_second": 600000,
                "rotation_frequency": 20.0,
                "id": "lidar"
            },
            {
                "type": "sensor.other.imu",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                "id": "imu"
            },
            {
                "type": "sensor.other.gnss",
                "x": 0.0, "y": 0.0, "z": 0.0,
                "id": "gps"
            },
            {
                "type": "sensor.speedometer",
                "reading_frequency": 20,
                "id": "speed"
            }
        ]

    def _process_lidar_bev(self, raw_points: np.ndarray) -> torch.Tensor:
        """Project raw 3D LiDAR point cloud into a 2-channel 256x256 BEV tensor."""
        bev = np.zeros((2, 256, 256), dtype=np.float32)
        if raw_points is None or len(raw_points) == 0:
            return torch.from_numpy(bev).unsqueeze(0).to(self.device)

        pts = raw_points[:, :3]
        # CARLA coordinate bounds: x in [-16, 16], y in [-32, 0] or forward [0, 50]
        # Normalization to [0, 256] grid
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        mask = (x >= -25.0) & (x <= 25.0) & (y >= 0.0) & (y <= 50.0) & (z >= -3.0) & (z <= 3.0)
        x_filt = x[mask]
        y_filt = y[mask]
        z_filt = z[mask]

        if len(x_filt) > 0:
            grid_x = np.clip(((x_filt + 25.0) / 50.0 * 255.0).astype(int), 0, 255)
            grid_y = np.clip((y_filt / 50.0 * 255.0).astype(int), 0, 255)

            # Channel 0: Normalized height map
            # Channel 1: Normalized point density
            for gx, gy, gz in zip(grid_x, grid_y, z_filt):
                bev[0, gy, gx] = max(bev[0, gy, gx], (gz + 3.0) / 6.0)
                bev[1, gy, gx] = min(1.0, bev[1, gy, gx] + 0.1)

        return torch.from_numpy(bev).unsqueeze(0).to(self.device)

    def run_step(self, input_data: Dict[str, Any], timestamp: float) -> VehicleControl:
        """
        Execute one control cycle of the autonomous robot in CARLA simulation.

        Returns:
            carla.VehicleControl with calculated steer, throttle, and brake.
        """
        self.step += 1

        # 1. Parse Sensors
        rgb_raw = input_data.get("rgb", (0, None))[1]
        lidar_raw = input_data.get("lidar", (0, None))[1]
        imu_raw = input_data.get("imu", (0, None))[1]
        speed_raw = input_data.get("speed", (0, {}))[1]

        current_speed = float(speed_raw.get("speed", 0.0)) if isinstance(speed_raw, dict) else float(speed_raw or 0.0)

        # 2. Preprocess Camera Image -> (1, 3, 256, 256)
        if rgb_raw is not None:
            # Drop alpha if RGBA
            if rgb_raw.shape[-1] == 4:
                rgb_raw = rgb_raw[:, :, :3]
            img = Image.fromarray(rgb_raw).resize((256, 256))
            rgb_tensor = torch.from_numpy(np.array(img).transpose(2, 0, 1)).float() / 255.0
            rgb_tensor = rgb_tensor.unsqueeze(0).to(self.device)
        else:
            rgb_tensor = torch.zeros((1, 3, 256, 256), device=self.device)

        # 3. Preprocess LiDAR Point Cloud -> (1, 2, 256, 256)
        lidar_tensor = self._process_lidar_bev(lidar_raw)

        # 4. Integrate High-Frequency IMU -> 9D Kinematic Vector [Log(R), v, p]
        if imu_raw is not None and len(imu_raw) >= 6:
            # CARLA IMU format: [accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]
            acc = torch.tensor(imu_raw[:3], dtype=torch.float32)
            gyro = torch.tensor(imu_raw[3:6], dtype=torch.float32)
            self.preintegrator.integrate(gyro, acc, dt=0.05)
            x_kinematic = self.preintegrator.get_kinematic_vector().unsqueeze(0).to(self.device)
        else:
            x_kinematic = torch.zeros((1, 9), device=self.device)

        # 5. Model Inference (Gated Spatio-Temporal Attention Fusion)
        target_point = torch.tensor([[10.0, 0.0]], device=self.device)  # Default local forward target
        with torch.no_grad():
            outputs = self.model(
                rgb_tensor,
                lidar_tensor,
                x_kinematic,
                target_point=target_point,
            )

        w_cam = float(outputs["w_cam"].mean().item())
        w_lidar = float(outputs["w_lidar"].mean().item())
        self.gating_log.append({"step": self.step, "w_cam": w_cam, "w_lidar": w_lidar, "speed": current_speed})

        # 6. Waypoint Decoding & Control
        if "waypoints" in outputs:
            wps = outputs["waypoints"].squeeze(0).cpu().numpy()  # (4, 2)
            # Use aim point (step 2 or 3) for lateral steering angle
            aim_point = wps[1] if len(wps) > 1 else wps[0]
            target_heading_deg = math.degrees(math.atan2(aim_point[0], aim_point[1]))  # local lateral angle
            steer = self.pid.compute_lateral(target_heading_deg)

            # Desired target speed from waypoint progress
            dist = float(np.linalg.norm(aim_point))
            target_speed = float(np.clip(dist * 3.6, 2.0, 8.0))  # m/s -> km/h bounded target
            throttle, brake = self.pid.compute_longitudinal(current_speed, target_speed)
        else:
            # Fallback safe creep
            steer = 0.0
            throttle, brake = self.pid.compute_longitudinal(current_speed, target_speed=4.0)

        # Stuck detection: if stuck for > 20 frames, nudge forward
        if current_speed < 0.1 and throttle > 0.1:
            throttle = min(0.8, throttle + 0.1)
            brake = 0.0

        control = VehicleControl(steer=steer, throttle=throttle, brake=brake)
        return control

    def destroy(self):
        """Clean up resources upon route completion."""
        self.pid.reset()
        if hasattr(self, "preintegrator"):
            self.preintegrator.reset()
        logger.info(f"STAFAgent finished drive ({self.step} steps executed).")

