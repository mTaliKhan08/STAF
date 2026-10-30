"""
Unit tests for STAF Phase 1 data pipeline:
- Data cleaning (camera, LiDAR, IMU filtering)
- Sensor synchronization
- BEV projection
- IMU pre-integration through dataset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def test_camera_cleaning_pass():
    """Normal camera frame should pass validation."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner(max_black_pixel_ratio=0.3)
    # Normal image (random pixels)
    image = np.random.randint(30, 220, (256, 256, 3), dtype=np.uint8)
    assert cleaner.check_camera_frame(image) is True
    print("[PASS] Normal camera frame passes cleaning")


def test_camera_cleaning_reject_black():
    """Mostly-black camera frame should be rejected."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner(max_black_pixel_ratio=0.3)
    # 80% black image
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:50, :, :] = 128  # Only top 20% has content
    assert cleaner.check_camera_frame(image) is False
    print("[PASS] Mostly-black camera frame rejected")


def test_lidar_cleaning_pass():
    """Normal LiDAR sweep should pass validation."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner(min_lidar_points=100)
    points = np.random.randn(5000, 4).astype(np.float32)
    assert cleaner.check_lidar_sweep(points) is True
    print("[PASS] Normal LiDAR sweep passes cleaning")


def test_lidar_cleaning_reject_sparse():
    """Sparse LiDAR sweep should be rejected."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner(min_lidar_points=100)
    points = np.random.randn(50, 4).astype(np.float32)  # Only 50 points
    assert cleaner.check_lidar_sweep(points) is False
    print("[PASS] Sparse LiDAR sweep rejected (50 < 100)")


def test_imu_gap_detection():
    """IMU timestamps with gaps should be detected."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner(max_imu_gap_ms=5.0)
    # Normal 200Hz timestamps with one 20ms gap
    ts = np.arange(0, 0.1, 0.005)  # 20 samples at 200Hz
    ts = np.concatenate([ts[:10], ts[10:] + 0.015])  # Insert 15ms gap
    gap_mask, _ = cleaner.check_and_fix_imu_timestamps(ts, expected_dt_ms=5.0)
    assert gap_mask[9] is True or gap_mask[9] == True, "Gap at index 9 not detected"
    print(f"[PASS] IMU gap detected: {np.sum(gap_mask)} gaps found")


def test_cleaning_stats():
    """Cleaning stats should track correctly."""
    from staf.data.cleaning import DataCleaner
    cleaner = DataCleaner()
    cleaner.validate_sample(
        image=np.random.randint(30, 220, (64, 64, 3), dtype=np.uint8),
        points=np.random.randn(500, 3).astype(np.float32),
    )
    cleaner.validate_sample(
        image=np.zeros((64, 64, 3), dtype=np.uint8),  # Black
        points=np.random.randn(500, 3).astype(np.float32),
    )
    stats = cleaner.get_stats()
    assert stats["total_samples"] == 2
    assert stats["camera_discarded"] == 1
    assert stats["samples_retained"] == 1
    print(f"[PASS] Cleaning stats: {stats}")


def test_sensor_synchronization():
    """Sensor synchronizer should match timestamps correctly."""
    from staf.data.sync import SensorSynchronizer
    sync = SensorSynchronizer(max_lidar_offset_ms=50.0)

    cam_ts = np.array([0.0, 0.05, 0.10, 0.15, 0.20])  # 20Hz
    lidar_ts = np.array([0.01, 0.11, 0.21])  # 10Hz
    imu_ts = np.arange(0, 0.25, 0.005)  # 200Hz = 50 samples

    samples = sync.synchronize(cam_ts, lidar_ts, imu_ts)

    assert len(samples) == 5, f"Expected 5 samples, got {len(samples)}"
    # First camera (0.0) should match first LiDAR (0.01) - 10ms offset
    assert samples[0].lidar_idx >= 0, "First cam should have LiDAR match"
    assert samples[0].time_offset_lidar_ms < 50.0, "LiDAR offset should be < 50ms"
    # All samples should have IMU data
    for s in samples:
        assert s.num_imu_samples > 0, f"Sample should have IMU data: {s}"
    print(f"[PASS] Sensor sync: {len(samples)} samples, all have IMU data")


def test_imu_window_extractor():
    """IMU window extractor should slice correct measurements."""
    from staf.data.sync import SensorSynchronizer, IMUWindowExtractor, SyncedSample

    extractor = IMUWindowExtractor()
    imu_ts = np.arange(0, 0.2, 0.005)  # 200Hz, 40 samples
    imu_gyro = np.random.randn(40, 3).astype(np.float32)
    imu_accel = np.random.randn(40, 3).astype(np.float32)
    imu_accel[:, 2] += 9.81  # Add gravity

    sample = SyncedSample(
        timestamp=0.1, camera_idx=0, lidar_idx=0,
        imu_start_idx=10, imu_end_idx=20,
        time_offset_lidar_ms=5.0, num_imu_samples=10,
    )

    window = extractor.extract_window(sample, imu_ts, imu_gyro, imu_accel)
    assert window["gyro"].shape == (10, 3), f"Gyro shape: {window['gyro'].shape}"
    assert window["accel"].shape == (10, 3), f"Accel shape: {window['accel'].shape}"
    assert len(window["dt"]) == 10, f"dt length: {len(window['dt'])}"
    print(f"[PASS] IMU window extraction: {window['gyro'].shape}")


def test_bev_projection():
    """BEV projection should produce valid 2-channel map."""
    from staf.data.dataset import STAFDataset

    # Create a minimal subclass for testing
    class TestDataset(STAFDataset):
        def _load_image(self, s): return None
        def _load_lidar(self, s): return None
        def _load_imu(self, s): return {"gyro": np.zeros((1, 3)), "accel": np.zeros((1, 3)), "dt": np.array([0.005])}

    ds = TestDataset(data_root="", apply_cleaning=False)

    # Generate random 3D points
    points = np.random.uniform(-30, 30, (1000, 3)).astype(np.float32)
    bev = ds._points_to_bev(points)

    assert bev.shape == (2, 256, 256), f"BEV shape: {bev.shape}"
    assert bev[0].max() > 0, "Height channel should have non-zero values"
    assert bev[1].max() <= 1.0, "Density channel should be normalized"
    print(f"[PASS] BEV projection: shape={bev.shape}, height_max={bev[0].max():.2f}, density_max={bev[1].max():.2f}")


def test_end_to_end_model_forward():
    """Full STAF model forward pass with synthetic data."""
    from staf.models.staf_model import STAFModel

    model = STAFModel(
        embed_dim=64,  # Small for testing
        num_heads=2,
        num_fusion_layers=2,
        ff_dim=128,
        queue_length=3,
        num_classes=10,
        pretrained_backbones=False,  # Skip downloading weights
    )
    model.eval()

    B = 2
    rgb = torch.randn(B, 3, 256, 256)
    lidar_bev = torch.randn(B, 2, 256, 256)
    x_kinematic = torch.randn(B, 9)

    with torch.no_grad():
        outputs = model(rgb, lidar_bev, x_kinematic)

    assert "fused_features" in outputs, "Missing fused_features"
    assert "w_cam" in outputs, "Missing w_cam"
    assert "w_lidar" in outputs, "Missing w_lidar"
    assert "logits" in outputs, "Missing logits"
    assert outputs["fused_features"].shape == (B, 64), f"Features shape: {outputs['fused_features'].shape}"
    assert outputs["logits"].shape == (B, 10), f"Logits shape: {outputs['logits'].shape}"

    params = model.get_param_summary()
    print(f"[PASS] E2E forward pass: features={outputs['fused_features'].shape}, logits={outputs['logits'].shape}")
    print(f"       Total params: {params['total']:,}")
    print(f"       Soft-gating params: {params['soft_gating']:,} ({100*params['soft_gating']/params['total']:.3f}%)")


def test_class_frequency_computation():
    """Class frequency computation for loss weighting."""
    from staf.data.sync import compute_class_frequencies

    labels = [
        np.array([0, 0, 0, 1, 1, 2]),
        np.array([0, 0, 3, 3, 3, 3]),
    ]
    counts = compute_class_frequencies(labels, num_classes=5)
    assert counts[0] == 5, f"Class 0 count: {counts[0]}"
    assert counts[1] == 2, f"Class 1 count: {counts[1]}"
    assert counts[2] == 1, f"Class 2 count: {counts[2]}"
    assert counts[3] == 4, f"Class 3 count: {counts[3]}"
    assert counts[4] == 0, f"Class 4 count: {counts[4]}"
    print(f"[PASS] Class frequencies: {counts}")



def test_carla_staf_dataset_synthetic(tmp_path=None):
    """Verify CARLASTAFDataset loads and formats synchronized multi-modal data and waypoints."""
    import tempfile
    import json
    from PIL import Image
    from staf.data.dataset import CARLASTAFDataset
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        route_dir = os.path.join(tmp_dir, "scenario1", "routes_Town01_00")
        os.makedirs(os.path.join(route_dir, "rgb"))
        os.makedirs(os.path.join(route_dir, "lidar"))
        os.makedirs(os.path.join(route_dir, "measurements"))
        os.makedirs(os.path.join(route_dir, "label_raw"))
        
        # Create 12 synthetic frames
        for i in range(12):
            fid = f"{i:04d}"
            # RGB
            img = Image.fromarray(np.random.randint(50, 200, (120, 160, 3), dtype=np.uint8))
            img.save(os.path.join(route_dir, "rgb", f"{fid}.png"))
            # LiDAR (XYZI points)
            pts = np.random.randn(200, 4).astype(np.float32)
            np.save(os.path.join(route_dir, "lidar", f"{fid}.npy"), pts)
            # Measurements (IMU & speed)
            with open(os.path.join(route_dir, "measurements", f"{fid}.json"), "w") as f:
                json.dump({"speed": 5.0, "accelerometer": [0.1, 0.0, 0.0], "gyroscope": [0.0, 0.0, 0.05]}, f)
            # Labels (Waypoints)
            with open(os.path.join(route_dir, "label_raw", f"{fid}.json"), "w") as f:
                json.dump({"x": float(i * 2.0), "y": 0.0, "theta": 0.0}, f)
                
        ds = CARLASTAFDataset(tmp_dir, split="train", pred_len=4)
        assert len(ds) > 0, "Dataset should have discovered synthetic route samples"
        sample = ds[0]
        
        assert "image" in sample, "Sample missing image tensor"
        assert "lidar_bev" in sample, "Sample missing lidar_bev tensor"
        assert "x_kinematic" in sample, "Sample missing x_kinematic tensor"
        assert "waypoints" in sample, "Sample missing waypoints"
        assert sample["waypoints"].shape == (4, 2), f"Expected (4, 2), got {sample['waypoints'].shape}"
        assert sample["x_kinematic"].shape == (9,), f"Expected (9,), got {sample['x_kinematic'].shape}"
        print(f"[PASS] CARLASTAFDataset synthetic: len={len(ds)}, waypoints shape={sample['waypoints'].shape}")



def test_staf_agent_simulation_step():
    """Verify STAFAgent runs in CARLA simulation environment and produces valid VehicleControl."""
    from staf.agents.staf_agent import STAFAgent
    
    agent = STAFAgent()
    agent.setup(path_to_conf_file=None)
    
    sensors = agent.sensors()
    assert len(sensors) >= 4, f"Expected at least 4 sensors, got {len(sensors)}"
    
    # Mock sensor tick matching CARLA simulation format
    mock_input = {
        "rgb": (1, np.random.randint(0, 255, (480, 960, 3), dtype=np.uint8)),
        "lidar": (1, np.random.randn(500, 4).astype(np.float32)),
        "imu": (1, np.array([0.05, 0.0, 9.81, 0.0, 0.0, 0.02], dtype=np.float32)),
        "speed": (1, {"speed": 5.2}),
    }
    
    control = agent.run_step(mock_input, timestamp=1.0)
    assert hasattr(control, "steer"), "Control output missing steer"
    assert hasattr(control, "throttle"), "Control output missing throttle"
    assert hasattr(control, "brake"), "Control output missing brake"
    assert -1.0 <= control.steer <= 1.0, f"Steer out of bounds: {control.steer}"
    assert 0.0 <= control.throttle <= 1.0, f"Throttle out of bounds: {control.throttle}"
    assert 0.0 <= control.brake <= 1.0, f"Brake out of bounds: {control.brake}"
    assert len(agent.gating_log) > 0, "Agent failed to record gating weights telemetry"
    
    agent.destroy()
    print(f"[PASS] STAFAgent closed-loop control step: steer={control.steer:.3f}, throttle={control.throttle:.3f}, brake={control.brake:.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("STAF Phase 1 Data Pipeline Tests")
    print("=" * 60)

    tests = [
        test_camera_cleaning_pass,
        test_camera_cleaning_reject_black,
        test_lidar_cleaning_pass,
        test_lidar_cleaning_reject_sparse,
        test_imu_gap_detection,
        test_cleaning_stats,
        test_sensor_synchronization,
        test_imu_window_extractor,
        test_bev_projection,
        test_end_to_end_model_forward,
        test_class_frequency_computation,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
