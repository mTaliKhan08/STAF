"""
Unit tests for STAF core modules:
- IMU Pre-integration (SO(3) maps, bias estimation, kinematic vector)
- Soft-Gating MLP (forward pass, parameter count)
- Class-Balanced Focal Loss (weight computation, forward pass)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

def test_so3_exp_identity():
    """Exp(0) should give identity rotation."""
    from staf.imu.preintegration import so3_exp
    omega = torch.zeros(3)
    R = so3_exp(omega)
    I = torch.eye(3)
    assert torch.allclose(R, I, atol=1e-6), f"Exp(0) != I: {R}"
    print("[PASS] so3_exp(0) = I")

def test_so3_exp_log_roundtrip():
    """Log(Exp(omega)) should return omega."""
    from staf.imu.preintegration import so3_exp, so3_log
    omega = torch.tensor([0.1, -0.2, 0.3])
    R = so3_exp(omega)
    omega_recovered = so3_log(R.unsqueeze(0)).squeeze(0)
    assert torch.allclose(omega, omega_recovered, atol=1e-5), \
        f"Roundtrip failed: {omega} != {omega_recovered}"
    print("[PASS] so3_log(so3_exp(omega)) = omega")

def test_so3_exp_orthogonality():
    """Exp(omega) should produce an orthogonal matrix with det=+1."""
    from staf.imu.preintegration import so3_exp
    omega = torch.tensor([0.5, -1.0, 0.7])
    R = so3_exp(omega)
    RtR = R.T @ R
    I = torch.eye(3)
    assert torch.allclose(RtR, I, atol=1e-5), f"R^T R != I: {RtR}"
    det = torch.det(R)
    assert torch.allclose(det, torch.tensor(1.0), atol=1e-5), f"det(R) != 1: {det}"
    print("[PASS] so3_exp produces valid SO(3) matrix")

def test_preintegrator_stationary():
    """Stationary IMU (zero gyro, gravity-only accel) should give identity rotation."""
    from staf.imu.preintegration import IMUPreintegrator
    preint = IMUPreintegrator()
    gravity = torch.tensor([0.0, 0.0, 9.81])
    zero_gyro = torch.zeros(3)
    dt = 0.005  # 200 Hz
    for _ in range(100):  # 0.5 seconds
        preint.integrate(zero_gyro, gravity, dt)
    
    x = preint.get_kinematic_vector()
    assert x.shape == (9,), f"Wrong shape: {x.shape}"
    # Rotation should be ~0 (identity)
    assert torch.allclose(x[:3], torch.zeros(3), atol=1e-4), \
        f"Rotation not identity: {x[:3]}"
    print(f"[PASS] Stationary preintegration: X_kinematic = {x.numpy().round(4)}")

def test_preintegrator_pure_rotation():
    """Constant angular velocity should integrate to expected rotation."""
    from staf.imu.preintegration import IMUPreintegrator, so3_log
    preint = IMUPreintegrator()
    # 1 rad/s around z-axis for 1 second at 200 Hz
    omega = torch.tensor([0.0, 0.0, 1.0])
    accel = torch.tensor([0.0, 0.0, 9.81])  # gravity only
    dt = 0.005
    for _ in range(200):
        preint.integrate(omega, accel, dt)
    
    x = preint.get_kinematic_vector()
    # Expected: ~1 radian around z-axis
    assert abs(x[2].item() - 1.0) < 0.05, f"Z rotation wrong: {x[2].item()}"
    print(f"[PASS] Pure rotation preintegration: rotation_z = {x[2].item():.4f} rad (expected ~1.0)")

def test_bias_estimation():
    """Bias estimation from stationary data should recover known biases."""
    from staf.imu.preintegration import estimate_initial_bias
    N = 200
    true_bg = torch.tensor([0.01, -0.02, 0.005])
    true_ba = torch.tensor([0.1, -0.05, 0.2])
    gravity = torch.tensor([0.0, 0.0, 9.81])
    
    # Simulate noisy stationary readings
    torch.manual_seed(42)
    gyro_data = true_bg.unsqueeze(0).expand(N, -1) + torch.randn(N, 3) * 0.001
    accel_data = (gravity + true_ba).unsqueeze(0).expand(N, -1) + torch.randn(N, 3) * 0.01
    
    est_bg, est_ba = estimate_initial_bias(gyro_data, accel_data)
    assert torch.allclose(est_bg, true_bg, atol=0.005), f"Gyro bias wrong: {est_bg}"
    assert torch.allclose(est_ba, true_ba, atol=0.05), f"Accel bias wrong: {est_ba}"
    print(f"[PASS] Bias estimation: bg={est_bg.numpy().round(4)}, ba={est_ba.numpy().round(4)}")

def test_soft_gating_forward():
    """Soft-gating MLP should produce weights in [0, 1]."""
    from staf.models.soft_gating import IMUGuidedSoftGating
    model = IMUGuidedSoftGating(input_dim=9, hidden_dim=64, dropout=0.1)
    model.eval()
    
    batch = torch.randn(4, 9)
    w_cam, w_lidar = model(batch)
    
    assert w_cam.shape == (4, 1), f"w_cam shape wrong: {w_cam.shape}"
    assert w_lidar.shape == (4, 1), f"w_lidar shape wrong: {w_lidar.shape}"
    assert (w_cam >= 0).all() and (w_cam <= 1).all(), "w_cam out of [0,1]"
    assert (w_lidar >= 0).all() and (w_lidar <= 1).all(), "w_lidar out of [0,1]"
    
    param_count = model.get_param_count()
    print(f"[PASS] Soft-gating forward: w_cam={w_cam.mean():.4f}, w_lidar={w_lidar.mean():.4f}")
    print(f"       Total parameters: {param_count}")

def test_focal_loss_forward():
    """Focal loss should compute without errors and be positive."""
    from staf.losses.focal_loss import ClassBalancedFocalLoss
    
    num_classes = 10
    class_counts = torch.tensor([10000, 8000, 500, 200, 50, 7000, 3000, 100, 6000, 9000], dtype=torch.float)
    
    loss_fn = ClassBalancedFocalLoss(num_classes=num_classes, class_counts=class_counts, gamma=2.0)
    
    logits = torch.randn(8, num_classes, 16, 16)  # B=8, C=10, H=W=16
    targets = torch.randint(0, num_classes, (8, 16, 16))
    
    loss = loss_fn(logits, targets)
    assert loss.item() > 0, f"Loss should be positive: {loss.item()}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"
    
    # Check that rare classes get higher weights
    weights = loss_fn.alpha
    # Class 4 (count=50) should have higher weight than class 0 (count=10000)
    assert weights[4] > weights[0], f"Rare class weight should be higher: w[4]={weights[4]:.4f} vs w[0]={weights[0]:.4f}"
    print(f"[PASS] Focal loss: value={loss.item():.4f}")
    print(f"       Class weights (first 5): {weights[:5].numpy().round(3)}")

def test_focal_loss_class_balance():
    """Verify effective number of samples weighting logic."""
    from staf.losses.focal_loss import ClassBalancedFocalLoss
    
    # Extreme imbalance: 1 class has 100K, another has 10
    counts = torch.tensor([100000.0, 10.0])
    loss_fn = ClassBalancedFocalLoss(num_classes=2, class_counts=counts, gamma=2.0)
    
    ratio = (loss_fn.alpha[1] / loss_fn.alpha[0]).item()
    assert ratio > 10, f"Rare class should have much higher weight, ratio={ratio:.2f}"
    print(f"[PASS] Class balance: rare/common weight ratio = {ratio:.1f}x")



def test_staf_model_waypoint_prediction():
    """Verify STAFModel waypoint prediction head for simulation driving."""
    from staf.models.staf_model import STAFModel
    
    B = 2
    model = STAFModel(embed_dim=128, pred_len=4, use_target_point=True, pretrained_backbones=False)
    rgb = torch.randn(B, 3, 256, 256)
    lidar = torch.randn(B, 2, 256, 256)
    imu = torch.randn(B, 9)
    target = torch.randn(B, 2)
    
    out = model(rgb, lidar, imu, target_point=target)
    assert "waypoints" in out, "Model output missing waypoints!"
    assert out["waypoints"].shape == (B, 4, 2), f"Expected (B, 4, 2), got {out['waypoints'].shape}"
    assert "w_cam" in out and "w_lidar" in out, "Missing gating weights!"
    print(f"[PASS] STAFModel waypoint prediction: shape={out['waypoints'].shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("STAF Core Module Unit Tests")
    print("=" * 60)
    
    tests = [
        test_so3_exp_identity,
        test_so3_exp_log_roundtrip,
        test_so3_exp_orthogonality,
        test_preintegrator_stationary,
        test_preintegrator_pure_rotation,
        test_bias_estimation,
        test_soft_gating_forward,
        test_focal_loss_forward,
        test_focal_loss_class_balance,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("=" * 60)
