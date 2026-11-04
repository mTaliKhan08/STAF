import torch
import pytest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if STAF_ROOT not in sys.path:
    sys.path.insert(0, STAF_ROOT)

from staf.models.staf_model import STAFModel, compute_cc_trajectory_torch

def test_constant_curvature_residual_initialization():
    """Verify that at step 0, the residual waypoint head predicts exactly constant curvature."""
    torch.manual_seed(42)
    model = STAFModel(
        embed_dim=256,
        pred_len=4,
        modality="full",
        gate_input_dim=3,
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    )
    model.eval()

    bs = 4
    rgb = torch.randn(bs, 3, 256, 256)
    lidar_bev = torch.randn(bs, 2, 256, 256)
    x_kin = torch.randn(bs, 3)
    target = torch.randn(bs, 2)
    speed = torch.tensor([2.0, 5.0, 8.0, 0.0])
    accel = torch.tensor([0.1, -0.5, 0.0, 1.0])
    yaw_rate = torch.tensor([0.1, -0.2, 0.0, 0.05])

    out = model(
        rgb=rgb,
        lidar_bev=lidar_bev,
        x_kinematic=x_kin,
        target_point=target,
        ego_speed=speed,
        accel=accel,
        body_yaw_rate=yaw_rate,
    )

    pred_wps = out["waypoints"]
    cc_wps = out["cc_waypoints"]
    delta_wps = out["delta_waypoints"]

    # Assert that initial delta is identically zero
    assert torch.allclose(delta_wps, torch.zeros_like(delta_wps), atol=1e-6)
    # Assert that predictions identically match Constant Curvature
    assert torch.allclose(pred_wps, cc_wps, atol=1e-6)

def test_all_modality_variants():
    """Verify forward and backward passes across all 4 retrained variant architectures."""
    bs = 2
    rgb = torch.randn(bs, 3, 256, 256)
    lidar_bev = torch.randn(bs, 2, 256, 256)
    x_kin = torch.randn(bs, 3)
    target = torch.randn(bs, 2)
    speed = torch.tensor([4.0, 6.0])
    accel = torch.tensor([0.2, -0.1])
    yaw_rate = torch.tensor([0.05, -0.05])

    gt_wps = torch.randn(bs, 4, 2)

    for mod in ["full", "camera_only", "lidar_only", "no_vision"]:
        model = STAFModel(
            embed_dim=256,
            pred_len=4,
            modality=mod,
            gate_input_dim=3,
            use_residual_cc=True,
            use_full_kinematics_head=True,
            pretrained_backbones=False,
        )
        model.train()

        out = model(
            rgb=rgb if mod in ["full", "camera_only"] else None,
            lidar_bev=lidar_bev if mod in ["full", "lidar_only"] else None,
            x_kinematic=x_kin if mod == "full" else None,
            target_point=target,
            ego_speed=speed,
            accel=accel,
            body_yaw_rate=yaw_rate,
        )

        loss = torch.nn.functional.l1_loss(out["waypoints"], gt_wps)
        loss.backward()

        assert out["waypoints"].shape == (bs, 4, 2)
        assert not torch.isnan(loss)
        # Check that waypoint head received gradients
        assert model.waypoint_head[-1].weight.grad is not None

