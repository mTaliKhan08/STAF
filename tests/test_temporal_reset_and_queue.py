import torch
import pytest
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if STAF_ROOT not in sys.path:
    sys.path.insert(0, STAF_ROOT)

from staf.models.staf_model import STAFModel
from staf.models.soft_gating import IMUGuidedSoftGating
from staf.data.dataset import RouteOrderedBatchSampler


def test_queue_length_1_equals_single_frame():
    """
    CHECK (a): Verify queue_length=1 output equals the single-frame path.
    With queue_length=1, the temporal memory holds only the current frame,
    so feeding a sequence yields the exact same output on frame t as feeding
    frame t fresh in isolation.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = STAFModel(
        embed_dim=64,
        num_heads=2,
        num_fusion_layers=2,
        ff_dim=128,
        queue_length=1,
        num_temporal_layers=1,
        pred_len=4,
        modality="full",
        gate_input_dim=3,
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    )
    model.eval()

    bs = 1
    # Create 3 sequential frames
    frames = []
    for i in range(3):
        frames.append({
            "rgb": torch.randn(bs, 3, 128, 128),
            "lidar_bev": torch.randn(bs, 2, 128, 128),
            "x_kin": torch.tensor([[1.0 + i * 0.5, 0.1 * i, 0.05 * i]]),
            "target": torch.tensor([[5.0 + i, 2.0 - i]]),
            "speed": torch.tensor([10.0 + i * 5.0]),
            "accel": torch.tensor([0.2 * i]),
            "yaw_rate": torch.tensor([0.05 * i]),
        })

    # Sequential pass: feed frame 0, frame 1, frame 2
    model.reset_temporal_state(bs, device)
    with torch.no_grad():
        for i in range(2):
            f = frames[i]
            _ = model(
                rgb=f["rgb"],
                lidar_bev=f["lidar_bev"],
                x_kinematic=f["x_kin"],
                target_point=f["target"],
                ego_speed=f["speed"],
                accel=f["accel"],
                body_yaw_rate=f["yaw_rate"],
            )

        # Output at frame 2 after sequential history
        f2 = frames[2]
        out2_seq = model(
            rgb=f2["rgb"],
            lidar_bev=f2["lidar_bev"],
            x_kinematic=f2["x_kin"],
            target_point=f2["target"],
            ego_speed=f2["speed"],
            accel=f2["accel"],
            body_yaw_rate=f2["yaw_rate"],
        )

    # Single-frame pass: reset queue and feed only frame 2
    model.reset_temporal_state(bs, device)
    with torch.no_grad():
        out2_single = model(
            rgb=f2["rgb"],
            lidar_bev=f2["lidar_bev"],
            x_kinematic=f2["x_kin"],
            target_point=f2["target"],
            ego_speed=f2["speed"],
            accel=f2["accel"],
            body_yaw_rate=f2["yaw_rate"],
        )

    # Both must match within numerical precision
    assert torch.allclose(out2_seq["fused_features"], out2_single["fused_features"], atol=1e-5), (
        "fused_features differ between sequential path and single-frame path for queue_length=1!"
    )
    assert torch.allclose(out2_seq["waypoints"], out2_single["waypoints"], atol=1e-5), (
        "Waypoints differ between sequential path and single-frame path for queue_length=1!"
    )


def test_queue_length_5_holds_order_and_resets_at_boundary():
    """
    CHECK (b): Verify queue_length=5 holds 5 frames from the same route in order,
    and resets cleanly at a route boundary.
    """
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = STAFModel(
        embed_dim=64,
        num_heads=2,
        num_fusion_layers=2,
        ff_dim=128,
        queue_length=5,
        num_temporal_layers=1,
        pred_len=4,
        modality="full",
        gate_input_dim=3,
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    )
    model.eval()

    bs = 2  # 2 routes in parallel

    # Start Route 1
    model.reset_temporal_state(bs, device)
    q_init = model.temporal_memory.get_queue_state()
    assert torch.all(q_init == 0), "Queue must initialize to all zeros upon reset!"

    # Feed 6 frames from Route 1
    with torch.no_grad():
        for t in range(6):
            rgb = torch.full((bs, 3, 128, 128), float(t + 1))
            lidar_bev = torch.full((bs, 2, 128, 128), float(t + 1))
            x_kin = torch.full((bs, 3), float(t + 1))
            target = torch.full((bs, 2), float(t + 1))
            speed = torch.full((bs,), float(t + 1))
            accel = torch.full((bs,), 0.0)
            yaw_rate = torch.full((bs,), 0.0)

            _ = model(
                rgb=rgb,
                lidar_bev=lidar_bev,
                x_kinematic=x_kin,
                target_point=target,
                ego_speed=speed,
                accel=accel,
                body_yaw_rate=yaw_rate,
            )

            fill_cnt = model.temporal_memory._fill_count
            expected_fill = min(t + 1, 5)
            assert fill_cnt == expected_fill, f"At step {t}, fill_count is {fill_cnt}, expected {expected_fill}"

    # Queue state now holds 5 frames
    q_state_r1 = model.temporal_memory.get_queue_state()
    assert q_state_r1.shape == (bs, 5, 64)

    # Route boundary occurs -> Route 2 starts
    model.reset_temporal_state(bs, device)
    q_reset = model.temporal_memory.get_queue_state()
    assert torch.all(q_reset == 0), "Queue must be all zeros after route boundary reset!"
    assert model.temporal_memory._fill_count == 0, "Fill count must reset to 0 at route boundary!"

    # Feed first frame of Route 2
    with torch.no_grad():
        rgb = torch.full((bs, 3, 128, 128), 100.0)
        lidar_bev = torch.full((bs, 2, 128, 128), 100.0)
        x_kin = torch.full((bs, 3), 100.0)
        target = torch.full((bs, 2), 100.0)
        speed = torch.full((bs,), 100.0)
        accel = torch.full((bs,), 0.0)
        yaw_rate = torch.full((bs,), 0.0)

        _ = model(
            rgb=rgb,
            lidar_bev=lidar_bev,
            x_kinematic=x_kin,
            target_point=target,
            ego_speed=speed,
            accel=accel,
            body_yaw_rate=yaw_rate,
        )

    # Queue should have fill_count = 1, containing ONLY Route 2 frame 0
    assert model.temporal_memory._fill_count == 1
    q_state_r2 = model.temporal_memory.get_queue_state()
    # Old positions (0 to 3) should still be zeros, only the latest position (-1) is filled
    assert torch.all(q_state_r2[:, :-1, :] == 0), "Earlier queue slots must be zero after route reset!"
    assert not torch.all(q_state_r2[:, -1, :] == 0), "Latest queue slot must be non-zero!"


def test_constant_gate_mode():
    """
    CHECK (STEP 2): Verify constant mode outputs exactly 0.5 for both weights on a test batch.
    """
    torch.manual_seed(42)
    bs = 4
    x_kin = torch.tensor([
        [1.0, 0.2, -0.1],
        [5.0, -1.0, 0.3],
        [0.0, 0.0, 0.0],
        [8.0, 2.0, -0.5],
    ])

    # Direct soft gating module check
    gating_const = IMUGuidedSoftGating(input_dim=3, gate_mode="constant")
    w_cam, w_lidar = gating_const(x_kin)

    assert w_cam.shape == (bs, 1)
    assert w_lidar.shape == (bs, 1)
    assert torch.all(w_cam == 0.5), f"Expected w_cam == 0.5, got {w_cam}"
    assert torch.all(w_lidar == 0.5), f"Expected w_lidar == 0.5, got {w_lidar}"

    # Full STAFModel check with constant gate
    model_const = STAFModel(
        embed_dim=64,
        num_heads=2,
        num_fusion_layers=2,
        pred_len=4,
        modality="full",
        gate_input_dim=3,
        gate_mode="constant",
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    )
    model_const.eval()

    rgb = torch.randn(bs, 3, 128, 128)
    lidar_bev = torch.randn(bs, 2, 128, 128)
    target = torch.randn(bs, 2)
    speed = torch.tensor([2.0, 5.0, 8.0, 0.0])
    accel = torch.tensor([0.1, -0.5, 0.0, 1.0])
    yaw_rate = torch.tensor([0.1, -0.2, 0.0, 0.05])

    with torch.no_grad():
        out = model_const(
            rgb=rgb,
            lidar_bev=lidar_bev,
            x_kinematic=x_kin,
            target_point=target,
            ego_speed=speed,
            accel=accel,
            body_yaw_rate=yaw_rate,
        )

    assert torch.all(out["w_cam"] == 0.5), f"Model out w_cam not 0.5: {out['w_cam']}"
    assert torch.all(out["w_lidar"] == 0.5), f"Model out w_lidar not 0.5: {out['w_lidar']}"

    # Contrast with learned mode on the same batch
    model_learned = STAFModel(
        embed_dim=64,
        num_heads=2,
        num_fusion_layers=2,
        pred_len=4,
        modality="full",
        gate_input_dim=3,
        gate_mode="learned",
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    )
    model_learned.eval()
    with torch.no_grad():
        out_learned = model_learned(
            rgb=rgb,
            lidar_bev=lidar_bev,
            x_kinematic=x_kin,
            target_point=target,
            ego_speed=speed,
            accel=accel,
            body_yaw_rate=yaw_rate,
        )
    # In learned mode, weights should NOT all be identical to 0.5
    assert not torch.all(out_learned["w_cam"] == 0.5)


def test_route_ordered_batch_sampler_structure():
    """Verify RouteOrderedBatchSampler groups routes and maintains chronological order."""
    class DummyDataset:
        def __init__(self):
            # 3 routes with different frame counts
            self.samples = []
            for r in ["route_A", "route_B", "route_C"]:
                n_frames = 10 if r == "route_A" else (8 if r == "route_B" else 12)
                for f in range(n_frames):
                    self.samples.append({
                        "route_dir": r,
                        "frame_idx": f,
                        "frame_id": f"{f:04d}",
                    })

    dataset = DummyDataset()
    sampler = RouteOrderedBatchSampler(dataset, batch_size=2, shuffle=False)

    batches = list(sampler)
    assert len(batches) > 0

    # First route chunk contains route_A and route_B (max len = 10)
    # Steps 0..9 should have 2 indices each
    for t in range(10):
        b = batches[t]
        assert len(b) == 2
        # Check that frame indices advance in order
        sample_0 = dataset.samples[b[0]]
        sample_1 = dataset.samples[b[1]]
        routes_in_batch = {sample_0["route_dir"], sample_1["route_dir"]}
        assert routes_in_batch == {"route_A", "route_B"}
        r_a = sample_0 if sample_0["route_dir"] == "route_A" else sample_1
        r_b = sample_1 if sample_1["route_dir"] == "route_B" else sample_0
        assert r_a["frame_idx"] == t
        assert r_b["frame_idx"] == min(t, 7)

