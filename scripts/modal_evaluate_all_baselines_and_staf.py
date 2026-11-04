"""
Modal Script to Evaluate ALL Baselines & STAF (Best Model: Epoch 1)
on the Full Validation Set (5,816 Frames).

Evaluates:
1. Constant Velocity (CV)
2. Constant Curvature (CC) with omega_body = -d_theta / dt
3. Kinematics MLP (No Goal, 3 in)
4. Kinematics MLP (With Goal, 5 in)
5. Kinematics MLP Residual over Constant Curvature (Residual over CC)
6. STAF Best Model (Epoch 1 checkpoint)

Subgroups:
- Overall Validation Set (All 5,816 frames)
- Turning Frames (|steer| > 0.05)
- Straight Frames (|steer| <= 0.05)

Standardized Metrics (Both Mean and Median):
- ADE [m] (Euclidean distance over horizons)
- FDE [m] (Euclidean distance at 2.0s)
- Lat-ADE [m] (mean |dy|)
- Lat-FDE [m] (|dy| at 2.0s)
- Mean L1 (sum) [m] (mean over horizons of |dx| + |dy|)
- Per-Horizon L1 (sum) [0.5s, 1.0s, 1.5s, 2.0s]
"""

import os
import sys
import math
import numpy as np
import modal

app = modal.App("staf-full-benchmark-evaluation")
data_volume = modal.Volume.from_name("staf-datasets")
checkpoint_volume = modal.Volume.from_name("staf-checkpoints")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/root/STAF"):
    staf_local_path = "/root/STAF"
else:
    staf_candidates = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../staf")),
    ]
    staf_local_path = None
    for c in staf_candidates:
        if os.path.exists(os.path.join(c, "staf", "models")):
            staf_local_path = c
            break
    assert staf_local_path is not None, f"Could not locate STAF root from {SCRIPT_DIR}"

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy",
        "scipy",
        "pillow",
        "timm",
        "einops",
    )
)

if not os.path.exists("/root/STAF") and staf_local_path:
    eval_image = eval_image.add_local_dir(
        staf_local_path,
        remote_path="/root/STAF",
        ignore=["**/__pycache__", "**/.git", "**/checkpoints", "**/*.pth", "**/*.pt"],
    )

@app.function(
    image=eval_image,
    gpu="A10G",
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,
)
def evaluate_all():
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from staf.data.dataset import (
        CARLASTAFDataset,
        wrap_angle,
        transform_future_ego_pose,
        transform_world_point_to_ego,
    )
    from staf.models.staf_model import STAFModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    print("Loading datasets...", flush=True)
    val_dataset = CARLASTAFDataset(carla_root, split="val", pred_len=4, augment_domain=False)
    train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=4, augment_domain=False)
    print(f"Loaded datasets: {len(train_dataset)} train samples, {len(val_dataset)} val samples", flush=True)

    time_horizons = [0.5, 1.0, 1.5, 2.0]

    def extract_features(dataset, max_samples=None):
        features = []
        indices = range(len(dataset)) if max_samples is None else range(min(max_samples, len(dataset)))
        for idx in indices:
            s_info = dataset.samples[idx]
            meas_path = s_info["meas_path"]
            m_dir = os.path.dirname(meas_path)
            frames = s_info["frames_list"]
            f_idx = s_info["frame_idx"]

            import json
            try:
                with open(meas_path, "r") as f:
                    d_cur = json.load(f)
                prev_p = os.path.join(m_dir, f"{frames[f_idx - 1]}.json")
                with open(prev_p, "r") as f:
                    d_prev = json.load(f)
            except Exception:
                continue

            wps, tgt, spd = dataset._load_waypoints(s_info)
            if wps is None:
                continue

            spd_prev = float(d_prev.get("speed", spd))
            ax = (spd - spd_prev) / 0.5

            th_cur = float(d_cur["theta"])
            th_prev = float(d_prev["theta"])
            d_th = wrap_angle(th_cur - th_prev)
            raw_w = d_th / 0.5
            body_w = -raw_w  # ISO 8855 right-handed body yaw-rate

            steer = float(d_cur.get("steer", 0.0))

            features.append({
                "idx": idx,
                "speed": spd,
                "accel": ax,
                "body_yaw_rate": body_w,
                "steer": steer,
                "target_point": tgt,       # (2,)
                "gt_waypoints": wps,       # (4, 2)
            })
        return features

    val_feats = extract_features(val_dataset)
    print(f"Extracted {len(val_feats)} valid feature samples from val dataset.", flush=True)

    train_feats = extract_features(train_dataset, max_samples=10000)
    print(f"Extracted {len(train_feats)} valid feature samples from train dataset.", flush=True)

    # 2. Train Kinematic MLPs
    print("Training Kinematics MLPs...", flush=True)
    class KinematicsMLPGoal(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
            )
        def forward(self, x):
            return self.net(x).view(-1, 4, 2)

    class KinematicsMLPNoGoal(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
            )
        def forward(self, x):
            return self.net(x).view(-1, 4, 2)

    class KinematicsMLPResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 8),
            )
        def forward(self, x):
            return self.net(x).view(-1, 4, 2)

    def compute_cc_traj(v, w):
        pts = []
        for t in time_horizons:
            if abs(w) < 1e-4:
                pts.append([v * t, 0.0])
            else:
                R = v / w
                psi = w * t
                pts.append([R * math.sin(psi), R * (1.0 - math.cos(psi))])
        return np.array(pts, dtype=np.float32)

    X_train_5 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_point"][0], s["target_point"][1]] for s in train_feats], dtype=torch.float32)
    X_train_3 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in train_feats], dtype=torch.float32)
    Y_train = torch.tensor(np.array([s["gt_waypoints"] for s in train_feats]), dtype=torch.float32)

    CC_train = torch.tensor(np.array([compute_cc_traj(s["speed"], s["body_yaw_rate"]) for s in train_feats]), dtype=torch.float32)
    Y_train_residual = Y_train - CC_train

    # Train MLP Goal
    mlp_goal = KinematicsMLPGoal()
    opt_goal = optim.Adam(mlp_goal.parameters(), lr=1e-3)
    loader_goal = DataLoader(torch.utils.data.TensorDataset(X_train_5, Y_train), batch_size=64, shuffle=True)
    for _ in range(15):
        for bx, by in loader_goal:
            opt_goal.zero_grad()
            nn.L1Loss()(mlp_goal(bx), by).backward()
            opt_goal.step()

    # Train MLP NoGoal
    mlp_nogoal = KinematicsMLPNoGoal()
    opt_nogoal = optim.Adam(mlp_nogoal.parameters(), lr=1e-3)
    loader_nogoal = DataLoader(torch.utils.data.TensorDataset(X_train_3, Y_train), batch_size=64, shuffle=True)
    for _ in range(15):
        for bx, by in loader_nogoal:
            opt_nogoal.zero_grad()
            nn.L1Loss()(mlp_nogoal(bx), by).backward()
            opt_nogoal.step()

    # Train MLP Residual
    mlp_resid = KinematicsMLPResidual()
    opt_resid = optim.Adam(mlp_resid.parameters(), lr=1e-3)
    loader_resid = DataLoader(torch.utils.data.TensorDataset(X_train_5, Y_train_residual), batch_size=64, shuffle=True)
    for _ in range(15):
        for bx, by in loader_resid:
            opt_resid.zero_grad()
            nn.L1Loss()(mlp_resid(bx), by).backward()
            opt_resid.step()

    print("Baselines trained. Generating baseline predictions on full val set...", flush=True)
    mlp_goal.eval()
    mlp_nogoal.eval()
    mlp_resid.eval()

    X_val_5 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_point"][0], s["target_point"][1]] for s in val_feats], dtype=torch.float32)
    X_val_3 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in val_feats], dtype=torch.float32)

    with torch.no_grad():
        preds_mlp_goal = mlp_goal(X_val_5).numpy()
        preds_mlp_nogoal = mlp_nogoal(X_val_3).numpy()
        preds_mlp_resid_offsets = mlp_resid(X_val_5).numpy()

    preds_cv = []
    preds_cc = []
    for s in val_feats:
        v = s["speed"]
        w = s["body_yaw_rate"]
        preds_cv.append([[v * t, 0.0] for t in time_horizons])
        preds_cc.append(compute_cc_traj(v, w))

    preds_cv = np.array(preds_cv, dtype=np.float32)
    preds_cc = np.array(preds_cc, dtype=np.float32)
    preds_mlp_resid = preds_cc + preds_mlp_resid_offsets

    # 3. Load STAF Best Model (Epoch 1) and Run Inference on Val Set
    print("Loading STAF best_model.pth (Epoch 1)...", flush=True)
    model = STAFModel(
        embed_dim=256,
        num_heads=4,
        num_fusion_layers=4,
        ff_dim=512,
        queue_length=5,
        num_temporal_layers=2,
        lidar_in_channels=2,
        dropout=0.0,
        num_classes=0,
        pred_len=4,
        use_target_point=True,
        use_ego_speed=True,
        pretrained_backbones=False,
    ).to(device)

    ckpt_path = "/checkpoints/staf_phase1/best_model.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Running STAF inference across {len(val_loader)} batches...", flush=True)

    preds_staf = []
    w_cams = []
    w_lidars = []
    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            rgb = batch["image"].to(device)
            lidar_bev = batch["lidar_bev"].to(device)
            x_kinematic = batch["x_kinematic"].to(device)
            target = batch.get("target_point").to(device)
            spd = batch.get("speed").to(device)
            bs = rgb.size(0)
            model.reset_temporal_state(bs, device)
            outputs = model(rgb, lidar_bev, x_kinematic, target_point=target, ego_speed=spd)
            preds_staf.append(outputs["waypoints"].cpu().numpy())
            w_cams.append(outputs["w_cam"].cpu().numpy())
            w_lidars.append(outputs["w_lidar"].cpu().numpy())

    preds_staf = np.concatenate(preds_staf, axis=0)
    w_cams = np.concatenate(w_cams, axis=0).flatten()
    w_lidars = np.concatenate(w_lidars, axis=0).flatten()

    w_cam_mean = float(np.mean(w_cams))
    w_cam_std = float(np.std(w_cams))
    w_lidar_mean = float(np.mean(w_lidars))
    w_lidar_std = float(np.std(w_lidars))
    print(f"Gating Weights across Full Val Set: w_cam={w_cam_mean:.4f} ± {w_cam_std:.4f}, w_lidar={w_lidar_mean:.4f} ± {w_lidar_std:.4f}", flush=True)

    # 4. Standardized Evaluation Metric Computation
    gts = np.array([s["gt_waypoints"] for s in val_feats], dtype=np.float32)
    steers = np.array([abs(s["steer"]) for s in val_feats])

    n_eval = min(len(gts), len(preds_staf))
    gts = gts[:n_eval]
    steers = steers[:n_eval]
    preds_cv = preds_cv[:n_eval]
    preds_cc = preds_cc[:n_eval]
    preds_mlp_nogoal = preds_mlp_nogoal[:n_eval]
    preds_mlp_goal = preds_mlp_goal[:n_eval]
    preds_mlp_resid = preds_mlp_resid[:n_eval]
    preds_staf = preds_staf[:n_eval]

    all_models = {
        "Constant Velocity": preds_cv,
        "Constant Curvature (omega_body)": preds_cc,
        "Kinematics MLP (No Goal, 3 in)": preds_mlp_nogoal,
        "Kinematics MLP (With Goal, 5 in)": preds_mlp_goal,
        "Kinematics MLP (Residual over CC)": preds_mlp_resid,
        "STAF Best Model (Epoch 1)": preds_staf,
    }

    def compute_standardized_metrics(preds, targets):
        diff = preds - targets  # (N, 4, 2)
        eucl = np.linalg.norm(diff, axis=-1)  # (N, 4)
        lat_err = np.abs(diff[..., 1])        # (N, 4)
        fwd_err = np.abs(diff[..., 0])        # (N, 4)
        l1_sum_horizons = fwd_err + lat_err   # (N, 4): sum of |dx| + |dy| per horizon
        mean_l1_sample = np.mean(l1_sum_horizons, axis=-1)  # (N,): mean L1 over horizons per sample

        return {
            "ade_mean": float(np.mean(eucl)),
            "ade_med": float(np.median(eucl)),
            "fde_mean": float(np.mean(eucl[:, 3])),
            "fde_med": float(np.median(eucl[:, 3])),
            "lat_ade_mean": float(np.mean(lat_err)),
            "lat_ade_med": float(np.median(lat_err)),
            "lat_fde_mean": float(np.mean(lat_err[:, 3])),
            "lat_fde_med": float(np.median(lat_err[:, 3])),
            "mean_l1_mean": float(np.mean(mean_l1_sample)),
            "mean_l1_med": float(np.median(mean_l1_sample)),
            "h_l1_mean": [float(x) for x in np.mean(l1_sum_horizons, axis=0)],
            "h_l1_med": [float(x) for x in np.median(l1_sum_horizons, axis=0)],
        }

    mask_turning = steers > 0.05
    mask_straight = steers <= 0.05

    subgroups = [
        ("ALL (Full Validation Set)", slice(None)),
        ("TURNING FRAMES (|steer| > 0.05)", mask_turning),
        ("STRAIGHT FRAMES (|steer| <= 0.05)", mask_straight),
    ]

    results_table = {}

    for s_name, mask in subgroups:
        sub_n = int(np.sum(mask)) if isinstance(mask, np.ndarray) else n_eval
        sub_pct = sub_n / n_eval * 100
        print(f"\n" + "="*110, flush=True)
        print(f"SUBGROUP: {s_name} | Samples: {sub_n} ({sub_pct:.1f}%)", flush=True)
        print("="*110, flush=True)
        print(f"{'Model':<36} | {'ADE (mean/med) [m]':<20} | {'FDE (mean/med) [m]':<20} | {'Lat-ADE (mean/med)':<20} | {'Mean L1 (sum)':<16}", flush=True)
        print("-"*110, flush=True)

        results_table[s_name] = {}
        for m_name, p in all_models.items():
            metrics = compute_standardized_metrics(p[mask], gts[mask])
            results_table[s_name][m_name] = metrics
            ade_str = f"{metrics['ade_mean']:.4f} / {metrics['ade_med']:.4f}"
            fde_str = f"{metrics['fde_mean']:.4f} / {metrics['fde_med']:.4f}"
            lat_str = f"{metrics['lat_ade_mean']:.4f} / {metrics['lat_ade_med']:.4f}"
            l1_str = f"{metrics['mean_l1_mean']:.4f} / {metrics['mean_l1_med']:.4f}"
            print(f"{m_name:<36} | {ade_str:<20} | {fde_str:<20} | {lat_str:<20} | {l1_str:<16}", flush=True)

        print("="*110, flush=True)

    return {
        "results": results_table,
        "gating": {
            "w_cam_mean": w_cam_mean,
            "w_cam_std": w_cam_std,
            "w_lidar_mean": w_lidar_mean,
            "w_lidar_std": w_lidar_std,
        }
    }

@app.local_entrypoint()
def main():
    print("Launching Comprehensive Full Benchmark Evaluation on Modal...", flush=True)
    res = evaluate_all.remote()
    print("Evaluation Complete!", flush=True)
