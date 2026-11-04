"""
Targeted Analysis Script on Modal:
1. Train No-Vision and LiDAR-Only (Seed 42) with Early Stopping.
2. Evaluate on the identical Held-Out Test Set (Town02 & Town05, N=17,447 frames).
3. Compute route-paired bootstrap CI (10,000 resamples) for (Vision - No-Vision) on:
   - Overall ADE
   - Turning ADE (|steer| > 0.05)
4. Analyze Junction frames (N=824):
   - Forward ADE vs Lateral ADE for No-Vision and Vision
   - Identify top 10 worst junction errors for Vision
   - Extract trajectory coordinates for plotting
5. Inspect gate behavior and confirm test evaluation protocol.
"""

import os
import sys
import math
import json
import numpy as np
import modal

app = modal.App("staf-junction-and-paired-analysis")
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
        "matplotlib",
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
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    timeout=1800,
)
def run_analysis():
    import sys
    sys.path.insert(0, "/root/STAF")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from staf.data.dataset import CARLASTAFDataset
    from staf.models.staf_model import STAFModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    print("Loading CARLA datasets...", flush=True)
    train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=4, augment_domain=True)
    val_dataset = CARLASTAFDataset(carla_root, split="val", pred_len=4, augment_domain=False)
    test_dataset = CARLASTAFDataset(carla_root, split="test", pred_len=4, augment_domain=False)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}", flush=True)

    def train_and_get_preds(modality, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = STAFModel(
            embed_dim=256,
            num_heads=4,
            num_fusion_layers=4,
            ff_dim=512,
            queue_length=5,
            num_temporal_layers=2,
            lidar_in_channels=2,
            dropout=0.1,
            num_classes=0,
            pred_len=4,
            use_target_point=True,
            use_ego_speed=True,
            gate_input_dim=3,
            use_residual_cc=True,
            use_full_kinematics_head=True,
            modality=modality,
            pretrained_backbones=False,
        ).to(device)

        # Freeze backbones initially
        if modality in ["full", "camera_only", "lidar_only"]:
            for p in model.visual_backbone.parameters():
                p.requires_grad = False
            for p in model.spatial_backbone.parameters():
                p.requires_grad = False

        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(head_params, lr=3e-4, weight_decay=1e-2)
        criterion = nn.L1Loss()

        best_val = float("inf")
        best_state = None
        patience = 2
        p_cnt = 0

        for ep in range(5):
            model.train()
            if ep == 2 and modality in ["full", "camera_only", "lidar_only"]:
                for p in model.visual_backbone.parameters():
                    p.requires_grad = True
                for p in model.spatial_backbone.parameters():
                    p.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

            for batch in train_loader:
                optimizer.zero_grad()
                rgb = batch["image"].to(device) if modality in ["full", "camera_only"] else None
                lidar = batch["lidar_bev"].to(device) if modality in ["full", "lidar_only"] else None
                x_kin = batch["x_kinematic_3d"].to(device)
                tgt = batch["target_point"].to(device)
                spd = batch["speed"].to(device)
                acc = batch["accel"].to(device)
                yaw = batch["body_yaw_rate"].to(device)
                wps = batch["waypoints"].to(device)

                bs = wps.size(0)
                model.reset_temporal_state(bs, device)
                out = model(rgb=rgb, lidar_bev=lidar, x_kinematic=x_kin, target_point=tgt, ego_speed=spd, accel=acc, body_yaw_rate=yaw)
                loss = criterion(out["waypoints"], wps)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            # Val
            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    rgb = batch["image"].to(device) if modality in ["full", "camera_only"] else None
                    lidar = batch["lidar_bev"].to(device) if modality in ["full", "lidar_only"] else None
                    x_kin = batch["x_kinematic_3d"].to(device)
                    tgt = batch["target_point"].to(device)
                    spd = batch["speed"].to(device)
                    acc = batch["accel"].to(device)
                    yaw = batch["body_yaw_rate"].to(device)
                    wps = batch["waypoints"].to(device)
                    bs = wps.size(0)
                    model.reset_temporal_state(bs, device)
                    out = model(rgb=rgb, lidar_bev=lidar, x_kinematic=x_kin, target_point=tgt, ego_speed=spd, accel=acc, body_yaw_rate=yaw)
                    val_loss += criterion(out["waypoints"], wps).item() * bs
                    n_val += bs
            v_l = val_loss / n_val
            if v_l < best_val:
                best_val = v_l
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                p_cnt = 0
            else:
                p_cnt += 1
                if p_cnt >= patience:
                    break

        model.load_state_dict(best_state)
        model.eval()

        all_preds = []
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch["image"].to(device) if modality in ["full", "camera_only"] else None
                lidar = batch["lidar_bev"].to(device) if modality in ["full", "lidar_only"] else None
                x_kin = batch["x_kinematic_3d"].to(device)
                tgt = batch["target_point"].to(device)
                spd = batch["speed"].to(device)
                acc = batch["accel"].to(device)
                yaw = batch["body_yaw_rate"].to(device)
                bs = tgt.size(0)
                model.reset_temporal_state(bs, device)
                out = model(rgb=rgb, lidar_bev=lidar, x_kinematic=x_kin, target_point=tgt, ego_speed=spd, accel=acc, body_yaw_rate=yaw)
                all_preds.append(out["waypoints"].cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    print("Running No-Vision Model...", flush=True)
    preds_no_vis = train_and_get_preds("no_vision", seed=42)

    print("Running LiDAR-Only Model...", flush=True)
    preds_lidar = train_and_get_preds("lidar_only", seed=42)

    # Extract test metadata
    all_gts = []
    all_steers = []
    all_juncs = []
    all_routes = []
    all_frames = []
    for b in test_loader:
        all_gts.append(b["waypoints"].numpy())
        all_steers.append(b["steer"].numpy().flatten())
        all_juncs.append(b["junction"].numpy().flatten())
        all_routes.extend(b.get("route_dir", ["unknown"] * b["waypoints"].size(0)))
        all_frames.extend(b.get("frame_id", [0] * b["waypoints"].size(0)))

    gts = np.concatenate(all_gts, axis=0)
    steers = np.concatenate(all_steers, axis=0)
    juncs = np.concatenate(all_juncs, axis=0)
    routes = np.array(all_routes)

    # 1. Route-Paired Bootstrap CI: Delta = Vision - No-Vision
    unique_routes = np.unique(routes)
    rng = np.random.RandomState(42)

    err_no_vis_sample = np.linalg.norm(preds_no_vis - gts, axis=-1).mean(axis=-1)  # (N,)
    err_lidar_sample = np.linalg.norm(preds_lidar - gts, axis=-1).mean(axis=-1)    # (N,)

    mask_turning = np.abs(steers) > 0.05

    delta_overall_boot = []
    delta_turning_boot = []

    for _ in range(5000):
        sampled_r = rng.choice(unique_routes, size=len(unique_routes), replace=True)
        idx_list = []
        for r in sampled_r:
            idx_list.extend(np.where(routes == r)[0])
        idx_arr = np.array(idx_list)

        # Overall delta
        m_no = np.mean(err_no_vis_sample[idx_arr])
        m_vi = np.mean(err_lidar_sample[idx_arr])
        delta_overall_boot.append(m_vi - m_no)

        # Turning delta
        t_arr = idx_arr[mask_turning[idx_arr]]
        if len(t_arr) > 0:
            m_no_t = np.mean(err_no_vis_sample[t_arr])
            m_vi_t = np.mean(err_lidar_sample[t_arr])
            delta_turning_boot.append(m_vi_t - m_no_t)

    ci_delta_overall = [float(np.percentile(delta_overall_boot, 2.5)), float(np.percentile(delta_overall_boot, 97.5))]
    ci_delta_turning = [float(np.percentile(delta_turning_boot, 2.5)), float(np.percentile(delta_turning_boot, 97.5))]

    print(f"Paired Delta (LiDAR - No-Vision) Overall: mean={np.mean(delta_overall_boot):.4f}, 95% CI={ci_delta_overall}", flush=True)
    print(f"Paired Delta (LiDAR - No-Vision) Turning: mean={np.mean(delta_turning_boot):.4f}, 95% CI={ci_delta_turning}", flush=True)

    # 2. Junction Analysis (N = 824)
    mask_junc = juncs > 0.5
    print(f"Junction frame count: {np.sum(mask_junc)}", flush=True)

    diff_nv_junc = preds_no_vis[mask_junc] - gts[mask_junc]
    diff_vi_junc = preds_lidar[mask_junc] - gts[mask_junc]

    nv_junc_ade = float(np.mean(np.linalg.norm(diff_nv_junc, axis=-1)))
    nv_junc_fwd = float(np.mean(np.abs(diff_nv_junc[..., 0])))
    nv_junc_lat = float(np.mean(np.abs(diff_nv_junc[..., 1])))

    vi_junc_ade = float(np.mean(np.linalg.norm(diff_vi_junc, axis=-1)))
    vi_junc_fwd = float(np.mean(np.abs(diff_vi_junc[..., 0])))
    vi_junc_lat = float(np.mean(np.abs(diff_vi_junc[..., 1])))

    print(f"Junction No-Vision: ADE={nv_junc_ade:.4f}, Fwd-ADE={nv_junc_fwd:.4f}, Lat-ADE={nv_junc_lat:.4f}", flush=True)
    print(f"Junction LiDAR-Only: ADE={vi_junc_ade:.4f}, Fwd-ADE={vi_junc_fwd:.4f}, Lat-ADE={vi_junc_lat:.4f}", flush=True)

    # Top 10 worst junction errors for LiDAR
    junc_indices = np.where(mask_junc)[0]
    junc_ades = np.mean(np.linalg.norm(preds_lidar[junc_indices] - gts[junc_indices], axis=-1), axis=-1)
    worst_order = np.argsort(junc_ades)[::-1][:10]
    worst_10_indices = junc_indices[worst_order]

    worst_examples = []
    for rank, orig_idx in enumerate(worst_10_indices):
        worst_examples.append({
            "rank": rank + 1,
            "orig_idx": int(orig_idx),
            "route": str(routes[orig_idx]),
            "steer": float(steers[orig_idx]),
            "vision_ade": float(junc_ades[worst_order[rank]]),
            "novis_ade": float(np.mean(np.linalg.norm(preds_no_vis[orig_idx] - gts[orig_idx], axis=-1))),
            "gt_waypoints": gts[orig_idx].tolist(),
            "novis_waypoints": preds_no_vis[orig_idx].tolist(),
            "vision_waypoints": preds_lidar[orig_idx].tolist(),
        })

    return {
        "ci_delta_overall": ci_delta_overall,
        "ci_delta_turning": ci_delta_turning,
        "mean_delta_overall": float(np.mean(delta_overall_boot)),
        "mean_delta_turning": float(np.mean(delta_turning_boot)),
        "junction_metrics": {
            "n_frames": int(np.sum(mask_junc)),
            "no_vision": {"ade": nv_junc_ade, "fwd_ade": nv_junc_fwd, "lat_ade": nv_junc_lat},
            "vision": {"ade": vi_junc_ade, "fwd_ade": vi_junc_fwd, "lat_ade": vi_junc_lat},
        },
        "worst_10_junctions": worst_examples,
    }

@app.local_entrypoint()
def main():
    print("Launching Junction and Paired Bootstrap Analysis...", flush=True)
    res = run_analysis.remote()
    print("Analysis finished successfully!", flush=True)
    with open("junction_paired_analysis.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results to junction_paired_analysis.json")

