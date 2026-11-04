"""
Fast Modal Script to:
1. Evaluate on test set junction frames (N=824) and turning frames in seconds.
2. Extract Forward ADE and Lateral ADE for Junction frames (Vision vs No-Vision vs Baselines).
3. Extract the 10 worst junction error cases and generate junction_10_worst_examples.png.
4. Compute the route-paired bootstrap CI for (Vision - No-Vision) on Overall and Turning ADE.
"""

import os
import sys
import math
import json
import numpy as np
import modal

app = modal.App("staf-fast-junction-analysis")
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
    timeout=600,
)
def run_fast_junction_analysis():
    import sys
    sys.path.insert(0, "/root/STAF")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from staf.data.dataset import CARLASTAFDataset
    from staf.models.staf_model import STAFModel, compute_cc_trajectory_torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    print("Loading test dataset (Town02 & Town05)...", flush=True)
    test_dataset = CARLASTAFDataset(carla_root, split="test", pred_len=4, augment_domain=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Loaded test dataset: {len(test_dataset)} frames across {len(test_loader)} batches.", flush=True)

    # Load STAF Best Model (Phase 1 checkpoint) for full visual reasoning
    ckpt_path = "/checkpoints/staf_phase1/best_model.pth"
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    vision_model = STAFModel(
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
    vision_model.load_state_dict(state_dict, strict=True)
    vision_model.eval()

    # Train fast No-Vision Kinematic Head over CC on training tabular features
    print("Extracting train tabular features for No-Vision baseline...", flush=True)
    train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=4, augment_domain=False)
    
    # Fast tabular extraction (skip image/lidar loading by calling _load_waypoints directly)
    train_feats = []
    for idx in range(min(12000, len(train_dataset))):
        s_info = train_dataset.samples[idx]
        wps, tgt, spd = train_dataset._load_waypoints(s_info)
        if wps is not None:
            train_feats.append({
                "speed": spd,
                "accel": s_info.get("accel", 0.0),
                "yaw_rate": s_info.get("body_yaw_rate", 0.0),
                "target_norm": np.clip(tgt / 20.0, -3.0, 3.0),
                "waypoints": wps,
            })

    X_tr = torch.tensor([[s["speed"], s["accel"], s["yaw_rate"], s["target_norm"][0], s["target_norm"][1]] for s in train_feats], dtype=torch.float32).to(device)
    Y_tr = torch.tensor(np.array([s["waypoints"] for s in train_feats]), dtype=torch.float32).to(device)
    spd_tr = torch.tensor([[s["speed"]] for s in train_feats], dtype=torch.float32).to(device)
    yaw_tr = torch.tensor([[s["yaw_rate"]] for s in train_feats], dtype=torch.float32).to(device)
    CC_tr = compute_cc_trajectory_torch(spd_tr, yaw_tr)
    Y_tr_resid = Y_tr - CC_tr

    novis_head = nn.Sequential(
        nn.Linear(5, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 8),
    ).to(device)
    opt_nv = optim.Adam(novis_head.parameters(), lr=1e-3)
    loader_tr = DataLoader(torch.utils.data.TensorDataset(X_tr, Y_tr_resid), batch_size=64, shuffle=True)
    for _ in range(15):
        for bx, by in loader_tr:
            opt_nv.zero_grad()
            pred = novis_head(bx).view(-1, 4, 2)
            nn.L1Loss()(pred, by).backward()
            opt_nv.step()
    novis_head.eval()
    print("No-Vision head trained.", flush=True)

    # Evaluate on held-out test set
    print("Running evaluation on test set...", flush=True)
    preds_vision = []
    preds_novis = []
    gts = []
    steers = []
    juncs = []
    routes = []
    speeds = []

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            rgb = batch["image"].to(device)
            lidar = batch["lidar_bev"].to(device)
            x_kin = batch["x_kinematic"].to(device)
            target = batch["target_point"].to(device)
            target_norm = batch["target_point_norm"].to(device)
            spd = batch["speed"].to(device)
            accel = batch["accel"].to(device)
            yaw_rate = batch["body_yaw_rate"].to(device)
            wps = batch["waypoints"]

            bs = wps.size(0)
            vision_model.reset_temporal_state(bs, device)
            out_vis = vision_model(
                rgb=rgb,
                lidar_bev=lidar,
                x_kinematic=x_kin,
                target_point=target,
                ego_speed=spd,
            )
            p_vis = out_vis["waypoints"].cpu().numpy()

            # No-vision residual over CC
            cc_traj = compute_cc_trajectory_torch(spd, yaw_rate)
            x_nv = torch.cat([spd, accel, yaw_rate, target_norm], dim=-1)
            nv_resid = novis_head(x_nv).view(-1, 4, 2)
            p_nv = (cc_traj + nv_resid).cpu().numpy()

            preds_vision.append(p_vis)
            preds_novis.append(p_nv)
            gts.append(wps.numpy())
            steers.append(batch["steer"].numpy().flatten())
            juncs.append(batch["junction"].numpy().flatten())
            speeds.append(spd.cpu().numpy().flatten())
            routes.extend(batch.get("route_dir", ["unknown"] * bs))

    preds_vision = np.concatenate(preds_vision, axis=0)
    preds_novis = np.concatenate(preds_novis, axis=0)
    gts = np.concatenate(gts, axis=0)
    steers = np.concatenate(steers, axis=0)
    juncs = np.concatenate(juncs, axis=0)
    speeds = np.concatenate(speeds, axis=0)
    routes = np.array(routes)

    # 1. Junction metrics breakdown
    mask_junc = juncs > 0.5
    n_junc = int(np.sum(mask_junc))
    print(f"Junction frames confirmed: N = {n_junc} ({n_junc / len(gts) * 100:.2f}%)", flush=True)

    diff_nv = preds_novis[mask_junc] - gts[mask_junc]
    diff_vi = preds_vision[mask_junc] - gts[mask_junc]

    nv_junc_ade = float(np.mean(np.linalg.norm(diff_nv, axis=-1)))
    nv_junc_fwd = float(np.mean(np.abs(diff_nv[..., 0])))
    nv_junc_lat = float(np.mean(np.abs(diff_nv[..., 1])))

    vi_junc_ade = float(np.mean(np.linalg.norm(diff_vi, axis=-1)))
    vi_junc_fwd = float(np.mean(np.abs(diff_vi[..., 0])))
    vi_junc_lat = float(np.mean(np.abs(diff_vi[..., 1])))

    print(f"Junction No-Vision: ADE={nv_junc_ade:.4f} m | Fwd-ADE={nv_junc_fwd:.4f} m | Lat-ADE={nv_junc_lat:.4f} m", flush=True)
    print(f"Junction Vision:    ADE={vi_junc_ade:.4f} m | Fwd-ADE={vi_junc_fwd:.4f} m | Lat-ADE={vi_junc_lat:.4f} m", flush=True)

    # 2. Top 10 worst junction examples by Vision error
    junc_indices = np.where(mask_junc)[0]
    vi_junc_ades = np.mean(np.linalg.norm(diff_vi, axis=-1), axis=-1)
    worst_order = np.argsort(vi_junc_ades)[::-1][:10]
    worst_10_indices = junc_indices[worst_order]

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    worst_data = []
    for rank, orig_idx in enumerate(worst_10_indices):
        ax = axes[rank]
        gt = gts[orig_idx]
        nv = preds_novis[orig_idx]
        vi = preds_vision[orig_idx]

        v_ade = float(np.mean(np.linalg.norm(vi - gt, axis=-1)))
        nv_ade = float(np.mean(np.linalg.norm(nv - gt, axis=-1)))

        ax.plot(0, 0, marker="o", color="black", markersize=8, label="Ego (0,0)")
        ax.plot(gt[:, 1], gt[:, 0], "-o", color="#2ca02c", linewidth=2.5, markersize=6, label="Ground Truth")
        ax.plot(nv[:, 1], nv[:, 0], "--s", color="#1f77b4", linewidth=2.0, markersize=5, label=f"No-Vision ({nv_ade:.2f}m)")
        ax.plot(vi[:, 1], vi[:, 0], ":^", color="#d62728", linewidth=2.0, markersize=5, label=f"Vision ({v_ade:.2f}m)")

        r_name = os.path.basename(routes[orig_idx])
        ax.set_title(f"Rank {rank+1} (Frame {orig_idx})\n{r_name} | Steer={steers[orig_idx]:.2f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Lateral (Left +y) [m]", fontsize=8)
        ax.set_ylabel("Forward (+x) [m]", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
        if rank == 0:
            ax.legend(loc="upper left", fontsize=8)

        worst_data.append({
            "rank": rank + 1,
            "frame_idx": int(orig_idx),
            "route": str(r_name),
            "steer": float(steers[orig_idx]),
            "speed": float(speeds[orig_idx]),
            "vision_ade": v_ade,
            "novis_ade": nv_ade,
            "fwd_error_vision": float(np.mean(np.abs(vi[:, 0] - gt[:, 0]))),
            "lat_error_vision": float(np.mean(np.abs(vi[:, 1] - gt[:, 1]))),
        })

    plt.suptitle("Top 10 Worst Junction Error Cases (Vision vs No-Vision vs Ground Truth in Ego Frame)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plot_path = "/checkpoints/junction_10_worst_examples.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {plot_path}", flush=True)

    # 3. Route-paired bootstrap CI: Delta = Vision - No-Vision
    unique_routes = np.unique(routes)
    rng = np.random.RandomState(42)

    err_nv_sample = np.linalg.norm(preds_novis - gts, axis=-1).mean(axis=-1)
    err_vi_sample = np.linalg.norm(preds_vision - gts, axis=-1).mean(axis=-1)

    mask_turning = np.abs(steers) > 0.05

    delta_overall_boot = []
    delta_turning_boot = []

    for _ in range(5000):
        sampled_r = rng.choice(unique_routes, size=len(unique_routes), replace=True)
        idx_list = []
        for r in sampled_r:
            idx_list.extend(np.where(routes == r)[0])
        idx_arr = np.array(idx_list)

        m_nv = np.mean(err_nv_sample[idx_arr])
        m_vi = np.mean(err_vi_sample[idx_arr])
        delta_overall_boot.append(m_vi - m_nv)

        t_arr = idx_arr[mask_turning[idx_arr]]
        if len(t_arr) > 0:
            m_nv_t = np.mean(err_nv_sample[t_arr])
            m_vi_t = np.mean(err_vi_sample[t_arr])
            delta_turning_boot.append(m_vi_t - m_nv_t)

    ci_delta_overall = [float(np.percentile(delta_overall_boot, 2.5)), float(np.percentile(delta_overall_boot, 97.5))]
    ci_delta_turning = [float(np.percentile(delta_turning_boot, 2.5)), float(np.percentile(delta_turning_boot, 97.5))]

    print(f"Route-Paired Delta (Vision - No-Vision) Overall: mean = {np.mean(delta_overall_boot):.4f} m, 95% CI = {ci_delta_overall}", flush=True)
    print(f"Route-Paired Delta (Vision - No-Vision) Turning: mean = {np.mean(delta_turning_boot):.4f} m, 95% CI = {ci_delta_turning}", flush=True)

    checkpoint_volume.commit()

    return {
        "n_junction": n_junc,
        "junction_novis": {"ade": nv_junc_ade, "fwd_ade": nv_junc_fwd, "lat_ade": nv_junc_lat},
        "junction_vision": {"ade": vi_junc_ade, "fwd_ade": vi_junc_fwd, "lat_ade": vi_junc_lat},
        "worst_10_junctions": worst_data,
        "ci_delta_overall": ci_delta_overall,
        "ci_delta_turning": ci_delta_turning,
        "mean_delta_overall": float(np.mean(delta_overall_boot)),
        "mean_delta_turning": float(np.mean(delta_turning_boot)),
    }

@app.local_entrypoint()
def main():
    print("Launching Fast Junction Analysis on Modal...", flush=True)
    res = run_fast_junction_analysis.remote()
    print("Finished! Output:", res)
    with open("fast_junction_results.json", "w") as f:
        json.dump(res, f, indent=2)
