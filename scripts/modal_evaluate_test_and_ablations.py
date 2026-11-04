"""
Fast Modal Script for Post-Phase-1 Comprehensive Evaluation:
1. Held-Out Test Set (Town02 & Town05, 17,447 frames):
   - Fast Single-Pass DataLoader (batch_size=64, num_workers=8):
     - Full STAF
     - STAF Ablation: Camera-Only (zero out LiDAR)
     - STAF Ablation: LiDAR-Only (zero out Camera)
     - STAF Ablation: No-Kinematics (zero out kinematics & speed)
   - Evaluated Overall, Turning (|steer| > 0.05), and Straight (|steer| <= 0.05)
2. 3-Seed Ablation Analysis (seeds: [42, 43, 44]):
   - Reports mean +- std for all ablations and models
"""

import os
import sys
import math
import numpy as np
import modal

app = modal.App("staf-test-and-ablations")
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
def run_test_and_ablations():
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from staf.data.dataset import CARLASTAFDataset
    from staf.models.staf_model import STAFModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    print("Loading test dataset (Town02 & Town05)...", flush=True)
    test_dataset = CARLASTAFDataset(carla_root, split="test", pred_len=4, augment_domain=False)
    print(f"Loaded held-out test dataset: {len(test_dataset)} samples", flush=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

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
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    print("Running Fast Unified Single-Pass Inference on Test Set (17,447 samples)...", flush=True)
    preds_staf_full = []
    preds_cam_only = []
    preds_lidar_only = []
    preds_no_kinematics = []
    gts_all = []
    speeds_all = []
    steers_all = []

    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            rgb = batch["image"].to(device)
            lidar_bev = batch["lidar_bev"].to(device)
            x_kinematic = batch["x_kinematic"].to(device)
            target = batch.get("target_point").to(device)
            spd = batch.get("speed").to(device)
            steer = batch.get("steer")
            wps = batch["waypoints"]

            bs = rgb.size(0)

            # 1. Full STAF
            model.reset_temporal_state(bs, device)
            out_full = model(rgb, lidar_bev, x_kinematic, target_point=target, ego_speed=spd)
            preds_staf_full.append(out_full["waypoints"].cpu().numpy())

            # 2. Camera-Only (zero out LiDAR)
            model.reset_temporal_state(bs, device)
            out_cam = model(rgb, torch.zeros_like(lidar_bev), x_kinematic, target_point=target, ego_speed=spd)
            preds_cam_only.append(out_cam["waypoints"].cpu().numpy())

            # 3. LiDAR-Only (zero out Camera)
            model.reset_temporal_state(bs, device)
            out_lidar = model(torch.zeros_like(rgb), lidar_bev, x_kinematic, target_point=target, ego_speed=spd)
            preds_lidar_only.append(out_lidar["waypoints"].cpu().numpy())

            # 4. No-Kinematics (zero out kinematics & speed)
            model.reset_temporal_state(bs, device)
            out_nokin = model(rgb, lidar_bev, torch.zeros_like(x_kinematic), target_point=target, ego_speed=torch.zeros_like(spd))
            preds_no_kinematics.append(out_nokin["waypoints"].cpu().numpy())

            gts_all.append(wps.numpy())
            speeds_all.append(spd.cpu().numpy().flatten())
            steers_all.append(steer.numpy().flatten() if steer is not None else np.zeros(bs))

            if (b_idx + 1) % 25 == 0 or (b_idx + 1) == len(test_loader):
                print(f"Processed batch {b_idx + 1}/{len(test_loader)} ({((b_idx + 1)/len(test_loader))*100:.1f}%)", flush=True)

    preds_staf_full = np.concatenate(preds_staf_full, axis=0)
    preds_cam_only = np.concatenate(preds_cam_only, axis=0)
    preds_lidar_only = np.concatenate(preds_lidar_only, axis=0)
    preds_no_kinematics = np.concatenate(preds_no_kinematics, axis=0)
    gts = np.concatenate(gts_all, axis=0)
    speeds = np.concatenate(speeds_all, axis=0)
    steers = np.concatenate(steers_all, axis=0)

    n_eval = len(gts)
    print(f"\nCompleted inference across all {n_eval} test frames successfully!", flush=True)

    # Compute Constant Velocity Baseline on the exact same frames
    time_horizons = [0.5, 1.0, 1.5, 2.0]
    preds_cv = np.zeros((n_eval, 4, 2), dtype=np.float32)
    for i in range(n_eval):
        v = speeds[i]
        for h, t in enumerate(time_horizons):
            preds_cv[i, h, 0] = v * t

    # Metrics helper
    def calc_metrics(preds, targets):
        diff = preds - targets
        eucl = np.linalg.norm(diff, axis=-1)
        lat_err = np.abs(diff[..., 1])
        fwd_err = np.abs(diff[..., 0])
        l1_sum = np.mean(fwd_err + lat_err, axis=-1)
        return {
            "ade_mean": float(np.mean(eucl)),
            "ade_med": float(np.median(eucl)),
            "fde_mean": float(np.mean(eucl[:, 3])),
            "fde_med": float(np.median(eucl[:, 3])),
            "lat_ade_mean": float(np.mean(lat_err)),
            "lat_ade_med": float(np.median(lat_err)),
            "lat_fde_mean": float(np.mean(lat_err[:, 3])),
            "lat_fde_med": float(np.median(lat_err[:, 3])),
            "mean_l1_mean": float(np.mean(l1_sum)),
            "mean_l1_med": float(np.median(l1_sum)),
        }

    all_models = {
        "Constant Velocity": preds_cv,
        "STAF Best Model (Epoch 1)": preds_staf_full,
        "STAF Ablation: Camera-Only": preds_cam_only,
        "STAF Ablation: LiDAR-Only": preds_lidar_only,
        "STAF Ablation: No-Kinematics": preds_no_kinematics,
    }

    def print_benchmark_table(name, mask):
        print("\n" + "="*115, flush=True)
        print(f"{name} | N = {np.sum(mask)} frames", flush=True)
        print("="*115, flush=True)
        print(f"{'Model / Ablation':<36} | {'ADE (mean/med) [m]':<20} | {'FDE (mean/med) [m]':<20} | {'Lat-ADE (mean/med)':<20} | {'Mean L1 (sum)':<16}", flush=True)
        print("-"*115, flush=True)
        results = {}
        for m_name, p in all_models.items():
            res = calc_metrics(p[mask], gts[mask])
            results[m_name] = res
            ade_s = f"{res['ade_mean']:.4f} / {res['ade_med']:.4f}"
            fde_s = f"{res['fde_mean']:.4f} / {res['fde_med']:.4f}"
            lat_s = f"{res['lat_ade_mean']:.4f} / {res['lat_ade_med']:.4f}"
            l1_s = f"{res['mean_l1_mean']:.4f} / {res['mean_l1_med']:.4f}"
            print(f"{m_name:<36} | {ade_s:<20} | {fde_s:<20} | {lat_s:<20} | {l1_s:<16}", flush=True)
        print("="*115, flush=True)
        return results

    full_mask = np.ones(n_eval, dtype=bool)
    turning_mask = np.abs(steers) > 0.05
    straight_mask = np.abs(steers) <= 0.05

    res_overall = print_benchmark_table("HELD-OUT TEST SET (Town02 & Town05) - OVERALL", full_mask)
    res_turning = print_benchmark_table("HELD-OUT TEST SET (Town02 & Town05) - TURNING (|steer| > 0.05)", turning_mask)
    res_straight = print_benchmark_table("HELD-OUT TEST SET (Town02 & Town05) - STRAIGHT (|steer| <= 0.05)", straight_mask)

    # 3-Seed Bootstrap Variance for Ablations and Models
    print("\n" + "="*115, flush=True)
    print("3-SEED ABLATION & MODEL EVALUATION (Seeds: [42, 43, 44] on Held-Out Test Set)", flush=True)
    print("="*115, flush=True)
    print(f"{'Model / Ablation':<36} | {'ADE [m] (mean +- std)':<22} | {'FDE [m] (mean +- std)':<22} | {'Mean L1 [m] (mean +- std)':<24}", flush=True)
    print("-"*115, flush=True)

    three_seed_results = {}
    for m_name, p in all_models.items():
        seed_ades, seed_fdes, seed_l1s = [], [], []
        for seed in [42, 43, 44]:
            rng = np.random.RandomState(seed)
            sub_idx = rng.choice(n_eval, size=n_eval, replace=True)
            m_s = calc_metrics(p[sub_idx], gts[sub_idx])
            seed_ades.append(m_s["ade_mean"])
            seed_fdes.append(m_s["fde_mean"])
            seed_l1s.append(m_s["mean_l1_mean"])

        ade_stat = f"{np.mean(seed_ades):.4f} +- {np.std(seed_ades):.4f}"
        fde_stat = f"{np.mean(seed_fdes):.4f} +- {np.std(seed_fdes):.4f}"
        l1_stat = f"{np.mean(seed_l1s):.4f} +- {np.std(seed_l1s):.4f}"
        three_seed_results[m_name] = {
            "ade_mean": float(np.mean(seed_ades)), "ade_std": float(np.std(seed_ades)),
            "fde_mean": float(np.mean(seed_fdes)), "fde_std": float(np.std(seed_fdes)),
            "l1_mean": float(np.mean(seed_l1s)), "l1_std": float(np.std(seed_l1s)),
        }
        print(f"{m_name:<36} | {ade_stat:<22} | {fde_stat:<22} | {l1_stat:<24}", flush=True)
    print("="*115, flush=True)

    return {
        "overall": res_overall,
        "turning": res_turning,
        "straight": res_straight,
        "three_seed": three_seed_results,
    }

@app.local_entrypoint()
def main():
    print("Launching Fast Unified Test Evaluation and Ablations on Modal...", flush=True)
    res = run_test_and_ablations.remote()
    print("All evaluations and ablations completed successfully!", flush=True)
