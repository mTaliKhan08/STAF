"""
Modal Retraining and Evaluation Suite for STAF v2.

Implements:
- STEP 1: Temporal memory reset strictly at route boundaries.
          RouteOrderedBatchSampler (shuffling routes, not frames).
- STEP 2: gate_mode in {"learned", "constant"}.
          In "constant" mode, w_cam = w_lidar = 0.5.
- STEP 3: Training runs:
          Group 1: A = full, learned gate, queue 5; B = full, constant gate, queue 5.
          Group 2: C = full, learned gate, queue 1.
          Group 3: D = camera_only, E = lidar_only, F = no_vision, queue 5.
          Seeds: 42, 43, 44.
- Output: results/v2_retrained_variants/{run_id}.json with config JSON beside each run.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal
import os
import json
import numpy as np

app = modal.App("staf-v2-training-suite")

data_volume = modal.Volume.from_name("staf-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("staf-checkpoints", create_if_missing=True)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/root/STAF"):
    staf_local_path = "/root/STAF"
else:
    staf_candidates = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../staf")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
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
    max_containers=2,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,
)
def train_and_eval_v2(run_config: dict) -> dict:
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from staf.data.dataset import CARLASTAFDataset, RouteOrderedBatchSampler
    from staf.models.staf_model import STAFModel

    variant = run_config["variant"]
    gate_mode = run_config["gate_mode"]
    queue_length = run_config["queue_length"]
    seed = run_config["seed"]
    run_id = run_config["run_id"]

    # Check cache on Modal Volume
    os.makedirs("/checkpoints/v2_metrics", exist_ok=True)
    cache_path = f"/checkpoints/v2_metrics/{run_id}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            print(f"[{run_id.upper()}] Loaded existing cached results from {cache_path}", flush=True)
            return cached
        except Exception as e:
            print(f"[{run_id.upper()}] Cache corrupted ({e}), retraining...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[{run_id.upper()}] Loading datasets...", flush=True)
    train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=4, augment_domain=True)
    val_dataset = CARLASTAFDataset(carla_root, split="val", pred_len=4, augment_domain=False)
    test_dataset = CARLASTAFDataset(carla_root, split="test", pred_len=4, augment_domain=False)

    route_batch_size = 8
    accum_steps = 8

    train_sampler = RouteOrderedBatchSampler(train_dataset, batch_size=route_batch_size, shuffle=True, seed=seed, drop_last=True)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)

    # For validation & testing: sequential route-by-route iteration (batch_size=1)
    # Guarantees EXACT unpadded frame counts (5,816 val, 17,447 test) with zero repeat padding
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[{run_id.upper()}] Initializing STAF Residual Model (modality={variant}, gate_mode={gate_mode}, queue_len={queue_length})...", flush=True)
    model = STAFModel(
        embed_dim=256,
        pred_len=4,
        modality=variant,
        gate_input_dim=3,
        gate_mode=gate_mode,
        queue_length=queue_length,
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    ).to(device)

    def get_optimizer(freeze_backbones: bool):
        backbone_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                if freeze_backbones:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    backbone_params.append(param)
            else:
                param.requires_grad = True
                head_params.append(param)

        param_groups = [{"params": head_params, "lr": 1e-3, "weight_decay": 1e-2}]
        if not freeze_backbones and len(backbone_params) > 0:
            param_groups.append({"params": backbone_params, "lr": 1e-5, "weight_decay": 1e-2})
        return optim.AdamW(param_groups)

    max_epochs = 5
    patience = 2
    best_val_ade = float("inf")
    best_weights = None
    patience_cnt = 0
    epoch_logs = []

    optimizer = get_optimizer(freeze_backbones=True)

    print(f"[{run_id.upper()}] Starting Route-Ordered training with early stopping on Val ADE...", flush=True)
    for epoch in range(max_epochs):
        train_sampler.set_epoch(epoch)
        if epoch == 2:
            print(f"[{run_id.upper()}] Epoch {epoch}: Unfreezing backbones at LR 1e-5", flush=True)
            optimizer = get_optimizer(freeze_backbones=False)

        model.train()
        train_l1 = 0.0
        n_train = 0
        optimizer.zero_grad()
        current_routes = None
        step_in_route = 0

        for batch in train_loader:
            routes = batch.get("route_id", [])
            bs = batch["waypoints"].size(0)

            # Reset temporal queue ONLY at route boundaries
            if routes != current_routes:
                model.reset_temporal_state(bs, device)
                current_routes = routes
                step_in_route = 0

            rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
            lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
            x_kin = batch["x_kinematic_3d"].to(device)
            target = batch["target_point"].to(device)
            spd = batch["speed"].to(device)
            accel = batch["accel"].to(device)
            yaw_rate = batch["body_yaw_rate"].to(device)
            wps = batch["waypoints"].to(device)

            out = model(
                rgb=rgb,
                lidar_bev=lidar_bev,
                x_kinematic=x_kin,
                target_point=target,
                ego_speed=spd,
                accel=accel,
                body_yaw_rate=yaw_rate,
            )

            pred_wps = out["waypoints"]
            loss = nn.L1Loss()(pred_wps, wps)
            (loss / accum_steps).backward()

            step_in_route += 1
            if step_in_route % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            train_l1 += loss.item() * bs
            n_train += bs

        # Final step flush if leftover gradients
        if step_in_route % accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        train_loss = train_l1 / max(1, n_train)

        # Validation pass
        model.eval()
        val_ades = []
        val_fdes = []
        val_w_cam = []
        val_w_lidar = []
        current_routes = None

        with torch.no_grad():
            for batch in val_loader:
                routes = batch.get("route_id", [])
                bs = batch["waypoints"].size(0)

                if routes != current_routes:
                    model.reset_temporal_state(bs, device)
                    current_routes = routes

                rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
                lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
                x_kin = batch["x_kinematic_3d"].to(device)
                target = batch["target_point"].to(device)
                spd = batch["speed"].to(device)
                accel = batch["accel"].to(device)
                yaw_rate = batch["body_yaw_rate"].to(device)
                wps = batch["waypoints"].to(device)

                out = model(
                    rgb=rgb,
                    lidar_bev=lidar_bev,
                    x_kinematic=x_kin,
                    target_point=target,
                    ego_speed=spd,
                    accel=accel,
                    body_yaw_rate=yaw_rate,
                )

                pred_wps = out["waypoints"]
                # ADE: mean Euclidean distance across 4 predicted steps
                ade_step = torch.norm(pred_wps - wps, dim=-1).mean(dim=-1)
                fde_step = torch.norm(pred_wps[:, -1, :] - wps[:, -1, :], dim=-1)

                val_ades.extend(ade_step.cpu().numpy().tolist())
                val_fdes.extend(fde_step.cpu().numpy().tolist())

                if out.get("w_cam") is not None:
                    val_w_cam.extend(out["w_cam"].squeeze(-1).cpu().numpy().tolist())
                if out.get("w_lidar") is not None:
                    val_w_lidar.extend(out["w_lidar"].squeeze(-1).cpu().numpy().tolist())

        val_ade = float(np.mean(val_ades))
        val_fde = float(np.mean(val_fdes))
        cam_mean = float(np.mean(val_w_cam)) if val_w_cam else 0.5
        cam_std = float(np.std(val_w_cam)) if val_w_cam else 0.0
        lidar_mean = float(np.mean(val_w_lidar)) if val_w_lidar else 0.5
        lidar_std = float(np.std(val_w_lidar)) if val_w_lidar else 0.0

        epoch_stat = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_ade": val_ade,
            "val_fde": val_fde,
            "gate_cam_mean": cam_mean,
            "gate_cam_std": cam_std,
            "gate_lidar_mean": lidar_mean,
            "gate_lidar_std": lidar_std,
            "queue_length": queue_length,
        }
        epoch_logs.append(epoch_stat)
        print(f"[{run_id.upper()}] Epoch {epoch}: Train Loss = {train_loss:.4f} m | Val ADE = {val_ade:.4f} m, Val FDE = {val_fde:.4f} m | Gate w_cam={cam_mean:.3f}+-{cam_std:.3f}", flush=True)

        if val_ade < best_val_ade:
            best_val_ade = val_ade
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[{run_id.upper()}] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    # Save best checkpoint locally on volume
    os.makedirs("/checkpoints/v2_checkpoints", exist_ok=True)
    ckpt_save_path = f"/checkpoints/v2_checkpoints/{run_id}_best.pth"
    torch.save({"state_dict": best_weights, "config": run_config, "val_ade": best_val_ade}, ckpt_save_path)
    checkpoint_volume.commit()

    # Load best checkpoint for held-out test evaluation
    model.load_state_dict(best_weights)
    model.eval()

    print(f"[{run_id.upper()}] Evaluating on Held-Out Test Set (Town02 & Town05)...", flush=True)
    all_preds = []
    all_gts = []
    all_steers = []
    all_speeds = []
    all_accels = []
    all_juncs = []
    all_veh_hazs = []
    all_light_hazs = []
    all_stop_hazs = []
    all_route_ids = []
    all_w_cam = []
    all_w_lidar = []
    all_ego_mats = []

    current_routes = None
    with torch.no_grad():
        for batch in test_loader:
            routes = batch.get("route_id", [])
            bs = batch["waypoints"].size(0)

            if routes != current_routes:
                model.reset_temporal_state(bs, device)
                current_routes = routes

            rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
            lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
            x_kin = batch["x_kinematic_3d"].to(device)
            target = batch["target_point"].to(device)
            spd = batch["speed"].to(device)
            accel = batch["accel"].to(device)
            yaw_rate = batch["body_yaw_rate"].to(device)
            wps = batch["waypoints"]

            out = model(
                rgb=rgb,
                lidar_bev=lidar_bev,
                x_kinematic=x_kin,
                target_point=target,
                ego_speed=spd,
                accel=accel,
                body_yaw_rate=yaw_rate,
            )

            all_preds.append(out["waypoints"].cpu().numpy())
            all_gts.append(wps.numpy())
            all_steers.append(batch["steer"].numpy().flatten())
            all_speeds.append(batch["speed"].numpy().flatten())
            all_accels.append(batch["accel"].numpy().flatten())
            all_juncs.append(batch["junction"].numpy().flatten())
            all_veh_hazs.append(batch["vehicle_hazard"].numpy().flatten())
            all_light_hazs.append(batch["light_hazard"].numpy().flatten())
            all_stop_hazs.append(batch["stop_sign_hazard"].numpy().flatten())
            all_route_ids.extend(routes)
            if "ego_matrix" in batch:
                all_ego_mats.append(batch["ego_matrix"].squeeze(0).numpy())

            if out.get("w_cam") is not None:
                all_w_cam.append(out["w_cam"].squeeze(-1).cpu().numpy())
            if out.get("w_lidar") is not None:
                all_w_lidar.append(out["w_lidar"].squeeze(-1).cpu().numpy())

    preds = np.concatenate(all_preds, axis=0)
    gts = np.concatenate(all_gts, axis=0)
    steers = np.concatenate(all_steers, axis=0)
    speeds = np.concatenate(all_speeds, axis=0)
    accels = np.concatenate(all_accels, axis=0)
    juncs = np.concatenate(all_juncs, axis=0)
    veh_hazs = np.concatenate(all_veh_hazs, axis=0)
    light_hazs = np.concatenate(all_light_hazs, axis=0)
    stop_hazs = np.concatenate(all_stop_hazs, axis=0)
    w_cam_arr = np.concatenate(all_w_cam, axis=0) if all_w_cam else np.full(len(preds), 0.5)
    w_lidar_arr = np.concatenate(all_w_lidar, axis=0) if all_w_lidar else np.full(len(preds), 0.5)

    diffs = preds - gts
    step_dists = np.linalg.norm(diffs, axis=-1)  # (N, 4)
    ades = step_dists.mean(axis=1)
    fdes = step_dists[:, -1]
    lat_ades = np.abs(diffs[:, :, 1]).mean(axis=1)

    # Coordinate-transformed trajectory flicker (model flicker and ground truth flicker)
    unique_routes = sorted(list(set(all_route_ids)))
    route_preds = {r: [] for r in unique_routes}
    route_gts = {r: [] for r in unique_routes}
    route_mats = {r: [] for r in unique_routes}

    for idx, (r_id, p, g) in enumerate(zip(all_route_ids, preds, gts)):
        route_preds[r_id].append(p)
        route_gts[r_id].append(g)
        if all_ego_mats:
            route_mats[r_id].append(all_ego_mats[idx])

    model_flickers = []
    gt_flickers = []
    for r_id, p_list in route_preds.items():
        if len(p_list) < 2 or not route_mats.get(r_id):
            continue
        m_list = route_mats[r_id]
        g_list = route_gts[r_id]
        for t in range(len(p_list) - 1):
            p_t = p_list[t]
            p_next = p_list[t + 1]
            g_t = g_list[t]
            g_next = g_list[t + 1]
            M_t = m_list[t]
            M_next = m_list[t + 1]

            T_rel = np.linalg.inv(M_t) @ M_next

            def transform_pts(pts):
                mat_pts = np.zeros((len(pts), 4), dtype=np.float64)
                mat_pts[:, 0] = pts[:, 0]
                mat_pts[:, 1] = -pts[:, 1]
                mat_pts[:, 3] = 1.0
                pts_t = (T_rel @ mat_pts.T).T
                fwd = pts_t[:, 0]
                left = -pts_t[:, 1]
                return np.column_stack([fwd, left])

            p_next_in_t = transform_pts(p_next)
            g_next_in_t = transform_pts(g_next)

            diff_model = np.linalg.norm(p_next_in_t - p_t, axis=-1).mean()
            diff_gt = np.linalg.norm(g_next_in_t - g_t, axis=-1).mean()
            model_flickers.append(float(diff_model))
            gt_flickers.append(float(diff_gt))

    mean_flicker = float(np.mean(model_flickers)) if model_flickers else 0.0
    mean_gt_flicker = float(np.mean(gt_flickers)) if gt_flickers else 0.0

    # Gate statistics: mean, std, and correlations with speed and yaw rate
    from scipy.stats import pearsonr, spearmanr
    corr_spd_p, corr_spd_s = 0.0, 0.0
    corr_yaw_p, corr_yaw_s = 0.0, 0.0
    if gate_mode == "learned" and len(w_cam_arr) > 1:
        try:
            corr_spd_p, _ = pearsonr(w_cam_arr, speeds)
            corr_spd_s, _ = spearmanr(w_cam_arr, speeds)
            corr_yaw_p, _ = pearsonr(w_cam_arr, np.abs(yaw_rates))
            corr_yaw_s, _ = spearmanr(w_cam_arr, np.abs(yaw_rates))
        except Exception:
            pass

    gate_stats = {
        "w_cam_mean": float(np.mean(w_cam_arr)),
        "w_cam_std": float(np.std(w_cam_arr)),
        "corr_speed_pearson": float(corr_spd_p),
        "corr_speed_spearman": float(corr_spd_s),
        "corr_yaw_pearson": float(corr_yaw_p),
        "corr_yaw_spearman": float(corr_yaw_s),
    }

    # Slices
    turning_mask = np.abs(steers) > 0.05
    straight_mask = np.abs(steers) <= 0.05

    def get_slice_metrics(mask):
        return {
            "n_frames": int(mask.sum()),
            "ade_mean": float(ades[mask].mean()),
            "ade_med": float(np.median(ades[mask])),
            "fde_mean": float(fdes[mask].mean()),
            "fde_med": float(np.median(fdes[mask])),
            "lat_ade_mean": float(lat_ades[mask].mean()),
        }

    overall_metrics = {
        "n_frames": int(len(ades)),
        "ade_mean": float(ades.mean()),
        "ade_med": float(np.median(ades)),
        "fde_mean": float(fdes.mean()),
        "fde_med": float(np.median(fdes)),
        "lat_ade_mean": float(lat_ades.mean()),
    }
    turning_metrics = get_slice_metrics(turning_mask)
    straight_metrics = get_slice_metrics(straight_mask)

    # Hazard slices
    hazard_metrics = {
        "red_light_ade": float(ades[light_hazs > 0.5].mean()) if (light_hazs > 0.5).any() else 0.0,
        "junction_ade": float(ades[juncs > 0.5].mean()) if (juncs > 0.5).any() else 0.0,
        "speed_change_ade": float(ades[np.abs(accels) > 1.0].mean()) if (np.abs(accels) > 1.0).any() else 0.0,
    }

    # Route-level aggregated ADE for paired bootstrap
    route_ade_map = {}
    route_turning_ade_map = {}
    for r_id in unique_routes:
        r_mask = np.array([rid == r_id for rid in all_route_ids])
        route_ade_map[r_id] = float(ades[r_mask].mean())
        r_turn_mask = r_mask & turning_mask
        if r_turn_mask.any():
            route_turning_ade_map[r_id] = float(ades[r_turn_mask].mean())

    result_payload = {
        "run_id": run_id,
        "config": run_config,
        "epoch_logs": epoch_logs,
        "best_val_ade": best_val_ade,
        "overall": overall_metrics,
        "turning": turning_metrics,
        "straight": straight_metrics,
        "flicker": mean_flicker,
        "gt_flicker": mean_gt_flicker,
        "gate_stats": gate_stats,
        "hazard_slices": hazard_metrics,
        "route_ades": route_ade_map,
        "route_turning_ades": route_turning_ade_map,
    }

    with open(cache_path, "w") as f:
        json.dump(result_payload, f, indent=2)
    checkpoint_volume.commit()
    print(f"[{run_id.upper()}] Successfully saved results to {cache_path}", flush=True)

    return result_payload


@app.local_entrypoint()
def main(group: int = 1):
    print(f"=== Modal STAF v2 Training Suite (Executing Group {group}) ===", flush=True)
    seeds = [42, 43, 44]
    jobs = []

    if group == 1:
        # Group 1: A = full, learned gate, queue 5; B = full, constant gate, queue 5.
        for s in seeds:
            jobs.append({
                "variant": "full",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group1_A_full_learned_q5_seed{s}",
            })
            jobs.append({
                "variant": "full",
                "gate_mode": "constant",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group1_B_full_constant_q5_seed{s}",
            })
    elif group == 2:
        # Group 2: C = full, learned gate, queue 1.
        for s in seeds:
            jobs.append({
                "variant": "full",
                "gate_mode": "learned",
                "queue_length": 1,
                "seed": s,
                "run_id": f"group2_C_full_learned_q1_seed{s}",
            })
    elif group == 3:
        # Group 3: D = camera_only, E = lidar_only, F = no_vision, queue 5.
        for s in seeds:
            jobs.append({
                "variant": "camera_only",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_D_camera_only_q5_seed{s}",
            })
            jobs.append({
                "variant": "lidar_only",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_E_lidar_only_q5_seed{s}",
            })
            jobs.append({
                "variant": "no_vision",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_F_no_vision_q5_seed{s}",
            })
    else:
        raise ValueError(f"Invalid group {group}. Expected 1, 2, or 3.")

    print(f"Submitting {len(jobs)} jobs for Group {group}...", flush=True)
    results = list(train_and_eval_v2.map(jobs))

    # Save to local repository results/v2_retrained_variants/
    local_out_dir = os.path.join(SCRIPT_DIR, "../results/v2_retrained_variants")
    os.makedirs(local_out_dir, exist_ok=True)

    for r in results:
        r_id = r["run_id"]
        res_file = os.path.join(local_out_dir, f"{r_id}.json")
        cfg_file = os.path.join(local_out_dir, f"{r_id}_config.json")
        with open(res_file, "w") as f:
            json.dump(r, f, indent=2)
        with open(cfg_file, "w") as f:
            json.dump(r["config"], f, indent=2)

    print(f"\nGroup {group} Complete! Saved all result and config JSONs to {local_out_dir}", flush=True)

    # Print summary table for Group
    print("\n" + "="*110, flush=True)
    print(f"GROUP {group} SUMMARY TABLE across 3 Seeds (42, 43, 44)", flush=True)
    print("="*110, flush=True)
    print(f"{'Run / Model':<32} | {'Overall ADE [m]':<18} | {'Turning ADE [m]':<18} | {'Flicker (L2)':<16} | {'Gate cam/lidar':<16}", flush=True)
    print("-"*110, flush=True)

    # Group by base name
    grouped = {}
    for r in results:
        base_name = "_".join(r["run_id"].split("_")[:-1])
        if base_name not in grouped:
            grouped[base_name] = []
        grouped[base_name].append(r)

    for base, runs in grouped.items():
        ades = [rn["overall"]["ade_mean"] for rn in runs]
        turn_ades = [rn["turning"]["ade_mean"] for rn in runs]
        flickers = [rn["flicker"] for rn in runs]
        cam_means = [rn["gate_stats"]["w_cam_mean"] for rn in runs]
        lidar_means = [rn["gate_stats"]["w_lidar_mean"] for rn in runs]

        ade_str = f"{np.mean(ades):.4f} +- {np.std(ades):.4f}"
        turn_str = f"{np.mean(turn_ades):.4f} +- {np.std(turn_ades):.4f}"
        flicker_str = f"{np.mean(flickers):.4f}"
        gate_str = f"{np.mean(cam_means):.2f} / {np.mean(lidar_means):.2f}"

        print(f"{base:<32} | {ade_str:<18} | {turn_str:<18} | {flicker_str:<16} | {gate_str:<16}", flush=True)

    print("="*110, flush=True)

