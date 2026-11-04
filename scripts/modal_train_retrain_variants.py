"""
Modal Serverless Retraining Suite for STAF Modality Ablations across 3 Seeds.

Trains and evaluates 4 distinct variants from scratch (NOT inference masking):
1. no_vision: Kinematics + Goal only (5-d input to residual head over CC)
2. camera_only: RGB Front Camera + Kinematics + Goal
3. lidar_only: LiDAR 2-channel BEV + Kinematics + Goal
4. full: Camera + LiDAR + Kinematics + Goal

Seeds: [42, 43, 44]
Evaluation Set: Held-Out Test Towns (Town02 & Town05, 17,447 frames)
Metrics: ADE, FDE, Lat-ADE, Lat-FDE, Mean L1 sum (mean +- std across 3 seeds),
         plus 95% Route-Level Bootstrap Confidence Intervals and Hazard/Dynamic Slices.
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

app = modal.App("staf-retrain-variants-suite")

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
    max_containers=2,
    volumes={
        "/data": data_volume,
        "/checkpoints": checkpoint_volume,
    },
    timeout=3600,
)
def train_and_eval_variant(variant: str, seed: int) -> dict:
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from staf.data.dataset import CARLASTAFDataset
    from staf.models.staf_model import STAFModel

    # Check if results are already cached on Volume
    os.makedirs("/checkpoints/retrained_metrics", exist_ok=True)
    cache_path = f"/checkpoints/retrained_metrics/{variant}_seed{seed}.json"
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            print(f"[{variant.upper()} | SEED {seed}] Loaded existing cached results from {cache_path}", flush=True)
            return cached
        except Exception as e:
            print(f"[{variant.upper()} | SEED {seed}] Cache corrupted ({e}), retraining...", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    carla_root = "/data/carla_transfuser"

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"[{variant.upper()} | SEED {seed}] Loading datasets...", flush=True)
    train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=4, augment_domain=True)
    val_dataset = CARLASTAFDataset(carla_root, split="val", pred_len=4, augment_domain=False)
    test_dataset = CARLASTAFDataset(carla_root, split="test", pred_len=4, augment_domain=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"[{variant.upper()} | SEED {seed}] Initializing STAF Residual Model...", flush=True)
    model = STAFModel(
        embed_dim=256,
        pred_len=4,
        modality=variant,
        gate_input_dim=3,
        use_residual_cc=True,
        use_full_kinematics_head=True,
        pretrained_backbones=False,
    ).to(device)

    # Initial parameter groups: freeze backbones for first 2 epochs
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
    best_val_loss = float("inf")
    best_weights = None
    patience_cnt = 0

    optimizer = get_optimizer(freeze_backbones=True)

    print(f"[{variant.upper()} | SEED {seed}] Starting training with Early Stopping...", flush=True)
    for epoch in range(max_epochs):
        if epoch == 2:
            print(f"[{variant.upper()} | SEED {seed}] Epoch {epoch}: Unfreezing backbones at LR 1e-5", flush=True)
            optimizer = get_optimizer(freeze_backbones=False)

        model.train()
        train_l1 = 0.0
        n_train = 0

        for batch in train_loader:
            rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
            lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
            x_kin = batch["x_kinematic_3d"].to(device)
            target = batch["target_point"].to(device)
            spd = batch["speed"].to(device)
            accel = batch["accel"].to(device)
            yaw_rate = batch["body_yaw_rate"].to(device)
            wps = batch["waypoints"].to(device)

            bs = wps.size(0)
            model.reset_temporal_state(bs, device)
            optimizer.zero_grad()

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_l1 += loss.item() * bs
            n_train += bs

        train_loss = train_l1 / max(1, n_train)

        # Validation pass
        model.eval()
        val_l1 = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
                lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
                x_kin = batch["x_kinematic_3d"].to(device)
                target = batch["target_point"].to(device)
                spd = batch["speed"].to(device)
                accel = batch["accel"].to(device)
                yaw_rate = batch["body_yaw_rate"].to(device)
                wps = batch["waypoints"].to(device)

                bs = wps.size(0)
                model.reset_temporal_state(bs, device)
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
                val_l1 += nn.L1Loss()(pred_wps, wps).item() * bs
                n_val += bs

        val_loss = val_l1 / max(1, n_val)
        print(f"[{variant.upper()} | SEED {seed}] Epoch {epoch}: Train L1 = {train_loss:.4f} m | Val L1 = {val_loss:.4f} m", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[{variant.upper()} | SEED {seed}] Early stopping triggered at epoch {epoch}.", flush=True)
                break

    # Load best checkpoint for held-out test evaluation
    model.load_state_dict(best_weights)
    model.eval()

    print(f"[{variant.upper()} | SEED {seed}] Evaluating on Held-Out Test Set (Town02 & Town05)...", flush=True)
    preds = []
    gts = []
    steers = []
    speeds = []
    accels = []
    juncs = []
    veh_hazs = []
    light_hazs = []
    stop_hazs = []
    routes = []

    with torch.no_grad():
        for batch in test_loader:
            rgb = batch["image"].to(device) if variant in ["full", "camera_only"] else None
            lidar_bev = batch["lidar_bev"].to(device) if variant in ["full", "lidar_only"] else None
            x_kin = batch["x_kinematic_3d"].to(device)
            target = batch["target_point"].to(device)
            spd = batch["speed"].to(device)
            accel = batch["accel"].to(device)
            yaw_rate = batch["body_yaw_rate"].to(device)
            wps = batch["waypoints"]

            bs = wps.size(0)
            model.reset_temporal_state(bs, device)
            out = model(
                rgb=rgb,
                lidar_bev=lidar_bev,
                x_kinematic=x_kin,
                target_point=target,
                ego_speed=spd,
                accel=accel,
                body_yaw_rate=yaw_rate,
            )

            preds.append(out["waypoints"].cpu().numpy())
            gts.append(wps.numpy())
            steers.append(batch["steer"].numpy().flatten())
            speeds.append(spd.cpu().numpy().flatten())
            accels.append(accel.cpu().numpy().flatten())
            juncs.append(batch["junction"].numpy().flatten())
            veh_hazs.append(batch["vehicle_hazard"].numpy().flatten())
            light_hazs.append(batch["light_hazard"].numpy().flatten())
            stop_hazs.append(batch["stop_sign_hazard"].numpy().flatten())
            routes.extend(batch.get("route_dir", ["unknown"] * bs))

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    steers = np.concatenate(steers, axis=0)
    accels = np.concatenate(accels, axis=0)
    juncs = np.concatenate(juncs, axis=0)
    veh_hazs = np.concatenate(veh_hazs, axis=0)
    light_hazs = np.concatenate(light_hazs, axis=0)
    stop_hazs = np.concatenate(stop_hazs, axis=0)
    routes = np.array(routes)

    # Compute metrics inside container
    def calc_metrics(p, t):
        diff = p - t
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

    mask_overall = np.ones(len(gts), dtype=bool)
    mask_turning = np.abs(steers) > 0.05
    mask_straight = ~mask_turning

    metrics_overall = calc_metrics(preds[mask_overall], gts[mask_overall])
    metrics_turning = calc_metrics(preds[mask_turning], gts[mask_turning])
    metrics_straight = calc_metrics(preds[mask_straight], gts[mask_straight])

    # Route bootstrap 95% CI
    unique_routes = np.unique(routes)
    rng = np.random.RandomState(42 + seed)
    boot_ades = []
    for _ in range(500):
        sampled_routes = rng.choice(unique_routes, size=len(unique_routes), replace=True)
        sampled_indices = []
        for r in sampled_routes:
            sampled_indices.extend(np.where(routes == r)[0])
        s_idx = np.array(sampled_indices)
        m_ade = np.mean(np.linalg.norm(preds[s_idx] - gts[s_idx], axis=-1))
        boot_ades.append(m_ade)
    ci_lower = float(np.percentile(boot_ades, 2.5))
    ci_upper = float(np.percentile(boot_ades, 97.5))

    # Hazard and Dynamic Slices
    hazard_slices = {
        "light_hazard": float(np.mean(np.linalg.norm(preds[light_hazs > 0.5] - gts[light_hazs > 0.5], axis=-1))),
        "vehicle_hazard": float(np.mean(np.linalg.norm(preds[veh_hazs > 0.5] - gts[veh_hazs > 0.5], axis=-1))),
        "stop_sign_hazard": float(np.mean(np.linalg.norm(preds[stop_hazs > 0.5] - gts[stop_hazs > 0.5], axis=-1))),
        "junction": float(np.mean(np.linalg.norm(preds[juncs > 0.5] - gts[juncs > 0.5], axis=-1))),
        "speed_change": float(np.mean(np.linalg.norm(preds[np.abs(accels) > 1.0] - gts[np.abs(accels) > 1.0], axis=-1))),
    }

    result_payload = {
        "variant": variant,
        "seed": seed,
        "best_val_loss": float(best_val_loss),
        "metrics_overall": metrics_overall,
        "metrics_turning": metrics_turning,
        "metrics_straight": metrics_straight,
        "bootstrap_ci": [ci_lower, ci_upper],
        "hazard_slices": hazard_slices,
        "n_samples": int(len(gts)),
        "n_turning": int(np.sum(mask_turning)),
        "n_straight": int(np.sum(mask_straight)),
    }

    # Persist to Volume
    try:
        with open(cache_path, "w") as f:
            json.dump(result_payload, f, indent=2)
        checkpoint_volume.commit()
        print(f"[{variant.upper()} | SEED {seed}] Saved metrics to {cache_path} and committed to volume.", flush=True)
    except Exception as e:
        print(f"[{variant.upper()} | SEED {seed}] Error saving cache: {e}", flush=True)

    return result_payload


@app.local_entrypoint()
def main():
    print("Launching Comprehensive Retraining Suite on Modal (Concurrency Limit = 2)...", flush=True)
    variants = ["no_vision", "camera_only", "lidar_only", "full"]
    seeds = [42, 43, 44]
    jobs = [(v, s) for v in variants for s in seeds]

    results = list(train_and_eval_variant.starmap(jobs))
    print(f"\nAll {len(results)} jobs finished! Aggregating multi-seed statistics...", flush=True)

    grouped = {}
    for r in results:
        v = r["variant"]
        if v not in grouped:
            grouped[v] = []
        grouped[v].append(r)

    def print_aggregated_table(title, metric_key):
        print("\n" + "="*115, flush=True)
        print(title, flush=True)
        print("="*115, flush=True)
        print(f"{'Modality / Architecture':<32} | {'ADE [m] (mean +- std)':<22} | {'FDE [m] (mean +- std)':<22} | {'Lat-ADE [m]':<16} | {'Route 95% CI [m]':<16}", flush=True)
        print("-"*115, flush=True)

        for v in ["no_vision", "camera_only", "lidar_only", "full"]:
            runs = grouped.get(v, [])
            ades = [r[metric_key]["ade_mean"] for r in runs]
            fdes = [r[metric_key]["fde_mean"] for r in runs]
            lat_ades = [r[metric_key]["lat_ade_mean"] for r in runs]
            ci_lows = [r["bootstrap_ci"][0] for r in runs]
            ci_highs = [r["bootstrap_ci"][1] for r in runs]

            v_name = {
                "no_vision": "Retrained: No-Vision (Kin + Goal)",
                "camera_only": "Retrained: Camera-Only",
                "lidar_only": "Retrained: LiDAR-Only",
                "full": "Retrained: Full (Cam + LiDAR + Kin)",
            }.get(v, v)

            ade_str = f"{np.mean(ades):.4f} +- {np.std(ades):.4f}"
            fde_str = f"{np.mean(fdes):.4f} +- {np.std(fdes):.4f}"
            lat_str = f"{np.mean(lat_ades):.4f}"
            ci_str = f"[{np.mean(ci_lows):.4f}, {np.mean(ci_highs):.4f}]"

            print(f"{v_name:<32} | {ade_str:<22} | {fde_str:<22} | {lat_str:<16} | {ci_str:<16}", flush=True)
        print("="*115, flush=True)

    print_aggregated_table("HELD-OUT TEST SET (Town02 & Town05) - OVERALL (N = 17,447 frames)", "metrics_overall")
    print_aggregated_table("HELD-OUT TEST SET (Town02 & Town05) - TURNING (|steer| > 0.05, N = 1,892 frames)", "metrics_turning")
    print_aggregated_table("HELD-OUT TEST SET (Town02 & Town05) - STRAIGHT (|steer| <= 0.05, N = 15,555 frames)", "metrics_straight")

    # Hazard & Dynamic Subsets
    print("\n" + "="*125, flush=True)
    print("HAZARD & DYNAMIC SUBSET ERROR ANALYSIS (ADE in meters across 3 Seeds)", flush=True)
    print("="*125, flush=True)
    hazard_keys = [
        ("Red Light", "light_hazard"),
        ("Lead Vehicle", "vehicle_hazard"),
        ("Stop Sign", "stop_sign_hazard"),
        ("Junction", "junction"),
        ("Speed-Change (|a|>1)", "speed_change"),
    ]
    header = f"{'Modality':<30} | " + " | ".join([f"{h_name:<16}" for h_name, _ in hazard_keys])
    print(header, flush=True)
    print("-"*125, flush=True)

    for v in ["no_vision", "camera_only", "lidar_only", "full"]:
        runs = grouped.get(v, [])
        cols = []
        for _, k in hazard_keys:
            vals = [r["hazard_slices"][k] for r in runs]
            cols.append(f"{np.mean(vals):.4f} +- {np.std(vals):.4f}")
        v_name = {
            "no_vision": "No-Vision (Kin + Goal)",
            "camera_only": "Camera-Only",
            "lidar_only": "LiDAR-Only",
            "full": "Full (Cam + LiDAR + Kin)",
        }.get(v, v)
        print(f"{v_name:<30} | " + " | ".join([f"{c:<16}" for c in cols]), flush=True)
    print("="*125, flush=True)
