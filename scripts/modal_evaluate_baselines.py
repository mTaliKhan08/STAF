"""
Modal script to evaluate Baselines on corrected CARLA labels:
1. Towns split:
   - Final Test Towns (Held-out): Town02, Town05 (including HD variants)
   - Training Towns: All other towns (Town01, Town03, Town04, Town06, Town07, Town10HD)
   - Route-level 80/20 train/val split within training towns
2. Corrected 2 Hz labels (stride 1 frame = 0.5s, pred_len=4 -> 0.5s, 1.0s, 1.5s, 2.0s)
3. Evaluates 3 Baselines:
   - Baseline 1: Constant Velocity (x = v * t, y = 0)
   - Baseline 2: Constant Curvature (Bicycle model with causal yaw rate / steer)
   - Baseline 3: Kinematics-only MLP (trained on train split, evaluated on val split)
4. Metrics reported per baseline:
   - Per-horizon L1 error (0.5s, 1.0s, 1.5s, 2.0s)
   - ADE (Average Displacement Error)
   - FDE (Final Displacement Error at 2.0s)
   - Lateral Error separately (ADE_lat, FDE_lat)
"""

import os
import re
import json
import math
import random
import numpy as np
import modal

app = modal.App("staf-baselines-evaluation")
data_volume = modal.Volume.from_name("staf-datasets")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy", "scipy")
)

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    timeout=600,
)
def run_baselines_eval():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    carla_root = "/data/carla_transfuser"
    routes = []
    for root, dirs, files in os.walk(carla_root):
        if "measurements" in dirs:
            routes.append(root)
    routes = sorted(list(set(routes)))
    print(f"Total discovered routes in {carla_root}: {len(routes)}", flush=True)

    # Town-based split:
    # Test Towns: Town02, Town05 (strictly held out)
    # Training Towns: All others
    test_routes = []
    train_town_routes = []

    for r in routes:
        m = re.search(r"Town\d+(?:HD)?", r)
        town = m.group(0) if m else "Unknown"
        if town.startswith("Town02") or town.startswith("Town05"):
            test_routes.append(r)
        else:
            train_town_routes.append(r)

    # Deterministic 80/20 route split within training towns
    random.seed(42)
    random.shuffle(train_town_routes)
    n_val = max(1, len(train_town_routes) // 5)
    val_routes = train_town_routes[:n_val]
    train_routes = train_town_routes[n_val:]

    print(f"\n--- DATASET SPLIT SUMMARY ---", flush=True)
    print(f"Training Towns Routes: {len(train_town_routes)} (Split: {len(train_routes)} train / {len(val_routes)} val)", flush=True)
    print(f"Held-Out Test Towns (Town02/Town05): {len(test_routes)} routes", flush=True)

    # Helper: SE(3) transform
    def transform_future_ego_pose(cur_matrix, fut_matrix):
        M_cur = np.array(cur_matrix, dtype=np.float64)
        M_fut = np.array(fut_matrix, dtype=np.float64)
        M_rel = np.linalg.inv(M_cur) @ M_fut
        fwd = float(M_rel[0, 3])
        left = float(-M_rel[1, 3])
        return np.array([fwd, left], dtype=np.float32)

    def transform_world_point_to_ego(cur_matrix, world_point_xy):
        M_cur = np.array(cur_matrix, dtype=np.float64)
        M_cur_inv = np.linalg.inv(M_cur)
        xw, yw = world_point_xy
        P_mat = np.array([yw, -xw, 0.0, 1.0], dtype=np.float64)
        P_ego = M_cur_inv @ P_mat
        fwd = float(P_ego[0])
        left = float(-P_ego[1])
        return np.array([fwd, left], dtype=np.float32)

    # High-efficiency cached reader
    def collect_samples_from_routes(route_list, max_samples=2000, split_name="val"):
        samples = []
        for r_idx, r_dir in enumerate(route_list):
            m_dir = os.path.join(r_dir, "measurements")
            if not os.path.exists(m_dir):
                continue
            fs = sorted([f for f in os.listdir(m_dir) if f.endswith(".json")])
            if len(fs) < 10:
                continue

            # Determine needed indices (stride 2 frames)
            # Up to 80 frames per route to ensure diverse route representation
            needed_steps = list(range(2, len(fs) - 5, 2))[:80]
            needed_file_indices = set()
            for i in needed_steps:
                needed_file_indices.add(i - 1)
                needed_file_indices.add(i)
                needed_file_indices.add(i + 1)
                needed_file_indices.add(i + 2)
                needed_file_indices.add(i + 3)
                needed_file_indices.add(i + 4)

            # Read each JSON file at most once into route_cache
            route_cache = {}
            for fi in needed_file_indices:
                p = os.path.join(m_dir, fs[fi])
                try:
                    with open(p, "r") as fp:
                        route_cache[fi] = json.load(fp)
                except Exception:
                    pass

            for i in needed_steps:
                if (i not in route_cache) or (i - 1 not in route_cache):
                    continue
                d_cur = route_cache[i]
                d_prev = route_cache[i - 1]

                fut_matrices = []
                valid_fut = True
                for st in [1, 2, 3, 4]:
                    if (i + st) not in route_cache:
                        valid_fut = False
                        break
                    fut_matrices.append(route_cache[i + st]["ego_matrix"])
                if not valid_fut:
                    continue

                cur_M = d_cur["ego_matrix"]
                gt_wps = []
                for f_M in fut_matrices:
                    gt_wps.append(transform_future_ego_pose(cur_M, f_M))
                gt_wps = np.array(gt_wps, dtype=np.float32)  # (4, 2)

                x = float(d_cur["x"])
                y = float(d_cur["y"])
                xc = float(d_cur.get("x_command", x))
                yc = float(d_cur.get("y_command", y))
                tgt = transform_world_point_to_ego(cur_M, (xc, yc))

                spd = float(d_cur.get("speed", 0.0))
                spd_prev = float(d_prev.get("speed", spd))
                ax = (spd - spd_prev) / 0.5  # causal acceleration (t-1 -> t)

                # Yaw rate from causal delta theta
                theta_cur = float(d_cur["theta"])
                theta_prev = float(d_prev["theta"])
                d_theta = (theta_cur - theta_prev + np.pi) % (2 * np.pi) - np.pi
                omega_z = d_theta / 0.5  # causal yaw rate
                steer = float(d_cur.get("steer", 0.0))

                samples.append({
                    "speed": spd,
                    "accel": ax,
                    "yaw_rate": omega_z,
                    "steer": steer,
                    "target_point": tgt,  # (2,)
                    "gt_waypoints": gt_wps,  # (4, 2)
                })
                if len(samples) >= max_samples:
                    break

            if (r_idx + 1) % 5 == 0 or len(samples) >= max_samples:
                print(f"[{split_name}] Processed {r_idx + 1}/{len(route_list)} routes -> {len(samples)}/{max_samples} samples collected", flush=True)

            if len(samples) >= max_samples:
                break

        return samples

    print("Collecting validation samples from i.i.d. val routes...", flush=True)
    val_samples = collect_samples_from_routes(val_routes, max_samples=2000, split_name="val")
    print(f"Collected {len(val_samples)} validation samples.", flush=True)

    print("\nCollecting training samples for Kinematics-only MLP...", flush=True)
    train_samples = collect_samples_from_routes(train_routes, max_samples=2000, split_name="train")
    print(f"Collected {len(train_samples)} training samples.", flush=True)

    time_horizons = [0.5, 1.0, 1.5, 2.0]

    # --- Baseline 1: Constant Velocity ---
    # Prediction: x_k = v * t_k, y_k = 0
    preds_cv = []
    gts = []
    for s in val_samples:
        v = s["speed"]
        pred = np.array([[v * t, 0.0] for t in time_horizons], dtype=np.float32)
        preds_cv.append(pred)
        gts.append(s["gt_waypoints"])

    preds_cv = np.array(preds_cv)  # (N, 4, 2)
    gts = np.array(gts)            # (N, 4, 2)

    # --- Baseline 2: Constant Curvature (Bicycle Model) ---
    # Prediction using causal yaw rate omega_z:
    # x_k = (v / w) * sin(w * t_k), y_k = (v / w) * (1 - cos(w * t_k)) [left is +y]
    preds_cc = []
    for s in val_samples:
        v = s["speed"]
        w = s["yaw_rate"]
        pred = []
        for t in time_horizons:
            if abs(w) < 1e-4:
                pred.append([v * t, 0.0])
            else:
                R = v / w
                psi = w * t
                # In left-positive frame: turning left (w > 0) curves into +y
                px = R * math.sin(psi)
                py = R * (1.0 - math.cos(psi))
                pred.append([px, py])
        preds_cc.append(pred)
    preds_cc = np.array(preds_cc, dtype=np.float32)

    # --- Baseline 3: Kinematics-Only MLP ---
    # Features: [speed, accel, yaw_rate, target_fwd, target_left] (dim = 5)
    # Output: (4, 2) = 8 dims
    class KinematicsMLP(nn.Module):
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

    # Train Kinematics MLP
    X_train = []
    Y_train = []
    for s in train_samples:
        X_train.append([s["speed"], s["accel"], s["yaw_rate"], s["target_point"][0], s["target_point"][1]])
        Y_train.append(s["gt_waypoints"])
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)

    X_val = []
    for s in val_samples:
        X_val.append([s["speed"], s["accel"], s["yaw_rate"], s["target_point"][0], s["target_point"][1]])
    X_val = torch.tensor(X_val, dtype=torch.float32)

    mlp_model = KinematicsMLP()
    optimizer = optim.Adam(mlp_model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    dataset_tensor = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset_tensor, batch_size=64, shuffle=True)

    print("\nTraining Kinematics-only MLP baseline on training routes (10 epochs)...", flush=True)
    mlp_model.train()
    for ep in range(10):
        ep_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            out = mlp_model(bx)
            loss = loss_fn(out, by)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item() * len(bx)
        ep_loss /= len(X_train)
        if (ep + 1) % 2 == 0:
            print(f"MLP Epoch {ep+1}/10 - Train L1: {ep_loss:.4f} m", flush=True)

    mlp_model.eval()
    with torch.no_grad():
        preds_mlp = mlp_model(X_val).numpy()

    # --- Compute Detailed Benchmark Metrics ---
    def compute_metrics(preds, gts):
        # preds, gts: (N, 4, 2)
        diff = preds - gts
        eucl = np.linalg.norm(diff, axis=-1)  # (N, 4)
        lat_err = np.abs(diff[..., 1])        # (N, 4)
        fwd_err = np.abs(diff[..., 0])        # (N, 4)

        # Per-horizon L1 (average of abs(dx) + abs(dy))
        l1_horizons = np.mean(fwd_err + lat_err, axis=0)  # (4,)
        ade = float(np.mean(eucl))
        fde = float(np.mean(eucl[:, 3]))

        ade_lat = float(np.mean(lat_err))
        fde_lat = float(np.mean(lat_err[:, 3]))

        ade_fwd = float(np.mean(fwd_err))
        fde_fwd = float(np.mean(fwd_err[:, 3]))

        return {
            "l1_05s": float(l1_horizons[0]),
            "l1_10s": float(l1_horizons[1]),
            "l1_15s": float(l1_horizons[2]),
            "l1_20s": float(l1_horizons[3]),
            "ade": ade,
            "fde": fde,
            "ade_lat": ade_lat,
            "fde_lat": fde_lat,
            "ade_fwd": ade_fwd,
            "fde_fwd": fde_fwd,
        }

    m_cv = compute_metrics(preds_cv, gts)
    m_cc = compute_metrics(preds_cc, gts)
    m_mlp = compute_metrics(preds_mlp, gts)

    print("\n" + "="*80, flush=True)
    print("STAF BASELINE BENCHMARK ON CORRECTED LABELS (Validation Set)", flush=True)
    print("="*80, flush=True)
    print(f"{'Metric':<20} | {'Constant Velocity':<18} | {'Constant Curvature':<18} | {'Kinematics MLP':<18}", flush=True)
    print("-" * 80, flush=True)
    print(f"{'ADE [m]':<20} | {m_cv['ade']:<18.4f} | {m_cc['ade']:<18.4f} | {m_mlp['ade']:<18.4f}", flush=True)
    print(f"{'FDE @ 2.0s [m]':<20} | {m_cv['fde']:<18.4f} | {m_cc['fde']:<18.4f} | {m_mlp['fde']:<18.4f}", flush=True)
    print(f"{'ADE Lateral [m]':<20} | {m_cv['ade_lat']:<18.4f} | {m_cc['ade_lat']:<18.4f} | {m_mlp['ade_lat']:<18.4f}", flush=True)
    print(f"{'FDE Lateral [m]':<20} | {m_cv['fde_lat']:<18.4f} | {m_cc['fde_lat']:<18.4f} | {m_mlp['fde_lat']:<18.4f}", flush=True)
    print(f"{'ADE Forward [m]':<20} | {m_cv['ade_fwd']:<18.4f} | {m_cc['ade_fwd']:<18.4f} | {m_mlp['ade_fwd']:<18.4f}", flush=True)
    print(f"{'L1 @ 0.5s [m]':<20} | {m_cv['l1_05s']:<18.4f} | {m_cc['l1_05s']:<18.4f} | {m_mlp['l1_05s']:<18.4f}", flush=True)
    print(f"{'L1 @ 1.0s [m]':<20} | {m_cv['l1_10s']:<18.4f} | {m_cc['l1_10s']:<18.4f} | {m_mlp['l1_10s']:<18.4f}", flush=True)
    print(f"{'L1 @ 1.5s [m]':<20} | {m_cv['l1_15s']:<18.4f} | {m_cc['l1_15s']:<18.4f} | {m_mlp['l1_15s']:<18.4f}", flush=True)
    print(f"{'L1 @ 2.0s [m]':<20} | {m_cv['l1_20s']:<18.4f} | {m_cc['l1_20s']:<18.4f} | {m_mlp['l1_20s']:<18.4f}", flush=True)
    print("="*80, flush=True)

    return {
        "m_cv": m_cv,
        "m_cc": m_cc,
        "m_mlp": m_mlp,
        "n_val": len(val_samples),
        "n_train": len(train_samples),
    }

@app.local_entrypoint()
def main():
    print("Launching Baselines Evaluation on Modal...", flush=True)
    res = run_baselines_eval.remote()
    print("Evaluation Complete!", flush=True)
