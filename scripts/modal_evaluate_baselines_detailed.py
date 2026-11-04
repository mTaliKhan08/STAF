"""
Detailed Evaluation of Baselines with Yaw-Rate Sign Correction & Subgroup Analysis:
- Turning frames (|steer| > 0.05) vs Straight frames (|steer| <= 0.05)
- Kinematics MLP WITH route goal (5 inputs) vs WITHOUT route goal (3 inputs)
- Constant Velocity vs Constant Curvature (corrected vs inverted sign)
"""

import os
import re
import json
import math
import random
import numpy as np
import modal

app = modal.App("staf-baselines-detailed")
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
def run_detailed_baselines():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    carla_root = "/data/carla_transfuser"
    routes = []
    for root, dirs, files in os.walk(carla_root):
        if "measurements" in dirs:
            routes.append(root)
    routes = sorted(list(set(routes)))

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

    print(f"Discovered {len(routes)} routes: {len(train_routes)} train, {len(val_routes)} val, {len(test_routes)} held-out test", flush=True)

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

    def collect_samples(route_list, max_samples=2000, split_name="val"):
        samples = []
        for r_idx, r_dir in enumerate(route_list):
            m_dir = os.path.join(r_dir, "measurements")
            if not os.path.exists(m_dir):
                continue
            fs = sorted([f for f in os.listdir(m_dir) if f.endswith(".json")])
            if len(fs) < 10:
                continue

            needed_steps = list(range(2, len(fs) - 5, 2))[:80]
            needed_file_indices = set()
            for i in needed_steps:
                needed_file_indices.update([i - 1, i, i + 1, i + 2, i + 3, i + 4])

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
                gt_wps = np.array(gt_wps, dtype=np.float32)

                x = float(d_cur["x"])
                y = float(d_cur["y"])
                xc = float(d_cur.get("x_command", x))
                yc = float(d_cur.get("y_command", y))
                tgt = transform_world_point_to_ego(cur_M, (xc, yc))

                spd = float(d_cur.get("speed", 0.0))
                spd_prev = float(d_prev.get("speed", spd))
                ax = (spd - spd_prev) / 0.5  # causal longitudinal acceleration [m/s^2]

                # Yaw rate analysis:
                theta_cur = float(d_cur["theta"])
                theta_prev = float(d_prev["theta"])
                d_theta = (theta_cur - theta_prev + np.pi) % (2 * np.pi) - np.pi
                raw_yaw_rate = d_theta / 0.5  # compass rate [rad/s]
                # Right-handed body yaw rate: CARLA compass increases clockwise, so body rate is -dtheta/dt
                body_yaw_rate = -raw_yaw_rate

                steer = float(d_cur.get("steer", 0.0))

                samples.append({
                    "speed": spd,
                    "accel": ax,
                    "raw_yaw_rate": raw_yaw_rate,
                    "body_yaw_rate": body_yaw_rate,
                    "steer": steer,
                    "target_point": tgt,  # (2,)
                    "gt_waypoints": gt_wps,  # (4, 2)
                })
                if len(samples) >= max_samples:
                    break
            if len(samples) >= max_samples:
                break
        return samples

    print("Collecting val and train samples...", flush=True)
    val_samples = collect_samples(val_routes, max_samples=2000, split_name="val")
    train_samples = collect_samples(train_routes, max_samples=2000, split_name="train")
    print(f"Collected {len(val_samples)} val samples, {len(train_samples)} train samples", flush=True)

    time_horizons = [0.5, 1.0, 1.5, 2.0]

    # --- Train Kinematics MLP WITH route goal (5 inputs) ---
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

    # --- Train Kinematics MLP WITHOUT route goal (3 inputs) ---
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

    X_train_goal = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_point"][0], s["target_point"][1]] for s in train_samples], dtype=torch.float32)
    X_train_nogoal = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in train_samples], dtype=torch.float32)
    Y_train = torch.tensor(np.array([s["gt_waypoints"] for s in train_samples]), dtype=torch.float32)

    X_val_goal = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_point"][0], s["target_point"][1]] for s in val_samples], dtype=torch.float32)
    X_val_nogoal = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in val_samples], dtype=torch.float32)

    # Train MLP with goal
    mlp_goal = KinematicsMLPGoal()
    opt_goal = optim.Adam(mlp_goal.parameters(), lr=1e-3)
    loader_goal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_goal, Y_train), batch_size=64, shuffle=True)
    for _ in range(10):
        for bx, by in loader_goal:
            opt_goal.zero_grad()
            loss = nn.L1Loss()(mlp_goal(bx), by)
            loss.backward()
            opt_goal.step()

    # Train MLP without goal
    mlp_nogoal = KinematicsMLPNoGoal()
    opt_nogoal = optim.Adam(mlp_nogoal.parameters(), lr=1e-3)
    loader_nogoal = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train_nogoal, Y_train), batch_size=64, shuffle=True)
    for _ in range(10):
        for bx, by in loader_nogoal:
            opt_nogoal.zero_grad()
            loss = nn.L1Loss()(mlp_nogoal(bx), by)
            loss.backward()
            opt_nogoal.step()

    mlp_goal.eval()
    mlp_nogoal.eval()
    with torch.no_grad():
        preds_mlp_goal = mlp_goal(X_val_goal).numpy()
        preds_mlp_nogoal = mlp_nogoal(X_val_nogoal).numpy()

    # Predictions for Constant Velocity
    preds_cv = []
    # Predictions for Constant Curvature (CORRECT SIGN: body_yaw_rate)
    preds_cc_correct = []
    # Predictions for Constant Curvature (INVERTED SIGN: raw_yaw_rate)
    preds_cc_inverted = []

    for s in val_samples:
        v = s["speed"]
        w_corr = s["body_yaw_rate"]
        w_inv = s["raw_yaw_rate"]

        preds_cv.append(np.array([[v * t, 0.0] for t in time_horizons], dtype=np.float32))

        # Correct
        p_c = []
        for t in time_horizons:
            if abs(w_corr) < 1e-4:
                p_c.append([v * t, 0.0])
            else:
                R = v / w_corr
                psi = w_corr * t
                p_c.append([R * math.sin(psi), R * (1.0 - math.cos(psi))])
        preds_cc_correct.append(p_c)

        # Inverted
        p_i = []
        for t in time_horizons:
            if abs(w_inv) < 1e-4:
                p_i.append([v * t, 0.0])
            else:
                R = v / w_inv
                psi = w_inv * t
                p_i.append([R * math.sin(psi), R * (1.0 - math.cos(psi))])
        preds_cc_inverted.append(p_i)

    preds_cv = np.array(preds_cv, dtype=np.float32)
    preds_cc_correct = np.array(preds_cc_correct, dtype=np.float32)
    preds_cc_inverted = np.array(preds_cc_inverted, dtype=np.float32)
    gts = np.array([s["gt_waypoints"] for s in val_samples], dtype=np.float32)
    steers = np.array([abs(s["steer"]) for s in val_samples])

    def calc_metrics(preds, targets):
        diff = preds - targets
        eucl = np.linalg.norm(diff, axis=-1)
        lat_err = np.abs(diff[..., 1])
        fwd_err = np.abs(diff[..., 0])
        l1_horizons = np.mean(fwd_err + lat_err, axis=0)
        return {
            "ade": float(np.mean(eucl)),
            "fde": float(np.mean(eucl[:, 3])),
            "ade_lat": float(np.mean(lat_err)),
            "fde_lat": float(np.mean(lat_err[:, 3])),
            "ade_fwd": float(np.mean(fwd_err)),
            "mean_l1": float(np.mean(l1_horizons)),
            "l1_05s": float(l1_horizons[0]),
            "l1_10s": float(l1_horizons[1]),
            "l1_15s": float(l1_horizons[2]),
            "l1_20s": float(l1_horizons[3]),
        }

    # Subsets:
    mask_turning = steers > 0.05
    mask_straight = steers <= 0.05

    print(f"\n--- EVALUATION SUBGROUPS ---", flush=True)
    print(f"Total val frames: {len(val_samples)}", flush=True)
    print(f"Turning frames (|steer| > 0.05): {np.sum(mask_turning)} ({np.mean(mask_turning):.1%})", flush=True)
    print(f"Straight frames (|steer| <= 0.05): {np.sum(mask_straight)} ({np.mean(mask_straight):.1%})", flush=True)

    models_dict = {
        "Constant Velocity": preds_cv,
        "Constant Curvature (Correct Sign)": preds_cc_correct,
        "Constant Curvature (Inverted Sign)": preds_cc_inverted,
        "Kinematics MLP (With Goal, 5 in)": preds_mlp_goal,
        "Kinematics MLP (No Goal, 3 in)": preds_mlp_nogoal,
    }

    results = {}
    for subset_name, mask in [("ALL", slice(None)), ("TURNING", mask_turning), ("STRAIGHT", mask_straight)]:
        results[subset_name] = {}
        print("\n" + "="*95, flush=True)
        print(f"SUBSET: {subset_name} (N = {np.sum(mask) if isinstance(mask, np.ndarray) else len(val_samples)})", flush=True)
        print("="*95, flush=True)
        print(f"{'Model':<35} | {'ADE [m]':<10} | {'FDE [m]':<10} | {'ADE Lat':<10} | {'FDE Lat':<10} | {'Mean L1':<10}", flush=True)
        print("-"*95, flush=True)
        for m_name, p in models_dict.items():
            m_res = calc_metrics(p[mask], gts[mask])
            results[subset_name][m_name] = m_res
            print(f"{m_name:<35} | {m_res['ade']:<10.4f} | {m_res['fde']:<10.4f} | {m_res['ade_lat']:<10.4f} | {m_res['fde_lat']:<10.4f} | {m_res['mean_l1']:<10.4f}", flush=True)
        print("="*95, flush=True)

    return results

@app.local_entrypoint()
def main():
    print("Launching Detailed Baselines Evaluation on Modal...", flush=True)
    res = run_detailed_baselines.remote()
    print("Done!", flush=True)

