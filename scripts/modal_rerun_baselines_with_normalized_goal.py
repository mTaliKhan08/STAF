"""
Modal Script to Rerun Baselines with Normalized Goal Features & Early Stopping:
1. Normalizes P_ego = clip(P_ego / 20.0, -3.0, 3.0).
2. Verifies speed is strictly from frame t only (causal).
3. Trains MLPs with validation early stopping on training towns.
4. Evaluates all baselines (CV, CC, MLP 3-in, MLP 5-in normalized, MLP Residual)
   on Held-Out Test Set (Town02 & Town05, 17,447 frames), broken down into:
   - Overall (N = 17,447)
   - Turning (|steer| > 0.05, N = 1,892)
   - Straight (|steer| <= 0.05, N = 15,555)
"""

import os
import sys
import math
import numpy as np
import modal

app = modal.App("staf-baselines-normalized-goal")
data_volume = modal.Volume.from_name("staf-datasets")

eval_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.0",
        "numpy",
        "scipy",
    )
)

@app.function(
    image=eval_image,
    volumes={"/data": data_volume},
    timeout=600,
)
def run_baselines():
    import json
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    carla_root = "/data/carla_transfuser"
    time_horizons = [0.5, 1.0, 1.5, 2.0]

    def wrap_angle(theta_diff: float) -> float:
        return (theta_diff + math.pi) % (2.0 * math.pi) - math.pi

    def transform_future_ego_pose(cur_M, fut_M):
        cur_inv = np.linalg.inv(np.array(cur_M))
        t_rel = cur_inv @ np.array(fut_M)
        return float(t_rel[0, 3]), float(t_rel[1, 3])

    def transform_world_point_to_ego(cur_M, world_point):
        x_w, y_w = world_point
        cur_inv = np.linalg.inv(np.array(cur_M))
        p_world = np.array([y_w, -x_w, 0.0, 1.0], dtype=np.float64)
        p_ego = cur_inv @ p_world
        return float(p_ego[0]), float(p_ego[1])

    # Discover routes
    all_routes = sorted([
        os.path.join(carla_root, d) for d in os.listdir(carla_root)
        if os.path.isdir(os.path.join(carla_root, d)) and os.path.exists(os.path.join(carla_root, d, "measurements"))
    ])

    test_routes = [r for r in all_routes if "Town02" in r or "Town05" in r]
    train_val_routes = [r for r in all_routes if "Town02" not in r and "Town05" not in r]

    import random
    rng = random.Random(42)
    shuffled_tv = list(train_val_routes)
    rng.shuffle(shuffled_tv)
    n_val = max(1, len(shuffled_tv) // 5)
    val_routes = sorted(shuffled_tv[:n_val])
    train_routes = sorted(shuffled_tv[n_val:])

    print(f"Routes: {len(train_routes)} train, {len(val_routes)} val, {len(test_routes)} test", flush=True)

    def extract_from_routes(routes, max_samples=None):
        samples = []
        for r_dir in routes:
            m_dir = os.path.join(r_dir, "measurements")
            frames = sorted([f.replace(".json", "") for f in os.listdir(m_dir) if f.endswith(".json")])
            for i in range(2, len(frames) - 5):
                f_cur = frames[i]
                f_prev = frames[i - 1]
                try:
                    with open(os.path.join(m_dir, f"{f_cur}.json"), "r") as f:
                        d_cur = json.load(f)
                    with open(os.path.join(m_dir, f"{f_prev}.json"), "r") as f:
                        d_prev = json.load(f)

                    # Extract future waypoints
                    cur_M = d_cur["ego_matrix"]
                    cur_x = float(d_cur["x"])
                    cur_y = float(d_cur["y"])
                    x_cmd = float(d_cur.get("x_command", cur_x))
                    y_cmd = float(d_cur.get("y_command", cur_y))

                    wps = []
                    for st in range(1, 5):
                        with open(os.path.join(m_dir, f"{frames[i + st]}.json"), "r") as f:
                            d_fut = json.load(f)
                        wps.append(transform_future_ego_pose(cur_M, d_fut["ego_matrix"]))

                    # Causal kinematics: frame t and t-1 only
                    spd_t = float(d_cur.get("speed", 0.0))
                    spd_prev = float(d_prev.get("speed", spd_t))
                    ax_t = (spd_t - spd_prev) / 0.5

                    th_t = float(d_cur["theta"])
                    th_prev = float(d_prev["theta"])
                    d_th = wrap_angle(th_t - th_prev)
                    body_w_t = -d_th / 0.5

                    tgt_x, tgt_y = transform_world_point_to_ego(cur_M, (x_cmd, y_cmd))
                    tgt_norm_x = float(np.clip(tgt_x / 20.0, -3.0, 3.0))
                    tgt_norm_y = float(np.clip(tgt_y / 20.0, -3.0, 3.0))

                    steer = float(d_cur.get("steer", 0.0))
                    junction = bool(d_cur.get("junction", False))
                    veh_haz = bool(d_cur.get("vehicle_hazard", False))
                    light_haz = bool(d_cur.get("light_hazard", False))
                    stop_haz = bool(d_cur.get("stop_sign_hazard", False))

                    samples.append({
                        "speed": spd_t,
                        "accel": ax_t,
                        "body_yaw_rate": body_w_t,
                        "target_raw": [tgt_x, tgt_y],
                        "target_norm": [tgt_norm_x, tgt_norm_y],
                        "steer": steer,
                        "junction": junction,
                        "vehicle_hazard": veh_haz,
                        "light_hazard": light_haz,
                        "stop_sign_hazard": stop_haz,
                        "waypoints": np.array(wps, dtype=np.float32),
                    })
                    if max_samples and len(samples) >= max_samples:
                        return samples
                except Exception:
                    continue
        return samples

    print("Extracting test samples...", flush=True)
    test_samples = extract_from_routes(test_routes)
    print(f"Extracted {len(test_samples)} test samples.", flush=True)

    print("Extracting train and val samples for MLP...", flush=True)
    train_samples = extract_from_routes(train_routes, max_samples=15000)
    val_samples = extract_from_routes(val_routes, max_samples=4000)
    print(f"Extracted {len(train_samples)} train and {len(val_samples)} val samples.", flush=True)

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

    # Prepare Tensors
    X_tr_3 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in train_samples], dtype=torch.float32)
    X_tr_5_norm = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_norm"][0], s["target_norm"][1]] for s in train_samples], dtype=torch.float32)
    Y_tr = torch.tensor(np.array([s["waypoints"] for s in train_samples]), dtype=torch.float32)
    CC_tr = torch.tensor(np.array([compute_cc_traj(s["speed"], s["body_yaw_rate"]) for s in train_samples]), dtype=torch.float32)
    Y_tr_resid = Y_tr - CC_tr

    X_val_3 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in val_samples], dtype=torch.float32)
    X_val_5_norm = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_norm"][0], s["target_norm"][1]] for s in val_samples], dtype=torch.float32)
    Y_val = torch.tensor(np.array([s["waypoints"] for s in val_samples]), dtype=torch.float32)
    CC_val = torch.tensor(np.array([compute_cc_traj(s["speed"], s["body_yaw_rate"]) for s in val_samples]), dtype=torch.float32)
    Y_val_resid = Y_val - CC_val

    X_te_3 = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"]] for s in test_samples], dtype=torch.float32)
    X_te_5_norm = torch.tensor([[s["speed"], s["accel"], s["body_yaw_rate"], s["target_norm"][0], s["target_norm"][1]] for s in test_samples], dtype=torch.float32)
    CC_te = np.array([compute_cc_traj(s["speed"], s["body_yaw_rate"]) for s in test_samples], dtype=np.float32)

    # Helper to train MLP with validation early stopping
    def train_mlp_with_early_stopping(in_dim, x_train, y_train, x_val, y_val, max_epochs=30, patience=5):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )
        opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loader = DataLoader(TensorDataset(x_train, y_train), batch_size=64, shuffle=True)
        
        best_loss = float("inf")
        best_weights = None
        patience_cnt = 0

        for ep in range(max_epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                pred = model(bx).view(-1, 4, 2)
                loss = nn.L1Loss()(pred, by)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(x_val).view(-1, 4, 2)
                val_loss = float(nn.L1Loss()(val_pred, y_val))

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
                if patience_cnt >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        return model

    print("Training MLPs with Early Stopping...", flush=True)
    mlp_3 = train_mlp_with_early_stopping(3, X_tr_3, Y_tr, X_val_3, Y_val)
    mlp_5_norm = train_mlp_with_early_stopping(5, X_tr_5_norm, Y_tr, X_val_5_norm, Y_val)
    mlp_resid = train_mlp_with_early_stopping(5, X_tr_5_norm, Y_tr_resid, X_val_5_norm, Y_val_resid)

    with torch.no_grad():
        preds_mlp_3 = mlp_3(X_te_3).view(-1, 4, 2).numpy()
        preds_mlp_5_norm = mlp_5_norm(X_te_5_norm).view(-1, 4, 2).numpy()
        preds_mlp_resid = CC_te + mlp_resid(X_te_5_norm).view(-1, 4, 2).numpy()

    # Constant Velocity baseline
    preds_cv = np.zeros((len(test_samples), 4, 2), dtype=np.float32)
    for i, s in enumerate(test_samples):
        v = s["speed"]
        for h, t in enumerate(time_horizons):
            preds_cv[i, h, 0] = v * t

    gts = np.array([s["waypoints"] for s in test_samples], dtype=np.float32)
    steers = np.array([abs(s["steer"]) for s in test_samples])

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
            "mean_l1_mean": float(np.mean(l1_sum)),
            "mean_l1_med": float(np.median(l1_sum)),
        }

    baselines = {
        "Constant Velocity": preds_cv,
        "Constant Curvature": CC_te,
        "Kinematics MLP (3-in, No Goal)": preds_mlp_3,
        "Kinematics MLP (5-in, Norm Goal)": preds_mlp_5_norm,
        "Kinematics MLP (Residual over CC)": preds_mlp_resid,
    }

    full_m = np.ones(len(test_samples), dtype=bool)
    turn_m = steers > 0.05
    str_m = steers <= 0.05

    for subset_name, mask in [("OVERALL", full_m), ("TURNING (|steer| > 0.05)", turn_m), ("STRAIGHT (|steer| <= 0.05)", str_m)]:
        print("\n" + "="*115, flush=True)
        print(f"HELD-OUT TEST SET - {subset_name} | N = {np.sum(mask)}", flush=True)
        print("="*115, flush=True)
        print(f"{'Model':<36} | {'ADE (mean/med) [m]':<20} | {'FDE (mean/med) [m]':<20} | {'Lat-ADE (mean/med)':<20} | {'Mean L1 (sum)':<16}", flush=True)
        print("-"*115, flush=True)
        for name, p in baselines.items():
            m = calc_metrics(p[mask], gts[mask])
            print(f"{name:<36} | {m['ade_mean']:.4f} / {m['ade_med']:.4f} | {m['fde_mean']:.4f} / {m['fde_med']:.4f} | {m['lat_ade_mean']:.4f} / {m['lat_ade_med']:.4f} | {m['mean_l1_mean']:.4f} / {m['mean_l1_med']:.4f}", flush=True)
        print("="*115, flush=True)

@app.local_entrypoint()
def main():
    run_baselines.remote()

