import json
import os
import numpy as np

audit_path = "results/v2_audit/audit_summary.json"
with open(audit_path, "r") as f:
    data = json.load(f)

b1 = data["audit_b1"]
reeval = data["reeval_group1"]
baselines = data["baselines"]
degradations = data["degradations"]

print("=" * 100)
print("AUDIT B1: TEST SET RECONCILIATION")
print("=" * 100)
print(f"Total Unique Test Frames (held-out Town02 & Town05): {b1['total_test_frames']}")
print(f"Total Unique Test Routes:                             {b1['num_routes']}")
print(f"Total Turning Frames (|steer| > 0.05):                {b1['total_turning_frames']}")
print(f"Total Straight Frames (|steer| <= 0.05):              {b1['total_straight_frames']}")
print(f"RouteOrderedBatchSampler (B=8) items yielded:         {b1['sampler_total_items']}")
print(f"Calculated Padding / Repeat Frames:                   {b1['total_padded_repeat_frames']}")
print(f"Verification: {b1['total_test_frames']} + {b1['total_padded_repeat_frames']} == {b1['sampler_total_items']}: {b1['total_test_frames'] + b1['total_padded_repeat_frames'] == b1['sampler_total_items']}")

print("\nChunk Padding Breakdown (first 5 chunks):")
for c_idx, c_size, min_l, max_l, pad in b1["per_chunk_padding"][:5]:
    print(f"  Chunk {c_idx:2d} (size {c_size}): lengths {min_l}..{max_l} -> padded {pad} frames")

print("\n" + "=" * 100)
print("AUDIT B1 & GROUP 1 RE-EVALUATION (EXACT 17,447 UNPADDED FRAMES)")
print("=" * 100)

var_a = [v for k, v in reeval.items() if "group1_A" in k]
var_b = [v for k, v in reeval.items() if "group1_B" in k]

def summarize_variant(items, label):
    ades = [it["overall"]["ade_mean"] for it in items]
    meds = [it["overall"]["ade_med"] for it in items]
    fdes = [it["overall"]["fde_mean"] for it in items]
    turn_ades = [it["turning"]["ade_mean"] for it in items]
    turn_meds = [it["turning"]["ade_med"] for it in items]
    turn_fdes = [it["turning"]["fde_mean"] for it in items]
    turn_lats = [it["turning"]["lat_ade_mean"] for it in items]
    str_ades = [it["straight"]["ade_mean"] for it in items]
    m_flickers = [it["flicker"]["model_flicker"] for it in items]
    gt_flickers = [it["flicker"]["gt_flicker"] for it in items]
    w_cam_means = [it["gate"]["w_cam_mean"] for it in items]
    w_cam_stds = [it["gate"]["w_cam_std"] for it in items]
    corr_spd_p = [it["gate"]["corr_speed_pearson"] for it in items]
    corr_spd_s = [it["gate"]["corr_speed_spearman"] for it in items]
    corr_yaw_p = [it["gate"]["corr_yaw_pearson"] for it in items]
    corr_yaw_s = [it["gate"]["corr_yaw_spearman"] for it in items]

    print(f"--- {label} (N={len(items)} seeds: 42, 43, 44) ---")
    for it in items:
        s = it["seed"]
        print(f"  Seed {s}: Overall ADE={it['overall']['ade_mean']:.4f} m (med: {it['overall']['ade_med']:.4f}) | Turn ADE={it['turning']['ade_mean']:.4f} m (med: {it['turning']['ade_med']:.4f}) | Transformed Flicker={it['flicker']['model_flicker']:.4f} m (GT={it['flicker']['gt_flicker']:.4f} m) | Gate w_cam={it['gate']['w_cam_mean']:.4f}+-{it['gate']['w_cam_std']:.4f}")

    print(f"  Aggregate (Mean +/- Std across 3 seeds):")
    print(f"    Overall ADE:        {np.mean(ades):.4f} +/- {np.std(ades, ddof=1):.4f} m (Median: {np.mean(meds):.4f} m)")
    print(f"    Overall FDE:        {np.mean(fdes):.4f} +/- {np.std(fdes, ddof=1):.4f} m")
    print(f"    Turning ADE:        {np.mean(turn_ades):.4f} +/- {np.std(turn_ades, ddof=1):.4f} m (Median: {np.mean(turn_meds):.4f} m)")
    print(f"    Turning FDE:        {np.mean(turn_fdes):.4f} +/- {np.std(turn_fdes, ddof=1):.4f} m")
    print(f"    Turning Lateral ADE:{np.mean(turn_lats):.4f} +/- {np.std(turn_lats, ddof=1):.4f} m")
    print(f"    Straight ADE:       {np.mean(str_ades):.4f} +/- {np.std(str_ades, ddof=1):.4f} m")
    print(f"    Transformed Flicker:{np.mean(m_flickers):.4f} +/- {np.std(m_flickers, ddof=1):.4f} m (GT: {np.mean(gt_flickers):.4f} m)")
    print(f"    Gate w_cam:         {np.mean(w_cam_means):.4f} +/- {np.std(w_cam_means, ddof=1):.4f} (std across frames: {np.mean(w_cam_stds):.4f})")
    print(f"    w_cam vs Speed:     Pearson r={np.mean(corr_spd_p):.4f}, Spearman rho={np.mean(corr_spd_s):.4f}")
    print(f"    w_cam vs |Yaw Rate|:Pearson r={np.mean(corr_yaw_p):.4f}, Spearman rho={np.mean(corr_yaw_s):.4f}")

summarize_variant(var_a, "Variant A: Full Fusion, Learned Gate (Q=5)")
print()
summarize_variant(var_b, "Variant B: Full Fusion, Constant Gate (w=0.5, Q=5)")

print("\n" + "=" * 100)
print("AUDIT B5: RECOMPUTED BASELINES (EXACT 17,447 TEST FRAMES)")
print("=" * 100)
for b_key, b_val in baselines.items():
    ov = b_val["overall"]
    tu = b_val["turning"]
    st = b_val["straight"]
    print(f"{b_val['name']:<40} | Overall ADE: {ov['ade_mean']:.4f} m (med: {ov['ade_med']:.4f}) | Turning ADE: {tu['ade_mean']:.4f} m (med: {tu['ade_med']:.4f}, lat: {tu['lat_ade_mean']:.4f}) | Straight: {st['ade_mean']:.4f} m")

print("\n" + "=" * 100)
print("REQUIREMENT D: MOTION-COUPLED BLUR & SENSOR DEGRADATIONS")
print("=" * 100)
for deg_k, deg_v in degradations.items():
    print(f"  {deg_k:<25} | Learned Gate ADE: {deg_v['learned_gate_ade']:.4f} m (w_cam={deg_v['learned_gate_w_cam']:.3f}) | Constant Gate ADE: {deg_v['constant_gate_ade']:.4f} m (w_cam={deg_v['constant_gate_w_cam']:.3f})")

print("\n" + "=" * 100)
print("REQUIREMENT C: ROUTE-LEVEL PAIRED BOOTSTRAP CI (B=5000)")
print("=" * 100)

import random
def run_paired_bootstrap(r_map_1, r_map_2, n_boot=5000, seed=42):
    rng = random.Random(seed)
    common_routes = sorted(list(set(r_map_1.keys()).intersection(set(r_map_2.keys()))))
    d1 = np.array([r_map_1[r] for r in common_routes])
    d2 = np.array([r_map_2[r] for r in common_routes])
    diffs = d1 - d2
    mean_diff = float(np.mean(diffs))

    boot_diffs = []
    n = len(diffs)
    for _ in range(n_boot):
        sample_diffs = rng.choices(diffs, k=n)
        boot_diffs.append(np.mean(sample_diffs))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    return len(common_routes), mean_diff, ci_lower, ci_upper

# Extract average route ADEs across the 3 seeds
common_routes = sorted(list(var_a[0]["route_ades"].keys()))
r_ades_a = {r: np.mean([it["route_ades"][r] for it in var_a]) for r in common_routes}
r_ades_b = {r: np.mean([it["route_ades"][r] for it in var_b]) for r in common_routes}

r_turn_routes = sorted(list(set(var_a[0]["route_turn_ades"].keys()).intersection(set(var_b[0]["route_turn_ades"].keys()))))
r_turn_ades_a = {r: np.mean([it["route_turn_ades"][r] for it in var_a]) for r in r_turn_routes}
r_turn_ades_b = {r: np.mean([it["route_turn_ades"][r] for it in var_b]) for r in r_turn_routes}

# Pair A vs B
n_r, diff, lo, hi = run_paired_bootstrap(r_ades_a, r_ades_b)
print(f"Overall ADE: Variant A - Variant B (N_routes={n_r}): Mean Diff = {diff:+.4f} m | 95% CI: [{lo:+.4f}, {hi:+.4f}] m | Crosses Zero: {lo <= 0 <= hi}")

n_rt, diff_t, lo_t, hi_t = run_paired_bootstrap(r_turn_ades_a, r_turn_ades_b)
print(f"Turning ADE: Variant A - Variant B (N_routes={n_rt}): Mean Diff = {diff_t:+.4f} m | 95% CI: [{lo_t:+.4f}, {hi_t:+.4f}] m | Crosses Zero: {lo_t <= 0 <= hi_t}")

# Pair Variant A vs Baselines
for b_key, b_name in [("mlp_goal", "Kinematics MLP (With Goal)"), ("residual_cc", "Residual over CC"), ("cc", "Constant Curvature (CC)"), ("cv", "Constant Velocity (CV)")]:
    b_r = baselines[b_key]["route_ades"]
    b_rt = baselines[b_key]["route_turn_ades"]
    n_r, diff, lo, hi = run_paired_bootstrap(r_ades_a, b_r)
    n_rt, diff_t, lo_t, hi_t = run_paired_bootstrap(r_turn_ades_a, b_rt)
    print(f"Overall ADE: Variant A - {b_name} (N={n_r}): Diff = {diff:+.4f} m | 95% CI: [{lo:+.4f}, {hi:+.4f}] m")
    print(f"Turning ADE: Variant A - {b_name} (N={n_rt}): Diff = {diff_t:+.4f} m | 95% CI: [{lo_t:+.4f}, {hi_t:+.4f}] m")
