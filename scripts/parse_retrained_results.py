import os
import json
import numpy as np

metrics_dir = "./retrained_metrics_dir"
variants = ["no_vision", "camera_only", "lidar_only", "full"]
seeds = [42, 43, 44]

data = {v: {} for v in variants}

for root, dirs, files in os.walk(metrics_dir):
    for f in files:
        if f.endswith(".json"):
            filepath = os.path.join(root, f)
            with open(filepath, "r") as fp:
                payload = json.load(fp)
                v = payload["variant"]
                s = payload["seed"]
                if v in data:
                    data[v][s] = payload

print("=== LOADED RUNS ===")
for v in variants:
    print(f"Variant: {v}, Seeds: {sorted(list(data[v].keys()))}")

def summarize_metric(v, subset_key, metric_name):
    vals = [data[v][s][subset_key][metric_name] for s in seeds if s in data[v]]
    return np.mean(vals), np.std(vals), vals

print("\n" + "="*115)
print("HELD-OUT TEST SET (Town02 & Town05) - OVERALL (N = 17,447 frames)")
print("="*115)
print(f"{'Modality':<20} | {'ADE Mean [m]':<18} | {'ADE Med [m]':<18} | {'FDE Mean [m]':<18} | {'FDE Med [m]':<18} | {'Lat-ADE [m]':<14}")
print("-"*115)

for v in variants:
    ade_m, ade_s, _ = summarize_metric(v, "metrics_overall", "ade_mean")
    ade_med_m, ade_med_s, _ = summarize_metric(v, "metrics_overall", "ade_med")
    fde_m, fde_s, _ = summarize_metric(v, "metrics_overall", "fde_mean")
    fde_med_m, fde_med_s, _ = summarize_metric(v, "metrics_overall", "fde_med")
    lat_m, lat_s, _ = summarize_metric(v, "metrics_overall", "lat_ade_mean")
    print(f"{v:<20} | {ade_m:.4f} +- {ade_s:.4f}   | {ade_med_m:.4f} +- {ade_med_s:.4f}   | {fde_m:.4f} +- {fde_s:.4f}   | {fde_med_m:.4f} +- {fde_med_s:.4f}   | {lat_m:.4f} +- {lat_s:.4f}")

print("\n" + "="*115)
print("HELD-OUT TEST SET - TURNING (|steer| > 0.05, N = 1,892 frames)")
print("="*115)
print(f"{'Modality':<20} | {'ADE Mean [m]':<18} | {'ADE Med [m]':<18} | {'FDE Mean [m]':<18} | {'FDE Med [m]':<18} | {'Lat-ADE [m]':<14}")
print("-"*115)
for v in variants:
    ade_m, ade_s, _ = summarize_metric(v, "metrics_turning", "ade_mean")
    ade_med_m, ade_med_s, _ = summarize_metric(v, "metrics_turning", "ade_med")
    fde_m, fde_s, _ = summarize_metric(v, "metrics_turning", "fde_mean")
    fde_med_m, fde_med_s, _ = summarize_metric(v, "metrics_turning", "fde_med")
    lat_m, lat_s, _ = summarize_metric(v, "metrics_turning", "lat_ade_mean")
    print(f"{v:<20} | {ade_m:.4f} +- {ade_s:.4f}   | {ade_med_m:.4f} +- {ade_med_s:.4f}   | {fde_m:.4f} +- {fde_s:.4f}   | {fde_med_m:.4f} +- {fde_med_s:.4f}   | {lat_m:.4f} +- {lat_s:.4f}")

print("\n" + "="*115)
print("HELD-OUT TEST SET - STRAIGHT (|steer| <= 0.05, N = 15,555 frames)")
print("="*115)
print(f"{'Modality':<20} | {'ADE Mean [m]':<18} | {'ADE Med [m]':<18} | {'FDE Mean [m]':<18} | {'FDE Med [m]':<18} | {'Lat-ADE [m]':<14}")
print("-"*115)
for v in variants:
    ade_m, ade_s, _ = summarize_metric(v, "metrics_straight", "ade_mean")
    ade_med_m, ade_med_s, _ = summarize_metric(v, "metrics_straight", "ade_med")
    fde_m, fde_s, _ = summarize_metric(v, "metrics_straight", "fde_mean")
    fde_med_m, fde_med_s, _ = summarize_metric(v, "metrics_straight", "fde_med")
    lat_m, lat_s, _ = summarize_metric(v, "metrics_straight", "lat_ade_mean")
    print(f"{v:<20} | {ade_m:.4f} +- {ade_s:.4f}   | {ade_med_m:.4f} +- {ade_med_s:.4f}   | {fde_m:.4f} +- {fde_s:.4f}   | {fde_med_m:.4f} +- {fde_med_s:.4f}   | {lat_m:.4f} +- {lat_s:.4f}")

print("\n" + "="*115)
print("ROUTE-LEVEL 95% BOOTSTRAP CONFIDENCE INTERVALS (500 resamples)")
print("="*115)
for v in variants:
    lows = [data[v][s]["bootstrap_ci"][0] for s in seeds if s in data[v]]
    highs = [data[v][s]["bootstrap_ci"][1] for s in seeds if s in data[v]]
    print(f"{v:<20} | 3-Seed Mean CI: [{np.mean(lows):.4f}, {np.mean(highs):.4f}] | Per-seed: {[f'[{l:.4f}, {h:.4f}]' for l, h in zip(lows, highs)]}")

print("\n" + "="*125)
print("HAZARD & DYNAMIC SUBSETS (ADE Mean +- Std across 3 seeds)")
print("="*125)
hazards = ["light_hazard", "vehicle_hazard", "junction", "speed_change"]
for v in variants:
    h_strs = []
    for h in hazards:
        vals = [data[v][s]["hazard_slices"][h] for s in seeds if s in data[v]]
        h_strs.append(f"{h}: {np.mean(vals):.4f}+-{np.std(vals):.4f}")
    print(f"{v:<20} | " + " | ".join(h_strs))

print("\n" + "="*115)
print("PER-SEED BREAKDOWN (ADE Mean [m] on Test)")
print("="*115)
for v in variants:
    ades = [data[v][s]["metrics_overall"]["ade_mean"] for s in seeds]
    fdes = [data[v][s]["metrics_overall"]["fde_mean"] for s in seeds]
    turn_ades = [data[v][s]["metrics_turning"]["ade_mean"] for s in seeds]
    print(f"{v:<20} | ADE: 42={ades[0]:.4f}, 43={ades[1]:.4f}, 44={ades[2]:.4f} | Turn-ADE: 42={turn_ades[0]:.4f}, 43={turn_ades[1]:.4f}, 44={turn_ades[2]:.4f}")
