import os
import json
import numpy as np

variants = ['no_vision', 'camera_only', 'lidar_only', 'full']
seeds = [42, 43, 44]

print("=== TURNING FRAMES: FORWARD VS LATERAL ADE ===")
for v in variants:
    ades = []
    lat_ades = []
    fwd_ades = []
    for s in seeds:
        with open(f'retrained_metrics_dir/{v}_seed{s}.json') as f:
            d = json.load(f)['metrics_turning']
            ades.append(d['ade_mean'])
            lat_ades.append(d['lat_ade_mean'])
            # Mean L1 is mean(|dx| + |dy|), so mean(|dx|) = Mean L1 - Mean Lat ADE
            fwd_ades.append(d['mean_l1_mean'] - d['lat_ade_mean'])
    print(f"{v:<15} | ADE: {np.mean(ades):.4f} +/- {np.std(ades, ddof=1):.4f} | Fwd-ADE: {np.mean(fwd_ades):.4f} +/- {np.std(fwd_ades, ddof=1):.4f} | Lat-ADE: {np.mean(lat_ades):.4f} +/- {np.std(lat_ades, ddof=1):.4f}")

print("\n=== OVERALL TEST SET: SAMPLE STD (DDOF=1) ===")
for v in variants:
    ades = []
    fdes = []
    lat_ades = []
    for s in seeds:
        with open(f'retrained_metrics_dir/{v}_seed{s}.json') as f:
            d = json.load(f)['metrics_overall']
            ades.append(d['ade_mean'])
            fdes.append(d['fde_mean'])
            lat_ades.append(d['lat_ade_mean'])
    print(f"{v:<15} | ADE: {np.mean(ades):.4f} +/- {np.std(ades, ddof=1):.4f} | FDE: {np.mean(fdes):.4f} +/- {np.std(fdes, ddof=1):.4f} | Lat-ADE: {np.mean(lat_ades):.4f} +/- {np.std(lat_ades, ddof=1):.4f}")

