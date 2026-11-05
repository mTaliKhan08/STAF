import json
import glob
import os
import numpy as np

files = sorted([f for f in glob.glob("results/v2_retrained_variants/group1_*.json") if not f.endswith("_config.json")])

print(f"Total result files found: {len(files)}")
runs = {}
for f in files:
    name = os.path.basename(f).replace(".json", "")
    with open(f) as fp:
        d = json.load(fp)
    runs[name] = d

# Group by variant
groups = {}
for name, d in sorted(runs.items()):
    c = d["config"]
    var_base = f"{c['variant']}_{c['gate_mode']}_q{c['queue_length']}"
    if var_base not in groups:
        groups[var_base] = []
    groups[var_base].append((name, d))

for base, items in sorted(groups.items()):
    print("=" * 110)
    print(f"VARIANT: {base} (N={len(items)} seeds)")
    print("=" * 110)
    
    val_ades = [it[1]["best_val_ade"] for it in items]
    test_ov_ades = [it[1]["overall"]["ade_mean"] for it in items]
    test_ov_meds = [it[1]["overall"]["ade_med"] for it in items]
    test_ov_fdes = [it[1]["overall"]["fde_mean"] for it in items]
    
    test_tu_ades = [it[1]["turning"]["ade_mean"] for it in items]
    test_tu_meds = [it[1]["turning"]["ade_med"] for it in items]
    test_tu_fdes = [it[1]["turning"]["fde_mean"] for it in items]
    test_tu_lats = [it[1]["turning"]["lat_ade_mean"] for it in items]

    test_st_ades = [it[1]["straight"]["ade_mean"] for it in items]
    
    flickers = [it[1]["flicker"] for it in items]
    w_cam_means = [it[1]["gate_stats"]["w_cam_mean"] for it in items]
    w_cam_stds = [it[1]["gate_stats"]["w_cam_std"] for it in items]

    for name, d in items:
        c = d["config"]
        s = c["seed"]
        v_ade = d["best_val_ade"]
        t_ade = d["overall"]["ade_mean"]
        t_med = d["overall"]["ade_med"]
        t_fde = d["overall"]["fde_mean"]
        tu_ade = d["turning"]["ade_mean"]
        tu_med = d["turning"]["ade_med"]
        tu_fde = d["turning"]["fde_mean"]
        tu_lat = d["turning"]["lat_ade_mean"]
        st_ade = d["straight"]["ade_mean"]
        flk = d["flicker"]
        w_mean = d["gate_stats"]["w_cam_mean"]
        w_std = d["gate_stats"]["w_cam_std"]
        n_epochs = len(d["epoch_logs"])

        print(f"  [Seed {s}] Epochs: {n_epochs} | Val ADE: {v_ade:.4f} m | Test ADE: {t_ade:.4f} m (med: {t_med:.4f}) | FDE: {t_fde:.4f} m")
        print(f"            Turning ADE: {tu_ade:.4f} m (med: {tu_med:.4f}, lat: {tu_lat:.4f}) | FDE: {tu_fde:.4f} m | Straight ADE: {st_ade:.4f} m")
        print(f"            Flicker: {flk:.4f} m | Gate w_cam: {w_mean:.4f} +/- {w_std:.4f}")
    
    print("-" * 110)
    print(f"  Aggregate Summary across {len(items)} seeds:")
    print(f"    Val ADE (best):     {np.mean(val_ades):.4f} +/- {np.std(val_ades, ddof=1):.4f} m")
    print(f"    Test Overall ADE:   {np.mean(test_ov_ades):.4f} +/- {np.std(test_ov_ades, ddof=1):.4f} m  (Median: {np.mean(test_ov_meds):.4f})")
    print(f"    Test Overall FDE:   {np.mean(test_ov_fdes):.4f} +/- {np.std(test_ov_fdes, ddof=1):.4f} m")
    print(f"    Test Turning ADE:   {np.mean(test_tu_ades):.4f} +/- {np.std(test_tu_ades, ddof=1):.4f} m  (Median: {np.mean(test_tu_meds):.4f})")
    print(f"    Test Turning FDE:   {np.mean(test_tu_fdes):.4f} +/- {np.std(test_tu_fdes, ddof=1):.4f} m")
    print(f"    Test Turning Lat:   {np.mean(test_tu_lats):.4f} +/- {np.std(test_tu_lats, ddof=1):.4f} m")
    print(f"    Test Straight ADE:  {np.mean(test_st_ades):.4f} +/- {np.std(test_st_ades, ddof=1):.4f} m")
    print(f"    Trajectory Flicker: {np.mean(flickers):.4f} +/- {np.std(flickers, ddof=1):.4f} m")
    print(f"    Gate w_cam Mean:    {np.mean(w_cam_means):.4f} +/- {np.std(w_cam_means, ddof=1):.4f}")
    print(f"    Gate w_cam Std:     {np.mean(w_cam_stds):.4f} +/- {np.std(w_cam_stds, ddof=1):.4f}")
print("=" * 110)
