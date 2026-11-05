import json
import glob
import os
import numpy as np

files = sorted([f for f in glob.glob("results/v2_retrained_variants/group1_*.json") if not f.endswith("_config.json")])

runs = {}
for f in files:
    name = os.path.basename(f).replace(".json", "")
    with open(f) as fp:
        d = json.load(fp)
    runs[name] = d

groups = {}
for name, d in sorted(runs.items()):
    c = d["config"]
    var_base = f"{c['variant']}_{c['gate_mode']}_q{c['queue_length']}"
    if var_base not in groups:
        groups[var_base] = []
    groups[var_base].append((name, d))

for base, items in sorted(groups.items()):
    print("=" * 110)
    print(f"HAZARD SLICES & BEHAVIOR: {base} (N={len(items)} seeds)")
    print("=" * 110)
    reds = [it[1]["hazard_slices"]["red_light_ade"] for it in items]
    juncs = [it[1]["hazard_slices"]["junction_ade"] for it in items]
    speeds = [it[1]["hazard_slices"]["speed_change_ade"] for it in items]

    for name, d in items:
        c = d["config"]
        s = c["seed"]
        hz = d["hazard_slices"]
        print(f"  [Seed {s}] Red-Light ADE: {hz['red_light_ade']:.4f} m | Junction ADE: {hz['junction_ade']:.4f} m | Speed-Change ADE: {hz['speed_change_ade']:.4f} m")
    
    print("-" * 110)
    print(f"  Aggregate Hazard Slices (Mean +/- Std):")
    print(f"    Red-Light ADE:    {np.mean(reds):.4f} +/- {np.std(reds, ddof=1):.4f} m")
    print(f"    Junction ADE:     {np.mean(juncs):.4f} +/- {np.std(juncs, ddof=1):.4f} m")
    print(f"    Speed-Change ADE: {np.mean(speeds):.4f} +/- {np.std(speeds, ddof=1):.4f} m")
print("=" * 110)

