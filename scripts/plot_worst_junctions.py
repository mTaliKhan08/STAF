import os
import json
import numpy as np
import matplotlib.pyplot as plt

json_path = "junction_paired_analysis.json"
out_fig_path = "C:/Users/talha/.gemini/antigravity/brain/07f55711-0fdd-48da-a54f-383c203e24a9/junction_10_worst_examples.png"

if not os.path.exists(json_path):
    print(f"Waiting for {json_path}...")
    exit(1)

with open(json_path, "r") as f:
    data = json.load(f)

worst = data["worst_10_junctions"]
print(f"Loaded {len(worst)} worst junction examples.")

fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.flatten()

for i, ex in enumerate(worst):
    ax = axes[i]
    gt = np.array(ex["gt_waypoints"])        # (4, 2): [x_fwd, y_lat]
    nv = np.array(ex["novis_waypoints"])     # (4, 2)
    vi = np.array(ex["vision_waypoints"])    # (4, 2)

    # Origin (ego vehicle)
    ax.plot(0, 0, marker="o", color="black", markersize=8, label="Ego (0,0)")

    # Trajectories: x is forward (vertical axis), y is lateral (horizontal axis)
    # BEV convention: x vertical (forward), y horizontal (left)
    ax.plot(gt[:, 1], gt[:, 0], "-o", color="#2ca02c", linewidth=2.5, markersize=6, label="Ground Truth")
    ax.plot(nv[:, 1], nv[:, 0], "--s", color="#1f77b4", linewidth=2.0, markersize=5, label=f"No-Vision (ADE={ex['novis_ade']:.2f}m)")
    ax.plot(vi[:, 1], vi[:, 0], ":^", color="#d62728", linewidth=2.0, markersize=5, label=f"Vision (ADE={ex['vision_ade']:.2f}m)")

    route_short = os.path.basename(ex["route"])
    ax.set_title(f"Rank {ex['rank']}: Steer={ex['steer']:.2f}\n{route_short} (Idx {ex['orig_idx']})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Lateral (Left +y) [m]", fontsize=8)
    ax.set_ylabel("Forward (+x) [m]", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    if i == 0:
        ax.legend(loc="upper left", fontsize=8)

plt.suptitle("Top 10 Worst Junction Error Cases (Vision vs No-Vision vs Ground Truth in Ego BEV Frame)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(out_fig_path, dpi=200, bbox_inches="tight")
print(f"Saved figure to {out_fig_path}")

