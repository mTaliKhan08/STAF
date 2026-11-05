"""
Sync results from Modal volume 'staf-checkpoints' (/v2_metrics/) to local results/v2_retrained_variants/.
Prints progress and summary table for completed runs.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import os
import json
import numpy as np
import modal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STAF_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
LOCAL_OUT_DIR = os.path.join(STAF_ROOT, "results", "v2_retrained_variants")
os.makedirs(LOCAL_OUT_DIR, exist_ok=True)

def main():
    vol = modal.Volume.from_name("staf-checkpoints")
    print(f">> Inspecting Modal Volume 'staf-checkpoints' in /v2_metrics/ ...")
    
    try:
        entries = vol.listdir("v2_metrics")
    except Exception as e:
        print(f"Error accessing volume: {e}")
        return

    files = [e.path for e in entries if e.path.endswith(".json")]
    print(f">> Found {len(files)} result files in cloud volume.")

    synced = []
    for f_path in files:
        fname = os.path.basename(f_path)
        local_path = os.path.join(LOCAL_OUT_DIR, fname)
        # Download from volume
        with open(local_path, "wb") as f_out:
            for chunk in vol.read_file(f_path):
                f_out.write(chunk)

        with open(local_path, "r") as f_in:
            data = json.load(f_in)
        synced.append(data)

        # Write config JSON beside each run
        cfg_path = os.path.join(LOCAL_OUT_DIR, f"{data['run_id']}_config.json")
        with open(cfg_path, "w") as f_cfg:
            json.dump(data["config"], f_cfg, indent=2)

    print(f">> Successfully synced {len(synced)} runs to {LOCAL_OUT_DIR}!")

    if not synced:
        print(">> No runs completed yet.")
        return

    # Group by base run type
    print("\n" + "=" * 115)
    print("CURRENT v2 RUN RESULTS SUMMARY TABLE across Seeds")
    print("=" * 115)
    print(f"{'Run / Model Base':<35} | {'Overall ADE [m]':<20} | {'Turning ADE [m]':<20} | {'Flicker (L2)':<16} | {'N Seeds':<8}")
    print("-" * 115)

    grouped = {}
    for r in synced:
        base = "_".join(r["run_id"].split("_")[:-1])
        if base not in grouped:
            grouped[base] = []
        grouped[base].append(r)

    for base, runs in grouped.items():
        ades = [rn["overall"]["ade_mean"] for rn in runs]
        turn_ades = [rn["turning"]["ade_mean"] for rn in runs]
        flickers = [rn["flicker"] for rn in runs]

        ade_str = f"{np.mean(ades):.4f} +- {np.std(ades):.4f}"
        turn_str = f"{np.mean(turn_ades):.4f} +- {np.std(turn_ades):.4f}"
        flicker_str = f"{np.mean(flickers):.4f}"

        print(f"{base:<35} | {ade_str:<20} | {turn_str:<20} | {flicker_str:<16} | {len(runs):<8}")

    print("=" * 115)

if __name__ == "__main__":
    main()

