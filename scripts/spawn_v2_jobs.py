"""
Spawn detached background training jobs on deployed Modal app 'staf-v2-training-suite'.
Runs independently on Modal's cloud GPU without depending on local network stability.
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import json
import modal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()

    fn = modal.Function.from_name("staf-v2-training-suite", "train_and_eval_v2")
    seeds = [42, 43, 44]
    jobs = []

    if args.group == 1:
        for s in seeds:
            jobs.append({
                "variant": "full",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group1_A_full_learned_q5_seed{s}",
            })
            jobs.append({
                "variant": "full",
                "gate_mode": "constant",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group1_B_full_constant_q5_seed{s}",
            })
    elif args.group == 2:
        for s in seeds:
            jobs.append({
                "variant": "full",
                "gate_mode": "learned",
                "queue_length": 1,
                "seed": s,
                "run_id": f"group2_C_full_learned_q1_seed{s}",
            })
    elif args.group == 3:
        for s in seeds:
            jobs.append({
                "variant": "camera_only",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_D_camera_only_q5_seed{s}",
            })
            jobs.append({
                "variant": "lidar_only",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_E_lidar_only_q5_seed{s}",
            })
            jobs.append({
                "variant": "no_vision",
                "gate_mode": "learned",
                "queue_length": 5,
                "seed": s,
                "run_id": f"group3_F_no_vision_q5_seed{s}",
            })

    print("=" * 80)
    print(f">> Spawning {len(jobs)} Detached Cloud Jobs for Group {args.group}...")
    print("=" * 80)
    spawned = []
    for j in jobs:
        call = fn.spawn(j)
        spawned.append({"run_id": j["run_id"], "call_id": call.object_id, "config": j})
        print(f"  [SPAWNED] {j['run_id']:<35} -> Call ID: {call.object_id}")

    print("\n" + "=" * 80)
    print(">> All jobs queued in Modal Cloud.")
    print(">> Dashboard: https://modal.com/apps/mtalikhan08/main/deployed/staf-v2-training-suite")
    print(">> Results will be written to Modal Volume 'staf-checkpoints' under /v2_metrics/")
    print("=" * 80)

if __name__ == "__main__":
    main()

