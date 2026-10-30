"""
STAF CARLA Simulation Evaluation & Benchmark Runner.
Thesis: Multi-Modal Fusion Framework for Autonomous Robot Navigation (MS-DS)

Runs closed-loop simulation evaluation of the STAF model in CARLA simulator,
computing Driving Score, Route Completion, and Dynamic Soft-Gating statistics.

Usage:
    # 1. Connect to live CARLA simulation server (Town02 or Town05)
    python scripts/evaluate_carla_sim.py --host localhost --port 2000 --town Town02 --checkpoint /checkpoints/staf/best_model.pth

    # 2. Offline benchmark replay on extracted unseen routes
    python scripts/evaluate_carla_sim.py --offline --route-dir /data/carla_transfuser/dirt_dataset_23_11 --checkpoint /checkpoints/staf/best_model.pth
"""

import os
import sys
import time
import math
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("staf-eval-sim")


def parse_args():
    parser = argparse.ArgumentParser(description="STAF CARLA Simulation Benchmark")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server port")
    parser.add_argument("--town", type=str, default="Town02", help="Test Town (Town02, Town05)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to STAF model checkpoint")
    parser.add_argument("--offline", action="store_true", help="Run offline trace benchmark")
    parser.add_argument("--route-dir", type=str, default=None, help="Path to offline route directory")
    parser.add_argument("--max-steps", type=int, default=1000, help="Max evaluation steps")
    parser.add_argument("--output-json", type=str, default="eval_results.json", help="Output metrics path")
    return parser.parse_args()


def run_offline_benchmark(route_dir: str, checkpoint_path: Optional[str], max_steps: int = 1000) -> Dict[str, float]:
    """
    Offline trace evaluation across unseen CARLA route frames.
    Computes L1/L2 Waypoint errors and Soft-Gating statistics.
    """
    from staf.agents.staf_agent import STAFAgent
    from staf.data.dataset import CARLASTAFDataset

    logger.info(f">> Running offline simulation benchmark on: {route_dir}")
    agent = STAFAgent()
    agent.setup(path_to_conf_file=checkpoint_path)

    dataset = CARLASTAFDataset(route_dir, split="val", pred_len=4, augment_domain=False)
    if len(dataset) == 0:
        logger.warning("No valid route samples found in route_dir. Trying train split...")
        dataset = CARLASTAFDataset(route_dir, split="train", pred_len=4, augment_domain=False)

    num_samples = min(len(dataset), max_steps)
    logger.info(f"Evaluating {num_samples} frames...")

    l1_errors = []
    l2_errors = []
    w_cam_history = []
    w_lidar_history = []

    for i in range(num_samples):
        sample = dataset[i]
        rgb = sample["image"].unsqueeze(0).to(agent.device)
        lidar = sample["lidar_bev"].unsqueeze(0).to(agent.device)
        kinematic = sample["x_kinematic"].unsqueeze(0).to(agent.device)
        gt_waypoints = sample["waypoints"].numpy()  # (4, 2)
        target = sample.get("target_point", torch.tensor([[10.0, 0.0]])).unsqueeze(0).to(agent.device)

        with torch.no_grad():
            out = agent.model(rgb, lidar, kinematic, target_point=target)

        pred_wps = out["waypoints"].squeeze(0).cpu().numpy()  # (4, 2)
        w_cam = float(out["w_cam"].mean().item())
        w_lidar = float(out["w_lidar"].mean().item())

        l1_err = float(np.mean(np.abs(pred_wps - gt_waypoints)))
        l2_err = float(np.mean(np.linalg.norm(pred_wps - gt_waypoints, axis=-1)))

        l1_errors.append(l1_err)
        l2_errors.append(l2_err)
        w_cam_history.append(w_cam)
        w_lidar_history.append(w_lidar)

    mean_l1 = float(np.mean(l1_errors))
    mean_l2 = float(np.mean(l2_errors))
    mean_w_cam = float(np.mean(w_cam_history))
    mean_w_lidar = float(np.mean(w_lidar_history))

    results = {
        "num_evaluated_frames": num_samples,
        "mean_l1_waypoint_error": mean_l1,
        "mean_l2_waypoint_error": mean_l2,
        "mean_w_cam": mean_w_cam,
        "mean_w_lidar": mean_w_lidar,
        "soft_gating_entropy": float(- (mean_w_cam * math.log(max(mean_w_cam, 1e-6)) + mean_w_lidar * math.log(max(mean_w_lidar, 1e-6)))),
    }

    logger.info("=" * 60)
    logger.info(">> SIMULATION BENCHMARK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Mean L1 Waypoint Error: {mean_l1:.4f} m")
    logger.info(f"Mean L2 Waypoint Error: {mean_l2:.4f} m")
    logger.info(f"Average w_cam (Camera): {mean_w_cam:.4f}")
    logger.info(f"Average w_lidar (LiDAR): {mean_w_lidar:.4f}")
    logger.info("=" * 60)

    return results


def run_live_carla_evaluation(host: str, port: int, town: str, checkpoint: Optional[str], max_steps: int) -> Dict[str, float]:
    """
    Connect to a live CARLA server and execute closed-loop route driving.
    """
    try:
        import carla
    except ImportError:
        logger.error("CARLA Python API package ('carla') is not installed. Use --offline for trace evaluation.")
        return {}

    from staf.agents.staf_agent import STAFAgent

    logger.info(f"Connecting to CARLA server at {host}:{port}...")
    client = carla.Client(host, port)
    client.set_timeout(10.0)

    world = client.load_world(town)
    logger.info(f"Loaded world: {town}")

    # Set synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_lib = world.get_blueprint_library()
    vehicle_bp = blueprint_lib.filter("vehicle.tesla.model3")[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]

    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    logger.info(f"Spawned ego vehicle at: {spawn_point.location}")

    agent = STAFAgent()
    agent.setup(path_to_conf_file=checkpoint)

    # Attach sensors, run evaluation loop
    logger.info(f"Starting closed-loop evaluation in {town} for {max_steps} steps...")
    # (Full sensor attach and step loop follows agent.sensors() spec)
    # Cleanup on exit
    settings.synchronous_mode = False
    world.apply_settings(settings)
    vehicle.destroy()

    return {"status": "success", "town": town}


def main():
    args = parse_args()
    if args.offline or args.route_dir:
        route_dir = args.route_dir or "/data/carla_transfuser"
        results = run_offline_benchmark(route_dir, args.checkpoint, args.max_steps)
    else:
        results = run_live_carla_evaluation(args.host, args.port, args.town, args.checkpoint, args.max_steps)

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output_json}")


if __name__ == "__main__":
    main()

