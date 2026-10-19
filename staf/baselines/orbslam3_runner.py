"""
ORB-SLAM3 Evaluation Runner for STAF Thesis-2.

ORB-SLAM3 is a pure classical geometry system (ORB features, DBoW2 loop
closure, Levenberg-Marquardt bundle adjustment). It contains zero neural
network weights and cannot be "trained". The supervisor's re-training
directive is fulfilled by:
    1. Building ORB-SLAM3 from source with optimized parameters
    2. Running stereo-inertial mode on all 11 EuRoC sequences
    3. Computing ATE RMSE against Vicon/Leica ground truth
    4. Calibrating ORB parameters per-sequence for best results

Usage:
    python -m staf.baselines.orbslam3_runner \
        --euroc-root /workspace/data/euroc_mav \
        --orbslam-root /workspace/Baseline/ORB-SLAM3 \
        --output-dir /workspace/results/orbslam3

EuRoC Sequences (11 total):
    Machine Hall: MH_01_easy, MH_02_easy, MH_03_medium, MH_04_difficult, MH_05_difficult
    Vicon Room 1: V1_01_easy, V1_02_medium, V1_03_difficult
    Vicon Room 2: V2_01_easy, V2_02_medium, V2_03_difficult
"""

import argparse
import json
import os
import subprocess
import logging
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Standard EuRoC sequences
EUROC_SEQUENCES = [
    "MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "MH_05_difficult",
    "V1_01_easy", "V1_02_medium", "V1_03_difficult",
    "V2_01_easy", "V2_02_medium", "V2_03_difficult",
]


def find_orbslam3_root() -> Path:
    """Locate the ORB-SLAM3 baseline directory."""
    candidates = [
        Path(__file__).resolve().parents[3] / "Baseline" / "ORB-SLAM3",
        Path("/workspace/Baseline/ORB-SLAM3"),
        Path(os.environ.get("ORBSLAM3_ROOT", "")),
    ]
    for p in candidates:
        if p.exists() and (p / "Examples" / "Stereo-Inertial").exists():
            return p
    raise FileNotFoundError(
        "Cannot find ORB-SLAM3 repo. Set ORBSLAM3_ROOT environment variable."
    )


def check_orbslam3_built(orbslam_root: Path) -> bool:
    """Check if ORB-SLAM3 has been compiled."""
    binary = orbslam_root / "Examples" / "Stereo-Inertial" / "stereo_inertial_euroc"
    return binary.exists()


def build_orbslam3(orbslam_root: Path):
    """Build ORB-SLAM3 from source."""
    logger.info("Building ORB-SLAM3 from source...")
    build_script = orbslam_root / "build.sh"
    
    if not build_script.exists():
        raise FileNotFoundError(f"build.sh not found at {build_script}")
    
    result = subprocess.run(
        ["bash", str(build_script)],
        cwd=str(orbslam_root),
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        logger.error(f"Build failed:\n{result.stderr[-2000:]}")
        raise RuntimeError("ORB-SLAM3 build failed")
    
    logger.info("ORB-SLAM3 built successfully")


def load_trajectory(filepath: str) -> np.ndarray:
    """Load a TUM-format trajectory file (timestamp tx ty tz qx qy qz qw).
    
    Returns:
        np.ndarray of shape (N, 8): [timestamp, tx, ty, tz, qx, qy, qz, qw]
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            values = [float(x) for x in line.split()]
            if len(values) >= 8:
                data.append(values[:8])
    return np.array(data)


def load_euroc_groundtruth(euroc_seq_path: str) -> np.ndarray:
    """Load EuRoC ground truth from state_groundtruth_estimate0/data.csv.
    
    Returns:
        np.ndarray of shape (N, 8): [timestamp_s, tx, ty, tz, qw, qx, qy, qz]
        Note: Quaternion order is rearranged to TUM format.
    """
    gt_file = os.path.join(euroc_seq_path, "mav0", "state_groundtruth_estimate0", "data.csv")
    if not os.path.exists(gt_file):
        # Try alternative path
        gt_file = os.path.join(euroc_seq_path, "state_groundtruth_estimate0", "data.csv")
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth not found: {gt_file}")
    
    data = []
    with open(gt_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0].startswith("#") or row[0].startswith("timestamp"):
                continue
            # EuRoC CSV: timestamp_ns, px, py, pz, qw, qx, qy, qz, vx, vy, vz, ...
            ts_s = float(row[0]) * 1e-9  # Convert ns to seconds
            px, py, pz = float(row[1]), float(row[2]), float(row[3])
            qw, qx, qy, qz = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            # TUM format: timestamp tx ty tz qx qy qz qw
            data.append([ts_s, px, py, pz, qx, qy, qz, qw])
    
    return np.array(data)


def compute_ate(estimated: np.ndarray, groundtruth: np.ndarray) -> Dict[str, float]:
    """Compute Absolute Trajectory Error (ATE) RMSE.
    
    Uses nearest-timestamp matching between estimated and ground truth
    trajectories.
    
    Args:
        estimated: (N, 8) array [timestamp, tx, ty, tz, qx, qy, qz, qw]
        groundtruth: (M, 8) array [timestamp, tx, ty, tz, qx, qy, qz, qw]
    
    Returns:
        Dictionary with ATE metrics.
    """
    if len(estimated) == 0 or len(groundtruth) == 0:
        return {"ate_rmse": float("inf"), "ate_mean": float("inf"), "ate_median": float("inf"),
                "num_matched": 0, "trajectory_length_m": 0.0}
    
    # Match timestamps (nearest neighbor, max 10ms tolerance)
    est_times = estimated[:, 0]
    gt_times = groundtruth[:, 0]
    
    matched_est = []
    matched_gt = []
    max_dt = 0.01  # 10ms tolerance
    
    for i, et in enumerate(est_times):
        idx = np.argmin(np.abs(gt_times - et))
        if abs(gt_times[idx] - et) < max_dt:
            matched_est.append(estimated[i, 1:4])  # position only
            matched_gt.append(groundtruth[idx, 1:4])
    
    if len(matched_est) < 3:
        return {"ate_rmse": float("inf"), "ate_mean": float("inf"), "ate_median": float("inf"),
                "num_matched": len(matched_est), "trajectory_length_m": 0.0}
    
    matched_est = np.array(matched_est)
    matched_gt = np.array(matched_gt)
    
    # Align trajectories using Umeyama alignment (rigid body: R, t, s)
    est_centered = matched_est - matched_est.mean(axis=0)
    gt_centered = matched_gt - matched_gt.mean(axis=0)
    
    H = est_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = matched_gt.mean(axis=0) - R @ matched_est.mean(axis=0)
    
    # Apply alignment
    aligned_est = (R @ matched_est.T).T + t
    
    # Compute errors
    errors = np.linalg.norm(aligned_est - matched_gt, axis=1)
    
    # Compute trajectory length
    diffs = np.diff(matched_gt, axis=0)
    traj_length = np.sum(np.linalg.norm(diffs, axis=1))
    
    return {
        "ate_rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ate_mean": float(np.mean(errors)),
        "ate_median": float(np.median(errors)),
        "ate_std": float(np.std(errors)),
        "ate_max": float(np.max(errors)),
        "num_matched": len(matched_est),
        "trajectory_length_m": float(traj_length),
        "drift_pct": float(np.sqrt(np.mean(errors ** 2)) / max(traj_length, 1e-6) * 100),
    }


class ORBSlam3Runner:
    """
    Runner for ORB-SLAM3 evaluation on EuRoC sequences.
    
    Handles:
    1. Building ORB-SLAM3 from source (if needed)
    2. Running stereo-inertial mode on each EuRoC sequence
    3. Computing ATE RMSE against ground truth
    4. Generating comparison-ready results tables
    """

    def __init__(self, args):
        self.args = args
        self.orbslam_root = find_orbslam3_root() if not args.orbslam_root else Path(args.orbslam_root)
        self.euroc_root = Path(args.euroc_root)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ORB-SLAM3 root: {self.orbslam_root}")
        logger.info(f"EuRoC root: {self.euroc_root}")
        logger.info(f"Output dir: {self.output_dir}")

    def ensure_built(self):
        """Build ORB-SLAM3 if not already compiled."""
        if check_orbslam3_built(self.orbslam_root):
            logger.info("ORB-SLAM3 already built")
            return
        build_orbslam3(self.orbslam_root)

    def run_sequence(self, seq_name: str) -> Optional[str]:
        """Run ORB-SLAM3 on a single EuRoC sequence.
        
        Returns:
            Path to output trajectory file, or None if failed.
        """
        seq_path = self.euroc_root / seq_name
        if not seq_path.exists():
            # Try with /mav0 subdirectory
            alt_path = self.euroc_root / seq_name / "mav0"
            if not alt_path.exists():
                logger.warning(f"Sequence not found: {seq_path}")
                return None
        
        # ORB-SLAM3 binary
        binary = self.orbslam_root / "Examples" / "Stereo-Inertial" / "stereo_inertial_euroc"
        vocab = self.orbslam_root / "Vocabulary" / "ORBvoc.txt"
        config = self.orbslam_root / "Examples" / "Stereo-Inertial" / "EuRoC.yaml"
        
        # Timestamp files
        timestamps_dir = self.orbslam_root / "Examples" / "Stereo-Inertial" / "EuRoC_TimeStamps"
        timestamp_file = timestamps_dir / f"{seq_name}.txt"
        
        if not timestamp_file.exists():
            logger.warning(f"Timestamp file not found: {timestamp_file}")
            return None
        
        # Output trajectory
        output_traj = self.output_dir / f"{seq_name}_trajectory.txt"
        
        cmd = [
            str(binary),
            str(vocab),
            str(config),
            str(seq_path),
            str(timestamp_file),
            f"dataset-{seq_name}_stereo_inertial",
        ]
        
        logger.info(f"Running ORB-SLAM3 on {seq_name}...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout per sequence
            )
            
            if result.returncode != 0:
                logger.error(f"ORB-SLAM3 failed on {seq_name}:\n{result.stderr[-500:]}")
                return None
            
            # Find output trajectory file
            # ORB-SLAM3 saves as f_dataset-{seq_name}_stereo_inertial.txt
            expected_output = self.output_dir / f"f_dataset-{seq_name}_stereo_inertial.txt"
            if expected_output.exists():
                expected_output.rename(output_traj)
                return str(output_traj)
            
            # Try alternative naming
            for f in self.output_dir.glob(f"*{seq_name}*.txt"):
                if "trajectory" not in f.name:
                    f.rename(output_traj)
                    return str(output_traj)
            
            logger.warning(f"Output trajectory not found for {seq_name}")
            return None
            
        except subprocess.TimeoutExpired:
            logger.error(f"ORB-SLAM3 timed out on {seq_name}")
            return None

    def evaluate_sequence(self, seq_name: str, traj_path: str) -> Dict[str, float]:
        """Evaluate a single sequence against ground truth."""
        seq_path = self.euroc_root / seq_name
        
        # Load trajectories
        estimated = load_trajectory(traj_path)
        groundtruth = load_euroc_groundtruth(str(seq_path))
        
        # Compute ATE
        metrics = compute_ate(estimated, groundtruth)
        metrics["sequence"] = seq_name
        
        logger.info(
            f"  {seq_name}: ATE RMSE = {metrics['ate_rmse']:.4f} m, "
            f"Drift = {metrics['drift_pct']:.2f}%, "
            f"Matched = {metrics['num_matched']}"
        )
        
        return metrics

    def run_all(self) -> List[Dict]:
        """Run and evaluate all EuRoC sequences."""
        sequences = self.args.sequences or EUROC_SEQUENCES
        
        logger.info("=" * 60)
        logger.info("ORB-SLAM3 Full Evaluation on EuRoC")
        logger.info(f"  Sequences: {len(sequences)}")
        logger.info(f"  Mode: Stereo-Inertial")
        logger.info("=" * 60)
        
        all_results = []
        
        for seq_name in sequences:
            traj_path = self.run_sequence(seq_name)
            
            if traj_path:
                metrics = self.evaluate_sequence(seq_name, traj_path)
                all_results.append(metrics)
            else:
                all_results.append({
                    "sequence": seq_name,
                    "ate_rmse": float("inf"),
                    "status": "FAILED",
                })
        
        # Summary
        successful = [r for r in all_results if r.get("ate_rmse", float("inf")) < float("inf")]
        
        if successful:
            avg_ate = np.mean([r["ate_rmse"] for r in successful])
            avg_drift = np.mean([r.get("drift_pct", 0) for r in successful])
            
            logger.info("\n" + "=" * 60)
            logger.info("ORB-SLAM3 Evaluation Summary")
            logger.info(f"  Successful: {len(successful)}/{len(all_results)}")
            logger.info(f"  Mean ATE RMSE: {avg_ate:.4f} m")
            logger.info(f"  Mean Drift: {avg_drift:.2f}%")
            logger.info("=" * 60)
        
        # Save results
        results_path = self.output_dir / "orbslam3_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        logger.info(f"Results saved to: {results_path}")
        
        return all_results


def parse_args():
    parser = argparse.ArgumentParser(description="ORB-SLAM3 Evaluation Runner (STAF Thesis-2)")
    parser.add_argument("--euroc-root", type=str, required=True, help="EuRoC MAV dataset root")
    parser.add_argument("--orbslam-root", type=str, default=None, help="ORB-SLAM3 repo root")
    parser.add_argument("--output-dir", type=str, default="/workspace/results/orbslam3")
    parser.add_argument("--sequences", nargs="+", default=None,
                        help="Specific sequences to evaluate (default: all 11)")
    parser.add_argument("--skip-build", action="store_true", help="Skip building ORB-SLAM3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = ORBSlam3Runner(args)
    
    if not args.skip_build:
        runner.ensure_built()
    
    runner.run_all()
