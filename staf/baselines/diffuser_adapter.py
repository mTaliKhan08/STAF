"""
DifFUSER Baseline Training Adapter for STAF Thesis-2.

Wraps the original DifFUSER (mmdetection3d + torchpack) training pipeline
with cost optimizations for re-training on RunPod (1x RTX 4090, 24GB).

The DifFUSER codebase uses torchpack's distributed launcher, so this
adapter generates the correct launch command and config overrides rather
than reimplementing the training loop.

Usage (on RunPod):
    python -m staf.baselines.diffuser_adapter \
        --data-root /workspace/data/nuscenes \
        --logdir /workspace/checkpoints/diffuser \
        --epochs 12 --batch-size 4 --use-amp

Original DifFUSER:
    - 20 epochs, LR=2e-4 (CyclicLR), BS=4/GPU
    - torchpack dist.launch for distributed training
    - mmdetection3d registry for model/dataset building
"""

import argparse
import json
import os
import sys
import subprocess
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_diffuser_root() -> Path:
    """Locate the DifFUSER baseline directory."""
    candidates = [
        Path(__file__).resolve().parents[3] / "Baseline" / "DifFUSER",
        Path("/workspace/Baseline/DifFUSER"),
        Path(os.environ.get("DIFFUSER_ROOT", "")),
    ]
    for p in candidates:
        if p.exists() and (p / "tools" / "train.py").exists():
            return p
    raise FileNotFoundError(
        "Cannot find DifFUSER repo. Set DIFFUSER_ROOT environment variable."
    )


def prepare_nuscenes_data(data_root: str, diffuser_root: Path):
    """Ensure nuScenes data is symlinked/accessible from DifFUSER's expected path.
    
    DifFUSER expects data at: <diffuser_root>/data/nuscenes/
    If the actual data is elsewhere, create a symlink.
    """
    expected_path = diffuser_root / "data" / "nuscenes"
    
    if expected_path.exists() or expected_path.is_symlink():
        logger.info(f"nuScenes data already accessible at: {expected_path}")
        return
    
    # Create symlink
    expected_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(data_root, str(expected_path))
    logger.info(f"Created symlink: {expected_path} -> {data_root}")


def create_nuscenes_infos(diffuser_root: Path, data_root: str):
    """Generate nuScenes info pickle files if they don't exist.
    
    These are pre-processed metadata files required by mmdetection3d.
    """
    info_file = os.path.join(data_root, "nuscenes_infos_train.pkl")
    if os.path.exists(info_file):
        logger.info("nuScenes info files already exist, skipping generation")
        return
    
    logger.info("Generating nuScenes info files (this may take 10-30 minutes)...")
    create_data_script = diffuser_root / "tools" / "create_data.py"
    
    cmd = [
        sys.executable, str(create_data_script),
        "nuscenes",
        "--root-path", data_root,
        "--out-dir", data_root,
        "--extra-tag", "nuscenes",
    ]
    
    result = subprocess.run(cmd, cwd=str(diffuser_root), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"create_data.py failed:\n{result.stderr}")
        raise RuntimeError("Failed to generate nuScenes info files")
    logger.info("nuScenes info files generated successfully")


def build_training_command(
    diffuser_root: Path,
    config_path: str,
    run_dir: str,
    num_gpus: int = 1,
    use_amp: bool = True,
    extra_opts: list = None,
) -> list:
    """Build the torchpack launch command for DifFUSER training.
    
    DifFUSER uses torchpack's distributed launcher:
        torchpack dist-run -np <N> python tools/train.py <config> [opts]
    """
    cmd = [
        "torchpack", "dist-run",
        "-np", str(num_gpus),
        sys.executable, str(diffuser_root / "tools" / "train.py"),
        config_path,
        "--run-dir", run_dir,
        "--nowandb",  # Disable wandb (we use our own logging)
    ]
    
    if extra_opts:
        cmd.extend(extra_opts)
    
    return cmd


def build_eval_command(
    diffuser_root: Path,
    config_path: str,
    checkpoint_path: str,
    num_gpus: int = 1,
    eval_set: str = "val",
) -> list:
    """Build the evaluation command for DifFUSER."""
    cmd = [
        "torchpack", "dist-run",
        "-np", str(num_gpus),
        sys.executable, str(diffuser_root / "tools" / "test.py"),
        config_path,
        checkpoint_path,
        "--eval", "bbox",  # Evaluate 3D bounding boxes
    ]
    return cmd


def sync_to_gdrive(logdir: str, remote_path: str = "gdrive:Thesis_Checkpoints/diffuser"):
    """Sync checkpoints to Google Drive via rclone (non-blocking)."""
    try:
        subprocess.Popen(
            ["rclone", "copy", logdir, remote_path,
             "--include", "*.pth", "--include", "*.json",
             "--include", "*.yaml", "--include", "*.log"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Background sync started: {logdir} -> {remote_path}")
    except FileNotFoundError:
        logger.warning("rclone not found - skipping Google Drive sync")


class DifFUSERAdapter:
    """
    Adapter for DifFUSER baseline training and evaluation.
    
    Since DifFUSER uses mmdetection3d's training infrastructure,
    this adapter:
    1. Prepares data paths and symlinks
    2. Generates config overrides for cost optimization
    3. Launches training via torchpack subprocess
    4. Monitors and logs progress
    5. Syncs checkpoints to Google Drive
    """

    def __init__(self, args):
        self.args = args
        self.diffuser_root = find_diffuser_root()
        logger.info(f"DifFUSER root: {self.diffuser_root}")
        
        # Resolve config
        if os.path.isabs(args.config):
            self.config_path = args.config
        else:
            self.config_path = str(self.diffuser_root / args.config)
        
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        logger.info(f"Config: {self.config_path}")
    
    def prepare(self):
        """Prepare data, symlinks, and info files."""
        logger.info("Preparing DifFUSER training environment...")
        
        # Ensure data is accessible
        prepare_nuscenes_data(self.args.data_root, self.diffuser_root)
        
        # Generate info pickle files
        create_nuscenes_infos(self.diffuser_root, self.args.data_root)
        
        # Create run directory
        os.makedirs(self.args.logdir, exist_ok=True)
        
        # Save adapter args
        with open(os.path.join(self.args.logdir, "adapter_args.json"), "w") as f:
            json.dump(vars(self.args), f, indent=2)
    
    def build_config_overrides(self) -> list:
        """Generate config overrides for cost optimization."""
        overrides = []
        
        # Override max epochs
        overrides.extend([f"max_epochs={self.args.epochs}"])
        
        # Override batch size
        overrides.extend([f"data.samples_per_gpu={self.args.batch_size}"])
        
        # Override data root
        overrides.extend([f"dataset_root={self.args.data_root}/"])
        
        # AMP flag (DifFUSER supports fp16 via mmcv runner)
        if self.args.use_amp:
            overrides.extend(["fp16.loss_scale=512.0"])
        
        return overrides

    def train(self):
        """Launch DifFUSER training."""
        self.prepare()
        
        overrides = self.build_config_overrides()
        
        cmd = build_training_command(
            diffuser_root=self.diffuser_root,
            config_path=self.config_path,
            run_dir=self.args.logdir,
            num_gpus=self.args.num_gpus,
            use_amp=self.args.use_amp,
            extra_opts=overrides,
        )
        
        logger.info("=" * 60)
        logger.info("DifFUSER Training Launch")
        logger.info(f"  Config: {self.config_path}")
        logger.info(f"  Epochs: {self.args.epochs}")
        logger.info(f"  Batch size: {self.args.batch_size}")
        logger.info(f"  AMP: {self.args.use_amp}")
        logger.info(f"  GPUs: {self.args.num_gpus}")
        logger.info(f"  Log dir: {self.args.logdir}")
        logger.info(f"  Command: {' '.join(cmd)}")
        logger.info("=" * 60)
        
        # Launch training subprocess
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.diffuser_root) + ":" + env.get("PYTHONPATH", "")
        
        process = subprocess.Popen(
            cmd,
            cwd=str(self.diffuser_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        # Stream output
        for line in process.stdout:
            print(line, end="")
            # Sync checkpoints periodically
            if "Saving checkpoint" in line and self.args.sync_gdrive:
                sync_to_gdrive(self.args.logdir)
        
        process.wait()
        
        if process.returncode != 0:
            logger.error(f"DifFUSER training exited with code {process.returncode}")
            raise RuntimeError("DifFUSER training failed")
        
        logger.info("DifFUSER training completed successfully")
        
        # Final sync
        if self.args.sync_gdrive:
            sync_to_gdrive(self.args.logdir)
    
    def evaluate(self, checkpoint_path: str = None):
        """Run DifFUSER evaluation."""
        if checkpoint_path is None:
            # Find latest checkpoint in logdir
            ckpts = sorted(Path(self.args.logdir).glob("epoch_*.pth"))
            if not ckpts:
                ckpts = sorted(Path(self.args.logdir).glob("latest.pth"))
            if not ckpts:
                raise FileNotFoundError(f"No checkpoints found in {self.args.logdir}")
            checkpoint_path = str(ckpts[-1])
        
        cmd = build_eval_command(
            diffuser_root=self.diffuser_root,
            config_path=self.config_path,
            checkpoint_path=checkpoint_path,
            num_gpus=self.args.num_gpus,
        )
        
        logger.info(f"Evaluating DifFUSER: {checkpoint_path}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.diffuser_root) + ":" + env.get("PYTHONPATH", "")
        
        result = subprocess.run(
            cmd, cwd=str(self.diffuser_root), env=env,
            capture_output=True, text=True,
        )
        
        print(result.stdout)
        if result.returncode != 0:
            logger.error(f"Evaluation failed:\n{result.stderr}")
        
        return result.stdout


def parse_args():
    parser = argparse.ArgumentParser(description="DifFUSER Training Adapter (STAF Thesis-2)")
    parser.add_argument("--data-root", type=str, required=True, help="nuScenes dataset root")
    parser.add_argument("--config", type=str,
                        default="configs/nuscenes/seg/diffuser-pillars.yaml",
                        help="DifFUSER config file (relative to DifFUSER root)")
    parser.add_argument("--logdir", type=str, default="/workspace/checkpoints/diffuser")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--sync-gdrive", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_amp:
        args.use_amp = False
    
    adapter = DifFUSERAdapter(args)
    
    if args.eval_only:
        adapter.evaluate(args.checkpoint)
    else:
        adapter.train()
