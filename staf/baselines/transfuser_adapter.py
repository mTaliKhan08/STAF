"""
TransFuser Baseline Training Adapter for STAF Thesis-2.

Wraps the original TransFuser training pipeline with cost optimizations
for re-training from scratch on RunPod (1x RTX 4090, 24GB VRAM).

Optimizations Applied:
    1. AMP (FP16 mixed precision) - 2x+ throughput gain
    2. OneCycleLR scheduler - converge in 20 epochs (vs 41 in original)
    3. torch.compile() - 15-20% kernel fusion speedup
    4. Gradient accumulation - effective BS=24 on single GPU
    5. Auto-resume from latest checkpoint
    6. Checkpoint sync to Google Drive via rclone

Usage (on RunPod):
    python -m staf.baselines.transfuser_adapter \
        --root-dir /workspace/data/carla_transfuser \
        --logdir /workspace/checkpoints/transfuser \
        --epochs 20 --batch-size 6 --accum-steps 4

Original TransFuser:
    - 41 epochs, LR=1e-4, step decay at epochs 30 and 40
    - Backbone: regnety_032 (image + LiDAR)
    - Loss: multi-task (wp, bev, depth, semantic, centernet)
"""

import argparse
import json
import os
import sys
import glob
import time
import subprocess
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    class SummaryWriter:
        def __init__(self, *args, **kwargs): pass
        def add_scalar(self, *args, **kwargs): pass
        def close(self): pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def find_transfuser_root() -> Path:
    """Locate the TransFuser baseline directory."""
    candidates = [
        Path(__file__).resolve().parents[3] / "Baseline" / "TransFuser",
        Path("/workspace/Baseline/TransFuser"),
        Path(os.environ.get("TRANSFUSER_ROOT", "")),
    ]
    for p in candidates:
        if p.exists() and (p / "team_code_transfuser" / "train.py").exists():
            return p
    raise FileNotFoundError(
        "Cannot find TransFuser repo. Set TRANSFUSER_ROOT environment variable."
    )


def setup_transfuser_imports(transfuser_root: Path):
    """Add TransFuser source to Python path."""
    code_dir = str(transfuser_root / "team_code_transfuser")
    if code_dir not in sys.path:
        sys.path.insert(0, code_dir)


def find_latest_checkpoint(logdir: str) -> tuple:
    """Find the latest checkpoint for auto-resume.

    Returns:
        (model_path, optimizer_path, epoch) or (None, None, 0)
    """
    if not os.path.isdir(logdir):
        return None, None, 0

    model_files = sorted(glob.glob(os.path.join(logdir, "model_*.pth")))
    if not model_files:
        return None, None, 0

    epochs = []
    for f in model_files:
        try:
            epoch = int(os.path.basename(f).replace("model_", "").replace(".pth", ""))
            epochs.append((epoch, f))
        except ValueError:
            continue

    if not epochs:
        return None, None, 0

    latest_epoch, latest_model = max(epochs, key=lambda x: x[0])
    latest_optimizer = latest_model.replace("model_", "optimizer_")

    if not os.path.exists(latest_optimizer):
        logger.warning(f"Optimizer checkpoint not found for epoch {latest_epoch}")
        return None, None, 0

    return latest_model, latest_optimizer, latest_epoch


def sync_to_gdrive(logdir: str, remote_path: str = "gdrive:Thesis_Checkpoints/transfuser"):
    """Sync checkpoints to Google Drive via rclone (non-blocking)."""
    try:
        subprocess.Popen(
            ["rclone", "copy", logdir, remote_path, "--include", "*.pth", "--include", "*.json"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(f"Background sync started: {logdir} -> {remote_path}")
    except FileNotFoundError:
        logger.warning("rclone not found - skipping Google Drive sync")


class OptimizedTransFuserTrainer:
    """
    Wraps TransFuser's Engine class with cost optimizations.

    Key differences from original:
    - AMP (FP16) via GradScaler
    - OneCycleLR instead of step decay
    - Gradient accumulation for larger effective batch size
    - Auto-resume from latest checkpoint
    - Periodic checkpoint sync to Google Drive
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Find and import TransFuser
        self.transfuser_root = find_transfuser_root()
        setup_transfuser_imports(self.transfuser_root)

        from config import GlobalConfig
        from model import LidarCenterNet
        from data import CARLA_Data

        # Build config
        self.config = GlobalConfig(root_dir=args.root_dir, setting=args.setting)
        self.config.use_target_point_image = bool(args.use_target_point_image)
        self.config.n_layer = args.n_layer
        self.config.use_point_pillars = False
        self.config.backbone = args.backbone
        if args.wp_only:
            self.config.multitask = False

        # Build model
        logger.info(f"Building TransFuser model (backbone={args.backbone}, "
                     f"image_arch={args.image_architecture}, lidar_arch={args.lidar_architecture})")
        self.model = LidarCenterNet(
            self.config, self.device, args.backbone,
            args.image_architecture, args.lidar_architecture,
            bool(args.use_velocity),
        ).to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total trainable parameters: {total_params:,}")

        # Optional: torch.compile for PyTorch 2.x speedup
        if args.use_compile and hasattr(torch, "compile"):
            logger.info("Applying torch.compile() - first iteration slow due to JIT")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Build optimizer (AdamW, same as original)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, weight_decay=1e-4)

        # AMP scaler
        self.scaler = GradScaler(enabled=args.use_amp)
        self.use_amp = args.use_amp

        # Build datasets
        logger.info(f"Loading CARLA dataset from: {args.root_dir}")
        self.train_set = CARLA_Data(root=self.config.train_data, config=self.config, shared_dict=None)
        self.val_set = CARLA_Data(root=self.config.val_data, config=self.config, shared_dict=None)
        logger.info(f"Train samples: {len(self.train_set)}, Val samples: {len(self.val_set)}")

        self.train_loader = DataLoader(
            self.train_set, shuffle=True, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_set, shuffle=False, batch_size=args.batch_size,
            num_workers=args.num_workers, pin_memory=True,
        )

        # OneCycleLR - converge faster than step decay
        steps_per_epoch = max(len(self.train_loader) // args.accum_steps, 1)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=args.lr * 10,
            epochs=args.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=10,
            final_div_factor=100,
        )

        # Loss weights (from TransFuser config)
        self.detailed_losses = self.config.detailed_losses
        if args.wp_only:
            weights = [1.0] + [0.0] * (len(self.detailed_losses) - 1)
        else:
            weights = self.config.detailed_losses_weights
        self.detailed_weights = {
            key: weights[idx] for idx, key in enumerate(self.detailed_losses)
        }

        # Logging
        os.makedirs(args.logdir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=args.logdir)
        with open(os.path.join(args.logdir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        # Auto-resume
        self.start_epoch = 0
        model_ckpt, opt_ckpt, resume_epoch = find_latest_checkpoint(args.logdir)
        if model_ckpt:
            logger.info(f"Auto-resuming from epoch {resume_epoch}: {model_ckpt}")
            self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
            self.optimizer.load_state_dict(torch.load(opt_ckpt, map_location=self.device))
            self.start_epoch = resume_epoch + 1

        self.best_val_loss = float("inf")
        self.training_log = []

    def _load_batch(self, data):
        """Move a data batch to GPU, handling optional fields."""
        batch = {
            "rgb": data["rgb"].to(self.device, dtype=torch.float32),
            "lidar": data["lidar"].to(self.device, dtype=torch.float32),
            "bev": data["bev"].to(self.device, dtype=torch.long),
            "label": data["label"].to(self.device, dtype=torch.float32),
            "ego_waypoint": data["ego_waypoint"].to(self.device, dtype=torch.float32),
            "target_point": data["target_point"].to(self.device, dtype=torch.float32),
            "target_point_image": data["target_point_image"].to(self.device, dtype=torch.float32),
            "ego_vel": data["speed"].to(self.device, dtype=torch.float32).reshape(-1, 1),
        }
        depth = data.get("depth")
        semantic = data.get("semantic")
        batch["depth"] = depth.to(self.device, dtype=torch.float32) if depth is not None else None
        batch["semantic"] = semantic.squeeze(1).to(self.device, dtype=torch.long) if semantic is not None else None
        return batch

    def _forward(self, batch):
        """Run TransFuser forward pass."""
        return self.model(
            batch["rgb"], batch["lidar"],
            ego_waypoint=batch["ego_waypoint"],
            target_point=batch["target_point"],
            target_point_image=batch["target_point_image"],
            ego_vel=batch["ego_vel"],
            bev=batch["bev"], label=batch["label"],
            save_path=None,
            depth=batch["depth"], semantic=batch["semantic"],
            num_points=None,
        )

    def train_one_epoch(self, epoch: int) -> dict:
        """Train for one epoch with AMP and gradient accumulation."""
        self.model.train()
        total_loss = 0.0
        detailed_losses_epoch = {key: 0.0 for key in self.detailed_losses}
        num_batches = 0
        accum_steps = self.args.accum_steps

        self.optimizer.zero_grad(set_to_none=True)

        for batch_idx, data in enumerate(self.train_loader):
            batch = self._load_batch(data)

            with autocast(enabled=self.use_amp):
                losses = self._forward(batch)
                loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
                for key, value in losses.items():
                    loss += self.detailed_weights[key] * value
                    detailed_losses_epoch[key] += float(self.detailed_weights[key] * value.item())
                loss = loss / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * accum_steps
            num_batches += 1

            if batch_idx % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"  Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {loss.item() * accum_steps:.4f} LR: {lr:.2e}"
                )

        avg_loss = total_loss / max(num_batches, 1)
        for key in detailed_losses_epoch:
            detailed_losses_epoch[key] /= max(num_batches, 1)

        return {"train_loss": avg_loss, **{f"train_{k}": v for k, v in detailed_losses_epoch.items()}}

    @torch.inference_mode()
    def validate(self, epoch: int) -> dict:
        """Validate on held-out data."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for data in self.val_loader:
            batch = self._load_batch(data)
            losses = self._forward(batch)
            loss = sum(self.detailed_weights[k] * v for k, v in losses.items())
            total_loss += float(loss.item())
            num_batches += 1

        return {"val_loss": total_loss / max(num_batches, 1)}

    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model and optimizer checkpoints."""
        model_path = os.path.join(self.args.logdir, f"model_{epoch}.pth")
        opt_path = os.path.join(self.args.logdir, f"optimizer_{epoch}.pth")

        state_dict = self.model.state_dict()
        if hasattr(self.model, "_orig_mod"):
            state_dict = self.model._orig_mod.state_dict()

        torch.save(state_dict, model_path)
        torch.save(self.optimizer.state_dict(), opt_path)

        metrics["epoch"] = epoch
        self.training_log.append(metrics)
        with open(os.path.join(self.args.logdir, "training_log.json"), "w") as f:
            json.dump(self.training_log, f, indent=2)

        logger.info(f"Saved checkpoint epoch {epoch}: {model_path}")

        if self.args.sync_gdrive:
            sync_to_gdrive(self.args.logdir)

    def run(self):
        """Run the full training loop."""
        logger.info("=" * 60)
        logger.info("TransFuser Optimized Re-Training")
        logger.info(f"  Epochs: {self.start_epoch} -> {self.args.epochs}")
        logger.info(f"  Batch size: {self.args.batch_size} x {self.args.accum_steps} accum = "
                     f"{self.args.batch_size * self.args.accum_steps} effective")
        logger.info(f"  AMP: {self.use_amp}")
        logger.info(f"  torch.compile: {self.args.use_compile}")
        logger.info(f"  Device: {self.device}")
        logger.info("=" * 60)

        for epoch in range(self.start_epoch, self.args.epochs):
            t0 = time.time()
            train_metrics = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)
            elapsed = time.time() - t0
            all_metrics = {**train_metrics, **val_metrics, "epoch_time_s": elapsed}

            for k, v in all_metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, epoch)

            logger.info(
                f"Epoch {epoch}/{self.args.epochs} ({elapsed:.0f}s) | "
                f"Train: {train_metrics['train_loss']:.4f} | "
                f"Val: {val_metrics['val_loss']:.4f}"
            )

            self.save_checkpoint(epoch, all_metrics)

            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                best_path = os.path.join(self.args.logdir, "best_model.pth")
                state_dict = self.model.state_dict()
                if hasattr(self.model, "_orig_mod"):
                    state_dict = self.model._orig_mod.state_dict()
                torch.save(state_dict, best_path)
                logger.info(f"  * New best model (val_loss={val_metrics['val_loss']:.4f})")

        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="TransFuser Optimized Re-Training (STAF Thesis-2)")
    parser.add_argument("--root-dir", type=str, required=True, help="CARLA dataset root")
    parser.add_argument("--setting", type=str, default="02_05_withheld",
                        choices=["all", "02_05_withheld"])
    parser.add_argument("--backbone", type=str, default="transFuser")
    parser.add_argument("--image-architecture", type=str, default="regnety_032")
    parser.add_argument("--lidar-architecture", type=str, default="regnety_032")
    parser.add_argument("--use-velocity", type=int, default=0)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--use-target-point-image", type=int, default=1)
    parser.add_argument("--wp-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--use-amp", action="store_true", default=True)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument("--logdir", type=str, default="/workspace/checkpoints/transfuser")
    parser.add_argument("--sync-gdrive", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.no_amp:
        args.use_amp = False
    trainer = OptimizedTransFuserTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
