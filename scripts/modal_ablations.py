"""
Modal Ablation Study Runner for STAF Thesis-2.

Runs the 4 ablation experiments (Phase 4.5) on Modal serverless GPUs.
Each ablation uses L4 ($0.80/hr) for cost efficiency.

Usage:
    # Run all 4 ablations
    modal run scripts/modal_ablations.py --ablation all

    # Run specific ablation
    modal run scripts/modal_ablations.py --ablation no-gating
    modal run scripts/modal_ablations.py --ablation no-temporal
    modal run scripts/modal_ablations.py --ablation ce-loss
    modal run scripts/modal_ablations.py --ablation noise-inject

Ablation Studies:
    1. no-gating:     STAF with w_cam = w_lidar = 1.0 (bypass soft-gating)
    2. no-temporal:   STAF with queue_length=0 (no temporal memory)
    3. ce-loss:       STAF with gamma=0.0 (cross-entropy instead of focal loss)
    4. noise-inject:  STAF with injected sensor noise (camera blur + LiDAR dropout)
"""

import modal
import os

app = modal.App("staf-ablation-studies")

data_volume = modal.Volume.from_name("staf-datasets", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("staf-checkpoints", create_if_missing=True)

# Locate STAF repo root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(SCRIPT_DIR, "../staf")):
    staf_local_path = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
elif os.path.exists(os.path.join(SCRIPT_DIR, "../../STAF/staf")):
    staf_local_path = os.path.abspath(os.path.join(SCRIPT_DIR, "../../STAF"))
else:
    staf_local_path = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy",
        "scipy",
        "pillow",
        "opencv-python-headless",
        "tensorboard",
        "timm",
        "einops",
        "nuscenes-devkit",
    )
    .add_local_dir(
        staf_local_path,
        remote_path="/root/STAF",
        ignore=["**/__pycache__", "**/.git", "**/checkpoints", "**/*.pth", "**/*.pt"],
    )
)


def _run_ablation(
    name: str,
    epochs: int,
    batch_size: int,
    queue_length: int = 5,
    gamma: float = 2.0,
    disable_gating: bool = False,
    inject_noise: bool = False,
):
    """Core ablation training logic shared by all experiments."""
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import time
    import glob
    import logging
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger(f"ablation-{name}")

    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"Ablation: {name}")

    from staf.models.staf_model import STAFModel
    from staf.losses.focal_loss import ClassBalancedFocalLoss
    from torch.cuda.amp import GradScaler, autocast

    # Build model
    model = STAFModel(
        embed_dim=256,
        num_heads=4,
        num_fusion_layers=4,
        queue_length=queue_length,
        num_classes=16,
        pretrained_backbones=True,
    ).to(device)

    logger.info(f"Model params: {model.get_param_summary()['total']:,}")

    # For the no-gating ablation, override the soft-gating to always return 1.0
    if disable_gating:
        logger.info("ABLATION: Disabling soft-gating (w_cam = w_lidar = 1.0)")

        class FixedGating(torch.nn.Module):
            def forward(self, x):
                B = x.shape[0]
                ones = torch.ones(B, 1, device=x.device)
                return ones, ones

        model.soft_gating = FixedGating()

    # Build optimizer, criterion, scaler
    lr = 3e-4
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and ("visual_backbone" in n or "lidar_backbone" in n)]
    new_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and "visual_backbone" not in n and "lidar_backbone" not in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": new_params, "lr": lr},
    ], weight_decay=1e-4)

    criterion = ClassBalancedFocalLoss(num_classes=16, gamma=gamma).to(device)
    scaler = GradScaler()

    # Auto-resume
    ckpt_dir = f"/checkpoints/ablation_{name}"
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    best_val_loss = float("inf")

    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")))
    if ckpts:
        latest = ckpts[-1]
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch}")

    # Build dataset (synthetic for now — replace with real data path)
    # When running on real data, change this to NuScenesDataset
    from torch.utils.data import TensorDataset, DataLoader

    # Check if real data exists
    nuscenes_path = "/data/nuscenes"
    if os.path.exists(os.path.join(nuscenes_path, "v1.0-trainval")):
        logger.info("Using real nuScenes data")
        from staf.data.dataset import NuScenesDataset
        train_dataset = NuScenesDataset(nuscenes_path, split="train", num_classes=16)
        val_dataset = NuScenesDataset(nuscenes_path, split="val", num_classes=16)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    else:
        logger.info("Using synthetic data (nuScenes not found in volume)")
        n_train, n_val = 128, 32
        train_dataset = TensorDataset(
            torch.randn(n_train, 3, 224, 224),
            torch.randn(n_train, 2, 200, 200),
            torch.randn(n_train, 9),
            torch.randint(0, 16, (n_train,)),
        )
        val_dataset = TensorDataset(
            torch.randn(n_val, 3, 224, 224),
            torch.randn(n_val, 2, 200, 200),
            torch.randn(n_val, 9),
            torch.randint(0, 16, (n_val,)),
        )

        def collate_dict(batch):
            imgs, lidars, kines, labels = zip(*batch)
            return {
                "image": torch.stack(imgs),
                "lidar_bev": torch.stack(lidars),
                "x_kinematic": torch.stack(kines),
                "label": torch.stack(labels),
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dict)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_dict)

    # Noise injection function for ablation 4
    def apply_noise(batch):
        """Add camera blur and LiDAR point dropout for noise ablation."""
        if not inject_noise:
            return batch
        import torch.nn.functional as F

        # Camera: Gaussian blur (sigma=2.0)
        img = batch["image"]
        kernel_size = 5
        sigma = 2.0
        channels = img.shape[1]
        x = torch.arange(kernel_size, dtype=torch.float32, device=img.device) - kernel_size // 2
        gauss = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = gauss / gauss.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d.expand(channels, 1, kernel_size, kernel_size)
        padding = kernel_size // 2
        batch["image"] = F.conv2d(img, kernel_2d, padding=padding, groups=channels)

        # LiDAR: Random point dropout (30% of channels zeroed)
        lidar = batch["lidar_bev"]
        mask = torch.rand_like(lidar) > 0.3
        batch["lidar_bev"] = lidar * mask.float()

        return batch

    # Scheduler
    warmup_epochs = 2
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Training loop
    results = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        model.reset_temporal_state(batch_size, device)

        total_loss, total_samples = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            batch = apply_noise(batch)  # No-op unless inject_noise=True

            rgb = batch["image"].to(device)
            lidar_bev = batch["lidar_bev"].to(device)
            x_kinematic = batch["x_kinematic"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with autocast(device_type="cuda"):
                outputs = model(rgb, lidar_bev, x_kinematic)
                loss = criterion(outputs["logits"], labels) if "logits" in outputs else torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * rgb.size(0)
            total_samples += rgb.size(0)

        # Validation
        model.eval()
        val_loss, val_samples = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["image"].to(device)
                lidar_bev = batch["lidar_bev"].to(device)
                x_kinematic = batch["x_kinematic"].to(device)
                labels = batch["label"].to(device)
                outputs = model(rgb, lidar_bev, x_kinematic)
                if "logits" in outputs:
                    loss = criterion(outputs["logits"], labels)
                    val_loss += loss.item() * rgb.size(0)
                val_samples += rgb.size(0)

        scheduler.step()
        train_avg = total_loss / max(total_samples, 1)
        val_avg = val_loss / max(val_samples, 1)
        elapsed = time.time() - t0

        logger.info(f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | Train: {train_avg:.4f} | Val: {val_avg:.4f}")

        is_best = val_avg < best_val_loss
        if is_best:
            best_val_loss = val_avg

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {"train_loss": train_avg, "val_loss": val_avg},
        }, os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth"))

        if is_best:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

        results.append({"epoch": epoch, "train_loss": train_avg, "val_loss": val_avg})
        checkpoint_volume.commit()

    logger.info(f"Ablation '{name}' complete. Best val loss: {best_val_loss:.4f}")
    return {"ablation": name, "best_val_loss": best_val_loss, "results": results}


# ============================================================
# Ablation Functions (one per experiment)
# ============================================================

@app.function(
    image=training_image, gpu="L4", timeout=43200,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    memory=32768,
)
def ablation_no_gating(epochs: int = 10, batch_size: int = 8):
    """Ablation 1: STAF without soft-gating (w=1.0 always)."""
    return _run_ablation("no_gating", epochs, batch_size, disable_gating=True)


@app.function(
    image=training_image, gpu="L4", timeout=43200,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    memory=32768,
)
def ablation_no_temporal(epochs: int = 10, batch_size: int = 8):
    """Ablation 2: STAF without temporal memory queue."""
    return _run_ablation("no_temporal", epochs, batch_size, queue_length=0)


@app.function(
    image=training_image, gpu="L4", timeout=43200,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    memory=32768,
)
def ablation_ce_loss(epochs: int = 10, batch_size: int = 8):
    """Ablation 3: STAF with cross-entropy loss (gamma=0)."""
    return _run_ablation("ce_loss", epochs, batch_size, gamma=0.0)


@app.function(
    image=training_image, gpu="L4", timeout=43200,
    volumes={"/data": data_volume, "/checkpoints": checkpoint_volume},
    memory=32768,
)
def ablation_noise_inject(epochs: int = 10, batch_size: int = 8):
    """Ablation 4: STAF with injected sensor noise."""
    return _run_ablation("noise_inject", epochs, batch_size, inject_noise=True)


# ============================================================
# CLI Entry Point
# ============================================================

@app.local_entrypoint()
def main(ablation: str = "all", epochs: int = 10, batch_size: int = 8):
    """
    Entry point for `modal run scripts/modal_ablations.py`.

    Args:
        ablation: Which ablation to run (all, no-gating, no-temporal, ce-loss, noise-inject)
        epochs: Number of epochs per ablation
        batch_size: Batch size
    """
    ablation_map = {
        "no-gating": ablation_no_gating,
        "no-temporal": ablation_no_temporal,
        "ce-loss": ablation_ce_loss,
        "noise-inject": ablation_noise_inject,
    }

    if ablation == "all":
        print(f"Launching ALL 4 ablation studies (epochs={epochs}, bs={batch_size})")
        # Launch all 4 in parallel for maximum efficiency
        futures = []
        for name, fn in ablation_map.items():
            print(f"  Starting ablation: {name}")
            futures.append((name, fn.spawn(epochs=epochs, batch_size=batch_size)))

        # Collect results
        results = {}
        for name, future in futures:
            result = future.get()
            results[name] = result
            print(f"  Completed: {name} -> best_val_loss={result['best_val_loss']:.4f}")

        print("\n=== All Ablation Results ===")
        for name, result in results.items():
            print(f"  {name}: best_val_loss = {result['best_val_loss']:.4f}")

    elif ablation in ablation_map:
        print(f"Launching ablation: {ablation}")
        result = ablation_map[ablation].remote(epochs=epochs, batch_size=batch_size)
        print(f"Result: {result}")

    else:
        valid = ", ".join(["all"] + list(ablation_map.keys()))
        raise ValueError(f"Unknown ablation: {ablation}. Choose from: {valid}")
