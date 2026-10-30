"""
Modal Serverless Training Launcher for STAF Thesis-2.

Deploys STAF and baseline training jobs on Modal's serverless GPU
infrastructure. Per-second billing = pay only for actual GPU time.

Usage:
    # STAF full training
    modal run scripts/modal_train.py --job staf --epochs 18

    # TransFuser re-training
    modal run scripts/modal_train.py --job transfuser --epochs 20

    # DifFUSER re-training
    modal run scripts/modal_train.py --job diffuser --epochs 12

    # Quick smoke test (2 epochs, synthetic data)
    modal run scripts/modal_train.py --job staf --epochs 2 --smoke-test

Prerequisites:
    pip install modal
    modal setup
    # Upload datasets to Modal Volume (see scripts/modal_upload_data.py)
"""

import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import modal
import os

# ============================================================
# Modal Configuration
# ============================================================

app = modal.App("staf-thesis-training")

# Persistent volumes for datasets and checkpoints
data_volume = modal.Volume.from_name("staf-datasets", create_if_missing=True)
nuscenes_volume = modal.Volume.from_name("staf-nuscenes", create_if_missing=True)
checkpoint_volume = modal.Volume.from_name("staf-checkpoints", create_if_missing=True)

# Locate STAF repo root and Baseline repos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.exists("/root/STAF"):
    staf_local_path = "/root/STAF"
    tf_code_path = "/root/Baseline/TransFuser/team_code_transfuser" if os.path.exists("/root/Baseline/TransFuser/team_code_transfuser") else None
else:
    staf_candidates = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../STAF")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../staf")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
    ]
    staf_local_path = None
    for c in staf_candidates:
        if os.path.exists(os.path.join(c, "staf", "models")):
            staf_local_path = c
            break
    assert staf_local_path is not None, f"Could not locate STAF root from {SCRIPT_DIR}"

    tf_code_candidates = [
        os.path.abspath(os.path.join(SCRIPT_DIR, "../team_code_transfuser")),
        os.path.abspath(os.path.join(SCRIPT_DIR, "../../Baseline/transfuser/team_code_transfuser")),
    ]
    tf_code_path = None
    for p in tf_code_candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train.py")):
            tf_code_path = p
            break

# GPU-ready container image with all dependencies and lightweight local repos mounted (Modal v1.0+ API)
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "wget", "unzip", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "torch==2.4.0",
        "torchvision==0.19.0",
        "torchaudio==2.4.0",
        "numpy",
        "scipy",
        "pillow",
        "opencv-python-headless",
        "tensorboard",
        "timm",
        "einops",
        "nuscenes-devkit",
        "ujson",
        "scikit-image",
        "tqdm",
    )
)

if not os.path.exists("/root/STAF") and staf_local_path:
    training_image = training_image.add_local_dir(
        staf_local_path,
        remote_path="/root/STAF",
        ignore=["**/__pycache__", "**/.git", "**/checkpoints", "**/*.pth", "**/*.pt", "**/runs", "**/weights"],
    )

if not os.path.exists("/root/Baseline/TransFuser/team_code_transfuser") and tf_code_path:
    training_image = training_image.add_local_dir(
        tf_code_path,
        remote_path="/root/Baseline/TransFuser/team_code_transfuser",
        ignore=["**/__pycache__", "**/*.pth", "**/*.pt"],
    )


# ============================================================
# Training Functions
# ============================================================

@app.function(
    image=training_image,
    gpu="A10G",  # 24GB VRAM, $1.10/hr — best value for our 33M param model
    timeout=86400,  # 24 hours max
    volumes={
        "/data": data_volume,
        "/nuscenes": nuscenes_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=32768,  # 32GB RAM
)
def train_staf(
    dataset_name: str = "carla",
    epochs: int = 18,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_classes: int = 0,
    pred_len: int = 4,
    queue_length: int = 5,
    gamma: float = 2.0,
    resume: bool = True,
    smoke_test: bool = False,
):
    """Train the STAF model on Modal GPU."""
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("modal-staf")

    device = torch.device("cuda")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from staf.models.staf_model import STAFModel
    from staf.losses.focal_loss import ClassBalancedFocalLoss
    try:
        from torch.amp import GradScaler, autocast
        scaler = GradScaler("cuda")
        amp_ctx = autocast("cuda")
    except Exception:
        from torch.cuda.amp import GradScaler, autocast
        scaler = GradScaler()
        amp_ctx = autocast()

    # Build model
    model = STAFModel(
        embed_dim=256,
        num_heads=4,
        num_fusion_layers=4,
        queue_length=queue_length,
        num_classes=num_classes,
        pred_len=pred_len,
        use_target_point=True,
        pretrained_backbones=not smoke_test,
    ).to(device)

    param_summary = model.get_param_summary()
    total_params = param_summary["total"]
    logger.info(f"Total parameters: {total_params:,}")

    # Build optimizer with differential LR
    backbone_params = []
    new_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "visual_backbone" in name or "lidar_backbone" in name:
            backbone_params.append(param)
        else:
            new_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": new_params, "lr": lr},
    ], weight_decay=1e-4)

    criterion = ClassBalancedFocalLoss(num_classes=num_classes, gamma=gamma).to(device)

    # Auto-resume from checkpoint
    start_epoch = 0
    best_val_loss = float("inf")
    ckpt_dir = "/checkpoints/staf"
    os.makedirs(ckpt_dir, exist_ok=True)

    if resume:
        import glob
        import re

        def extract_epoch(path):
            m = re.search(r"checkpoint_epoch_(\d+)\.pth", path)
            return int(m.group(1)) if m else -1

        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch_*.pth")), key=extract_epoch)
        if ckpts:
            latest = ckpts[-1]
            try:
                ckpt = torch.load(latest, map_location=device, weights_only=False)
                incompatible = model.load_state_dict(ckpt["model_state_dict"], strict=False)
                if incompatible.missing_keys:
                    logger.info(f"Loaded checkpoint with missing keys: {incompatible.missing_keys}")
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        logger.warning("Optimizer state not matched, continuing with fresh optimizer")
                start_epoch = ckpt.get("epoch", 0) + 1
                best_val_loss = ckpt.get("metrics", {}).get("val_loss", float("inf"))
                logger.info(f"Resumed from {latest} (epoch {start_epoch})")
            except Exception as e:
                logger.warning(f"Could not load {latest}: {e}. Starting fresh.")

    # Build dataset
    if smoke_test:
        logger.info("SMOKE TEST MODE: Using synthetic data")
        from torch.utils.data import TensorDataset, DataLoader
        import numpy as np

        np.random.seed(42)
        n = 64
        train_labels = torch.randint(0, num_classes, (n,)) if num_classes > 0 else torch.zeros(n, dtype=torch.long)
        val_labels = torch.randint(0, num_classes, (n // 2,)) if num_classes > 0 else torch.zeros(n // 2, dtype=torch.long)
        train_dataset = TensorDataset(
            torch.rand(n, 3, 256, 256),
            torch.rand(n, 2, 256, 256),
            torch.randn(n, 9) * 0.1,
            train_labels,
            torch.randn(n, pred_len, 2),
            torch.randn(n, 2),
        )
        val_dataset = TensorDataset(
            torch.rand(n // 2, 3, 256, 256),
            torch.rand(n // 2, 2, 256, 256),
            torch.randn(n // 2, 9) * 0.1,
            val_labels,
            torch.randn(n // 2, pred_len, 2),
            torch.randn(n // 2, 2),
        )

        def collate_dict(batch):
            imgs, lidars, kines, labels, wps, tgts = zip(*batch)
            return {
                "image": torch.stack(imgs),
                "lidar_bev": torch.stack(lidars),
                "x_kinematic": torch.stack(kines),
                "label": torch.stack(labels),
                "waypoints": torch.stack(wps),
                "target_point": torch.stack(tgts),
            }

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_dict)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_dict)
    elif dataset_name == "carla":
        from staf.data.dataset import CARLASTAFDataset
        from torch.utils.data import DataLoader

        logger.info("Loading CARLASTAFDataset (simulation routes with domain augmentations)...")
        carla_root = "/data/carla_transfuser" if os.path.exists("/data/carla_transfuser") else "/data"
        train_dataset = CARLASTAFDataset(carla_root, split="train", pred_len=pred_len, augment_domain=True)
        val_dataset = CARLASTAFDataset(carla_root, split="val", pred_len=pred_len, augment_domain=False)
        logger.info(f"CARLA samples: {len(train_dataset)} train, {len(val_dataset)} val")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    elif dataset_name == "all":
        from staf.data.dataset import CARLASTAFDataset, NuScenesDataset
        from torch.utils.data import DataLoader, ConcatDataset

        logger.info("Loading JOINT MULTI-DOMAIN DATASET (CARLA Sim + nuScenes Real-World)...")
        carla_root = "/data/carla_transfuser" if os.path.exists("/data/carla_transfuser") else "/data"
        carla_train = CARLASTAFDataset(carla_root, split="train", pred_len=pred_len, augment_domain=True)
        carla_val = CARLASTAFDataset(carla_root, split="val", pred_len=pred_len, augment_domain=False)

        nusc_train = NuScenesDataset("/nuscenes", split="train", pred_len=pred_len, num_classes=num_classes)
        nusc_val = NuScenesDataset("/nuscenes", split="val", pred_len=pred_len, num_classes=num_classes)

        train_dataset = ConcatDataset([carla_train, nusc_train])
        val_dataset = ConcatDataset([carla_val, nusc_val])
        logger.info(f"Joint Dataset: {len(train_dataset)} train ({len(carla_train)} CARLA + {len(nusc_train)} nuScenes), {len(val_dataset)} val")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )
    else:
        from staf.data.dataset import NuScenesDataset
        from torch.utils.data import DataLoader

        logger.info("Loading NuScenesDataset (real-world urban driving with trajectory waypoints)...")
        train_dataset = NuScenesDataset("/nuscenes", split="train", pred_len=pred_len, num_classes=num_classes)
        val_dataset = NuScenesDataset("/nuscenes", split="val", pred_len=pred_len, num_classes=num_classes)
        logger.info(f"nuScenes samples: {len(train_dataset)} train, {len(val_dataset)} val")
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )

    # Scheduler: cosine with warmup
    import numpy as np
    warmup_epochs = 3

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    for g in optimizer.param_groups:
        g.setdefault("initial_lr", g["lr"])
    last_epoch = start_epoch - 1 if start_epoch > 0 else -1
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

    # Training loop
    import time

    logger.info(f"Starting training: epochs {start_epoch} -> {epochs}")
    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        model.train()
        model.reset_temporal_state(batch_size, device)

        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch in enumerate(train_loader):
            rgb = batch["image"].to(device)
            lidar_bev = batch["lidar_bev"].to(device)
            x_kinematic = batch["x_kinematic"].to(device)
            target = batch.get("target_point")
            if target is not None:
                target = target.to(device)

            optimizer.zero_grad()
            with amp_ctx:
                outputs = model(rgb, lidar_bev, x_kinematic, target_point=target)
                loss = torch.tensor(0.0, device=device)
                if "waypoints" in outputs and "waypoints" in batch:
                    gt_wps = batch["waypoints"].to(device)
                    loss = loss + torch.nn.functional.l1_loss(outputs["waypoints"], gt_wps)
                if "logits" in outputs and "label" in batch:
                    loss = loss + criterion(outputs["logits"], batch["label"].to(device))
                if loss.item() == 0.0:
                    loss = outputs["fused_features"].mean() * 0.0

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            bs = rgb.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            if batch_idx % 50 == 0:
                logger.info(
                    f"  Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"w_cam: {outputs['w_cam'].mean():.3f} "
                    f"w_lidar: {outputs['w_lidar'].mean():.3f}"
                )

        # Validation
        model.eval()
        val_loss = 0.0
        val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["image"].to(device)
                lidar_bev = batch["lidar_bev"].to(device)
                x_kinematic = batch["x_kinematic"].to(device)
                target = batch.get("target_point")
                if target is not None:
                    target = target.to(device)
                outputs = model(rgb, lidar_bev, x_kinematic, target_point=target)
                loss = torch.tensor(0.0, device=device)
                if "waypoints" in outputs and "waypoints" in batch:
                    loss = loss + torch.nn.functional.l1_loss(outputs["waypoints"], batch["waypoints"].to(device))
                if "logits" in outputs and "label" in batch:
                    loss = loss + criterion(outputs["logits"], batch["label"].to(device))
                val_loss += loss.item() * rgb.size(0)
                val_samples += rgb.size(0)

        scheduler.step()
        train_avg_loss = total_loss / max(total_samples, 1)
        val_avg_loss = val_loss / max(val_samples, 1)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) | "
            f"Train: {train_avg_loss:.4f} | Val: {val_avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Save checkpoint
        is_best = val_avg_loss < best_val_loss
        if is_best:
            best_val_loss = val_avg_loss

        ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": {"train_loss": train_avg_loss, "val_loss": val_avg_loss},
        }, ckpt_path)

        if is_best:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

        # Commit volume to persist checkpoints
        checkpoint_volume.commit()
        logger.info(f"  Checkpoint saved & committed: {ckpt_path}")

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    return {"best_val_loss": best_val_loss, "total_epochs": epochs}


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=86400,
    volumes={
        "/data": data_volume,
        "/nuscenes": nuscenes_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=32768,
)
def train_transfuser(
    epochs: int = 20,
    batch_size: int = 6,
    accum_steps: int = 4,
    setting: str = "all",
    wp_only: bool = True,
):
    """Train TransFuser baseline on Modal GPU."""
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("modal-transfuser")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # The adapter will find TransFuser from TRANSFUSER_ROOT
    os.environ["TRANSFUSER_ROOT"] = "/root/Baseline/TransFuser"

    from staf.baselines.transfuser_adapter import main as tf_main
    cmd_args = [
        "transfuser_adapter",
        "--root-dir", "/data/carla_transfuser",
        "--logdir", "/checkpoints/transfuser",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--accum-steps", str(accum_steps),
        "--setting", setting,
        "--use-amp",
    ]
    if wp_only:
        cmd_args.append("--wp-only")
    sys.argv = cmd_args
    tf_main()

    checkpoint_volume.commit()
    logger.info("TransFuser training complete. Checkpoints committed.")


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=86400,
    volumes={
        "/data": data_volume,
        "/nuscenes": nuscenes_volume,
        "/checkpoints": checkpoint_volume,
    },
    memory=32768,
)
def train_diffuser(epochs: int = 12, batch_size: int = 4):
    """Train DifFUSER baseline on Modal GPU."""
    import sys
    sys.path.insert(0, "/root/STAF")

    import torch
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("modal-diffuser")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.environ["DIFFUSER_ROOT"] = "/root/Baseline/DifFUSER"

    from staf.baselines.diffuser_adapter import main as df_main
    sys.argv = [
        "diffuser_adapter",
        "--data-root", "/nuscenes",
        "--logdir", "/checkpoints/diffuser",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--use-amp",
    ]
    df_main()

    checkpoint_volume.commit()
    logger.info("DifFUSER training complete. Checkpoints committed.")


# ============================================================
# CLI Entry Point
# ============================================================

@app.local_entrypoint()
def main(
    job: str = "staf",
    dataset: str = "carla",
    epochs: int = 18,
    batch_size: int = 8,
    lr: float = 3e-4,
    setting: str = "all",
    wp_only: bool = True,
    smoke_test: bool = False,
):
    """
    Entry point for `modal run scripts/modal_train.py`.

    Args:
        job: Which model to train (staf, transfuser, diffuser)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        setting: Setting for TransFuser ('all' or '02_05_withheld')
        wp_only: Train waypoint navigation only (recommended for streamlined CARLA)
        smoke_test: If True, use synthetic data for quick validation
    """
    print(f"Launching {job} training on Modal (epochs={epochs}, bs={batch_size})")

    if job == "staf":
        result = train_staf.remote(
            dataset_name=dataset,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            smoke_test=smoke_test,
        )
    elif job == "transfuser":
        result = train_transfuser.remote(
            epochs=epochs,
            batch_size=batch_size,
            setting=setting,
            wp_only=wp_only,
        )
    elif job == "diffuser":
        result = train_diffuser.remote(epochs=epochs, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown job: {job}. Choose from: staf, transfuser, diffuser")

    print(f"Training result: {result}")
