with open("scripts/modal_train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_block = """    elif dataset_name == "carla":
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
    else:
        from staf.data.dataset import NuScenesDataset
        from torch.utils.data import DataLoader

        train_dataset = NuScenesDataset("/nuscenes", split="train", num_classes=num_classes)
        val_dataset = NuScenesDataset("/nuscenes", split="val", num_classes=num_classes)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True,
        )"""

new_block = """    elif dataset_name == "carla":
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
        )"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open("scripts/modal_train.py", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print("Successfully patched modal_train.py with joint multi-domain support and nuScenes pred_len")
else:
    print("ERROR: old_block not found in scripts/modal_train.py")

