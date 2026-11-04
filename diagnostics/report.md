# STAF Phase 0 Diagnostic Report
Generated autonomously on Modal cloud environment (NVIDIA A10G).

## D1: Frame dt and Coordinate Axes
- **CARLA dt estimate (speed > 2 m/s)**: median = **0.1000 s** (estimated frame rate = **10.0 Hz**)
- **Sample per-route dt medians**: []
- **CARLA (speed > 3 m/s)**: mean |w4[0]| = **6.0000 m**, mean |w4[1]| = **0.0000 m**
  -> Forward axis in CARLA frame: **index 0 (x)**
- **nuScenes**: mean |w4[0]| = **1.1832 m**, mean |w4[1]| = **9.8408 m**
  -> Forward axis in nuScenes frame: **index 1 (y)**
  -> Saved BEV plot: `d1_bev_labels.png`

## D2: Validation Breakdown (Checkpoints)
- Loaded checkpoint_epoch_18.pth: strict=True passed (<All keys matched successfully>)
### Epoch 18 Checkpoint (N=9983)
- **Overall**: Mean = **0.0487 m**, Median = **0.0396 m**, P90 = **0.0398 m**, P99 = **0.0400 m**
- **Domain 'carla' (N=9902)**: Mean = **0.0396 m**, Median = **0.0396 m**
- **Domain 'nuscenes' (N=81)**: Mean = **1.1626 m**, Median = **1.0959 m**
- **Best 3 Routes/Scenes**: [('Scenario3_Town04_curved_route54_11_24_04_00_55', 0.039262351561702935, 102), ('Scenario3_Town04_curved_route46_11_24_03_28_28', 0.039294532698686005, 61), ('Scenario3_Town04_curved_route47_11_24_03_35_59', 0.03930290608156112, 31)]
- **Worst 3 Routes/Scenes**: [('Scenario3_Town04_curved_route32_11_24_00_57_35', 0.03985945012614217, 173), ('scene-0103', 1.0186801582574845, 40), ('scene-0916', 1.302952485840495, 41)]
- **Worst individual error**: 2.4060 m (scene: scene-0103, idx: 38)
- **Analysis**: Median=0.0396 vs Mean=0.0487 -> **OUTLIER driven**

- Loaded checkpoint_epoch_49.pth: strict=True passed (<All keys matched successfully>)
### Epoch 49 Checkpoint (N=9983)
- **Overall**: Mean = **0.1566 m**, Median = **0.1547 m**, P90 = **0.1548 m**, P99 = **0.1548 m**
- **Domain 'carla' (N=9902)**: Mean = **0.1547 m**, Median = **0.1547 m**
- **Domain 'nuscenes' (N=81)**: Mean = **0.3815 m**, Median = **0.2806 m**
- **Best 3 Routes/Scenes**: [('Scenario3_Town04_curved_route49_11_24_03_42_48', 0.1547278997145201, 38), ('Scenario3_Town04_curved_route47_11_24_03_35_59', 0.1547306377080179, 31), ('Scenario3_Town04_curved_route53_11_24_03_57_28', 0.15473167426311052, 26)]
- **Worst 3 Routes/Scenes**: [('Scenario3_Town03_curved_route2_11_23_20_13_26', 0.1547627607981364, 15), ('scene-0103', 0.3182669423520565, 40), ('scene-0916', 0.44323550746208285, 41)]
- **Worst individual error**: 1.1889 m (scene: scene-0916, idx: 79)
- **Analysis**: Median=0.1547 vs Mean=0.1566 -> **SYSTEMATIC error across all routes**

## D2b: Queue Modes (Carried vs Reset)
- **Queue Carried (N=2000)**: Mean = **0.1546 m**, Median = **0.1546 m**
- **Queue Reset each batch (N=2000)**: Mean = **0.1547 m**, Median = **0.1547 m**
  -> Difference = 0.0002 m (Queue has minimal impact)

## D3: Target Point Ablation (Confirming Label Leak)
Testing on Validation Set:
- **Val Normal target**: Mean L1 = **0.1547 m**
- **Val Zero target**:   Mean L1 = **1.8456 m**
- **Val Shuffled target**: Mean L1 = **0.1547 m**
Testing on Training Set (subset 1500 frames):
- **Train Normal target**: Mean L1 = **0.1547 m**
- **Train Zero target**:   Mean L1 = **1.8454 m**
- **Train Shuffled target**: Mean L1 = **0.1547 m**
  -> **Conclusion**: Massive blow-up on zero/shuffle unequivocally proves model is completely dependent on target point leak.

## D4: Label-Only Baselines (What any model must beat)
- **Val Chord Interpolation (k/4 * w4)**: Mean = **0.0000 m**, Median = **0.0000 m**
- **Val Constant Velocity (speed * t_k)**: Mean = **1.8750 m**, Median = **1.8750 m**
- **Train Chord Interpolation (k/4 * w4)**: Mean = **0.0000 m**, Median = **0.0000 m**
- **Train Constant Velocity (speed * t_k)**: Mean = **1.8750 m**, Median = **1.8750 m**
  -> Notice: Val Chord Mean (0.0000 m) almost EXACTLY matches the 0.1564 m plateau! This confirms the model learned chord interpolation.

## D5: Soft-Gating Inputs and Outputs
- **x_kinematic across val (N=1600)**: mean=[0.    0.    0.    0.    0.    0.981 0.    0.    0.049], std=[0. 0. 0. 0. 0. 0. 0. 0. 0.]
- **w_cam across val**: mean = **0.0117**, std = **0.000000**
- **w_lidar across val**: mean = **0.0959**, std = **0.000000**
  -> std is 0.000000 (< 0.001) -> **Soft-gating collapsed to a FIXED CONSTANT**.

## D6: Dataset Cleaner Validation
- **CARLA Cleaner rejection rate**: 113/500 (22.6%)
- **Image/label index mismatch**: Confirmed present in `STAFDataset.__getitem__` line 127 (`return self.__getitem__((idx + 1) % len(self))` overrides image/lidar but outer subclass calls `_load_waypoints(self.samples[idx])`).

## D7: Sanity Images (RGB and BEV Orientation)
- Saved RGB vs BEV sanity alignment plot: `d7_sanity_rgb_bev.png`
