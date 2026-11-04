# STAF Project — Comprehensive Critical Analysis Report

> [!IMPORTANT]
> **Report Scope:** This report is evidence-based and critically objective. Every metric value is sourced directly from experiment files in the STAF repository. Claims are explicitly labelled as Supported, Not Supported, or Invalidated by experimental evidence. No new claims are made.
>
> **Source of Truth:** All experiment results from `E:\MS-DS\Thesis-1\Multi-Modal Fusion Framework for Autonomous Robot\STAF\results\` and associated log files. All implementation facts from `STAF\staf\` Python package source code.

---

## §1 Executive Summary

| # | Critical Finding |
|---|---|
| 1 | **Scope pivot:** The empirical work pivoted entirely from the 3-benchmark system (CARLA DS, nuScenes mIoU, EuRoC ATE) to CARLA open-loop waypoint prediction (ADE/FDE). The original benchmark targets (DS≥72, mIoU≥73.5%, ATE≤0.025m) were **never evaluated**. |
| 2 | **Best overall model is LiDAR-Only, not Full Fusion.** STAF LiDAR-Only achieves 0.1799±0.0015m overall ADE. Full Fusion (0.1899±0.0075m) is worse than LiDAR-Only and shows high seed variance (seed 42: 0.1996m vs. seed 43: 0.1814m). |
| 3 | **Vision modalities help only on turning frames** (CI camera vs. no-vision turning: [−0.080, −0.025]m) but are **not significantly better overall** (CI: [−0.005, +0.001]m). The claim that "vision improves trajectory prediction" is only conditionally supported. |
| 4 | **Soft gating is near-static,** not dynamic. Gate weights vary weakly across validation frames (w_cam std=0.025, w_lidar std=0.015). Correlation with speed: r=−0.175. The "dynamic IMU-guided gating" mechanism is **not demonstrated** to respond meaningfully to sensor quality. |
| 5 | **SO(3) IMU pre-integration is implemented but not used in training.** The CARLA dataset stub returns near-zero IMU data (`gyro = np.zeros((20,3))`). The actual gate input is a 3-d causal kinematic vector `[v/10, a, ω]`, not the 9-d pre-integrated vector. |
| 6 | **Phase 1 STAF was worse than even the Constant Curvature baseline** (0.3883 vs. 0.3088m overall ADE), due to unnormalized 9-d kinematics with zero-padding and no residual head. The remediated suite corrects this. |
| 7 | **The strongest demonstrated contribution** is the residual CC head + normalized goal + causal kinematics design: No-Vision retrained STAF achieves 0.1824±0.0052m, outperforming the best standalone kinematic MLP (Residual CC: 0.2030m) by ~10%. Vision modalities (camera, LiDAR) add a further ~1% overall and ~21% on turning frames vs. no-vision. |

---

## §2 Original Research Scope and Objectives

### 2.1 Three-Benchmark System (from `STAF_implementation_plan.md`)

The proposal specified three independent benchmarks, each comparing STAF against a different baseline:

| Benchmark | Task | Baseline | Primary Metric | Target |
|-----------|------|----------|----------------|--------|
| CARLA Longest6 | End-to-end driving (IL) | TransFuser | Driving Score (DS) | ≥ 72.0 |
| nuScenes val | 3D detection + BEV segmentation | DifFUSER | mIoU | ≥ 73.5% |
| EuRoC MAV | Visual-inertial trajectory estimation | ORB-SLAM3 | ATE RMSE | ≤ 0.025 m |
| Gazebo (closed-loop) | Safety under degradation | All three | CTE, collision/km, SFR | — |

### 2.2 Five-Phase Plan

| Phase | Objective | Status |
|-------|-----------|--------|
| **Phase 0** | Literature gap, taxonomy, limitations | Partially Done (taxonomy notes in plan; no published matrix) |
| **Phase 1** | Full dataset pipeline, cleaning, class-balancing | Partially Done (CARLA pipeline complete; nuScenes/EuRoC stubs only) |
| **Phase 2** | IMU pre-integration & kinematic encoding | Partially Done (SO(3) module implemented and unit-tested; not used in production training) |
| **Phase 3** | STAF architecture + baseline setup | Partially Done (STAF model implemented; TransFuser adapter stub; DifFUSER/ORB-SLAM3 not evaluated) |
| **Phase 4** | GPU training + 4 ablation studies | Partially Done (3-seed retrained suite done; original 4 ablations not executed) |
| **Phase 5** | ROS 2 deployment + Gazebo simulation | **Not Started** |

### 2.3 Proposed Metric Targets vs. Current Status

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| CARLA Driving Score | ≥ 72.0 | Not evaluated | **Not Demonstrated** |
| CARLA Route Completion | Report | Not evaluated | **Not Demonstrated** |
| nuScenes mIoU | ≥ 73.5% | Not evaluated | **Not Demonstrated** |
| nuScenes NDS | Report | Not evaluated | **Not Demonstrated** |
| EuRoC ATE RMSE | ≤ 0.025 m | Not evaluated | **Not Demonstrated** |
| Gazebo latency | < 20 ms | Not evaluated | **Not Demonstrated** |
| **CARLA open-loop ADE** | Not originally proposed | 0.1799±0.0015m (LiDAR-Only) | **Evaluated (Pivot)** |

---

## §3 Proposed Architecture and Techniques

### 3.1 Architecture Overview (from Implementation Plan §§3.1–3.4)

The proposed STAF architecture consists of five components:

1. **Visual Stream:** ResNet34 (ImageNet pretrained), matching TransFuser backbone for fair ablation.
2. **LiDAR Stream:** ResNet18 on 2-channel BEV-projected point clouds.
3. **IMU-Guided Soft-Gating MLP:** Two independent MLPs (input: 9-d `X_kinematic`), each producing a scalar weight `w ∈ [0,1]`. Total: ~1,666 parameters.
4. **Gated Multi-Scale Cross-Attention Transformer:** 4 layers, 4 heads, embed_dim=256.  
   Formula: `f_fused = w_cam · Attn(Q_lidar, K_cam, V_cam) + w_lidar · Attn(Q_cam, K_lidar, V_lidar)`
5. **Spatio-Temporal FIFO Memory Queue:** T=5 frames, with a 2-layer temporal transformer encoder.

### 3.2 Proposed Training Configuration

- PyTorch AMP (FP16) for throughput
- OneCycleLR / CosineAnnealing with warmup
- ImageNet pretrained backbone initialization
- Differential LR: backbone at 0.1× LR, fusion/gating at 1.0× LR
- Gradient clipping (max norm = 1.0)
- `torch.compile()` for speed

### 3.3 Originally Proposed Ablations (Phase 4.5)

| # | Ablation | What It Tests |
|---|----------|---------------|
| 1 | STAF vs. STAF w/o soft-gating (w=1.0) | Quantitative value of IMU-guided gating |
| 2 | With vs. without temporal memory queue | Value of spatio-temporal consistency |
| 3 | Class-Balanced Focal Loss vs. Cross-Entropy | Impact on minority class recall |
| 4 | Performance under injected sensor noise (blur, LiDAR dropout) | Robustness to sensor degradation |

> [!WARNING]
> None of these 4 original ablations were executed as designed. The retrained modality variants (no-vision / camera / LiDAR / full) are modality ablations, not the originally proposed architectural ablations.

---

## §4 Current Implementation Analysis

### 4.1 Implemented and Verified Components

| Component | File | Verified Status | Key Details |
|-----------|------|-----------------|-------------|
| `STAFModel` | `staf/models/staf_model.py` (338 lines) | ✅ Implemented | ResNet34+ResNet18 backbones; residual CC head; goal normalization; modality switching |
| `IMUGuidedSoftGating` | `staf/models/soft_gating.py` (93 lines) | ✅ Implemented | Two independent MLP heads; `gate_input_dim` configurable (3 or 9); Sigmoid output |
| `GatedFusionTransformer` | `staf/models/fusion_transformer.py` (222 lines) | ✅ Implemented | 4-layer bidirectional gated cross-attention; gating applied by scaling key/value tokens before projection |
| `SpatioTemporalMemoryQueue` | `staf/models/temporal_memory.py` (176 lines) | ✅ Implemented | FIFO queue T=5; 2-layer temporal transformer encoder; learnable position embeddings; partial-fill handling |
| `IMUPreintegrator` | `staf/imu/preintegration.py` (225 lines) | ✅ Implemented | Full SO(3) on-manifold integration with Jacobians; Lie group Exp/Log maps; numerical stability handling |
| `ClassBalancedFocalLoss` | `staf/losses/focal_loss.py` (115 lines) | ✅ Implemented | Effective Number of Samples weighting (Cui et al., CVPR 2019); γ=2.0, β=0.999 |
| `CARLASTAFDataset` | `staf/data/dataset.py` (1048 lines) | ✅ Implemented | Multi-town CARLA data at 2 Hz; train/val/test splits; augmentation; causal kinematics |
| Residual CC head | `staf_model.py` | ✅ Implemented | `compute_cc_trajectory_torch()`; zero-init last layer; `use_residual_cc=True` |
| Goal normalization | `dataset.py` | ✅ Implemented | `target_point_norm = clamp(target_point/20.0, −3.0, 3.0)` |
| Causal 3-d kinematics | `dataset.py` | ✅ Implemented | `x_kinematic_3d = [v/10, a, ω]` from frames t and t−1; angle wrap to [−π, π] |
| Domain augmentation | `dataset.py` | ✅ Implemented | Color jitter on camera; LiDAR dropout (p=0.3–0.5); IMU noise injection |
| Data cleaning | `staf/data/cleaning.py` | ✅ Implemented but **disabled** | Comment in code: disabled for CARLA simulation data (would falsely reject valid frames) |

### 4.2 Critical Implementation Gap: IMU in CARLA Training

> [!CAUTION]
> This is the most significant implementation discrepancy between plan and reality.

The CARLA dataset's `_load_imu()` method constructs a 20-step IMU array by **pre-filling with zeros** and appending only the final frame's accelerometer/gyroscope reading:

```python
gyro = np.zeros((20, 3))
accel = np.zeros((20, 3))
# ... appends single-frame reading at index -1
```

**Consequence:** When `IMUPreintegrator.forward()` runs on this data, it integrates 19 near-identity rotations followed by one real measurement. The resulting 9-d pre-integrated vector is essentially equivalent to a single-frame kinematic state, **not** a true pre-integrated measurement.

**What is actually used as gate input in production training (`modal_train_retrain_variants.py`, line 124):**
```python
model = STAFModel(
    gate_input_dim=3,          # ← 3-d causal vector, NOT 9-d pre-integrated
    ...
)
```

The actual gate input is `x_kinematic_3d = [v/10, a, ω]` — a causal 3-d vector computed from two consecutive frames. The SO(3) pre-integration module exists, passes all unit tests, but is **not used as the gate input in any training run**.

### 4.3 Training Configuration (Retrained Suite — Confirmed from `modal_train_retrain_variants.py`)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Head LR | 1e-3 |
| Backbone LR (epochs ≥ 2) | 1e-5 |
| Backbone state (epochs 0–1) | **Frozen** |
| Weight decay | 1e-2 |
| Batch size | 64 |
| Max epochs | 5 |
| Early stopping patience | 2 |
| Gradient clipping | max\_norm=1.0 |
| Loss function | L1 (mean absolute waypoint error) |
| Test evaluation | Once, on best val checkpoint |
| Seeds | 42, 43, 44 |

> [!NOTE]
> The Class-Balanced Focal Loss (`focal_loss.py`) is implemented and unit-tested but **was not used** in the retrained suite. The actual training loss is plain L1 (`nn.L1Loss()`). The focal loss is designed for classification tasks; waypoint regression uses L1.

### 4.4 Phase 1 Training (50-Epoch Original)

From [`logs/original_50epochs/staf_original_50epochs_progression.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/logs/original_50epochs/staf_original_50epochs_progression.csv):

- Best checkpoint: epoch 18, val_loss=0.0487
- Train loss from epoch 18 → 49: 0.1143 → 0.0034 (severe overfitting)
- Val loss from epoch 18 → 49: 0.0487 → 0.1564 (plateau/increase from epoch 20)
- Gate weights collapsed by epoch 27: mean_w_cam fell from 0.217 to 0.012; mean_w_lidar from 0.274 to 0.097
- **Root cause:** 9-d unnormalized kinematics with zero-padding caused gradient collapse in soft-gating

---

## §5 Baseline vs. STAF Results

### 5.1 Master Comparison Table — Overall Test ADE (17,447 frames, Town02 & Town05)

| Category | Model | Overall ADE (mean) [m] | Overall ADE (median) [m] | Turning ADE (mean) [m] | Straight ADE (mean) [m] |
|----------|-------|------------------------|--------------------------|------------------------|-------------------------|
| **Kinematic Baselines** | Constant Velocity (CV) | 0.3709 | 0.2025 | 0.6881 | 0.3323 |
| | Constant Curvature (CC) | 0.3088 | 0.1698 | 0.3760 | 0.3006 |
| | Kin. MLP (No Goal, 3-in) | — | — | 0.2353 | 0.1601 |
| | Kin. MLP (With Goal, 5-in) | 0.2415 | 0.0695 | 0.2216 | 0.1240 |
| | Kin. MLP (Residual over CC) | 0.2030 | 0.0637 | 0.2470 | 0.1215 |
| **Phase 1 (Unremediated)** | STAF Epoch 1 (9-d, zero-padded) | 0.3883 | 0.1086 | 0.6320 | 0.3586 |
| **Retrained 3-Seed Suite** | No-Vision (Kinematics+Goal) | **0.1824 ± 0.0052** | 0.0479 | 0.2802 ± 0.0122 | 0.1705 ± 0.0043 |
| | Camera-Only | 0.1819 ± 0.0020 | 0.0553 | 0.2220 ± 0.0026 | 0.1770 ± 0.0019 |
| | **LiDAR-Only** | **0.1799 ± 0.0015** | 0.0546 | **0.2204 ± 0.0019** | 0.1750 ± 0.0015 |
| | Full Fusion | 0.1899 ± 0.0075 | 0.0593 | 0.2435 ± 0.0192 | 0.1833 ± 0.0061 |

> [!NOTE]
> Turning frames defined as `|steer| > 0.05` (N=1,892 out of 17,447). Straight frames: `|steer| ≤ 0.05` (N=15,555).

### 5.2 Per-Seed Breakdown — Full Fusion (High Variance)

| Seed | Full Fusion Overall ADE | Full Fusion Turning ADE | Best Val Loss |
|------|------------------------|------------------------|---------------|
| 42 | 0.1996 | 0.2681 | 0.1164 |
| 43 | 0.1814 | 0.2212 | 0.1035 |
| 44 | 0.1886 | 0.2413 | 0.1077 |
| **Mean ± Std** | **0.1899 ± 0.0075** | **0.2435 ± 0.0192** | — |

Seed 42 is notably worse (0.1996m vs. 0.1814m for seed 43), accounting for the high variance. This suggests Full Fusion training is unstable relative to single-modality variants.

### 5.3 Paired Bootstrap Confidence Intervals (132 total routes, 84 turning routes)

| Comparison | Overall ADE (mean, CI) | Turning ADE (mean, CI) | Significant? |
|------------|------------------------|------------------------|--------------|
| Camera vs. No-Vision | −0.0014m, [−0.005, +0.001] | −0.0506m, [−0.080, −0.025] | **Overall: No; Turning: Yes** |
| LiDAR vs. No-Vision | −0.0028m, [−0.006, −0.000] | −0.0520m, [−0.082, −0.025] | **Overall: marginal; Turning: Yes** |

### 5.4 Key Comparative Findings

1. **Retrained No-Vision STAF (0.1824m) outperforms the best kinematic MLP (Residual CC: 0.2030m) by ~10%** — attributable to the residual CC head design and normalized goal, not to vision.
2. **LiDAR-Only is the best single model overall (0.1799m)**, not Full Fusion.
3. **Full Fusion is NOT consistently better than its components** — it underperforms LiDAR-Only overall and shows high seed variance.
4. **Phase 1 STAF (0.3883m) was worse than the pure Constant Curvature baseline (0.3088m)** — a clear failure of the original unnormalized configuration.

---

## §6 Training, Validation, and Testing Status

### 6.1 Completed Training Runs

| Experiment | Status | Evidence |
|------------|--------|---------|
| Original 50-epoch STAF training | ✅ Complete | `logs/original_50epochs/staf_original_50epochs_progression.csv` (33 rows, epochs 18–49) |
| Phase 1 unremediated test evaluation | ✅ Complete | `results/phase1_unremediated/staf_phase1_unremediated_test_evaluation.json` |
| Retrained 3-seed suite (12 runs) | ✅ Complete | `results/retrained_variants/` (12 JSON files, all 4 variants × 3 seeds) |
| Shuffled vision controls (6 runs) | ✅ Complete | `results/slices_and_controls/shuffled_vision_controls.csv` |
| Hazard/dynamic slice analysis | ✅ Complete | `results/slices_and_controls/hazard_and_dynamic_slices.csv` |
| Paired bootstrap CI computation | ✅ Complete | `results/slices_and_controls/comprehensive_analysis_results.json` |

### 6.2 NOT Completed / Not Started

| Experiment | Status | Notes |
|------------|--------|-------|
| CARLA Longest6 closed-loop evaluation (Driving Score) | ❌ Not Done | Local TransFuser sim: 0% success (100% blocked); no STAF closed-loop runs |
| nuScenes training + evaluation | ❌ Not Done | IMU loading is a stub; no nuScenes results files |
| EuRoC ATE evaluation | ❌ Not Done | No EuRoC results files found |
| DifFUSER baseline | ❌ Not Done | `staf/baselines/diffuser_adapter.py` exists but no training/results |
| ORB-SLAM3 evaluation | ❌ Not Done | `staf/baselines/orbslam3_runner.py` exists but no results |
| Gazebo ROS 2 deployment | ❌ Not Started | Phase 5 not initiated |
| Original 4 ablations (§4.5 of plan) | ❌ Not Done | No STAF w/o soft-gating or temporal memory ablation results |
| Hyperparameter grid search | ❌ Not Done | No grid search evidence |
| AMP / `torch.compile()` training | ❌ Not Used | Retrained suite uses standard PyTorch training loop |
| Within-route / speed-bin-matched shuffled controls | ❌ Not Done | Only cross-route shuffled controls were run |

### 6.3 Data Integrity Checks

- **Test leakage:** No leakage detected. Test frames are from held-out towns (Town02, Town05). Val-based model selection is confirmed in `modal_train_retrain_variants.py` (line 248–249): best weights loaded from val checkpoint before test evaluation.
- **Val leakage in speed/kinematics:** Speed (`batch["speed"]`) comes from frame t only (confirmed in dataset code). No future frame data is used as model input.
- **Test evaluated once:** `modal_train_retrain_variants.py` runs test evaluation once per seed after training, not during training. Confirmed.
- **MLP baselines training data:** Kinematic MLP baselines were trained on the same training split, evaluated on the same test split as the STAF retrained suite.

---

## §7 Proposed Techniques vs. Actual Implementation — Classification Table

| Technique / Claim | Category | Evidence |
|-------------------|----------|---------|
| ResNet34 + ResNet18 backbone pair | ✅ **Completed** | Confirmed in `staf_model.py`; `gate_input_dim=3` in production |
| IMU SO(3) pre-integration module | ✅ **Completed** (not used in production training) | `preintegration.py` 225 lines; passes unit tests; CARLA IMU stub returns zeros |
| Soft-gating MLP (two independent heads) | ✅ **Completed** (input=3d, not 9d) | `soft_gating.py`; production training uses `gate_input_dim=3` |
| Gated cross-attention transformer (4L, 4H) | ✅ **Completed** | `fusion_transformer.py` 222 lines; bidirectional cross-attention |
| Spatio-temporal FIFO memory queue (T=5) | ✅ **Completed** | `temporal_memory.py` 176 lines; temporal transformer encoder |
| Class-balanced focal loss | ✅ **Completed** (unit-tested, not used in waypoint regression) | `focal_loss.py`; production loss is plain L1 |
| Residual CC head + zero-init last layer | ✅ **Completed** | `staf_model.py`; `use_residual_cc=True`; zero-init confirmed |
| Goal normalization (clamp/20) | ✅ **Completed** | `dataset.py`; `target_point_norm = clamp(target/20, −3, 3)` |
| Domain augmentation (color jitter, LiDAR dropout, IMU noise) | ✅ **Completed** | `dataset.py`; `augment_domain=True` for training sets |
| Backbone freeze for first 2 epochs | ✅ **Completed** | `modal_train_retrain_variants.py` lines 158–164 |
| AdamW + weight decay 1e-2 | ✅ **Completed** | Lines 147–150 of training script |
| Early stopping on val (patience=2) | ✅ **Completed** | Lines 152–246 of training script |
| ImageNet pretrained backbones | ⚠️ **Partially** | Plan required pretrained; production uses `pretrained_backbones=False` (not pretrained) |
| AMP (FP16) training | ❌ **Not Implemented** | Standard PyTorch loop; no `torch.cuda.amp` calls in training script |
| `torch.compile()` | ❌ **Not Implemented** | No compile calls in production training |
| OneCycleLR / CosineAnnealing | ❌ **Not Implemented** | Fixed LR with no scheduler in retrained suite |
| Hyperparameter grid search | ❌ **Not Implemented** | No grid search evidence |
| 9-d SO(3) kinematic gate input | ❌ **Not Used in Training** | Plan: 9-d; production: `gate_input_dim=3` |
| IMU bias estimation module | ❌ **Not Implemented** | Plan §2.3 describes online update; not in `preintegration.py` |
| CARLA Driving Score ≥ 72.0 | ❌ **Claim Not Demonstrated** | No closed-loop evaluation |
| nuScenes mIoU ≥ 73.5% | ❌ **Claim Not Demonstrated** | No nuScenes training or evaluation |
| EuRoC ATE ≤ 0.025 m | ❌ **Claim Not Demonstrated** | No EuRoC evaluation |
| Gazebo simulation validation | ❌ **Not Implemented** | Phase 5 not started |
| Dynamic gating adjusts to sensor quality | ❌ **Claim Not Supported** | gate std=0.015–0.025; r=−0.175 with speed; weakly varying |
| Full Fusion outperforms single modalities | ❌ **Claim Invalidated** | LiDAR-Only (0.1799m) beats Full Fusion (0.1899m) overall |
| Vision improves trajectory prediction | ⚠️ **Conditionally Supported** | Turning: CI [−0.080, −0.025] ✓; Overall: CI [−0.005, +0.001] ✗ |
| STAF outperforms kinematic-only baseline | ⚠️ **Weakly Supported** | No-vision STAF outperforms best kinematic MLP by ~10% (architecture credit, not vision) |
| Sensor degradation robustness | ❌ **Not Demonstrated** | No noise injection experiments; only cross-route shuffled controls |

---

## §8 Completed / Partially Completed / Pending Work

### ✅ Completed (Fully Implemented and Evaluated)

- CARLA STAF dataset pipeline (train/val/test splits, 17,447 test frames)
- Residual CC head + normalized goal design
- 3-seed retrained modality suite (no-vision, camera, LiDAR, full)
- Kinematic baselines (CV, CC, MLP variants) on same test frames and metric
- Hazard slice analysis (red-light, junction, speed-change)
- Shuffled vision controls (3 seeds each, cross-route)
- Paired bootstrap CIs (camera and LiDAR vs. no-vision, overall and turning)
- Gate statistics (mean, std, speed correlation)
- Repository reorganization (logs, results, diagnostics, figures)
- Unit tests: SO(3), IMU, soft-gating, focal loss, waypoint prediction (9 tests)

### ⚠️ Partially Completed

- **SO(3) IMU pre-integration:** Implemented and unit-tested; not evaluated end-to-end in production training (CARLA IMU stub returns near-zeros; gate input is 3-d causal)
- **Data cleaning:** `cleaning.py` implemented; disabled for CARLA data
- **TransFuser baseline:** Checkpoint verified functionally correct (181.9 FPS on T4); local closed-loop CARLA sim: 0% success rate due to environment issues
- **Ablation studies:** Modality ablations done (retrained variants); original 4 architectural ablations (no gating, no temporal, focal vs. CE, noise injection) not done

### ❌ Not Completed

- Closed-loop CARLA evaluation (Driving Score, Route Completion, Infraction Score)
- nuScenes training and evaluation (DifFUSER comparison)
- EuRoC ATE evaluation (ORB-SLAM3 comparison)
- Gazebo + ROS 2 deployment (Phase 5)
- Hyperparameter grid search
- 4 original architectural ablations from §4.5 of the plan
- Within-route and speed-bin-matched shuffled controls
- AMP, `torch.compile()`, OneCycleLR training optimizations
- IMU bias estimation online update module
- CARLA Longest6 benchmark metrics for STAF

---

## §9 Invalidated, Unsupported, and Conditionally Supported Claims

### 9.1 Invalidated Claims

| Claim | Evidence Against |
|-------|-----------------|
| **Full Fusion is the best model** | LiDAR-Only (0.1799m) outperforms Full Fusion (0.1899m) on overall ADE. Full Fusion has highest seed variance (std=0.0075). |
| **Phase 1 STAF shows meaningful learning** | Phase 1 STAF (0.3883m) is worse than even the Constant Curvature baseline (0.3088m). Gate weights collapsed to near-zero by epoch 27. |
| **Dynamic gating responds meaningfully to sensor quality** | Gate std=0.015–0.025 (weakly varying). Correlation with speed r=−0.175 (weak). Gate is essentially a learned fixed scalar per modality. |

### 9.2 Not Supported Claims

| Claim | Reason Unsupported |
|-------|--------------------|
| CARLA Driving Score ≥ 72.0 | No closed-loop evaluation performed. |
| nuScenes mIoU ≥ 73.5% | No nuScenes training or evaluation performed. |
| EuRoC ATE ≤ 0.025 m | No EuRoC evaluation performed. |
| STAF outperforms TransFuser on driving metrics | Not evaluated. Local closed-loop sim blocked. |
| Sensor degradation robustness | Only cross-route shuffled controls run; no noise injection ablations. |
| IMU pre-integration improves gating quality | 9-d gate input was never used in production; CARLA IMU returns near-zeros. |

### 9.3 Conditionally Supported Claims

| Claim | Condition |
|-------|-----------|
| Vision improves trajectory prediction | Only on turning frames (CI: [−0.080, −0.025]m). Not significant overall (CI: [−0.005, +0.001]m). |
| STAF outperforms kinematic-only baselines | No-Vision STAF (0.1824m) outperforms best kinematic MLP (0.2030m) by ~10% — this is due to architecture design (residual CC + normalized goal), not vision. |
| Camera and LiDAR provide useful signal | Supported only on turning. Shuffled controls confirm real images help on turns (real: 0.222m vs. shuffled: 0.287m). |

### 9.4 Hazard Slice Findings — Hypotheses Only

> [!WARNING]
> The following are **observations**, not established mechanisms:

| Slice | N | No-Vision ADE | Camera ADE | LiDAR ADE | Full ADE | Observation |
|-------|---|---------------|------------|-----------|----------|-------------|
| Red Light | 56 | 0.028 | 0.090 | 0.066 | 0.203 | All 56 frames are stopped (v < 0.1 m/s). No-vision correctly predicts near-zero displacement; vision models overshoot. Vision is WORSE here. |
| Junction | 174 | 1.063 | 1.994 | 1.998 | 1.935 | Vision models ~2× worse than no-vision. **Hypothesis:** visual geometric ambiguity at junctions misleads the cross-attention; mechanism not established. |
| Speed-Change | 3,088 | 0.801 | 0.857 | 0.852 | 0.831 | All models 0.80–0.86m; no meaningful differentiation. |

**Junction/Red-Light overlap:** 0 frames (confirmed from `comprehensive_analysis_results.json`). These are independent slices.

---

## §10 Techniques and Claims Now Out of Scope

The following were proposed in the original plan but are no longer part of the current research direction:

| Item | Reason Out of Scope |
|------|---------------------|
| nuScenes BEV segmentation (mIoU) | Dataset not trained on; task differs from current open-loop waypoint prediction |
| EuRoC ATE evaluation | SLAM/VIO task; not pursued |
| DifFUSER baseline comparison | No nuScenes training; adapter is a stub |
| ORB-SLAM3 evaluation | No EuRoC experiments |
| Gazebo simulation + 5 degradation scenarios | Phase 5 not started |
| ROS 2 node + TorchScript/TensorRT export | Phase 5 not started |
| B2B functional equivalence testing | Phase 5 not started |
| Three-benchmark unified summary table | Benchmarks 2 and 3 not evaluated |
| Hyperparameter grid search (LR, γ, T) | Not executed |

---

## §11 Remaining Training and Experiments Required

### 11.1 Critical Gaps (Required for Original Thesis Claims)

| Gap | Required For | Feasibility |
|-----|-------------|-------------|
| **Closed-loop CARLA evaluation (Driving Score)** | Core thesis claim: "STAF outperforms TransFuser" | Requires CARLA server + GPU; local sim blocked; Modal or RunPod viable |
| **nuScenes training + DifFUSER comparison** | nuScenes mIoU claim | Significant compute; requires fixing IMU stub |
| **EuRoC ATE evaluation** | EuRoC ATE claim | Moderate effort; requires EuRoC data access |
| **4 architectural ablations** (§4.5 of plan) | "Soft-gating adds value" claim | Manageable; same training infrastructure |

### 11.2 If Closed-Loop Evaluation Is Out of Scope (Explicit Scope Reduction)

If the thesis scope is formally reduced to open-loop waypoint prediction on CARLA:

- Acknowledge openly in thesis: "This work evaluates open-loop trajectory prediction accuracy; closed-loop driving performance is not evaluated."
- Replace DS/RC/IS targets with ADE/FDE/CI comparisons (already done).
- Compare against TransFuser only on shared ADE metric (requires running TransFuser's waypoint head on the same test frames).
- Document why closed-loop evaluation was infeasible (local sim: 0% success; cloud CARLA: requires $X budget).

### 11.3 Recommended Additional Experiments (Feasible, High Value)

| Experiment | Expected Value | Effort |
|------------|---------------|--------|
| Proper 9-d gate input run (fix CARLA IMU stub, retrain 1 seed) | Tests whether SO(3) pre-integration actually helps gating | Medium |
| Run STAF w/ constant gate (w=1.0) to isolate gating contribution | Answers ablation 1 from §4.5 | Low |
| Run STAF w/ queue length T=1 (no temporal memory) | Answers ablation 2 from §4.5 | Low |
| Pretrained backbone initialization (flip `pretrained_backbones=True`) | Tests plan claim that ImageNet init helps | Low |
| Within-route shuffled vision control | Stronger control than cross-route | Medium |

---

## §12 Final Research Contributions and Achievements

### 12.1 Demonstrated Contributions (Evidence-Based)

| Contribution | Evidence | Strength |
|-------------|----------|---------|
| Residual CC head + normalized goal design achieves state-of-the-art on CARLA open-loop waypoint prediction vs. kinematic-only baselines | No-Vision STAF (0.1824m) outperforms Residual CC MLP (0.2030m) by ~10% | Moderate (3 seeds, same test set) |
| Camera and LiDAR modalities improve trajectory prediction on turning frames | Turning ADE CI: [−0.080, −0.025]m; shuffled control confirms real images help (0.222 vs. 0.287m) | Moderate (CI verified, shuffled control) |
| Full SO(3) IMU pre-integration module with Jacobians (academic contribution) | `preintegration.py` 225 lines; 9 unit tests passing | Moderate (implementation only; not used in experiments) |
| Identification of STAF failure modes on Phase 1 training (gate weight collapse, unnormalized kinematics) and systematic remediation | Phase 1 ADE 0.388m → Retrained 0.179m (54% improvement) | Strong (before/after comparison) |
| Hazard slice analysis revealing vision models underperform no-vision on stopped frames and junctions | Red-light: no-vision 0.028m vs. full 0.203m; junction: no-vision 1.063m vs. camera 1.994m | Descriptive (mechanism is hypothesis) |

### 12.2 Architecture Components (Implemented, Not Fully Evaluated)

The following are implemented and unit-tested but not evaluated in a full benchmark:

- Bidirectional gated cross-attention fusion transformer (4 layers, 4 heads)
- Spatio-temporal FIFO memory queue with temporal transformer encoder
- Class-balanced focal loss (Cui et al., CVPR 2019 weighting)
- Full SO(3) IMU pre-integration with on-manifold Jacobians

### 12.3 What Cannot Be Claimed in the Thesis (Without New Experiments)

- STAF outperforms TransFuser on CARLA Longest6 (Driving Score)
- STAF achieves mIoU ≥ 73.5% on nuScenes
- STAF achieves ATE ≤ 0.025m on EuRoC
- Full Fusion is the best STAF configuration (LiDAR-Only is currently best)
- Dynamic gating adjusts to sensor quality (gate is near-static)
- IMU pre-integration provides end-to-end benefit (9-d gate never tested in production)

---

## §13 Recommended Next Steps

### Priority 1: Scope Clarification (Required First)

**Decision needed:** Is the thesis scope being formally reduced from the 3-benchmark system to CARLA open-loop only?

- **If YES (scope reduced):** Rewrite thesis objectives; strengthen open-loop results with the experiments in §11.3; document all out-of-scope items with justification.
- **If NO (original scope maintained):** Plan closed-loop CARLA evaluation (Modal/RunPod + CARLA container); fix nuScenes IMU stub and retrain; run EuRoC ATE evaluation.

### Priority 2: High-Impact, Low-Effort Experiments (Recommended Regardless of Scope)

1. **Constant-gate ablation (w=1.0):** Verify that soft-gating adds value over fixed uniform weights. One training run per seed = 3 runs.
2. **9-d gate input with fixed CARLA IMU stub:** Confirm that the 3-d causal vector is equivalent to or better than the 9-d pre-integrated vector in this setting.
3. **Pretrained backbone test:** Flip `pretrained_backbones=True` and train 1 seed to test the plan's ImageNet initialization claim.

### Priority 3: Thesis Writing Corrections

| Issue | Required Fix |
|-------|-------------|
| Dynamic gating claim | Replace "dynamic gating" with "learned fixed-ratio gating (std=0.015–0.025 across frames)" |
| Full Fusion is best | Replace with "LiDAR-Only achieves best overall ADE; Full Fusion adds variance without consistent gain" |
| Vision outperforms | Replace with "vision modalities help significantly on turning frames (CI: [−0.080, −0.025]m) but not overall" |
| Junction/red-light mechanisms | Label as hypotheses; add "mechanism not established" |
| IMU gate input | Clarify that 3-d causal kinematics were used as gate input in all production runs; SO(3) module is implemented but not gate-connected in training |
| Limitations section | Add: open-loop only, expert-defined turning slices (|steer|>0.05), limited routes (132), single dataset, 3 seeds, no AMP/compile optimizations |

---

## §14 Data Sources and File Index

| File | Contents |
|------|----------|
| [`results/benchmarks/staf_master_benchmark.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/benchmarks/staf_master_benchmark.csv) | All 9 model rows; overall+turning+straight ADE/FDE |
| [`results/benchmarks/test_baselines_recomputed.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/benchmarks/test_baselines_recomputed.csv) | Kinematic baselines on turning/straight subsets |
| [`results/retrained_variants/retrained_variants_summary.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/retrained_variants/retrained_variants_summary.csv) | 12 rows (4 variants × 3 seeds); all metrics per seed |
| [`results/slices_and_controls/hazard_and_dynamic_slices.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/slices_and_controls/hazard_and_dynamic_slices.csv) | Red-light (N=56), speed-change (N=3,088), junction (N=174) ADE per model |
| [`results/slices_and_controls/shuffled_vision_controls.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/slices_and_controls/shuffled_vision_controls.csv) | Camera and LiDAR shuffled turning ADE (3 seeds each) |
| [`results/slices_and_controls/comprehensive_analysis_results.json`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/slices_and_controls/comprehensive_analysis_results.json) | Paired bootstrap CIs; gate statistics; route counts; junction/red-light overlap |
| [`results/phase1_unremediated/staf_phase1_unremediated_test_evaluation.json`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/results/phase1_unremediated/staf_phase1_unremediated_test_evaluation.json) | Phase 1 STAF test results; inference-time masking (OOD) results |
| [`logs/original_50epochs/staf_original_50epochs_progression.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/logs/original_50epochs/staf_original_50epochs_progression.csv) | 50-epoch training progression; gate weight collapse evidence |
| [`logs/phase1_progression/phase1_epoch_progression.csv`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/logs/phase1_progression/phase1_epoch_progression.csv) | Phase 1 (7 epoch) progression; overfitting from epoch 2 |
| [`scripts/modal_train_retrain_variants.py`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/scripts/modal_train_retrain_variants.py) | Production training script; gate_input_dim=3; L1 loss; AdamW; early stopping |
| [`staf/models/staf_model.py`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/staf/models/staf_model.py) | Complete STAFModel implementation |
| [`staf/models/fusion_transformer.py`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/staf/models/fusion_transformer.py) | GatedFusionTransformer; bidirectional cross-attention |
| [`staf/models/temporal_memory.py`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/staf/models/temporal_memory.py) | SpatioTemporalMemoryQueue; FIFO + temporal transformer |
| [`staf/imu/preintegration.py`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/staf/imu/preintegration.py) | Full SO(3) IMU pre-integration with Jacobians |
| [`STAF_implementation_plan.md`](file:///E:/MS-DS/Thesis-1/Multi-Modal%20Fusion%20Framework%20for%20Autonomous%20Robot/STAF/STAF_implementation_plan.md) | Complete original 5-phase plan; source of all original scope claims |

---

*Report generated: 2026-11-04. All metric values sourced directly from experiment files. No interpolated or estimated values.*
