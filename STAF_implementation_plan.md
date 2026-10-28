# Spatio-Temporal Adaptive Fusion (STAF) — Implementation Plan v3 (Thesis-2)

This plan incorporates all feedback from the **Thesis-1 Examination Panel**, all 12 fixes from the [deep review](file:///C:/Users/talha/.gemini/antigravity-ide/brain/07f55711-0fdd-48da-a54f-383c203e24a9/plan_review.md), the user's updated specifications, and **all decisions from the September 2026 planning sessions**.

---

## Key Decisions & Confirmations

> [!IMPORTANT]
> 1. **Supervisor Directive:** All three baselines (TransFuser, DifFUSER, ORB-SLAM3) must be **re-trained / re-evaluated from scratch** on the full datasets — not using pretrained author checkpoints for final results.
> 2. **Compute:** Local Dell G15 (RTX 3060 6GB) used **only for code development, unit testing, and Gazebo simulation**. All neural network training runs on **RunPod Spot/Community GPU** (1× RTX 4090 24GB).
> 3. **Dataset Storage:** All three datasets permanently stored on **5TB Google Drive** (free). Transferred to cloud pod NVMe via `rclone` at pod boot time.
> 4. **Estimated Cloud Budget:** ~$27–$35 total (down from ~$145 via 6 optimization strategies).
> 5. **Metrics Rigor:** All benchmarks report standard, auditable metrics. The Thesis-1 "100% validation success" is reframed as a checkpoint integrity test (Phase 0.4).

---

## Phase 0: Literature Gap Fortification, Novelty Defense & Limitations

*Objective: Address Panel Comments #1 and #5. Justify novelty AND honestly document STAF's own limitations.*

### 0.1 — Taxonomy & Positioning Matrix
- Map existing adaptive/uncertainty fusion literature (Won et al., CLS-3D, BEVFusion, DifFUSER, TransFuser).
- Differentiate STAF: Existing methods compute uncertainty from visual/point-cloud features **post-hoc** (already corrupted during chassis vibration/blur) or use IMUs strictly for pose estimation. STAF is the first to feed pre-integrated high-frequency IMU kinematics ($200\text{ Hz}$) directly into the cross-attention gating layer as a **predictive, prior** semantic regulator.

### 0.2 — Mathematical Formalization of Novelty
- Formalize soft-gating vs. discrete thresholding vs. standard attention weighting to prove noise suppression boundaries.

### 0.3 — Formal Limitations Section *(Fix #1)*
- Document known boundary conditions of the soft-gating approach:
  - **IMU drift accumulation:** Over long sequences ($>5\text{ min}$), pre-integrated biases will drift unless periodically reset.
  - **Bias initialization sensitivity:** Poor initial bias calibration ($b^g_0, b^a_0$) degrades gating accuracy.
  - **Dual-sensor failure:** When *both* camera and LiDAR degrade simultaneously (e.g., heavy rain + night), the soft-gating mechanism has no "clean" modality to fall back on.
  - **Temporal queue latency:** The FIFO memory queue adds $T \times \Delta t$ latency to the perception pipeline.

### 0.4 — Reframe Thesis-1 Baseline Claims *(Fix #2)*
- Present the Thesis-1 TransFuser validation as a **checkpoint integrity and functional correctness test**: all 2,316 parameter groups loaded, all 36 Longest6 routes executed without runtime errors on a Tesla T4.
- Clearly distinguish this from the actual performance baselines: **Driving Score = 47.30**, **Route Completion = 93.38%** (from the official paper), which are the true comparison anchors.

---

## Phase 1: Full-Scale Data Pipeline, Cleaning & Class Balancing

*Objective: Build robust ingestion pipelines for complete datasets with data cleaning and class-imbalance correction.*

### 1.1 — Full Dataset Acquisition & Storage ✅

All datasets are downloaded and permanently stored on the user's **5TB Google Drive**:

| Dataset | Size (Compressed) | Google Drive Location | Purpose |
|---------|-------------------|----------------------|---------|
| **nuScenes v1.0 Full** | ~320 GB (10 blobs + meta + maps + CAN bus) | `Thesis_Datasets/nuscenes/` | DifFUSER training, STAF training |
| **CARLA TransFuser** | ~120 GB (13 scenario zips) | `Thesis_Datasets/carla_transfuser/` | TransFuser training |
| **EuRoC MAV** | ~27 GB (3 room zips + calibration) | `Thesis_Datasets/euroc_mav/` | ORB-SLAM3 evaluation, IMU pre-integration validation |

> [!NOTE]
> Datasets are kept **compressed** on Google Drive. They are extracted only on the cloud pod's local NVMe at training time. This avoids Google Drive's file-count quotas and the FUSE cache issue that caused disk-full errors on Colab.

### 1.2 — Per-Benchmark Comparison Strategy *(Fix #3)*

> [!IMPORTANT]
> TransFuser, DifFUSER, and ORB-SLAM3 operate on **different datasets and tasks**. Comparison is **per-benchmark**, not a single-table "apple-to-apple" ranking.

| Benchmark | Task | STAF vs. | Metrics |
|-----------|------|----------|---------|
| **CARLA Longest6** | End-to-end driving (imitation learning) | TransFuser | Driving Score (DS), Route Completion (RC), Infraction Score (IS) |
| **nuScenes val** | 3D detection + BEV segmentation | DifFUSER | mIoU, NDS, mAP |
| **EuRoC** | Visual-inertial trajectory estimation | ORB-SLAM3 | ATE (RMSE), drift (%), real-time factor |
| **Gazebo (closed-loop)** | Safety under degradation | All three | CTE, collision rate/km, SFR, latency |

A **unified summary table** in the thesis will show where each model excels and where STAF's soft-gating provides the strongest advantage.

### 1.3 — Data Cleaning with Reproducible Rules *(Fix #5)*
- Camera frames with $>30\%$ black/saturated pixels → discard.
- LiDAR sweeps with $<100$ valid points → discard.
- IMU timestamps with $>5\text{ ms}$ gap from expected interval → flag and linearly interpolate.
- Document the **exact count** of filtered samples per dataset in the thesis appendix.
- Retain natural edge-case anomalies (near-miss pedestrians, partial occlusions) in the validation split for realistic evaluation.

### 1.4 — Class-Balancing & Long-Tail Category Weighting
- Compute per-class frequency distributions across the entire training set.
- Implement **Effective Number of Samples Weighting** and **Class-Balanced Focal Loss**:
$$\mathcal{L}_{\text{balanced}} = -\alpha_c (1 - p_t)^\gamma \log(p_t)$$
- Ensure minority and safety-critical classes (pedestrians, cyclists, small obstacles, potholes, construction markers) receive equal gradient priority.

### 1.5 — Asynchronous Spatio-Temporal Synchronization
- Synchronize multi-rate sensor inputs (Camera @ $20\text{ Hz}$, LiDAR @ $10\text{ Hz}$, IMU @ $100\text{–}200\text{ Hz}$) using nearest-timestamp matching with linear interpolation for IMU.

---

## Phase 2: IMU Pre-integration & Motion Dynamics Encoding

*Objective: Implement and verify the mathematical transformation of high-frequency IMU readings into compact kinematic tensors.*

### 2.1 — Manifold Pre-integration Engine
- Implement on-manifold integration of raw accelerometer ($a^B_k$) and gyroscope ($\omega^B_k$) data:

$$\Delta R_{ij} = \prod_{k=i}^{j-1} \text{Exp}\left( (\omega^B_k - b^g_k - \eta^g_k)\Delta t \right)$$

$$\Delta v_{ij} = \sum_{k=i}^{j-1} \Delta R_{ik} (a^B_k - b^a_k - \eta^a_k) \Delta t$$

$$\Delta p_{ij} = \sum_{k=i}^{j-1} \left( \Delta v_{ik} \Delta t + \frac{1}{2} \Delta R_{ik} (a^B_k - b^a_k - \eta^a_k) \Delta t^2 \right)$$

### 2.2 — Kinematic Delta State Representation
$$X_{\text{kinematic}} = \left[ \text{Log}(\Delta R_{ij})^\top, \Delta v_{ij}^\top, \Delta p_{ij}^\top \right]^\top \in \mathbb{R}^9$$

### 2.3 — IMU Bias Estimation Module *(Fix #6)*
- **Initialization:** Estimate initial biases ($b^g_0, b^a_0$) from the first $N=200$ stationary frames (vehicle at rest) by averaging gyroscope and accelerometer readings.
- **Online Update:** Implement a lightweight sliding-window bias correction using the first-order Jacobian of the pre-integrated measurements w.r.t. bias ($\frac{\partial \Delta R}{\partial b^g}$, $\frac{\partial \Delta v}{\partial b^a}$), following the standard VINS-Mono / ORB-SLAM3 bias update formulation.
- **Validation:** Compare pre-integration trajectory against ground truth on EuRoC V1_01 sequence to verify bias estimation accuracy ($< 0.5°$ rotational drift per minute).

### 2.4 — Numerical Stability Testing
- Benchmark under extreme synthetic accelerations ($>5g$) and angular velocities ($>500°/s$) to verify numerical stability of the SO(3) exponential map.

---

## Phase 3: STAF Architecture Development & Baseline Setup

*Objective: Build the STAF model with committed architectural choices and prepare all 3 baselines for re-training.*

### 3.1 — Multi-Scale Feature Extractors *(Fix #7 — Committed Choices)*
- **Visual Stream:** ResNet34 (matching TransFuser's architecture for fair comparison).
- **LiDAR Stream:** ResNet18 on BEV-projected point clouds (matching TransFuser).
- Both backbones initialized with **ImageNet pretrained weights** (standard practice — same as TransFuser and BEVFusion papers). Only the fusion layers, gating heads, and task heads train from random init.
- **Rationale:** Using the same backbones as the primary baseline (TransFuser) isolates the contribution of the soft-gating module.

### 3.2 — IMU-Guided Learned Soft-Gating Module *(Fix #8 — Full Specification)*
Two independent MLP heads (one per modality), each with the following architecture:

```
Input: X_kinematic ∈ ℝ⁹
  → Linear(9 → 64) → LayerNorm(64) → ReLU → Dropout(0.1)
  → Linear(64 → 1) → Sigmoid
Output: w ∈ [0, 1]
```

- **Camera head:** Produces $w_{\text{cam}}$. **LiDAR head:** Produces $w_{\text{lidar}}$.
- **Total additional parameters:** ~1,666 (verified in code: $2 \times (9 \times 64 + 64 + 64 + 64 \times 1 + 1) = 1{,}666$). Negligible overhead (~0.005% of 32.8M total).
- **Soft-scaling:** $\hat{f}_{\text{cam}} = w_{\text{cam}} \cdot f_{\text{cam}}$, $\hat{f}_{\text{lidar}} = w_{\text{lidar}} \cdot f_{\text{lidar}}$.

### 3.3 — Gated Multi-Scale Cross-Attention Transformer
- Fuse dynamically weighted visual and spatial tokens:
$$f_{\text{fused}} = w_{\text{cam}} \cdot \text{Attn}(Q_{\text{lidar}}, K_{\text{cam}}, V_{\text{cam}}) + w_{\text{lidar}} \cdot \text{Attn}(Q_{\text{cam}}, K_{\text{lidar}}, V_{\text{lidar}})$$

### 3.4 — Spatio-Temporal Memory Queue
- FIFO feature token buffer: $M_{\text{temp}} = \text{Concat}(\hat{F}_{t-T}, \dots, \hat{F}_t)$ with temporal transformer encoder.
- Default queue length: $T = 5$ frames (tunable in ablation).

### 3.5 — Baseline Training Adapters

> [!IMPORTANT]
> **Supervisor Requirement:** All baselines must be re-trained from scratch, not just evaluated from author-provided checkpoints.

| Baseline | Framework | Task | Dataset | Training Status |
|----------|-----------|------|---------|-----------------|
| **TransFuser** | PyTorch (custom) | Imitation learning / driving | CARLA (8 towns) | Re-train from scratch |
| **DifFUSER** | mmdetection3d + torchpack | 3D detection + BEV seg | nuScenes (1,000 scenes) | Re-train from scratch |
| **ORB-SLAM3** | C++ (Pangolin, g2o, Eigen3) | Visual-inertial SLAM | EuRoC (11 sequences) | **No training** (classical geometry — tune parameters and evaluate) |

> [!NOTE]
> **ORB-SLAM3 clarification:** ORB-SLAM3 is a pure classical geometry system. It uses ORB feature detection, DBoW2 bag-of-words loop closure, and Levenberg-Marquardt bundle adjustment. It contains **zero neural network weights** and **cannot be trained**. The "re-training" requirement from the supervisor is fulfilled by **calibrating its parameters** (ORB feature threshold, number of keypoints, BA iterations) for each EuRoC sequence and running complete evaluation. This is the standard procedure described in the ORB-SLAM3 paper.

---

## Phase 4: High-Compute GPU Training & Ablation Suite

*Objective: Full-scale training on cloud GPU with rigorous ablation experiments, using 6 cost-optimization strategies.*

### 4.1 — Training Optimizations (6 Cost-Reduction Strategies)

| # | Strategy | Effect |
|---|----------|--------|
| 1 | **Automatic Mixed Precision (AMP — FP16)** | 2.2–2.8× faster throughput per epoch, 40% lower VRAM |
| 2 | **OneCycleLR / CosineAnnealing with Warmup** | Convergence in 15–20 epochs instead of 40–50 |
| 3 | **ImageNet Pretrained Backbone Init** | Skips 20+ wasted epochs learning low-level features |
| 4 | **RunPod Spot/Community GPU ($0.34/hr)** | 50% cheaper than On-Demand ($0.70/hr) |
| 5 | **`torch.compile()` (PyTorch 2.x)** | 15–20% automatic speed boost via fused CUDA kernels |
| 6 | **Pre-rasterized BEV Caching on NVMe** | Eliminates CPU dataloader bottleneck → 100% GPU utilization |

### 4.2 — Training Configuration & Hyperparameter Search *(Fix #9)*
- **Initial Grid Search** (on 10% validation subset, ~4h):
  - Learning rate: $\{1 \times 10^{-4}, 3 \times 10^{-4}, 5 \times 10^{-4}\}$
  - Focal loss $\gamma$: $\{1.0, 2.0, 3.0\}$
  - Temporal queue length $T$: $\{3, 5, 8\}$
- **Full Training:** PyTorch AMP, gradient clipping (max norm = 1.0), `torch.compile()`.
- **Differential Learning Rates:** Backbone params at $0.1 \times \text{LR}$, new modules (gating, fusion, temporal) at $1.0 \times \text{LR}$.

### 4.3 — Optimized Training Cost Estimate *(Fix #10 — Updated)*

| Model | Epochs | GPU-Hours (1× RTX 4090) | Cost @ $0.34/hr (Spot) |
|-------|--------|-------------------------|------------------------|
| TransFuser re-training | 20 | ~12 hrs | ~$4.10 |
| DifFUSER re-training | 12 | ~20 hrs | ~$6.80 |
| STAF full training | 18 | ~27 hrs | ~$9.20 |
| Ablation experiments (4×) | 10 each | ~20 hrs | ~$6.80 |
| **Total** | — | **~79 hrs** | **~$26.90** |

> [!NOTE]
> Previous estimate was ~264 hrs / ~$93 (before optimizations). The 6 strategies above reduce this by **70%**.

### 4.4 — Baselines Re-training
- **TransFuser:** Re-train on full CARLA dataset (8 towns, all weather presets) using the official `team_code_transfuser` training script with our AMP + OneCycleLR optimizations.
- **DifFUSER:** Re-train on full nuScenes dataset under the BEVFusion training protocol with `torchpack` and our cached BEV optimization.
- **ORB-SLAM3:** Run full evaluation on all 11 EuRoC sequences (MH01–MH05, V101–V103, V201–V203) in stereo-inertial mode. Calibrate ORB parameters per-sequence. Report ATE RMSE against Vicon/Leica ground truth.

### 4.5 — Ablation Studies (Addressing Panel Comment #4)

| # | Ablation | What It Proves |
|---|----------|----------------|
| 1 | STAF vs. STAF w/o soft-gating ($w = 1.0$) | Quantitative impact of IMU-guided gating |
| 2 | STAF with vs. without temporal memory queue | Value of spatio-temporal consistency |
| 3 | Class-Balanced Focal Loss vs. Cross-Entropy | Impact on minority class recall |
| 4 | Performance under injected noise ($\sigma_{\text{blur}}$, $p_{\text{drop}}$) | Robustness to sensor degradation |

### 4.6 — Cloud Deployment & Data Pipeline

Training workflow for the RunPod instance:

```
Boot Pod (1× RTX 4090, 1TB NVMe, 64GB RAM)
  │
  ├─ Install rclone → Configure Google Drive remote
  ├─ rclone copy gdrive:Thesis_Datasets/ /workspace/data/ (~45 min)
  ├─ Extract archives on NVMe (~15 min)
  │
  ├─ Train TransFuser → Auto-sync checkpoints to Drive
  ├─ Train DifFUSER → Auto-sync checkpoints to Drive
  ├─ Train STAF → Auto-sync checkpoints to Drive
  ├─ Run 4 Ablation experiments → Auto-sync results to Drive
  │
  └─ Terminate pod (stop billing)
```

- **Auto-resume:** Every epoch checkpoint is saved locally AND synced to Google Drive. If the Spot instance is interrupted, reconnect and resume from latest epoch.
- **Total pod uptime:** ~80 hours spread across multiple sessions.

---

## Phase 5: ROS 2 Deployment & Gazebo Closed-Loop Validation

*Objective: Deploy as a modular ROS 2 node and validate closed-loop safety in prioritized simulation scenarios.*

### 5.1 — C++ LibTorch ROS 2 Node
- Export via TorchScript / TensorRT.
- Asynchronous ROS 2 node: subscribes to `/camera/image_raw`, `/lidar/point_cloud`, `/imu/data_raw`.
- Publishes to `/staf/semantic_grid` (`nav_msgs/OccupancyGrid`).

### 5.2 — Back-to-Back (B2B) Functional Equivalence *(Fix #12)*
- Play `.rosbag` through both offline Python model and live C++ ROS 2 node.
- **Equivalence Criteria** (following Autoware Toolbox 2 MBD methodology [cheng2025autoware]):
  - Per-pixel semantic label agreement $> 99.5\%$ (Cohen's $\kappa > 0.99$).
  - Maximum floating-point divergence in attention weight tensors $< 10^{-4}$.

### 5.3 — Prioritized Gazebo Scenarios *(Fix #11 — Trimmed to 5 Core)*

> [!TIP]
> Instead of 16+ scenarios, we focus on **5 high-impact scenarios** that directly demonstrate the soft-gating mechanism's value. Each scenario targets a specific sensor degradation mode.

| # | Scenario | Primary Degradation | Expected STAF Response |
|---|----------|--------------------|-----------------------|
| 1 | **High-speed cornering** | Camera motion blur | $w_{\text{cam}} \to 0$, rely on LiDAR |
| 2 | **Rough terrain / potholes** | LiDAR rolling skew + chassis vibration | $w_{\text{lidar}}$ reduced, temporal memory stabilizes |
| 3 | **Tunnel entry/exit** | Sudden lighting change (glare → darkness) | $w_{\text{cam}}$ drops during transition |
| 4 | **Heavy rain / fog** | LiDAR scattering + reduced camera visibility | Both weights adjust; temporal memory maintains context |
| 5 | **Nominal urban driving** | None (baseline comparison) | $w_{\text{cam}} \approx w_{\text{lidar}} \approx 1.0$ |

- Use existing Gazebo community worlds as starting points (e.g., `turtlebot3_world`, `aws-robomaker-small-house-world`).
- Add synthetic noise **programmatically** (Gaussian blur kernel on camera, random point dropout on LiDAR) rather than building photorealistic environments from scratch.

### 5.4 — Safety-Critical Benchmarking
Compare STAF against all 3 baselines:

| Metric Category | Metrics |
|-----------------|---------|
| **Driving Safety** | ATE (RMSE), Cross-Track Error (CTE), Collision Rate per km |
| **Perception Quality** | mIoU (clean vs. degraded), Semantic Flickering Rate (SFR), Minority class recall |
| **Computational** | Inference latency (ms), throughput (FPS), GPU memory footprint |

---

## Verification Plan

### Offline ML Metrics
- mIoU and mAP on nuScenes val split (STAF vs. DifFUSER).
- Driving Score, RC, IS on CARLA Longest6 (STAF vs. TransFuser).
- ATE on EuRoC sequences (STAF IMU module vs. ORB-SLAM3).
- Class-Balanced Focal Loss impact on minority class recall.

### Online Simulation & ROS 2 Validation
- Automated Gazebo test batches across all 5 prioritized scenarios.
- End-to-end latency target: $< 20\text{ ms}$ ($\geq 50\text{ FPS}$).
- Publication-ready comparison tables and TikZ/PGFPlots diagrams.
