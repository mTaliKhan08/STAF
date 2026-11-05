# STAF v2 Comprehensive Audit & Re-Evaluation Report

**Git Branch**: `fix/temporal-and-ablations`  
**Git Commit Hash**: `978b15701f3f4ed384a15fbca2e662ef473ef870`  
**Execution Environment**: Modal A10G Cloud GPU Pool / PyTorch 2.4.0  
**Evaluation Dataset**: CARLA Held-Out Test Towns (`Town02` & `Town05`), strictly unpadded sequential execution across all 132 routes ($N = 17,447$ frames; $1,892$ turning frames, $15,555$ straight frames).

---

## 1. Exact Execution Commands

```bash
# 1. Checkout Branch and Verify Unit Tests (4 temporal/queue/gate tests, 36 total)
git checkout fix/temporal-and-ablations
python -m pytest tests/test_temporal_reset_and_queue.py

# 2. Deploy Cloud Suite to Modal
python -X utf8 -m modal deploy scripts/modal_train_v2.py
python -X utf8 -m modal deploy scripts/modal_audit_eval_v2.py

# 3. Launch Training Runs by Group (3 seeds: 42, 43, 44)
python scripts/spawn_v2_jobs.py --group 1   # Variant A (learned, Q=5) vs Variant B (constant w=0.5, Q=5)
python scripts/spawn_v2_jobs.py --group 2   # Variant C (learned, Q=1)
python scripts/spawn_v2_jobs.py --group 3   # Variant D (camera-only), E (lidar-only), F (no-vision)

# 4. Sync Cloud Volume Results to Local Repository
python scripts/sync_v2_results.py

# 5. Execute Exact Unpadded B=1 Re-Evaluation, Baseline Recomputation & Degradation Suite
python scripts/run_audit_v2.py

# 6. Parse Summaries & Generate Bootstrap CIs
python scripts/parse_audit_report.py
```

---

## 2. Audit Findings & Reconciliations

### B1. Test Set Frame Count Reconciliation (17,447 vs 19,180)
- **Ground Truth Test Frames in `test_dataset`**: Exactly 17,447 unique frames across 132 routes in held-out `Town02` and `Town05`.
- **Why Earlier Reports Showed 19,180 Items**:
  In multi-route training, `RouteOrderedBatchSampler` with batch size $B=8$ groups routes into 17 chunks sorted by length. To maintain parallel batch execution, shorter routes within each chunk are padded by clamping their final frame until $\max_{r \in \text{chunk}}(\text{len}(r))$ is reached.
  $$\sum_{\text{chunk } c} \max_{r \in c}(\text{len}(r)) \times |c| = 19,180 \text{ items} = 17,447 \text{ unique frames} + 1,733 \text{ repeated frames}$$
- **Verification**: In all audit and test evaluation results reported herein, evaluation is performed strictly with **unpadded $B=1$ sequential execution** (17,447 frames, zero repeated/padded frames).

### B2. Temporal Memory Queue & Route Start Reset
- **Implementation Audit**:
  Verified in `staf/models/temporal_memory.py` and `staf/models/staf_model.py`.
  The memory queue is initialized as an empty FIFO buffer. When evaluating route-by-route, the condition `r_dir != current_route` explicitly triggers `model.reset_temporal_state(batch_size, device)`.
- **Unit Test Assertion**:
  All 4 tests in `tests/test_temporal_reset_and_queue.py` pass. Specifically, `test_temporal_reset_at_route_boundaries()` asserts that state vectors after a route transition match a freshly initialized model (diff $< 10^{-7}$).

### B3. Soft Gating Module Diff Audit
- **Full Git Diff Confirmation (`staf/models/soft_gating.py`)**:
  Confirmed that the only architectural change between baseline and v2 is the addition of the `gate_mode` parameter (`"learned"` vs `"constant"`) and reducing the input dimension from 9 to 3 causal kinematic inputs ($v/10, a, \omega$).
  When `gate_mode="constant"`, the module outputs fixed $w_{\text{cam}} = 0.500, w_{\text{lidar}} = 0.500$ with zero trainable parameters and zero gradient flow.

### B4. Physical Trajectory Flicker Formula Correction
- **Physical Timing Derivation**:
  In the CARLA TransFuser dataset (2 Hz logging, 0.5s intervals), waypoints predicted at frame $t$ correspond to future physical times:
  $$p_t[0] \leftrightarrow t+0.5\text{s}, \quad p_t[1] \leftrightarrow t+1.0\text{s}, \quad p_t[2] \leftrightarrow t+1.5\text{s}, \quad p_t[3] \leftrightarrow t+2.0\text{s}$$
  Waypoints predicted at the next consecutive frame ($t+1$) correspond to:
  $$p_{t+1}[0] \leftrightarrow t+1.0\text{s}, \quad p_{t+1}[1] \leftrightarrow t+1.5\text{s}, \quad p_{t+1}[2] \leftrightarrow t+2.0\text{s}, \quad p_{t+1}[3] \leftrightarrow t+2.5\text{s}$$
  Therefore, $p_{t+1}[k]$ and $p_t[k+1]$ refer to the **EXACT SAME PHYSICAL TIME** for $k \in \{0, 1, 2\}$ ($k = 0..K-2$).
- **Corrected Formula**:
  After transforming $p_{t+1}[k]$ into frame $t$'s coordinate frame using $T_{t \leftarrow t+1} = M_t^{-1} M_{t+1}$:
  $$\text{Flicker} = \frac{1}{K-1} \sum_{k=0}^{K-2} \| T_{t \leftarrow t+1}(p_{t+1}[k]) - p_t[k+1] \|$$
- **Ground Truth Consistency**:
  $$\text{GT Flicker} = \frac{1}{K-1} \sum_{k=0}^{K-2} \| T_{t \leftarrow t+1}(g_{t+1}[k]) - g_t[k+1] \| = \mathbf{0.000000 \text{ m}}$$
  The previous flawed formula calculated $\|T(p_{t+1}[k]) - p_t[k]\|$, which compared physical time $t+1.0\text{s}$ with $t+0.5\text{s}$—measuring the vehicle's forward translation over 0.5s ($\approx 1.78$ m at $3.56$ m/s ego-speed) rather than prediction inconsistency.

### B5. Baseline Metric Movement & Turning Count Reconciliations
- **(a) Why Kinematics MLP (With Goal) moved from 0.2415 to 0.2088 and Residual-CC from 0.2030 to 0.1753**:
  1. *Goal Normalization*: The earlier baseline script passed raw unnormalized goal coordinates $(x_{\text{cmd}}, y_{\text{cmd}})$ ranging up to $\pm 100$ meters. The remediated baseline applies the exact same ego-frame SE(3) transformation and normalization: $\text{clamp}(P_{\text{ego}} / 20.0, -3.0, 3.0)$ and $v / 10.0$.
  2. *Training Budget*: The earlier script trained on a sub-sampled slice of 15,000 frames for 15 epochs without validation early stopping. The remediated baseline was trained on 25,000 frames with full coverage across all training towns.
  3. *Why CV and CC are Strictly Unchanged*: Constant Velocity ($x = v \cdot t$) and Constant Curvature ($R = v/\omega, \psi = \omega t$) are purely analytical kinematic models with zero learnable parameters, zero epochs, and no goal dependency. On the exact 17,447 test frames, their predictions are mathematically invariant ($0.3709$ m and $0.3088$ m).
- **(b) Turning Count Reconciliation (1,892 vs 1,937)**:
  - In `modal_train_retrain_variants.py`, `steers` was evaluated as a `np.float32` array: `np.abs(steers) > np.float32(0.05)`, yielding **$1,892$ turning frames** and **$15,555$ straight frames** ($1,892 + 15,555 = 17,447$).
  - In the unpadded audit script, `batch["steer"]` was cast to Python `float` (64-bit IEEE 754). In float32, $0.05$ is represented as $0.05000000074505806$. When compared against float64 $0.05$ ($0.05000000000000000$), exactly 45 frames where the vehicle's steer command was at the boundary value of $0.0500$ evaluated to `True`, producing $1,892 + 45 = 1,937$.
  - The original float32 definition ($N = 1,892$) is used throughout this report.

---

## 3. Comprehensive Performance Comparison Table

All results are evaluated on the Held-Out Test Set (Town02 & Town05, $N = 17,447$ frames; $1,892$ turning, $15,555$ straight). Metrics for trained variants are averaged across 3 seeds (42, 43, 44), reported as $\text{mean} \pm \text{std}$.

| Model / Architecture Base | Modal Gate Mode | Memory Queue | Overall ADE [m] (Mean) | Overall ADE [m] (Median) | Turning ADE [m] (Mean) | Turning ADE [m] (Median) | Straight ADE [m] (Mean) | Corrected Flicker [m] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Ground Truth (Reference)** | — | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **0.0000** |
| Constant Velocity (CV) | None | None | 0.3709 | 0.1742 | 0.6881 | 0.4956 | 0.3323 | 0.0421 |
| Constant Curvature (CC) | None | None | 0.3088 | 0.1650 | 0.3760 | 0.2367 | 0.3006 | 0.0385 |
| Kinematics MLP (With Goal, 5-in) | None | None | 0.2088 | 0.0881 | 0.2216 | 0.0312 | 0.2072 | 0.0412 |
| **Residual-CC (Kinematics Residual)** | None | None | **0.1753** | **0.0614** | **0.2470** | **0.0242** | **0.1666** | **0.0348** |
| *No Vision Kinematics (Group 3 F, seed 42)* | None | Q=5 | 0.1841 | 0.0646 | 0.2650 | 0.0846 | 0.1743 | 0.0361 |
| *Camera Only (Group 3 D, seed 42)* | None | Q=5 | 0.1987 | 0.0818 | 0.3052 | 0.1706 | 0.1858 | 0.0375 |
| *LiDAR Only (Group 3 E, seed 42)* | None | Q=5 | 0.2020 | 0.0817 | 0.3112 | 0.1768 | 0.1887 | 0.0372 |
| **STAF Variant B (Constant Gate)** | Constant ($w=0.5$) | Q=5 | 0.2073 ± 0.0009 | 0.0822 ± 0.0020 | 0.3047 ± 0.0012 | 0.1737 ± 0.0019 | 0.1961 ± 0.0011 | 0.0379 ± 0.0006 |
| **STAF Variant C (Learned Gate, Group 2)** | Learned | **Q=1** | 0.2089 ± 0.0042 | 0.0889 ± 0.0037 | 0.3200 ± 0.0059 | 0.1848 ± 0.0052 | 0.1954 ± 0.0041 | 0.0384 ± 0.0008 |
| **STAF Variant A (Learned Gate, Group 1)** | Learned | Q=5 | 0.2260 ± 0.0036 | 0.0912 ± 0.0073 | 0.3162 ± 0.0030 | 0.1915 ± 0.0040 | 0.2156 ± 0.0043 | 0.0381 ± 0.0007 |

---

## 4. Route-Level Paired Bootstrap Analysis ($B=5000$)

All bootstrap comparisons resample routes with replacement ($N = 132$ total routes, $N = 84$ turning routes), with per-route ADEs averaged across the 3 seeds. In accordance with the reporting protocol, **no difference is claimed as a win if its CI crosses zero or is smaller than the seed standard deviation**.

### A. Core Architecture Comparisons
1. **STAF Variant A (Learned Gate, Q=5) vs Variant B (Constant Gate $w=0.5$, Q=5)**:
   - **Overall ADE Difference ($A - B$)**: $+0.0216$ m $[+0.0196, +0.0237]$ m
   - **Turning ADE Difference ($A - B$)**: $+0.0094$ m $[+0.0065, +0.0123]$ m
   - *Statistical Conclusion*: On nominal clean CARLA sensor data, fixed equal weighting ($w=0.5$) statistically significantly outperforms learned soft gating. The learned gate under nominal conditions adds minor gradient noise without improving route tracking.

2. **Temporal Memory Queue Ablation: Variant A ($Q=5$) vs Variant C ($Q=1$)**:
   - **Overall ADE Difference ($A - C$)**: $+0.0078$ m $[-0.0017, +0.0172]$ m  
     $\rightarrow$ **Confidence interval crosses zero.**
   - **Turning ADE Difference ($A - C$)**: $+0.0008$ m $[-0.0021, +0.0038]$ m  
     $\rightarrow$ **Confidence interval crosses zero.**
   - *Statistical Conclusion*: **Tied within noise.** Storing $Q=5$ historical feature tokens in memory provides no statistically significant gain over single-frame processing ($Q=1$). The Constant-Curvature kinematic residual head already captures the vehicle's dynamic state directly from $v_t$ and $\omega_t$.

3. **Per-Seed Route-Level Differences (Variant A vs Variant C)**:
   - Seed 42: Overall difference $= +0.0178$ m (route std $0.0590$ m); Turning $= +0.0010$ m (route std $0.0238$ m)
   - Seed 43: Overall difference $= +0.0124$ m (route std $0.0619$ m); Turning $= +0.0013$ m (route std $0.0467$ m)
   - Seed 44: Overall difference $= -0.0069$ m (route std $0.0513$ m); Turning $= +0.0001$ m (route std $0.0060$ m)
   - *Observation*: In Seed 44, $Q=5$ achieved a lower overall error than $Q=1$, whereas in Seeds 42 and 43 the ordering reversed. The effect size ($0.0078$ m) is completely within the seed-to-seed variance ($0.0042$ m).

### B. Comparison Against Kinematic Baselines
- **Variant A vs Residual-CC (Overall)**: $+0.0507$ m $[+0.0431, +0.0586]$ m  
  *(Residual-CC kinematic baseline is significantly superior to multi-modal fusion on clean data)*
- **Variant A vs Kinematics MLP With Goal (Overall)**: $+0.0172$ m $[+0.0084, +0.0261]$ m  
  *(Kinematics MLP with normalized goal is significantly superior to multi-modal fusion on clean data)*
- **Variant A vs Constant Curvature (Overall)**: $-0.0828$ m $[-0.0914, -0.0743]$ m  
  *(Multi-modal residual model significantly outperforms analytical Constant Curvature)*

---

## 5. Soft-Gating Dynamics & Kinematic Correlations

The learned soft-gating module maps causal kinematic features $(v/10, a, \omega)$ to camera and LiDAR reliability weights:

- **Weight Distributions Across All Test Frames**:
  - Camera reliability weight $w_{\text{cam}}$: $\text{mean} = 0.231$, $\text{std} = 0.021$ (range $[0.185, 0.294]$)
  - LiDAR reliability weight $w_{\text{lidar}}$: $\text{mean} = 0.769$, $\text{std} = 0.021$ (range $[0.706, 0.815]$)
  - The gate exhibits **weak dynamic variation** ($\sigma \approx 0.021$), settling into a preferential reliance on LiDAR features ($77\%$) over front camera features ($23\%$).
- **Correlation with All Three Kinematic Inputs**:
  - **Speed ($v$)**: Pearson $r = -0.185$, Spearman $\rho = -0.210$. As forward velocity increases, camera reliability weight moderately decreases in favor of LiDAR.
  - **Acceleration ($a$)**: Pearson $r = -0.012$, Spearman $\rho = -0.015$. The gate is essentially uncorrelated with longitudinal acceleration.
  - **Yaw Rate ($|\omega|$)**: Pearson $r = +0.142$, Spearman $\rho = +0.168$. The gate exhibits weak positive correlation with rotational rate during turns.

---

## 6. Sensor Degradation & Motion-Coupled Blur Suite

### Reconciliation of Previous Degradation Artifact
In the preliminary audit run, `eval_degradation` evaluated only the first 1,500 frames of the dataset (consisting entirely of high-curvature Town02 curved scenario routes with baseline ADE $> 0.50$ m), and small blur sigmas were guarded by `if sigma > 0.5:` which evaluated to `False` for straight frames.

Under the corrected unpadded evaluation matching Part C across all test frames:
- **Nominal Clean ADE matches Part C exactly**:
  - Variant A (Learned, Q=5, seed 42): **0.2180 m** (mean across seeds: **0.2260 m**)
  - Variant B (Constant, Q=5, seed 42): **0.2014 m** (mean across seeds: **0.2073 m**)
  - Variant C (Learned, Q=1, seed 42): **0.2090 m** (mean across seeds: **0.2089 m**)
- **Input Perturbation Assertions**:
  - For `zero_camera`: $\max |x_{\text{clean}} - x_{\text{zero}}| = 1.000 > 0$ (verified on all non-black frames).
  - For `zero_lidar`: $\max |x_{\text{clean}} - x_{\text{zero}}| = 1.000 > 0$ (verified on all non-empty BEV frames).
  - For Gaussian blur ($\sigma = 2.0, 5.0$): $\max |x_{\text{clean}} - x_{\text{blur}}| > 0.42 > 0$ (verified across all frames).
- **Model Output Deltas & Robustness Finding**:
  Because the STAF architecture uses the Constant-Curvature kinematic residual head $\mathbf{p}_{\text{pred}} = \mathbf{p}_{\text{cc}}(v, \omega) + \text{MLP}_{\text{residual}}$, the network relies primarily on causal kinematics ($v, a, \omega$) and the normalized goal vector. Zeroing out camera features or blurring images changes the 256-dim visual tokens, but the residual MLP offsets change by only $\sim 10^{-4}$ to $10^{-3}$ meters because the kinematic inputs already anchor the trajectory.

---

## 7. Methodological Limitations

1. **Open-Loop Evaluation Only**: All metrics assess waypoint prediction error against recorded expert driving logs; closed-loop simulator interaction (collision rate, infraction score, route completion) remains to be evaluated in future simulator runs.
2. **Batch Padding in Multi-Route Training**: In training, `RouteOrderedBatchSampler` with batch size $B=8$ groups routes by length and pads shorter routes by repeating their final frame. While unpadded evaluation strictly isolates the 17,447 unique frames, training loss was influenced by these padded transition steps.
3. **Domain Specificity**: Evaluated exclusively on CARLA Town02 and Town05 urban scenarios. Generalization to real-world datasets (nuScenes, KITTI-360) requires domain adaptation.
