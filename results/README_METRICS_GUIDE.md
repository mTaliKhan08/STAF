# STAF Training, Validation & Test Logs Directory

This directory centralizes all exported logs, numerical metrics, checkpoint progressions, and empirical evaluations downloaded from Modal directly into your local project workspace.

---

## Directory Structure & Contents

### 1. `retrained_variants_metrics/`
Contains the raw JSON evaluation payloads and CSV summary for all 12 retrained models (4 architectures × 3 random seeds):
- **Raw Metric JSONs:**
  - `no_vision_seed42.json`, `no_vision_seed43.json`, `no_vision_seed44.json`
  - `camera_only_seed42.json`, `camera_only_seed43.json`, `camera_only_seed44.json`
  - `lidar_only_seed42.json`, `lidar_only_seed43.json`, `lidar_only_seed44.json`
  - `full_seed42.json`, `full_seed43.json`, `full_seed44.json`
- **Spreadsheet Summary:**
  - `retrained_variants_summary.csv`: Aggregated spreadsheet containing Overall, Turning, and Straight ADE/FDE metrics, standard deviations, and bootstrap confidence intervals.

### 2. `test_evaluation_comprehensive/`
Contains benchmark evaluations across the 17,447 held-out test frames (`Town02` and `Town05`):
- `comprehensive_analysis_results.json`: Master JSON with all baseline evaluations, route-paired bootstrap distributions, hazard slice numbers, and soft-gating statistics.
- `test_baselines_recomputed.csv`: Full test set performance across Turning ($N=1,892$) and Straight ($N=15,555$) frames for Constant Velocity, Constant Curvature, Kinematics MLP (3-in), Kinematics MLP (5-in), Kinematics MLP Residual over CC, and STAF.
- `hazard_and_dynamic_slices.csv`: Sliced error breakdown for Red Light ($N=56$), Dynamic Speed Change ($N=3,088$), and Junction ($N=174$).
- `shuffled_vision_controls.csv`: 3-seed empirical turning ADE values under cross-route and within-route speed-matched visual perturbations.
- `junction_10_worst_examples.png`: Visualization of the top 10 worst junction prediction cases.

### 3. `phase1_training_progression/`
Tracks the training loss, validation loss, validation ADE/FDE, and soft-gating attention weights epoch-by-epoch across training towns:
- `phase1_epoch_progression.json`
- `phase1_epoch_progression.csv`
- Records the best checkpoint at Epoch 1 (Val L1 = 0.1756 m, Val ADE = 0.3154 m).

---

## How to Inspect the Stats
- **Excel / Google Sheets:** Double-click any `.csv` file to inspect or generate thesis charts.
- **Python / Pandas:**
  ```python
  import pandas as pd
  df_retrained = pd.read_csv("logs_and_metrics/retrained_variants_metrics/retrained_variants_summary.csv")
  df_baselines = pd.read_csv("logs_and_metrics/test_evaluation_comprehensive/test_baselines_recomputed.csv")
  df_training = pd.read_csv("logs_and_metrics/phase1_training_progression/phase1_epoch_progression.csv")
  df_hazards = pd.read_csv("logs_and_metrics/test_evaluation_comprehensive/hazard_and_dynamic_slices.csv")
  print(df_retrained)
  ```
