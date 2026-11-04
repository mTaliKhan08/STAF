# STAF Phase 0.5: Task 4 & Task 5 Empirical Investigation

## Task 5: Root-Cause Breakdown of Cleaner Rejection
Auditing DataCleaner over 150 samples...
### Cleaner Breakdown (N=150):
- **Camera Rejections (>30% bad pixels)**: **28 / 150** (18.7%)
  - Mean black pixel ratio: **15.36%** (hood + shadows)
  - Mean saturated pixel ratio: **0.01%** (sky/glare)
- **LiDAR Loader Format/Read Failures**: **0 / 150** (0.0%)
- **LiDAR Genuinely Sparse (<100 points)**: **0 / 150** (0.0%)
- **LiDAR Valid Sweep Point Counts**: Mean = **31478**, Min = **30279**, Max = **32864**
- **Combined Filter Rejection Rate**: **28 / 150** (18.7%)

## Task 4: Heading Convention Verification & 30-Route BEV Plot
Evaluating 100 high-speed sequences (speed > 3.0 m/s):
- **Mode `direct`**: Mean |forward_disp - speed * t| = **19.0700 m**
- **Mode `theta_minus_pi2`**: Mean |forward_disp - speed * t| = **7.1145 m**
- **Mode `transfuser_lbc`**: Mean |forward_disp - speed * t| = **3.1723 m**
- **Mode `matrix`**: Mean |forward_disp - speed * t| = **19.0617 m**

-> **Selected Heading Convention**: **`transfuser_lbc`** (minimum forward kinematic error: **3.1723 m**)

Collected 30 distinct route trajectories:
- Forward displacement range at 2.0s: **[-26.56 m, 26.59 m]** (mean = 1.21 m)
- Lateral displacement range at 2.0s: **[-39.10 m, -1.21 m]** (mean = -34.61 m)
- Lateral standard deviation across 30 routes: **7.5715 m** (> 0 confirmed!)
- Saved BEV plot of 30 route trajectories: `phase05_bev_30_routes.png`
