# STAF Experimental Results & Benchmarks Directory

This directory stores all empirical benchmarks, evaluation payloads, and ablation metrics for the Spatio-Temporal Attention Fusion (STAF) framework across held-out test environments (CARLA `Town02` and `Town05`, 17,447 frames).

## Subdirectories:
1. `benchmarks/`:
   - `staf_master_benchmark.csv`: Master cross-architecture comparison containing all kinematic baselines, Phase 1 unremediated STAF, and remediated STAF variants (mean +- std and median).
   - `test_baselines_recomputed.csv`: Detailed baseline evaluations across Turning (N=1,892) and Straight (N=15,555) frames.
2. `retrained_variants/`:
   - The 12 independently retrained models across seeds [42, 43, 44] for Full Fusion, Camera-Only, LiDAR-Only, and No-Vision.
   - `retrained_variants_summary.csv`: Aggregated spreadsheet with bootstrap confidence intervals.
3. `phase1_unremediated/`:
   - `staf_phase1_unremediated_test_evaluation.json`: Exact evaluation of the initial Phase 1 STAF checkpoint (Epoch 1) and inference-time masking diagnostics.
4. `slices_and_controls/`:
   - `hazard_and_dynamic_slices.csv`: Sliced error breakdown for Red Light (N=56), Dynamic Speed Change (N=3,088), and Junction (N=174).
   - `shuffled_vision_controls.csv`: Visual perturbation sanity checks (speed-matched and cross-route).
   - `comprehensive_analysis_results.json`: Full nested payload with route-paired bootstrap distributions.
