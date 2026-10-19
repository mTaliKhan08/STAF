# STAF: Spatio-Temporal Attention Fusion Framework for Autonomous Robot Navigation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research: Master's Thesis](https://img.shields.io/badge/Research-MS--DS%20Thesis-purple.svg)](https://github.com/mTaliKhan08/STAF)

> **Official PyTorch implementation of the Spatio-Temporal Attention Fusion (STAF) framework for robust, multi-modal autonomous robot navigation and Bird's-Eye-View (BEV) perception under adverse environmental conditions.**

---

## 📌 Abstract & Research Overview

Reliable autonomous robot navigation in dynamic, real-world, and weather-degraded scenarios demands robust perception across complementary sensing modalities. Conventional fusion paradigms frequently exhibit single-modality dominance and catastrophic failure when individual sensors degrade (e.g., optical blinding under sudden glare or darkness, LiDAR point cloud scattering in heavy rain/fog, and visual odometry drift under aggressive kinematic shocks).

**STAF (Spatio-Temporal Attention Fusion)** addresses these fundamental challenges by integrating:
1. **Multi-Modal Cross-Attention:** Learns rich cross-sensory representations across synchronized Camera imagery, LiDAR point clouds, and high-frequency IMU streams.
2. **Dynamic Soft-Gating Mechanism:** Dynamically reweights sensory modalities via an ultra-lightweight MLP (only **1,666 parameters**, 0.005% compute overhead) conditioned on environmental degradation signals.
3. **On-Manifold $SO(3)$ Kinematic Prior:** Formulates continuous-time IMU pre-integration to stabilize spatial cross-attention matrices during rapid ego-motion.
4. **Spatio-Temporal FIFO Memory Queue:** Maintains sliding-window temporal feature representations to maintain persistent tracking during transient sensor blackouts.

---

## 🔬 Cross-Baseline Comparison Matrix

STAF is benchmarked against canonical state-of-the-art baselines across their respective gold-standard evaluation suites:

| Evaluation Criterion | TransFuser Baseline | DifFUSER Baseline | ORB-SLAM3 Baseline | **Proposed STAF Framework** |
| :--- | :---: | :---: | :---: | :---: |
| **Primary Reference** | CVPR 2021 / IEEE TPAMI 2022 | arXiv 2023 / 2024 | IEEE T-RO 2021 | **Proposed Thesis Work (2026)** |
| **Benchmark Suite** | CARLA Longest6 (36 Routes) | nuScenes v1.0 (Full Dataset) | EuRoC MAV (`MH_01_easy`) | **nuScenes v1.0 + CARLA Longest6** |
| **Target Task** | End-to-End Driving | BEV Map Segmentation | Metric 6-DoF Pose | **Unified Perception & Trajectory** |
| **Primary Metric** | Driving Score: `47.30` | Mean IoU: `59.85%` | ATE RMSE: `0.035 m` | **Target DS: $>55.0$ / NDS: $>0.62$** |
| **Input Modalities** | Camera + LiDAR | Camera + LiDAR + HD Map | Stereo Camera + IMU | **Camera + LiDAR + 200Hz IMU** |
| **IMU Integration** | ❌ None (velocity scalar) | ❌ None | ✅ Pre-integration (Pose only) | **✅ On-manifold $SO(3)$ Kinematic Prior** |
| **Degraded Vision Resilience** | ❌ Severe drop ($>40\%$) | ❌ Severe drop ($>35\%$) | ❌ Feature tracking lost | **✅ Adaptive Soft-Gating ($lpha \in [0, 1]$)** |
| **LiDAR Dropout Resilience** | ❌ Severe drop | ❌ Diffusion collapses | N/A | **✅ Temporal Queue Memory Support** |
| **Inference Latency** | $0.47\text{ ms}$ (Raw) | $\sim 45\text{ ms}$ (Multi-step) | $\sim 28\text{ ms}$ (Local BA) | **$< 20.0\text{ ms}$ (Real-Time)** |
| **Computational Footprint** | $38.4\text{ M}$ parameters | $\sim 48.0\text{ M}$ parameters | Classical C++ (Zero NN) | **$32.8\text{ M}$ parameters (15% lighter)** |
| **Soft-Gating Overhead** | None | None | None | **1,666 Parameters ($<0.1\text{ ms}$)** |

---

## 🏗️ Model Architecture

```
                        +----------------------------+
                        | Camera Sweep (6-Cam RGB)   |
                        +--------------+-------------+
                                       |
                                [ResNet / ViT]
                                       |
+--------------------------+           v
| LiDAR Point Cloud Sweeps |---> [Voxel/Pillar] ---> [Cross-Attention Fusion] <---> [Spatio-Temporal FIFO Queue]
+--------------------------+           ^                         |
                                       |                         v
+--------------------------+           |             [Dynamic Soft-Gating]
| 200Hz IMU Stream         |---> [SO(3) Manifold]                | (Reweighting weights: w_cam, w_lidar)
+--------------------------+    [Pre-integration]                v
                                                    [Task Heads: BEV Map / Waypoints]
```

### Key Modules:
* **`staf/models/staf_model.py`**: End-to-end multi-modal architecture with feature backbones and prediction heads.
* **`staf/models/soft_gating.py`**: Adaptive soft-gating MLP computing continuous sensory confidence weights.
* **`staf/models/temporal_memory.py`**: Spatio-temporal transformer encoder with detached FIFO queue state.
* **`staf/imu/preintegration.py`**: Lie-group kinematic pre-integration on $SO(3)$ algebra.
* **`staf/losses/focal_loss.py`**: Class-balanced multi-class focal loss addressing extreme dataset imbalance.

---

## 📁 Repository Structure

```text
STAF/
├── configs/
│   └── default_config.py          # Master configuration (backbone, fusion, loss, training)
├── scripts/
│   ├── runpod_setup.sh            # Automated cloud container deployment script
│   └── train.py                   # Full end-to-end training loop with validation
├── staf/
│   ├── baselines/                 # Adapter interfaces for baseline benchmarks
│   │   ├── diffuser_adapter.py    # DifFUSER pipeline & logging adapter
│   │   ├── orbslam3_runner.py     # ORB-SLAM3 trajectory execution runner
│   │   └── transfuser_adapter.py  # TransFuser CARLA Longest6 evaluation adapter
│   ├── data/                      # Multi-modal data loaders & preprocessing
│   │   ├── cleaning.py            # Quality filters for camera/LiDAR/IMU
│   │   ├── dataset.py             # nuScenes & CARLA PyTorch datasets
│   │   └── sync.py                # Multi-rate timestamp synchronization
│   ├── imu/
│   │   └── preintegration.py      # On-manifold SO(3) IMU pre-integration
│   ├── losses/
│   │   └── focal_loss.py          # Class-balanced multi-class focal loss
│   ├── models/
│   │   ├── fusion_transformer.py  # Spatial cross-attention fusion layers
│   │   ├── soft_gating.py         # Dynamic soft-gating network (1,666 params)
│   │   ├── staf_model.py          # Master STAF neural network architecture
│   │   └── temporal_memory.py     # Spatio-temporal FIFO memory queue
│   └── utils/
├── tests/
│   ├── test_core.py               # Unit tests: SO(3), soft-gating, focal loss
│   └── test_pipeline.py           # Unit tests: Data cleaning, sync, E2E forward pass
├── requirements.txt               # Python package dependencies
├── LICENSE                        # MIT License
└── README.md                      # Comprehensive academic documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/mTaliKhan08/STAF.git
cd STAF
```

### 2. Environment Setup
We recommend Python 3.10+ in a dedicated virtual environment or Conda environment:
```bash
conda create -n staf python=3.10 -y
conda activate staf

# Install PyTorch with CUDA support (adjust cuda version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install STAF dependencies
pip install -r requirements.txt
```

### 3. Run Unit Tests & Verification
Verify that all core mathematical modules and pipeline components pass validation:
```bash
# Run Core Mathematical & Module Tests (9 tests)
python tests/test_core.py

# Run Data Pipeline & End-to-End Forward Pass Tests (11 tests)
python tests/test_pipeline.py
```

Expected output:
```text
Results: 9 passed, 0 failed out of 9 tests
Results: 11 passed, 0 failed out of 11 tests
E2E forward pass verified: Total params: 32,790,220 (~32.8M)
```

---

## 🏋️ Training & Evaluation

### Local / Single-GPU Training
```bash
python scripts/train.py --config configs/default_config.py --batch_size 4 --epochs 35
```

### Cloud Deployment (RunPod / AWS G5)
For automated setup on GPU instances (NVIDIA RTX 4090 / A10G):
```bash
chmod +x scripts/runpod_setup.sh
./scripts/runpod_setup.sh
```

---

## 📊 Baseline Benchmark Reproducibility

Detailed step-by-step logs, model checkpoints, and evaluation reports for all three baselines are cataloged under the thesis baseline suite:
* **TransFuser:** 36/36 CARLA Longest6 routes validated with 100% success rate and 0.47 ms inference latency.
* **DifFUSER:** 11,650 backpropagation iterations on AWS A10G achieving 73% multi-task loss reduction.
* **ORB-SLAM3:** Visual-Inertial tracking on EuRoC MAV (`MH_01_easy`) achieving ~0.036 m ATE RMSE.

---

## 📄 License & Attribution

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Citation
If you find this research or code useful in your academic work, please consider citing:
```bibtex
@mastersthesis{talikhan2026staf,
  author  = {M. Talikhan},
  title   = {Multi-Modal Fusion Framework for Autonomous Robot Navigation (STAF)},
  school  = {Department of Data Science},
  year    = {2026},
  month   = {October}
}
```
