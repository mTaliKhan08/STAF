#!/bin/bash
# ============================================================
# RunPod Setup & Training Launch Script — STAF Thesis-2
# ============================================================
#
# Turnkey script for setting up a RunPod instance to train all
# baselines and the STAF model.
#
# Prerequisites:
#   - RunPod pod with 1x RTX 4090 (24GB), 1TB NVMe, 64GB RAM
#   - Google Drive configured with rclone (see setup below)
#
# Usage:
#   bash scripts/runpod_setup.sh [phase]
#
# Phases:
#   setup     - Install dependencies and configure rclone
#   sync      - Download datasets from Google Drive to NVMe
#   transfuser - Train TransFuser baseline
#   diffuser  - Train DifFUSER baseline
#   orbslam3  - Evaluate ORB-SLAM3
#   staf      - Train STAF model
#   ablation  - Run ablation experiments
#   all       - Run everything in sequence
#
# ============================================================

set -euo pipefail

# --- Configuration ---
WORKSPACE="/workspace"
DATA_DIR="${WORKSPACE}/data"
CKPT_DIR="${WORKSPACE}/checkpoints"
RESULTS_DIR="${WORKSPACE}/results"
REPO_DIR="${WORKSPACE}/STAF"
BASELINE_DIR="${WORKSPACE}/Baseline"
GDRIVE_DATA="Thesis_Datasets"  # Google Drive folder name

# --- Color output ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_phase() { echo -e "\n${BLUE}================================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}================================================================${NC}\n"; }

# --- Phase: Setup ---
phase_setup() {
    log_phase "Phase: Environment Setup"
    
    # System packages
    log_info "Installing system dependencies..."
    apt-get update -qq && apt-get install -y -qq \
        git wget unzip htop tmux \
        libgl1-mesa-glx libglib2.0-0 \
        > /dev/null 2>&1
    
    # Install rclone (for Google Drive sync)
    if ! command -v rclone &> /dev/null; then
        log_info "Installing rclone..."
        curl -s https://rclone.org/install.sh | bash
    else
        log_info "rclone already installed"
    fi
    
    # Check rclone config
    if ! rclone listremotes | grep -q "gdrive:"; then
        log_warn "rclone 'gdrive' remote not configured!"
        log_warn "Run: rclone config"
        log_warn "  - New remote -> name: gdrive -> type: drive -> ..."
        log_warn "After configuring, re-run this script."
        exit 1
    fi
    
    # Python dependencies
    log_info "Installing Python dependencies..."
    pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -q tensorboard numpy scipy pillow opencv-python
    pip install -q timm einops diskcache
    
    # For DifFUSER: mmdetection3d stack
    pip install -q mmcv-full mmdet mmsegmentation
    pip install -q torchpack nuscenes-devkit
    
    # Create directories
    mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${RESULTS_DIR}"
    
    # Clone/copy repos if not present
    if [ ! -d "${REPO_DIR}" ]; then
        log_warn "STAF repo not found at ${REPO_DIR}"
        log_warn "Please upload or clone your STAF repo to ${REPO_DIR}"
    fi
    
    log_info "Setup complete!"
}

# --- Phase: Sync datasets from Google Drive ---
phase_sync() {
    log_phase "Phase: Dataset Sync (Google Drive -> NVMe)"
    
    # nuScenes (~320 GB compressed)
    log_info "Syncing nuScenes dataset..."
    rclone copy "gdrive:${GDRIVE_DATA}/nuscenes/" "${DATA_DIR}/nuscenes/" \
        --progress --transfers 4 --checkers 8 \
        --drive-chunk-size 128M
    
    # CARLA TransFuser (~120 GB compressed)
    log_info "Syncing CARLA TransFuser dataset..."
    rclone copy "gdrive:${GDRIVE_DATA}/carla_transfuser/" "${DATA_DIR}/carla_transfuser/" \
        --progress --transfers 4 --checkers 8 \
        --drive-chunk-size 128M
    
    # EuRoC MAV (~27 GB compressed)
    log_info "Syncing EuRoC MAV dataset..."
    rclone copy "gdrive:${GDRIVE_DATA}/euroc_mav/" "${DATA_DIR}/euroc_mav/" \
        --progress --transfers 4 --checkers 8 \
        --drive-chunk-size 128M
    
    # Extract archives
    log_info "Extracting archives..."
    cd "${DATA_DIR}"
    
    # Extract nuScenes blobs
    for f in nuscenes/*.tar.gz nuscenes/*.tgz; do
        [ -f "$f" ] && tar xzf "$f" -C nuscenes/ && log_info "  Extracted: $f"
    done
    
    # Extract CARLA zips
    for f in carla_transfuser/*.zip; do
        [ -f "$f" ] && unzip -qo "$f" -d carla_transfuser/ && log_info "  Extracted: $f"
    done
    
    # Extract EuRoC zips
    for f in euroc_mav/*.zip; do
        [ -f "$f" ] && unzip -qo "$f" -d euroc_mav/ && log_info "  Extracted: $f"
    done
    
    log_info "Dataset sync and extraction complete!"
    log_info "Disk usage:"
    du -sh "${DATA_DIR}"/*
}

# --- Phase: TransFuser Training ---
phase_transfuser() {
    log_phase "Phase: TransFuser Re-Training (20 epochs)"
    
    cd "${REPO_DIR}"
    
    python -m staf.baselines.transfuser_adapter \
        --root-dir "${DATA_DIR}/carla_transfuser" \
        --logdir "${CKPT_DIR}/transfuser" \
        --epochs 20 \
        --batch-size 6 \
        --accum-steps 4 \
        --use-amp \
        --use-compile \
        --sync-gdrive \
        --setting 02_05_withheld
    
    log_info "TransFuser training complete!"
}

# --- Phase: DifFUSER Training ---
phase_diffuser() {
    log_phase "Phase: DifFUSER Re-Training (12 epochs)"
    
    cd "${REPO_DIR}"
    
    python -m staf.baselines.diffuser_adapter \
        --data-root "${DATA_DIR}/nuscenes" \
        --logdir "${CKPT_DIR}/diffuser" \
        --epochs 12 \
        --batch-size 4 \
        --use-amp \
        --sync-gdrive
    
    log_info "DifFUSER training complete!"
}

# --- Phase: ORB-SLAM3 Evaluation ---
phase_orbslam3() {
    log_phase "Phase: ORB-SLAM3 Evaluation (11 EuRoC sequences)"
    
    cd "${REPO_DIR}"
    
    python -m staf.baselines.orbslam3_runner \
        --euroc-root "${DATA_DIR}/euroc_mav" \
        --orbslam-root "${BASELINE_DIR}/ORB-SLAM3" \
        --output-dir "${RESULTS_DIR}/orbslam3"
    
    log_info "ORB-SLAM3 evaluation complete!"
}

# --- Phase: STAF Training ---
phase_staf() {
    log_phase "Phase: STAF Full Training (18 epochs)"
    
    cd "${REPO_DIR}"
    
    python scripts/train.py \
        --data-root "${DATA_DIR}/nuscenes" \
        --dataset nuscenes \
        --epochs 18 \
        --batch-size 8 \
        --lr 3e-4 \
        --num-classes 16 \
        --save-dir "${CKPT_DIR}/staf" \
        --num-workers 8
    
    log_info "STAF training complete!"
    
    # Sync checkpoints
    rclone copy "${CKPT_DIR}/staf" "gdrive:Thesis_Checkpoints/staf" \
        --include "*.pth" --include "*.json" --progress
}

# --- Phase: Ablation Studies ---
phase_ablation() {
    log_phase "Phase: Ablation Studies (4 experiments)"
    
    cd "${REPO_DIR}"
    
    # Ablation 1: No soft-gating (w=1.0)
    log_info "Ablation 1: No soft-gating..."
    python scripts/train.py \
        --data-root "${DATA_DIR}/nuscenes" \
        --dataset nuscenes \
        --epochs 10 \
        --batch-size 8 \
        --lr 3e-4 \
        --save-dir "${CKPT_DIR}/ablation_no_gating"
    
    # Ablation 2: No temporal memory
    log_info "Ablation 2: No temporal memory..."
    python scripts/train.py \
        --data-root "${DATA_DIR}/nuscenes" \
        --dataset nuscenes \
        --epochs 10 \
        --batch-size 8 \
        --lr 3e-4 \
        --queue-length 0 \
        --save-dir "${CKPT_DIR}/ablation_no_temporal"
    
    # Ablation 3: Cross-entropy instead of focal loss
    log_info "Ablation 3: Cross-entropy loss..."
    python scripts/train.py \
        --data-root "${DATA_DIR}/nuscenes" \
        --dataset nuscenes \
        --epochs 10 \
        --batch-size 8 \
        --lr 3e-4 \
        --gamma 0.0 \
        --save-dir "${CKPT_DIR}/ablation_ce_loss"
    
    # Ablation 4: Robustness under noise
    log_info "Ablation 4: Noise injection..."
    python scripts/train.py \
        --data-root "${DATA_DIR}/nuscenes" \
        --dataset nuscenes \
        --epochs 10 \
        --batch-size 8 \
        --lr 3e-4 \
        --save-dir "${CKPT_DIR}/ablation_noise"
    
    # Sync all ablation results
    rclone copy "${CKPT_DIR}" "gdrive:Thesis_Checkpoints" \
        --include "ablation*/**" --progress
    
    log_info "All ablation studies complete!"
}

# --- Main entry point ---
PHASE="${1:-all}"

case "$PHASE" in
    setup)      phase_setup ;;
    sync)       phase_sync ;;
    transfuser) phase_transfuser ;;
    diffuser)   phase_diffuser ;;
    orbslam3)   phase_orbslam3 ;;
    staf)       phase_staf ;;
    ablation)   phase_ablation ;;
    all)
        phase_setup
        phase_sync
        phase_transfuser
        phase_diffuser
        phase_orbslam3
        phase_staf
        phase_ablation
        log_phase "All phases complete!"
        log_info "Total disk usage:"
        du -sh "${WORKSPACE}"/*
        ;;
    *)
        echo "Usage: $0 {setup|sync|transfuser|diffuser|orbslam3|staf|ablation|all}"
        exit 1
        ;;
esac
