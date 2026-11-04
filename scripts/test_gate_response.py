import sys
import os
import torch

sys.path.insert(0, r"E:\MS-DS\Thesis-1\Multi-Modal Fusion Framework for Autonomous Robot\STAF")
from staf.models.soft_gating import IMUGuidedSoftGating

gating = IMUGuidedSoftGating(input_dim=9, hidden_dim=64, dropout=0.0)
gating.eval()

# Test 1: Stopped input (all zeros)
x_stopped = torch.zeros(1, 9)
w_cam_stop, w_lidar_stop = gating(x_stopped)

# Test 2: Fast forward motion (v=10 m/s, a=2 m/s^2)
x_fast = torch.tensor([[5.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
w_cam_fast, w_lidar_fast = gating(x_fast)

# Test 3: Fast turning motion (omega = 0.5 rad/s)
x_turn = torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.5]], dtype=torch.float32)
w_cam_turn, w_lidar_turn = gating(x_turn)

print(f"Untrained Gating Head Response:")
print(f"Stopped: w_cam={w_cam_stop.item():.4f}, w_lidar={w_lidar_stop.item():.4f}")
print(f"Fast:    w_cam={w_cam_fast.item():.4f}, w_lidar={w_lidar_fast.item():.4f}")
print(f"Turn:    w_cam={w_cam_turn.item():.4f}, w_lidar={w_lidar_turn.item():.4f}")

