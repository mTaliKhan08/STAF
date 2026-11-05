"""
STAF: Spatio-Temporal Adaptive Fusion Model.

Main model class that integrates all STAF components:
1. ResNet34 visual backbone (camera)
2. ResNet18 spatial backbone (LiDAR BEV)
3. IMU-Guided Soft-Gating MLP (supports 3-d causal vector or 9-d pre-integrated IMU)
4. Gated Multi-Scale Cross-Attention Transformer
5. Spatio-Temporal Memory Queue
6. Constant-Curvature Residual Waypoint Prediction Head
"""

import math
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple

from staf.models.soft_gating import IMUGuidedSoftGating
from staf.models.fusion_transformer import GatedFusionTransformer
from staf.models.temporal_memory import SpatioTemporalMemoryQueue


def compute_cc_trajectory_torch(
    speed: torch.Tensor,
    body_yaw_rate: torch.Tensor,
    time_horizons: list = [0.5, 1.0, 1.5, 2.0],
) -> torch.Tensor:
    """
    Compute Constant Curvature future trajectory in ego body frame (+x Forward, +y Left).
    speed: (B, 1) or (B,) in m/s
    body_yaw_rate: (B, 1) or (B,) in rad/s (right-handed ISO body frame)
    Returns: (B, len(time_horizons), 2)
    """
    if speed.ndim == 1:
        speed = speed.unsqueeze(-1)
    if body_yaw_rate.ndim == 1:
        body_yaw_rate = body_yaw_rate.unsqueeze(-1)

    pts = []
    for t in time_horizons:
        psi = body_yaw_rate * t
        is_straight = torch.abs(body_yaw_rate) < 1e-4
        safe_w = torch.where(is_straight, torch.ones_like(body_yaw_rate) * 1e-4, body_yaw_rate)
        R = speed / safe_w
        x_curve = R * torch.sin(psi)
        y_curve = R * (1.0 - torch.cos(psi))
        x_straight = speed * t
        y_straight = torch.zeros_like(speed)
        x = torch.where(is_straight, x_straight, x_curve)
        y = torch.where(is_straight, y_straight, y_curve)
        pts.append(torch.stack([x, y], dim=-1))
    return torch.cat(pts, dim=1)


class VisualBackbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4
        self.channel_dims = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list:
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s2, s3, s4, s5]


class LiDARBackbone(nn.Module):
    def __init__(self, in_channels: int = 2, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                orig_weight = resnet.conv1.weight
                avg_weight = orig_weight.mean(dim=1, keepdim=True)
                self.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1)[:, :in_channels])
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4
        self.channel_dims = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list:
        s1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s2, s3, s4, s5]


class MultiScaleProjection(nn.Module):
    def __init__(self, channel_dims: list, embed_dim: int = 256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(8),
                nn.Conv2d(ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in channel_dims
        ])

    def forward(self, features: list) -> torch.Tensor:
        tokens = []
        for proj, feat in zip(self.projections, features):
            projected = proj(feat)
            B, D, H, W = projected.shape
            tokens.append(projected.flatten(2).transpose(1, 2))
        return torch.cat(tokens, dim=1)


class STAFModel(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_fusion_layers: int = 4,
        ff_dim: int = 512,
        queue_length: int = 5,
        num_temporal_layers: int = 2,
        lidar_in_channels: int = 2,
        dropout: float = 0.1,
        num_classes: int = 0,
        pred_len: int = 4,
        use_target_point: bool = True,
        use_ego_speed: bool = True,
        pretrained_backbones: bool = True,
        modality: str = "full",
        gate_input_dim: int = 9,
        gate_mode: str = "learned",
        use_residual_cc: bool = False,
        use_full_kinematics_head: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pred_len = pred_len
        self.use_target_point = use_target_point
        self.use_ego_speed = use_ego_speed
        self.modality = modality
        self.gate_input_dim = gate_input_dim
        self.gate_mode = gate_mode
        self.use_residual_cc = use_residual_cc
        self.use_full_kinematics_head = use_full_kinematics_head

        # Backbones
        if self.modality in ["full", "camera_only"]:
            self.visual_backbone = VisualBackbone(pretrained=pretrained_backbones)
            self.cam_projection = MultiScaleProjection(self.visual_backbone.channel_dims, embed_dim)
        else:
            self.visual_backbone = None
            self.cam_projection = None

        if self.modality in ["full", "lidar_only"]:
            self.lidar_backbone = LiDARBackbone(in_channels=lidar_in_channels, pretrained=pretrained_backbones)
            self.lidar_projection = MultiScaleProjection(self.lidar_backbone.channel_dims, embed_dim)
        else:
            self.lidar_backbone = None
            self.lidar_projection = None

        # Soft-Gating
        if self.modality == "full":
            self.soft_gating = IMUGuidedSoftGating(input_dim=gate_input_dim, hidden_dim=64, dropout=dropout, gate_mode=gate_mode)
            self.fusion_transformer = GatedFusionTransformer(embed_dim=embed_dim, num_heads=num_heads, num_layers=num_fusion_layers, ff_dim=ff_dim, dropout=dropout)
        else:
            self.soft_gating = None
            self.fusion_transformer = None

        if self.modality != "no_vision":
            self.spatial_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1))
            self.temporal_memory = SpatioTemporalMemoryQueue(embed_dim=embed_dim, queue_length=queue_length, num_heads=num_heads, num_layers=num_temporal_layers, ff_dim=ff_dim, dropout=dropout)
        else:
            self.spatial_pool = None
            self.temporal_memory = None

        if num_classes > 0 and self.modality != "no_vision":
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

        # Waypoint Head
        if pred_len > 0:
            if self.use_full_kinematics_head:
                # [z, normalized v, a, omega, goal (2d)]
                vis_dim = embed_dim if self.modality != "no_vision" else 0
                in_dim = vis_dim + 1 + 1 + 1 + 2
            else:
                vis_dim = embed_dim if self.modality != "no_vision" else 0
                in_dim = vis_dim + (2 if use_target_point else 0) + (1 if use_ego_speed else 0)

            self.waypoint_head = nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, pred_len * 2),
            )
            if self.use_residual_cc:
                nn.init.zeros_(self.waypoint_head[-1].weight)
                nn.init.zeros_(self.waypoint_head[-1].bias)
        else:
            self.waypoint_head = None

    def forward(
        self,
        rgb: Optional[torch.Tensor] = None,
        lidar_bev: Optional[torch.Tensor] = None,
        x_kinematic: Optional[torch.Tensor] = None,
        target_point: Optional[torch.Tensor] = None,
        ego_speed: Optional[torch.Tensor] = None,
        accel: Optional[torch.Tensor] = None,
        body_yaw_rate: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        output = {}
        temporally_smoothed = None
        w_cam, w_lidar = None, None

        if self.modality == "full":
            cam_features = self.visual_backbone(rgb)
            lidar_features = self.lidar_backbone(lidar_bev)
            cam_tokens = self.cam_projection(cam_features)
            lidar_tokens = self.lidar_projection(lidar_features)
            w_cam, w_lidar = self.soft_gating(x_kinematic)
            fused = self.fusion_transformer(cam_tokens, lidar_tokens, w_cam, w_lidar)
            pooled = self.spatial_pool(fused.transpose(1, 2)).squeeze(-1)
            temporally_smoothed = self.temporal_memory(pooled)
        elif self.modality == "camera_only":
            cam_features = self.visual_backbone(rgb)
            cam_tokens = self.cam_projection(cam_features)
            pooled = self.spatial_pool(cam_tokens.transpose(1, 2)).squeeze(-1)
            temporally_smoothed = self.temporal_memory(pooled)
        elif self.modality == "lidar_only":
            lidar_features = self.lidar_backbone(lidar_bev)
            lidar_tokens = self.lidar_projection(lidar_features)
            pooled = self.spatial_pool(lidar_tokens.transpose(1, 2)).squeeze(-1)
            temporally_smoothed = self.temporal_memory(pooled)

        output["fused_features"] = temporally_smoothed
        output["w_cam"] = w_cam
        output["w_lidar"] = w_lidar

        if self.classifier is not None and temporally_smoothed is not None:
            output["logits"] = self.classifier(temporally_smoothed)

        if self.waypoint_head is not None:
            # Batch size reference
            if temporally_smoothed is not None:
                bs = temporally_smoothed.size(0)
                device = temporally_smoothed.device
                dtype = temporally_smoothed.dtype
            elif ego_speed is not None:
                bs = ego_speed.size(0)
                device = ego_speed.device
                dtype = ego_speed.dtype
            else:
                bs = 1
                device = torch.device("cpu")
                dtype = torch.float32

            if ego_speed is None:
                ego_speed = torch.zeros((bs, 1), device=device, dtype=dtype)
            elif ego_speed.ndim == 1:
                ego_speed = ego_speed.unsqueeze(-1)

            if accel is None:
                accel = torch.zeros((bs, 1), device=device, dtype=dtype)
            elif accel.ndim == 1:
                accel = accel.unsqueeze(-1)

            if body_yaw_rate is None:
                body_yaw_rate = torch.zeros((bs, 1), device=device, dtype=dtype)
            elif body_yaw_rate.ndim == 1:
                body_yaw_rate = body_yaw_rate.unsqueeze(-1)

            if target_point is None:
                target_point = torch.zeros((bs, 2), device=device, dtype=dtype)

            features = []
            if temporally_smoothed is not None:
                features.append(temporally_smoothed)

            if self.use_full_kinematics_head:
                # [z, normalized v, a, omega, normalized goal]
                norm_v = ego_speed / 10.0
                norm_tgt = torch.clamp(target_point / 20.0, -3.0, 3.0)
                features.extend([norm_v, accel, body_yaw_rate, norm_tgt])
            else:
                if self.use_target_point:
                    features.append(target_point)
                if self.use_ego_speed:
                    features.append(ego_speed)

            wp_input = torch.cat(features, dim=-1)
            head_offsets = self.waypoint_head(wp_input).view(-1, self.pred_len, 2)

            if self.use_residual_cc:
                p_cc = compute_cc_trajectory_torch(ego_speed, body_yaw_rate)
                output["waypoints"] = p_cc + head_offsets
                output["cc_waypoints"] = p_cc
                output["delta_waypoints"] = head_offsets
            else:
                output["waypoints"] = head_offsets

        return output

    def reset_temporal_state(self, batch_size: int, device: torch.device):
        if self.temporal_memory is not None:
            self.temporal_memory.reset_queue(batch_size, device)

    def get_param_summary(self) -> Dict[str, int]:
        def count_params(module):
            if module is None:
                return 0
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            'visual_backbone': count_params(self.visual_backbone),
            'lidar_backbone': count_params(self.lidar_backbone),
            'cam_projection': count_params(self.cam_projection),
            'lidar_projection': count_params(self.lidar_projection),
            'soft_gating': count_params(self.soft_gating),
            'fusion_transformer': count_params(self.fusion_transformer),
            'temporal_memory': count_params(self.temporal_memory),
            'classifier': count_params(self.classifier) if self.classifier else 0,
            'total': count_params(self),
        }
