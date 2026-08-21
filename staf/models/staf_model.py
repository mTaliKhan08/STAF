"""
STAF: Spatio-Temporal Adaptive Fusion Model.

Main model class that integrates all STAF components:
1. ResNet34 visual backbone (camera)
2. ResNet18 spatial backbone (LiDAR BEV)
3. IMU-Guided Soft-Gating MLP
4. Gated Multi-Scale Cross-Attention Transformer
5. Spatio-Temporal Memory Queue

Architecture design follows TransFuser's backbone choices (Fix #7)
for fair ablation comparison, with the novel soft-gating and temporal
memory components added.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Optional, Tuple

from staf.models.soft_gating import IMUGuidedSoftGating
from staf.models.fusion_transformer import GatedFusionTransformer
from staf.models.temporal_memory import SpatioTemporalMemoryQueue


class VisualBackbone(nn.Module):
    """
    Camera feature extractor using ResNet34.

    Extracts multi-scale features from RGB images. Outputs features
    at 4 scales matching TransFuser's multi-scale fusion design.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # Split into stages for multi-scale feature extraction
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.stage2 = resnet.layer1  # 64 channels
        self.stage3 = resnet.layer2  # 128 channels
        self.stage4 = resnet.layer3  # 256 channels
        self.stage5 = resnet.layer4  # 512 channels

        self.channel_dims = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: RGB image tensor, shape (B, 3, H, W).

        Returns:
            List of multi-scale features [s2, s3, s4, s5].
        """
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s2, s3, s4, s5]


class LiDARBackbone(nn.Module):
    """
    LiDAR BEV feature extractor using ResNet18.

    Processes BEV-projected LiDAR point clouds. Extracts features at
    4 scales matching the visual backbone's resolution hierarchy.
    """

    def __init__(self, in_channels: int = 2, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Modify first conv for BEV input (2 channels: height + intensity)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # If pretrained, average the original 3-channel weights for initialization
        if pretrained:
            with torch.no_grad():
                orig_weight = resnet.conv1.weight  # (64, 3, 7, 7)
                avg_weight = orig_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                self.conv1.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1)[:, :in_channels])

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.stage2 = resnet.layer1  # 64 channels
        self.stage3 = resnet.layer2  # 128 channels
        self.stage4 = resnet.layer3  # 256 channels
        self.stage5 = resnet.layer4  # 512 channels

        self.channel_dims = [64, 128, 256, 512]

    def forward(self, x: torch.Tensor) -> list:
        """
        Args:
            x: BEV LiDAR tensor, shape (B, C_in, H, W).

        Returns:
            List of multi-scale features [s2, s3, s4, s5].
        """
        s1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        s5 = self.stage5(s4)
        return [s2, s3, s4, s5]


class MultiScaleProjection(nn.Module):
    """
    Projects multi-scale features to a common embedding dimension
    and flattens spatial dimensions into token sequences.
    """

    def __init__(self, channel_dims: list, embed_dim: int = 256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(8),  # Uniform spatial size
                nn.Conv2d(ch, embed_dim, 1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            )
            for ch in channel_dims
        ])

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: List of feature maps at different scales.

        Returns:
            Token sequence, shape (B, N_tokens, D).
        """
        tokens = []
        for proj, feat in zip(self.projections, features):
            projected = proj(feat)  # (B, D, H', W')
            B, D, H, W = projected.shape
            tokens.append(projected.flatten(2).transpose(1, 2))  # (B, H'*W', D)
        return torch.cat(tokens, dim=1)  # (B, sum(H'*W'), D)


class STAFModel(nn.Module):
    """
    Spatio-Temporal Adaptive Fusion (STAF) Model.

    Complete end-to-end architecture integrating:
    - Dual-backbone feature extraction (ResNet34 + ResNet18)
    - IMU-guided learned soft-gating
    - Gated multi-scale cross-attention transformer
    - Spatio-temporal memory queue

    Input: RGB image, LiDAR BEV, IMU kinematic vector
    Output: Fused perception features for downstream tasks
    """

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
        pretrained_backbones: bool = True,
    ):
        super().__init__()

        # === Backbones (Phase 3.1) ===
        self.visual_backbone = VisualBackbone(pretrained=pretrained_backbones)
        self.lidar_backbone = LiDARBackbone(
            in_channels=lidar_in_channels,
            pretrained=pretrained_backbones,
        )

        # === Multi-scale feature projection ===
        self.cam_projection = MultiScaleProjection(
            self.visual_backbone.channel_dims, embed_dim
        )
        self.lidar_projection = MultiScaleProjection(
            self.lidar_backbone.channel_dims, embed_dim
        )

        # === IMU-Guided Soft-Gating (Phase 3.2) ===
        self.soft_gating = IMUGuidedSoftGating(
            input_dim=9,
            hidden_dim=64,
            dropout=dropout,
        )

        # === Gated Fusion Transformer (Phase 3.3) ===
        self.fusion_transformer = GatedFusionTransformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_fusion_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # === Spatial pooling for temporal memory input ===
        self.spatial_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
        )

        # === Spatio-Temporal Memory Queue (Phase 3.4) ===
        self.temporal_memory = SpatioTemporalMemoryQueue(
            embed_dim=embed_dim,
            queue_length=queue_length,
            num_heads=num_heads,
            num_layers=num_temporal_layers,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        # === Optional classification head ===
        if num_classes > 0:
            self.classifier = nn.Linear(embed_dim, num_classes)
        else:
            self.classifier = None

    def forward(
        self,
        rgb: torch.Tensor,
        lidar_bev: torch.Tensor,
        x_kinematic: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Full STAF forward pass.

        Args:
            rgb: RGB camera image, shape (B, 3, H, W).
            lidar_bev: BEV-projected LiDAR, shape (B, C_lidar, H', W').
            x_kinematic: Pre-integrated kinematic vector, shape (B, 9).

        Returns:
            Dictionary with:
                'fused_features': Temporally-smoothed fused representation (B, D).
                'w_cam': Camera gating weight (B, 1).
                'w_lidar': LiDAR gating weight (B, 1).
                'logits': Class logits if num_classes > 0 (B, C).
        """
        # 1. Multi-scale feature extraction
        cam_features = self.visual_backbone(rgb)        # List of 4 feature maps
        lidar_features = self.lidar_backbone(lidar_bev)  # List of 4 feature maps

        # 2. Project to token sequences
        cam_tokens = self.cam_projection(cam_features)    # (B, N_cam, D)
        lidar_tokens = self.lidar_projection(lidar_features)  # (B, N_lidar, D)

        # 3. IMU-guided soft-gating
        w_cam, w_lidar = self.soft_gating(x_kinematic)  # (B, 1), (B, 1)

        # 4. Gated cross-attention fusion
        fused = self.fusion_transformer(
            cam_tokens, lidar_tokens, w_cam, w_lidar
        )  # (B, N_total, D)

        # 5. Spatial pooling -> (B, D)
        pooled = fused.transpose(1, 2)  # (B, D, N)
        pooled = self.spatial_pool(pooled).squeeze(-1)  # (B, D)

        # 6. Temporal memory smoothing
        temporally_smoothed = self.temporal_memory(pooled)  # (B, D)

        # Build output
        output = {
            'fused_features': temporally_smoothed,
            'w_cam': w_cam,
            'w_lidar': w_lidar,
        }

        if self.classifier is not None:
            output['logits'] = self.classifier(temporally_smoothed)

        return output

    def get_param_summary(self) -> Dict[str, int]:
        """Return parameter counts for each component."""
        def count_params(module):
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

    def reset_temporal_state(self, batch_size: int, device: torch.device):
        """Reset the temporal memory queue (call at sequence boundaries)."""
        self.temporal_memory.reset_queue(batch_size, device)
