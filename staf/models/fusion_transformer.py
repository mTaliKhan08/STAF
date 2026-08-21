"""
Gated Multi-Scale Cross-Attention Transformer for STAF Framework.

Implements the core fusion mechanism where soft-gated visual and spatial
feature tokens are fused using cross-attention:

    f_fused = w_cam * Attn(Q_lidar, K_cam, V_cam) + w_lidar * Attn(Q_cam, K_lidar, V_lidar)

The gating weights (w_cam, w_lidar) are produced by the IMU-Guided
Soft-Gating MLP and applied BEFORE the cross-attention computation.

Architecture choices match TransFuser's baseline (ResNet34 + ResNet18 + GPT-style
transformer fusion) for fair ablation comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatedCrossAttention(nn.Module):
    """
    Single gated cross-attention layer.

    Computes cross-attention between two modalities where the key/value
    tokens are pre-scaled by the soft-gating weight.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        gate_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: Query tokens from modality A, shape (B, N_q, D).
            key_value: Key/Value tokens from modality B, shape (B, N_kv, D).
            gate_weight: Soft-gating weight for modality B, shape (B, 1).

        Returns:
            Fused output tokens, shape (B, N_q, D).
        """
        B, N_q, D = query.shape
        N_kv = key_value.shape[1]

        # Apply soft-gating: scale the key/value tokens
        key_value_gated = key_value * gate_weight.unsqueeze(1)  # (B, N_kv, D)

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key_value_gated)
        V = self.v_proj(key_value_gated)

        # Reshape for multi-head attention: (B, num_heads, N, head_dim)
        Q = Q.reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N_kv, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, N_q, N_kv)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = attn_weights @ V  # (B, H, N_q, head_dim)
        out = out.transpose(1, 2).reshape(B, N_q, D)
        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out


class GatedFusionBlock(nn.Module):
    """
    Bidirectional gated cross-attention fusion block.

    Performs cross-attention in both directions:
        cam_to_lidar: LiDAR queries attend to gated camera tokens
        lidar_to_cam: Camera queries attend to gated LiDAR tokens

    The fused representation combines both directions:
        f_fused = cam_cross + lidar_cross

    Followed by a feed-forward network with residual connections.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, ff_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        # Cross-attention: LiDAR queries, Camera keys/values
        self.cross_attn_cam_to_lidar = GatedCrossAttention(embed_dim, num_heads, dropout)
        # Cross-attention: Camera queries, LiDAR keys/values
        self.cross_attn_lidar_to_cam = GatedCrossAttention(embed_dim, num_heads, dropout)

        # Layer norms
        self.norm_cam = nn.LayerNorm(embed_dim)
        self.norm_lidar = nn.LayerNorm(embed_dim)
        self.norm_ff_cam = nn.LayerNorm(embed_dim)
        self.norm_ff_lidar = nn.LayerNorm(embed_dim)

        # Feed-forward networks
        self.ff_cam = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ff_lidar = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        f_cam: torch.Tensor,
        f_lidar: torch.Tensor,
        w_cam: torch.Tensor,
        w_lidar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f_cam: Camera feature tokens, shape (B, N_cam, D).
            f_lidar: LiDAR feature tokens, shape (B, N_lidar, D).
            w_cam: Camera reliability weight, shape (B, 1).
            w_lidar: LiDAR reliability weight, shape (B, 1).

        Returns:
            Updated (f_cam, f_lidar) after gated cross-attention fusion.
        """
        # Cross-attention with residual connections
        # Camera enriched by LiDAR
        f_cam_cross = self.cross_attn_lidar_to_cam(
            query=self.norm_cam(f_cam),
            key_value=f_lidar,
            gate_weight=w_lidar,
        )
        f_cam = f_cam + f_cam_cross

        # LiDAR enriched by Camera
        f_lidar_cross = self.cross_attn_cam_to_lidar(
            query=self.norm_lidar(f_lidar),
            key_value=f_cam,
            gate_weight=w_cam,
        )
        f_lidar = f_lidar + f_lidar_cross

        # Feed-forward with residual connections
        f_cam = f_cam + self.ff_cam(self.norm_ff_cam(f_cam))
        f_lidar = f_lidar + self.ff_lidar(self.norm_ff_lidar(f_lidar))

        return f_cam, f_lidar


class GatedFusionTransformer(nn.Module):
    """
    Multi-layer Gated Fusion Transformer.

    Stacks multiple GatedFusionBlocks to progressively fuse camera and
    LiDAR features under dynamic soft-gating control.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedFusionBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        f_cam: torch.Tensor,
        f_lidar: torch.Tensor,
        w_cam: torch.Tensor,
        w_lidar: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            f_cam: Camera features (B, N_cam, D).
            f_lidar: LiDAR features (B, N_lidar, D).
            w_cam: Camera gating weight (B, 1).
            w_lidar: LiDAR gating weight (B, 1).

        Returns:
            Fused feature representation (B, N_cam + N_lidar, D).
        """
        for layer in self.layers:
            f_cam, f_lidar = layer(f_cam, f_lidar, w_cam, w_lidar)

        # Concatenate and normalize
        fused = torch.cat([f_cam, f_lidar], dim=1)
        fused = self.final_norm(fused)
        return fused
