"""
Spatio-Temporal Memory Queue for STAF Framework.

Implements a FIFO feature token buffer coupled with a temporal transformer
encoder to enforce temporal consistency and eliminate inter-frame semantic
flickering.

    M_temp = Concat(F_hat_{t-T}, ..., F_hat_t)

The temporal transformer processes the concatenated history to produce
temporally-smoothed feature representations.

Default queue length T=5 (tunable in ablation).
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Optional


class TemporalTransformerEncoder(nn.Module):
    """
    Lightweight temporal transformer encoder.

    Processes a sequence of feature tokens from consecutive frames
    to enforce temporal consistency.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 8,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            temporal_features: Stacked feature tokens from T frames,
                shape (B, T, D) where T is the queue length.

        Returns:
            Temporally-smoothed features, shape (B, D).
            Uses the most recent frame's output (last position).
        """
        B, T, D = temporal_features.shape

        # Add temporal position embeddings
        temporal_features = temporal_features + self.temporal_pos_embed[:, :T, :]

        # Temporal self-attention
        out = self.encoder(temporal_features)  # (B, T, D)
        out = self.output_norm(out)

        # Return the most recent frame's temporally-contextualized representation
        return out[:, -1, :]  # (B, D)


class SpatioTemporalMemoryQueue(nn.Module):
    """
    FIFO Memory Queue with Temporal Transformer.

    Maintains a sliding window of the most recent T fused feature
    representations and applies a temporal transformer to produce
    temporally-consistent outputs.

    This addresses the semantic flickering problem identified in
    Thesis-1 Gap #3: consecutive frames may produce inconsistent
    classifications without temporal context.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        queue_length: int = 5,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.queue_length = queue_length

        # Feature projection (in case fused features have different dim)
        self.feature_proj = nn.Linear(embed_dim, embed_dim)

        # Temporal transformer
        self.temporal_encoder = TemporalTransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=dropout,
            max_seq_len=queue_length,
        )

        # Internal queue (not a parameter, managed manually)
        self._queue = None  # Will be initialized on first forward pass

    def reset_queue(self, batch_size: int, device: torch.device):
        """Initialize or reset the memory queue with zeros."""
        self._queue = torch.zeros(
            batch_size, self.queue_length, self.embed_dim,
            device=device
        )
        self._fill_count = 0

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        Push new fused features into the queue and return temporally-smoothed output.

        Args:
            fused_features: Current frame's fused features, shape (B, D).
                These should be the spatially-pooled output of the
                GatedFusionTransformer.

        Returns:
            Temporally-smoothed features, shape (B, D).
        """
        B, D = fused_features.shape

        # Initialize queue if needed
        if self._queue is None or self._queue.shape[0] != B:
            self.reset_queue(B, fused_features.device)

        # Project features
        projected = self.feature_proj(fused_features)  # (B, D)

        # FIFO push: shift left and append new frame at the end
        self._queue = torch.cat([
            self._queue[:, 1:, :],           # Drop oldest
            projected.unsqueeze(1),            # Append newest
        ], dim=1)  # (B, T, D)

        self._fill_count = min(self._fill_count + 1, self.queue_length)

        # Apply temporal transformer
        # Only use filled positions to avoid attending to zero-padding
        if self._fill_count < self.queue_length:
            # Partial queue: use only filled entries
            active = self._queue[:, -self._fill_count:, :]
        else:
            active = self._queue

        temporally_smoothed = self.temporal_encoder(active)  # (B, D)

        return temporally_smoothed

    @torch.no_grad()
    def get_queue_state(self) -> Optional[torch.Tensor]:
        """Return the current queue state for debugging/visualization."""
        return self._queue.clone() if self._queue is not None else None
