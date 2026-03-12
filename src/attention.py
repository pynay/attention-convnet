"""
Attention modules for augmenting convolutional networks.

Two modules are provided:
- SpatialSelfAttention: Multi-head self-attention over spatial positions (H*W).
  Uses 1x1 convolutions for Q/K/V projections, computes scaled dot-product
  attention, and adds a residual connection. Returns (output, attn_weights)
  so attention maps can be visualized without hooks.

- ChannelAttention: SE-Net style squeeze-and-excitation block for ablation
  comparison. Returns (output, None) for API consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    """Multi-head spatial self-attention via 1x1 convolutions.

    For an input of shape (B, C, H, W), flattens spatial dims to (B, C, H*W),
    computes multi-head scaled dot-product attention over the H*W positions,
    projects back, and adds a residual connection.

    Args:
        in_channels: Number of input channels.
        head_dim: Dimension per attention head. Total projection dim = head_dim * num_heads.
        num_heads: Number of attention heads.
        dropout: Dropout rate on attention weights.
    """

    def __init__(self, in_channels: int, head_dim: int = 32, num_heads: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = head_dim * num_heads

        # Linear projections for Q, K, V operating on channel dimension.
        # We use nn.Linear instead of 1x1 Conv2d to avoid MPS backward
        # compatibility issues with view/reshape on conv output strides.
        # Input is reshaped to (B, N, C) before projection.
        self.query = nn.Linear(in_channels, inner_dim, bias=False)
        self.key = nn.Linear(in_channels, inner_dim, bias=False)
        self.value = nn.Linear(in_channels, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, in_channels, bias=False)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_dim ** -0.5

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, C, H, W) — attention-augmented features with residual.
            attn_weights: (B, num_heads, H*W, H*W) — attention maps for visualization.
        """
        B, C, H, W = x.shape
        N = H * W
        nh, hd = self.num_heads, self.head_dim

        # Reshape to (B, N, C) for linear projections.
        # .contiguous() calls before every reshape ensure MPS backward compatibility,
        # since MPS autograd can produce non-contiguous gradient tensors after permute.
        x_flat = x.permute(0, 2, 3, 1).contiguous().reshape(B, N, C)

        # Project to Q, K, V and reshape to (B, nh, N, hd)
        q = self.query(x_flat).reshape(B, N, nh, hd).permute(0, 2, 1, 3).contiguous()
        k = self.key(x_flat).reshape(B, N, nh, hd).permute(0, 2, 1, 3).contiguous()
        v = self.value(x_flat).reshape(B, N, nh, hd).permute(0, 2, 1, 3).contiguous()

        # Scaled dot-product attention: (B, nh, N, hd) @ (B, nh, hd, N) -> (B, nh, N, N)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn_weights = attn  # save before dropout for visualization
        attn = self.dropout(attn)

        # Apply attention to values: (B, nh, N, N) @ (B, nh, N, hd) -> (B, nh, N, hd)
        out = torch.matmul(attn, v)
        # Reshape back: (B, nh, N, hd) -> (B, N, inner_dim) -> project -> (B, N, C)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, N, nh * hd)
        out = self.out_proj(out)

        # Post-norm residual
        out = x_flat + out
        out = self.norm(out)

        # Back to (B, C, H, W)
        out = out.permute(0, 2, 1).contiguous().reshape(B, C, H, W)

        return out, attn_weights


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention block.

    Global average pools spatial dims to get a channel descriptor, passes it
    through a bottleneck MLP (squeeze -> excite), and scales the original
    channels. Captures channel interdependencies without spatial attention.

    Included for ablation comparison against SpatialSelfAttention.

    Args:
        in_channels: Number of input channels.
        reduction: Bottleneck reduction ratio for the MLP hidden dim.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        mid = max(in_channels // reduction, 8)  # floor at 8 to avoid degenerate bottleneck
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, C, H, W) — channel-recalibrated features.
            attn_weights: None (no spatial attention maps to visualize).
        """
        B, C, _, _ = x.shape
        # Squeeze: global average pool -> (B, C)
        desc = x.mean(dim=(2, 3))
        # Excite: bottleneck MLP -> per-channel scale factors
        scale = self.fc(desc).view(B, C, 1, 1)
        return x * scale, None
