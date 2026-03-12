"""
CNN architectures: baseline and attention-augmented variants.

- ConvBlock: Reusable building block (2x Conv3x3-BN-ReLU + optional MaxPool).
- BaselineCNN: 4 ConvBlocks + classifier head (~2.5M params).
- AttentionCNN: Same backbone with configurable attention insertion points.
  A single `attention_positions` list controls which blocks get attention,
  enabling all ablation variants without separate model classes.
- count_parameters / get_model: Utilities for param counting and config-based construction.
"""

import torch
import torch.nn as nn

from .attention import SpatialSelfAttention, ChannelAttention


class ConvBlock(nn.Module):
    """Two consecutive Conv3x3-BN-ReLU layers with optional MaxPool.

    Using two 3x3 convs per block (VGG-style) gives an effective 5x5 receptive
    field with fewer parameters than a single 5x5 conv, plus an extra nonlinearity.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        pool: Whether to append 2x2 max pooling.
    """

    def __init__(self, in_channels: int, out_channels: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BaselineCNN(nn.Module):
    """Standard CNN baseline: 4 ConvBlocks + classifier head.

    Architecture:
        Input (3, 32, 32)
        -> ConvBlock0: 3->64,   pool -> (64, 16, 16)
        -> ConvBlock1: 64->128, pool -> (128, 8, 8)
        -> ConvBlock2: 128->256, pool -> (256, 4, 4)
        -> ConvBlock3: 256->256, pool -> (256, 2, 2)
        -> AdaptiveAvgPool2d(1) -> FC(256->128) -> ReLU -> Dropout -> FC(128->num_classes)

    Args:
        num_classes: Number of output classes.
        dropout: Dropout rate in the classifier head.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x, {}


class AttentionCNN(nn.Module):
    """Attention-augmented CNN with configurable attention placement.

    Same backbone as BaselineCNN, but inserts attention modules after specified
    ConvBlocks. The `attention_positions` list controls which blocks get attention
    (e.g. [1, 2] inserts attention after blocks 1 and 2). This single parameter
    enables all ablation variants without needing separate model classes.

    Args:
        num_classes: Number of output classes.
        dropout: Dropout rate in the classifier head.
        attention_positions: List of block indices (0-3) after which to insert attention.
        attention_type: 'spatial' for SpatialSelfAttention, 'channel' for ChannelAttention.
        head_dim: Dimension per attention head (spatial attention only).
        num_heads: Number of attention heads (spatial attention only).
    """

    def __init__(
        self,
        num_classes: int = 10,
        dropout: float = 0.5,
        attention_positions: list[int] | None = None,
        attention_type: str = "spatial",
        head_dim: int = 32,
        num_heads: int = 1,
    ):
        super().__init__()
        if attention_positions is None:
            attention_positions = [1, 2]

        # Channel dims after each ConvBlock
        channels = [64, 128, 256, 256]

        self.blocks = nn.ModuleList([
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
        ])

        # Build attention modules only for requested positions.
        # Using nn.ModuleDict with string keys for proper registration.
        self.attention_modules = nn.ModuleDict()
        for pos in attention_positions:
            c = channels[pos]
            if attention_type == "spatial":
                self.attention_modules[str(pos)] = SpatialSelfAttention(
                    c, head_dim=head_dim, num_heads=num_heads
                )
            elif attention_type == "channel":
                self.attention_modules[str(pos)] = ChannelAttention(c)
            else:
                raise ValueError(f"Unknown attention_type: {attention_type}")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor):
        attn_maps = {}
        for i, block in enumerate(self.blocks):
            x = block(x)
            if str(i) in self.attention_modules:
                x, attn = self.attention_modules[str(i)](x)
                attn_maps[f"block_{i}"] = attn

        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x, attn_maps


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(config: dict) -> nn.Module:
    """Factory function to create a model from a config dict.

    Config keys:
        model_type: 'baseline' | 'attention'
        num_classes: int (default 10)
        dropout: float (default 0.5)
        attention_positions: list[int] (default [1, 2], attention model only)
        attention_type: str (default 'spatial', attention model only)
        head_dim: int (default 32)
        num_heads: int (default 1)
    """
    model_type = config.get("model_type", "baseline")
    num_classes = config.get("num_classes", 10)
    dropout = config.get("dropout", 0.5)

    if model_type == "baseline":
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    elif model_type == "attention":
        return AttentionCNN(
            num_classes=num_classes,
            dropout=dropout,
            attention_positions=config.get("attention_positions", [1, 2]),
            attention_type=config.get("attention_type", "spatial"),
            head_dim=config.get("head_dim", 32),
            num_heads=config.get("num_heads", 1),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
