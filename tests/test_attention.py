"""Tests for attention modules."""

import torch
import pytest
from src.attention import SpatialSelfAttention, ChannelAttention


class TestSpatialSelfAttention:
    """Tests for the SpatialSelfAttention module."""

    def test_output_shape(self):
        """Output should match input shape (B, C, H, W)."""
        module = SpatialSelfAttention(in_channels=64)
        x = torch.randn(2, 64, 8, 8)
        out, attn = module(x)
        assert out.shape == (2, 64, 8, 8)

    def test_attention_weights_shape(self):
        """Attention weights should be (B, num_heads, N, N) where N=H*W."""
        module = SpatialSelfAttention(in_channels=64, num_heads=2, head_dim=16)
        x = torch.randn(2, 64, 8, 8)
        _, attn = module(x)
        assert attn.shape == (2, 2, 64, 64)  # N=8*8=64

    def test_attention_weights_sum_to_one(self):
        """Each row of attention weights should sum to 1 (softmax output)."""
        module = SpatialSelfAttention(in_channels=32)
        x = torch.randn(1, 32, 4, 4)
        _, attn = module(x)
        row_sums = attn.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_different_spatial_sizes(self):
        """Should work with various spatial dimensions."""
        module = SpatialSelfAttention(in_channels=128)
        for size in [4, 8, 16]:
            x = torch.randn(1, 128, size, size)
            out, attn = module(x)
            assert out.shape == x.shape
            assert attn.shape == (1, 1, size * size, size * size)

    def test_multi_head(self):
        """Multi-head attention should produce correct shapes."""
        module = SpatialSelfAttention(in_channels=64, num_heads=4, head_dim=16)
        x = torch.randn(2, 64, 4, 4)
        out, attn = module(x)
        assert out.shape == (2, 64, 4, 4)
        assert attn.shape == (2, 4, 16, 16)  # 4 heads, N=4*4=16

    def test_gradient_flow(self):
        """Gradients should flow through the module."""
        module = SpatialSelfAttention(in_channels=32)
        x = torch.randn(1, 32, 4, 4, requires_grad=True)
        out, _ = module(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestChannelAttention:
    """Tests for the ChannelAttention (SE) module."""

    def test_output_shape(self):
        """Output should match input shape."""
        module = ChannelAttention(in_channels=64)
        x = torch.randn(2, 64, 8, 8)
        out, attn = module(x)
        assert out.shape == (2, 64, 8, 8)

    def test_no_attention_weights(self):
        """Channel attention returns None for attention weights."""
        module = ChannelAttention(in_channels=64)
        x = torch.randn(1, 64, 4, 4)
        _, attn = module(x)
        assert attn is None

    def test_scale_range(self):
        """SE scale factors should be in [0, 1] due to Sigmoid."""
        module = ChannelAttention(in_channels=32)
        x = torch.randn(2, 32, 4, 4)
        out, _ = module(x)
        # Output = input * scale, where scale in [0,1]
        # So |output| <= |input| element-wise
        assert (out.abs() <= x.abs() + 1e-6).all()

    def test_gradient_flow(self):
        """Gradients should flow through the module."""
        module = ChannelAttention(in_channels=64)
        x = torch.randn(1, 64, 4, 4, requires_grad=True)
        out, _ = module(x)
        out.sum().backward()
        assert x.grad is not None

    def test_min_bottleneck_size(self):
        """Bottleneck dimension should not go below 8."""
        # With in_channels=16 and reduction=16, raw mid = 1, but should be clamped to 8
        module = ChannelAttention(in_channels=16, reduction=16)
        mid_dim = module.fc[0].in_features  # first linear input
        out_dim = module.fc[0].out_features  # first linear output (the bottleneck)
        assert out_dim >= 8
