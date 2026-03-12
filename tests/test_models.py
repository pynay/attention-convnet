"""Tests for CNN model architectures."""

import torch
import pytest
from src.models import ConvBlock, BaselineCNN, AttentionCNN, count_parameters, get_model


class TestConvBlock:
    """Tests for the ConvBlock building block."""

    def test_output_shape_with_pool(self):
        """With pooling, spatial dims should halve."""
        block = ConvBlock(3, 64, pool=True)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 16, 16)

    def test_output_shape_without_pool(self):
        """Without pooling, spatial dims should be preserved."""
        block = ConvBlock(64, 128, pool=False)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        assert out.shape == (2, 128, 8, 8)

    def test_channel_expansion(self):
        """Output channels should match out_channels arg."""
        block = ConvBlock(32, 256, pool=True)
        x = torch.randn(1, 32, 16, 16)
        out = block(x)
        assert out.shape[1] == 256


class TestBaselineCNN:
    """Tests for the BaselineCNN model."""

    def test_output_shape(self):
        """Should produce (B, num_classes) logits."""
        model = BaselineCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        logits, attn_maps = model(x)
        assert logits.shape == (4, 10)

    def test_empty_attention_maps(self):
        """Baseline should return empty attention maps dict."""
        model = BaselineCNN()
        x = torch.randn(1, 3, 32, 32)
        _, attn_maps = model(x)
        assert attn_maps == {}

    def test_custom_num_classes(self):
        """Should support different numbers of output classes."""
        model = BaselineCNN(num_classes=100)
        x = torch.randn(1, 3, 32, 32)
        logits, _ = model(x)
        assert logits.shape == (1, 100)

    def test_gradient_flow(self):
        """Gradients should flow from loss to input."""
        model = BaselineCNN()
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        logits, _ = model(x)
        logits.sum().backward()
        assert x.grad is not None


class TestAttentionCNN:
    """Tests for the AttentionCNN model."""

    def test_output_shape(self):
        """Should produce same output shape as baseline."""
        model = AttentionCNN(num_classes=10, attention_positions=[1, 2])
        x = torch.randn(4, 3, 32, 32)
        logits, _ = model(x)
        assert logits.shape == (4, 10)

    def test_attention_maps_returned(self):
        """Should return attention maps for configured positions."""
        model = AttentionCNN(attention_positions=[1, 2])
        x = torch.randn(2, 3, 32, 32)
        _, attn_maps = model(x)
        assert "block_1" in attn_maps
        assert "block_2" in attn_maps
        assert len(attn_maps) == 2

    def test_single_attention_position(self):
        """Should work with attention at only one position."""
        model = AttentionCNN(attention_positions=[2])
        x = torch.randn(2, 3, 32, 32)
        _, attn_maps = model(x)
        assert "block_2" in attn_maps
        assert len(attn_maps) == 1

    def test_all_attention_positions(self):
        """Should work with attention at positions 0, 1, 2."""
        model = AttentionCNN(attention_positions=[0, 1, 2])
        x = torch.randn(1, 3, 32, 32)
        _, attn_maps = model(x)
        assert len(attn_maps) == 3

    def test_channel_attention_type(self):
        """Channel attention should return None for attention weights."""
        model = AttentionCNN(attention_type="channel", attention_positions=[1])
        x = torch.randn(2, 3, 32, 32)
        logits, attn_maps = model(x)
        assert logits.shape == (2, 10)
        assert attn_maps["block_1"] is None

    def test_gradient_flow(self):
        """Gradients should flow through attention modules."""
        model = AttentionCNN(attention_positions=[1, 2])
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        logits, _ = model(x)
        logits.sum().backward()
        assert x.grad is not None

    def test_invalid_attention_type_raises(self):
        """Unknown attention type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown attention_type"):
            AttentionCNN(attention_type="invalid")


class TestCountParameters:
    """Tests for parameter counting utility."""

    def test_baseline_param_count(self):
        """Baseline should have ~2.36M params."""
        model = BaselineCNN()
        params = count_parameters(model)
        assert 2_000_000 < params < 3_000_000

    def test_attention_has_more_params(self):
        """Attention model should have more params than baseline."""
        baseline = BaselineCNN()
        attention = AttentionCNN(attention_positions=[1, 2])
        assert count_parameters(attention) > count_parameters(baseline)

    def test_more_attention_more_params(self):
        """More attention positions should mean more parameters."""
        attn_1 = AttentionCNN(attention_positions=[1])
        attn_12 = AttentionCNN(attention_positions=[1, 2])
        assert count_parameters(attn_12) > count_parameters(attn_1)


class TestGetModel:
    """Tests for the model factory function."""

    def test_baseline_factory(self):
        """Should create BaselineCNN from config."""
        model = get_model({"model_type": "baseline"})
        assert isinstance(model, BaselineCNN)

    def test_attention_factory(self):
        """Should create AttentionCNN from config."""
        model = get_model({"model_type": "attention", "attention_positions": [1]})
        assert isinstance(model, AttentionCNN)

    def test_invalid_model_type_raises(self):
        """Unknown model type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_model({"model_type": "transformer"})

    def test_factory_respects_num_classes(self):
        """Factory should pass through num_classes."""
        model = get_model({"model_type": "baseline", "num_classes": 100})
        x = torch.randn(1, 3, 32, 32)
        logits, _ = model(x)
        assert logits.shape == (1, 100)
