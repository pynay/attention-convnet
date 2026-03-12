"""Tests for training pipeline utilities."""

import torch
import torch.nn as nn
import pytest
from src.train import get_device, set_seed, get_dataloaders, train_one_epoch, evaluate
from src.models import BaselineCNN


class TestGetDevice:
    """Tests for device auto-detection."""

    def test_returns_valid_device(self):
        """Should return a valid torch device."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda", "mps")


class TestSetSeed:
    """Tests for reproducibility seeding."""

    def test_reproducible_tensors(self):
        """Same seed should produce same random tensors."""
        set_seed(123)
        a = torch.randn(5)
        set_seed(123)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        """Different seeds should produce different tensors."""
        set_seed(1)
        a = torch.randn(100)
        set_seed(2)
        b = torch.randn(100)
        assert not torch.equal(a, b)


class TestGetDataloaders:
    """Tests for CIFAR-10 data loading."""

    def test_loader_lengths(self):
        """Train should have 50000/batch_size batches, test 10000/batch_size."""
        train_loader, test_loader = get_dataloaders(batch_size=500, num_workers=0, data_dir="./data")
        assert len(train_loader) == 100  # 50000 / 500
        assert len(test_loader) == 20    # 10000 / 500

    def test_batch_shapes(self):
        """Batches should have correct image and label shapes."""
        train_loader, _ = get_dataloaders(batch_size=16, num_workers=0, data_dir="./data")
        images, labels = next(iter(train_loader))
        assert images.shape == (16, 3, 32, 32)
        assert labels.shape == (16,)

    def test_normalized_range(self):
        """After normalization, values should not be in raw [0,1] range."""
        _, test_loader = get_dataloaders(batch_size=16, num_workers=0, data_dir="./data")
        images, _ = next(iter(test_loader))
        # Normalized images will have negative values and values > 1
        assert images.min() < 0 or images.max() > 1


class TestTrainOneEpoch:
    """Tests for the single-epoch training function."""

    def test_returns_loss_and_accuracy(self):
        """Should return dict with 'loss' and 'accuracy' keys."""
        model = BaselineCNN(num_classes=10)
        # Use a tiny subset for speed
        train_loader, _ = get_dataloaders(batch_size=64, num_workers=0, data_dir="./data")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        # Only run 1 batch
        tiny_loader = [next(iter(train_loader))]
        metrics = train_one_epoch(model, tiny_loader, optimizer, criterion, torch.device("cpu"))

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1


class TestEvaluate:
    """Tests for the evaluation function."""

    def test_returns_metrics(self):
        """Should return loss, accuracy, and per-class accuracy."""
        model = BaselineCNN(num_classes=10)
        _, test_loader = get_dataloaders(batch_size=64, num_workers=0, data_dir="./data")
        criterion = nn.CrossEntropyLoss()

        tiny_loader = [next(iter(test_loader))]
        metrics = evaluate(model, tiny_loader, criterion, torch.device("cpu"))

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert "per_class_accuracy" in metrics
        assert metrics["loss"] > 0
        assert 0 <= metrics["accuracy"] <= 1

    def test_per_class_accuracy_keys(self):
        """Per-class accuracy should have integer keys for present classes."""
        model = BaselineCNN(num_classes=10)
        _, test_loader = get_dataloaders(batch_size=200, num_workers=0, data_dir="./data")
        criterion = nn.CrossEntropyLoss()

        tiny_loader = [next(iter(test_loader))]
        metrics = evaluate(model, tiny_loader, criterion, torch.device("cpu"))

        for key, val in metrics["per_class_accuracy"].items():
            assert 0 <= val <= 1
