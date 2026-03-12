"""
Training pipeline for CIFAR-10 image classification.

Provides data loading with standard augmentation, train/eval loops, and a
full training pipeline with CosineAnnealingLR scheduling. Supports MPS
(Apple Silicon), CUDA, and CPU backends.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .models import get_model, count_parameters

# CIFAR-10 channel-wise mean and std, precomputed from the training set.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = 42):
    """Set seeds for reproducibility across all RNG sources."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Auto-detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_dataloaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "./data",
) -> tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 train and test data loaders.

    Training augmentation: RandomCrop(32, pad=4) + RandomHorizontalFlip.
    These are standard CIFAR-10 augmentations that provide regularization
    without being overly aggressive for 32x32 images.

    Both splits are normalized with precomputed CIFAR-10 channel statistics.
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # pin_memory speeds up CPU->GPU transfers on CUDA but is unsupported on MPS
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
    )
    return train_loader, test_loader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Run one training epoch. Returns dict with 'loss' and 'accuracy'."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset. Returns dict with 'loss' and 'accuracy'.

    Also returns per-class accuracy for detailed analysis.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        for label, pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            class_total[label] = class_total.get(label, 0) + 1
            if label == pred:
                class_correct[label] = class_correct.get(label, 0) + 1

    per_class_acc = {
        k: class_correct.get(k, 0) / class_total[k]
        for k in sorted(class_total.keys())
    }

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "per_class_accuracy": per_class_acc,
    }


def train(config: dict) -> dict:
    """Full training pipeline.

    Args:
        config: Dict with keys:
            model_type, num_classes, dropout, attention_positions, attention_type,
            head_dim, num_heads — forwarded to get_model()
            epochs (default 50), batch_size (default 128), lr (default 1e-3),
            weight_decay (default 1e-4), num_workers (default 2),
            data_dir (default './data'), save_dir (default './results'),
            seed (default 42), run_name (default auto-generated)

    Returns:
        Dict with training history and final test metrics.
    """
    # Extract training hyperparameters with defaults
    epochs = config.get("epochs", 50)
    batch_size = config.get("batch_size", 128)
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-4)
    num_workers = config.get("num_workers", 2)
    data_dir = config.get("data_dir", "./data")
    save_dir = config.get("save_dir", "./results")
    seed = config.get("seed", 42)

    # Auto-generate run name from config if not provided
    run_name = config.get("run_name")
    if run_name is None:
        model_type = config.get("model_type", "baseline")
        if model_type == "attention":
            positions = config.get("attention_positions", [1, 2])
            attn_type = config.get("attention_type", "spatial")
            run_name = f"{attn_type}_attn_pos{''.join(map(str, positions))}"
        else:
            run_name = "baseline"

    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Data
    train_loader, test_loader = get_dataloaders(batch_size, num_workers, data_dir)

    # Model
    model = get_model(config).to(device)
    num_params = count_parameters(model)
    print(f"Model: {run_name} | Parameters: {num_params:,}")

    # Optimizer and scheduler
    # Adam chosen over SGD for faster convergence on small datasets.
    # CosineAnnealingLR smoothly decays LR to near zero, avoiding the need
    # to hand-tune a step schedule.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_acc"].append(test_metrics["accuracy"])
        history["lr"].append(optimizer.param_groups[0]["lr"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
            f"Test Loss: {test_metrics['loss']:.4f} Acc: {test_metrics['accuracy']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best checkpoint
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            save_path = Path(save_dir) / f"{run_name}_best.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)

    # Final evaluation
    best_path = Path(save_dir) / f"{run_name}_best.pt"
    model.load_state_dict(torch.load(best_path, weights_only=True))
    final_metrics = evaluate(model, test_loader, criterion, device)
    print(f"\nBest test accuracy: {final_metrics['accuracy']:.4f}")

    # Save results
    results = {
        "config": config,
        "run_name": run_name,
        "num_params": num_params,
        "best_test_accuracy": final_metrics["accuracy"],
        "per_class_accuracy": {str(k): v for k, v in final_metrics["per_class_accuracy"].items()},
        "history": history,
    }
    results_path = Path(save_dir) / f"{run_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train CNN on CIFAR-10")
    parser.add_argument("--model-type", type=str, default="baseline", choices=["baseline", "attention"])
    parser.add_argument("--attention-positions", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--attention-type", type=str, default="spatial", choices=["spatial", "channel"])
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    config = vars(args)
    # Convert attention_positions to list (argparse already does this, but be explicit)
    config["attention_positions"] = list(config["attention_positions"])

    train(config)


if __name__ == "__main__":
    main()
