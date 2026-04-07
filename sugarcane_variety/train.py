from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights


@dataclass
class TrainSummary:
    best_val_acc: float
    test_acc: float
    classes: list[str]
    checkpoint_path: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloaders(
    data_dir: Path,
    image_size: int,
    batch_size: int,
    workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    return train_dl, val_dl, test_dl, train_ds.classes


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        accs.append(_accuracy(logits, labels))
    return float(np.mean(losses)), float(np.mean(accs))


def run_training(
    prepared_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    workers: int = 4,
    seed: int = 42,
) -> TrainSummary:
    _set_seed(seed)
    data_dir = Path(prepared_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split folder: {split_path}. Run preprocessing first."
            )

    train_dl, val_dl, test_dl, classes = _build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_acc = -1.0
    best_ckpt = out_dir / "best_model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        batch_accs = []

        for images, labels in train_dl:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            batch_accs.append(_accuracy(logits, labels))

        scheduler.step()

        train_loss = float(np.mean(batch_losses))
        train_acc = float(np.mean(batch_accs))
        val_loss, val_acc = _eval_epoch(model, val_dl, device, criterion)

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": image_size,
                },
                best_ckpt,
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = _eval_epoch(model, test_dl, device, criterion)
    print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f}")

    metrics: Dict[str, object] = {
        "classes": classes,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "image_size": image_size,
        "seed": seed,
        "device": str(device),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainSummary(
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        classes=classes,
        checkpoint_path=str(best_ckpt),
    )

