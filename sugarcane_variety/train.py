from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.transforms import functional as TF
from torchvision.models import ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class TrainSummary:
    best_val_acc: float
    test_acc: float
    classes: list[str]
    checkpoint_path: str


@dataclass
class EvalSummary:
    test_loss: float
    test_acc: float
    num_samples: int
    classes: list[str]
    checkpoint_path: str
    device: str
    variety_acc: float
    maturity_acc: float | None
    per_class: List[Dict[str, object]]
    top_confusions: List[Dict[str, object]]
    friendly_outcome: str
    interpretation_points: List[str]
    summary_json_path: str


def _decode_class_name(class_name: str) -> Dict[str, str]:
    if "__" in class_name:
        variety, maturity = class_name.split("__", 1)
        return {
            "class_name": class_name,
            "variety": variety,
            "maturity_status": maturity,
        }
    return {
        "class_name": class_name,
        "variety": class_name,
        "maturity_status": "",
    }


def _friendly_outcome_text(test_acc: float) -> str:
    if test_acc >= 0.9:
        return (
            "Excellent result. The model is highly reliable on this test set and is a "
            "strong candidate for pilot deployment."
        )
    if test_acc >= 0.8:
        return (
            "Strong result. The model is performing well, with room to improve edge cases "
            "and confusing classes."
        )
    if test_acc >= 0.7:
        return (
            "Promising result. The model has learned useful patterns, but needs more data "
            "or tuning before production use."
        )
    if test_acc >= 0.6:
        return (
            "Moderate result. The model is partially useful, but still makes many mistakes. "
            "Focus on dataset quality and class balance."
        )
    return (
        "Early-stage result. The model is not yet reliable. Prioritize cleaner labeling, "
        "more samples per class, and improved preprocessing."
    )


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
    raw_tfms = transforms.PILToTensor()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=raw_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=raw_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=raw_tfms)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate_raw_images,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate_raw_images,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate_raw_images,
    )
    return train_dl, val_dl, test_dl, train_ds.classes


def _collate_raw_images(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def _resize_tensor(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=size,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _center_crop_tensor(image: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, height, width = image.shape
    top = max((height - crop_size) // 2, 0)
    left = max((width - crop_size) // 2, 0)
    return image[:, top : top + crop_size, left : left + crop_size]


def _resize_shorter_edge(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    if height < width:
        new_height = size
        new_width = int(width * size / height)
    else:
        new_width = size
        new_height = int(height * size / width)
    return _resize_tensor(image, (new_height, new_width))


def _apply_gpu_color_jitter(image: torch.Tensor) -> torch.Tensor:
    adjustments = [
        lambda x: TF.adjust_brightness(x, random.uniform(0.8, 1.2)),
        lambda x: TF.adjust_contrast(x, random.uniform(0.8, 1.2)),
        lambda x: TF.adjust_saturation(x, random.uniform(0.8, 1.2)),
        lambda x: TF.adjust_hue(x, random.uniform(-0.05, 0.05)),
    ]
    random.shuffle(adjustments)
    for adjustment in adjustments:
        image = adjustment(image)
    return image


def _prepare_images_on_device(
    images: list[torch.Tensor],
    device: torch.device,
    image_size: int,
    training: bool,
) -> torch.Tensor:
    processed: list[torch.Tensor] = []
    for image in images:
        image = image.to(device=device, non_blocking=True, dtype=torch.float32).div_(255.0)
        if training:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=(0.75, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            )
            image = image[:, i : i + h, j : j + w]
            image = _resize_tensor(image, (image_size, image_size))
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
            image = _apply_gpu_color_jitter(image)
        else:
            image = _resize_shorter_edge(image, int(image_size * 1.15))
            image = _center_crop_tensor(image, image_size)

        processed.append(image)

    batch = torch.stack(processed)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    image_size: int,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    accs = []
    for images, labels in dataloader:
        images = _prepare_images_on_device(
            images,
            device=device,
            image_size=image_size,
            training=False,
        )
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
            images = _prepare_images_on_device(
                images,
                device=device,
                image_size=image_size,
                training=True,
            )
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
        val_loss, val_acc = _eval_epoch(model, val_dl, device, criterion, image_size)

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
    test_loss, test_acc = _eval_epoch(model, test_dl, device, criterion, image_size)
    print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f}")

    metrics: Dict[str, object] = {
        "classes": classes,
        "class_label_components": [_decode_class_name(name) for name in classes],
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


def run_evaluation(
    prepared_dir: str,
    checkpoint_path: str,
    batch_size: int = 32,
    workers: int = 4,
) -> EvalSummary:
    data_dir = Path(prepared_dir).expanduser().resolve()
    ckpt_path = Path(checkpoint_path).expanduser().resolve()

    test_path = data_dir / "test"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test folder: {test_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(ckpt_path, map_location=device)
    classes = checkpoint["classes"]
    image_size = int(checkpoint["image_size"])

    test_ds = datasets.ImageFolder(test_path, transform=transforms.PILToTensor())
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        collate_fn=_collate_raw_images,
    )

    if test_ds.classes != classes:
        raise ValueError(
            "Class mismatch between checkpoint and test data.\n"
            f"Checkpoint classes: {classes}\n"
            f"Test classes: {test_ds.classes}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss_sum = 0.0
    num_samples = 0
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for images, labels in test_dl:
            images = _prepare_images_on_device(
                images,
                device=device,
                image_size=image_size,
                training=False,
            )
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size_now = labels.size(0)

            loss_sum += loss.item() * batch_size_now
            num_samples += batch_size_now
            all_targets.extend(labels.detach().cpu().tolist())
            all_preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())

    if num_samples == 0:
        raise ValueError(f"No test samples found in: {test_path}")

    test_loss = loss_sum / num_samples
    test_acc = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    print(f"Test only | loss={test_loss:.4f} acc={test_acc:.4f}")

    decoded = [_decode_class_name(name) for name in classes]
    class_support = [0] * len(classes)
    class_correct = [0] * len(classes)
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)

    variety_correct = 0
    maturity_correct = 0
    has_maturity = any(item["maturity_status"] for item in decoded)

    for target_idx, pred_idx in zip(all_targets, all_preds):
        class_support[target_idx] += 1
        if target_idx == pred_idx:
            class_correct[target_idx] += 1
        confusion[target_idx, pred_idx] += 1

        target_info = decoded[target_idx]
        pred_info = decoded[pred_idx]
        if target_info["variety"] == pred_info["variety"]:
            variety_correct += 1
        if has_maturity and target_info["maturity_status"] == pred_info["maturity_status"]:
            maturity_correct += 1

    variety_acc = variety_correct / num_samples
    maturity_acc = (maturity_correct / num_samples) if has_maturity else None

    per_class: List[Dict[str, object]] = []
    for idx, class_name in enumerate(classes):
        support = class_support[idx]
        correct = class_correct[idx]
        class_acc = (correct / support) if support > 0 else 0.0
        info = decoded[idx]
        per_class.append(
            {
                "class_name": class_name,
                "variety": info["variety"],
                "maturity_status": info["maturity_status"],
                "support": support,
                "correct": correct,
                "accuracy": class_acc,
            }
        )

    confusion_pairs: List[Dict[str, object]] = []
    for true_idx in range(len(classes)):
        for pred_idx in range(len(classes)):
            if true_idx == pred_idx:
                continue
            count = int(confusion[true_idx, pred_idx])
            if count > 0:
                confusion_pairs.append(
                    {
                        "true_class": classes[true_idx],
                        "predicted_class": classes[pred_idx],
                        "count": count,
                    }
                )
    confusion_pairs.sort(key=lambda x: int(x["count"]), reverse=True)
    top_confusions = confusion_pairs[:10]

    interpretation_points = [
        f"Exact label accuracy (variety + maturity): {test_acc:.2%}",
        f"Variety-only accuracy: {variety_acc:.2%}",
    ]
    if maturity_acc is not None:
        interpretation_points.append(f"Maturity-only accuracy: {maturity_acc:.2%}")
    interpretation_points.append(
        "Check top_confusions to see which classes the model mixes up most often."
    )

    friendly_outcome = _friendly_outcome_text(test_acc)
    summary_json_path = ckpt_path.parent / "test_summary.json"
    summary_payload: Dict[str, object] = {
        "test_loss": test_loss,
        "test_acc": test_acc,
        "num_samples": num_samples,
        "classes": classes,
        "checkpoint_path": str(ckpt_path),
        "device": str(device),
        "variety_acc": variety_acc,
        "maturity_acc": maturity_acc,
        "per_class": per_class,
        "top_confusions": top_confusions,
        "friendly_outcome": friendly_outcome,
        "interpretation_points": interpretation_points,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return EvalSummary(
        test_loss=test_loss,
        test_acc=test_acc,
        num_samples=num_samples,
        classes=classes,
        checkpoint_path=str(ckpt_path),
        device=str(device),
        variety_acc=variety_acc,
        maturity_acc=maturity_acc,
        per_class=per_class,
        top_confusions=top_confusions,
        friendly_outcome=friendly_outcome,
        interpretation_points=interpretation_points,
        summary_json_path=str(summary_json_path),
    )
