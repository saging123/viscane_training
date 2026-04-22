from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Tuple

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
DEFAULT_ONNX_OPSETS = (16, 17)
DEFAULT_TRAIN_NOISE_STD = 0.04
DEFAULT_TRAIN_BLUR_PROB = 0.20
DEFAULT_TRAIN_ERASE_PROB = 0.20
DEFAULT_TRAIN_ROTATION_DEGREES = 12.0
ModelType = Literal["resnet18", "yolov8"]
ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class TrainSummary:
    best_val_acc: float
    test_acc: float
    classes: list[str]
    checkpoint_path: str
    model_type: str = "resnet18"
    android_artifact_path: str | None = None
    onnx_artifact_path: str | None = None
    android_metadata_path: str | None = None


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
    model_type: str = "resnet18"


def _raise_ultralytics_dependency_error(action: str, exc: ImportError) -> None:
    message = str(exc)
    if "libGL.so.1" in message:
        raise RuntimeError(
            f"YOLOv8 {action} could not import OpenCV because libGL.so.1 is missing. "
            "This usually happens on headless or Termux-backed environments. "
            "Install the missing system library (for Debian/Ubuntu: sudo apt install -y libgl1 libglib2.0-0) "
            "or switch to a headless OpenCV build."
        ) from exc
    raise RuntimeError(
        f"YOLOv8 {action} requires the optional 'ultralytics' package and its native dependencies. "
        f"Install them with: pip install ultralytics opencv-python-headless. "
        f"Original import error: {type(exc).__name__}: {exc}"
    ) from exc


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


def _apply_random_rotation(image: torch.Tensor, max_degrees: float) -> torch.Tensor:
    if max_degrees <= 0:
        return image
    angle = random.uniform(-max_degrees, max_degrees)
    return TF.rotate(
        image,
        angle=angle,
        interpolation=transforms.InterpolationMode.BILINEAR,
    )


def _apply_gaussian_noise(image: torch.Tensor, noise_std: float) -> torch.Tensor:
    if noise_std <= 0:
        return image
    noise = torch.randn_like(image).mul_(noise_std)
    return image.add(noise).clamp_(0.0, 1.0)


def _apply_random_blur(image: torch.Tensor, blur_prob: float) -> torch.Tensor:
    if blur_prob <= 0 or random.random() >= blur_prob:
        return image
    kernel_size = random.choice([3, 5])
    sigma = random.uniform(0.2, 1.4)
    return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])


def _apply_random_erasing(image: torch.Tensor, erase_prob: float) -> torch.Tensor:
    if erase_prob <= 0 or random.random() >= erase_prob:
        return image

    _, height, width = image.shape
    erase_h = random.randint(max(1, height // 12), max(1, height // 4))
    erase_w = random.randint(max(1, width // 12), max(1, width // 4))
    top = random.randint(0, max(height - erase_h, 0))
    left = random.randint(0, max(width - erase_w, 0))
    fill_value = random.uniform(0.0, 1.0)
    image[:, top : top + erase_h, left : left + erase_w] = fill_value
    return image


def _prepare_images_on_device(
    images: list[torch.Tensor],
    device: torch.device,
    image_size: int,
    training: bool,
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    erase_prob: float = DEFAULT_TRAIN_ERASE_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
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
            image = _apply_random_rotation(image, rotation_degrees)
            image = _apply_gpu_color_jitter(image)
            image = _apply_random_blur(image, blur_prob)
            image = _apply_gaussian_noise(image, noise_std)
            image = _apply_random_erasing(image, erase_prob)
        else:
            image = _resize_shorter_edge(image, int(image_size * 1.15))
            image = _center_crop_tensor(image, image_size)
            if augment_validation:
                image = _apply_random_rotation(image, rotation_degrees * 0.5)
                image = _apply_random_blur(image, blur_prob * 0.5)
                image = _apply_gaussian_noise(image, noise_std * 0.5)

        processed.append(image)

    batch = torch.stack(processed)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return batch.sub_(mean).div_(std)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def _validate_prepared_dir(data_dir: Path) -> None:
    for split in ["train", "val", "test"]:
        split_path = data_dir / split
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split folder: {split_path}. Run preprocessing first."
            )


def _build_resnet18(num_classes: int, pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _get_onnx_export_opsets() -> tuple[int, ...]:
    raw_value = os.getenv("ONNX_EXPORT_OPSETS", "")
    if not raw_value.strip():
        return DEFAULT_ONNX_OPSETS

    parsed: list[int] = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            parsed.append(int(chunk))
        except ValueError:
            print(f"Ignoring invalid ONNX opset value: {chunk!r}")

    return tuple(parsed) or DEFAULT_ONNX_OPSETS


def _export_onnx_with_opset_fallback(
    model: nn.Module,
    example: torch.Tensor,
    target_path: Path,
) -> str | None:
    last_error: Exception | None = None
    for opset_version in _get_onnx_export_opsets():
        try:
            torch.onnx.export(
                model,
                example,
                str(target_path),
                input_names=["images"],
                output_names=["logits"],
                dynamic_axes={
                    "images": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=opset_version,
            )
            return str(target_path)
        except Exception as exc:
            last_error = exc
            print(
                f"ONNX export failed with opset {opset_version}: "
                f"{type(exc).__name__}: {exc}"
            )

    if last_error is not None:
        print(f"Skipping ONNX export after all opset attempts failed: {type(last_error).__name__}: {last_error}")
    return None


def _export_resnet18_android_artifacts(
    model: nn.Module,
    out_dir: Path,
    image_size: int,
) -> tuple[str | None, str | None]:
    model_cpu = model.to("cpu").eval()
    android_path = out_dir / "resnet18_android.ptl"
    onnx_path = out_dir / "resnet18_android.onnx"
    example = torch.randn(1, 3, image_size, image_size)

    android_artifact_path: str | None = None
    onnx_artifact_path: str | None = None

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile

        scripted = torch.jit.script(model_cpu)
        optimized = optimize_for_mobile(scripted)
        optimized._save_for_lite_interpreter(str(android_path))
        android_artifact_path = str(android_path)
    except Exception as exc:
        print(f"Skipping ResNet18 Android Lite export: {type(exc).__name__}: {exc}")

    onnx_artifact_path = _export_onnx_with_opset_fallback(
        model=model_cpu,
        example=example,
        target_path=onnx_path,
    )

    return android_artifact_path, onnx_artifact_path


def _export_yolov8_onnx_fallback(
    best_model: Any,
    out_dir: Path,
    image_size: int,
) -> str | None:
    onnx_path = out_dir / "best.onnx"
    model = getattr(best_model, "model", None)
    if model is None:
        return None

    model_cpu = model.to("cpu").eval()
    example = torch.randn(1, 3, image_size, image_size)
    return _export_onnx_with_opset_fallback(
        model=model_cpu,
        example=example,
        target_path=onnx_path,
    )


def _write_android_metadata(
    out_dir: Path,
    model_type: str,
    classes: list[str],
    image_size: int,
    checkpoint_path: str,
    onnx_artifact_path: str | None,
    android_artifact_path: str | None,
) -> str:
    metadata_path = out_dir / f"{model_type}_android_metadata.json"
    payload: Dict[str, object] = {
        "model_type": model_type,
        "classes": classes,
        "class_label_components": [_decode_class_name(name) for name in classes],
        "image_size": image_size,
        "checkpoint_path": checkpoint_path,
        "android_artifact_path": android_artifact_path,
        "onnx_artifact_path": onnx_artifact_path,
        "input_format": "RGB image resized/cropped to CHW float tensor",
        "input_range": "0.0 to 1.0 before normalization",
        "normalization_mean": IMAGENET_MEAN if model_type == "resnet18" else None,
        "normalization_std": IMAGENET_STD if model_type == "resnet18" else None,
        "output": "class logits/probabilities ordered by the classes array",
    }
    metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(metadata_path)


@torch.no_grad()
def _eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    image_size: int,
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
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
            augment_validation=augment_validation,
            noise_std=noise_std,
            blur_prob=blur_prob,
            rotation_degrees=rotation_degrees,
        )
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        accs.append(_accuracy(logits, labels))
    return float(np.mean(losses)), float(np.mean(accs))


def _run_resnet18_training(
    prepared_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    workers: int = 4,
    seed: int = 42,
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    erase_prob: float = DEFAULT_TRAIN_ERASE_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
    progress_callback: ProgressCallback | None = None,
) -> TrainSummary:
    _set_seed(seed)
    data_dir = Path(prepared_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _validate_prepared_dir(data_dir)

    train_dl, val_dl, test_dl, classes = _build_dataloaders(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        workers=workers,
    )
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_resnet18(num_classes=len(classes), pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_acc = -1.0
    best_ckpt = out_dir / "best_model.pt"
    epoch_history: List[Dict[str, object]] = []
    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_started",
                "model_type": "resnet18",
                "epochs": epochs,
                "classes": classes,
                "output_dir": str(out_dir),
                "device": str(device),
                "augmentation": {
                    "augment_validation": augment_validation,
                    "noise_std": noise_std,
                    "blur_prob": blur_prob,
                    "erase_prob": erase_prob,
                    "rotation_degrees": rotation_degrees,
                },
            }
        )

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
                noise_std=noise_std,
                blur_prob=blur_prob,
                erase_prob=erase_prob,
                rotation_degrees=rotation_degrees,
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
        current_lr = float(scheduler.get_last_lr()[0])
        val_loss, val_acc = _eval_epoch(
            model,
            val_dl,
            device,
            criterion,
            image_size,
            augment_validation=augment_validation,
            noise_std=noise_std,
            blur_prob=blur_prob,
            rotation_degrees=rotation_degrees,
        )
        epoch_record = {
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_val_acc": best_val_acc if best_val_acc > val_acc else val_acc,
            "lr": current_lr,
        }
        epoch_history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{epochs} "
            f"| train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_type": "resnet18",
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "image_size": image_size,
                },
                best_ckpt,
            )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "epoch_completed",
                    "model_type": "resnet18",
                    "checkpoint_path": str(best_ckpt),
                    **epoch_record,
                }
            )

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = _eval_epoch(
        model,
        test_dl,
        device,
        criterion,
        image_size,
        augment_validation=augment_validation,
        noise_std=noise_std,
        blur_prob=blur_prob,
        rotation_degrees=rotation_degrees,
    )
    print(f"Final test | loss={test_loss:.4f} acc={test_acc:.4f}")
    android_artifact_path, onnx_artifact_path = _export_resnet18_android_artifacts(
        model=model,
        out_dir=out_dir,
        image_size=image_size,
    )
    android_metadata_path = _write_android_metadata(
        out_dir=out_dir,
        model_type="resnet18",
        classes=classes,
        image_size=image_size,
        checkpoint_path=str(best_ckpt),
        onnx_artifact_path=onnx_artifact_path,
        android_artifact_path=android_artifact_path,
    )

    metrics: Dict[str, object] = {
        "model_type": "resnet18",
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
        "augmentation": {
            "augment_validation": augment_validation,
            "noise_std": noise_std,
            "blur_prob": blur_prob,
            "erase_prob": erase_prob,
            "rotation_degrees": rotation_degrees,
        },
        "epoch_history": epoch_history,
        "android_artifact_path": android_artifact_path,
        "onnx_artifact_path": onnx_artifact_path,
        "android_metadata_path": android_metadata_path,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_completed",
                "model_type": "resnet18",
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "checkpoint_path": str(best_ckpt),
                "metrics_path": str(metrics_path),
                "android_artifact_path": android_artifact_path,
                "onnx_artifact_path": onnx_artifact_path,
                "android_metadata_path": android_metadata_path,
            }
        )

    return TrainSummary(
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        classes=classes,
        checkpoint_path=str(best_ckpt),
        model_type="resnet18",
        android_artifact_path=android_artifact_path,
        onnx_artifact_path=onnx_artifact_path,
        android_metadata_path=android_metadata_path,
    )


def _run_yolov8_training(
    prepared_dir: str,
    output_dir: str,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    workers: int = 4,
    seed: int = 42,
    yolo_weights: str = "yolov8n-cls.pt",
    progress_callback: ProgressCallback | None = None,
) -> TrainSummary:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        _raise_ultralytics_dependency_error("training", exc)

    _set_seed(seed)
    data_dir = Path(prepared_dir).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _validate_prepared_dir(data_dir)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transforms.PILToTensor())
    classes = train_ds.classes
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes to train a classifier.")

    model = YOLO(yolo_weights)
    epoch_history: List[Dict[str, object]] = []
    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_started",
                "model_type": "yolov8",
                "epochs": epochs,
                "classes": classes,
                "output_dir": str(out_dir),
                "yolo_weights": yolo_weights,
            }
        )

        def _on_yolo_epoch_end(trainer: object) -> None:
            epoch = int(getattr(trainer, "epoch", -1)) + 1
            metrics = getattr(trainer, "metrics", {}) or {}
            optimizer = getattr(trainer, "optimizer", None)
            current_lr = None
            if optimizer is not None:
                try:
                    current_lr = float(optimizer.param_groups[0].get("lr"))
                except Exception:
                    current_lr = None
            epoch_record = {
                "epoch": epoch,
                "epochs": epochs,
                "lr": current_lr,
                "metrics": _json_safe(metrics),
            }
            epoch_history.append(epoch_record)
            progress_callback(
                {
                    "event": "epoch_completed",
                    "model_type": "yolov8",
                    **epoch_record,
                }
            )

        try:
            model.add_callback("on_train_epoch_end", _on_yolo_epoch_end)
        except Exception as exc:
            progress_callback(
                {
                    "event": "progress_callback_warning",
                    "model_type": "yolov8",
                    "message": f"YOLO epoch callback could not be attached: {type(exc).__name__}: {exc}",
                }
            )

    train_result = model.train(
        data=str(data_dir),
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        workers=workers,
        seed=seed,
        lr0=lr,
        weight_decay=weight_decay,
        project=str(out_dir),
        name="yolov8",
        exist_ok=True,
    )

    save_dir = Path(getattr(train_result, "save_dir", out_dir / "yolov8"))
    best_ckpt = save_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        best_ckpt = save_dir / "weights" / "last.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"YOLOv8 checkpoint was not found under: {save_dir}")

    best_model = YOLO(str(best_ckpt))
    val_metrics = best_model.val(data=str(data_dir), split="val", imgsz=image_size)
    test_metrics = best_model.val(data=str(data_dir), split="test", imgsz=image_size)
    best_val_acc = _extract_yolo_top1(val_metrics)
    test_acc = _extract_yolo_top1(test_metrics)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "evaluation_completed",
                "model_type": "yolov8",
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "checkpoint_path": str(best_ckpt),
            }
        )

    onnx_artifact_path: str | None = None
    try:
        exported = best_model.export(format="onnx", imgsz=image_size)
        onnx_artifact_path = str(exported)
    except Exception as exc:
        print(f"Ultralytics YOLOv8 ONNX export failed: {type(exc).__name__}: {exc}")
        onnx_artifact_path = _export_yolov8_onnx_fallback(
            best_model=best_model,
            out_dir=save_dir / "weights",
            image_size=image_size,
        )
    android_metadata_path = _write_android_metadata(
        out_dir=out_dir,
        model_type="yolov8",
        classes=classes,
        image_size=image_size,
        checkpoint_path=str(best_ckpt),
        onnx_artifact_path=onnx_artifact_path,
        android_artifact_path=onnx_artifact_path,
    )

    metadata_path = out_dir / "yolov8_metadata.json"
    metadata: Dict[str, object] = {
        "model_type": "yolov8",
        "classes": classes,
        "class_label_components": [_decode_class_name(name) for name in classes],
        "checkpoint_path": str(best_ckpt),
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "image_size": image_size,
        "seed": seed,
        "yolo_weights": yolo_weights,
        "epoch_history": epoch_history,
        "onnx_artifact_path": onnx_artifact_path,
        "android_metadata_path": android_metadata_path,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if progress_callback is not None:
        progress_callback(
            {
                "event": "training_completed",
                "model_type": "yolov8",
                "best_val_acc": best_val_acc,
                "test_acc": test_acc,
                "checkpoint_path": str(best_ckpt),
                "metrics_path": str(metrics_path),
                "android_artifact_path": onnx_artifact_path,
                "onnx_artifact_path": onnx_artifact_path,
                "android_metadata_path": android_metadata_path,
            }
        )

    return TrainSummary(
        best_val_acc=best_val_acc,
        test_acc=test_acc,
        classes=classes,
        checkpoint_path=str(best_ckpt),
        model_type="yolov8",
        android_artifact_path=onnx_artifact_path,
        onnx_artifact_path=onnx_artifact_path,
        android_metadata_path=android_metadata_path,
    )


def _extract_yolo_top1(metrics: object) -> float:
    for path in [
        ("top1",),
        ("metrics", "top1"),
        ("results_dict", "metrics/accuracy_top1"),
        ("results_dict", "top1"),
    ]:
        value: object = metrics
        for attr in path:
            if isinstance(value, dict):
                value = value.get(attr)
            else:
                value = getattr(value, attr, None)
            if value is None:
                break
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    erase_prob: float = DEFAULT_TRAIN_ERASE_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
    model_type: ModelType = "resnet18",
    yolo_weights: str = "yolov8n-cls.pt",
    progress_callback: ProgressCallback | None = None,
) -> TrainSummary:
    if model_type == "resnet18":
        return _run_resnet18_training(
            prepared_dir=prepared_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            image_size=image_size,
            workers=workers,
            seed=seed,
            augment_validation=augment_validation,
            noise_std=noise_std,
            blur_prob=blur_prob,
            erase_prob=erase_prob,
            rotation_degrees=rotation_degrees,
            progress_callback=progress_callback,
        )
    if model_type == "yolov8":
        return _run_yolov8_training(
            prepared_dir=prepared_dir,
            output_dir=output_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            image_size=image_size,
            workers=workers,
            seed=seed,
            yolo_weights=yolo_weights,
            progress_callback=progress_callback,
        )
    raise ValueError("model_type must be 'resnet18' or 'yolov8'.")


def run_evaluation(
    prepared_dir: str,
    checkpoint_path: str,
    batch_size: int = 32,
    workers: int = 4,
    model_type: ModelType | None = None,
) -> EvalSummary:
    data_dir = Path(prepared_dir).expanduser().resolve()
    ckpt_path = Path(checkpoint_path).expanduser().resolve()

    test_path = data_dir / "test"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test folder: {test_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if model_type == "yolov8":
        return _run_yolov8_evaluation(
            prepared_dir=prepared_dir,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            workers=workers,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
    except Exception:
        if model_type is None:
            return _run_yolov8_evaluation(
                prepared_dir=prepared_dir,
                checkpoint_path=checkpoint_path,
                batch_size=batch_size,
                workers=workers,
            )
        raise
    inferred_model_type = model_type or str(checkpoint.get("model_type", "resnet18"))
    if inferred_model_type == "yolov8":
        return _run_yolov8_evaluation(
            prepared_dir=prepared_dir,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            workers=workers,
        )
    if inferred_model_type != "resnet18":
        raise ValueError("model_type must be 'resnet18' or 'yolov8'.")

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

    model = _build_resnet18(num_classes=len(classes), pretrained=False)
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
        model_type="resnet18",
    )


def _run_yolov8_evaluation(
    prepared_dir: str,
    checkpoint_path: str,
    batch_size: int = 32,
    workers: int = 4,
) -> EvalSummary:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        _raise_ultralytics_dependency_error("evaluation", exc)

    data_dir = Path(prepared_dir).expanduser().resolve()
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    test_path = data_dir / "test"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test folder: {test_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = YOLO(str(ckpt_path))
    metrics = model.val(
        data=str(data_dir),
        split="test",
        batch=batch_size,
        workers=workers,
    )
    test_acc = _extract_yolo_top1(metrics)
    classes = [
        name
        for _, name in sorted(
            getattr(model, "names", {}).items(),
            key=lambda item: int(item[0]),
        )
    ]
    if not classes:
        test_ds = datasets.ImageFolder(test_path, transform=transforms.PILToTensor())
        classes = test_ds.classes
    num_samples = len(datasets.ImageFolder(test_path, transform=transforms.PILToTensor()))

    variety_acc = test_acc
    maturity_acc = None
    if any(_decode_class_name(name)["maturity_status"] for name in classes):
        maturity_acc = test_acc

    interpretation_points = [
        f"YOLOv8 top-1 accuracy: {test_acc:.2%}",
        "For detailed YOLOv8 confusion plots, inspect the validation output folder.",
    ]
    friendly_outcome = _friendly_outcome_text(test_acc)
    summary_json_path = ckpt_path.parent.parent / "test_summary.json"
    summary_payload: Dict[str, object] = {
        "model_type": "yolov8",
        "test_loss": 0.0,
        "test_acc": test_acc,
        "num_samples": num_samples,
        "classes": classes,
        "checkpoint_path": str(ckpt_path),
        "device": "ultralytics",
        "variety_acc": variety_acc,
        "maturity_acc": maturity_acc,
        "per_class": [],
        "top_confusions": [],
        "friendly_outcome": friendly_outcome,
        "interpretation_points": interpretation_points,
    }
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return EvalSummary(
        test_loss=0.0,
        test_acc=test_acc,
        num_samples=num_samples,
        classes=classes,
        checkpoint_path=str(ckpt_path),
        device="ultralytics",
        variety_acc=variety_acc,
        maturity_acc=maturity_acc,
        per_class=[],
        top_confusions=[],
        friendly_outcome=friendly_outcome,
        interpretation_points=interpretation_points,
        summary_json_path=str(summary_json_path),
        model_type="yolov8",
    )
