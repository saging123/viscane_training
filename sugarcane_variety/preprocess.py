from __future__ import annotations

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, UnidentifiedImageError

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PreprocessDevice = Literal["auto", "cuda", "cpu"]


@dataclass
class PreprocessSummary:
    classes: List[str]
    class_counts: Dict[str, int]
    train_count: int
    val_count: int
    test_count: int
    skipped_corrupt: int


@dataclass
class FlatPreprocessSummary:
    classes: List[str]
    class_counts: Dict[str, int]
    total_count: int
    skipped_corrupt: int


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def _collect_images(raw_dir: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        images = sorted(
            [p for p in class_dir.rglob("*") if p.is_file() and _is_image_file(p)]
        )
        if images:
            class_to_images[class_dir.name] = images
    if not class_to_images:
        raise ValueError(
            f"No class folders with images found in '{raw_dir}'. "
            "Expected structure: raw_dir/<variety_name>/*.jpg"
        )
    return class_to_images


def _collect_images_variety_maturity(raw_dir: Path) -> Dict[str, List[Path]]:
    class_to_images: Dict[str, List[Path]] = {}
    for variety_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        for maturity_dir in sorted(p for p in variety_dir.iterdir() if p.is_dir()):
            images = sorted(
                [p for p in maturity_dir.rglob("*") if p.is_file() and _is_image_file(p)]
            )
            if images:
                class_name = f"{variety_dir.name}__{maturity_dir.name}"
                class_to_images[class_name] = images
    if not class_to_images:
        raise ValueError(
            f"No images found in '{raw_dir}'. "
            "Expected structure: raw_dir/<variety_name>/<maturity_status>/*.jpg"
        )
    return class_to_images


def _collect_images_by_mode(
    raw_dir: Path,
    label_mode: Literal["variety", "variety_maturity"],
) -> Dict[str, List[Path]]:
    if label_mode == "variety":
        return _collect_images(raw_dir)
    if label_mode == "variety_maturity":
        return _collect_images_variety_maturity(raw_dir)
    raise ValueError(f"Unsupported label_mode: {label_mode}")


def _split_class_items(
    items: List[Path],
    val_ratio: float,
    test_ratio: float,
) -> Dict[str, List[Path]]:
    n_total = len(items)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    n_train = n_total - n_val - n_test

    # Keep at least one training sample when possible.
    if n_train <= 0 and n_total > 0:
        n_train = 1
        if n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def _resolve_preprocess_device(device: PreprocessDevice) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "preprocess_device='cuda' was requested, but CUDA is not available."
            )
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported preprocess_device: {device}")


def _center_square_crop_tensor(image: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    crop_size = min(height, width)
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return image[:, top : top + crop_size, left : left + crop_size]


def _resize_image_on_device(
    img: Image.Image,
    image_size: int,
    device: torch.device,
) -> Image.Image:
    img = img.convert("RGB")
    tensor = torch.as_tensor(bytearray(img.tobytes()), dtype=torch.uint8)
    tensor = tensor.reshape(img.height, img.width, 3).permute(2, 0, 1)
    tensor = _center_square_crop_tensor(tensor).to(device=device, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    )
    tensor = tensor.squeeze(0).clamp_(0, 255).to(dtype=torch.uint8).cpu()
    array = tensor.permute(1, 2, 0).numpy()
    return Image.fromarray(array, mode="RGB")


def _save_processed_image(
    src: Path,
    dst: Path,
    image_size: int | None,
    device: torch.device,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    if image_size is None:
        shutil.copy2(src, dst)
        return

    with Image.open(src) as img:
        if device.type == "cuda":
            img = _resize_image_on_device(img, image_size=image_size, device=device)
        else:
            img = img.convert("RGB")
            img = ImageOps.fit(
                img,
                (image_size, image_size),
                method=Image.Resampling.BICUBIC,
            )
        img.save(dst.with_suffix(".jpg"), format="JPEG", quality=95)


def _unique_dst_path(dst_dir: Path, src: Path, force_jpg: bool) -> Path:
    digest = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    suffix = ".jpg" if force_jpg else src.suffix.lower()
    base_name = f"{src.stem}_{digest}{suffix}"
    return dst_dir / base_name


def run_preprocess(
    raw_dir: str,
    output_dir: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    image_size: int | None = None,
    label_mode: Literal["variety", "variety_maturity"] = "variety",
    preprocess_device: PreprocessDevice = "auto",
) -> PreprocessSummary:
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Use ratios where val_ratio >= 0, test_ratio >= 0, and sum < 1.")

    class_to_images = _collect_images_by_mode(raw_path, label_mode=label_mode)
    device = _resolve_preprocess_device(preprocess_device)
    rng = random.Random(seed)

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    skipped_corrupt = 0
    class_counts: Dict[str, int] = {}
    split_totals = {"train": 0, "val": 0, "test": 0}

    for class_name, paths in class_to_images.items():
        valid_paths: List[Path] = []
        for path in paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                valid_paths.append(path)
            except (UnidentifiedImageError, OSError):
                skipped_corrupt += 1

        rng.shuffle(valid_paths)
        class_counts[class_name] = len(valid_paths)
        splits = _split_class_items(valid_paths, val_ratio=val_ratio, test_ratio=test_ratio)

        for split_name, split_paths in splits.items():
            dst_dir = out_path / split_name / class_name
            for src in split_paths:
                dst = _unique_dst_path(dst_dir, src, force_jpg=image_size is not None)
                _save_processed_image(
                    src,
                    dst,
                    image_size=image_size,
                    device=device,
                )
                split_totals[split_name] += 1

    return PreprocessSummary(
        classes=sorted(class_to_images.keys()),
        class_counts=class_counts,
        train_count=split_totals["train"],
        val_count=split_totals["val"],
        test_count=split_totals["test"],
        skipped_corrupt=skipped_corrupt,
    )


def run_preprocess_flat(
    raw_dir: str,
    output_dir: str,
    image_size: int | None = None,
    label_mode: Literal["variety", "variety_maturity"] = "variety",
    preprocess_device: PreprocessDevice = "auto",
) -> FlatPreprocessSummary:
    """
    Validate and preprocess dataset while preserving folder-per-class structure:
      raw/<class_name>/* -> processed/<class_name>/*
    """
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")

    if label_mode == "variety":
        class_to_images = _collect_images(raw_path)
    elif label_mode == "variety_maturity":
        class_to_images = _collect_images_variety_maturity(raw_path)
    else:
        raise ValueError(f"Unsupported label_mode: {label_mode}")

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    device = _resolve_preprocess_device(preprocess_device)
    skipped_corrupt = 0
    total_count = 0
    class_counts: Dict[str, int] = {}

    for class_name, paths in class_to_images.items():
        if label_mode == "variety_maturity":
            variety_name, maturity_name = class_name.split("__", 1)
            dst_dir = out_path / variety_name / maturity_name
        else:
            dst_dir = out_path / class_name
        valid_count = 0
        for src in paths:
            try:
                with Image.open(src) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError):
                skipped_corrupt += 1
                continue

            dst = _unique_dst_path(dst_dir, src, force_jpg=image_size is not None)
            _save_processed_image(src, dst, image_size=image_size, device=device)
            valid_count += 1
            total_count += 1
        class_counts[class_name] = valid_count

    return FlatPreprocessSummary(
        classes=sorted(class_to_images.keys()),
        class_counts=class_counts,
        total_count=total_count,
        skipped_corrupt=skipped_corrupt,
    )
