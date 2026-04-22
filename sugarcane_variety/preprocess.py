from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
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


@dataclass
class SplitLeakageSummary:
    prepared_dir: str
    total_images: int
    exact_duplicate_groups: int
    near_duplicate_groups: int
    cross_split_exact_groups: int
    cross_split_near_groups: int
    suspicious_examples: List[Dict[str, object]]
    summary_json_path: str


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def _average_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
        pixels = list(img.getdata())
    mean_value = sum(pixels) / max(len(pixels), 1)
    bits = "".join("1" if pixel >= mean_value else "0" for pixel in pixels)
    return f"{int(bits, 2):0{hash_size * hash_size // 4}x}"


def _hamming_distance(hash_a: str, hash_b: str) -> int:
    return (int(hash_a, 16) ^ int(hash_b, 16)).bit_count()


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


def _verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError):
        return False


def _verify_image_worker(path: str) -> tuple[str, bool]:
    return path, _verify_image(Path(path))


def _filter_valid_paths(
    paths: List[Path],
    workers: int,
) -> tuple[List[Path], int]:
    if workers <= 1:
        valid_paths: List[Path] = []
        skipped_corrupt = 0
        for path in paths:
            if _verify_image(path):
                valid_paths.append(path)
            else:
                skipped_corrupt += 1
        return valid_paths, skipped_corrupt

    valid_paths = []
    skipped_corrupt = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for path, is_valid in executor.map(_verify_image_worker, [str(p) for p in paths]):
            if is_valid:
                valid_paths.append(Path(path))
            else:
                skipped_corrupt += 1
    return valid_paths, skipped_corrupt


def _save_processed_image_worker(
    task: tuple[str, str, int | None, str],
) -> None:
    src, dst, image_size, device_type = task
    _save_processed_image(
        Path(src),
        Path(dst),
        image_size=image_size,
        device=torch.device(device_type),
    )


def _audit_image_worker(path: str) -> tuple[str, str, str]:
    path_obj = Path(path)
    content_hash = hashlib.sha1(path_obj.read_bytes()).hexdigest()
    perceptual_hash = _average_hash(path_obj)
    return path, content_hash, perceptual_hash


def _save_processed_images(
    tasks: List[tuple[Path, Path]],
    image_size: int | None,
    device: torch.device,
    workers: int,
) -> None:
    if not tasks:
        return

    if workers <= 1 or device.type == "cuda":
        for src, dst in tasks:
            _save_processed_image(src, dst, image_size=image_size, device=device)
        return

    worker_tasks = [
        (str(src), str(dst), image_size, device.type)
        for src, dst in tasks
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_save_processed_image_worker, worker_tasks))


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
    preprocess_workers: int = 1,
) -> PreprocessSummary:
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1:
        raise ValueError("Use ratios where val_ratio >= 0, test_ratio >= 0, and sum < 1.")
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers must be at least 1.")

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
        valid_paths, class_skipped = _filter_valid_paths(
            paths,
            workers=preprocess_workers,
        )
        skipped_corrupt += class_skipped

        rng.shuffle(valid_paths)
        class_counts[class_name] = len(valid_paths)
        splits = _split_class_items(valid_paths, val_ratio=val_ratio, test_ratio=test_ratio)

        save_tasks: List[tuple[Path, Path]] = []
        for split_name, split_paths in splits.items():
            dst_dir = out_path / split_name / class_name
            for src in split_paths:
                dst = _unique_dst_path(dst_dir, src, force_jpg=image_size is not None)
                save_tasks.append((src, dst))
                split_totals[split_name] += 1
        _save_processed_images(
            save_tasks,
            image_size=image_size,
            device=device,
            workers=preprocess_workers,
        )

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
    preprocess_workers: int = 1,
) -> FlatPreprocessSummary:
    """
    Validate and preprocess dataset while preserving folder-per-class structure:
      raw/<class_name>/* -> processed/<class_name>/*
    """
    raw_path = Path(raw_dir).expanduser().resolve()
    out_path = Path(output_dir).expanduser().resolve()

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset path does not exist: {raw_path}")
    if preprocess_workers < 1:
        raise ValueError("preprocess_workers must be at least 1.")

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
        valid_paths, class_skipped = _filter_valid_paths(
            paths,
            workers=preprocess_workers,
        )
        skipped_corrupt += class_skipped

        save_tasks: List[tuple[Path, Path]] = []
        for src in valid_paths:
            dst = _unique_dst_path(dst_dir, src, force_jpg=image_size is not None)
            total_count += 1
            save_tasks.append((src, dst))
        _save_processed_images(
            save_tasks,
            image_size=image_size,
            device=device,
            workers=preprocess_workers,
        )
        class_counts[class_name] = len(valid_paths)

    return FlatPreprocessSummary(
        classes=sorted(class_to_images.keys()),
        class_counts=class_counts,
        total_count=total_count,
        skipped_corrupt=skipped_corrupt,
    )


def audit_prepared_splits(
    prepared_dir: str,
    near_duplicate_distance: int = 5,
    max_examples: int = 25,
    workers: int = 1,
) -> SplitLeakageSummary:
    prepared_path = Path(prepared_dir).expanduser().resolve()
    if not prepared_path.exists():
        raise FileNotFoundError(f"Prepared dataset path does not exist: {prepared_path}")
    if near_duplicate_distance < 0:
        raise ValueError("near_duplicate_distance must be non-negative.")
    if max_examples < 1:
        raise ValueError("max_examples must be at least 1.")

    image_paths = sorted(
        path
        for split_name in ("train", "val", "test")
        for path in (prepared_path / split_name).rglob("*")
        if path.is_file() and _is_image_file(path)
    )
    if not image_paths:
        raise ValueError(f"No prepared images found under: {prepared_path}")

    if workers <= 1:
        audit_rows = [_audit_image_worker(str(path)) for path in image_paths]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            audit_rows = list(executor.map(_audit_image_worker, [str(path) for path in image_paths]))

    image_infos: List[Dict[str, object]] = []
    by_content_hash: Dict[str, List[Dict[str, object]]] = {}
    for raw_path, content_hash, perceptual_hash in audit_rows:
        path = Path(raw_path)
        relative = path.relative_to(prepared_path)
        parts = relative.parts
        split_name = parts[0] if parts else "unknown"
        class_name = parts[1] if len(parts) > 1 else "unknown"
        info = {
            "path": str(path),
            "relative_path": str(relative),
            "split": split_name,
            "class_name": class_name,
            "content_hash": content_hash,
            "perceptual_hash": perceptual_hash,
        }
        image_infos.append(info)
        by_content_hash.setdefault(content_hash, []).append(info)

    suspicious_examples: List[Dict[str, object]] = []

    exact_duplicate_groups = 0
    cross_split_exact_groups = 0
    for content_hash, items in by_content_hash.items():
        if len(items) < 2:
            continue
        exact_duplicate_groups += 1
        splits = sorted({str(item["split"]) for item in items})
        if len(splits) > 1 and len(suspicious_examples) < max_examples:
            cross_split_exact_groups += 1
            suspicious_examples.append(
                {
                    "type": "exact_duplicate",
                    "content_hash": content_hash,
                    "splits": splits,
                    "items": [
                        {
                            "relative_path": str(item["relative_path"]),
                            "class_name": str(item["class_name"]),
                        }
                        for item in items[:6]
                    ],
                }
            )
        elif len(splits) > 1:
            cross_split_exact_groups += 1

    near_duplicate_groups = 0
    cross_split_near_groups = 0
    for index, current in enumerate(image_infos):
        current_split = str(current["split"])
        current_hash = str(current["perceptual_hash"])
        for other in image_infos[index + 1 :]:
            if str(current["content_hash"]) == str(other["content_hash"]):
                continue
            distance = _hamming_distance(current_hash, str(other["perceptual_hash"]))
            if distance > near_duplicate_distance:
                continue
            near_duplicate_groups += 1
            if current_split != str(other["split"]):
                cross_split_near_groups += 1
                if len(suspicious_examples) < max_examples:
                    suspicious_examples.append(
                        {
                            "type": "near_duplicate",
                            "distance": distance,
                            "splits": sorted({current_split, str(other["split"])}),
                            "items": [
                                {
                                    "relative_path": str(current["relative_path"]),
                                    "class_name": str(current["class_name"]),
                                },
                                {
                                    "relative_path": str(other["relative_path"]),
                                    "class_name": str(other["class_name"]),
                                },
                            ],
                        }
                    )

    summary = {
        "prepared_dir": str(prepared_path),
        "total_images": len(image_infos),
        "exact_duplicate_groups": exact_duplicate_groups,
        "near_duplicate_groups": near_duplicate_groups,
        "cross_split_exact_groups": cross_split_exact_groups,
        "cross_split_near_groups": cross_split_near_groups,
        "suspicious_examples": suspicious_examples,
    }
    summary_json_path = prepared_path / "split_leakage_audit.json"
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return SplitLeakageSummary(
        prepared_dir=str(prepared_path),
        total_images=len(image_infos),
        exact_duplicate_groups=exact_duplicate_groups,
        near_duplicate_groups=near_duplicate_groups,
        cross_split_exact_groups=cross_split_exact_groups,
        cross_split_near_groups=cross_split_near_groups,
        suspicious_examples=suspicious_examples,
        summary_json_path=str(summary_json_path),
    )
