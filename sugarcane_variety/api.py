from __future__ import annotations

import csv
import html
import io
import json
import os
import threading
import time
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import torch
import torch.nn.functional as F
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import transforms

from sugarcane_variety.train import (
    DEFAULT_TRAIN_BLUR_PROB,
    DEFAULT_TRAIN_ERASE_PROB,
    DEFAULT_TRAIN_NOISE_STD,
    DEFAULT_TRAIN_ROTATION_DEGREES,
    DEFAULT_USE_BALANCED_SAMPLER,
    _build_resnet18,
    _build_resnet18_two_head,
    _combine_two_head_logits,
    _decode_class_name,
    _prepare_images_on_device,
    _raise_ultralytics_dependency_error,
)
from sugarcane_variety.colab_compatible import run_all_for_colab, test_for_colab


DEFAULT_ARTIFACTS_DIR = "content/data/sugarcane_artifacts"
DEFAULT_PREPARED_DIR = "content/data/prepared"
DEFAULT_RESNET_CHECKPOINT_PATH = "content/data/sugarcane_artifacts/resnet18/best_model.pt"
DEFAULT_YOLO_CHECKPOINT_PATH = "content/data/sugarcane_artifacts/yolov8/yolov8/weights/best.pt"
ARTIFACT_EXTENSIONS = {".pt", ".ptl", ".onnx", ".json"}
VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
DATASET_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SUPPORTED_MODELS = {"resnet18", "resnet18_two_head", "yolov8"}
REPORT_MODEL_SLOTS = ("resnet18", "yolov8")


app = FastAPI(title="Sugarcane Variety Classifier API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.PILToTensor()
loaded_model_type = "resnet18"
loaded_models: dict[str, Any | None] = {model_type: None for model_type in SUPPORTED_MODELS}
model_classes: dict[str, list[str]] = {model_type: [] for model_type in SUPPORTED_MODELS}
model_image_sizes: dict[str, int] = {model_type: 224 for model_type in SUPPORTED_MODELS}
model_load_errors: dict[str, str | None] = {model_type: None for model_type in SUPPORTED_MODELS}
model_checkpoints: dict[str, Path] = {}
training_lock = threading.Lock()
training_status: dict[str, Any] = {
    "state": "idle",
    "message": "No training job has been started.",
}
report_lock = threading.Lock()
current_training_report: dict[str, Any] = {
    "state": "idle",
    "events": [],
    "epoch_history": [],
}


class TrainingRequest(BaseModel):
    model_type: str = "resnet18"
    raw_dir: str = "content/data/raw/DATASETSFINAL"
    prepared_dir: str = "content/data/prepared"
    output_dir: str = "content/data/sugarcane_artifacts"
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    resize: int | None = 256
    epochs: int = Field(default=35, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=5e-4, ge=0.0)
    image_size: int = Field(default=224, ge=1)
    workers: int = Field(default=8, ge=0)
    seed: int = 42
    augment_validation: bool = False
    noise_std: float = Field(default=DEFAULT_TRAIN_NOISE_STD, ge=0.0)
    blur_prob: float = Field(default=DEFAULT_TRAIN_BLUR_PROB, ge=0.0, le=1.0)
    erase_prob: float = Field(default=DEFAULT_TRAIN_ERASE_PROB, ge=0.0, le=1.0)
    rotation_degrees: float = Field(default=DEFAULT_TRAIN_ROTATION_DEGREES, ge=0.0)
    early_stopping_patience: int = Field(default=8, ge=0)
    early_stopping_min_delta: float = Field(default=0.002, ge=0.0)
    use_class_weights: bool = True
    use_balanced_sampler: bool = DEFAULT_USE_BALANCED_SAMPLER
    label_mode: str = "variety_maturity"
    preprocess_device: str = "auto"
    preprocess_workers: int = Field(default=8, ge=1)
    perform_preprocess: bool = True
    yolo_weights: str = "yolov8n-cls.pt"


class ModelLoadRequest(BaseModel):
    checkpoint_path: str
    model_type: str | None = None


def _open_uploaded_image(contents: bytes) -> tuple[Image.Image, torch.Tensor]:
    try:
        with Image.open(io.BytesIO(contents)) as img:
            pil_image = img.convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    return pil_image, to_tensor(pil_image)


def _model_checkpoint_path(model_type: str) -> Path:
    if model_type == "yolov8":
        env_name = "MODEL_CHECKPOINT_YOLO"
        default_path = DEFAULT_YOLO_CHECKPOINT_PATH
    else:
        env_name = "MODEL_CHECKPOINT_RESNET"
        default_path = DEFAULT_RESNET_CHECKPOINT_PATH
    return Path(os.getenv(env_name, default_path)).expanduser().resolve()


def _checkpoint_path() -> Path:
    return _model_checkpoint_path("resnet18")


def _best_model_type_from_training_report() -> str | None:
    report_path = _artifacts_dir() / "full_training_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    best_result = report.get("best_result")
    if not isinstance(best_result, dict):
        return None
    model_type = str(best_result.get("model_type", ""))
    if model_type not in SUPPORTED_MODELS:
        return None
    return model_type


def _active_checkpoint_path() -> Path:
    if loaded_models.get(loaded_model_type) is not None:
        return model_checkpoints.get(loaded_model_type, _model_checkpoint_path(loaded_model_type))
    return _checkpoint_path()


def _predict_probabilities(image: torch.Tensor, model_type: str) -> torch.Tensor:
    selected_model = loaded_models.get(model_type)
    if selected_model is None:
        raise HTTPException(status_code=503, detail=f"{model_type} model is not loaded yet.")

    if model_type == "yolov8":
        try:
            result = selected_model(_tensor_to_pil(image), verbose=False)[0]
            probs = result.probs.data
            return probs.detach().to(device="cpu")
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"YOLOv8 prediction failed: {type(exc).__name__}: {exc}",
            )

    with torch.no_grad():
        batch = _prepare_images_on_device(
            [image],
            device=device,
            image_size=model_image_sizes[model_type],
            training=False,
        )
        if model_type == "resnet18_two_head":
            variety_logits, maturity_logits = selected_model(batch)
            class_to_variety_idx = torch.tensor(
                getattr(selected_model, "class_to_variety_idx"),
                device=device,
                dtype=torch.long,
            )
            class_to_maturity_idx = torch.tensor(
                getattr(selected_model, "class_to_maturity_idx"),
                device=device,
                dtype=torch.long,
            )
            logits = _combine_two_head_logits(
                variety_logits=variety_logits,
                maturity_logits=maturity_logits,
                class_to_variety_idx=class_to_variety_idx,
                class_to_maturity_idx=class_to_maturity_idx,
            )
        else:
            logits = selected_model(batch)
        return F.softmax(logits, dim=1)[0]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(array)


def _build_predictions(
    probabilities: torch.Tensor,
    top_k: int,
    class_names: list[str],
) -> list[dict[str, Any]]:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    if not class_names:
        raise HTTPException(status_code=503, detail="Model class labels are not loaded.")
    top_k = min(top_k, len(class_names))

    scores, indexes = torch.topk(probabilities, k=top_k)
    predictions = []
    for score, index in zip(scores.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
        class_name = class_names[index]
        predictions.append(
            {
                "class_index": index,
                "confidence": score,
                **_decode_class_name(class_name),
            }
        )
    return predictions


def _maturity_reason(probabilities: torch.Tensor, class_names: list[str]) -> dict[str, Any]:
    maturity_scores: dict[str, float] = {}
    for index, class_name in enumerate(class_names):
        decoded = _decode_class_name(class_name)
        maturity_status = decoded["maturity_status"] or "unknown"
        maturity_scores[maturity_status] = (
            maturity_scores.get(maturity_status, 0.0)
            + float(probabilities[index].detach().cpu().item())
        )

    ranked = sorted(maturity_scores.items(), key=lambda item: item[1], reverse=True)
    predicted_maturity, predicted_score = ranked[0]
    next_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "maturity_status": predicted_maturity,
        "maturity_probability": predicted_score,
        "next_best_probability": next_score,
        "margin": predicted_score - next_score,
        "maturity_probabilities": maturity_scores,
        "reason": (
            "maturity_status is selected by summing softmax probabilities for all "
            "classes with the same decoded maturity label and taking the highest "
            "marginal maturity probability."
        ),
    }


def _draw_prediction_overlay(
    image: Image.Image,
    maturity_reason: dict[str, Any],
) -> Image.Image:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated, "RGBA")

    left = 0
    top = 0
    right = annotated.width - 1
    bottom = annotated.height - 1

    maturity_status = str(maturity_reason["maturity_status"]).lower()
    if maturity_status == "mature":
        color = (31, 180, 90, 255)
    elif maturity_status in {"not_mature", "not mature", "immature"}:
        color = (220, 55, 55, 255)
    else:
        color = (255, 210, 65, 255)

    line_width = max(3, min(12, min(annotated.width, annotated.height) // 80))
    draw.rectangle((left, top, right, bottom), outline=color, width=line_width)

    return annotated

def _artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", DEFAULT_ARTIFACTS_DIR)).expanduser().resolve()


def _model_artifacts_dir(model_type: str) -> Path:
    base_dir = _artifacts_dir()
    if model_type == "yolov8":
        return base_dir / "yolov8"
    return base_dir / "resnet18"


def _build_artifacts_zip(artifacts_dir: Path) -> io.BytesIO:
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Artifacts folder not found: {artifacts_dir}",
        )

    artifact_files = sorted(
        path
        for path in artifacts_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in ARTIFACT_EXTENSIONS
    )
    if not artifact_files:
        raise HTTPException(
            status_code=404,
            detail=f"No .pt or .json artifact files found in: {artifacts_dir}",
        )

    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in artifact_files:
            archive.write(path, arcname=path.relative_to(artifacts_dir))
    output.seek(0)
    return output


def _summary_to_dict(summary: Any) -> dict[str, Any] | None:
    if summary is None:
        return None
    if is_dataclass(summary):
        return asdict(summary)
    if isinstance(summary, dict):
        return summary
    return {"value": str(summary)}


def _set_training_status(**updates: Any) -> None:
    training_status.update(updates)


def _set_report_state(**updates: Any) -> None:
    with report_lock:
        current_training_report.update(updates)


def _record_training_event(event: dict[str, Any]) -> None:
    timestamped = {
        "timestamp": time.time(),
        **event,
    }
    with report_lock:
        current_training_report["state"] = training_status.get("state", "running")
        current_training_report["model_type"] = event.get(
            "model_type",
            current_training_report.get("model_type"),
        )
        current_training_report["last_event"] = timestamped
        events = current_training_report.setdefault("events", [])
        events.append(timestamped)
        del events[:-100]

        if event.get("event") == "epoch_completed":
            epoch_history = current_training_report.setdefault("epoch_history", [])
            epoch_history.append(timestamped)
            del epoch_history[:-200]
        if event.get("event") == "dataset_analyzed":
            dataset_summary = event.get("dataset_summary")
            if isinstance(dataset_summary, dict):
                current_training_report["dataset_summary"] = dataset_summary
                _set_training_status(
                    dataset_summary=dataset_summary,
                    dataset_split_counts=dataset_summary.get("split_counts"),
                    dataset_total_images=dataset_summary.get("total_images"),
                    dataset_analysis_path=dataset_summary.get("summary_json_path"),
                )


def _read_json_file(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _find_report_file(artifacts_dir: Path, name: str) -> Path | None:
    candidate = artifacts_dir / name
    if candidate.exists():
        return candidate
    matches = sorted(artifacts_dir.rglob(name))
    return matches[0] if matches else None


def _prepared_dir() -> Path:
    return Path(os.getenv("PREPARED_DIR", DEFAULT_PREPARED_DIR)).expanduser().resolve()


def _full_training_report() -> dict[str, Any]:
    report_path = _artifacts_dir() / "full_training_report.json"
    payload = _read_json_file(report_path) if report_path.exists() else None
    return payload if isinstance(payload, dict) else {}


def _dataset_analysis_from_report(report: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
    results = report.get("results")
    if not isinstance(results, list):
        return None

    for result in results:
        if not isinstance(result, dict):
            continue
        analysis = result.get("split_analysis")
        if isinstance(analysis, dict) and analysis:
            source = str(result.get("prepared_dir") or analysis.get("prepared_dir") or "full_training_report.json")
            return analysis, f"embedded in full_training_report.json ({source})"
    return None


def _candidate_prepared_dirs(report: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []

    def add_candidate(value: Any) -> None:
        if not value:
            return
        try:
            path = Path(str(value)).expanduser().resolve()
        except OSError:
            return
        if path not in candidates:
            candidates.append(path)

    with report_lock:
        live_request = current_training_report.get("request")
    if isinstance(live_request, dict):
        add_candidate(live_request.get("prepared_dir"))

    status_request = training_status.get("request")
    if isinstance(status_request, dict):
        add_candidate(status_request.get("prepared_dir"))

    settings = report.get("settings")
    if isinstance(settings, dict):
        add_candidate(settings.get("prepared_dir"))

    results = report.get("results")
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            add_candidate(result.get("prepared_dir"))
            split_analysis = result.get("split_analysis")
            if isinstance(split_analysis, dict):
                add_candidate(split_analysis.get("prepared_dir"))

    add_candidate(os.getenv("PREPARED_DIR"))
    add_candidate(DEFAULT_PREPARED_DIR)
    return candidates


def _count_dataset_images(class_dir: Path) -> int:
    return sum(
        1
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in DATASET_IMAGE_EXTENSIONS
    )


def _analyze_prepared_dataset_directory(
    prepared_dir: Path,
    low_sample_threshold: int = 20,
) -> dict[str, Any] | None:
    if not prepared_dir.exists() or not prepared_dir.is_dir():
        return None

    split_names = ("train", "val", "test")
    split_counts: dict[str, int] = {}
    class_counts_by_split: dict[str, dict[str, int]] = {}
    overall_class_counts: dict[str, int] = {}
    variety_counts_by_split: dict[str, dict[str, int]] = {}
    maturity_counts_by_split: dict[str, dict[str, int]] = {}

    total_images = 0
    for split_name in split_names:
        split_dir = prepared_dir / split_name
        split_class_counts: dict[str, int] = {}
        split_variety_counts: dict[str, int] = {}
        split_maturity_counts: dict[str, int] = {}
        split_total = 0

        if split_dir.exists() and split_dir.is_dir():
            for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
                image_count = _count_dataset_images(class_dir)
                if image_count <= 0:
                    continue

                class_name = class_dir.name
                split_class_counts[class_name] = image_count
                overall_class_counts[class_name] = overall_class_counts.get(class_name, 0) + image_count
                split_total += image_count

                if "__" in class_name:
                    variety_name, maturity_name = class_name.split("__", 1)
                    split_variety_counts[variety_name] = split_variety_counts.get(variety_name, 0) + image_count
                    split_maturity_counts[maturity_name] = split_maturity_counts.get(maturity_name, 0) + image_count

        split_counts[split_name] = split_total
        class_counts_by_split[split_name] = split_class_counts
        variety_counts_by_split[split_name] = split_variety_counts
        maturity_counts_by_split[split_name] = split_maturity_counts
        total_images += split_total

    if total_images <= 0:
        return None

    class_count_values = list(overall_class_counts.values())
    max_class_count = max(class_count_values) if class_count_values else 0
    min_class_count = min(class_count_values) if class_count_values else 0
    class_distribution_ratio = (
        float(min_class_count) / float(max_class_count) if max_class_count > 0 else 0.0
    )

    low_sample_warnings: list[dict[str, Any]] = []
    val_test_floor = max(1, low_sample_threshold // 3)
    for class_name in sorted(overall_class_counts):
        split_details = {
            split_name: class_counts_by_split.get(split_name, {}).get(class_name, 0)
            for split_name in split_names
        }
        total_for_class = overall_class_counts[class_name]
        warning_reasons: list[str] = []
        if split_details["train"] < low_sample_threshold:
            warning_reasons.append("low_train_samples")
        if split_details["val"] < val_test_floor:
            warning_reasons.append("low_val_samples")
        if split_details["test"] < val_test_floor:
            warning_reasons.append("low_test_samples")
        if total_for_class < (low_sample_threshold * 2):
            warning_reasons.append("low_total_samples")
        if warning_reasons:
            low_sample_warnings.append(
                {
                    "class_name": class_name,
                    "counts": split_details,
                    "total": total_for_class,
                    "reasons": warning_reasons,
                }
            )

    return {
        "prepared_dir": str(prepared_dir),
        "total_images": total_images,
        "split_counts": split_counts,
        "class_counts_by_split": class_counts_by_split,
        "overall_class_counts": overall_class_counts,
        "class_distribution_ratio": class_distribution_ratio,
        "low_sample_threshold": low_sample_threshold,
        "low_sample_warnings": low_sample_warnings,
        "variety_counts_by_split": variety_counts_by_split,
        "maturity_counts_by_split": maturity_counts_by_split,
        "computed_at_runtime": True,
    }


def _load_dataset_analysis() -> tuple[dict[str, Any], str, dict[str, Any]]:
    report = _full_training_report()
    with report_lock:
        live_dataset_summary = current_training_report.get("dataset_summary")
    if isinstance(live_dataset_summary, dict) and live_dataset_summary:
        return live_dataset_summary, "current runtime training report", report

    explicit_path = os.getenv("DATASET_ANALYSIS_PATH")
    candidate_paths: list[Path] = []
    if explicit_path:
        candidate_paths.append(Path(explicit_path).expanduser().resolve())
    candidate_paths.append(_prepared_dir() / "prepared_dataset_analysis.json")

    artifact_match = _find_report_file(_artifacts_dir(), "prepared_dataset_analysis.json")
    if artifact_match is not None:
        candidate_paths.append(artifact_match)

    seen: set[Path] = set()
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not candidate.exists():
            continue
        payload = _read_json_file(candidate)
        if isinstance(payload, dict):
            return payload, str(candidate), report

    from_report = _dataset_analysis_from_report(report)
    if from_report is not None:
        analysis, source = from_report
        return analysis, source, report

    for prepared_dir in _candidate_prepared_dirs(report):
        analysis = _analyze_prepared_dataset_directory(prepared_dir)
        if analysis is not None:
            return analysis, f"computed from prepared dataset folder ({prepared_dir})", report

    return {}, "", report



def _normalize_metric_key(key: str) -> str:
    return "".join(ch.lower() for ch in key if ch.isalnum())


def _extract_csv_float(row: dict[str, str], candidates: list[str]) -> float | None:
    normalized = {_normalize_metric_key(key): value for key, value in row.items()}
    for candidate in candidates:
        raw = normalized.get(_normalize_metric_key(candidate))
        if raw in (None, ""):
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return None


def _load_epoch_history_from_results_csv(artifacts_dir: Path) -> list[dict[str, Any]]:
    results_path = _find_report_file(artifacts_dir, "results.csv")
    if results_path is None:
        return []

    history: list[dict[str, Any]] = []
    try:
        with results_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            best_val_acc = 0.0
            for index, row in enumerate(reader, start=1):
                epoch = _extract_csv_float(row, ["epoch"])
                val_acc = _extract_csv_float(
                    row,
                    [
                        "metrics/accuracy_top1",
                        "metrics/top1",
                        "val/acc_top1",
                        "accuracy_top1",
                        "top1",
                    ],
                )
                if val_acc is not None:
                    best_val_acc = max(best_val_acc, val_acc)
                history.append(
                    {
                        "epoch": int(epoch) + 1 if epoch is not None else index,
                        "train_loss": _extract_csv_float(row, ["train/loss", "train_loss"]),
                        "train_acc": _extract_csv_float(row, ["train/acc", "train_acc", "metrics/accuracy_top1"]),
                        "val_loss": _extract_csv_float(row, ["val/loss", "val_loss"]),
                        "val_acc": val_acc,
                        "best_val_acc": best_val_acc if val_acc is not None else None,
                        "lr": _extract_csv_float(row, ["lr/pg0", "lr0", "lr"]),
                    }
                )
    except OSError:
        return []
    return history


def _collect_visual_assets(artifacts_dir: Path) -> list[Path]:
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return []
    return sorted(
        path
        for path in artifacts_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VISUAL_EXTENSIONS
    )[:24]


def _format_metric_value(value: Any) -> str:
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value:.2%}"
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    if isinstance(value, list):
        return str(len(value))
    return str(value)


def _render_definition_rows(data: dict[str, Any], keys: list[tuple[str, str]]) -> str:
    rows: list[str] = []
    for label, key in keys:
        if key not in data:
            continue
        rows.append(
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(_format_metric_value(data.get(key)))}</td></tr>"
        )
    if not rows:
        return "<tr><td colspan='2'>No metrics available.</td></tr>"
    return "".join(rows)


def _split_class_label(class_name: str) -> tuple[str, str]:
    decoded = _decode_class_name(class_name)
    variety = decoded.get("variety") or "n/a"
    maturity = decoded.get("maturity_status") or "n/a"
    return variety, maturity


def _format_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _coerce_count(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _display_split_name(split_name: str) -> str:
    return {
        "train": "Training",
        "val": "Validation",
        "test": "Testing",
    }.get(split_name, split_name)


def _render_count_rows(
    counts_by_split: dict[str, dict[str, Any]],
    totals: dict[str, Any],
) -> str:
    if not totals:
        return "<tr><td colspan='7'>No dataset count details are available.</td></tr>"

    rows: list[str] = []
    for class_name in sorted(totals):
        variety, maturity = _split_class_label(class_name)
        rows.append(
            "<tr>"
            f"<td>{html.escape(class_name)}</td>"
            f"<td>{html.escape(variety)}</td>"
            f"<td>{html.escape(maturity)}</td>"
            f"<td>{html.escape(_format_count(counts_by_split.get('train', {}).get(class_name, 0)))}</td>"
            f"<td>{html.escape(_format_count(counts_by_split.get('val', {}).get(class_name, 0)))}</td>"
            f"<td>{html.escape(_format_count(counts_by_split.get('test', {}).get(class_name, 0)))}</td>"
            f"<td>{html.escape(_format_count(totals.get(class_name, 0)))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_rollup_rows(counts_by_split: dict[str, dict[str, Any]]) -> str:
    labels = sorted(
        {
            label
            for split_counts in counts_by_split.values()
            if isinstance(split_counts, dict)
            for label in split_counts
        }
    )
    if not labels:
        return "<tr><td colspan='5'>No rollup counts are available.</td></tr>"

    rows: list[str] = []
    for label in labels:
        train = counts_by_split.get("train", {}).get(label, 0)
        val = counts_by_split.get("val", {}).get(label, 0)
        test = counts_by_split.get("test", {}).get(label, 0)
        total = _coerce_count(train) + _coerce_count(val) + _coerce_count(test)
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(label))}</td>"
            f"<td>{html.escape(_format_count(train))}</td>"
            f"<td>{html.escape(_format_count(val))}</td>"
            f"<td>{html.escape(_format_count(test))}</td>"
            f"<td>{html.escape(_format_count(total))}</td>"
            "</tr>"
        )
    return "".join(rows)


def _render_low_sample_warnings(warnings: Any) -> str:
    if not isinstance(warnings, list) or not warnings:
        return "<p class='muted'>No low-sample warnings were recorded for the configured threshold.</p>"

    rows: list[str] = []
    for warning in warnings[:12]:
        if not isinstance(warning, dict):
            continue
        counts = warning.get("counts") if isinstance(warning.get("counts"), dict) else {}
        reasons = warning.get("reasons")
        reason_text = ", ".join(str(item) for item in reasons) if isinstance(reasons, list) else str(reasons or "n/a")
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(warning.get('class_name', 'n/a')))}</td>"
            f"<td>{html.escape(_format_count(counts.get('train', 0)))}</td>"
            f"<td>{html.escape(_format_count(counts.get('val', 0)))}</td>"
            f"<td>{html.escape(_format_count(counts.get('test', 0)))}</td>"
            f"<td>{html.escape(_format_count(warning.get('total', 0)))}</td>"
            f"<td>{html.escape(reason_text)}</td>"
            "</tr>"
        )
    if not rows:
        return "<p class='muted'>No low-sample warnings were recorded for the configured threshold.</p>"
    return (
        "<table><thead><tr><th>Class</th><th>Training</th><th>Validation</th><th>Testing</th><th>Total</th><th>Reason</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_dataset_documentation_section() -> str:
    analysis, source, report = _load_dataset_analysis()
    settings = report.get("settings") if isinstance(report.get("settings"), dict) else {}

    if not analysis:
        expected_path = _prepared_dir() / "prepared_dataset_analysis.json"
        return f"""
        <article class="card">
          <h2>Dataset and Image Summary</h2>
          <p class="muted">
            No prepared dataset analysis was found. Generate it with
            <code>python main.py analyze-prepared --prepared-dir {html.escape(str(_prepared_dir()))}</code>
            or place a JSON report at <code>{html.escape(str(expected_path))}</code>.
          </p>
        </article>
        """

    split_counts = analysis.get("split_counts") if isinstance(analysis.get("split_counts"), dict) else {}
    class_counts_by_split = (
        analysis.get("class_counts_by_split")
        if isinstance(analysis.get("class_counts_by_split"), dict)
        else {}
    )
    overall_class_counts = (
        analysis.get("overall_class_counts")
        if isinstance(analysis.get("overall_class_counts"), dict)
        else {}
    )
    variety_counts_by_split = (
        analysis.get("variety_counts_by_split")
        if isinstance(analysis.get("variety_counts_by_split"), dict)
        else {}
    )
    maturity_counts_by_split = (
        analysis.get("maturity_counts_by_split")
        if isinstance(analysis.get("maturity_counts_by_split"), dict)
        else {}
    )
    low_sample_warnings = analysis.get("low_sample_warnings")

    total_images = analysis.get("total_images")
    prepared_dir = analysis.get("prepared_dir") or settings.get("prepared_dir") or str(_prepared_dir())
    raw_dir = report.get("raw_dir") or "n/a"
    label_mode = settings.get("label_mode") or "n/a"
    preprocess_resize = settings.get("preprocess_resize")
    image_size = settings.get("image_size")
    class_count = len(overall_class_counts)
    distribution_ratio = analysis.get("class_distribution_ratio")
    distribution_text = _format_metric_value(distribution_ratio) if distribution_ratio is not None else "n/a"

    split_summary_rows = "".join(
        "<tr>"
        f"<td>{html.escape(_display_split_name(split_name))}</td>"
        f"<td>{html.escape(_format_count(count))}</td>"
        "</tr>"
        for split_name, count in split_counts.items()
    ) or "<tr><td colspan='2'>No split counts are available.</td></tr>"

    return f"""
    <section class="dataset-section">
      <div class="section-header">
        <h2>Dataset and Image Summary</h2>
      </div>
      <p class="muted">
        This section describes the prepared image dataset used for the documented training run.
        Counts are loaded from {html.escape(source or 'the prepared dataset analysis report')}.
      </p>
      <div class="doc-grid">
        <article class="card">
          <h3>Dataset Configuration</h3>
          <table><tbody>
            <tr><th>Raw Directory</th><td>{html.escape(str(raw_dir))}</td></tr>
            <tr><th>Prepared Directory</th><td>{html.escape(str(prepared_dir))}</td></tr>
            <tr><th>Label Mode</th><td>{html.escape(str(label_mode))}</td></tr>
            <tr><th>Total Images</th><td>{html.escape(_format_count(total_images))}</td></tr>
            <tr><th>Class Count</th><td>{html.escape(_format_count(class_count))}</td></tr>
            <tr><th>Class Balance Ratio</th><td>{html.escape(distribution_text)}</td></tr>
            <tr><th>Preprocess Resize</th><td>{html.escape(_format_metric_value(preprocess_resize))}</td></tr>
            <tr><th>Model Image Size</th><td>{html.escape(_format_metric_value(image_size))}</td></tr>
          </tbody></table>
        </article>
        <article class="card">
          <h3>Split Totals</h3>
          <table><thead><tr><th>Dataset Section</th><th>Number of Images</th></tr></thead><tbody>{split_summary_rows}</tbody></table>
        </article>
      </div>
      <article class="card">
        <h3>Class Image Counts</h3>
        <table>
          <thead>
            <tr><th>Class</th><th>Variety</th><th>Maturity</th><th>Training</th><th>Validation</th><th>Testing</th><th>Total</th></tr>
          </thead>
          <tbody>{_render_count_rows(class_counts_by_split, overall_class_counts)}</tbody>
        </table>
      </article>
      <div class="doc-grid">
        <article class="card">
          <h3>Variety Rollup</h3>
          <table><thead><tr><th>Variety</th><th>Training</th><th>Validation</th><th>Testing</th><th>Total</th></tr></thead>
          <tbody>{_render_rollup_rows(variety_counts_by_split)}</tbody></table>
        </article>
        <article class="card">
          <h3>Maturity Rollup</h3>
          <table><thead><tr><th>Maturity</th><th>Training</th><th>Validation</th><th>Testing</th><th>Total</th></tr></thead>
          <tbody>{_render_rollup_rows(maturity_counts_by_split)}</tbody></table>
        </article>
      </div>
      <article class="card">
        <h3>Low-Sample Warnings</h3>
        {_render_low_sample_warnings(low_sample_warnings)}
      </article>
    </section>
    """


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_series_value(item: dict[str, Any], key: str) -> float | None:
    if key in item:
        return _coerce_float(item.get(key))
    metrics = item.get("metrics")
    if isinstance(metrics, dict):
        for candidate in [key, key.replace("_", "/"), key.replace("_", "-"), key.replace("_", "")]:
            if candidate in metrics:
                return _coerce_float(metrics.get(candidate))
    return None


def _load_epoch_history(model_type: str, metrics: dict[str, Any]) -> list[dict[str, Any]]:
    with report_lock:
        live_model_type = str(current_training_report.get("model_type", ""))
        live_history = list(current_training_report.get("epoch_history", []))

    if live_model_type == model_type and live_history:
        return live_history

    history = metrics.get("epoch_history")
    if isinstance(history, list):
        return [item for item in history if isinstance(item, dict)]
    return _load_epoch_history_from_results_csv(_model_artifacts_dir(model_type))


def _render_epoch_history_table(model_type: str, metrics: dict[str, Any]) -> str:
    epoch_history = _load_epoch_history(model_type, metrics)

    if not epoch_history:
        return "<p class='muted'>No epoch history is available for this model.</p>"

    header = (
        "<tr><th>Epoch</th><th>Train Loss</th><th>Train Acc</th>"
        "<th>Val Loss</th><th>Val Acc</th><th>Best Val Acc</th><th>Learning Rate</th></tr>"
    )
    rows: list[str] = []
    for item in epoch_history[-20:]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('epoch', 'n/a')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'train_loss')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'train_acc')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'val_loss')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'val_acc')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'best_val_acc')))}</td>"
            f"<td>{html.escape(_format_metric_value(_extract_series_value(item, 'lr')))}</td>"
            "</tr>"
        )
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def _render_line_chart(
    epoch_history: list[dict[str, Any]],
    title: str,
    series_specs: list[tuple[str, str, str]],
) -> str:
    if not epoch_history:
        return "<p class='muted'>No chart data available.</p>"

    values_by_series: list[tuple[str, str, list[float]]] = []
    for label, key, color in series_specs:
        values = [
            value
            for item in epoch_history
            if (value := _extract_series_value(item, key)) is not None
        ]
        if values:
            values_by_series.append((label, color, values))

    if not values_by_series:
        return "<p class='muted'>No numeric chart series were found in the saved epoch history.</p>"

    all_values = [value for _, _, values in values_by_series for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    width = 760
    height = 280
    pad_left = 56
    pad_right = 18
    pad_top = 18
    pad_bottom = 36
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    total_points = max(len(epoch_history) - 1, 1)

    def point_xy(index: int, value: float) -> tuple[float, float]:
        x = pad_left + (plot_width * index / total_points)
        y = pad_top + plot_height - ((value - min_value) / (max_value - min_value) * plot_height)
        return x, y

    grid_lines = []
    for step in range(5):
        y = pad_top + plot_height * step / 4
        value = max_value - ((max_value - min_value) * step / 4)
        grid_lines.append(
            f"<line x1='{pad_left}' y1='{y:.1f}' x2='{width - pad_right}' y2='{y:.1f}' stroke='#d8cfbf' stroke-dasharray='4 4' />"
            f"<text x='8' y='{y + 4:.1f}' font-size='11' fill='#5f6b61'>{html.escape(_format_metric_value(value))}</text>"
        )

    series_paths: list[str] = []
    legend: list[str] = []
    for idx, (label, color, values) in enumerate(values_by_series):
        path_parts: list[str] = []
        for point_index, value in enumerate(values):
            x, y = point_xy(point_index, value)
            command = "M" if point_index == 0 else "L"
            path_parts.append(f"{command}{x:.1f},{y:.1f}")
        series_paths.append(
            f"<path d='{' '.join(path_parts)}' fill='none' stroke='{color}' stroke-width='2.6' stroke-linecap='round' stroke-linejoin='round' />"
        )
        legend.append(
            f"<span class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{html.escape(label)}</span>"
        )

    x_labels = []
    for point_index, item in enumerate(epoch_history):
        if point_index in {0, len(epoch_history) - 1} or point_index % max(len(epoch_history) // 6, 1) == 0:
            x, _ = point_xy(point_index, min_value)
            x_labels.append(
                f"<text x='{x:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f6b61'>E{html.escape(str(item.get('epoch', point_index + 1)))}</text>"
            )

    return (
        f"<div class='chart-card'><h4>{html.escape(title)}</h4>"
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"{''.join(grid_lines)}"
        f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#7d857d' />"
        f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' stroke='#7d857d' />"
        f"{''.join(series_paths)}"
        f"{''.join(x_labels)}"
        "</svg></div>"
    )


def _render_training_graphs(model_type: str, metrics: dict[str, Any]) -> str:
    epoch_history = _load_epoch_history(model_type, metrics)
    if not epoch_history:
        return "<p class='muted'>No saved epoch history is available for graph rendering.</p>"

    charts = [
        _render_line_chart(
            epoch_history,
            "Loss Across Epochs",
            [("Train Loss", "train_loss", "#8a3d2f"), ("Validation Loss", "val_loss", "#3a6b48")],
        ),
        _render_line_chart(
            epoch_history,
            "Accuracy Across Epochs",
            [("Train Accuracy", "train_acc", "#255f85"), ("Validation Accuracy", "val_acc", "#5e8c31"), ("Best Val Accuracy", "best_val_acc", "#d08b2b")],
        ),
        _render_line_chart(
            epoch_history,
            "Learning Rate Schedule",
            [("Learning Rate", "lr", "#7b4ec9")],
        ),
    ]
    return f"<div class='chart-grid'>{''.join(charts)}</div>"


def _render_multi_history_chart(
    histories: dict[str, list[dict[str, Any]]],
    title: str,
    series_specs: list[tuple[str, str, str, str]],
) -> str:
    available_histories = {
        model_type: history for model_type, history in histories.items() if history
    }
    if not available_histories:
        return "<p class='muted'>No cross-model chart data is available.</p>"

    max_epochs = max(len(history) for history in available_histories.values())
    width = 920
    height = 320
    pad_left = 56
    pad_right = 18
    pad_top = 18
    pad_bottom = 36
    plot_width = width - pad_left - pad_right
    plot_height = height - pad_top - pad_bottom
    total_points = max(max_epochs - 1, 1)

    available_series: list[tuple[str, str, str, list[float]]] = []
    for model_type, label, key, color in series_specs:
        history = available_histories.get(model_type, [])
        values = [
            value
            for item in history
            if (value := _extract_series_value(item, key)) is not None
        ]
        if values:
            available_series.append((label, color, key, values))

    if not available_series:
        return "<p class='muted'>Epoch history exists, but no comparable chart series were found.</p>"

    all_values = [value for _, _, _, values in available_series for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def point_xy(index: int, value: float) -> tuple[float, float]:
        x = pad_left + (plot_width * index / total_points)
        y = pad_top + plot_height - ((value - min_value) / (max_value - min_value) * plot_height)
        return x, y

    grid_lines = []
    for step in range(5):
        y = pad_top + plot_height * step / 4
        value = max_value - ((max_value - min_value) * step / 4)
        grid_lines.append(
            f"<line x1='{pad_left}' y1='{y:.1f}' x2='{width - pad_right}' y2='{y:.1f}' stroke='#d8cfbf' stroke-dasharray='4 4' />"
            f"<text x='8' y='{y + 4:.1f}' font-size='11' fill='#5f6b61'>{html.escape(_format_metric_value(value))}</text>"
        )

    legend: list[str] = []
    paths: list[str] = []
    for label, color, _key, values in available_series:
        path_parts: list[str] = []
        for point_index, value in enumerate(values):
            x, y = point_xy(point_index, value)
            command = "M" if point_index == 0 else "L"
            path_parts.append(f"{command}{x:.1f},{y:.1f}")
        paths.append(
            f"<path d='{' '.join(path_parts)}' fill='none' stroke='{color}' stroke-width='2.8' stroke-linecap='round' stroke-linejoin='round' />"
        )
        legend.append(
            f"<span class='legend-item'><span class='legend-swatch' style='background:{color}'></span>{html.escape(label)}</span>"
        )

    x_labels = []
    for point_index in range(max_epochs):
        if point_index in {0, max_epochs - 1} or point_index % max(max_epochs // 8, 1) == 0:
            x, _ = point_xy(point_index, min_value)
            x_labels.append(
                f"<text x='{x:.1f}' y='{height - 10}' font-size='11' text-anchor='middle' fill='#5f6b61'>E{point_index + 1}</text>"
            )

    return (
        f"<div class='chart-card'><h4>{html.escape(title)}</h4>"
        f"<div class='legend'>{''.join(legend)}</div>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)}'>"
        f"{''.join(grid_lines)}"
        f"<line x1='{pad_left}' y1='{height - pad_bottom}' x2='{width - pad_right}' y2='{height - pad_bottom}' stroke='#7d857d' />"
        f"<line x1='{pad_left}' y1='{pad_top}' x2='{pad_left}' y2='{height - pad_bottom}' stroke='#7d857d' />"
        f"{''.join(paths)}"
        f"{''.join(x_labels)}"
        "</svg></div>"
    )


def _render_comparison_graphs() -> str:
    histories: dict[str, list[dict[str, Any]]] = {}
    for model_type in REPORT_MODEL_SLOTS:
        payload = _build_model_doc_payload(model_type)
        history = _load_epoch_history(model_type, payload["metrics"])
        if history:
            histories[model_type] = history

    if not histories:
        return "<p class='muted'>No cross-model epoch history is available yet. Retrain with the updated code or keep YOLO results.csv in the artifact folder.</p>"
    charts = [
        _render_multi_history_chart(
            histories,
            "Combined Training Loss",
            [
                ("resnet18", "ResNet18 Train Loss", "train_loss", "#8a3d2f"),
                ("yolov8", "YOLOv8 Train Loss", "train_loss", "#255f85"),
            ],
        ),
        _render_multi_history_chart(
            histories,
            "Combined Validation Loss",
            [
                ("resnet18", "ResNet18 Validation Loss", "val_loss", "#3a6b48"),
                ("yolov8", "YOLOv8 Validation Loss", "val_loss", "#7b4ec9"),
            ],
        ),
        _render_multi_history_chart(
            histories,
            "Combined Training Accuracy",
            [
                ("resnet18", "ResNet18 Train Accuracy", "train_acc", "#d08b2b"),
                ("yolov8", "YOLOv8 Train Accuracy", "train_acc", "#5e8c31"),
            ],
        ),
        _render_multi_history_chart(
            histories,
            "Combined Validation Accuracy",
            [
                ("resnet18", "ResNet18 Validation Accuracy", "val_acc", "#3a6b48"),
                ("yolov8", "YOLOv8 Validation Accuracy", "val_acc", "#8a3d2f"),
                ("resnet18", "ResNet18 Best Validation Accuracy", "best_val_acc", "#7fb069"),
                ("yolov8", "YOLOv8 Best Validation Accuracy", "best_val_acc", "#d08b2b"),
            ],
        ),
    ]

    return (
        "<article class='card'>"
        "<h2>Cross-Model Training and Validation Comparison</h2>"
        "<p class='muted'>These shared charts normalize saved ResNet18 and YOLOv8 epoch history into comparable train/validation loss and accuracy views. YOLO falls back to results.csv when JSON epoch history is not available.</p>"
        f"<div class='chart-grid'>{''.join(charts)}</div>"
        "</article>"
    )


def _render_top_confusions(test_summary: dict[str, Any] | None) -> str:
    if not isinstance(test_summary, dict):
        return "<p class='muted'>No test summary available.</p>"
    confusions = test_summary.get("top_confusions")
    if not isinstance(confusions, list) or not confusions:
        return "<p class='muted'>No confusion pairs were recorded for this model.</p>"

    rows: list[str] = []
    for row in confusions[:10]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('true_class', 'n/a')))}</td>"
            f"<td>{html.escape(str(row.get('predicted_class', 'n/a')))}</td>"
            f"<td>{html.escape(str(row.get('count', 'n/a')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>True Class</th><th>Predicted Class</th><th>Count</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_per_class_table(test_summary: dict[str, Any] | None) -> str:
    if not isinstance(test_summary, dict):
        return "<p class='muted'>No per-class metrics are available.</p>"
    per_class = test_summary.get("per_class")
    if not isinstance(per_class, list) or not per_class:
        return "<p class='muted'>This model did not save per-class metrics.</p>"

    rows: list[str] = []
    for row in per_class[:12]:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('class_name', 'n/a')))}</td>"
            f"<td>{html.escape(_format_metric_value(row.get('accuracy')))}</td>"
            f"<td>{html.escape(str(row.get('support', 'n/a')))}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Class</th><th>Accuracy</th><th>Support</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _render_visual_gallery(model_type: str, artifacts_dir: Path) -> str:
    visuals = _collect_visual_assets(artifacts_dir)
    if not visuals:
        return "<p class='muted'>No image assets were found in this artifact folder.</p>"

    cards: list[str] = []
    for path in visuals:
        rel_path = path.relative_to(artifacts_dir)
        asset_url = f"/reports/assets/{model_type}/{quote(str(rel_path), safe='/')}"
        cards.append(
            "<figure class='visual-card'>"
            f"<img src='{asset_url}' alt='{html.escape(path.name)}' loading='lazy' />"
            f"<figcaption>{html.escape(str(rel_path))}</figcaption>"
            "</figure>"
        )
    return f"<div class='visual-grid'>{''.join(cards)}</div>"


def _build_model_doc_payload(model_type: str) -> dict[str, Any]:
    artifacts_dir = _model_artifacts_dir(model_type)
    metrics_path = _find_report_file(artifacts_dir, "metrics.json")
    test_summary_path = _find_report_file(artifacts_dir, "test_summary.json")
    android_metadata_path = _find_report_file(artifacts_dir, f"{model_type}_android_metadata.json")
    metadata = _read_json_file(metrics_path) if metrics_path else None
    test_summary = _read_json_file(test_summary_path) if test_summary_path else None
    android_metadata = _read_json_file(android_metadata_path) if android_metadata_path else None
    return {
        "model_type": model_type,
        "artifacts_dir": artifacts_dir,
        "metrics": metadata if isinstance(metadata, dict) else {},
        "test_summary": test_summary if isinstance(test_summary, dict) else {},
        "android_metadata": android_metadata if isinstance(android_metadata, dict) else {},
    }


def _extract_class_count_from_source(source: dict[str, Any]) -> int:
    for key in ("classes", "class_names", "names"):
        value = source.get(key)

        if isinstance(value, list):
            return len(value)
        
        if isinstance(value, dict):
            return len(value)

    per_class = source.get("per_class")
    if isinstance(per_class, list):
        class_names = {
            str(row.get("class_name"))
            for row in per_class
            if isinstance(row, dict) and row.get("class_name")
        }
        if class_names:
            return len(class_names)


    overall_class_counts = source.get("overall_class_counts")
    if isinstance(overall_class_counts, dict):
        return len(overall_class_counts)

    return 0

def _class_count_from_payload(payload: dict[str, Any], model_type: str) -> int:

    loaded_classes = model_classes.get(model_type, [])
    if loaded_classes:
        return len(loaded_classes)


    for key in ("android_metadata", "metrics", "test_summary"):
        source = payload.get(key)
        if not isinstance(source, dict):
            continue

        count = _extract_class_count_from_source(source)
        if count > 0:
            return count

    return 0


def _render_model_doc_section(model_type: str) -> str:
    payload = _build_model_doc_payload(model_type)
    metrics = payload["metrics"]
    test_summary = payload["test_summary"]
    android_metadata = payload["android_metadata"]
    artifacts_dir = payload["artifacts_dir"]
    displayed_model_type = str(metrics.get("model_type") or model_type)
    status = "Loaded" if loaded_models.get(model_type) is not None else "Unavailable"
    error_text = model_load_errors.get(model_type)

    metric_rows = _render_definition_rows(
        metrics,
        [
            ("Best Validation Accuracy", "best_val_acc"),
            ("Test Accuracy", "test_acc"),
            ("Epochs", "epochs"),
            ("Batch Size", "batch_size"),
            ("Learning Rate", "learning_rate"),
            ("Weight Decay", "weight_decay"),
            ("Image Size", "image_size"),
            ("Checkpoint Path", "checkpoint_path"),
        ],
    )
    test_rows = _render_definition_rows(
        test_summary,
        [
            ("Test Accuracy", "test_acc"),
            ("Variety Accuracy", "variety_acc"),
            ("Maturity Accuracy", "maturity_acc"),
            ("Samples", "num_samples"),
            ("Friendly Outcome", "friendly_outcome"),
        ],
    )
    android_rows = _render_definition_rows(
        android_metadata,
        [
            ("Android Artifact", "android_artifact_path"),
            ("ONNX Artifact", "onnx_artifact_path"),
            ("Image Size", "image_size"),
        ],
    )
    technical_rows = _render_definition_rows(
        metrics,
        [
            ("Device", "device"),
            ("Batch Size", "batch_size"),
            ("Epochs", "epochs"),
            ("Learning Rate", "learning_rate"),
            ("Weight Decay", "weight_decay"),
            ("Seed", "seed"),
            ("YOLO Base Weights", "yolo_weights"),
        ],
    )
    augmentation_rows = _render_definition_rows(
        metrics.get("augmentation", {}) if isinstance(metrics.get("augmentation"), dict) else {},
        [
            ("Validation Augmented", "augment_validation"),
            ("Train Resize Size", "train_resize_size"),
            ("Train Crop Size", "train_crop_size"),
            ("Validation Resize Size", "validation_resize_size"),
            ("Test Resize Size", "test_resize_size"),
            ("Noise Std", "noise_std"),
            ("Blur Probability", "blur_prob"),
            ("Erase Probability", "erase_prob"),
            ("Rotation Degrees", "rotation_degrees"),
            ("Crop Scale", "crop_scale"),
            ("Crop Ratio", "crop_ratio"),
            ("Horizontal Flip Probability", "horizontal_flip_prob"),
            ("Brightness Range", "brightness_range"),
            ("Contrast Range", "contrast_range"),
            ("Saturation Range", "saturation_range"),
            ("Hue Range", "hue_range"),
            ("Translate", "translate"),
            ("Scale", "scale"),
            ("Shear", "shear"),
            ("Perspective", "perspective"),
            ("Flip Left/Right", "fliplr"),
            ("Flip Up/Down", "flipud"),
            ("HSV Hue", "hsv_h"),
            ("HSV Saturation", "hsv_s"),
            ("HSV Value", "hsv_v"),
            ("Mosaic", "mosaic"),
            ("MixUp", "mixup"),
            ("Copy Paste", "copy_paste"),
            ("Close Mosaic", "close_mosaic"),
        ],
    )

    return f"""
    <section class="model-section">
      <div class="section-header">
        <h2>{html.escape(displayed_model_type.upper())}</h2>
        <span class="status {'loaded' if loaded_models.get(model_type) is not None else 'failed'}">{html.escape(status)}</span>
      </div>
      <p><strong>Checkpoint:</strong> {html.escape(str(model_checkpoints.get(model_type, _model_checkpoint_path(model_type))))}</p>
      <p><strong>Artifacts Directory:</strong> {html.escape(str(artifacts_dir))}</p>
      <p><strong>Load Error:</strong> {html.escape(error_text or 'none')}</p>
      <div class="doc-grid">
        <article class="card">
          <h3>Training Summary</h3>
          <table><tbody>{metric_rows}</tbody></table>
        </article>
        <article class="card">
          <h3>Evaluation Summary</h3>
          <table><tbody>{test_rows}</tbody></table>
        </article>
        <article class="card">
          <h3>Android / Deployment Metadata</h3>
          <table><tbody>{android_rows}</tbody></table>
        </article>
        <article class="card">
          <h3>Training Configuration</h3>
          <table><tbody>{technical_rows}</tbody></table>
        </article>
        <article class="card">
          <h3>Augmentation and Robustness Settings</h3>
          <table><tbody>{augmentation_rows}</tbody></table>
        </article>
      </div>
      <article class="card">
        <h3>Training Curves</h3>
        { _render_training_graphs(model_type, metrics) }
      </article>
      <article class="card">
        <h3>Training and Validation History</h3>
        { _render_epoch_history_table(model_type, metrics) }
      </article>
      <article class="card">
        <h3>Per-Class Performance</h3>
        { _render_per_class_table(test_summary) }
      </article>
      <article class="card">
        <h3>Top Confusions</h3>
        { _render_top_confusions(test_summary) }
      </article>
      <article class="card">
        <h3>Artifact Images</h3>
        { _render_visual_gallery(model_type, artifacts_dir) }
      </article>
    </section>
    """


def _build_technical_documentation_html() -> str:
    comparison_rows: list[str] = []
    for model_type in REPORT_MODEL_SLOTS:
        payload = _build_model_doc_payload(model_type)
        metrics = payload["metrics"]
        test_summary = payload["test_summary"]
        class_count = _class_count_from_payload(payload, model_type)
        displayed_model_type = str(metrics.get("model_type") or model_type)
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(displayed_model_type)}</td>"
            f"<td>{html.escape('loaded' if loaded_models.get(model_type) is not None else 'not loaded')}</td>"
            f"<td>{html.escape(_format_metric_value(metrics.get('best_val_acc')))}</td>"
            f"<td>{html.escape(_format_metric_value(test_summary.get('test_acc', metrics.get('test_acc'))))}</td>"
            f"<td>{html.escape(str(class_count))}</td>"
            f"<td>{html.escape(str(payload['artifacts_dir']))}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Sugarcane Model Technical Documentation</title>
  <style>
    :root {{
      --bg: #f3efe5;
      --panel: #fffaf2;
      --ink: #1f2a1f;
      --muted: #5f6b61;
      --line: #d8cfbf;
      --accent: #3a6b48;
      --accent-soft: #dfeadf;
      --warn: #8a3d2f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(58,107,72,0.12), transparent 22rem),
        linear-gradient(180deg, #f7f1e7 0%, var(--bg) 100%);
    }}
    main {{ max-width: 1200px; margin: 0 auto; padding: 32px 20px 48px; }}
    h1, h2, h3 {{ margin: 0 0 12px; line-height: 1.1; }}
    p {{ line-height: 1.6; }}
    .hero {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 28px;
      box-shadow: 0 18px 45px rgba(49, 58, 45, 0.08);
    }}
    .hero p {{ color: var(--muted); max-width: 75ch; }}
    .card {{
      background: rgba(255, 250, 242, 0.95);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      margin-top: 18px;
    }}
    .doc-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
      margin-top: 18px;
    }}
    .comparison-table, table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{ width: 32%; color: var(--muted); font-weight: 600; }}
    .comparison-table th, .comparison-table td {{ width: auto; }}
    .model-section {{ margin-top: 28px; }}
    .dataset-section {{ margin-top: 24px; }}
    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }}
    code {{
      background: #efe6d6;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 2px 5px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 0.92em;
    }}
    .status {{
      display: inline-flex;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 0.9rem;
      font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .status.loaded {{ background: var(--accent-soft); color: var(--accent); }}
    .status.failed {{ background: #f7d9d2; color: var(--warn); }}
    .muted {{ color: var(--muted); }}
    .visual-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
    }}
    .chart-card {{
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf8;
      padding: 14px;
    }}
    .chart-card h4 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .chart-card svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-bottom: 8px;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .legend-item {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .legend-swatch {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
    }}
    .visual-card {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow: hidden;
      background: #fff;
    }}
    .visual-card img {{
      width: 100%;
      display: block;
      aspect-ratio: 4 / 3;
      object-fit: cover;
      background: #ece3d4;
    }}
    .visual-card figcaption {{
      padding: 10px 12px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    @media (max-width: 720px) {{
      main {{ padding: 20px 14px 36px; }}
      .hero, .card {{ border-radius: 14px; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Sugarcane Variety Classifier Technical Documentation</h1>
      <p>
        This document compares the ResNet18 and YOLOv8 pipelines using the currently configured
        artifact folders. It summarizes training outcomes, validation and test performance,
        deployment-ready outputs, and any generated visual evidence stored in the artifact tree.
      </p>
      <article class="card">
        <h2>Model Comparison Overview</h2>
        <table class="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Status</th>
              <th>Best Val Acc</th>
              <th>Test Acc</th>
              <th>Artifact Classes</th>
              <th>Artifacts Directory</th>
            </tr>
          </thead>
          <tbody>{''.join(comparison_rows)}</tbody>
        </table>
      </article>
      {_render_dataset_documentation_section()}
      {_render_comparison_graphs()}
    </section>
    {_render_model_doc_section("resnet18")}
    {_render_model_doc_section("yolov8")}
  </main>
</body>
</html>"""


def _collect_report_jsons(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return []

    reports: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.rglob("*.json")):
        payload = _read_json_file(path)
        if payload is None:
            continue
        reports.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": str(path.relative_to(artifacts_dir)),
                "content": payload,
            }
        )
    return reports


def _artifact_file_index(artifacts_dir: Path) -> list[dict[str, Any]]:
    if not artifacts_dir.exists() or not artifacts_dir.is_dir():
        return []

    files: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in ARTIFACT_EXTENSIONS:
            continue
        try:
            size_bytes = path.stat().st_size
        except OSError:
            size_bytes = None
        files.append(
            {
                "name": path.name,
                "path": str(path),
                "relative_path": str(path.relative_to(artifacts_dir)),
                "suffix": path.suffix.lower(),
                "size_bytes": size_bytes,
            }
        )
    return files


def _build_current_report(artifacts_dir: Path | None = None) -> dict[str, Any]:
    with report_lock:
        live_report = {
            key: value
            for key, value in current_training_report.items()
        }
    selected_artifacts_dir = artifacts_dir
    if selected_artifacts_dir is None and live_report.get("artifacts_dir"):
        selected_artifacts_dir = Path(str(live_report["artifacts_dir"])).expanduser().resolve()
    if selected_artifacts_dir is None:
        selected_artifacts_dir = _artifacts_dir()
    dataset_analysis, dataset_source, _ = _load_dataset_analysis()

    return {
        "purpose": "Model comparison and research documentation",
        "generated_at_unix": time.time(),
        "loaded_model": {
            "model_type": loaded_model_type,
            "checkpoint": str(_active_checkpoint_path()),
            "model_loaded": loaded_models.get(loaded_model_type) is not None,
            "model_load_error": model_load_errors.get(loaded_model_type),
            "checkpoints": {
                model_type: str(model_checkpoints.get(model_type, _model_checkpoint_path(model_type)))
                for model_type in sorted(SUPPORTED_MODELS)
            },
            "models": {
                model_type: {
                    "model_loaded": loaded_models.get(model_type) is not None,
                    "model_load_error": model_load_errors.get(model_type),
                    "checkpoint": str(model_checkpoints.get(model_type, _model_checkpoint_path(model_type))),
                    "classes": [
                        {
                            "index": index,
                            **_decode_class_name(class_name),
                        }
                        for index, class_name in enumerate(model_classes[model_type])
                    ],
                    "image_size": model_image_sizes[model_type],
                }
                for model_type in sorted(SUPPORTED_MODELS)
            },
            "device": str(device),
        },
        "training": {
            "status": training_status,
            "live_report": live_report,
        },
        "dataset": {
            "source": dataset_source,
            "analysis": dataset_analysis,
        },
        "artifacts": {
            "dir": str(selected_artifacts_dir),
            "files": _artifact_file_index(selected_artifacts_dir),
            "json_reports": _collect_report_jsons(selected_artifacts_dir),
        },
    }


def _run_training_job(request: TrainingRequest) -> None:
    try:
        _set_report_state(
            state="running",
            model_type=request.model_type,
            request=request.dict(),
            artifacts_dir=request.output_dir,
            events=[],
            epoch_history=[],
            dataset_summary={},
        )
        _set_training_status(
            state="running",
            message="Training is running.",
            request=request.dict(),
            dataset_summary={},
            dataset_split_counts=None,
            dataset_total_images=None,
            dataset_analysis_path=None,
        )

        prep_summary, train_summary = run_all_for_colab(
            raw_dir=request.raw_dir,
            prepared_dir=request.prepared_dir,
            output_dir=request.output_dir,
            val_ratio=request.val_ratio,
            test_ratio=request.test_ratio,
            resize=request.resize,
            epochs=request.epochs,
            batch_size=request.batch_size,
            lr=request.lr,
            weight_decay=request.weight_decay,
            image_size=request.image_size,
            workers=request.workers,
            seed=request.seed,
            augment_validation=request.augment_validation,
            noise_std=request.noise_std,
            blur_prob=request.blur_prob,
            erase_prob=request.erase_prob,
            rotation_degrees=request.rotation_degrees,
            early_stopping_patience=request.early_stopping_patience,
            early_stopping_min_delta=request.early_stopping_min_delta,
            use_class_weights=request.use_class_weights,
            use_balanced_sampler=request.use_balanced_sampler,
            label_mode=request.label_mode,
            preprocess_device=request.preprocess_device,
            preprocess_workers=request.preprocess_workers,
            perform_preprocess=request.perform_preprocess,
            model_type=request.model_type,
            yolo_weights=request.yolo_weights,
            progress_callback=_record_training_event,
        )

        eval_summary = test_for_colab(
            prepared_dir=request.prepared_dir,
            checkpoint_path=train_summary.checkpoint_path,
            batch_size=request.batch_size,
            workers=request.workers,
            model_type=train_summary.model_type,
        )

        if train_summary.model_type == "yolov8":
            os.environ["MODEL_CHECKPOINT_YOLO"] = train_summary.checkpoint_path
        else:
            os.environ["MODEL_CHECKPOINT_RESNET"] = train_summary.checkpoint_path
        _load_model(
            checkpoint_path=Path(train_summary.checkpoint_path),
            model_type=train_summary.model_type,
        )

        with report_lock:
            dataset_summary = current_training_report.get("dataset_summary")
        dataset_summary = dataset_summary if isinstance(dataset_summary, dict) else {}

        _set_training_status(
            state="completed",
            message="Training completed. The API model was reloaded from the new checkpoint.",
            preprocess_summary=_summary_to_dict(prep_summary),
            train_summary=_summary_to_dict(train_summary),
            eval_summary=_summary_to_dict(eval_summary),
            dataset_summary=dataset_summary,
            dataset_split_counts=dataset_summary.get("split_counts"),
            dataset_total_images=dataset_summary.get("total_images"),
            dataset_analysis_path=dataset_summary.get("summary_json_path"),
            checkpoint_path=train_summary.checkpoint_path,
            model_type=train_summary.model_type,
            android_artifact_path=train_summary.android_artifact_path,
            onnx_artifact_path=train_summary.onnx_artifact_path,
            android_metadata_path=train_summary.android_metadata_path,
        )
        _set_report_state(
            state="completed",
            preprocess_summary=_summary_to_dict(prep_summary),
            train_summary=_summary_to_dict(train_summary),
            eval_summary=_summary_to_dict(eval_summary),
            dataset_summary=dataset_summary,
            checkpoint_path=train_summary.checkpoint_path,
            artifacts_dir=request.output_dir,
        )
    except Exception as exc:
        _set_training_status(
            state="failed",
            message="Training failed.",
            error=f"{type(exc).__name__}: {exc}",
        )
        _set_report_state(
            state="failed",
            error=f"{type(exc).__name__}: {exc}",
            artifacts_dir=request.output_dir,
        )
    finally:
        training_lock.release()


def _load_model(
    checkpoint_path: Path | None = None,
    model_type: str | None = None,
) -> None:
    global loaded_model_type

    requested_model_type = model_type
    if checkpoint_path is None and requested_model_type is not None:
        checkpoint_path = _model_checkpoint_path(requested_model_type)
    checkpoint_path = checkpoint_path or _checkpoint_path()
    requested_model_type = requested_model_type or _infer_model_type(checkpoint_path)
    if not checkpoint_path.exists():
        if requested_model_type == "yolov8":
            hint = "Train first, or set MODEL_CHECKPOINT_YOLO=/path/to/content/data/sugarcane_artifacts/yolov8/yolov8/weights/best.pt."
        else:
            hint = "Train first, or set MODEL_CHECKPOINT_RESNET=/path/to/content/data/sugarcane_artifacts/resnet18/best_model.pt."
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. {hint}")

    if requested_model_type == "yolov8":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            _raise_ultralytics_dependency_error("inference", exc)

        loaded_model = YOLO(str(checkpoint_path))
        yolo_names = getattr(loaded_model, "names", {})
        classes = [
            name
            for _, name in sorted(yolo_names.items(), key=lambda item: int(item[0]))
        ]
        model_classes["yolov8"] = classes
        model_image_sizes["yolov8"] = int(os.getenv("MODEL_IMAGE_SIZE_YOLO", os.getenv("MODEL_IMAGE_SIZE", "224")))
        loaded_models["yolov8"] = loaded_model
        loaded_model_type = "yolov8"
        model_load_errors["yolov8"] = None
        model_checkpoints["yolov8"] = checkpoint_path
        return

    if requested_model_type not in {"resnet18", "resnet18_two_head"}:
        raise ValueError("model_type must be 'resnet18', 'resnet18_two_head', or 'yolov8'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = list(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])

    if requested_model_type == "resnet18_two_head":
        loaded_model = _build_resnet18_two_head(
            num_varieties=len(checkpoint["varieties"]),
            num_maturities=len(checkpoint["maturities"]),
            pretrained=False,
        )
        setattr(loaded_model, "class_to_variety_idx", list(checkpoint["class_to_variety_idx"]))
        setattr(loaded_model, "class_to_maturity_idx", list(checkpoint["class_to_maturity_idx"]))
    else:
        loaded_model = _build_resnet18(num_classes=len(classes), pretrained=False)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    loaded_models[requested_model_type] = loaded_model
    loaded_model_type = requested_model_type
    model_classes[requested_model_type] = classes
    model_image_sizes[requested_model_type] = image_size
    model_load_errors[requested_model_type] = None
    model_checkpoints[requested_model_type] = checkpoint_path


def _infer_model_type(checkpoint_path: Path) -> str:
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            return str(checkpoint.get("model_type", "resnet18"))
    except Exception:
        pass
    return "yolov8"


@app.on_event("startup")
def startup() -> None:
    global loaded_model_type

    for model_type in sorted(SUPPORTED_MODELS):
        try:
            _load_model(model_type=model_type)
        except Exception as exc:
            loaded_models[model_type] = None
            model_classes[model_type] = []
            model_load_errors[model_type] = f"{type(exc).__name__}: {exc}"
    preferred_model_type = _best_model_type_from_training_report()
    if preferred_model_type and loaded_models.get(preferred_model_type) is not None:
        loaded_model_type = preferred_model_type
    elif loaded_models.get("resnet18_two_head") is not None:
        loaded_model_type = "resnet18_two_head"
    elif loaded_models.get("resnet18") is not None:
        loaded_model_type = "resnet18"
    elif loaded_models.get("yolov8") is not None:
        loaded_model_type = "yolov8"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": any(loaded_models.values()),
        "model_load_error": model_load_errors.get(loaded_model_type),
        "device": str(device),
        "checkpoint": str(_active_checkpoint_path()),
        "model_type": loaded_model_type,
        "models": {
            model_type: {
                "loaded": loaded_models.get(model_type) is not None,
                "checkpoint": str(model_checkpoints.get(model_type, _model_checkpoint_path(model_type))),
                "classes": len(model_classes[model_type]),
                "image_size": model_image_sizes[model_type],
                "error": model_load_errors.get(model_type),
            }
            for model_type in sorted(SUPPORTED_MODELS)
        },
    }


@app.get("/models")
def list_supported_models() -> dict[str, Any]:
    return {
        "supported_models": [
            {
                "model_type": "resnet18",
                "training_endpoint": "/training/start",
                "checkpoint": "PyTorch .pt",
                "android_exports": ["resnet18_android.ptl", "resnet18_android.onnx"],
            },
            {
                "model_type": "resnet18_two_head",
                "training_endpoint": "/training/start",
                "checkpoint": "PyTorch .pt",
                "android_exports": [],
            },
            {
                "model_type": "yolov8",
                "training_endpoint": "/training/start",
                "checkpoint": "Ultralytics YOLOv8 .pt",
                "android_exports": ["best.onnx"],
            },
        ],
        "loaded_model_type": loaded_model_type,
        "loaded_models": [
            model_type for model_type in sorted(SUPPORTED_MODELS) if loaded_models.get(model_type) is not None
        ],
    }


@app.get("/reports/current")
def current_model_report(artifacts_dir: str | None = None) -> dict[str, Any]:
    selected_artifacts_dir = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir is not None
        else None
    )
    return _build_current_report(artifacts_dir=selected_artifacts_dir)


@app.get("/reports/technical-documentation", response_class=HTMLResponse)
@app.get("/reports/model-comparison", response_class=HTMLResponse)
def technical_documentation() -> HTMLResponse:
    return HTMLResponse(_build_technical_documentation_html())


@app.get("/reports/assets/{model_type}/{asset_path:path}")
def report_asset(model_type: str, asset_path: str) -> FileResponse:
    if model_type not in SUPPORTED_MODELS:
        raise HTTPException(status_code=400, detail="model_type must be 'resnet18', 'resnet18_two_head', or 'yolov8'.")

    base_dir = _model_artifacts_dir(model_type).resolve()
    candidate = (base_dir / asset_path).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid asset path.") from exc
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"Asset not found: {asset_path}")
    return FileResponse(candidate)


@app.post("/models/load")
def load_model_endpoint(request: ModelLoadRequest) -> dict[str, Any]:
    if request.model_type is not None and request.model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'resnet18', 'resnet18_two_head', or 'yolov8'.",
        )
    checkpoint_path = Path(request.checkpoint_path).expanduser().resolve()
    _load_model(checkpoint_path=checkpoint_path, model_type=request.model_type)
    resolved_model_type = request.model_type or _infer_model_type(checkpoint_path)
    if resolved_model_type == "yolov8":
        os.environ["MODEL_CHECKPOINT_YOLO"] = str(checkpoint_path)
    else:
        os.environ["MODEL_CHECKPOINT_RESNET"] = str(checkpoint_path)
    return {
        "message": "Model loaded.",
        "checkpoint_path": str(checkpoint_path),
        "model_type": resolved_model_type,
        "classes": len(model_classes[resolved_model_type]),
        "image_size": model_image_sizes[resolved_model_type],
    }


@app.get("/classes")
def list_classes() -> dict[str, Any]:
    return {
        "models": {
            model_type: [
                {
                    "index": index,
                    **_decode_class_name(class_name),
                }
                for index, class_name in enumerate(model_classes[model_type])
            ]
            for model_type in sorted(SUPPORTED_MODELS)
        }
    }


@app.get("/artifacts/download")
def download_artifacts() -> StreamingResponse:
    artifacts_dir = _artifacts_dir()
    output = _build_artifacts_zip(artifacts_dir)
    headers = {
        "Content-Disposition": 'attachment; filename="sugarcane_artifacts.zip"',
        "X-Artifacts-Dir": str(artifacts_dir),
    }
    return StreamingResponse(output, media_type="application/zip", headers=headers)


@app.post("/training/start", status_code=202)
def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    if request.val_ratio + request.test_ratio >= 1:
        raise HTTPException(
            status_code=400,
            detail="Use ratios where val_ratio + test_ratio is less than 1.",
        )
    if request.label_mode not in {"variety", "maturity", "variety_maturity"}:
        raise HTTPException(
            status_code=400,
            detail="label_mode must be 'variety', 'maturity', or 'variety_maturity'.",
        )
    if request.preprocess_device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(
            status_code=400,
            detail="preprocess_device must be 'auto', 'cuda', or 'cpu'.",
        )
    if request.model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'resnet18', 'resnet18_two_head', or 'yolov8'.",
        )

    if not training_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="A training job is already running.",
        )

    _set_training_status(
        state="queued",
        message="Training job was accepted and will start in the background.",
        request=request.dict(),
        dataset_summary={},
        dataset_split_counts=None,
        dataset_total_images=None,
        dataset_analysis_path=None,
    )
    background_tasks.add_task(_run_training_job, request)
    return {
        "state": "queued",
        "message": "Training job accepted. Poll /training/status for progress.",
    }


@app.get("/training/status")
def get_training_status() -> dict[str, Any]:
    return training_status


@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 3) -> dict[str, Any]:
    contents = await file.read()
    _, image = _open_uploaded_image(contents)
    results: dict[str, Any] = {}
    for model_type in sorted(SUPPORTED_MODELS):
        if loaded_models.get(model_type) is None:
            continue
        probabilities = _predict_probabilities(image, model_type=model_type)
        predictions = _build_predictions(
            probabilities,
            top_k=top_k,
            class_names=model_classes[model_type],
        )
        results[model_type] = {
            "prediction": predictions[0],
            "top_k": predictions,
        }

    if not results:
        raise HTTPException(status_code=503, detail="No models are loaded yet.")

    return {
        "filename": file.filename,
        "models": results,
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...), top_k: int = 3) -> StreamingResponse:
    contents = await file.read()
    pil_image, image = _open_uploaded_image(contents)
    reason_model_type = "resnet18" if loaded_models.get("resnet18") is not None else "yolov8"
    if loaded_models.get(reason_model_type) is None:
        raise HTTPException(status_code=503, detail="No models are loaded yet.")
    probabilities = _predict_probabilities(image, model_type=reason_model_type)
    predictions = _build_predictions(
        probabilities,
        top_k=top_k,
        class_names=model_classes[reason_model_type],
    )
    reason = _maturity_reason(probabilities, class_names=model_classes[reason_model_type])

    annotated = _draw_prediction_overlay(
        image=pil_image,
        maturity_reason=reason,
    )
    output = io.BytesIO()
    annotated.save(output, format="PNG")
    output.seek(0)

    headers = {
        "X-Model-Type": reason_model_type,
        "X-Predicted-Class": str(predictions[0]["class_name"]),
        "X-Predicted-Variety": str(predictions[0]["variety"]),
        "X-Predicted-Maturity": str(reason["maturity_status"]),
        "X-Maturity-Probability": f"{reason['maturity_probability']:.6f}",
        "X-Next-Best-Maturity-Probability": f"{reason['next_best_probability']:.6f}",
        "X-Maturity-Margin": f"{reason['margin']:.6f}",
        "X-Maturity-Reason": str(reason["reason"]),
    }
    return StreamingResponse(output, media_type="image/png", headers=headers)
