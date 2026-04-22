from __future__ import annotations

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
    _build_resnet18,
    _decode_class_name,
    _prepare_images_on_device,
    _raise_ultralytics_dependency_error,
)
from sugarcane_variety.colab_compatible import run_all_for_colab, test_for_colab


DEFAULT_ARTIFACTS_DIR = "content/data/sugarcane_artifacts"
DEFAULT_RESNET_CHECKPOINT_PATH = "content/data/sugarcane_artifacts/resnet18/best_model.pt"
DEFAULT_YOLO_CHECKPOINT_PATH = "content/data/sugarcane_artifacts/yolov8/yolov8/weights/best.pt"
ARTIFACT_EXTENSIONS = {".pt", ".ptl", ".onnx", ".json"}
VISUAL_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_MODELS = {"resnet18", "yolov8"}


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
    raw_dir: str = "content/data/raw"
    prepared_dir: str = "content/data/prepared"
    output_dir: str = "content/data/sugarcane_artifacts"
    val_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, lt=1.0)
    resize: int | None = 256
    epochs: int = Field(default=25, ge=1)
    batch_size: int = Field(default=32, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    image_size: int = Field(default=224, ge=1)
    workers: int = Field(default=8, ge=0)
    seed: int = 42
    augment_validation: bool = False
    noise_std: float = Field(default=0.04, ge=0.0)
    blur_prob: float = Field(default=0.20, ge=0.0, le=1.0)
    erase_prob: float = Field(default=0.20, ge=0.0, le=1.0)
    rotation_degrees: float = Field(default=12.0, ge=0.0)
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

    crop_image_size = model_image_sizes.get("resnet18") or model_image_sizes.get("yolov8") or 224
    crop_scale = crop_image_size / int(crop_image_size * 1.15)
    crop_size = min(annotated.width, annotated.height) * crop_scale
    left = (annotated.width - crop_size) / 2
    top = (annotated.height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size

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
    return []


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


def _render_model_doc_section(model_type: str) -> str:
    payload = _build_model_doc_payload(model_type)
    metrics = payload["metrics"]
    test_summary = payload["test_summary"]
    android_metadata = payload["android_metadata"]
    artifacts_dir = payload["artifacts_dir"]
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
            ("Noise Std", "noise_std"),
            ("Blur Probability", "blur_prob"),
            ("Erase Probability", "erase_prob"),
            ("Rotation Degrees", "rotation_degrees"),
        ],
    )

    return f"""
    <section class="model-section">
      <div class="section-header">
        <h2>{html.escape(model_type.upper())}</h2>
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
    for model_type in sorted(SUPPORTED_MODELS):
        payload = _build_model_doc_payload(model_type)
        metrics = payload["metrics"]
        test_summary = payload["test_summary"]
        comparison_rows.append(
            "<tr>"
            f"<td>{html.escape(model_type)}</td>"
            f"<td>{html.escape('loaded' if loaded_models.get(model_type) is not None else 'not loaded')}</td>"
            f"<td>{html.escape(_format_metric_value(metrics.get('best_val_acc')))}</td>"
            f"<td>{html.escape(_format_metric_value(test_summary.get('test_acc', metrics.get('test_acc'))))}</td>"
            f"<td>{html.escape(str(len(model_classes.get(model_type, []))))}</td>"
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
    .section-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
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
              <th>Loaded Classes</th>
              <th>Artifacts Directory</th>
            </tr>
          </thead>
          <tbody>{''.join(comparison_rows)}</tbody>
        </table>
      </article>
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
        )
        _set_training_status(
            state="running",
            message="Training is running.",
            request=request.dict(),
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

        _set_training_status(
            state="completed",
            message="Training completed. The API model was reloaded from the new checkpoint.",
            preprocess_summary=_summary_to_dict(prep_summary),
            train_summary=_summary_to_dict(train_summary),
            eval_summary=_summary_to_dict(eval_summary),
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

    if requested_model_type != "resnet18":
        raise ValueError("model_type must be 'resnet18' or 'yolov8'.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = list(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])

    loaded_model = _build_resnet18(num_classes=len(classes), pretrained=False)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    loaded_models["resnet18"] = loaded_model
    loaded_model_type = "resnet18"
    model_classes["resnet18"] = classes
    model_image_sizes["resnet18"] = image_size
    model_load_errors["resnet18"] = None
    model_checkpoints["resnet18"] = checkpoint_path


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
    if loaded_models.get("resnet18") is not None:
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
        raise HTTPException(status_code=400, detail="model_type must be 'resnet18' or 'yolov8'.")

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
            detail="model_type must be 'resnet18' or 'yolov8'.",
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
    if request.label_mode not in {"variety", "variety_maturity"}:
        raise HTTPException(
            status_code=400,
            detail="label_mode must be 'variety' or 'variety_maturity'.",
        )
    if request.preprocess_device not in {"auto", "cuda", "cpu"}:
        raise HTTPException(
            status_code=400,
            detail="preprocess_device must be 'auto', 'cuda', or 'cpu'.",
        )
    if request.model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'resnet18' or 'yolov8'.",
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
