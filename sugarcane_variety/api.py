from __future__ import annotations

import io
import json
import os
import threading
import time
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import transforms

from sugarcane_variety.train import (
    _build_resnet18,
    _decode_class_name,
    _prepare_images_on_device,
)
from sugarcane_variety.colab_compatible import run_all_for_colab, test_for_colab


DEFAULT_CHECKPOINT_PATH = "artifacts/best_model.pt"
ARTIFACT_EXTENSIONS = {".pt", ".ptl", ".onnx", ".json"}
SUPPORTED_MODELS = {"resnet18", "yolov8"}


app = FastAPI(title="Sugarcane Variety Classifier API")

model: Any | None = None
loaded_model_type = "resnet18"
classes: list[str] = []
image_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.PILToTensor()
model_load_error: str | None = None
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


def _predict_probabilities(image: torch.Tensor) -> torch.Tensor:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if loaded_model_type == "yolov8":
        try:
            result = model(_tensor_to_pil(image), verbose=False)[0]
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
            image_size=image_size,
            training=False,
        )
        logits = model(batch)
        return F.softmax(logits, dim=1)[0]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.permute(1, 2, 0).detach().cpu().numpy()
    return Image.fromarray(array)


def _build_predictions(probabilities: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    if not classes:
        raise HTTPException(status_code=503, detail="Model class labels are not loaded.")
    top_k = min(top_k, len(classes))

    scores, indexes = torch.topk(probabilities, k=top_k)
    predictions = []
    for score, index in zip(scores.detach().cpu().tolist(), indexes.detach().cpu().tolist()):
        class_name = classes[index]
        predictions.append(
            {
                "class_index": index,
                "confidence": score,
                **_decode_class_name(class_name),
            }
        )
    return predictions


def _maturity_reason(probabilities: torch.Tensor) -> dict[str, Any]:
    maturity_scores: dict[str, float] = {}
    for index, class_name in enumerate(classes):
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

    crop_scale = image_size / int(image_size * 1.15)
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


def _checkpoint_path() -> Path:
    return Path(os.getenv("MODEL_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)).expanduser().resolve()


def _artifacts_dir() -> Path:
    return Path(os.getenv("ARTIFACTS_DIR", _checkpoint_path().parent)).expanduser().resolve()


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
            "checkpoint": str(_checkpoint_path()),
            "model_loaded": model is not None,
            "model_load_error": model_load_error,
            "classes": [
                {
                    "index": index,
                    **_decode_class_name(class_name),
                }
                for index, class_name in enumerate(classes)
            ],
            "image_size": image_size,
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

        os.environ["MODEL_CHECKPOINT"] = train_summary.checkpoint_path
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
    global model, loaded_model_type, classes, image_size, model_load_error

    checkpoint_path = checkpoint_path or _checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first, or set MODEL_CHECKPOINT=/path/to/best_model.pt."
        )

    requested_model_type = model_type or _infer_model_type(checkpoint_path)
    if requested_model_type == "yolov8":
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLOv8 inference requires the optional 'ultralytics' package. "
                "Install it with: pip install ultralytics"
            ) from exc

        loaded_model = YOLO(str(checkpoint_path))
        yolo_names = getattr(loaded_model, "names", {})
        classes = [
            name
            for _, name in sorted(yolo_names.items(), key=lambda item: int(item[0]))
        ]
        image_size = int(os.getenv("MODEL_IMAGE_SIZE", "224"))
        model = loaded_model
        loaded_model_type = "yolov8"
        model_load_error = None
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

    model = loaded_model
    loaded_model_type = "resnet18"
    model_load_error = None


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
    global model_load_error

    try:
        _load_model()
    except Exception as exc:
        model_load_error = f"{type(exc).__name__}: {exc}"


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_load_error": model_load_error,
        "device": str(device),
        "checkpoint": str(_checkpoint_path()),
        "model_type": loaded_model_type,
        "classes": len(classes),
        "image_size": image_size,
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
    }


@app.get("/reports/current")
def current_model_report(artifacts_dir: str | None = None) -> dict[str, Any]:
    selected_artifacts_dir = (
        Path(artifacts_dir).expanduser().resolve()
        if artifacts_dir is not None
        else None
    )
    return _build_current_report(artifacts_dir=selected_artifacts_dir)


@app.post("/models/load")
def load_model_endpoint(request: ModelLoadRequest) -> dict[str, Any]:
    if request.model_type is not None and request.model_type not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail="model_type must be 'resnet18' or 'yolov8'.",
        )
    checkpoint_path = Path(request.checkpoint_path).expanduser().resolve()
    _load_model(checkpoint_path=checkpoint_path, model_type=request.model_type)
    os.environ["MODEL_CHECKPOINT"] = str(checkpoint_path)
    return {
        "message": "Model loaded.",
        "checkpoint_path": str(checkpoint_path),
        "model_type": loaded_model_type,
        "classes": len(classes),
        "image_size": image_size,
    }


@app.get("/classes")
def list_classes() -> dict[str, Any]:
    return {
        "classes": [
            {
                "index": index,
                **_decode_class_name(class_name),
            }
            for index, class_name in enumerate(classes)
        ]
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
    probabilities = _predict_probabilities(image)
    predictions = _build_predictions(probabilities, top_k=top_k)

    return {
        "filename": file.filename,
        "prediction": predictions[0],
        "top_k": predictions,
    }


@app.post("/predict/annotated")
async def predict_annotated(file: UploadFile = File(...), top_k: int = 3) -> StreamingResponse:
    contents = await file.read()
    pil_image, image = _open_uploaded_image(contents)
    probabilities = _predict_probabilities(image)
    predictions = _build_predictions(probabilities, top_k=top_k)
    reason = _maturity_reason(probabilities)

    annotated = _draw_prediction_overlay(
        image=pil_image,
        maturity_reason=reason,
    )
    output = io.BytesIO()
    annotated.save(output, format="PNG")
    output.seek(0)

    headers = {
        "X-Predicted-Class": str(predictions[0]["class_name"]),
        "X-Predicted-Variety": str(predictions[0]["variety"]),
        "X-Predicted-Maturity": str(reason["maturity_status"]),
        "X-Maturity-Probability": f"{reason['maturity_probability']:.6f}",
        "X-Next-Best-Maturity-Probability": f"{reason['next_best_probability']:.6f}",
        "X-Maturity-Margin": f"{reason['margin']:.6f}",
        "X-Maturity-Reason": str(reason["reason"]),
    }
    return StreamingResponse(output, media_type="image/png", headers=headers)
