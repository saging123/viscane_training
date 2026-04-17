from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms

from sugarcane_variety.train import _decode_class_name, _prepare_images_on_device


DEFAULT_CHECKPOINT_PATH = "artifacts/best_model.pt"


app = FastAPI(title="Sugarcane Variety Classifier API")

model: nn.Module | None = None
classes: list[str] = []
image_size = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.PILToTensor()


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

    with torch.no_grad():
        batch = _prepare_images_on_device(
            [image],
            device=device,
            image_size=image_size,
            training=False,
        )
        logits = model(batch)
        return F.softmax(logits, dim=1)[0]


def _build_predictions(probabilities: torch.Tensor, top_k: int) -> list[dict[str, Any]]:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
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


def _load_model() -> None:
    global model, classes, image_size

    checkpoint_path = _checkpoint_path()
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train first, or set MODEL_CHECKPOINT=/path/to/best_model.pt."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = list(checkpoint["classes"])
    image_size = int(checkpoint["image_size"])

    loaded_model = models.resnet18(weights=None)
    loaded_model.fc = nn.Linear(loaded_model.fc.in_features, len(classes))
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.to(device)
    loaded_model.eval()

    model = loaded_model


@app.on_event("startup")
def startup() -> None:
    _load_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "checkpoint": str(_checkpoint_path()),
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
