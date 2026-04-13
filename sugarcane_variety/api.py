from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
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
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be at least 1.")
    top_k = min(top_k, len(classes))

    contents = await file.read()
    try:
        with Image.open(io.BytesIO(contents)) as img:
            image = to_tensor(img.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    with torch.no_grad():
        batch = _prepare_images_on_device(
            [image],
            device=device,
            image_size=image_size,
            training=False,
        )
        logits = model(batch)
        probabilities = F.softmax(logits, dim=1)[0]
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

    return {
        "filename": file.filename,
        "prediction": predictions[0],
        "top_k": predictions,
    }
