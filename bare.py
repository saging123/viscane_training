from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import shutil
import time
from typing import Any

from sugarcane_variety.colab_compatible import (
    print_eval_summary,
    run_all_for_colab,
    test_for_colab,
)
from sugarcane_variety.preprocess import analyze_prepared_dataset, audit_prepared_splits


RAW_DIR = "content/data/raw/DATASETSFINAL"
PREPARED_DIR = "content/data/prepared"
BASE_ARTIFACTS_DIR = Path("content/data/sugarcane_artifacts")
RESNET_OUTPUT_DIR = str(BASE_ARTIFACTS_DIR / "resnet18")
YOLO_OUTPUT_DIR = str(BASE_ARTIFACTS_DIR / "yolov8")
REPORT_PATH = BASE_ARTIFACTS_DIR / "full_training_report.json"

# T4-friendly defaults for 8 vCPU + T4.
BATCH_SIZE = 32
WORKERS = 8
PREPROCESS_WORKERS = 8
EPOCHS = 45
PREPROCESS_RESIZE = 384
IMAGE_SIZE = 320
LR = 2e-4
WEIGHT_DECAY = 1e-3
LABEL_SMOOTHING = 0.10
FREEZE_BACKBONE_EPOCHS = 4
LABEL_MODE = "variety_maturity"
MODEL_TYPES = ("resnet18", "yolov8")
YOLO_WEIGHTS = "yolov8n-cls.pt"
NOISE_STD = 0.02
BLUR_PROB = 0.05
ERASE_PROB = 0.05
ROTATION_DEGREES = 8.0


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(_json_safe(report), indent=2), encoding="utf-8")


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _clean_previous_outputs() -> None:
    _remove_path(Path(PREPARED_DIR))
    _remove_path(Path(RESNET_OUTPUT_DIR))
    _remove_path(Path(YOLO_OUTPUT_DIR))
    _remove_path(REPORT_PATH)


def _make_progress_callback(report: dict[str, Any], run_name: str):
    def _callback(event: dict[str, Any]) -> None:
        event = _json_safe(event)
        event["run"] = run_name
        event["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report.setdefault("events", []).append(event)

        event_name = event.get("event", "progress")
        if event_name == "epoch_completed":
            epoch = event.get("epoch")
            epochs = event.get("epochs")
            train_acc = event.get("train_acc")
            val_acc = event.get("val_acc")
            lr = event.get("lr")
            if train_acc is not None and val_acc is not None and lr is not None:
                print(
                    f"[{run_name}] epoch {epoch}/{epochs} "
                    f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.6g}"
                )
            else:
                print(f"[{run_name}] epoch {epoch}/{epochs}: {event}")
        elif event_name in {"training_started", "training_completed", "early_stopping_triggered"}:
            print(f"[{run_name}] {event_name}: {event}")

        _write_report(report)

    return _callback


def _output_dir_for_model(model_type: str) -> str:
    if model_type == "yolov8":
        return YOLO_OUTPUT_DIR
    return RESNET_OUTPUT_DIR


def _run_model_training(
    report: dict[str, Any],
    model_type: str,
    perform_preprocess: bool,
) -> dict[str, Any]:
    output_dir = _output_dir_for_model(model_type)
    print(f"\n=== Running {model_type} full training ({LABEL_MODE}) ===")
    prep, train_summary = run_all_for_colab(
        raw_dir=RAW_DIR,
        prepared_dir=PREPARED_DIR,
        output_dir=output_dir,
        resize=PREPROCESS_RESIZE,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        image_size=IMAGE_SIZE,
        workers=WORKERS,
        seed=42,
        noise_std=NOISE_STD,
        blur_prob=BLUR_PROB,
        erase_prob=ERASE_PROB,
        rotation_degrees=ROTATION_DEGREES,
        early_stopping_patience=8,
        early_stopping_min_delta=0.002,
        use_class_weights=True,
        label_smoothing=LABEL_SMOOTHING,
        freeze_backbone_epochs=FREEZE_BACKBONE_EPOCHS,
        label_mode=LABEL_MODE,
        preprocess_device="cpu",
        preprocess_workers=PREPROCESS_WORKERS,
        perform_preprocess=perform_preprocess,
        model_type=model_type,
        yolo_weights=YOLO_WEIGHTS,
        progress_callback=_make_progress_callback(report, model_type),
    )

    split_analysis = analyze_prepared_dataset(
        prepared_dir=PREPARED_DIR,
        low_sample_threshold=20,
    )
    split_audit = audit_prepared_splits(
        prepared_dir=PREPARED_DIR,
        near_duplicate_distance=5,
        max_examples=10,
        workers=PREPROCESS_WORKERS,
    )
    eval_summary = test_for_colab(
        prepared_dir=PREPARED_DIR,
        checkpoint_path=train_summary.checkpoint_path,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        model_type=train_summary.model_type,
    )

    print(f"\n{model_type} evaluation")
    print_eval_summary(eval_summary)

    result = {
        "name": model_type,
        "label_mode": LABEL_MODE,
        "model_type": train_summary.model_type,
        "prepared_dir": PREPARED_DIR,
        "output_dir": output_dir,
        "augmentation": {
            "noise_std": NOISE_STD,
            "blur_prob": BLUR_PROB,
            "erase_prob": ERASE_PROB,
            "rotation_degrees": ROTATION_DEGREES,
        },
        "regularization": {
            "label_smoothing": LABEL_SMOOTHING,
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "weight_decay": WEIGHT_DECAY,
        },
        "preprocess": prep,
        "split_analysis": split_analysis,
        "split_audit": split_audit,
        "train": train_summary,
        "evaluation": eval_summary,
    }
    report.setdefault("results", []).append(result)
    _write_report(report)
    return result


def run_full_training(report: dict[str, Any]) -> dict[str, Any]:
    results = []
    for index, model_type in enumerate(MODEL_TYPES):
        results.append(
            _run_model_training(
                report=report,
                model_type=model_type,
                perform_preprocess=index == 0,
            )
        )

    best_result = max(
        results,
        key=lambda item: float(item["evaluation"].test_acc),
    )
    report["best_result"] = {
        "model_type": best_result["model_type"],
        "checkpoint_path": best_result["train"].checkpoint_path,
        "test_acc": best_result["evaluation"].test_acc,
        "output_dir": best_result["output_dir"],
    }
    _write_report(report)
    return best_result


def _add_findings(report: dict[str, Any]) -> None:
    results = report.get("results", [])
    best_result = report.get("best_result", {})
    findings: list[str] = []

    if results:
        score_parts = [
            f"{item['model_type']}={float(item['evaluation'].test_acc):.4f}"
            for item in results
        ]
        findings.append(f"Model comparison test accuracy: {', '.join(score_parts)}.")
        if best_result:
            findings.append(
                f"Best model is {best_result['model_type']} at {float(best_result['test_acc']):.4f}; checkpoint={best_result['checkpoint_path']}."
            )

    for result in results:
        eval_summary = result["evaluation"]
        test_acc = float(eval_summary.test_acc)
        if test_acc < 0.50:
            findings.append(
                f"{result['model_type']} accuracy is below 50%; run diagnose.py next to separate variety, maturity, and augmentation issues."
            )
        elif test_acc < 0.70:
            findings.append(
                f"{result['model_type']} learned useful signal but is not deployment-ready; inspect per-class recall and top confusions."
            )

        audit = result["split_audit"]
        if audit.cross_split_exact_groups > 0:
            findings.append(
                "Exact duplicate leakage exists across splits; fix this before trusting the score."
            )
        if audit.cross_split_near_groups > 0:
            findings.append(
                "Near-duplicate images exist across splits; review whether captures from the same cane/stalk are leaking."
            )

        analysis = result["split_analysis"]
        if analysis.low_sample_warnings:
            findings.append(
                "Some classes have low train/val/test samples; class imbalance may be limiting performance."
            )

        if getattr(eval_summary, "maturity_acc", None) is not None:
            findings.append(
                f"{result['model_type']} variety-only acc={eval_summary.variety_acc:.4f}; maturity-only acc={eval_summary.maturity_acc:.4f}."
            )
        else:
            findings.append(f"{result['model_type']} variety-only acc={eval_summary.variety_acc:.4f}.")

    report["findings"] = findings
    _write_report(report)


def main() -> None:
    _clean_previous_outputs()
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": RAW_DIR,
        "settings": {
            "label_mode": LABEL_MODE,
            "model_types": list(MODEL_TYPES),
            "yolo_weights": YOLO_WEIGHTS,
            "prepared_dir": PREPARED_DIR,
            "resnet_output_dir": RESNET_OUTPUT_DIR,
            "yolo_output_dir": YOLO_OUTPUT_DIR,
            "api_resnet_checkpoint": str(BASE_ARTIFACTS_DIR / "resnet18" / "best_model.pt"),
            "api_yolo_checkpoint": str(BASE_ARTIFACTS_DIR / "yolov8" / "yolov8" / "weights" / "best.pt"),
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "preprocess_workers": PREPROCESS_WORKERS,
            "epochs": EPOCHS,
            "preprocess_resize": PREPROCESS_RESIZE,
            "image_size": IMAGE_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "label_smoothing": LABEL_SMOOTHING,
            "freeze_backbone_epochs": FREEZE_BACKBONE_EPOCHS,
            "noise_std": NOISE_STD,
            "blur_prob": BLUR_PROB,
            "erase_prob": ERASE_PROB,
            "rotation_degrees": ROTATION_DEGREES,
        },
        "events": [],
    }
    _write_report(report)
    result = run_full_training(report)
    _add_findings(report)

    print(f"\nFull training report saved to: {REPORT_PATH}")
    print(f"Best model: {result['model_type']}")
    print(f"Best checkpoint: {result['train'].checkpoint_path}")
    print(f"API ResNet checkpoint: {BASE_ARTIFACTS_DIR / 'resnet18' / 'best_model.pt'}")
    print(f"API YOLO checkpoint: {BASE_ARTIFACTS_DIR / 'yolov8' / 'yolov8' / 'weights' / 'best.pt'}")
    print("Findings:")
    for finding in report.get("findings", []):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
