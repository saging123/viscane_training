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
BASE_PREPARED_DIR = Path("content/data")
BASE_ARTIFACTS_DIR = Path("content/data/sugarcane_artifacts")
REPORT_PATH = BASE_ARTIFACTS_DIR / "diagnostic_training_report.json"

# T4-friendly defaults for 8 vCPU + T4.
BATCH_SIZE = 32
WORKERS = 8
PREPROCESS_WORKERS = 8
EPOCHS = 35
LR = 5e-4
WEIGHT_DECAY = 5e-4

EXPERIMENTS = [
    {
        "name": "resnet18_maturity_low_aug_320",
        "label_mode": "maturity",
        "resize": 384,
        "image_size": 320,
        "noise_std": 0.02,
        "blur_prob": 0.05,
        "erase_prob": 0.05,
        "rotation_degrees": 8.0,
    },
    {
        "name": "resnet18_joint_low_aug_224",
        "label_mode": "variety_maturity",
        "resize": 256,
        "image_size": 224,
        "noise_std": 0.02,
        "blur_prob": 0.05,
        "erase_prob": 0.05,
        "rotation_degrees": 8.0,
    },
    {
        "name": "resnet18_joint_low_aug_320",
        "label_mode": "variety_maturity",
        "resize": 384,
        "image_size": 320,
        "noise_std": 0.02,
        "blur_prob": 0.05,
        "erase_prob": 0.05,
        "rotation_degrees": 8.0,
    },
    {
        "name": "resnet18_joint_full_context_320",
        "label_mode": "variety_maturity",
        "resize": None,
        "image_size": 320,
        "noise_std": 0.02,
        "blur_prob": 0.05,
        "erase_prob": 0.05,
        "rotation_degrees": 8.0,
    },
]

LEGACY_EXPERIMENT_NAMES = [
    "resnet18_variety",
    "resnet18_maturity",
    "resnet18_joint",
    "resnet18_joint_low_aug",
]


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
    experiment_names = [str(item["name"]) for item in EXPERIMENTS] + LEGACY_EXPERIMENT_NAMES
    for name in experiment_names:
        _remove_path(BASE_PREPARED_DIR / f"prepared_{name}")
        _remove_path(BASE_ARTIFACTS_DIR / name)
    _remove_path(REPORT_PATH)


def _make_progress_callback(report: dict[str, Any], experiment_name: str):
    def _callback(event: dict[str, Any]) -> None:
        event = _json_safe(event)
        event["experiment"] = experiment_name
        event["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        report.setdefault("events", []).append(event)

        event_name = event.get("event", "progress")
        if event_name == "epoch_completed":
            epoch = event.get("epoch")
            epochs = event.get("epochs")
            train_acc = event.get("train_acc")
            val_acc = event.get("val_acc")
            lr = event.get("lr")
            print(
                f"[{experiment_name}] epoch {epoch}/{epochs} "
                f"train_acc={train_acc:.4f} val_acc={val_acc:.4f} lr={lr:.6g}"
            )
        elif event_name in {"training_started", "training_completed", "early_stopping_triggered"}:
            print(f"[{experiment_name}] {event_name}: {event}")

        _write_report(report)

    return _callback


def _run_resnet_experiment(
    report: dict[str, Any],
    name: str,
    label_mode: str,
    resize: int | None,
    image_size: int,
    noise_std: float,
    blur_prob: float,
    erase_prob: float,
    rotation_degrees: float,
) -> dict[str, Any]:
    prepared_dir = str(BASE_PREPARED_DIR / f"prepared_{name}")
    output_dir = str(BASE_ARTIFACTS_DIR / name)
    print(f"\n=== Running {name} ({label_mode}) ===")

    prep, train_summary = run_all_for_colab(
        raw_dir=RAW_DIR,
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        resize=resize,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        image_size=image_size,
        workers=WORKERS,
        seed=42,
        noise_std=noise_std,
        blur_prob=blur_prob,
        erase_prob=erase_prob,
        rotation_degrees=rotation_degrees,
        early_stopping_patience=8,
        early_stopping_min_delta=0.002,
        use_class_weights=True,
        label_mode=label_mode,
        preprocess_device="cpu",
        preprocess_workers=PREPROCESS_WORKERS,
        perform_preprocess=True,
        model_type="resnet18",
        progress_callback=_make_progress_callback(report, name),
    )

    split_analysis = analyze_prepared_dataset(
        prepared_dir=prepared_dir,
        low_sample_threshold=20,
    )
    split_audit = audit_prepared_splits(
        prepared_dir=prepared_dir,
        near_duplicate_distance=5,
        max_examples=10,
        workers=PREPROCESS_WORKERS,
    )
    eval_summary = test_for_colab(
        prepared_dir=prepared_dir,
        checkpoint_path=train_summary.checkpoint_path,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        model_type="resnet18",
    )

    print(f"\n{name} evaluation")
    print_eval_summary(eval_summary)

    result = {
        "name": name,
        "label_mode": label_mode,
        "prepared_dir": prepared_dir,
        "output_dir": output_dir,
        "augmentation": {
            "resize": resize,
            "image_size": image_size,
            "noise_std": noise_std,
            "blur_prob": blur_prob,
            "erase_prob": erase_prob,
            "rotation_degrees": rotation_degrees,
        },
        "preprocess": prep,
        "split_analysis": split_analysis,
        "split_audit": split_audit,
        "train": train_summary,
        "evaluation": eval_summary,
    }
    report.setdefault("experiments", []).append(result)
    _write_report(report)
    return result


def _add_findings(report: dict[str, Any]) -> None:
    experiments = report.get("experiments", [])
    by_name = {item["name"]: item for item in experiments}
    findings: list[str] = []

    maturity = by_name.get("resnet18_maturity_low_aug_320")
    joint_low_aug_224 = by_name.get("resnet18_joint_low_aug_224")
    joint_low_aug_320 = by_name.get("resnet18_joint_low_aug_320")
    joint_full_context = by_name.get("resnet18_joint_full_context_320")

    if maturity:
        maturity_acc = float(maturity["evaluation"].test_acc)
        if maturity_acc < 0.60:
            findings.append(
                "Maturity remains the main bottleneck after label cleanup; prioritize maturity label review and more balanced maturity samples."
            )

    if joint_low_aug_224 and joint_low_aug_320:
        acc_224 = float(joint_low_aug_224["evaluation"].test_acc)
        acc_320 = float(joint_low_aug_320["evaluation"].test_acc)
        if acc_320 > acc_224 + 0.02:
            findings.append(
                "Higher resolution helped; keep 320px training / 384px preprocessing in bare.py."
            )
        elif acc_224 > acc_320 + 0.02:
            findings.append(
                "Higher resolution did not help; keep the smaller 224px recipe to avoid overfitting."
            )
        else:
            findings.append(
                "224px and 320px are close; choose based on speed unless per-class maturity recall improves at 320px."
            )

    if joint_low_aug_320 and joint_full_context:
        cropped_acc = float(joint_low_aug_320["evaluation"].test_acc)
        full_context_acc = float(joint_full_context["evaluation"].test_acc)
        if full_context_acc > cropped_acc + 0.02:
            findings.append(
                "Full image context helped; avoid square preprocessing crops for final training."
            )
        elif cropped_acc > full_context_acc + 0.02:
            findings.append(
                "Square preprocessing performed better than full-context originals; keep the 384px prepared images."
            )
        else:
            findings.append(
                "Full-context and square-preprocessed runs are close; focus next on labels and low-sample classes."
            )

    for item in experiments:
        audit = item["split_audit"]
        if audit.cross_split_exact_groups > 0:
            findings.append(
                f"{item['name']} still has exact duplicate leakage across splits; fix this before trusting accuracy."
            )
        analysis = item["split_analysis"]
        if analysis.low_sample_warnings:
            findings.append(
                f"{item['name']} has low-sample class warnings; class imbalance may be limiting performance."
            )

    report["findings"] = findings
    _write_report(report)


def main() -> None:
    _clean_previous_outputs()
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": RAW_DIR,
        "settings": {
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "preprocess_workers": PREPROCESS_WORKERS,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "experiments": [],
        "events": [],
    }
    _write_report(report)

    for experiment in EXPERIMENTS:
        _run_resnet_experiment(report=report, **experiment)

    _add_findings(report)
    print(f"\nDiagnostic report saved to: {REPORT_PATH}")
    print("Findings:")
    for finding in report.get("findings", []):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
