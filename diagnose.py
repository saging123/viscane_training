from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
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

# T4-friendly defaults: larger batch uses VRAM better; 8 workers keeps input loading busy.
BATCH_SIZE = 64
WORKERS = 8
PREPROCESS_WORKERS = 7
EPOCHS = 35
IMAGE_SIZE = 224
LR = 5e-4
WEIGHT_DECAY = 5e-4


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
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        image_size=IMAGE_SIZE,
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

    variety = by_name.get("resnet18_variety")
    maturity = by_name.get("resnet18_maturity")
    joint = by_name.get("resnet18_joint")
    joint_low_aug = by_name.get("resnet18_joint_low_aug")

    if variety and maturity:
        variety_acc = float(variety["evaluation"].test_acc)
        maturity_acc = float(maturity["evaluation"].test_acc)
        if variety_acc > maturity_acc + 0.10:
            findings.append(
                "Maturity is likely the main bottleneck: variety-only accuracy is much higher than maturity-only accuracy."
            )
        elif maturity_acc > variety_acc + 0.10:
            findings.append(
                "Variety separation is likely the main bottleneck: maturity-only accuracy is much higher than variety-only accuracy."
            )
        else:
            findings.append(
                "Variety and maturity have similar difficulty; inspect labels, image quality, and split leakage for both tasks."
            )

    if joint and joint_low_aug:
        joint_acc = float(joint["evaluation"].test_acc)
        low_aug_acc = float(joint_low_aug["evaluation"].test_acc)
        if low_aug_acc > joint_acc + 0.03:
            findings.append(
                "Over-augmentation is likely hurting accuracy: the low-augmentation joint run performed better."
            )
        elif joint_acc > low_aug_acc + 0.03:
            findings.append(
                "The default augmentation is probably helping: the normal joint run performed better than low augmentation."
            )
        else:
            findings.append(
                "Augmentation strength is probably not the primary issue: normal and low-augmentation joint runs are close."
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
    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_dir": RAW_DIR,
        "settings": {
            "batch_size": BATCH_SIZE,
            "workers": WORKERS,
            "preprocess_workers": PREPROCESS_WORKERS,
            "epochs": EPOCHS,
            "image_size": IMAGE_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "experiments": [],
        "events": [],
    }
    _write_report(report)

    _run_resnet_experiment(
        report=report,
        name="resnet18_variety",
        label_mode="variety",
        noise_std=0.07,
        blur_prob=0.30,
        erase_prob=0.30,
        rotation_degrees=18.0,
    )
    _run_resnet_experiment(
        report=report,
        name="resnet18_maturity",
        label_mode="maturity",
        noise_std=0.07,
        blur_prob=0.30,
        erase_prob=0.30,
        rotation_degrees=18.0,
    )
    _run_resnet_experiment(
        report=report,
        name="resnet18_joint",
        label_mode="variety_maturity",
        noise_std=0.07,
        blur_prob=0.30,
        erase_prob=0.30,
        rotation_degrees=18.0,
    )
    _run_resnet_experiment(
        report=report,
        name="resnet18_joint_low_aug",
        label_mode="variety_maturity",
        noise_std=0.02,
        blur_prob=0.05,
        erase_prob=0.05,
        rotation_degrees=8.0,
    )

    _add_findings(report)
    print(f"\nDiagnostic report saved to: {REPORT_PATH}")
    print("Findings:")
    for finding in report.get("findings", []):
        print(f"- {finding}")


if __name__ == "__main__":
    main()
