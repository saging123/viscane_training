from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from sugarcane_variety.preprocess import (
    PreparedDatasetAnalysisSummary,
    PreprocessSummary,
    analyze_prepared_dataset,
    run_preprocess,
)
from sugarcane_variety.train import (
    DEFAULT_TRAIN_BLUR_PROB,
    DEFAULT_TRAIN_ERASE_PROB,
    DEFAULT_TRAIN_NOISE_STD,
    DEFAULT_TRAIN_ROTATION_DEGREES,
    DEFAULT_USE_BALANCED_SAMPLER,
    EvalSummary,
    TrainSummary,
    run_evaluation,
    run_training,
)


def install_requirements(requirements_path: str = "requirements.txt") -> None:
    """Install project dependencies in Colab."""
    req = Path(requirements_path)
    if not req.exists():
        raise FileNotFoundError(f"requirements file not found: {req.resolve()}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])


def mount_drive(mount_point: str = "/content/drive") -> None:
    """Mount Google Drive when running in Colab."""
    try:
        from google.colab import drive  # type: ignore
    except ImportError as exc:
        raise RuntimeError("google.colab is not available. Run this inside Colab.") from exc
    drive.mount(mount_point)


def preprocess_for_colab(
    raw_dir: str,
    prepared_dir: str = "/content/data/prepared",
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
    resize: int | None = 256,
    label_mode: str = "variety",
    preprocess_device: str = "auto",
    preprocess_workers: int = 8,
) -> PreprocessSummary:
    """Run preprocessing with Colab-friendly defaults."""
    return run_preprocess(
        raw_dir=raw_dir,
        output_dir=prepared_dir,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        image_size=resize,
        label_mode=label_mode,  # "variety" or "variety_maturity"
        preprocess_device=preprocess_device,  # "auto", "cuda", or "cpu"
        preprocess_workers=preprocess_workers,
    )


def train_for_colab(
    prepared_dir: str = "/content/data/prepared",
    output_dir: str = "/content/artifacts",
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    image_size: int = 224,
    workers: int = 8,
    seed: int = 42,
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    erase_prob: float = DEFAULT_TRAIN_ERASE_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 0.002,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    freeze_backbone_epochs: int = 0,
    use_balanced_sampler: bool = DEFAULT_USE_BALANCED_SAMPLER,
    model_type: str = "resnet18",
    yolo_weights: str = "yolov8n-cls.pt",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainSummary:
    """Run training with Colab-friendly defaults."""
    return run_training(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        image_size=image_size,
        workers=workers,
        seed=seed,
        augment_validation=augment_validation,
        noise_std=noise_std,
        blur_prob=blur_prob,
        erase_prob=erase_prob,
        rotation_degrees=rotation_degrees,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_class_weights=use_class_weights,
        label_smoothing=label_smoothing,
        freeze_backbone_epochs=freeze_backbone_epochs,
        use_balanced_sampler=use_balanced_sampler,
        model_type=model_type,
        yolo_weights=yolo_weights,
        progress_callback=progress_callback,
    )


def test_for_colab(
    prepared_dir: str = "/content/data/prepared",
    checkpoint_path: str = "/content/artifacts/best_model.pt",
    batch_size: int = 32,
    workers: int = 8,
    model_type: str | None = None,
) -> EvalSummary:
    """Run test-only evaluation in Colab."""
    return run_evaluation(
        prepared_dir=prepared_dir,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        workers=workers,
        model_type=model_type,
    )


def print_eval_summary(summary: EvalSummary) -> None:
    """Pretty-print evaluation summary in Colab output cells."""
    print("Evaluation complete")
    print(f"Model type: {summary.model_type}")
    print(f"Samples: {summary.num_samples}")
    print(f"Test loss: {summary.test_loss:.4f}")
    print(f"Exact label acc: {summary.test_acc:.4f}")
    print(f"Variety-only acc: {summary.variety_acc:.4f}")
    if summary.maturity_acc is not None:
        print(f"Maturity-only acc: {summary.maturity_acc:.4f}")
    print(f"Device: {summary.device}")
    print(f"Checkpoint: {summary.checkpoint_path}")
    print(f"Summary JSON: {summary.summary_json_path}")
    print("Interpretation:")
    for point in summary.interpretation_points:
        print(f"- {point}")
    print(f"Friendly outcome: {summary.friendly_outcome}")
    if summary.top_confusions:
        print("Top confusions:")
        for row in summary.top_confusions[:5]:
            print(
                f"- true={row['true_class']} predicted={row['predicted_class']} "
                f"count={row['count']}"
            )


def _dataset_analysis_payload(
    summary: PreparedDatasetAnalysisSummary,
) -> dict[str, Any]:
    return {
        "prepared_dir": summary.prepared_dir,
        "total_images": summary.total_images,
        "split_counts": summary.split_counts,
        "class_counts_by_split": summary.class_counts_by_split,
        "overall_class_counts": summary.overall_class_counts,
        "class_distribution_ratio": summary.class_distribution_ratio,
        "low_sample_threshold": summary.low_sample_threshold,
        "low_sample_warnings": summary.low_sample_warnings,
        "variety_counts_by_split": summary.variety_counts_by_split,
        "maturity_counts_by_split": summary.maturity_counts_by_split,
        "summary_json_path": summary.summary_json_path,
    }


def run_all_for_colab(
    raw_dir: str,
    prepared_dir: str = "/content/data/prepared",
    output_dir: str = "/content/artifacts",
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    resize: int | None = 256,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 5e-4,
    image_size: int = 224,
    workers: int = 8,
    seed: int = 42,
    augment_validation: bool = False,
    noise_std: float = DEFAULT_TRAIN_NOISE_STD,
    blur_prob: float = DEFAULT_TRAIN_BLUR_PROB,
    erase_prob: float = DEFAULT_TRAIN_ERASE_PROB,
    rotation_degrees: float = DEFAULT_TRAIN_ROTATION_DEGREES,
    early_stopping_patience: int = 8,
    early_stopping_min_delta: float = 0.002,
    use_class_weights: bool = True,
    label_smoothing: float = 0.0,
    freeze_backbone_epochs: int = 0,
    use_balanced_sampler: bool = DEFAULT_USE_BALANCED_SAMPLER,
    label_mode: str = "variety",
    preprocess_device: str = "auto",
    preprocess_workers: int = 8,
    perform_preprocess: bool = True,
    model_type: str = "resnet18",
    yolo_weights: str = "yolov8n-cls.pt",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[PreprocessSummary | None, TrainSummary]:
    """
    End-to-end preprocessing + training wrapper for Colab notebooks.
    """
    prep_summary = None
    if perform_preprocess:
        prep_summary = preprocess_for_colab(
            raw_dir=raw_dir,
            prepared_dir=prepared_dir,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            resize=resize,
            label_mode=label_mode,
            preprocess_device=preprocess_device,
            preprocess_workers=preprocess_workers,
        )

    dataset_analysis = analyze_prepared_dataset(
        prepared_dir=prepared_dir,
        low_sample_threshold=20,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "dataset_analyzed",
                "model_type": model_type,
                "label_mode": label_mode,
                "dataset_summary": _dataset_analysis_payload(dataset_analysis),
            }
        )

    train_summary = train_for_colab(
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        image_size=image_size,
        workers=workers,
        seed=seed,
        augment_validation=augment_validation,
        noise_std=noise_std,
        blur_prob=blur_prob,
        erase_prob=erase_prob,
        rotation_degrees=rotation_degrees,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        use_class_weights=use_class_weights,
        label_smoothing=label_smoothing,
        freeze_backbone_epochs=freeze_backbone_epochs,
        use_balanced_sampler=use_balanced_sampler,
        model_type=model_type,
        yolo_weights=yolo_weights,
        progress_callback=progress_callback,
    )
    return prep_summary, train_summary
