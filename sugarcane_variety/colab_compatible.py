from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sugarcane_variety.preprocess import PreprocessSummary, run_preprocess
from sugarcane_variety.train import EvalSummary, TrainSummary, run_evaluation, run_training


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
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    resize: int | None = 256,
    label_mode: str = "variety",
    preprocess_device: str = "auto",
    preprocess_workers: int = 1,
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
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    workers: int = 2,
    seed: int = 42,
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
    )


def test_for_colab(
    prepared_dir: str = "/content/data/prepared",
    checkpoint_path: str = "/content/artifacts/best_model.pt",
    batch_size: int = 32,
    workers: int = 2,
) -> EvalSummary:
    """Run test-only evaluation in Colab."""
    return run_evaluation(
        prepared_dir=prepared_dir,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        workers=workers,
    )


def print_eval_summary(summary: EvalSummary) -> None:
    """Pretty-print evaluation summary in Colab output cells."""
    print("Evaluation complete")
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


def run_all_for_colab(
    raw_dir: str,
    prepared_dir: str = "/content/data/prepared",
    output_dir: str = "/content/artifacts",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    resize: int | None = 256,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    image_size: int = 224,
    workers: int = 2,
    seed: int = 42,
    label_mode: str = "variety",
    preprocess_device: str = "auto",
    preprocess_workers: int = 1,
    perform_preprocess: bool = True,
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
    )
    return prep_summary, train_summary
