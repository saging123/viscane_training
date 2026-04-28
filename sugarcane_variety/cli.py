from __future__ import annotations

import argparse

from sugarcane_variety.preprocess import (
    analyze_prepared_dataset,
    audit_prepared_splits,
    run_preprocess,
    run_preprocess_flat,
)
from sugarcane_variety.train import (
    DEFAULT_TRAIN_BLUR_PROB,
    DEFAULT_TRAIN_ERASE_PROB,
    DEFAULT_TRAIN_NOISE_STD,
    DEFAULT_TRAIN_ROTATION_DEGREES,
    DEFAULT_USE_BALANCED_SAMPLER,
    run_evaluation,
    run_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sugarcane variety classifier pipeline (preprocess + training)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("preprocess", help="Validate and split dataset.")
    prep.add_argument("--raw-dir", required=True, help="Path to raw dataset root.")
    prep.add_argument(
        "--prepared-dir",
        default="data/prepared",
        help="Where split dataset is written.",
    )
    prep.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    prep.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    prep.add_argument("--seed", type=int, default=42, help="Random seed.")
    prep.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize during preprocessing (e.g., 256).",
    )
    prep.add_argument(
        "--label-mode",
        choices=["variety", "maturity", "variety_maturity"],
        default="variety",
        help="Labeling mode: variety only, maturity only, or variety+maturity.",
    )
    prep.add_argument(
        "--preprocess-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for resize preprocessing. 'auto' uses CUDA when available.",
    )
    prep.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="CPU workers for image validation and preprocessing.",
    )

    prep_flat = subparsers.add_parser(
        "preprocess-flat",
        help="Validate/preprocess and keep folder-per-variety output.",
    )
    prep_flat.add_argument("--raw-dir", required=True, help="Path to raw dataset root.")
    prep_flat.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Where processed dataset is written as processed/<variety>/images.",
    )
    prep_flat.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize (e.g., 256).",
    )
    prep_flat.add_argument(
        "--label-mode",
        choices=["variety", "maturity", "variety_maturity"],
        default="variety",
        help="Labeling mode: variety only, maturity only, or variety+maturity.",
    )
    prep_flat.add_argument(
        "--preprocess-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for resize preprocessing. 'auto' uses CUDA when available.",
    )
    prep_flat.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="CPU workers for image validation and preprocessing.",
    )

    audit = subparsers.add_parser(
        "audit-splits",
        help="Audit prepared train/val/test folders for duplicate leakage.",
    )
    audit.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    audit.add_argument(
        "--near-duplicate-distance",
        type=int,
        default=5,
        help="Maximum perceptual hash Hamming distance to flag as near-duplicate.",
    )
    audit.add_argument(
        "--max-examples",
        type=int,
        default=25,
        help="Maximum suspicious examples to include in the report.",
    )
    audit.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for image hashing during the audit.",
    )

    analyze = subparsers.add_parser(
        "analyze-prepared",
        help="Summarize class balance, split counts, and low-sample risks.",
    )
    analyze.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    analyze.add_argument(
        "--low-sample-threshold",
        type=int,
        default=20,
        help="Warn when a class has fewer than this many train samples.",
    )

    train = subparsers.add_parser("train", help="Train model on prepared dataset.")
    train.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    train.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model checkpoints and metrics.",
    )
    train.add_argument("--epochs", type=int, default=35, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    train.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    train.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay for AdamW."
    )
    train.add_argument("--image-size", type=int, default=224, help="Input size.")
    train.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")
    train.add_argument(
        "--augment-validation",
        action="store_true",
        help="Deprecated compatibility flag; validation and test images are never augmented.",
    )
    train.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_TRAIN_NOISE_STD,
        help="Gaussian noise strength for ResNet training images.",
    )
    train.add_argument(
        "--blur-prob",
        type=float,
        default=DEFAULT_TRAIN_BLUR_PROB,
        help="Probability of Gaussian blur for ResNet training images.",
    )
    train.add_argument(
        "--erase-prob",
        type=float,
        default=DEFAULT_TRAIN_ERASE_PROB,
        help="Probability of random erasing for ResNet training images.",
    )
    train.add_argument(
        "--rotation-degrees",
        type=float,
        default=DEFAULT_TRAIN_ROTATION_DEGREES,
        help="Maximum random rotation in degrees for ResNet training images.",
    )
    train.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop training after this many epochs without meaningful validation improvement.",
    )
    train.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.002,
        help="Minimum validation-accuracy gain counted as an improvement for ResNet18.",
    )
    train.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting for ResNet18 training loss.",
    )
    train.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        default=DEFAULT_USE_BALANCED_SAMPLER,
        help="Enable balanced ResNet18 batch sampling for imbalanced classes.",
    )
    train.add_argument(
        "--model-type",
        choices=["resnet18", "resnet18_two_head", "yolov8"],
        default="resnet18",
        help="Model pipeline to train.",
    )
    train.add_argument(
        "--yolo-weights",
        default="yolov8n-cls.pt",
        help="YOLOv8 classification weights to fine-tune.",
    )

    test = subparsers.add_parser("test", help="Evaluate checkpoint on test split.")
    test.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    test.add_argument(
        "--checkpoint-path",
        default="artifacts/best_model.pt",
        help="Path to trained checkpoint.",
    )
    test.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    test.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    test.add_argument(
        "--model-type",
        choices=["resnet18", "resnet18_two_head", "yolov8"],
        default=None,
        help="Checkpoint model type. Auto-detected for ResNet18 checkpoints.",
    )

    all_cmd = subparsers.add_parser("all", help="Run preprocess then train.")
    all_cmd.add_argument("--raw-dir", help="Path to raw dataset root.")
    all_cmd.add_argument(
        "--prepared-dir",
        default="data/prepared",
        help="Where split dataset is written.",
    )
    all_cmd.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model checkpoints and metrics.",
    )
    all_cmd.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio.")
    all_cmd.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio.")
    all_cmd.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Optional square resize during preprocessing.",
    )
    all_cmd.add_argument("--epochs", type=int, default=35, help="Training epochs.")
    all_cmd.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    all_cmd.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    all_cmd.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight decay for AdamW."
    )
    all_cmd.add_argument("--image-size", type=int, default=224, help="Input size.")
    all_cmd.add_argument("--workers", type=int, default=8, help="DataLoader workers.")
    all_cmd.add_argument("--seed", type=int, default=42, help="Random seed.")
    all_cmd.add_argument(
        "--augment-validation",
        action="store_true",
        help="Deprecated compatibility flag; validation and test images are never augmented.",
    )
    all_cmd.add_argument(
        "--noise-std",
        type=float,
        default=DEFAULT_TRAIN_NOISE_STD,
        help="Gaussian noise strength for ResNet training images.",
    )
    all_cmd.add_argument(
        "--blur-prob",
        type=float,
        default=DEFAULT_TRAIN_BLUR_PROB,
        help="Probability of Gaussian blur for ResNet training images.",
    )
    all_cmd.add_argument(
        "--erase-prob",
        type=float,
        default=DEFAULT_TRAIN_ERASE_PROB,
        help="Probability of random erasing for ResNet training images.",
    )
    all_cmd.add_argument(
        "--rotation-degrees",
        type=float,
        default=DEFAULT_TRAIN_ROTATION_DEGREES,
        help="Maximum random rotation in degrees for ResNet training images.",
    )
    all_cmd.add_argument(
        "--early-stopping-patience",
        type=int,
        default=8,
        help="Stop training after this many epochs without meaningful validation improvement.",
    )
    all_cmd.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.002,
        help="Minimum validation-accuracy gain counted as an improvement for ResNet18.",
    )
    all_cmd.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weighting for ResNet18 training loss.",
    )
    all_cmd.add_argument(
        "--use-balanced-sampler",
        action="store_true",
        default=DEFAULT_USE_BALANCED_SAMPLER,
        help="Enable balanced ResNet18 batch sampling for imbalanced classes.",
    )
    all_cmd.add_argument(
        "--model-type",
        choices=["resnet18", "resnet18_two_head", "yolov8"],
        default="resnet18",
        help="Model pipeline to train.",
    )
    all_cmd.add_argument(
        "--yolo-weights",
        default="yolov8n-cls.pt",
        help="YOLOv8 classification weights to fine-tune.",
    )
    all_cmd.add_argument(
        "--label-mode",
        choices=["variety", "maturity", "variety_maturity"],
        default="variety",
        help="Labeling mode for preprocessing.",
    )
    all_cmd.add_argument(
        "--preprocess-device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device for resize preprocessing. 'auto' uses CUDA when available.",
    )
    all_cmd.add_argument(
        "--preprocess-workers",
        type=int,
        default=8,
        help="CPU workers for image validation and preprocessing.",
    )
    all_cmd.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Assume prepared-dir already exists and run training only.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "preprocess":
        summary = run_preprocess(
            raw_dir=args.raw_dir,
            output_dir=args.prepared_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            image_size=args.resize,
            label_mode=args.label_mode,
            preprocess_device=args.preprocess_device,
            preprocess_workers=args.preprocess_workers,
        )
        print("Preprocess complete")
        print(f"Classes: {len(summary.classes)} -> {summary.classes}")
        print(
            f"Split counts: train={summary.train_count} "
            f"val={summary.val_count} test={summary.test_count}"
        )
        print(f"Skipped corrupt images: {summary.skipped_corrupt}")
        return

    if args.command == "preprocess-flat":
        summary = run_preprocess_flat(
            raw_dir=args.raw_dir,
            output_dir=args.processed_dir,
            image_size=args.resize,
            label_mode=args.label_mode,
            preprocess_device=args.preprocess_device,
            preprocess_workers=args.preprocess_workers,
        )
        print("Preprocess flat complete")
        print(f"Classes: {len(summary.classes)} -> {summary.classes}")
        print(f"Total images: {summary.total_count}")
        print(f"Skipped corrupt images: {summary.skipped_corrupt}")
        return

    if args.command == "audit-splits":
        summary = audit_prepared_splits(
            prepared_dir=args.prepared_dir,
            near_duplicate_distance=args.near_duplicate_distance,
            max_examples=args.max_examples,
            workers=args.workers,
        )
        print("Split audit complete")
        print(f"Prepared dir: {summary.prepared_dir}")
        print(f"Total images: {summary.total_images}")
        print(f"Exact duplicate groups: {summary.exact_duplicate_groups}")
        print(f"Cross-split exact duplicate groups: {summary.cross_split_exact_groups}")
        print(f"Near-duplicate groups: {summary.near_duplicate_groups}")
        print(f"Cross-split near-duplicate groups: {summary.cross_split_near_groups}")
        print(f"Summary JSON: {summary.summary_json_path}")
        if summary.suspicious_examples:
            print("Suspicious examples:")
            for example in summary.suspicious_examples[:10]:
                print(f"- type={example['type']} splits={example['splits']}")
                for item in example["items"]:
                    print(
                        f"  path={item['relative_path']} class={item['class_name']}"
                    )
        return

    if args.command == "analyze-prepared":
        summary = analyze_prepared_dataset(
            prepared_dir=args.prepared_dir,
            low_sample_threshold=args.low_sample_threshold,
        )
        print("Prepared dataset analysis complete")
        print(f"Prepared dir: {summary.prepared_dir}")
        print(f"Total images: {summary.total_images}")
        print(
            f"Split counts: train={summary.split_counts.get('train', 0)} "
            f"val={summary.split_counts.get('val', 0)} "
            f"test={summary.split_counts.get('test', 0)}"
        )
        print(f"Classes: {len(summary.overall_class_counts)}")
        print(
            "Class distribution ratio "
            f"(minority/majority): {summary.class_distribution_ratio:.4f}"
        )
        print(f"Summary JSON: {summary.summary_json_path}")
        if summary.low_sample_warnings:
            print("Low-sample warnings:")
            for warning in summary.low_sample_warnings[:10]:
                counts = warning["counts"]
                print(
                    f"- {warning['class_name']}: train={counts['train']} "
                    f"val={counts['val']} test={counts['test']} "
                    f"total={warning['total']} reasons={','.join(warning['reasons'])}"
                )
        return

    if args.command == "train":
        summary = run_training(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            workers=args.workers,
            seed=args.seed,
            augment_validation=args.augment_validation,
            noise_std=args.noise_std,
            blur_prob=args.blur_prob,
            erase_prob=args.erase_prob,
            rotation_degrees=args.rotation_degrees,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            use_class_weights=not args.disable_class_weights,
            use_balanced_sampler=args.use_balanced_sampler,
            model_type=args.model_type,
            yolo_weights=args.yolo_weights,
        )
        print("Training complete")
        print(f"Model type: {summary.model_type}")
        print(f"Best val acc: {summary.best_val_acc:.4f}")
        print(f"Test acc: {summary.test_acc:.4f}")
        print(f"Checkpoint: {summary.checkpoint_path}")
        if summary.android_artifact_path:
            print(f"Android artifact: {summary.android_artifact_path}")
        if summary.onnx_artifact_path:
            print(f"ONNX artifact: {summary.onnx_artifact_path}")
        if summary.android_metadata_path:
            print(f"Android metadata: {summary.android_metadata_path}")
        return

    if args.command == "test":
        summary = run_evaluation(
            prepared_dir=args.prepared_dir,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            workers=args.workers,
            model_type=args.model_type,
        )
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
        return

    if args.command == "all":
        if args.skip_preprocess:
            print(f"Skipping preprocess; using prepared dataset at {args.prepared_dir}")
        else:
            if not args.raw_dir:
                raise ValueError("--raw-dir is required unless --skip-preprocess is set.")
            prep_summary = run_preprocess(
                raw_dir=args.raw_dir,
                output_dir=args.prepared_dir,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
                image_size=args.resize,
                label_mode=args.label_mode,
                preprocess_device=args.preprocess_device,
                preprocess_workers=args.preprocess_workers,
            )
            print(
                f"Preprocess complete | train={prep_summary.train_count} "
                f"val={prep_summary.val_count} test={prep_summary.test_count}"
            )
        train_summary = run_training(
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            image_size=args.image_size,
            workers=args.workers,
            seed=args.seed,
            augment_validation=args.augment_validation,
            noise_std=args.noise_std,
            blur_prob=args.blur_prob,
            erase_prob=args.erase_prob,
            rotation_degrees=args.rotation_degrees,
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_min_delta=args.early_stopping_min_delta,
            use_class_weights=not args.disable_class_weights,
            use_balanced_sampler=args.use_balanced_sampler,
            model_type=args.model_type,
            yolo_weights=args.yolo_weights,
        )
        print("End-to-end complete")
        print(f"Model type: {train_summary.model_type}")
        print(f"Best val acc: {train_summary.best_val_acc:.4f}")
        print(f"Test acc: {train_summary.test_acc:.4f}")
        print(f"Checkpoint: {train_summary.checkpoint_path}")
        if train_summary.android_artifact_path:
            print(f"Android artifact: {train_summary.android_artifact_path}")
        if train_summary.onnx_artifact_path:
            print(f"ONNX artifact: {train_summary.onnx_artifact_path}")
        if train_summary.android_metadata_path:
            print(f"Android metadata: {train_summary.android_metadata_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")
