from __future__ import annotations

import argparse

from sugarcane_variety.preprocess import run_preprocess, run_preprocess_flat
from sugarcane_variety.train import run_evaluation, run_training


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
        choices=["variety", "variety_maturity"],
        default="variety",
        help="Labeling mode: variety only or variety+maturity.",
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
        choices=["variety", "variety_maturity"],
        default="variety",
        help="Labeling mode: variety only or variety+maturity.",
    )

    train = subparsers.add_parser("train", help="Train model on prepared dataset.")
    train.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    train.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory for model checkpoints and metrics.",
    )
    train.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    train.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    train.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    train.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW."
    )
    train.add_argument("--image-size", type=int, default=224, help="Input size.")
    train.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    train.add_argument("--seed", type=int, default=42, help="Random seed.")

    test = subparsers.add_parser("test", help="Evaluate checkpoint on test split.")
    test.add_argument("--prepared-dir", required=True, help="Prepared dataset root.")
    test.add_argument(
        "--checkpoint-path",
        default="artifacts/best_model.pt",
        help="Path to trained checkpoint.",
    )
    test.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    test.add_argument("--workers", type=int, default=4, help="DataLoader workers.")

    all_cmd = subparsers.add_parser("all", help="Run preprocess then train.")
    all_cmd.add_argument("--raw-dir", required=True, help="Path to raw dataset root.")
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
    all_cmd.add_argument("--epochs", type=int, default=20, help="Training epochs.")
    all_cmd.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    all_cmd.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    all_cmd.add_argument(
        "--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW."
    )
    all_cmd.add_argument("--image-size", type=int, default=224, help="Input size.")
    all_cmd.add_argument("--workers", type=int, default=4, help="DataLoader workers.")
    all_cmd.add_argument("--seed", type=int, default=42, help="Random seed.")
    all_cmd.add_argument(
        "--label-mode",
        choices=["variety", "variety_maturity"],
        default="variety",
        help="Labeling mode for preprocessing.",
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
        )
        print("Preprocess flat complete")
        print(f"Classes: {len(summary.classes)} -> {summary.classes}")
        print(f"Total images: {summary.total_count}")
        print(f"Skipped corrupt images: {summary.skipped_corrupt}")
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
        )
        print("Training complete")
        print(f"Best val acc: {summary.best_val_acc:.4f}")
        print(f"Test acc: {summary.test_acc:.4f}")
        print(f"Checkpoint: {summary.checkpoint_path}")
        return

    if args.command == "test":
        summary = run_evaluation(
            prepared_dir=args.prepared_dir,
            checkpoint_path=args.checkpoint_path,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        print("Evaluation complete")
        print(f"Test loss: {summary.test_loss:.4f}")
        print(f"Test acc: {summary.test_acc:.4f}")
        print(f"Device: {summary.device}")
        print(f"Checkpoint: {summary.checkpoint_path}")
        return

    if args.command == "all":
        prep_summary = run_preprocess(
            raw_dir=args.raw_dir,
            output_dir=args.prepared_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            image_size=args.resize,
            label_mode=args.label_mode,
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
        )
        print("End-to-end complete")
        print(f"Best val acc: {train_summary.best_val_acc:.4f}")
        print(f"Test acc: {train_summary.test_acc:.4f}")
        print(f"Checkpoint: {train_summary.checkpoint_path}")
        return

    raise ValueError(f"Unknown command: {args.command}")
