from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Google Drive folder into a local dataset directory."
    )
    parser.add_argument(
        "folder",
        help="Google Drive folder URL or folder ID.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Target directory for downloaded dataset.",
    )
    return parser.parse_args()


def normalize_drive_folder_url(folder: str) -> str:
    if folder.startswith("http://") or folder.startswith("https://"):
        return folder
    return f"https://drive.google.com/drive/folders/{folder}"


def main() -> int:
    args = parse_args()
    try:
        import gdown
    except ImportError:
        print(
            "Error: gdown is required to download Google Drive folders.",
            file=sys.stderr,
        )
        print("Install it with: pip install gdown", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folder_url = normalize_drive_folder_url(args.folder)
    print(f"Downloading Google Drive folder to: {output_dir}")
    print(f"Source: {folder_url}")

    gdown.download_folder(
        folder_url,
        output=str(output_dir),
        quiet=False,
        use_cookies=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
