"""Normalize Colab absolute paths in WasteAI experiment CSVs.

Run from the repository root:
    python scripts/normalize_dataset_paths.py

Creates timestamped backups before editing. It only changes the image_path
column in the dataset manifest and saved split CSVs; it does not change images,
model weights, or experiment metrics.
"""
from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_FILES = [
    ROOT / "outputs" / "dataset_manifest.csv",
    ROOT / "outputs" / "splits" / "train.csv",
    ROOT / "outputs" / "splits" / "validation.csv",
    ROOT / "outputs" / "splits" / "test.csv",
]
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")


def portable_image_path(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    marker = "/dataset/"
    if marker in normalized:
        return "dataset/" + normalized.split(marker, 1)[1].lstrip("/")
    if normalized.startswith("dataset/"):
        return normalized
    return raw


def process_csv(path: Path) -> tuple[int, int]:
    if not path.exists():
        print(f"SKIP (not found): {path.relative_to(ROOT)}")
        return 0, 0

    backup = path.with_name(path.name + f".backup-{STAMP}")
    shutil.copy2(path, backup)

    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            print(f"SKIP (empty CSV): {path.relative_to(ROOT)}")
            return 0, 0
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    if "image_path" not in fieldnames:
        print(f"SKIP (no image_path column): {path.relative_to(ROOT)}")
        return len(rows), 0

    changed = 0
    for row in rows:
        before = row.get("image_path", "")
        after = portable_image_path(before)
        if before != after:
            row["image_path"] = after
            changed += 1

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"UPDATED: {path.relative_to(ROOT)} | rows={len(rows)} | paths_normalized={changed} | backup={backup.name}")
    return len(rows), changed


def main() -> None:
    total_rows = 0
    total_changed = 0
    for csv_path in CSV_FILES:
        rows, changed = process_csv(csv_path)
        total_rows += rows
        total_changed += changed
    print("\nNormalization complete.")
    print(f"Rows inspected: {total_rows}")
    print(f"Paths normalized: {total_changed}")
    print("Review the CSVs before committing. Dataset images themselves were not modified.")


if __name__ == "__main__":
    main()
