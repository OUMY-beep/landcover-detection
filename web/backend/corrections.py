"""Persistent, user-made corrections for uploaded segmentation images."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def image_key_for_bytes(raw: bytes) -> str:
    """Return a stable key for an image's exact contents, not its filename."""
    return hashlib.sha256(raw).hexdigest()


def _correction_path(directory: Path, image_key: str) -> Path:
    if len(image_key) != 64 or any(char not in "0123456789abcdef" for char in image_key):
        raise ValueError("Invalid image key")
    return directory / f"{image_key}.json"


def load_corrections(directory: Path, image_key: str) -> list[dict[str, Any]]:
    """Load saved clicks for an image; malformed saved data is ignored safely."""
    path = _correction_path(directory, image_key)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_correction(
    directory: Path,
    image_key: str,
    correction: dict[str, Any],
) -> list[dict[str, Any]]:
    """Append one correction and replace its file atomically."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _correction_path(directory, image_key)
    corrections = load_corrections(directory, image_key)
    corrections.append(correction)

    temporary_path = path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(corrections, separators=(",", ":")), encoding="utf-8")
    temporary_path.replace(path)
    return corrections


def apply_corrections_to_mask(
    prediction: np.ndarray,
    corrections: list[dict[str, Any]],
    num_classes: int,
) -> np.ndarray:
    """Paint valid normalized click and rectangle corrections on a mask."""
    if prediction.ndim != 2:
        raise ValueError(f"Expected a 2D prediction mask, got shape {prediction.shape}")

    corrected = prediction.copy()
    height, width = corrected.shape
    rows, columns = np.ogrid[:height, :width]

    for item in corrections:
        try:
            class_id = int(item["class_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= class_id < num_classes:
            continue

        if item.get("type") == "rectangle":
            try:
                x1 = float(item["x1"])
                y1 = float(item["y1"])
                x2 = float(item["x2"])
                y2 = float(item["y2"])
            except (KeyError, TypeError, ValueError):
                continue
            if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
                continue

            left, right = sorted((round(x1 * (width - 1)), round(x2 * (width - 1))))
            top, bottom = sorted((round(y1 * (height - 1)), round(y2 * (height - 1))))
            corrected[top:bottom + 1, left:right + 1] = class_id
            continue

        if item.get("type") == "stroke":
            try:
                radius = max(1, min(int(item.get("radius", 12)), 64))
                points = item["points"]
            except (KeyError, TypeError, ValueError):
                continue
            if not isinstance(points, list):
                continue

            for point in points:
                try:
                    x = float(point["x"])
                    y = float(point["y"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    continue
                center_x = round(x * (width - 1))
                center_y = round(y * (height - 1))
                brush = (columns - center_x) ** 2 + (rows - center_y) ** 2 <= radius**2
                corrected[brush] = class_id
            continue

        try:
            x = float(item["x"])
            y = float(item["y"])
            radius = int(item.get("radius", 8))
        except (KeyError, TypeError, ValueError):
            continue

        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            continue
        radius = max(1, min(radius, 64))
        center_x = round(x * (width - 1))
        center_y = round(y * (height - 1))
        brush = (columns - center_x) ** 2 + (rows - center_y) ** 2 <= radius**2
        corrected[brush] = class_id

    return corrected
