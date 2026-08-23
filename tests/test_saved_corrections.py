"""Regression checks for persistent, click-based prediction corrections."""

from __future__ import annotations

import runpy
import tempfile
from pathlib import Path

import numpy as np


MODULE = runpy.run_path(str(Path("web/backend/corrections.py")))
apply_corrections_to_mask = MODULE["apply_corrections_to_mask"]
image_key_for_bytes = MODULE["image_key_for_bytes"]
load_corrections = MODULE["load_corrections"]
save_correction = MODULE["save_correction"]


def test_correction_is_saved_by_content_key_and_reloaded() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        directory = Path(temporary_directory)
        key = image_key_for_bytes(b"same image bytes")
        saved = save_correction(directory, key, {"x": 0.5, "y": 0.5, "class_id": 6, "radius": 3})

        assert len(saved) == 1
        assert load_corrections(directory, key) == saved
        assert image_key_for_bytes(b"same image bytes") == key
        assert image_key_for_bytes(b"different image bytes") != key


def test_correction_uses_normalized_position_and_preserves_other_pixels() -> None:
    prediction = np.full((11, 11), 4, dtype=np.int64)
    corrected = apply_corrections_to_mask(
        prediction,
        [{"x": 0.5, "y": 0.5, "class_id": 6, "radius": 1}],
        num_classes=8,
    )

    assert corrected[5, 5] == 6
    assert corrected[0, 0] == 4
    assert prediction[5, 5] == 4


def test_invalid_correction_is_ignored() -> None:
    prediction = np.full((5, 5), 4, dtype=np.int64)
    corrected = apply_corrections_to_mask(
        prediction,
        [{"x": 1.2, "y": 0.5, "class_id": 6, "radius": 3}],
        num_classes=8,
    )
    assert np.array_equal(corrected, prediction)


def test_rectangle_correction_paints_the_selected_area() -> None:
    prediction = np.full((11, 11), 4, dtype=np.int64)
    corrected = apply_corrections_to_mask(
        prediction,
        [{"type": "rectangle", "x1": 0.2, "y1": 0.3, "x2": 0.6, "y2": 0.7, "class_id": 7}],
        num_classes=8,
    )

    assert corrected[3, 2] == 7
    assert corrected[7, 6] == 7
    assert corrected[2, 2] == 4
    assert corrected[8, 6] == 4


def test_brush_correction_paints_a_freeform_area() -> None:
    prediction = np.full((11, 11), 4, dtype=np.int64)
    corrected = apply_corrections_to_mask(
        prediction,
        [{"type": "stroke", "points": [{"x": 0.2, "y": 0.2}, {"x": 0.8, "y": 0.8}], "class_id": 6, "radius": 1}],
        num_classes=8,
    )

    assert corrected[2, 2] == 6
    assert corrected[8, 8] == 6
    assert corrected[0, 10] == 4


if __name__ == "__main__":
    test_correction_is_saved_by_content_key_and_reloaded()
    test_correction_uses_normalized_position_and_preserves_other_pixels()
    test_invalid_correction_is_ignored()
    test_rectangle_correction_paints_the_selected_area()
    test_brush_correction_paints_a_freeform_area()
    print("Saved correction checks passed")
