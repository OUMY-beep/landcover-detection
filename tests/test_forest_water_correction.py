"""Regression coverage for the shifted forest/water inference IDs."""

import ast
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image
from scipy import ndimage as ndi


def _function(name: str, namespace: dict):
    source = (Path(__file__).resolve().parents[1] / "web" / "backend" / "inference.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)
    return namespace[name]


def test_only_uncertain_water_pixels_become_forest():
    correct = _function(
        "resolve_forest_water_ambiguity",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    probs = np.zeros((8, 1, 2), dtype=np.float32)
    probs[4] = [[0.55, 0.90]]
    probs[6] = [[0.50, 0.50]]
    pred = np.array([[4, 4]], dtype=np.int64)

    corrected = correct(pred, probs)

    assert corrected.tolist() == [[6, 4]]


def test_probability_calibration_keeps_strong_water_but_recovers_close_agriculture():
    calibrate = _function(
        "calibrate_landcover_probabilities",
        {
            "np": np,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    probs = np.zeros((8, 1, 2), dtype=np.float32)
    probs[4] = [[0.55, 0.97]]
    probs[7] = [[0.43, 0.01]]
    probs[1] = [[0.02, 0.02]]

    calibrated = calibrate(probs)

    assert np.argmax(calibrated, axis=0).tolist() == [[7, 4]]
    assert np.allclose(calibrated.sum(axis=0), 1.0)


def test_textured_water_prediction_with_forest_support_becomes_forest():
    correct = _function(
        "resolve_forest_water_ambiguity",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    pixels = np.full((24, 24, 3), 90, dtype=np.uint8)
    pixels[:, :12] = np.indices((24, 12)).sum(axis=0)[..., None] % 2 * 160 + 40
    image = Image.fromarray(pixels, "RGB")
    probs = np.zeros((8, 24, 24), dtype=np.float32)
    probs[4] = 0.90
    probs[6] = 0.02  # SegFormer can still rank forest first among alternatives.
    probs[7] = 0.01
    pred = np.full((24, 24), 4, dtype=np.int64)

    corrected = correct(pred, probs, image=image)

    assert np.all(corrected[8:16, 3:9] == 6)
    assert np.all(corrected[8:16, 15:21] == 4)


def test_dark_textured_canopy_overrides_confident_false_water_only():
    correct = _function(
        "resolve_forest_water_ambiguity",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    pixels = np.full((32, 32, 3), (65, 122, 118), dtype=np.uint8)
    checkerboard = np.indices((32, 16)).sum(axis=0) % 2
    pixels[:, :16] = np.where(
        checkerboard[..., None] == 1,
        (60, 105, 62),
        (35, 75, 38),
    )
    probs = np.zeros((8, 32, 32), dtype=np.float32)
    probs[4] = 0.98
    probs[1] = 0.01
    probs[6] = 0.005
    pred = np.full((32, 32), 4, dtype=np.int64)

    corrected = correct(pred, probs, image=Image.fromarray(pixels, "RGB"))

    assert np.all(corrected[10:22, 4:12] == 6)
    assert np.all(corrected[10:22, 20:28] == 4)


def test_textured_open_land_with_land_support_becomes_agricultural():
    correct = _function(
        "resolve_forest_water_ambiguity",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    pixels = np.full((32, 32, 3), (62, 113, 126), dtype=np.uint8)
    stripes = np.indices((32, 16))[1] % 2
    pixels[:, :16] = np.where(
        stripes[..., None] == 1,
        (150, 137, 85),
        (130, 120, 75),
    )
    probs = np.zeros((8, 32, 32), dtype=np.float32)
    probs[4] = 0.85
    probs[7] = 0.08
    probs[5] = 0.03
    probs[1] = 0.02
    pred = np.full((32, 32), 4, dtype=np.int64)

    corrected = correct(pred, probs, image=Image.fromarray(pixels, "RGB"))

    assert np.all(corrected[10:22, 4:12] == 7)
    assert np.all(corrected[10:22, 20:28] == 4)


def test_close_water_vs_barren_or_agriculture_prefers_land():
    correct = _function(
        "resolve_forest_water_ambiguity",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    probs = np.zeros((8, 1, 3), dtype=np.float32)
    probs[4] = [[0.74, 0.73, 0.97]]
    probs[7] = [[0.66, 0.03, 0.04]]
    probs[5] = [[0.04, 0.65, 0.03]]
    pred = np.full((1, 3), 4, dtype=np.int64)

    corrected = correct(pred, probs)

    assert corrected.tolist() == [[7, 5, 4]]


def test_large_compact_field_like_water_becomes_land_but_lake_and_river_stay_water():
    component_correct = _function(
        "resolve_field_like_water_components",
        {
            "np": np,
            "cv2": cv2,
            "Image": Image,
            "WATER_CLASS": 4,
            "FOREST_CLASS": 6,
            "AGRICULTURAL_CLASS": 7,
            "BARREN_CLASS": 5,
        },
    )
    pixels = np.full((80, 120, 3), (80, 80, 80), dtype=np.uint8)
    pixels[10:40, 10:40] = (130, 150, 85)   # Compact green agricultural field.
    pixels[10:40, 70:100] = (174, 200, 153) # Blue-green lake.
    pixels[45:75, 105:112] = (150, 145, 129) # Elongated dark river.
    pred = np.ones((80, 120), dtype=np.int64)
    pred[10:40, 10:40] = 4
    pred[10:40, 70:100] = 4
    pred[45:75, 105:112] = 4
    probs = np.zeros((8, 80, 120), dtype=np.float32)
    probs[4] = 0.90
    probs[7] = 0.03
    probs[5] = 0.02

    corrected = component_correct(pred, probs, Image.fromarray(pixels, "RGB"), min_area=50)

    assert np.all(corrected[15:35, 15:35] == 7)
    assert np.all(corrected[15:35, 75:95] == 4)
    assert np.all(corrected[50:70, 106:111] == 4)


def test_large_image_tiles_stitch_probability_maps_not_class_ids():
    class TileModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            logits = torch.zeros((1, 8, 2, 2))
            logits[:, self.calls] = 1.0
            self.calls += 1
            return logits

    tiled_predict = _function(
        "predict_large_image_with_tiles",
        {
            "np": np,
            "torch": torch,
            "Image": Image,
            "IMAGE_SIZE": (4, 4),
            "NUM_CLASSES": 8,
            "DEVICE": torch.device("cpu"),
            "prepare_image_tensor": lambda _image: torch.zeros((1, 3, 2, 2)),
        },
    )

    probabilities = tiled_predict(
        TileModel(), Image.new("RGB", (8, 8)), grid_size=2, global_weight=0.0
    )

    assert probabilities.shape == (8, 4, 4)
    assert np.argmax(probabilities, axis=0).tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
    ]


def test_overlapping_tiles_return_normalized_probability_maps():
    class TileModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            logits = torch.zeros((1, 8, 2, 2))
            logits[:, self.calls % 8] = 1.0
            self.calls += 1
            return logits

    tiled_predict = _function(
        "predict_large_image_with_tiles",
        {
            "np": np,
            "torch": torch,
            "Image": Image,
            "IMAGE_SIZE": (8, 8),
            "NUM_CLASSES": 8,
            "DEVICE": torch.device("cpu"),
            "prepare_image_tensor": lambda _image: torch.zeros((1, 3, 2, 2)),
        },
    )

    probabilities = tiled_predict(
        TileModel(), Image.new("RGB", (32, 32)), grid_size=2, global_weight=0.25
    )

    assert np.allclose(probabilities.sum(axis=0), 1.0, atol=1e-6)


def test_finer_tiles_keep_each_small_region_separate():
    class TileModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, _tensor):
            logits = torch.zeros((1, 8, 2, 2))
            logits[:, self.calls % 8] = 2.0
            self.calls += 1
            return logits

    tiled_predict = _function(
        "predict_large_image_with_tiles",
        {
            "np": np,
            "torch": torch,
            "Image": Image,
            "IMAGE_SIZE": (8, 8),
            "NUM_CLASSES": 8,
            "DEVICE": torch.device("cpu"),
            "prepare_image_tensor": lambda _image: torch.zeros((1, 3, 2, 2)),
        },
    )

    probabilities = tiled_predict(
        TileModel(), Image.new("RGB", (64, 64)), grid_size=4, global_weight=0.0
    )

    assert np.argmax(probabilities, axis=0)[1, 1] == 0
    assert np.argmax(probabilities, axis=0)[1, 3] == 1
    assert np.argmax(probabilities, axis=0)[3, 1] == 4
    assert np.argmax(probabilities, axis=0)[3, 3] == 5


def test_class_thresholds_use_class_channel_not_image_row():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "advanced.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_class_specific_thresholds"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)

    probs = np.zeros((8, 1, 2), dtype=np.float32)
    probs[4, 0, 0] = 0.90  # Water exceeds its 0.85 threshold.
    probs[6, 0, 1] = 0.20  # Forest exceeds its 0.15 threshold.
    refined = namespace["apply_class_specific_thresholds"](probs)

    assert refined.tolist() == [[4, 6]]


def test_morphology_preserves_valid_small_land_cover_regions():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "morphology.py"
    ).read_text(encoding="utf-8")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "apply_morphology"
    )
    namespace = {"np": np, "cv2": cv2, "ndi": ndi}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)

    mask = np.ones((9, 9), dtype=np.int64)
    mask[3:6, 3:6] = 2  # A compact building should not be opened away.
    mask[0, 0] = 0       # Ignore must always be preserved.
    refined = namespace["apply_morphology"](mask, max_hole_size=4)

    assert np.all(refined[3:6, 3:6] == 2)
    assert refined[0, 0] == 0


def test_segearth_verifier_does_not_overwrite_existing_land_cover_classes():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "advanced.py"
    ).read_text(encoding="utf-8")
    verifier_start = source.index("def verify_segmentation_with_segearth")
    verifier_source = source[verifier_start:source.index("def verify_background_with_segearth", verifier_start)]

    assert "backgroundish = segformer_pred == 0" in verifier_source
    assert "strong_foreground" not in verifier_source


def test_open_vocabulary_model_preserves_the_existing_viewer_class_ids():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "advanced.py"
    ).read_text(encoding="utf-8")

    for segearth_class, viewer_class in {
        0: 1, 1: 5, 2: 7, 3: 3, 4: 3, 5: 6, 6: 4, 7: 7, 8: 2,
    }.items():
        assert f"{segearth_class}: {viewer_class}" in source
    assert "def predict_open_vocabulary_landcover(" in source


def test_hybrid_fusion_only_replaces_uncertain_segformer_pixels():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "advanced.py"
    ).read_text(encoding="utf-8")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "fuse_segearth_with_segformer"
    )
    namespace = {
        "np": np,
        "Image": Image,
        "_predict_with_segearth_logits": lambda _image: (
            np.array([[5, 5]], dtype=np.int64),
            np.array([[0.95, 0.95]], dtype=np.float32),
        ),
        "_resize_mask": lambda array, _size: array,
        "_map_segearth_to_landcover": lambda array: array + 1,
        "_local_support_mask": lambda array: np.ones_like(array, dtype=bool),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)

    pred = np.array([[4, 4]], dtype=np.int64)
    probs = np.zeros((8, 1, 2), dtype=np.float32)
    probs[4] = [[0.40, 0.90]]
    fused = namespace["fuse_segearth_with_segformer"](Image.new("RGB", (2, 1)), pred, probs)

    assert fused.tolist() == [[6, 4]]
