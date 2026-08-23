"""Regression coverage for DenseCRF probability-map layout."""

import ast
from pathlib import Path

import numpy as np


def _load_probability_adapter():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "crf.py"
    ).read_text(encoding="utf-8")
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_as_chw_probabilities"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<test>", "exec"), namespace)
    return namespace["_as_chw_probabilities"]


def test_crf_probability_adapter_preserves_channels_first_shape():
    adapter = _load_probability_adapter()
    probs = np.zeros((8, 2, 3), dtype=np.float32)
    probs[4] = 1.0

    result = adapter(probs, 8, 2, 3)

    assert result.shape == (8, 2, 3)
    assert np.allclose(result.sum(axis=0), 1.0)
    assert np.all(np.argmax(result, axis=0) == 4)


def test_crf_probability_adapter_accepts_hwc_without_changing_classes():
    adapter = _load_probability_adapter()
    probs = np.zeros((2, 3, 8), dtype=np.float32)
    probs[:, :, 6] = 1.0

    result = adapter(probs, 8, 2, 3)

    assert result.shape == (8, 2, 3)
    assert np.all(np.argmax(result, axis=0) == 6)


def test_dense_crf_has_a_large_image_memory_guard():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "postprocessing"
        / "crf.py"
    ).read_text(encoding="utf-8")

    assert "MAX_DENSE_CRF_PIXELS = 512 * 512" in source
    assert "if h * w > MAX_DENSE_CRF_PIXELS:" in source
