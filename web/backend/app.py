"""
app.py
======
Flask API server for the QGIS-inspired land cover segmentation viewer.

Endpoints
---------
GET /api/images                         → list all test image filenames
GET /api/images/<filename>              → serve resized satellite image as PNG
GET /api/masks/<filename>               → serve colored GT mask as PNG
GET /api/predict/<model>/<filename>     → run inference, return colored mask PNG
POST /api/predict/upload                → run inference on uploaded image, return colored mask PNG
GET /api/metrics/image/<model>/<filename> → per-image mIoU + IoU per class
GET /api/metrics/global                 → pre-computed evaluation metrics JSON
GET /api/compare/<filename>             → both predictions + per-image metrics
GET /api/reports/confusion/<model>      → serve confusion matrix PNG

Start:
    cd web/backend
    python app.py
"""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file
from flask_cors import CORS
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

DATA_DIR    = PROJECT_ROOT / "data" / "splits" / "test"
IMAGES_DIR  = DATA_DIR / "images"
MASKS_DIR   = DATA_DIR / "masks"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Lazy import inference to avoid loading models at import time
from inference import (
    CLASS_NAMES,
    colorize_mask,
    compute_per_image_metrics,
    load_mask,
    predict_from_image,
    predict_image,
    serve_pil_image,
    serve_raw_image,
)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/images", methods=["GET"])
def list_images():
    """Return a sorted list of all test image filenames."""
    files = sorted(f.name for f in IMAGES_DIR.glob("*.png"))
    return jsonify({"images": files, "total": len(files)})


@app.route("/api/images/<filename>", methods=["GET"])
def get_image(filename: str):
    """Serve a resized satellite image as PNG."""
    path = IMAGES_DIR / filename
    if not path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404

    png_bytes = serve_raw_image(path)
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/masks/<filename>", methods=["GET"])
def get_mask(filename: str):
    """Serve the ground-truth mask as a colored semi-transparent PNG."""
    path = MASKS_DIR / filename
    if not path.exists():
        return jsonify({"error": f"Mask not found: {filename}"}), 404

    mask = load_mask(path)
    png_bytes = colorize_mask(mask)
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/predict/<model>/<filename>", methods=["GET"])
def get_prediction(model: str, filename: str):
    """Run inference and return the colored prediction mask as PNG."""
    if model not in ("unet", "segformer"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404

    postprocess = request.args.get("postprocess", "false").lower() == "true"

    pred = predict_image(model, image_path, postprocess=postprocess)
    png_bytes = colorize_mask(pred)
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/predict/upload", methods=["POST"])
def predict_upload():
    """Run inference on an uploaded image and return the colored prediction mask."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(uploaded.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "error": f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }), 400

    model = request.form.get("model", "unet")
    if model not in ("unet", "segformer"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    postprocess = request.form.get("postprocess", "false").lower() == "true"

    raw = uploaded.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}), 400
    if len(raw) == 0:
        return jsonify({"error": "Empty file"}), 400

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as e:
        return jsonify({"error": f"Could not read image file: {str(e)}"}), 400

    try:
        pred = predict_from_image(model, img, postprocess=postprocess)
        png_bytes = colorize_mask(pred)
        return Response(png_bytes, mimetype="image/png")
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error during prediction with {model}: {error_detail}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/upload/preview", methods=["POST"])
def upload_preview():
    """Return a resized preview of an uploaded image (matches model input size)."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded = request.files["file"]
    if not uploaded.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = Path(uploaded.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    raw = uploaded.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "File too large"}), 400

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    png_bytes = serve_pil_image(img)
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/metrics/image/<model>/<filename>", methods=["GET"])
def get_image_metrics(model: str, filename: str):
    """Compute per-image metrics (mIoU, pixel accuracy, IoU per class)."""
    if model not in ("unet", "segformer"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    image_path = IMAGES_DIR / filename
    mask_path  = MASKS_DIR  / filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404
    if not mask_path.exists():
        return jsonify({"error": f"Mask not found: {filename}"}), 404

    postprocess = request.args.get("postprocess", "false").lower() == "true"

    pred = predict_image(model, image_path, postprocess=postprocess)
    gt   = load_mask(mask_path)
    metrics = compute_per_image_metrics(pred, gt)
    metrics["model"] = model
    metrics["filename"] = filename
    return jsonify(metrics)


@app.route("/api/compare/<filename>", methods=["GET"])
def compare_models(filename: str):
    """Return predictions + metrics from both models for side-by-side comparison."""
    image_path = IMAGES_DIR / filename
    mask_path  = MASKS_DIR  / filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404
    if not mask_path.exists():
        return jsonify({"error": f"Mask not found: {filename}"}), 404

    postprocess = request.args.get("postprocess", "false").lower() == "true"
    gt = load_mask(mask_path)

    results = {}
    for model_name in ("unet", "segformer"):
        pred = predict_image(model_name, image_path, postprocess=postprocess)
        metrics = compute_per_image_metrics(pred, gt)
        metrics["model"] = model_name
        results[model_name] = metrics

    return jsonify({"filename": filename, "results": results})


@app.route("/api/metrics/global", methods=["GET"])
def get_global_metrics():
    """Return pre-computed evaluation metrics from outputs/reports/."""
    output = {}
    for model_name in ("unet", "segformer"):
        path = REPORTS_DIR / f"metrics_{model_name}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                output[model_name] = json.load(f)
        else:
            output[model_name] = None
    return jsonify(output)


@app.route("/api/reports/confusion/<model>", methods=["GET"])
def get_confusion_matrix(model: str):
    """Serve the confusion matrix PNG."""
    if model not in ("unet", "segformer"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    path = REPORTS_DIR / f"confusion_{model}.png"
    if not path.exists():
        return jsonify({"error": f"Confusion matrix not found for {model}"}), 404

    return send_file(path, mimetype="image/png")


@app.route("/api/classes", methods=["GET"])
def get_classes():
    """Return class names and colors."""
    return jsonify({
        "classes": [
            {"id": i, "name": name}
            for i, name in enumerate(CLASS_NAMES)
        ]
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n  Project root : {PROJECT_ROOT}")
    print(f"  Images dir   : {IMAGES_DIR}")
    print(f"  Masks dir    : {MASKS_DIR}")
    print(f"  Reports dir  : {REPORTS_DIR}")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)
