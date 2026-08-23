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
import torch
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
BACKEND_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parents[1]
SRC_ROOT     = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

DATA_DIR    = PROJECT_ROOT / "data" / "splits" / "test"
IMAGES_DIR  = DATA_DIR / "images"
MASKS_DIR   = DATA_DIR / "masks"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CORRECTIONS_DIR = PROJECT_ROOT / "data" / "corrections"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# Lazy import inference to avoid loading models at import time
from inference import (
    CLASS_NAMES,
    IMAGE_SIZE,
    calibrate_landcover_probabilities,
    colorize_mask,
    compute_per_image_metrics,
    load_mask,
    predict_from_image,
    predict_image,
    resolve_field_like_water_components,
    resolve_forest_water_ambiguity,
    serve_pil_image,
    serve_raw_image,
)
from tta import TestTimeAugmentation
from corrections import (
    apply_corrections_to_mask,
    image_key_for_bytes,
    load_corrections,
    save_correction,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB


def apply_saved_upload_corrections(prediction: np.ndarray, image_key: str) -> np.ndarray:
    """Overlay corrections previously made for the exact uploaded image."""
    corrections = load_corrections(CORRECTIONS_DIR, image_key)
    if not corrections:
        return prediction
    return apply_corrections_to_mask(prediction, corrections, num_classes=len(CLASS_NAMES))


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
    if model != "segformer":
        return jsonify({"error": f"Unknown model: {model}"}), 400

    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404

    postprocess = request.args.get("postprocess", "false").lower() == "true"

    pred = predict_image(model, image_path, postprocess=postprocess)
    png_bytes = colorize_mask(pred)
    return Response(png_bytes, mimetype="image/png")


@app.route("/api/corrections/upload", methods=["POST"])
def save_upload_correction():
    """Save a click or rectangular class correction for an uploaded image."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    raw = request.files["file"].read()
    if not raw:
        return jsonify({"error": "Empty file"}), 400
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}), 400

    try:
        class_id = int(request.form["class_id"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "A valid class is required"}), 400
    if not 0 <= class_id < len(CLASS_NAMES):
        return jsonify({"error": "Unknown class"}), 400

    if "points" in request.form:
        try:
            points = json.loads(request.form["points"])
            radius = int(request.form.get("radius", "12"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return jsonify({"error": "A valid brush stroke is required"}), 400
        if not isinstance(points, list) or not 1 <= len(points) <= 2000:
            return jsonify({"error": "Brush stroke must contain between 1 and 2000 points"}), 400
        if not 1 <= radius <= 64:
            return jsonify({"error": "Brush radius must be between 1 and 64 pixels"}), 400
        try:
            normalized_points = [{"x": float(point["x"]), "y": float(point["y"])} for point in points]
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "Brush stroke contains an invalid point"}), 400
        if not all(0.0 <= point["x"] <= 1.0 and 0.0 <= point["y"] <= 1.0 for point in normalized_points):
            return jsonify({"error": "Brush stroke must be within the image"}), 400
        correction = {"type": "stroke", "points": normalized_points, "class_id": class_id, "radius": radius}
    elif all(field in request.form for field in ("x1", "y1", "x2", "y2")):
        try:
            x1 = float(request.form["x1"])
            y1 = float(request.form["y1"])
            x2 = float(request.form["x2"])
            y2 = float(request.form["y2"])
        except (TypeError, ValueError):
            return jsonify({"error": "A valid selected area is required"}), 400
        if not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
            return jsonify({"error": "Selected area must be within the image"}), 400
        correction = {"type": "rectangle", "x1": x1, "y1": y1, "x2": x2, "y2": y2, "class_id": class_id}
    else:
        try:
            x = float(request.form["x"])
            y = float(request.form["y"])
            radius = int(request.form.get("radius", "8"))
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "A valid position or selected area is required"}), 400
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return jsonify({"error": "Click position must be within the image"}), 400
        if not 1 <= radius <= 64:
            return jsonify({"error": "Brush radius must be between 1 and 64 pixels"}), 400
        correction = {"x": x, "y": y, "class_id": class_id, "radius": radius}

    image_key = image_key_for_bytes(raw)
    corrections = save_correction(CORRECTIONS_DIR, image_key, correction)
    return jsonify({"image_key": image_key, "count": len(corrections)})


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

    model = request.form.get("model", "segformer")
    if model not in ("segformer", "segearth", "hybrid"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    postprocess = request.form.get("postprocess", "true").lower() == "true"  # Default to true
    use_tta = request.form.get("tta", "true").lower() == "true"  # Default to true for better quality
    use_advanced = request.form.get("advanced", "true").lower() == "true"  # Default to true
    use_crf = request.form.get("crf", "false").lower() == "true"
    confidence_threshold = float(request.form.get("confidence_threshold", "0.6"))  # Default 0.6
    use_multi_scale = request.form.get("multi_scale", "true").lower() == "true"  # Multi-scale for complex images
    temperature = float(request.form.get("temperature", "0.8"))  # Default 0.8 for sharper predictions
    use_api_verify = request.form.get("api_verify", "false").lower() == "true"
    use_background_verify = request.form.get("background_verify", "false").lower() == "true"

    # Refinement controls are a SegFormer-only workflow.  Other model choices
    # retain their own prediction paths without accepting SegFormer refinements.
    if model != "segformer":
        postprocess = False
        use_tta = False
        use_advanced = False
        use_crf = False
        use_api_verify = False
        use_background_verify = False

    raw = uploaded.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}), 400
    if len(raw) == 0:
        return jsonify({"error": "Empty file"}), 400
    image_key = image_key_for_bytes(raw)

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as e:
        return jsonify({"error": f"Could not read image file: {str(e)}"}), 400

    try:
        if model == "hybrid":
            from postprocessing.advanced import fuse_segearth_with_segformer

            pred, probs = predict_from_image(
                "segformer",
                img,
                postprocess=postprocess,
                use_crf=use_crf,
                use_advanced=use_advanced,
                confidence_threshold=confidence_threshold,
                use_multi_scale=use_multi_scale,
                temperature=temperature,
                return_probabilities=True,
            )
            pred = fuse_segearth_with_segformer(img, pred, probs)
        elif model == "segearth":
            from postprocessing.advanced import predict_open_vocabulary_landcover

            pred = predict_open_vocabulary_landcover(
                img, output_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
            )
            if pred is None:
                return jsonify({
                    "error": "General imagery model could not initialize. Check the SegEarth-OV installation."
                }), 503
        elif use_tta:
            from inference import get_model
            model_instance = get_model(model)
            tta = TestTimeAugmentation(model_instance, DEVICE)
            probs = tta.predict_proba_with_tta(img)
            probs = calibrate_landcover_probabilities(probs)
            pred = resolve_forest_water_ambiguity(np.argmax(probs, axis=0), probs, image=img)
            pred = resolve_field_like_water_components(pred, probs, image=img)
        else:
            # Enable CRF and advanced post-processing by default
            pred = predict_from_image(
                model, 
                img, 
                postprocess=postprocess,  # Enable morphology
                use_crf=use_crf,          # Enable CRF post-processing
                use_advanced=use_advanced,  # Enable advanced post-processing
                confidence_threshold=confidence_threshold,  # Set confidence threshold
                use_multi_scale=use_multi_scale,  # Enable multi-scale for complex images
                temperature=temperature  # Apply temperature scaling for sharper predictions
            )

        if use_api_verify or use_background_verify:
            from postprocessing.advanced import (
                verify_background_with_segearth,
                verify_segmentation_with_segearth,
            )

            if use_api_verify:
                pred = verify_segmentation_with_segearth(img, pred)
            if use_background_verify:
                pred = verify_background_with_segearth(img, pred)

        pred = apply_saved_upload_corrections(pred, image_key)
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
    if model != "segformer":
        return jsonify({"error": f"Unknown model: {model}"}), 400

    image_path = IMAGES_DIR / filename
    mask_path  = MASKS_DIR  / filename
    if not image_path.exists():
        return jsonify({"error": f"Image not found: {filename}"}), 404
    if not mask_path.exists():
        return jsonify({"error": f"Mask not found: {filename}"}), 404

    postprocess = request.args.get("postprocess", "true").lower() == "true"  # Default to true
    use_crf = request.args.get("use_crf", "true").lower() == "true"  # Default to true
    use_advanced = request.args.get("use_advanced", "true").lower() == "true"  # Default to true
    confidence_threshold = float(request.args.get("confidence_threshold", "0.6"))  # Default 0.6
    use_multi_scale = request.args.get("multi_scale", "true").lower() == "true"  # Multi-scale for complex images
    temperature = float(request.args.get("temperature", "0.8"))  # Default 0.8 for sharper predictions

    pred = predict_image(
        model, 
        image_path, 
        postprocess=postprocess,
        use_crf=use_crf,
        use_advanced=use_advanced,
        confidence_threshold=confidence_threshold,
        use_multi_scale=use_multi_scale,
        temperature=temperature
    )
    gt   = load_mask(mask_path)
    metrics = compute_per_image_metrics(pred, gt)
    metrics["model"] = model
    metrics["filename"] = filename
    return jsonify(metrics)


@app.route("/api/metrics/global", methods=["GET"])
def get_global_metrics():
    """Return pre-computed evaluation metrics from outputs/reports/."""
    output = {}
    for model_name in ("segformer",):
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
    if model != "segformer":
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


@app.route("/api/export/prediction", methods=["POST"])
def export_prediction():
    """Export the prediction mask as a downloadable PNG file with original image comparison."""
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

    model = request.form.get("model", "segformer")
    if model not in ("segformer", "segearth", "hybrid"):
        return jsonify({"error": f"Unknown model: {model}"}), 400

    postprocess = request.form.get("postprocess", "true").lower() == "true"
    use_tta = request.form.get("tta", "true").lower() == "true"
    use_advanced = request.form.get("advanced", "true").lower() == "true"
    use_crf = request.form.get("crf", "false").lower() == "true"
    confidence_threshold = float(request.form.get("confidence_threshold", "0.6"))
    use_multi_scale = request.form.get("multi_scale", "true").lower() == "true"
    temperature = float(request.form.get("temperature", "0.8"))

    if model != "segformer":
        postprocess = False
        use_tta = False
        use_advanced = False
        use_crf = False

    raw = uploaded.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        return jsonify({"error": f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}), 400
    if len(raw) == 0:
        return jsonify({"error": "Empty file"}), 400
    image_key = image_key_for_bytes(raw)

    try:
        img = Image.open(BytesIO(raw))
        img.load()
    except Exception as e:
        return jsonify({"error": f"Could not read image file: {str(e)}"}), 400

    try:
        if model == "hybrid":
            from postprocessing.advanced import fuse_segearth_with_segformer

            pred, probs = predict_from_image(
                "segformer",
                img,
                postprocess=postprocess,
                use_crf=use_crf,
                use_advanced=use_advanced,
                confidence_threshold=confidence_threshold,
                use_multi_scale=use_multi_scale,
                temperature=temperature,
                return_probabilities=True,
            )
            pred = fuse_segearth_with_segformer(img, pred, probs)
        elif model == "segearth":
            from postprocessing.advanced import predict_open_vocabulary_landcover

            pred = predict_open_vocabulary_landcover(
                img, output_size=(IMAGE_SIZE[1], IMAGE_SIZE[0])
            )
            if pred is None:
                return jsonify({
                    "error": "General imagery model could not initialize. Check the SegEarth-OV installation."
                }), 503
        elif use_tta:
            from inference import get_model
            model_instance = get_model(model)
            tta = TestTimeAugmentation(model_instance, DEVICE)
            probs = tta.predict_proba_with_tta(img)
            probs = calibrate_landcover_probabilities(probs)
            pred = resolve_forest_water_ambiguity(np.argmax(probs, axis=0), probs, image=img)
            pred = resolve_field_like_water_components(pred, probs, image=img)
        else:
            pred = predict_from_image(
                model, 
                img, 
                postprocess=postprocess,
                use_crf=use_crf,
                use_advanced=use_advanced,
                confidence_threshold=confidence_threshold,
                use_multi_scale=use_multi_scale,
                temperature=temperature
            )

        pred = apply_saved_upload_corrections(pred, image_key)
        
        # Create comparison image (original + segmentation side by side)
        # Resize original image to match prediction size
        original_resized = img.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.LANCZOS)
        
        # Create colored mask from prediction
        mask_bytes = colorize_mask(pred)
        mask_rgba = Image.open(BytesIO(mask_bytes))
        
        # Create side-by-side comparison with small separator
        width, height = IMAGE_SIZE[1], IMAGE_SIZE[0]
        separator_width = 10
        comparison = Image.new('RGB', (width * 2 + separator_width, height), (255, 255, 255))
        
        # Paste original image on left
        comparison.paste(original_resized, (0, 0))
        
        # Paste segmentation mask on right
        comparison.paste(mask_rgba.convert('RGB'), (width + separator_width, 0))
        
        # Convert to bytes
        buf = BytesIO()
        comparison.save(buf, format='PNG')
        png_bytes = buf.getvalue()
        
        return Response(
            png_bytes, 
            mimetype="image/png"
        )
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error during export with {model}: {error_detail}")
        return jsonify({"error": f"Export failed: {str(e)}"}), 500


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
