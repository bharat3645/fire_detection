"""Core, framework-free logic for the Fire Detection app.

Kept separate from app.py (which owns all Streamlit/UI concerns) so that
preprocessing, validation, classification, and Grad-CAM explainability can
be unit-tested without booting a Streamlit session or touching the network.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Matches the saved model's input shape: (None, 64, 64, 3).
INPUT_SHAPE = (64, 64)

# index 0 = "No Fire", index 1 = "Fire" — the model's single sigmoid output
# neuron is P(class 1) = P(Fire).
CLASS_NAMES = ["No Fire", "Fire"]

# Generous but bounded, to stop someone from uploading a 500MB "image" and
# hanging the app's memory/CPU during resize/inference.
MAX_UPLOAD_SIZE_MB = 10

# Name of the last convolutional layer in fire_detection_model.h5, used as
# the activation map for Grad-CAM. Verified against the shipped model via
# model.summary(): conv2d -> maxpool -> conv2d_1 -> maxpool -> flatten -> ...
LAST_CONV_LAYER_NAME = "conv2d_1"

# Where prediction audit records are appended (one JSON object per line),
# overridable per-deployment (e.g. a mounted volume in Docker) via the
# FIRE_DETECTION_LOG_PATH env var. This fallback is intentionally *not*
# resolved from the environment here at import time: core.py can end up
# imported (directly or transitively) before a caller has finished setting
# up its environment — e.g. pytest imports every test module during
# collection, before any fixture runs — which would otherwise freeze this
# constant to the wrong value for the rest of the process. log_prediction()
# below reads the env var itself, at call time, instead.
DEFAULT_LOG_PATH = os.path.join("logs", "predictions.jsonl")


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be used as model input."""


# ---------------------------------------------------------------------------
# Validation + preprocessing
# ---------------------------------------------------------------------------

def validate_file_size(file_bytes: bytes, max_size_mb: float = MAX_UPLOAD_SIZE_MB) -> None:
    """Raise ImageValidationError if the uploaded bytes are empty or too large."""
    if not file_bytes:
        raise ImageValidationError("Uploaded file is empty.")
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ImageValidationError(
            f"File is {size_mb:.1f} MB, which exceeds the {max_size_mb:g} MB limit. "
            "Please upload a smaller image."
        )


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL Image, raising a friendly error on failure.

    Catches truncated files, non-image files (e.g. a renamed .pdf), and
    zero-byte payloads, all of which crash a bare ``Image.open`` call.

    Also normalizes EXIF orientation (``ImageOps.exif_transpose``) before
    returning. Phone/drone photos frequently store pixel data "as captured"
    plus an EXIF orientation tag telling viewers how to rotate it; skipping
    this step silently feeds the model a sideways or upside-down image while
    every *displayed* preview of the same file looks upright.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # force full decode now so corrupt/truncated data is caught here
    except UnidentifiedImageError as exc:
        raise ImageValidationError(
            "This file isn't a recognizable image format (expected JPG/PNG)."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        raise ImageValidationError(f"Could not read this image file: {exc}") from exc

    image = ImageOps.exif_transpose(image)  # no-op if there's no orientation tag
    return image.convert("RGB")


def preprocess_image(image: Image.Image, target_size: tuple[int, int] = INPUT_SHAPE) -> np.ndarray:
    """Resize + normalize a PIL image into the model's (1, H, W, 3) float input."""
    resized = image.resize(target_size)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    expected_shape = (*target_size, 3)
    if array.shape != expected_shape:
        raise ImageValidationError(
            f"Unexpected image shape {array.shape} after conversion; expected {expected_shape}. "
            "Is this a valid RGB image?"
        )
    return array.reshape(1, *target_size, 3)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_fire_model(model_path: str = "fire_detection_model.h5"):
    """Load the pretrained Keras fire-detection model from disk.

    Framework-free (no Streamlit dependency) so the same loader can back the
    Streamlit app, the REST API, and the CLI without duplicating this logic
    three times. Callers own their own caching strategy (e.g. Streamlit's
    ``st.cache_resource``, or a plain module-level singleton for api.py/cli.py).
    """
    from tensorflow.keras.models import load_model

    return load_model(model_path)


# ---------------------------------------------------------------------------
# Inference + classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    label: str
    predicted_class: int
    fire_probability: float
    confidence_pct: float


def predict_fire_probability(model, img_array: np.ndarray) -> float:
    """Run the model and return P(Fire) as a plain float."""
    prediction = model.predict(img_array, verbose=0)
    return float(prediction[0][0])


def classify(
    fire_probability: float,
    threshold: float = 0.5,
    class_names: list[str] | None = None,
) -> ClassificationResult:
    """Turn a raw sigmoid probability into a labeled result with confidence.

    The model has a single sigmoid output neuron, so ``fire_probability`` IS
    P(Fire) directly (never argmax it — with one output that's always 0).
    """
    if not 0.0 <= fire_probability <= 1.0:
        raise ValueError(f"fire_probability must be in [0, 1], got {fire_probability!r}")

    class_names = class_names if class_names is not None else CLASS_NAMES

    predicted_class = 1 if fire_probability >= threshold else 0
    confidence = fire_probability * 100 if predicted_class == 1 else (1 - fire_probability) * 100

    return ClassificationResult(
        label=class_names[predicted_class],
        predicted_class=predicted_class,
        fire_probability=fire_probability,
        confidence_pct=confidence,
    )


# ---------------------------------------------------------------------------
# Prediction audit logging
# ---------------------------------------------------------------------------

def log_prediction(record: dict, log_path: str | None = None) -> None:
    """Append one JSON-line audit record for a prediction.

    Shared by app.py, api.py, and cli.py so every prediction made through any
    interface leaves a consistent, greppable trail (timestamp, source,
    filename, verdict, probability, threshold) — useful for after-the-fact
    triage of what the model called and when, which a stateless demo app
    otherwise has no record of.

    Best-effort by design: a logging failure (read-only filesystem, disk
    full, bad path) is swallowed rather than raised, since a prediction that
    already succeeded should never fail *because logging it* failed.
    """
    path = log_path if log_path is not None else os.environ.get("FIRE_DETECTION_LOG_PATH", DEFAULT_LOG_PATH)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), **record}
    try:
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Grad-CAM explainability
# ---------------------------------------------------------------------------

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str = LAST_CONV_LAYER_NAME,
) -> np.ndarray:
    """Compute a Grad-CAM heatmap (values in [0, 1]) showing which regions of
    ``img_array`` most influenced the model's fire prediction.

    Rebuilds a small functional graph that reuses the loaded model's layers
    (Keras 3 Sequential models loaded via load_model don't expose a usable
    `.output` on their layers, so we re-call each layer on a fresh Input to
    get symbolic tensors for tf.GradientTape to differentiate through).
    """
    import tensorflow as tf

    inputs = tf.keras.Input(shape=img_array.shape[1:])
    x = inputs
    conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            conv_output = x
    if conv_output is None:
        raise ValueError(f"Layer '{last_conv_layer_name}' not found in model.")

    grad_model = tf.keras.Model(inputs, [conv_output, x])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("Could not compute gradients for Grad-CAM on this model.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.math.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, max_val)
    return heatmap.numpy()


# Jet-like colormap control points (position, R, G, B), 0..255.
_CMAP_POS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
_CMAP_R = np.array([0, 0, 0, 255, 255])
_CMAP_G = np.array([0, 255, 255, 255, 0])
_CMAP_B = np.array([143, 255, 0, 0, 0])


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Resize ``heatmap`` to the image size, colorize it, and alpha-blend it
    over the original image for a Grad-CAM visualization.

    Implemented with a small hand-rolled colormap (vectorized via
    ``np.interp``) so this stays dependency-free — no matplotlib required.
    """
    base = image.convert("RGB")

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(base.size, resample=Image.BILINEAR)
    heatmap_arr = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    flat = heatmap_arr.flatten()

    r = np.interp(flat, _CMAP_POS, _CMAP_R)
    g = np.interp(flat, _CMAP_POS, _CMAP_G)
    b = np.interp(flat, _CMAP_POS, _CMAP_B)
    colored = np.stack([r, g, b], axis=-1).reshape(*heatmap_arr.shape, 3).astype(np.uint8)

    overlay = Image.fromarray(colored, mode="RGB")
    return Image.blend(base, overlay, alpha=alpha)
