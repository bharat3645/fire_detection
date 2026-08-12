"""Unit tests for core.py — the framework-free preprocessing / classification /
explainability logic behind the Streamlit app.

Run with:
    pytest -v
"""

import io
import json
import os

import numpy as np
import pytest
from PIL import Image

import core

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_bytes(size=(80, 60), color=(200, 30, 10), fmt="PNG"):
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class _TinyFireModel:
    """A minimal 2-conv Keras model with the same layer names / IO contract
    as fire_detection_model.h5, so Grad-CAM and prediction plumbing can be
    exercised without loading the real 19MB model file.
    """

    def __init__(self):
        import tensorflow as tf

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(64, 64, 3)),
                tf.keras.layers.Conv2D(4, 3, activation="relu", name="conv2d"),
                tf.keras.layers.MaxPooling2D(name="max_pooling2d"),
                tf.keras.layers.Conv2D(4, 3, activation="relu", name="conv2d_1"),
                tf.keras.layers.MaxPooling2D(name="max_pooling2d_1"),
                tf.keras.layers.Flatten(name="flatten"),
                tf.keras.layers.Dense(8, activation="relu", name="dense"),
                tf.keras.layers.Dense(1, activation="sigmoid", name="dense_1"),
            ]
        )
        self._model = model
        self.layers = model.layers

    def predict(self, x, verbose=0):
        return self._model.predict(x, verbose=verbose)


# ---------------------------------------------------------------------------
# validate_file_size
# ---------------------------------------------------------------------------

def test_validate_file_size_accepts_normal_file():
    core.validate_file_size(b"x" * 1024, max_size_mb=10)  # should not raise


def test_validate_file_size_rejects_empty_file():
    with pytest.raises(core.ImageValidationError, match="empty"):
        core.validate_file_size(b"")


def test_validate_file_size_rejects_oversized_file():
    too_big = b"x" * (2 * 1024 * 1024)
    with pytest.raises(core.ImageValidationError, match="exceeds"):
        core.validate_file_size(too_big, max_size_mb=1)


# ---------------------------------------------------------------------------
# load_image_from_bytes
# ---------------------------------------------------------------------------

def test_load_image_from_bytes_decodes_valid_image():
    data = _make_image_bytes()
    image = core.load_image_from_bytes(data)
    assert image.mode == "RGB"
    assert image.size == (80, 60)


def test_load_image_from_bytes_rejects_garbage_bytes():
    with pytest.raises(core.ImageValidationError):
        core.load_image_from_bytes(b"this is not an image, just text pretending to be one")


def test_load_image_from_bytes_converts_non_rgb_to_rgb():
    img = Image.new("RGBA", (10, 10), (1, 2, 3, 4))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = core.load_image_from_bytes(buf.getvalue())
    assert result.mode == "RGB"


def test_load_image_from_bytes_applies_exif_orientation():
    # A landscape 60x30 image tagged "needs a 90-degree rotation to display
    # upright" must come back portrait (30x60). Without normalizing this,
    # real-world phone/drone photos get fed to the model sideways while
    # looking perfectly upright in every image viewer.
    img = Image.new("RGB", (60, 30), (200, 30, 10))
    exif = img.getexif()
    exif[0x0112] = 6  # Orientation tag: rotate 90 CW needed
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)

    result = core.load_image_from_bytes(buf.getvalue())
    assert result.size == (30, 60)


def test_load_image_from_bytes_handles_missing_exif_gracefully():
    data = _make_image_bytes(size=(80, 60))
    image = core.load_image_from_bytes(data)
    assert image.size == (80, 60)


# ---------------------------------------------------------------------------
# preprocess_image
# ---------------------------------------------------------------------------

def test_preprocess_image_shape_and_range():
    image = Image.new("RGB", (200, 150), (255, 128, 0))
    array = core.preprocess_image(image)
    assert array.shape == (1, 64, 64, 3)
    assert array.dtype == np.float32
    assert array.min() >= 0.0
    assert array.max() <= 1.0


def test_preprocess_image_normalizes_known_pixel_value():
    image = Image.new("RGB", (64, 64), (255, 0, 0))
    array = core.preprocess_image(image)
    assert np.allclose(array[0, 0, 0], [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# classify — the exact logic that was previously broken (argmax on a
# single-neuron sigmoid output always returned index 0 / "No Fire").
# ---------------------------------------------------------------------------

def test_classify_high_probability_is_fire():
    result = core.classify(0.93)
    assert result.predicted_class == 1
    assert result.label == "Fire"
    assert result.confidence_pct == pytest.approx(93.0)


def test_classify_low_probability_is_no_fire():
    result = core.classify(0.02)
    assert result.predicted_class == 0
    assert result.label == "No Fire"
    assert result.confidence_pct == pytest.approx(98.0)


def test_classify_respects_custom_threshold():
    # 0.4 would be "No Fire" at the default 0.5 threshold, but "Fire" at 0.3.
    assert core.classify(0.4, threshold=0.5).label == "No Fire"
    assert core.classify(0.4, threshold=0.3).label == "Fire"


def test_classify_boundary_at_threshold_is_positive_class():
    result = core.classify(0.5, threshold=0.5)
    assert result.predicted_class == 1


def test_classify_rejects_out_of_range_probability():
    with pytest.raises(ValueError):
        core.classify(1.5)
    with pytest.raises(ValueError):
        core.classify(-0.1)


# ---------------------------------------------------------------------------
# predict_fire_probability + end-to-end with a tiny real Keras model
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_model():
    return _TinyFireModel()


def test_predict_fire_probability_returns_float_in_range(tiny_model):
    img_array = np.random.rand(1, 64, 64, 3).astype(np.float32)
    prob = core.predict_fire_probability(tiny_model, img_array)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

def test_make_gradcam_heatmap_shape_and_range(tiny_model):
    img_array = np.random.rand(1, 64, 64, 3).astype(np.float32)
    heatmap = core.make_gradcam_heatmap(img_array, tiny_model, last_conv_layer_name="conv2d_1")
    assert heatmap.ndim == 2
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6


def test_make_gradcam_heatmap_raises_on_unknown_layer(tiny_model):
    img_array = np.random.rand(1, 64, 64, 3).astype(np.float32)
    with pytest.raises(ValueError, match="not found"):
        core.make_gradcam_heatmap(img_array, tiny_model, last_conv_layer_name="does_not_exist")


def test_overlay_heatmap_on_image_matches_base_size():
    image = Image.new("RGB", (100, 50), (10, 20, 30))
    heatmap = np.random.rand(8, 8).astype(np.float32)
    overlay = core.overlay_heatmap_on_image(image, heatmap, alpha=0.5)
    assert overlay.size == image.size
    assert overlay.mode == "RGB"


def test_overlay_heatmap_handles_all_zero_heatmap():
    # Regression guard: an all-zero heatmap must not raise (e.g. via div-by-zero).
    image = Image.new("RGB", (32, 32), (5, 5, 5))
    heatmap = np.zeros((4, 4), dtype=np.float32)
    overlay = core.overlay_heatmap_on_image(image, heatmap)
    assert overlay.size == image.size


# ---------------------------------------------------------------------------
# Integration test against the real shipped model (skipped if not present)
# ---------------------------------------------------------------------------

_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fire_detection_model.h5")


@pytest.mark.skipif(not os.path.exists(_MODEL_PATH), reason="fire_detection_model.h5 not present")
def test_real_model_end_to_end_pipeline():
    from tensorflow.keras.models import load_model

    model = load_model(_MODEL_PATH)
    image = Image.new("RGB", (64, 64), (255, 60, 0))  # fire-orange solid image
    img_array = core.preprocess_image(image)

    prob = core.predict_fire_probability(model, img_array)
    assert 0.0 <= prob <= 1.0

    result = core.classify(prob)
    assert result.label in core.CLASS_NAMES

    heatmap = core.make_gradcam_heatmap(img_array, model)
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0


@pytest.mark.skipif(not os.path.exists(_MODEL_PATH), reason="fire_detection_model.h5 not present")
def test_load_fire_model_returns_a_usable_model():
    # This is the loader shared by app.py, api.py, and cli.py — a regression
    # here would silently break all three entry points at once.
    model = core.load_fire_model(_MODEL_PATH)
    assert hasattr(model, "predict")


def test_load_fire_model_raises_on_missing_file(tmp_path):
    with pytest.raises(Exception):  # noqa: B017 - Keras raises its own IOError/ValueError variants
        core.load_fire_model(str(tmp_path / "does_not_exist.h5"))


# ---------------------------------------------------------------------------
# log_prediction
# ---------------------------------------------------------------------------

def test_log_prediction_appends_jsonl_lines(tmp_path):
    log_path = tmp_path / "nested" / "predictions.jsonl"  # nested dir must be auto-created

    core.log_prediction({"source": "test", "prediction": "Fire"}, log_path=str(log_path))
    core.log_prediction({"source": "test", "prediction": "No Fire"}, log_path=str(log_path))

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    first = json.loads(lines[0])
    assert first["prediction"] == "Fire"
    assert "timestamp" in first


def test_log_prediction_never_raises_when_parent_path_is_not_a_directory(tmp_path):
    # Logging is best-effort: a broken log destination must never crash the
    # caller (Streamlit app / API / CLI), since the prediction it's logging
    # already succeeded.
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("x")
    bad_log_path = blocking_file / "predictions.jsonl"

    core.log_prediction({"source": "test"}, log_path=str(bad_log_path))  # must not raise
