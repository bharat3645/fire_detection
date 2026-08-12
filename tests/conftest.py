"""Shared pytest fixtures.

Provides a tiny real Keras model with the same layer names / IO contract as
fire_detection_model.h5 (2x Conv2D+MaxPool -> Flatten -> Dense -> sigmoid),
so tests for core.py, api.py, and cli.py can exercise the full
predict -> classify -> (optionally Grad-CAM) pipeline without loading the
19MB shipped model.
"""

import pytest


def _build_tiny_model():
    import tensorflow as tf

    return tf.keras.Sequential(
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


@pytest.fixture(scope="session")
def tiny_model():
    """An in-memory tiny Keras model — fast, no disk I/O."""
    return _build_tiny_model()


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    """The same tiny model, saved to a temp .h5 file — for code that loads by path (api.py, cli.py)."""
    model = _build_tiny_model()
    path = tmp_path_factory.mktemp("model") / "tiny_fire_model.h5"
    model.save(path)
    return path
