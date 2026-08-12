"""REST API for the fire-detection model, built on FastAPI.

Gives the model a programmatic interface alongside the Streamlit UI, so it
can be called from a script, a cron job, or another service instead of a
human clicking a file uploader — the "REST API endpoint" item from the
project roadmap.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000

Then either use the interactive docs at http://localhost:8000/docs, or:
    curl -F "file=@image.jpg" "http://localhost:8000/predict?threshold=0.5"

Environment variables:
    FIRE_MODEL_PATH         Path to the .h5 model (default: fire_detection_model.h5)
    FIRE_DETECTION_LOG_PATH Path to the JSONL prediction audit log (see core.log_prediction)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from core import (
    CLASS_NAMES,
    INPUT_SHAPE,
    MAX_UPLOAD_SIZE_MB,
    ImageValidationError,
    classify,
    load_fire_model,
    load_image_from_bytes,
    log_prediction,
    predict_fire_probability,
    preprocess_image,
    validate_file_size,
)

_model = None
_model_error: str | None = None


def _resolve_model_path() -> str:
    return os.environ.get("FIRE_MODEL_PATH", "fire_detection_model.h5")


def get_model():
    """Lazily load and cache the model as a module-level singleton.

    Deliberately not a Streamlit ``@st.cache_resource`` (this module has no
    Streamlit dependency) and not ``functools.lru_cache`` (so a failed load
    can be retried on a later request instead of caching the exception).
    """
    global _model, _model_error
    if _model is None and _model_error is None:
        try:
            _model = load_fire_model(_resolve_model_path())
        except Exception as exc:  # noqa: BLE001 - surfaced via /health and 503s
            _model_error = str(exc)
    return _model


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_model()  # warm the model once at startup instead of on the first request
    yield


app = FastAPI(
    title="Fire Detection API",
    description=(
        "Upload a satellite/aerial image and get back a Fire / No Fire verdict "
        "from the pretrained CNN, with confidence and the raw P(Fire) score."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    error: str | None = None


class PredictResponse(BaseModel):
    filename: str
    prediction: str
    predicted_class: int
    fire_probability: float
    confidence_pct: float
    threshold: float


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Report whether the model is loaded and ready to serve predictions."""
    model = get_model()
    return HealthResponse(
        status="ok" if model is not None else "model_unavailable",
        model_loaded=model is not None,
        error=_model_error,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(
    file: UploadFile = File(..., description="A JPG or PNG image."),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="P(Fire) at or above this value is classified as Fire."),
) -> PredictResponse:
    """Classify one uploaded image as Fire or No Fire."""
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {_model_error}")

    file_bytes = await file.read()
    try:
        validate_file_size(file_bytes)
        image = load_image_from_bytes(file_bytes)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        img_array = preprocess_image(image, target_size=INPUT_SHAPE)
        fire_probability = predict_fire_probability(model, img_array)
        result = classify(fire_probability, threshold=threshold, class_names=CLASS_NAMES)
    except Exception as exc:  # noqa: BLE001 - turned into a clean 500 for API clients
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    log_prediction(
        {
            "source": "api",
            "filename": file.filename,
            "prediction": result.label,
            "predicted_class": result.predicted_class,
            "fire_probability": result.fire_probability,
            "confidence_pct": result.confidence_pct,
            "threshold": threshold,
        }
    )

    return PredictResponse(
        filename=file.filename or "unknown",
        prediction=result.label,
        predicted_class=result.predicted_class,
        fire_probability=round(result.fire_probability, 6),
        confidence_pct=round(result.confidence_pct, 2),
        threshold=threshold,
    )


@app.get("/", tags=["meta"])
def root() -> dict:
    """Basic service info, plus a pointer to the interactive docs."""
    return {
        "service": "fire-detection-api",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "max_upload_size_mb": MAX_UPLOAD_SIZE_MB,
    }
