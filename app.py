import csv
import io

import streamlit as st
from PIL import Image

from core import (
    CLASS_NAMES,
    INPUT_SHAPE,
    MAX_UPLOAD_SIZE_MB,
    ImageValidationError,
    classify,
    load_fire_model,
    load_image_from_bytes,
    log_prediction,
    make_gradcam_heatmap,
    overlay_heatmap_on_image,
    predict_fire_probability,
    preprocess_image,
    validate_file_size,
)

# Set up app config
st.set_page_config(page_title="Fire Detection from Satellite", layout="centered")

st.title("🔥 Fire Detection from Satellite Images")
st.write("Upload a satellite image to detect the presence of fire.")


@st.cache_resource(show_spinner="Loading model...")
def get_cached_model():
    return load_fire_model("fire_detection_model.h5")


try:
    model = get_cached_model()
except Exception as exc:  # noqa: BLE001 - this is a hard stop for the whole app
    st.error(
        "Couldn't load `fire_detection_model.h5`. Make sure the model file is present "
        f"next to app.py and isn't corrupted.\n\nDetails: {exc}"
    )
    st.stop()


def run_inference(image: Image.Image, threshold: float):
    """Preprocess, predict, classify, and Grad-CAM one image. Returns
    (ClassificationResult, heatmap_overlay: PIL.Image | None, error: str | None).
    """
    try:
        img_array = preprocess_image(image, target_size=INPUT_SHAPE)
        fire_probability = predict_fire_probability(model, img_array)
        result = classify(fire_probability, threshold=threshold, class_names=CLASS_NAMES)
    except Exception as exc:  # noqa: BLE001 - surfaced per-image, doesn't crash the app
        return None, None, str(exc)

    overlay = None
    try:
        heatmap = make_gradcam_heatmap(img_array, model)
        overlay = overlay_heatmap_on_image(image, heatmap, alpha=0.45)
    except Exception:  # noqa: BLE001 - explainability is best-effort, never fatal
        overlay = None

    return result, overlay, None


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
mode = st.sidebar.radio("Mode", ["Single Image", "Batch Upload"])
threshold = st.sidebar.slider(
    "Fire decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="P(Fire) at or above this value is classified as Fire. Lower it to make "
    "the detector more sensitive (more false alarms); raise it to be more conservative.",
)
show_gradcam = st.sidebar.checkbox(
    "Show Grad-CAM heatmap", value=True, help="Highlights the image regions that most influenced the prediction."
)
st.sidebar.caption(f"Max upload size: {MAX_UPLOAD_SIZE_MB:g} MB per image.")

# ---------------------------------------------------------------------------
# Single image mode
# ---------------------------------------------------------------------------

if mode == "Single Image":
    uploaded_file = st.file_uploader("📷 Upload Satellite Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        try:
            validate_file_size(file_bytes)
            image = load_image_from_bytes(file_bytes)
        except ImageValidationError as exc:
            st.error(str(exc))
            st.stop()

        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🧠 Detect Fire"):
            with st.spinner("Analyzing image..."):
                result, overlay, error = run_inference(image, threshold)

            if error:
                st.error(f"Prediction failed: {error}")
            else:
                log_prediction(
                    {
                        "source": "streamlit-single",
                        "filename": uploaded_file.name,
                        "prediction": result.label,
                        "predicted_class": result.predicted_class,
                        "fire_probability": result.fire_probability,
                        "confidence_pct": result.confidence_pct,
                        "threshold": threshold,
                    }
                )
                st.subheader("Result:")
                if result.predicted_class == 1:
                    st.error(f"Prediction: **🔥 {result.label}**")
                else:
                    st.success(f"Prediction: **{result.label}**")
                st.info(f"Confidence: **{result.confidence_pct:.2f}%**  ·  P(Fire) = {result.fire_probability:.4f}")

                if show_gradcam:
                    if overlay is not None:
                        st.subheader("Why the model thinks this:")
                        st.image(
                            overlay,
                            caption="Grad-CAM: warmer colors = regions that most influenced the prediction",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Grad-CAM heatmap could not be generated for this image.")

# ---------------------------------------------------------------------------
# Batch upload mode
# ---------------------------------------------------------------------------

else:
    uploaded_files = st.file_uploader(
        "📷 Upload Satellite Images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected.")

        if st.button(f"🧠 Detect Fire in {len(uploaded_files)} image(s)"):
            rows = []
            progress = st.progress(0.0)

            for i, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.getvalue()
                row = {
                    "filename": uploaded_file.name,
                    "prediction": None,
                    "confidence_pct": None,
                    "fire_probability": None,
                    "error": None,
                }
                try:
                    validate_file_size(file_bytes)
                    image = load_image_from_bytes(file_bytes)
                    result, _overlay, error = run_inference(image, threshold)
                    if error:
                        row["error"] = error
                    else:
                        row["prediction"] = result.label
                        row["confidence_pct"] = round(result.confidence_pct, 2)
                        row["fire_probability"] = round(result.fire_probability, 4)
                        log_prediction(
                            {
                                "source": "streamlit-batch",
                                "filename": uploaded_file.name,
                                "prediction": result.label,
                                "predicted_class": result.predicted_class,
                                "fire_probability": result.fire_probability,
                                "confidence_pct": result.confidence_pct,
                                "threshold": threshold,
                            }
                        )
                except ImageValidationError as exc:
                    row["error"] = str(exc)

                rows.append(row)
                progress.progress((i + 1) / len(uploaded_files))

            st.subheader("Batch Results")
            st.dataframe(rows, use_container_width=True)

            n_fire = sum(1 for r in rows if r["prediction"] == "Fire")
            n_errors = sum(1 for r in rows if r["error"])
            st.caption(f"🔥 {n_fire} flagged as Fire · {n_errors} failed to process · {len(rows)} total")

            csv_buffer = io.StringIO()
            writer = csv.DictWriter(
                csv_buffer, fieldnames=["filename", "prediction", "confidence_pct", "fire_probability", "error"]
            )
            writer.writeheader()
            writer.writerows(rows)

            st.download_button(
                "⬇️ Download results as CSV",
                data=csv_buffer.getvalue(),
                file_name="fire_detection_results.csv",
                mime="text/csv",
            )
