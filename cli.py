"""Command-line batch inference for the fire-detection model.

Runs predictions over a single image or a whole directory of images without
launching Streamlit or a browser — useful for scripted/scheduled use (e.g. a
nightly cron job scanning a folder of newly captured tiles) where the
Streamlit batch-upload mode, which needs a human to click through a browser,
doesn't fit.

Usage:
    python cli.py --input path/to/images/            # CSV to stdout
    python cli.py --input image.jpg --format json
    python cli.py --input path/to/images/ --output results.csv --threshold 0.3

Exit codes (designed to be checked by a calling script/cron job):
    0  ran successfully, no image was flagged as Fire
    2  ran successfully, at least one image was flagged as Fire
    1  fatal error (bad --input path, no images found, model failed to load)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from core import (
    CLASS_NAMES,
    INPUT_SHAPE,
    ImageValidationError,
    classify,
    load_fire_model,
    load_image_from_bytes,
    log_prediction,
    predict_fire_probability,
    preprocess_image,
    validate_file_size,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ROW_FIELDS = ["filename", "prediction", "confidence_pct", "fire_probability", "error"]


def iter_image_paths(input_path: Path):
    """Yield image file paths: the file itself, or every image under a directory (recursive)."""
    if input_path.is_file():
        yield input_path
        return
    for path in sorted(input_path.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def run_batch(model, paths, threshold: float, log: bool) -> list[dict]:
    """Classify every path in ``paths``, returning one result row per image.

    A single unreadable/corrupt image is recorded as an error row and never
    aborts the rest of the batch.
    """
    rows = []
    for path in paths:
        row = {
            "filename": str(path),
            "prediction": None,
            "confidence_pct": None,
            "fire_probability": None,
            "error": None,
        }
        try:
            file_bytes = path.read_bytes()
            validate_file_size(file_bytes)
            image = load_image_from_bytes(file_bytes)
            img_array = preprocess_image(image, target_size=INPUT_SHAPE)
            fire_probability = predict_fire_probability(model, img_array)
            result = classify(fire_probability, threshold=threshold, class_names=CLASS_NAMES)
            row["prediction"] = result.label
            row["confidence_pct"] = round(result.confidence_pct, 2)
            row["fire_probability"] = round(result.fire_probability, 4)
            if log:
                log_prediction(
                    {
                        "source": "cli",
                        "filename": str(path),
                        "prediction": result.label,
                        "predicted_class": result.predicted_class,
                        "fire_probability": result.fire_probability,
                        "confidence_pct": result.confidence_pct,
                        "threshold": threshold,
                    }
                )
        except ImageValidationError as exc:
            row["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - keep the batch going, surface per-file
            row["error"] = f"Unexpected error: {exc}"
        rows.append(row)
    return rows


def write_csv(rows: list[dict], stream) -> None:
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch fire detection over a directory or single image, from the command line."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to an image file, or a directory of images (searched recursively).",
    )
    parser.add_argument("--output", type=Path, default=None, help="Where to write results. Defaults to stdout.")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="Output format (default: csv).")
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="P(Fire) cutoff for the Fire label (default: 0.5)."
    )
    parser.add_argument(
        "--model", type=Path, default=Path("fire_detection_model.h5"), help="Path to the .h5 model file."
    )
    parser.add_argument("--no-log", action="store_true", help="Skip writing to the prediction audit log.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input.exists():
        print(f"error: input path does not exist: {args.input}", file=sys.stderr)
        return 1

    try:
        model = load_fire_model(str(args.model))
    except Exception as exc:  # noqa: BLE001 - fatal, reported to the user directly
        print(f"error: could not load model from {args.model}: {exc}", file=sys.stderr)
        return 1

    paths = list(iter_image_paths(args.input))
    if not paths:
        print(f"error: no image files (.jpg/.jpeg/.png) found under {args.input}", file=sys.stderr)
        return 1

    rows = run_batch(model, paths, args.threshold, log=not args.no_log)

    if args.format == "csv":
        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                write_csv(rows, f)
        else:
            write_csv(rows, sys.stdout)
    else:
        text = json.dumps(rows, indent=2)
        if args.output:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text)

    n_fire = sum(1 for r in rows if r["prediction"] == "Fire")
    n_errors = sum(1 for r in rows if r["error"])
    print(f"\n{len(rows)} image(s) processed - {n_fire} flagged as Fire - {n_errors} failed", file=sys.stderr)

    return 2 if n_fire > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
