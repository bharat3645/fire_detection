"""Tests for the headless batch-inference CLI (cli.py)."""

import csv
import json

from PIL import Image

import cli


def _write_image(path, size=(64, 64), color=(220, 40, 10)):
    Image.new("RGB", size, color).save(path)


# ---------------------------------------------------------------------------
# iter_image_paths
# ---------------------------------------------------------------------------

def test_iter_image_paths_single_file(tmp_path):
    img_path = tmp_path / "a.png"
    _write_image(img_path)
    assert list(cli.iter_image_paths(img_path)) == [img_path]


def test_iter_image_paths_directory_filters_extensions(tmp_path):
    _write_image(tmp_path / "a.png")
    _write_image(tmp_path / "b.jpg")
    (tmp_path / "notes.txt").write_text("not an image")
    found = {p.name for p in cli.iter_image_paths(tmp_path)}
    assert found == {"a.png", "b.jpg"}


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------

def test_run_batch_produces_rows_for_valid_and_invalid_images(tmp_path, tiny_model):
    good = tmp_path / "good.png"
    _write_image(good)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    rows = cli.run_batch(tiny_model, [good, bad], threshold=0.5, log=False)
    by_name = {r["filename"]: r for r in rows}

    assert by_name[str(good)]["prediction"] in ("Fire", "No Fire")
    assert by_name[str(good)]["error"] is None
    assert by_name[str(bad)]["prediction"] is None
    assert by_name[str(bad)]["error"] is not None


# ---------------------------------------------------------------------------
# main() — CLI entry point, exit codes, output formats
# ---------------------------------------------------------------------------

def test_main_writes_csv_and_returns_success_exit_code(tmp_path, tiny_model_path):
    img = tmp_path / "img.png"
    _write_image(img)
    out = tmp_path / "results.csv"

    code = cli.main(
        ["--input", str(img), "--output", str(out), "--model", str(tiny_model_path), "--threshold", "0.0", "--no-log"]
    )

    assert code == 2  # threshold 0.0 forces every prediction to Fire
    assert out.exists()
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["prediction"] == "Fire"


def test_main_no_fire_detected_returns_zero(tmp_path, tiny_model_path):
    img = tmp_path / "img.png"
    _write_image(img)
    out = tmp_path / "results.csv"

    code = cli.main(
        ["--input", str(img), "--output", str(out), "--model", str(tiny_model_path), "--threshold", "1.01", "--no-log"]
    )

    assert code == 0  # a threshold above 1.0 forces every prediction to No Fire


def test_main_json_format(tmp_path, tiny_model_path):
    img = tmp_path / "img.png"
    _write_image(img)
    out = tmp_path / "results.json"

    cli.main(
        ["--input", str(img), "--output", str(out), "--format", "json", "--model", str(tiny_model_path), "--no-log"]
    )

    data = json.loads(out.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["filename"] == str(img)


def test_main_errors_on_missing_input_path(tmp_path, tiny_model_path):
    code = cli.main(["--input", str(tmp_path / "does_not_exist.png"), "--model", str(tiny_model_path)])
    assert code == 1


def test_main_errors_on_missing_model(tmp_path):
    img = tmp_path / "img.png"
    _write_image(img)
    code = cli.main(["--input", str(img), "--model", str(tmp_path / "no_such_model.h5")])
    assert code == 1


def test_main_errors_on_directory_with_no_images(tmp_path, tiny_model_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = cli.main(["--input", str(empty_dir), "--model", str(tiny_model_path)])
    assert code == 1


def test_main_logs_predictions_when_enabled(tmp_path, tiny_model_path, monkeypatch):
    log_path = tmp_path / "audit.jsonl"
    monkeypatch.setenv("FIRE_DETECTION_LOG_PATH", str(log_path))

    img = tmp_path / "img.png"
    _write_image(img)
    out = tmp_path / "results.csv"

    cli.main(["--input", str(img), "--output", str(out), "--model", str(tiny_model_path)])

    assert log_path.exists()
    # Parse as JSON rather than substring-matching str(img): on Windows,
    # path separators are backslashes, which JSON re-escapes ("\\") when
    # serialized, so the raw path string never appears verbatim in the file.
    logged = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert logged["filename"] == str(img)
