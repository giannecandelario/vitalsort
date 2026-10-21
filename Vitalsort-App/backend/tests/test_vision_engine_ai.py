import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing this module is itself the regression test for the removed
# `from cv2 import textureFlattening` line, which aborted the whole backend.
from vision_engine_ai import AdvancedVisionEngine


@pytest.fixture(scope="session")
def engine():
    return AdvancedVisionEngine()


def synthetic_image_bytes(seed: int, red: int, noise: int) -> bytes:
    rng = np.random.default_rng(seed)
    # int16 accumulation keeps the red channel from wrapping past 255 and
    # silently dropping below the engine's >150 "red" threshold.
    base = np.zeros((200, 200, 3), dtype=np.int16)
    base[:, :, 0] = red
    base += rng.integers(0, noise, size=base.shape, dtype=np.int16)
    arr = np.clip(base, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_image_succeeds_without_fallback_error(engine):
    result = engine.analyze_image(synthetic_image_bytes(seed=1, red=200, noise=60))

    assert "error" not in result, f"engine hit its exception fallback: {result.get('error')}"
    assert result["source"] == "Advanced Visual Analysis"
    assert isinstance(result["extracted_symptoms"], list) and result["extracted_symptoms"]
    assert result["inferred_acuity"] in {"Routine", "Urgent/Emergency Risk"}
    assert 0.0 <= result["confidence"] <= 1.0


def test_red_high_texture_image_maps_to_inflammation(engine):
    result = engine.analyze_image(synthetic_image_bytes(seed=2, red=210, noise=120))

    metrics = result["visual_metrics"]
    assert "red" in metrics["dominant_colors"]
    assert metrics["texture_score"] > 0
    assert "inflammation" in result["extracted_symptoms"]


def test_determine_acuity_honours_texture_score(engine):
    # Before the fix this branch compared the imported cv2 name instead of the
    # measured texture, so the score never influenced acuity at all.
    high = engine._determine_acuity(["inflammation"], 70.0, ["red"], 95.0)
    low = engine._determine_acuity(["inflammation"], 70.0, ["red"], 12.0)
    many = engine._determine_acuity(["rash", "lesion", "swelling"], 30.0, [], 5.0)

    assert high == "Urgent/Emergency Risk"
    assert low == "Routine"
    assert many == "Urgent/Emergency Risk"


def test_corrupt_payload_still_degrades_gracefully(engine):
    result = engine.analyze_image(b"not an image at all")

    assert "error" in result
    assert result["inferred_acuity"] == "Routine"
