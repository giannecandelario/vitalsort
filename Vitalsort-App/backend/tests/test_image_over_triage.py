"""Over-triage on the visual ingestion path.

Three defects stacked up so that essentially every upload stored as
Level 1: Emergency:

1. `_extract_dominant_colors` compared each channel against 150 in isolation, so
   any bright image - a flat gray square at value 200 - was "red".
2. "red" expanded into three weighted aliases of the same signal, inflammation
   (3) + erythema (3) + rash (2), which is exactly the 8-point Emergency
   threshold on its own, and `lesion` (5) made it trivially exceed it.
3. `main.visual_ingestion` then graded the image on that score, so a warm-toned
   or merely noisy photograph declared an emergency.

Live evidence before the fix: the gray square above was queued as
"Level 1: Emergency" / ["inflammation", "erythema", "rash"] / Dermatology.
"""
import asyncio
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from vision_engine_ai import AdvancedVisionEngine  # noqa: E402


def flat_image_bytes(value: int) -> bytes:
    arr = np.full((200, 200, 3), value, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def tinted_image_bytes(rgb, noise: int = 0, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    arr = np.zeros((200, 200, 3), dtype=np.int16) + np.array(rgb, dtype=np.int16)
    if noise:
        arr += rng.integers(-noise, noise, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def engine():
    return AdvancedVisionEngine()


# --- 1 and 2: what the engine sees ------------------------------------------

def test_bright_neutral_image_is_not_read_as_red(engine):
    """A light gray photo has no colour, but every channel clears 150."""
    metrics = engine.analyze_image(flat_image_bytes(200))["visual_metrics"]

    assert metrics["dominant_colors"] == ["neutral"]


def test_dark_neutral_image_is_still_not_read_as_a_colour(engine):
    metrics = engine.analyze_image(flat_image_bytes(90))["visual_metrics"]

    assert metrics["dominant_colors"] == ["neutral"]


def test_genuinely_red_image_is_still_detected(engine):
    metrics = engine.analyze_image(tinted_image_bytes((205, 120, 110)))["visual_metrics"]

    assert "red" in metrics["dominant_colors"]


def test_warm_skin_tone_image_is_detected_as_red_not_gray(engine):
    """The case that matters: a real photograph of flushed skin is warm overall,
    with all three channels high, and must still register."""
    metrics = engine.analyze_image(tinted_image_bytes((200, 165, 140), noise=12))["visual_metrics"]

    assert "red" in metrics["dominant_colors"]


def test_one_colour_yields_one_sign_not_three_aliases(engine):
    """The additive score counts each token, so emitting three names for the same
    redness silently multiplied a single visual feature by three."""
    symptoms = engine.analyze_image(tinted_image_bytes((205, 110, 100), noise=90))[
        "extracted_symptoms"
    ]

    assert "inflammation" in symptoms
    assert "erythema" not in symptoms
    assert "rash" not in symptoms


def test_red_alone_scores_below_the_emergency_threshold(engine):
    from engines import calculate_triage_level

    symptoms = engine.analyze_image(tinted_image_bytes((205, 110, 100)))["extracted_symptoms"]

    label, score = calculate_triage_level(symptoms)

    assert score < 8, f"{symptoms} scored {score} on redness alone"
    assert label != "Level 1: Emergency"


# --- 3: what the endpoint stores --------------------------------------------

@pytest.fixture(scope="module")
def main_module(tmp_path_factory):
    """Import main with the process CWD moved into a throwaway directory.

    DBEngine, ContactManager and AnalyticsEngine all default to the relative
    path "vitalsort.db", so importing main otherwise creates or migrates the
    committed database in the repo.
    """
    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("isolate-db"))
    try:
        import main
        yield main
    finally:
        os.chdir(previous)


class FakeSocket:
    def __init__(self, payload):
        self.payload = payload
        self.sent = []

    async def accept(self):
        pass

    async def receive_bytes(self):
        return self.payload

    async def send_json(self, message):
        self.sent.append(message)


def run_vision(main_module, monkeypatch, context, payload=b"\x00\x01\x02"):
    monkeypatch.setattr(main_module.vision_engine, "analyze_image", lambda data: context)
    captured = {}
    monkeypatch.setattr(main_module.db_engine, "insert_record",
                        lambda **kwargs: captured.update(kwargs) or 999)
    socket = FakeSocket(payload)
    asyncio.run(main_module.visual_ingestion(socket))
    return socket.sent[-1], captured


def test_a_high_scoring_sign_is_capped_at_urgent(main_module, monkeypatch):
    """The colour and texture heuristics cannot support "call 911", so a score in
    the Emergency band is stored as Urgent, and the row's action matches."""
    sent, captured = run_vision(main_module, monkeypatch, {
        "extracted_symptoms": ["inflammation", "lesion", "discoloration"],
        "inferred_acuity": "Urgent/Emergency Risk",
        "confidence": 0.09,
    })

    assert sent["is_emergency"] is False
    assert captured["triage_data"]["triage_level"] == "Level 2: Urgent"
    assert captured["triage_data"]["recommended_action"] == (
        "Seek clinical evaluation the same day."
    )
    assert captured["triage_score"] == 10


def test_a_trigger_phrase_in_a_visual_sign_still_reaches_emergency(main_module, monkeypatch):
    """The cap must not remove the escalation lane: a sign that matches an
    un-bypassable trigger still stores Level 1 with its protocol."""
    sent, captured = run_vision(main_module, monkeypatch, {
        "extracted_symptoms": ["severe swelling"],
        "inferred_acuity": "Routine",
        "confidence": 0.09,
    })

    assert sent["is_emergency"] is True
    assert captured["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert "911" in captured["triage_data"]["emergency_protocol"]


def test_a_bright_gray_upload_is_routine(engine, main_module, monkeypatch):
    """End to end on the synthetic image that used to be an emergency."""
    monkeypatch.setattr(main_module.vision_engine, "analyze_image", engine.analyze_image)
    captured = {}
    monkeypatch.setattr(main_module.db_engine, "insert_record",
                        lambda **kwargs: captured.update(kwargs) or 999)
    socket = FakeSocket(flat_image_bytes(200))

    asyncio.run(main_module.visual_ingestion(socket))

    assert socket.sent[-1]["is_emergency"] is False
    assert captured["triage_data"]["triage_level"] == "Level 4/5: Routine"


def test_a_flushed_skin_upload_is_not_an_emergency(engine, main_module, monkeypatch):
    monkeypatch.setattr(main_module.vision_engine, "analyze_image", engine.analyze_image)
    captured = {}
    monkeypatch.setattr(main_module.db_engine, "insert_record",
                        lambda **kwargs: captured.update(kwargs) or 999)
    socket = FakeSocket(tinted_image_bytes((205, 110, 100), noise=90))

    asyncio.run(main_module.visual_ingestion(socket))

    assert socket.sent[-1]["is_emergency"] is False
    assert captured["triage_data"]["triage_level"] != "Level 1: Emergency"
