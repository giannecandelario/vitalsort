"""What may declare an emergency.

CriticalHeuristicTriage.detect_red_flags treated `inferred_acuity ==
"Urgent/Emergency Risk"` as an un-bypassable red flag on its own. That string is
produced by _determine_acuity whenever the zero-shot severity classifier answers
"severe" and two or more symptoms were extracted - and it answers "severe" for
almost every complaint. Live rows proved it: "My ankle is swollen and it hurts"
(weighted score 6) was stored as Level 1: Emergency with the respiratory airway
protocol. Widening the intake vocabulary made multi-symptom extraction the norm,
so the rule started doing this constantly.

An emergency now needs an actual triggered flag, or a weighted score that is
already Level 1. Both directions are pinned here: the false positive is gone and
the true positives - including the image path, whose only previous emergency
signal was that same acuity string - still escalate.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from triage_logic import CriticalHeuristicTriage  # noqa: E402


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


@pytest.fixture
def captured_record(main_module, monkeypatch):
    captured = {}

    def fake_insert(**kwargs):
        captured.update(kwargs)
        return 999

    monkeypatch.setattr(main_module.db_engine, "insert_record", fake_insert)
    return captured


def fake_nlp(main_module, monkeypatch, symptoms, acuity, severity="severe"):
    """Pins the extraction the endpoint sees, so the assertions are about the
    escalation rule and not about which inflections the vocabulary happens to
    cover, or what BART decided on this run."""
    monkeypatch.setattr(main_module.nlp_engine, "process_symptoms", lambda text: {
        "extracted_symptoms": symptoms,
        "inferred_acuity": acuity,
        "symptom_severity": severity,
        "processed_tokens": 8,
        "confidence": 0.85,
    })


# --- the rule itself -------------------------------------------------------

def test_inferred_acuity_alone_is_not_an_emergency():
    """Score 6, acuity says risk, no trigger phrase matched."""
    flags = CriticalHeuristicTriage.detect_red_flags(
        ["pain", "swelling", "joint pain"], "Urgent/Emergency Risk",
        "My ankle is swollen and it hurts")

    assert flags["is_emergency"] is False
    assert flags["triggered_flags"] == []
    assert flags["protocol"] is None


def test_routine_acuity_with_no_flags_is_not_an_emergency():
    flags = CriticalHeuristicTriage.detect_red_flags(
        ["rash"], "Routine", "My head is itchy")

    assert flags["is_emergency"] is False


def test_trigger_phrase_in_the_text_is_still_an_emergency():
    flags = CriticalHeuristicTriage.detect_red_flags(
        ["pain", "chest", "severe"], "Routine", "I have severe chest pain")

    assert flags["is_emergency"] is True
    assert "cardiovascular" in flags["triggered_flags"]
    assert "911" in flags["emergency_instructions"]


def test_trigger_phrase_in_the_symptom_list_is_still_an_emergency():
    """Flags come from the extracted symptoms as well as the raw text: the vision
    path only ever has the category tokens."""
    flags = CriticalHeuristicTriage.detect_red_flags(
        ["severe swelling"], "Routine", "[IMAGE UPLOADED]")

    assert flags["is_emergency"] is True
    assert "anaphylaxis" in flags["triggered_flags"]


# --- the text endpoint -----------------------------------------------------

def test_acuity_risk_row_is_not_stored_as_an_emergency(main_module, captured_record, monkeypatch):
    fake_nlp(main_module, monkeypatch, ["pain", "swelling", "joint pain"], "Urgent/Emergency Risk")
    monkeypatch.setattr(main_module.ai_clinician, "analyze_symptoms", lambda *a, **k: None)

    result = asyncio.run(main_module.process_nlp({"text": "My ankle is swollen and it hurts"}))

    assert captured_record["triage_data"]["triage_level"] == "Level 2: Urgent"
    assert captured_record["triage_score"] == 6
    assert result["is_emergency"] is False
    assert "emergency_protocol" not in result


def test_weighted_score_still_reaches_the_emergency_lane(main_module, captured_record, monkeypatch):
    """Gating the flag check must not demote a real emergency that simply has no
    trigger phrase, and a Level 1 row may not be answered with reassurance."""
    fake_nlp(main_module, monkeypatch, ["difficulty swallowing", "swelling"], "Routine")
    monkeypatch.setattr(main_module.ai_clinician, "analyze_symptoms", lambda *a, **k: None)

    result = asyncio.run(main_module.process_nlp({"text": "I cant swallow and my tongue is big"}))

    assert captured_record["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert result["is_emergency"] is True
    assert "911" in captured_record["triage_data"]["recommended_action"]
    assert "911" in result["emergency_protocol"]


# --- the image endpoint ----------------------------------------------------

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


def test_image_with_a_trigger_phrase_reaches_emergency(main_module, monkeypatch):
    """len(symptoms) > 2 used to be the whole emergency signal on this path; with
    the flag check gated, an image escalates on a trigger phrase the same way text
    does. What a score alone may reach is pinned in test_image_over_triage.py."""
    sent, captured = run_vision(main_module, monkeypatch, {
        "extracted_symptoms": ["severe swelling", "erythema"],
        "inferred_acuity": "Routine",
        "confidence": 0.09,
    })

    assert sent["is_emergency"] is True
    assert "anaphylaxis" in sent["triage"]["triggered_flags"]
    assert captured["triage_data"]["triage_level"] == "Level 1: Emergency"


def test_image_with_one_minor_finding_is_not_an_emergency(main_module, monkeypatch):
    sent, captured = run_vision(main_module, monkeypatch, {
        "extracted_symptoms": ["rash"],
        "inferred_acuity": "Urgent/Emergency Risk",
        "confidence": 0.09,
    })

    assert sent["is_emergency"] is False
    assert captured["triage_data"]["triage_level"] != "Level 1: Emergency"
