"""Intake-path behaviour with the GPT clinician attached.

The clinician is stubbed here: these tests pin the agreement between the
deterministic engines and the model, namely that the model can only ever raise a
level, that a deterministic red flag still outranks it, and that intake is
unchanged when no analysis comes back.
"""
import asyncio
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="module")
def main_module(tmp_path_factory):
    """Import main with CWD moved to a throwaway directory.

    DBEngine and friends default to the relative path "vitalsort.db", so without
    this the suite migrates the committed database.
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


def clinician(urgency, citations=None, actions=None):
    return {
        "urgency": urgency,
        "red_flags": ["facial swelling", "cannot swallow"],
        "differential": [{"condition": "angioedema", "likelihood": "high",
                          "rationale": "swelling with dysphagia"}],
        "immediate_actions": actions or ["Call 911 immediately."],
        "specialist": "Emergency Medicine",
        "citations": citations if citations is not None else [
            {"source": "MedlinePlus", "claim": "Trouble swallowing with swelling needs emergency care."}
        ],
        "confidence_note": "Based on the description alone.",
        "model": "gpt-4.1-mini",
    }


def stub(main_module, monkeypatch, analysis):
    monkeypatch.setattr(main_module.ai_clinician, "analyze_symptoms",
                        lambda *a, **k: analysis)


# Phrasing the widened keyword vocabulary still misses, so the deterministic path
# scores 0 and assigns Routine. This is the gap the model is meant to catch.
# (It used to be "My eyes are swollen and I can't drink water" - the extractor now
# recognises that one, which is the point of the vocabulary expansion.)
UNRECOGNISED_DYSPHAGIA = "My tongue feels big and every mouthful gets stuck"


def test_model_escalates_a_routine_heuristic_row(main_module, captured_record, monkeypatch):
    stub(main_module, monkeypatch, clinician("emergency"))

    result = asyncio.run(main_module.process_nlp({"text": UNRECOGNISED_DYSPHAGIA}))

    assert captured_record["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert captured_record["triage_data"]["is_emergency"] is True
    assert captured_record["triage_data"]["specialist"] == "Emergency Medicine"
    assert captured_record["triage_data"]["recommended_action"].startswith("AI clinician escalated")
    assert result["ai_escalated"] is True
    assert result["is_emergency"] is True
    assert "911" in result["emergency_protocol"]


def test_escalation_is_recorded_for_audit(main_module, captured_record, monkeypatch):
    stub(main_module, monkeypatch, clinician("emergency"))

    asyncio.run(main_module.process_nlp({"text": UNRECOGNISED_DYSPHAGIA}))

    assert captured_record["triage_data"]["ai_analysis"]["urgency"] == "emergency"


def test_urgent_model_escalates_without_claiming_emergency(main_module, captured_record, monkeypatch):
    stub(main_module, monkeypatch, clinician("urgent", actions=["Be assessed the same day."]))

    result = asyncio.run(main_module.process_nlp({"text": UNRECOGNISED_DYSPHAGIA}))

    assert captured_record["triage_data"]["triage_level"] == "Level 2: Urgent"
    assert captured_record["triage_data"]["is_emergency"] is False
    assert result["is_emergency"] is False


def test_model_cannot_de_escalate_a_red_flag_emergency(main_module, captured_record, monkeypatch):
    # This text trips the deterministic red flag, so the row is already Level 1.
    stub(main_module, monkeypatch, clinician("routine", citations=[]))

    result = asyncio.run(main_module.process_nlp({
        "text": "I have severe chest pain and difficulty breathing"
    }))

    assert captured_record["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert captured_record["triage_data"]["ai_analysis"]["urgency"] == "routine"
    assert result["ai_escalated"] is False
    # The deterministic protocol still wins over anything the model suggests.
    assert captured_record["triage_data"]["recommended_action"].startswith("🚨 EMERGENCY")


def test_uncited_emergency_is_shown_but_not_acted_on(main_module, captured_record, monkeypatch):
    stub(main_module, monkeypatch, clinician("emergency", citations=[]))

    result = asyncio.run(main_module.process_nlp({"text": UNRECOGNISED_DYSPHAGIA}))

    assert captured_record["triage_data"]["triage_level"] == "Level 4/5: Routine"
    assert result["ai_escalated"] is False
    # The analysis is still returned so the reviewer can see and judge it.
    assert result["ai_analysis"]["urgency"] == "emergency"


def test_intake_is_unchanged_when_the_model_returns_nothing(main_module, captured_record, monkeypatch):
    stub(main_module, monkeypatch, None)

    result = asyncio.run(main_module.process_nlp({
        "text": "I have a mild fever for three days"
    }))

    assert result["ai_analysis"] is None
    assert result["ai_escalated"] is False
    assert captured_record["triage_data"]["ai_analysis"] is None
    assert captured_record["triage_data"]["recommended_action"] == "Consult physician if symptoms persist."
