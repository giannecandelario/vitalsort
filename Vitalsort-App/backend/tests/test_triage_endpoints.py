"""Endpoint-level guards for emergency escalation.

Before the fix, a red-flag emergency was returned to the client as
is_emergency=true while the row written to triage_queue kept the weighted-score
label ("Level 4/5: Routine"), so the provider queue, analytics and PDF dossier
all understated acuity.
"""
import asyncio
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="module")
def main_module(tmp_path_factory):
    """Import main with the process CWD moved into a throwaway directory.

    DBEngine, ContactManager and AnalyticsEngine all default to the relative
    path "vitalsort.db", so importing main otherwise creates or migrates the
    committed database in the repo.
    """
    import os

    previous = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("isolate-db"))
    try:
        import main
        yield main
    finally:
        os.chdir(previous)


@pytest.fixture
def captured_record(main_module, monkeypatch):
    """Stubs persistence so the tests can inspect what would be stored."""
    captured = {}

    def fake_insert(**kwargs):
        captured.update(kwargs)
        return 999

    monkeypatch.setattr(main_module.db_engine, "insert_record", fake_insert)
    return captured


def test_red_flag_text_escalates_stored_priority(main_module, captured_record):
    result = asyncio.run(main_module.process_nlp({
        "text": "I have severe chest pain and difficulty breathing"
    }))

    assert result["is_emergency"] is True
    assert captured_record["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert captured_record["triage_score"] >= 8


def test_non_emergency_text_is_not_escalated(main_module, captured_record):
    result = asyncio.run(main_module.process_nlp({
        "text": "I have a mild fever for three days"
    }))

    assert result["is_emergency"] is False
    assert captured_record["triage_data"]["triage_level"] != "Level 1: Emergency"


def test_emergency_row_keeps_anaphylaxis_protocol(main_module, captured_record):
    # Phrasing deliberately matches an entry in EMERGENCY_TRIGGERS. Note that
    # "my throat is closing" does NOT match the stored phrase "throat closing";
    # red-flag detection is exact substring matching and is brittle against
    # ordinary patient phrasing.
    asyncio.run(main_module.process_nlp({
        "text": "I am having a severe allergic reaction and my throat closing"
    }))

    assert captured_record["triage_data"]["is_emergency"] is True
    assert captured_record["triage_data"]["triage_level"] == "Level 1: Emergency"
    assert "911" in captured_record["triage_data"]["recommended_action"]
