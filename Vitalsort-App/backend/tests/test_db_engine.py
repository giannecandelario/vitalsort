"""Specialist referral must survive the round trip into storage.

The intake endpoints resolve a specialist and return it to the client, but
triage_queue never stored it, so the provider dashboard column rendered by
frontend/app.js was permanently empty.
"""
import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_engine import DBEngine

# The schema as it exists in the committed database -- no specialist column.
LEGACY_SCHEMA = """
CREATE TABLE triage_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    raw_text TEXT,
    symptoms TEXT,
    acuity TEXT,
    priority_level TEXT,
    recommended_action TEXT
, triage_score INTEGER DEFAULT 0)
"""

LEGACY_ROWS = [
    ("2026-05-24 14:06:15", "I have a headache and fever", '["fever"]',
     "Routine", "Level 4/5: Routine", "Monitor.", 4),
]


def new_record(triage_data=None, nlp_data=None):
    return dict(
        raw_text="I have severe chest pain",
        nlp_data=nlp_data or {"extracted_symptoms": ["pain", "chest"], "inferred_acuity": "Urgent/Emergency Risk"},
        triage_data=triage_data or {"triage_level": "Level 1: Emergency",
                                    "recommended_action": "Call 911.",
                                    "specialist": "Cardiology"},
        triage_score=13,
    )


def test_specialist_is_persisted_and_returned(tmp_path):
    engine = DBEngine(db_path=str(tmp_path / "fresh.db"))

    record_id = engine.insert_record(**new_record())
    row = [r for r in engine.get_active_queue() if r["id"] == record_id][0]

    assert row["specialist"] == "Cardiology"


def test_specialist_column_is_added_to_existing_databases(tmp_path):
    """Migration path: committed databases predate the column."""
    db = tmp_path / "legacy.db"
    conn = sqlite3.connect(db)
    conn.execute(LEGACY_SCHEMA)
    conn.executemany(
        "INSERT INTO triage_queue (timestamp, raw_text, symptoms, acuity, priority_level,"
        " recommended_action, triage_score) VALUES (?, ?, ?, ?, ?, ?, ?)",
        LEGACY_ROWS,
    )
    conn.commit()
    conn.close()

    engine = DBEngine(db_path=str(db))
    record_id = engine.insert_record(**new_record())

    columns = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(triage_queue)")]
    assert "specialist" in columns

    queue = engine.get_active_queue()
    by_id = {r["id"]: r for r in queue}
    assert by_id[record_id]["specialist"] == "Cardiology"
    # Pre-existing rows simply have no referral rather than breaking reads.
    assert by_id[1]["specialist"] in (None, "")


def test_missing_specialist_does_not_break_insert(tmp_path):
    engine = DBEngine(db_path=str(tmp_path / "nofield.db"))

    record_id = engine.insert_record(**new_record(triage_data={
        "triage_level": "Level 4/5: Routine",
        "recommended_action": "Monitor.",
    }))

    row = [r for r in engine.get_active_queue() if r["id"] == record_id][0]
    assert row["specialist"] in (None, "")
    assert json.loads(row["symptoms"]) == ["pain", "chest"]


def test_ai_analysis_is_stored_as_json(tmp_path):
    """The analysis must be auditable: a row escalated by the model has to say so."""
    analysis = {
        "urgency": "emergency",
        "red_flags": ["cannot swallow"],
        "differential": [{"condition": "angioedema", "likelihood": "high", "rationale": "facial swelling"}],
        "immediate_actions": ["Call 911"],
        "specialist": "Emergency Medicine",
        "citations": [{"source": "MedlinePlus", "claim": "Swelling with dysphagia is an emergency."}],
        "confidence_note": "Text only.",
        "model": "gpt-4.1-mini",
    }
    engine = DBEngine(db_path=str(tmp_path / "ai.db"))

    record_id = engine.insert_record(**new_record(triage_data={
        "triage_level": "Level 1: Emergency",
        "recommended_action": "Call 911.",
        "specialist": "Emergency Medicine",
        "ai_analysis": analysis,
    }))

    row = [r for r in engine.get_active_queue() if r["id"] == record_id][0]
    assert json.loads(row["ai_analysis"]) == analysis


def test_rows_without_an_analysis_store_null(tmp_path):
    engine = DBEngine(db_path=str(tmp_path / "noai.db"))

    record_id = engine.insert_record(**new_record())
    row = [r for r in engine.get_active_queue() if r["id"] == record_id][0]

    assert row["ai_analysis"] is None


def test_ai_analysis_column_is_added_to_existing_databases(tmp_path):
    db = tmp_path / "legacy2.db"
    conn = sqlite3.connect(db)
    conn.execute(LEGACY_SCHEMA)
    conn.commit()
    conn.close()

    engine = DBEngine(db_path=str(db))
    engine.insert_record(**new_record(triage_data={
        "triage_level": "Level 4/5: Routine",
        "recommended_action": "Monitor.",
        "ai_analysis": {"urgency": "routine", "red_flags": [], "differential": [],
                        "immediate_actions": [], "specialist": "Primary Care",
                        "citations": [{"source": "NHS", "claim": "c"}],
                        "confidence_note": "n", "model": "m"},
    }))

    columns = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(triage_queue)")]
    assert "ai_analysis" in columns
    assert json.loads(engine.get_active_queue()[0]["ai_analysis"])["urgency"] == "routine"
