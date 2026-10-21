import sys
from pathlib import Path

import pytest
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from predictive_analytics_ai import PredictiveAnalyticsEngine

CREATE_SQL = """
CREATE TABLE triage_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    raw_text TEXT,
    symptoms TEXT,
    acuity TEXT,
    priority_level TEXT,
    recommended_action TEXT,
    triage_score INTEGER DEFAULT 0
)
"""


def make_engine(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute(CREATE_SQL)
    conn.executemany(
        """INSERT INTO triage_queue
           (timestamp, raw_text, symptoms, acuity, priority_level, recommended_action, triage_score)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            # acuity deliberately differs from priority_level so a positional
            # mix-up cannot pass silently.
            ("2026-05-24 13:33:28", "I have severe chest pain", '["pain"]',
             "Urgent", "Level 1: Emergency", "Call 911.", 8),
            ("2026-05-24 14:06:15", "I have a headache", '["pain"]',
             "Routine", "Level 4/5: Routine", "Monitor.", 1),
            ("2026-05-24 13:41:51", "[IMAGE UPLOADED]", '["rash"]',
             "Urgent/Emergency Risk", "Level 4/5: Routine", "Monitor.", 2),
            ("2026-05-24 15:00:00", "I have a fever", '["fever"]',
             "Routine", "Level 1: Emergency", "Call 911.", 4),
        ],
    )
    conn.commit()
    conn.close()
    return PredictiveAnalyticsEngine(db_path=str(db_path))


def test_emergencies_are_counted_from_priority_level(tmp_path):
    insights = make_engine(tmp_path / "vitalsort.db").get_population_insights()

    assert insights["status"] == "success"
    assert insights["total_patients"] == 4
    assert insights["emergency_cases"] == 2
    assert insights["high_risk_patients"] == 2
    assert insights["risk_percentage"] == 50.0


def test_hospitalization_counts_emergency_and_urgent_rows(tmp_path):
    insights = make_engine(tmp_path / "vitalsort.db").get_population_insights()

    # Rows 1 and 4 are Level 1: Emergency. The "Urgent/Emergency Risk" acuity on
    # row 3 must not inflate this -- its priority level is Routine.
    assert insights["likely_hospitalization"] == 2
    assert insights["hospitalization_percentage"] == 50.0


def test_empty_population_returns_error_status(tmp_path):
    conn = sqlite3.connect(tmp_path / "vitalsort.db")
    conn.execute(CREATE_SQL)
    conn.commit()
    conn.close()

    insights = PredictiveAnalyticsEngine(db_path=str(tmp_path / "vitalsort.db")).get_population_insights()

    assert insights["status"] == "error"
