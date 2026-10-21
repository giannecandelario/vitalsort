import sqlite3
import json
from datetime import datetime

class DBEngine:
    def __init__(self, db_path="vitalsort.db"):
        """
        Initializes the local SQLite database for the MVP queue.
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS triage_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    raw_text TEXT,
                    symptoms TEXT,
                    acuity TEXT,
                    priority_level TEXT,
                    recommended_action TEXT,
                    specialist TEXT,
                    ai_analysis TEXT
                )
            ''')
            # Check if triage_score column exists, if not, add it
            cursor.execute("PRAGMA table_info(triage_queue)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'triage_score' not in columns:
                cursor.execute("ALTER TABLE triage_queue ADD COLUMN triage_score INTEGER DEFAULT 0")
            # Same migration pattern for the specialist referral
            if 'specialist' not in columns:
                cursor.execute("ALTER TABLE triage_queue ADD COLUMN specialist TEXT")
            # And for the GPT clinician analysis stored alongside the decision
            if 'ai_analysis' not in columns:
                cursor.execute("ALTER TABLE triage_queue ADD COLUMN ai_analysis TEXT")
            conn.commit()

    # In backend/db_engine.py

    def insert_record(self, raw_text: str, nlp_data: dict, triage_data: dict, triage_score: int) -> int:
        analysis = triage_data.get("ai_analysis")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Ensure your table has the 'triage_score' column!
            cursor.execute('''
                INSERT INTO triage_queue 
                (timestamp, raw_text, symptoms, acuity, priority_level, recommended_action, triage_score, specialist, ai_analysis)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                raw_text,
                json.dumps(nlp_data.get("extracted_symptoms", [])),
                nlp_data.get("inferred_acuity", "Routine"),
                triage_data.get("triage_level", "Level 4/5: Routine"),
                triage_data.get("recommended_action", ""),
                triage_score, # New field stored here
                triage_data.get("specialist"),
                json.dumps(analysis) if analysis else None
            ))
            conn.commit()
            return cursor.lastrowid

    def get_active_queue(self) -> list:
        """Retrieves the most recent triage records for the provider dashboard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Allows accessing columns by name
            cursor = conn.cursor()
            # Fetch the 50 most recent records, highest ID first
            cursor.execute('SELECT * FROM triage_queue ORDER BY id DESC LIMIT 50')
            
            # Convert SQLite rows to standard Python dictionaries
            return [dict(row) for row in cursor.fetchall()]