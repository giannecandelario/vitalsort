import sqlite3
import json
from datetime import datetime

class ContactManager:
    """
    Manages patient contact information for follow-up care coordination.
    Stores contact details separately for HIPAA compliance and privacy.
    """
    
    def __init__(self, db_path="vitalsort.db"):
        self.db_path = db_path
        self._initialize_contacts_table()
    
    def _initialize_contacts_table(self):
        """Creates contacts table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS patient_contacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    phone TEXT NOT NULL,
                    notes TEXT,
                    timestamp TEXT NOT NULL,
                    follow_up_scheduled TEXT DEFAULT 'pending',
                    contact_status TEXT DEFAULT 'active'
                )
            ''')
            conn.commit()
    
    def save_contact(self, name: str, email: str, phone: str, notes: str = "") -> int:
        """
        Saves patient contact information.
        
        Args:
            name: Patient full name
            email: Patient email address
            phone: Patient phone number
            notes: Optional additional notes
            
        Returns:
            Contact record ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO patient_contacts 
                (name, email, phone, notes, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                name,
                email,
                phone,
                notes,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_all_contacts(self) -> list:
        """Retrieves all patient contacts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patient_contacts ORDER BY id DESC')
            return [dict(row) for row in cursor.fetchall()]
    
    def get_contact_by_id(self, contact_id: int) -> dict:
        """Retrieves specific contact by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM patient_contacts WHERE id = ?', (contact_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_follow_up_status(self, contact_id: int, status: str):
        """Updates follow-up scheduling status."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE patient_contacts SET follow_up_scheduled = ? WHERE id = ?',
                (status, contact_id)
            )
            conn.commit()
    
    def get_pending_follow_ups(self) -> list:
        """Retrieves contacts pending follow-up."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                'SELECT * FROM patient_contacts WHERE follow_up_scheduled = ? ORDER BY timestamp',
                ('pending',)
            )
            return [dict(row) for row in cursor.fetchall()]