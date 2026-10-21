import sqlite3
from datetime import datetime, timedelta
from collections import Counter

class AnalyticsEngine:
    """
    Provides statistical analysis and trending insights for the provider dashboard.
    Helps clinicians understand patient patterns and system performance.
    """
    
    def __init__(self, db_path="vitalsort.db"):
        self.db_path = db_path
    
    def get_triage_distribution(self, days: int = 30) -> dict:
        """
        Returns distribution of triage levels over the past N days.
        
        Args:
            days: Number of days to analyze (default 30)
            
        Returns:
            Dict with triage level counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get records from past N days
            date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            cursor.execute('''
                SELECT priority_level, COUNT(*) as count
                FROM triage_queue
                WHERE timestamp > ?
                GROUP BY priority_level
            ''', (date_threshold,))
            
            results = cursor.fetchall()
            
            return {
                "period_days": days,
                "distribution": {
                    row[0]: row[1] for row in results
                }
            }
    
    def get_symptom_frequency(self, days: int = 30, limit: int = 10) -> dict:
        """
        Returns most common symptoms over the past N days.
        
        Args:
            days: Number of days to analyze
            limit: Max symptoms to return
            
        Returns:
            Dict with symptom frequency
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            cursor.execute('''
                SELECT symptoms FROM triage_queue
                WHERE timestamp > ?
            ''', (date_threshold,))
            
            # Parse JSON symptoms and count
            symptom_counter = Counter()
            for row in cursor.fetchall():
                if row[0]:
                    symptoms = row[0].replace('[', '').replace(']', '').replace('"', '').split(',')
                    for symptom in symptoms:
                        symptom = symptom.strip()
                        if symptom:
                            symptom_counter[symptom] += 1
            
            return {
                "period_days": days,
                "top_symptoms": dict(symptom_counter.most_common(limit))
            }
    
    def get_acuity_trends(self, days: int = 30) -> dict:
        """
        Returns daily trend of high-acuity cases.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with daily acuity counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            date_threshold = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
            
            cursor.execute('''
                SELECT DATE(timestamp) as date, COUNT(*) as count,
                       SUM(CASE WHEN priority_level LIKE '%Emergency%' THEN 1 ELSE 0 END) as emergencies,
                       SUM(CASE WHEN priority_level LIKE '%Urgent%' THEN 1 ELSE 0 END) as urgent,
                       AVG(triage_score) as avg_score
                FROM triage_queue
                WHERE timestamp > ?
                GROUP BY DATE(timestamp)
                ORDER BY date DESC
            ''', (date_threshold,))
            
            results = cursor.fetchall()
            
            trends = []
            for row in results:
                trends.append({
                    "date": row[0],
                    "total_cases": row[1],
                    "emergencies": row[2] or 0,
                    "urgent": row[3] or 0,
                    "avg_triage_score": round(row[4], 2) if row[4] else 0
                })
            
            return {
                "period_days": days,
                "daily_trends": trends
            }
    
    def get_system_stats(self) -> dict:
        """Returns overall system statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute('SELECT COUNT(*) FROM triage_queue')
            total_records = cursor.fetchone()[0]
            
            # Emergency count
            cursor.execute(
                "SELECT COUNT(*) FROM triage_queue WHERE priority_level LIKE '%Emergency%'"
            )
            emergency_count = cursor.fetchone()[0]
            
            # Average triage score
            cursor.execute('SELECT AVG(triage_score) FROM triage_queue')
            avg_score = cursor.fetchone()[0]
            
            # Records today
            today = datetime.now().strftime("%Y-%m-%d")
            cursor.execute(
                "SELECT COUNT(*) FROM triage_queue WHERE DATE(timestamp) = ?",
                (today,)
            )
            today_count = cursor.fetchone()[0]
            
            return {
                "total_records": total_records,
                "emergency_count": emergency_count,
                "emergency_percentage": round((emergency_count / max(total_records, 1)) * 100, 2),
                "average_triage_score": round(avg_score, 2) if avg_score else 0,
                "records_today": today_count
            }
    
    def get_specialist_referral_stats(self) -> dict:
        """
        Returns statistics on specialist referrals.
        (Note: This is a placeholder - actual specialists are tracked at intake time)
        """
        return {
            "total_referrals": "Coming soon",
            "top_specialists": "Coming soon",
            "note": "Specialist data will be populated as more records are created"
        }