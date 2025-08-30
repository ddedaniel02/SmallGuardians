import sqlite3
from datetime import datetime, timezone



class DBLogger:
    def __init__(self, db_file="Monitoring_Tool/monitoring.db"):
        self.DB_FILE = db_file
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.DB_FILE)
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS input_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            classificator TEXT NOT NULL,
            input TEXT NOT NULL,
            classification TEXT NOT NULL,
            action_taken TEXT,
            comment TEXT
        )
        """)
        c.execute("""
        CREATE TABLE IF NOT EXISTS output_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            classificator TEXT NOT NULL,
            output TEXT NOT NULL,
            classification TEXT NOT NULL,
            action_taken TEXT,
            comment TEXT
        )
        """)
        conn.commit()
        conn.close()

    def insert_log_event(self, event, classificator, text, classification, action_taken=None, comment=None):
        if event == "input":
            table = "input_events"
            param = "input"
        elif event == "output":
            table = "output_events"
            param = "output"
        else:
            raise ValueError("Invalid event type")

        with sqlite3.connect(self.DB_FILE) as conn:
            c = conn.cursor()
            c.execute(f"""
            INSERT INTO {table} (timestamp, classificator, {param}, classification, action_taken, comment)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
            datetime.now(timezone.utc).isoformat(),
            classificator,
            text,
            classification,
            action_taken,
            comment
        ))

    def fetch_all_events(self, table):
        with sqlite3.connect(self.DB_FILE) as conn:
            c = conn.cursor()
            c.execute(f"SELECT * FROM {table} ORDER BY timestamp DESC")
            return c.fetchall()
