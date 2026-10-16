"""
SQLite database management module for storing prediction history and alert logs.

Uses safe parameterized queries, structured persistence functions, and clean
connection lifecycle management.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
import sqlite3
import json

from src.config import DEFAULT_DB_FILE
from src.utils import get_timestamp, setup_logger

logger = setup_logger("database")


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Open and return a connection to the SQLite database.

    Args:
        db_path: Path to the SQLite database file. Defaults to DEFAULT_DB_FILE.

    Returns:
        sqlite3.Connection instance with row factory set to sqlite3.Row.
    """
    path = db_path or DEFAULT_DB_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_context(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager yielding an open database connection and ensuring it is
    reliably closed upon exit.
    """
    conn = get_connection(db_path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Optional[Path] = None) -> None:
    """
    Initialize SQLite database tables for prediction logs and alert events.

    Args:
        db_path: Path to the SQLite database file.
    """
    path = db_path or DEFAULT_DB_FILE
    logger.info(f"Initializing SQLite database at: {path}")

    with get_db_context(path) as conn:
        cursor = conn.cursor()

        # Predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                device_type TEXT,
                air_temp_k REAL,
                process_temp_k REAL,
                rotational_speed_rpm REAL,
                torque_nm REAL,
                tool_wear_min REAL,
                predicted_failure INTEGER NOT NULL,
                prediction_score REAL NOT NULL,
                alert_level TEXT NOT NULL,
                is_simulated INTEGER NOT NULL DEFAULT 0,
                extra_metadata TEXT
            );
            """
        )

        # Alerts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                message TEXT NOT NULL,
                is_simulated INTEGER NOT NULL DEFAULT 0,
                resolved INTEGER NOT NULL DEFAULT 0
            );
            """
        )

        conn.commit()
    logger.info("Database schema initialized successfully.")


def log_prediction(
    features: Dict[str, Any],
    predicted_failure: int,
    prediction_score: float,
    alert_level: str,
    is_simulated: bool = False,
    extra_metadata: Optional[Dict[str, Any]] = None,
    db_path: Optional[Path] = None,
) -> int:
    """
    Log a model prediction event to the database using parameterized SQL.

    Returns:
        ID of the inserted prediction row.
    """
    timestamp = get_timestamp()
    meta_json = json.dumps(extra_metadata) if extra_metadata else None

    query = """
        INSERT INTO predictions (
            timestamp, device_type, air_temp_k, process_temp_k,
            rotational_speed_rpm, torque_nm, tool_wear_min,
            predicted_failure, prediction_score, alert_level,
            is_simulated, extra_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    params = (
        timestamp,
        str(features.get("Type", "Unknown")),
        float(features.get("Air temperature [K]", 0.0)),
        float(features.get("Process temperature [K]", 0.0)),
        float(features.get("Rotational speed [rpm]", 0.0)),
        float(features.get("Torque [Nm]", 0.0)),
        float(features.get("Tool wear [min]", 0.0)),
        int(predicted_failure),
        float(prediction_score),
        str(alert_level),
        1 if is_simulated else 0,
        meta_json,
    )

    with get_db_context(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid or 0


def log_alert(
    alert_level: str,
    risk_score: float,
    message: str,
    is_simulated: bool = False,
    db_path: Optional[Path] = None,
) -> int:
    """
    Log an alert dispatch event to the database using parameterized SQL.

    Returns:
        ID of the inserted alert row.
    """
    timestamp = get_timestamp()
    query = """
        INSERT INTO alerts (
            timestamp, alert_level, risk_score, message, is_simulated, resolved
        ) VALUES (?, ?, ?, ?, ?, 0);
    """
    params = (
        timestamp,
        str(alert_level),
        float(risk_score),
        str(message),
        1 if is_simulated else 0,
    )

    with get_db_context(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        return cursor.lastrowid or 0


def get_recent_predictions(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch the most recent prediction records."""
    query = "SELECT * FROM predictions ORDER BY id DESC LIMIT ?;"
    with get_db_context(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_recent_alerts(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Fetch the most recent alert records."""
    query = "SELECT * FROM alerts ORDER BY id DESC LIMIT ?;"
    with get_db_context(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(query, (limit,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


if __name__ == "__main__":
    init_db()
    print("Database initialization check complete.")
