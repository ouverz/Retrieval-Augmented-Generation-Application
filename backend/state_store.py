# app/services/state_store.py
from typing import Optional
import psycopg
from psycopg.rows import dict_row

DDL = """
CREATE TABLE IF NOT EXISTS rag_state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


class StateStore:
    def __init__(self, dsn: str):
        self._dsn = dsn
        self._init_db()

    def get(self, key: str) -> Optional[str]:
        try:
            with (
                psycopg.connect(self._dsn, autocommit=True, connect_timeout=5) as conn,
                conn.cursor(row_factory=dict_row) as cur,
            ):
                cur.execute("SELECT value FROM rag_state WHERE key=%s", (key,))
                row = cur.fetchone()
                return row["value"] if row else None
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"StateStore get failed for key '{key}': {e}")
            return None

    def _init_db(self) -> None:
        """Initialize database table with timeout and error handling."""
        try:
            with psycopg.connect(
                self._dsn, 
                autocommit=True, 
                connect_timeout=5  # 5 second timeout
            ) as conn, conn.cursor() as cur:
                cur.execute(DDL)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"StateStore database initialization failed: {e}")
            # Don't raise - allow app to continue without state persistence
    
    def set(self, key: str, value: str) -> None:
        try:
            with psycopg.connect(self._dsn, autocommit=True, connect_timeout=5) as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_state(key, value)
                    VALUES (%s,%s)
                    ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
                    """,
                    (key, value),
                )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"StateStore set failed for key '{key}': {e}")
            # Don't raise - allow app to continue without state persistence
