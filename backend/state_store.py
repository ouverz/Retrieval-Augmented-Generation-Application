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
        with psycopg.connect(self._dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(DDL)

    def get(self, key: str) -> Optional[str]:
        with (
            psycopg.connect(self._dsn, autocommit=True) as conn,
            conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute("SELECT value FROM rag_state WHERE key=%s", (key,))
            row = cur.fetchone()
            return row["value"] if row else None

    def set(self, key: str, value: str) -> None:
        with psycopg.connect(self._dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO rag_state(key, value)
                VALUES (%s,%s)
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value, updated_at=NOW()
                """,
                (key, value),
            )
