from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rvclaw.utils import ensure_dir, utc_now_iso


class SQLiteEventStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._init_schema()

    def _connect_raw(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connect(self):
        connection = self._connect_raw()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _init_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  content TEXT NOT NULL,
                  tags TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_events_kind ON events(kind)")

    def seed_defaults(self) -> None:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if count:
            return
        self.append_event(
            "device_profile",
            {
                "zone": "A-03",
                "device": "edge-inference-node-a03",
                "expected_status": "normal",
                "risk_policy": "stop on smoke, high temperature, blocked path",
            },
            ["seed", "device", "A-03"],
        )
        self.append_event(
            "inspection_history",
            {
                "zone": "A-03",
                "last_result": "normal",
                "last_report": "No abnormal vibration or overheating found.",
            },
            ["seed", "inspection", "A-03"],
        )

    def append_event(self, kind: str, content: dict[str, Any] | str, tags: list[str] | None = None) -> int:
        encoded = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO events(ts, kind, content, tags) VALUES (?, ?, ?, ?)",
                (utc_now_iso(), kind, encoded, json.dumps(tags or [], ensure_ascii=False)),
            )
            return int(cursor.lastrowid)

    def query(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        terms = [term for term in query.replace("，", " ").replace(",", " ").split() if term]
        clauses = []
        params: list[str] = []
        for term in terms[:8]:
            clauses.append("(kind LIKE ? OR content LIKE ? OR tags LIKE ?)")
            like = f"%{term}%"
            params.extend([like, like, like])
        where = " OR ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM events WHERE {where} ORDER BY id DESC LIMIT ?"
        params.append(str(limit))
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._decode_row(row) for row in rows]

    @staticmethod
    def _decode_row(row: sqlite3.Row) -> dict[str, Any]:
        content = row["content"]
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass
        return {
            "id": row["id"],
            "ts": row["ts"],
            "kind": row["kind"],
            "content": content,
            "tags": json.loads(row["tags"]),
        }
