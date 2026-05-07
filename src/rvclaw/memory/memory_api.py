from __future__ import annotations

from pathlib import Path
from typing import Any

from .sqlite_event_store import SQLiteEventStore


class MemoryManager:
    def __init__(self, event_store: SQLiteEventStore):
        self.event_store = event_store

    @classmethod
    def from_path(cls, db_path: str | Path) -> "MemoryManager":
        store = SQLiteEventStore(db_path)
        store.seed_defaults()
        return cls(store)

    def query(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.event_store.query(query, limit=limit)

    def remember(self, kind: str, content: dict[str, Any] | str, tags: list[str] | None = None) -> int:
        return self.event_store.append_event(kind, content, tags=tags)
