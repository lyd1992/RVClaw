from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9\u4e00-\u9fff-]+", text.lower()))


@dataclass(slots=True)
class FlatVectorRecord:
    record_id: str
    text: str
    metadata: dict


class FlatVectorStore:
    """Tiny lexical baseline standing in for a future vector backend."""

    def __init__(self, records: Iterable[FlatVectorRecord] | None = None):
        self.records = list(records or [])

    def add(self, record_id: str, text: str, metadata: dict | None = None) -> None:
        self.records.append(FlatVectorRecord(record_id, text, metadata or {}))

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_tokens = _tokens(query)
        scored = []
        for record in self.records:
            score = len(query_tokens & _tokens(record.text))
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "record_id": record.record_id,
                "score": score,
                "text": record.text,
                "metadata": record.metadata,
            }
            for score, record in scored[:limit]
        ]
