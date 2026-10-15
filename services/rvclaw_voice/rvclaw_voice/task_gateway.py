"""Persistent, fail-closed task proposal gateway."""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

LOG = logging.getLogger(__name__)

EVENT_TYPES = frozenset({"TASK_CREATED", "TASK_ACCEPTED", "TASK_REJECTED", "TASK_STARTED", "STEP_STARTED", "STEP_SUCCEEDED", "STEP_FAILED", "TASK_PAUSED", "TASK_CANCELLED", "TASK_COMPLETED", "TASK_FAILED", "TASK_ABORTED", "MODE_CHANGED", "ESTOP_TRIGGERED"})
MOTION_INTENTS = frozenset({"start_inspection", "start_mapping", "turn_left", "return_home", "resume_inspection", "resume_inspection_relocalize"})
SOURCES = frozenset({"local", "cloud", "m260c_command"})


class TaskGateway:
    """Local trust boundary for typed proposals and durable task events."""

    def __init__(self, database: str = ":memory:", *, epoch: int = 0,
                 clock=time.time, max_ttl: float = 120.0,
                 max_clock_skew: float = 30.0) -> None:
        self.clock = clock
        self.max_ttl = float(max_ttl)
        self.max_clock_skew = float(max_clock_skew)
        if database != ":memory:":
            Path(database).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, check_same_thread=False)
        # Write-ahead logging (WAL) keeps proposals/events durable while readers
        # inspect status concurrently with the executor.
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS mode_state(name TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS proposals(
          proposal_id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
          source TEXT NOT NULL, intent TEXT NOT NULL, payload TEXT NOT NULL,
          status TEXT NOT NULL, reason TEXT, created REAL NOT NULL,
          task_id TEXT, owner_epoch INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS tasks(
          task_id TEXT PRIMARY KEY, proposal_id TEXT NOT NULL UNIQUE,
          state TEXT NOT NULL, owner_epoch INTEGER NOT NULL,
          updated REAL NOT NULL, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(
          seq INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT NOT NULL UNIQUE,
          task_id TEXT, type TEXT NOT NULL, payload TEXT NOT NULL,
          owner_epoch INTEGER NOT NULL, timestamp REAL NOT NULL);
        """)
        # Migrate databases created by the first gateway revision in place.
        # SQLite allows adding nullable columns without rewriting event history.
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(proposals)")}
        if "task_id" not in columns:
            self.db.execute("ALTER TABLE proposals ADD COLUMN task_id TEXT")
        if "owner_epoch" not in columns:
            self.db.execute("ALTER TABLE proposals ADD COLUMN owner_epoch INTEGER NOT NULL DEFAULT 0")
        self.db.commit()
        row = self.db.execute("SELECT value FROM mode_state WHERE name='owner_epoch'").fetchone()
        if row is None:
            self.owner_epoch = int(epoch)
            self.db.execute("INSERT INTO mode_state VALUES('owner_epoch',?)", (str(self.owner_epoch),))
            self.db.commit()
        else:
            self.owner_epoch = int(row[0])

    def set_owner_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < self.owner_epoch:
            raise ValueError("owner epoch cannot move backwards")
        with self.db:
            self.owner_epoch = epoch
            self.db.execute("INSERT OR REPLACE INTO mode_state VALUES('owner_epoch',?)", (str(epoch),))

    def submit(self, proposal: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(proposal, dict):
            return self._reject({}, "schema_invalid")
        reason = self._validate(proposal)
        if reason:
            return self._reject(proposal, reason)
        request_id = proposal["request_id"]
        old = self.db.execute("SELECT payload,status,reason FROM proposals WHERE request_id=?", (request_id,)).fetchone()
        if old:
            previous = json.loads(old[0])
            if previous.get("proposal_id") != proposal.get("proposal_id") or previous.get("intent") != proposal.get("intent"):
                return self._reject(proposal, "idempotency_conflict")
            previous.update(status=old[1], reason=old[2], replayed=True)
            return previous
        task_id = str(proposal.get("task_id") or proposal["proposal_id"])
        payload = dict(proposal, task_id=task_id, owner_epoch=self.owner_epoch)
        now = float(self.clock())
        try:
            with self.db:
                self.db.execute("INSERT INTO proposals VALUES(?,?,?,?,?,?,?,?,?,?)", (proposal["proposal_id"], request_id, proposal["source"], proposal["intent"], json.dumps(payload, ensure_ascii=False), "accepted", None, now, task_id, self.owner_epoch))
                self.db.execute("INSERT INTO tasks VALUES(?,?,?,?,?,?)", (task_id, proposal["proposal_id"], "accepted", self.owner_epoch, now, json.dumps(payload, ensure_ascii=False)))
                self._event(task_id, "TASK_CREATED", payload, timestamp=now)
                self._event(task_id, "TASK_ACCEPTED", payload, timestamp=now)
        except sqlite3.IntegrityError:
            return self._reject(proposal, "idempotency_conflict")
        return dict(payload, status="accepted", reason="proposal_accepted")

    def _validate(self, proposal: Dict[str, Any]) -> Optional[str]:
        required = ("schema_version", "proposal_id", "request_id", "source", "intent", "created_at", "expires_at")
        if any(not proposal.get(k) for k in required) or proposal.get("schema_version") != 1:
            return "schema_invalid"
        if not all(isinstance(proposal[k], str) for k in ("proposal_id", "request_id", "intent")):
            return "schema_invalid"
        if proposal["source"] not in SOURCES:
            return "source_invalid"
        if not isinstance(proposal.get("parameters", {}), dict):
            return "parameters_invalid"
        try:
            created = self._parse_timestamp(proposal["created_at"])
            expires = self._parse_timestamp(proposal["expires_at"])
        except (TypeError, ValueError, OverflowError):
            return "timestamp_invalid"
        now = float(self.clock())
        if created > now + self.max_clock_skew:
            return "timestamp_invalid"
        if expires <= created or expires <= now:
            return "proposal_expired"
        if expires - created > self.max_ttl:
            return "ttl_exceeded"
        try:
            owner_epoch = int(proposal.get("owner_epoch", self.owner_epoch))
        except (TypeError, ValueError):
            return "owner_epoch_invalid"
        if owner_epoch != self.owner_epoch:
            return "stale_owner_epoch"
        confidence = proposal.get("confidence")
        if confidence is not None:
            try:
                if not 0 <= float(confidence) <= 1:
                    return "confidence_invalid"
            except (TypeError, ValueError):
                return "confidence_invalid"
        if proposal.get("intent") in MOTION_INTENTS and (confidence is None or float(confidence) < 0.78):
            return "low_confidence_motion"
        return None

    @staticmethod
    def _parse_timestamp(value: Any) -> float:
        """Accept Unix seconds and RFC(Request for Comments) 3339 timestamps."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            text = value.strip()
            if text:
                try:
                    return float(text)
                except ValueError:
                    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    return parsed.timestamp()
        raise ValueError("invalid timestamp")

    def _reject(self, proposal: Dict[str, Any], reason: str) -> Dict[str, Any]:
        result = {"proposal_id": proposal.get("proposal_id"), "request_id": proposal.get("request_id"), "status": "rejected", "reason": reason}
        task_id = proposal.get("task_id") or proposal.get("proposal_id")
        if isinstance(task_id, str) and task_id:
            try:
                with self.db:
                    self._event(task_id, "TASK_REJECTED", {**proposal, "reason": reason})
            except sqlite3.IntegrityError:
                pass
        return result

    def _event(self, task_id: Optional[str], event_type: str, payload: Dict[str, Any], *, timestamp: Optional[float] = None) -> None:
        if event_type not in EVENT_TYPES:
            raise ValueError("unknown task event: %s" % event_type)
        stamp = self.clock() if timestamp is None else timestamp
        event_id = "%s:%s:%s:%s" % (task_id or "", event_type, stamp, uuid4().hex)
        self.db.execute("INSERT INTO events(event_id,task_id,type,payload,owner_epoch,timestamp) VALUES(?,?,?,?,?,?)", (event_id, task_id, event_type, json.dumps(payload, ensure_ascii=False), self.owner_epoch, stamp))

    def record_event(self, task_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None, *, owner_epoch: Optional[int] = None) -> Dict[str, Any]:
        if owner_epoch is not None and int(owner_epoch) != self.owner_epoch:
            raise ValueError("stale_owner_epoch")
        payload = dict(payload or {})
        state = {"TASK_STARTED": "running", "TASK_PAUSED": "paused", "TASK_CANCELLED": "cancelled", "TASK_COMPLETED": "completed", "TASK_FAILED": "failed", "TASK_ABORTED": "aborted"}.get(event_type)
        with self.db:
            self._event(task_id, event_type, payload)
            if state:
                self.db.execute("UPDATE tasks SET state=?,updated=? WHERE task_id=?", (state, self.clock(), task_id))
                self.db.execute("UPDATE proposals SET status=?,reason=? WHERE task_id=?", (state, event_type.lower(), task_id))
        return {"task_id": task_id, "type": event_type, "state": state}

    def complete_task(self, task_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.record_event(task_id, "TASK_COMPLETED", payload)

    def fail_task(self, task_id: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.record_event(task_id, "TASK_FAILED", payload)

    def reconcile_stale_tasks(self, *, stale_after: float = 60.0,
                              reason: str = "owner_process_gone") -> List[str]:
        """Close task rows whose owner died before it could publish a terminal event.

        A route task is held "running" by whichever process owns the route client.
        If that process disappears -- harness teardown, a service restart, a power
        cut -- nothing is left to observe the chassis exit, so the row would stay
        "running" forever and neither TASK_COMPLETED nor TASK_FAILED is ever
        written. Those rows are indistinguishable from real work in progress, which
        is exactly what makes them a nuisance.

        The caller is expected to run this where it can be sure nothing is actually
        driving the chassis (see ``service.build_router``, which additionally
        requires that no route runner process exists). Rows touched within
        ``stale_after`` seconds are skipped so a just-created task is never closed
        out from under a racing caller.

        Returns the ids of the tasks that were reclassified.
        """
        cutoff = float(self.clock()) - float(stale_after)
        rows = self.db.execute(
            "SELECT task_id, updated FROM tasks WHERE state='running' AND updated<? ORDER BY updated",
            (cutoff,),
        ).fetchall()
        closed: List[str] = []
        for task_id, updated in rows:
            try:
                self.record_event(task_id, "TASK_ABORTED", {
                    "reason": reason,
                    "previous_state": "running",
                    "last_update": updated,
                })
            except Exception:  # noqa: BLE001 - one bad row must not stop the sweep
                LOG.warning("failed to reconcile task %s", task_id, exc_info=True)
                continue
            closed.append(task_id)
        if closed:
            LOG.warning("reconciled %d stale running task(s) as aborted: %s",
                        len(closed), ", ".join(closed))
        return closed

    def task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.execute("SELECT task_id,state,owner_epoch,updated,payload FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        if not row:
            return None
        result = json.loads(row[4]); result.update(task_id=row[0], state=row[1], owner_epoch=row[2], updated=row[3])
        return result

    def latest_active_task_id(self, intents) -> Optional[str]:
        """Return the newest running task whose durable payload matches an intent.

        This is used only when the voice service restarts while an external
        chassis wrapper is still alive. The wrapper remains the execution
        authority; recovering the task id merely lets its eventual terminal
        state close the pre-existing database row instead of creating a second
        task or leaving the first one permanently running.
        """

        expected = {str(intent) for intent in intents}
        rows = self.db.execute(
            "SELECT task_id,payload FROM tasks WHERE state='running' ORDER BY updated DESC"
        ).fetchall()
        for task_id, payload in rows:
            try:
                intent = str(json.loads(payload).get("intent", ""))
            except (TypeError, ValueError):
                continue
            if intent in expected:
                return str(task_id)
        return None

    def events(self, task_id: Optional[str] = None, *, after_seq: int = 0):
        query = "SELECT seq,type,payload,owner_epoch,timestamp FROM events WHERE seq>?"
        args: tuple[Any, ...] = (int(after_seq),)
        if task_id:
            query += " AND task_id=?"; args += (task_id,)
        return self.db.execute(query + " ORDER BY seq", args).fetchall()

    def event_records(self, task_id: Optional[str] = None, *, after_seq: int = 0):
        """Return contract-shaped event dictionaries for sync/feedback users."""
        rows = self.db.execute(
            "SELECT seq,event_id,task_id,type,payload,owner_epoch,timestamp "
            "FROM events WHERE seq>?" + (" AND task_id=?" if task_id else "") + " ORDER BY seq",
            (int(after_seq), task_id) if task_id else (int(after_seq),),
        ).fetchall()
        return [{"seq": row[0], "event_id": row[1], "task_id": row[2], "type": row[3],
                 "payload": json.loads(row[4]), "owner_epoch": row[5], "timestamp": row[6]} for row in rows]
