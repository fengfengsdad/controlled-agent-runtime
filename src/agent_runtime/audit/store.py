from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from agent_runtime.models.schemas import AuditEvent, AuditEventType


class AuditStore:
    """Persists a structured audit chain per workflow."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_workflow ON audit_events(workflow_id)"
            )

    def append(
        self,
        workflow_id: str,
        event_type: AuditEventType,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            workflow_id=workflow_id,
            event_type=event_type,
            payload=payload or {},
        )
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events(event_id, workflow_id, event_type, timestamp, payload)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.workflow_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    json.dumps(event.payload, ensure_ascii=False),
                ),
            )
        return event

    def list_for_workflow(self, workflow_id: str) -> list[AuditEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, workflow_id, event_type, timestamp, payload
                FROM audit_events
                WHERE workflow_id = ?
                ORDER BY timestamp ASC
                """,
                (workflow_id,),
            ).fetchall()
        events: list[AuditEvent] = []
        for row in rows:
            ts = row["timestamp"]
            timestamp = datetime.fromisoformat(ts) if isinstance(ts, str) else ts
            events.append(
                AuditEvent(
                    event_id=row["event_id"],
                    workflow_id=row["workflow_id"],
                    event_type=AuditEventType(row["event_type"]),
                    timestamp=timestamp,
                    payload=json.loads(row["payload"]),
                )
            )
        return events
