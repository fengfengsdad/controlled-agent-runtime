from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class CheckpointStore:
    """SQLite checkpoints for service-restart recovery and idempotent workflow lookup."""

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
                CREATE TABLE IF NOT EXISTS checkpoints (
                    workflow_id TEXT PRIMARY KEY,
                    idempotency_key TEXT UNIQUE,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        workflow_id: str,
        idempotency_key: str,
        status: str,
        state: dict[str, Any],
        updated_at: str,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints(
                    workflow_id, idempotency_key, status, state_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    workflow_id,
                    idempotency_key,
                    status,
                    json.dumps(state, ensure_ascii=False),
                    updated_at,
                ),
            )

    def get(self, workflow_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "workflow_id": row["workflow_id"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "state": json.loads(row["state_json"]),
            "updated_at": row["updated_at"],
        }

    def get_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE idempotency_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "workflow_id": row["workflow_id"],
            "idempotency_key": row["idempotency_key"],
            "status": row["status"],
            "state": json.loads(row["state_json"]),
            "updated_at": row["updated_at"],
        }
