"""SQLite persistence for training jobs, their log lines and their metrics.

Standard library only (`sqlite3`); one connection per call so the HTTP threads and the
training threads never share a handle. WAL keeps readers (SSE) from blocking the writer.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    algorithm     TEXT NOT NULL,
    status        TEXT NOT NULL,
    mode          TEXT NOT NULL,
    created_at    REAL NOT NULL,
    started_at    REAL,
    finished_at   REAL,
    params        TEXT NOT NULL,
    command       TEXT NOT NULL,
    output_dir    TEXT,
    exit_code     INTEGER,
    message       TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id  TEXT NOT NULL,
    ts      REAL NOT NULL,
    stream  TEXT NOT NULL,
    line    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_job ON events (job_id, id);
CREATE TABLE IF NOT EXISTS metrics (
    job_id  TEXT NOT NULL,
    step    INTEGER NOT NULL,
    name    TEXT NOT NULL,
    value   REAL NOT NULL,
    PRIMARY KEY (job_id, step, name)
);
CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL,
    updated_at  REAL NOT NULL
);
"""

TERMINAL = {"finished", "failed", "stopped"}


class Store:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    # ------------------------------------------------------------------ jobs
    def create_job(
        self,
        *,
        name: str,
        algorithm: str,
        mode: str,
        params: dict[str, Any],
        command: list[str],
        output_dir: str,
        job_id: str | None = None,
    ) -> dict:
        job_id = job_id or f"job-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (id, name, algorithm, status, mode, created_at, params, command, output_dir)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    name,
                    algorithm,
                    "created",
                    mode,
                    time.time(),
                    json.dumps(params, ensure_ascii=False),
                    json.dumps(command, ensure_ascii=False),
                    output_dir,
                ),
            )
        return self.get_job(job_id)  # type: ignore[return-value]

    def get_job(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._job_row(row) if row else None

    def list_jobs(self, limit: int = 100) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [self._job_row(row) for row in rows]

    def update_job(self, job_id: str, **fields: Any) -> None:
        allowed = {"status", "started_at", "finished_at", "exit_code", "message", "output_dir", "command", "params"}
        updates = {k: v for k, v in fields.items() if k in allowed}
        if not updates:
            return
        assignments = ", ".join(f"{k} = ?" for k in updates)
        values = [
            json.dumps(v, ensure_ascii=False) if k in {"params", "command"} else v for k, v in updates.items()
        ]
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", (*values, job_id))

    def delete_job(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if job is None or job["status"] not in TERMINAL | {"created"}:
            return False
        with self._conn() as conn:
            conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            conn.execute("DELETE FROM events WHERE job_id = ?", (job_id,))
            conn.execute("DELETE FROM metrics WHERE job_id = ?", (job_id,))
        return True

    @staticmethod
    def _job_row(row: sqlite3.Row) -> dict:
        job = dict(row)
        job["params"] = json.loads(job["params"])
        job["command"] = json.loads(job["command"])
        return job

    # ---------------------------------------------------------------- events
    def append_event(self, job_id: str, line: str, stream: str = "stdout") -> int:
        with self._conn() as conn:
            cursor = conn.execute(
                "INSERT INTO events (job_id, ts, stream, line) VALUES (?,?,?,?)",
                (job_id, time.time(), stream, line),
            )
            return int(cursor.lastrowid or 0)

    def get_events(self, job_id: str, since: int = 0, limit: int = 2000) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, ts, stream, line FROM events WHERE job_id = ? AND id > ? ORDER BY id LIMIT ?",
                (job_id, since, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def last_event_id(self, job_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT MAX(id) AS last FROM events WHERE job_id = ?", (job_id,)).fetchone()
        return int(row["last"] or 0)

    # --------------------------------------------------------------- metrics
    def append_metric(self, job_id: str, step: int, name: str, value: float) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO metrics (job_id, step, name, value) VALUES (?,?,?,?)",
                (job_id, step, name, float(value)),
            )

    # -------------------------------------------------------------- settings
    def get_settings(self) -> dict[str, str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT key, value FROM settings ORDER BY key").fetchall()
        return {row["key"]: row["value"] for row in rows}

    def set_settings(self, values: Mapping[str, str | None]) -> None:
        """Write the given settings; an empty or null value deletes the key."""

        now = time.time()
        with self._conn() as conn:
            for key, value in values.items():
                if value is None or not str(value).strip():
                    conn.execute("DELETE FROM settings WHERE key = ?", (key,))
                else:
                    conn.execute(
                        "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)"
                        " ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                        (key, str(value), now),
                    )

    # --------------------------------------------------------------- metrics
    def get_metrics(self, job_id: str) -> dict[str, list[dict]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT step, name, value FROM metrics WHERE job_id = ? ORDER BY step", (job_id,)
            ).fetchall()
        series: dict[str, list[dict]] = {}
        for row in rows:
            series.setdefault(row["name"], []).append({"step": row["step"], "value": row["value"]})
        return series
