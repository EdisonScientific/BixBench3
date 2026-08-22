from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock


class DecisionCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = Lock()
        self.connection.execute(
            "create table if not exists decisions "
            "(policy_version text, url text, payload text, "
            "primary key(policy_version, url))"
        )

    def get(self, policy_version: str, url: str) -> dict[str, object] | None:
        with self._lock:
            row = self.connection.execute(
                "select payload from decisions where policy_version = ? and url = ?",
                (policy_version, url),
            ).fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        return payload if isinstance(payload, dict) else None

    def put(
        self,
        policy_version: str,
        url: str,
        payload: dict[str, object],
    ) -> None:
        with self._lock:
            self.connection.execute(
                "insert or replace into decisions(policy_version, url, payload) values (?, ?, ?)",
                (policy_version, url, json.dumps(payload, sort_keys=True)),
            )
            self.connection.commit()
