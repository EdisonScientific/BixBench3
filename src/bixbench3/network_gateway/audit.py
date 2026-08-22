from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, payload: dict[str, object]) -> None:
        record = {"timestamp": datetime.now(UTC).isoformat(), **payload}
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
