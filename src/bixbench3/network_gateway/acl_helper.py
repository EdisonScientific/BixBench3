from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

from .allowlist_store import AllowListStore
from .policy import NetworkPolicy


def main() -> int:
    log_dir = Path(
        os.environ.get(
            "BIXBENCH3_BROKER_LOG_DIR",
            "/tmp/bixbench3-broker-logs",
        )
    )
    try:
        policy = NetworkPolicy.from_path(Path(os.environ["BIXBENCH3_BROKER_POLICY"]))
        store = AllowListStore(
            log_dir / "allowlist.jsonl",
            policy,
            writable=False,
        )
        for line in sys.stdin:
            request = line.strip().split()
            value = request[0] if request else ""
            allowed = store.allows_url(value) if value else False
            print("OK" if allowed else "ERR", flush=True)
        return 0
    except Exception:
        log_dir.mkdir(parents=True, exist_ok=True)
        with (log_dir / "acl_helper_error.log").open("a", encoding="utf-8") as handle:
            traceback.print_exc(file=handle)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
