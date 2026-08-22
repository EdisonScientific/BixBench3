from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .policy import NetworkPolicy, domain_matches, host_from_url, normalize_url

AllowEntry = dict[str, str]


class AllowListStore:
    def __init__(
        self,
        path: Path,
        initial_policy: NetworkPolicy,
        *,
        writable: bool = True,
    ) -> None:
        self.path = path
        self.policy = initial_policy
        self.writable = writable
        self._lock = Lock()
        self._entries: list[AllowEntry] = []
        self._load_initial()
        self._load_persisted()
        if self.writable:
            self._persist_unlocked()

    def add(self, entry: AllowEntry) -> None:
        if not self.writable:
            raise PermissionError("read-only allowlist store cannot add entries")
        normalized = normalize_entry(entry)
        with self._lock:
            if not self._contains_entry(normalized):
                self._entries.append(normalized)
                self._persist_unlocked()

    def snapshot(self) -> list[AllowEntry]:
        with self._lock:
            return [dict(entry) for entry in self._entries]

    def allows_url(self, url_or_host: str) -> bool:
        with self._lock:
            self._load_persisted()
            return _allows(url_or_host, self._entries, self.policy)

    def _load_initial(self) -> None:
        for domain in self.policy.base_allow_domains:
            self._entries.append({"kind": "domain", "value": domain, "source": "base"})
        for entry in self.policy.paper_allow_domains:
            value = entry.get("value")
            if isinstance(value, str):
                self._entries.append({"kind": "domain", "value": value, "source": "paper"})

    def _load_persisted(self) -> None:
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            entry = {
                key: value
                for key in ("kind", "value", "source", "reason")
                if isinstance((value := payload.get(key)), str)
            }
            if "kind" in entry and "value" in entry and not self._contains_entry(entry):
                self._entries.append(normalize_entry(entry))

    def _persist_unlocked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in self._entries),
            encoding="utf-8",
        )

    def _contains_entry(self, entry: AllowEntry) -> bool:
        return any(
            existing.get("kind") == entry.get("kind")
            and existing.get("value") == entry.get("value")
            for existing in self._entries
        )


def normalize_entry(entry: AllowEntry) -> AllowEntry:
    kind = entry.get("kind")
    value = entry.get("value", "")
    if kind == "prefix":
        return {**entry, "value": normalize_prefix(value)}
    if kind == "exact":
        return {**entry, "value": normalize_url(value)}
    if kind == "domain":
        return {**entry, "value": value.strip().lower().rstrip(".")}
    raise ValueError(f"unsupported allowlist entry kind: {kind!r}")


def normalize_prefix(value: str) -> str:
    return normalize_url(value).rstrip("/")


def _allows(
    url_or_host: str,
    entries: list[AllowEntry],
    policy: NetworkPolicy,
) -> bool:
    if "://" in url_or_host:
        try:
            normalized = normalize_url(url_or_host)
        except ValueError:
            return False
        if policy.block_reason(normalized):
            return False
        host = host_from_url(normalized)
    else:
        normalized = ""
        host = url_or_host.split(":", 1)[0].lower().rstrip(".")
    for entry in entries:
        kind = entry.get("kind")
        value = entry.get("value", "").strip()
        if not value:
            continue
        if kind == "domain" and domain_matches(host, value):
            return True
        if normalized and kind == "exact" and normalized == value:
            return True
        if normalized and kind == "prefix" and _matches_prefix(normalized, value):
            return True
    return False


def _matches_prefix(normalized_url: str, prefix: str) -> bool:
    clean_prefix = prefix.rstrip("/")
    return normalized_url == clean_prefix or normalized_url.startswith(f"{clean_prefix}/")
