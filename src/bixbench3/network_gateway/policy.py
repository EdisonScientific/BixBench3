from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

PERMITTED_DOMAIN_SCOPES = (
    "cran.r-project.org",
    "cloud.r-project.org",
    "bioconductor.org",
    "archive.bioconductor.org",
    "pypi.org",
    "files.pythonhosted.org",
    "pythonhosted.org",
    "conda.anaconda.org",
    "repo.anaconda.com",
    "anaconda.org",
    "registry.npmjs.org",
    "crates.io",
    "static.crates.io",
    "proxy.golang.org",
    "sum.golang.org",
    "astral.sh",
    "packagemanager.posit.co",
    "packages.posit.co",
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
)

RESTRICTED_DOMAIN_SCOPES = (
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "zenodo.org",
    "figshare.com",
    "osf.io",
    "drive.google.com",
    "dropbox.com",
)

FORBIDDEN_BASE_ALLOW_DOMAINS = (
    "storage.googleapis.com",
    "storage.cloud.google.com",
    "www.googleapis.com",
)

DEFAULT_PROTECTED_GCS_BUCKETS = ("bixbench3-ground-truth",)

BlockReason = Literal["denied_protected_source"]
ScopeKind = Literal["exact", "prefix", "domain"]
PolicyEntry = dict[str, str | None]


class ScopeRejection(ValueError):
    def __init__(self, code: str, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code)


def normalize_url(value: str) -> str:
    raw = value.strip()
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"expected an http(s) URL, got {value!r}")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise ValueError(f"expected a URL host, got {value!r}")
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError(f"invalid URL port in {value!r}") from error
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))


def host_from_url(value: str) -> str:
    return (urlsplit(normalize_url(value)).hostname or "").lower().rstrip(".")


def domain_matches(host: str, domain: str) -> bool:
    normalized_host = host.lower().rstrip(".")
    normalized_domain = domain.lower().rstrip(".")
    return normalized_host == normalized_domain or normalized_host.endswith(f".{normalized_domain}")


class NetworkPolicy:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.mode = _string(payload.get("mode"), "allowlist_with_tool")
        self.policy_version = _string(
            payload.get("policy_version"),
            "bixbench3-network-policy-v1",
        )
        self.base_allow_domains = _domains(payload.get("base_allow_domains"))
        self.answer_source_urls = [
            normalize_url(url) for url in _raw_strings(payload.get("answer_source_urls"))
        ]
        self.answer_source_domains = _domains(payload.get("answer_source_domains"))
        self.permitted_domain_scopes = _domains(payload.get("permitted_domain_scopes")) or list(
            PERMITTED_DOMAIN_SCOPES
        )
        restricted = _domains(payload.get("restricted_domain_scopes")) or list(
            RESTRICTED_DOMAIN_SCOPES
        )
        for domain in _domains(payload.get("fetch_instead_of_allow_domains")):
            if domain not in restricted:
                restricted.append(domain)
        self.restricted_domain_scopes = restricted
        self.paper_allow_domains = _entries(payload.get("paper_allow_domains"))
        self.paper_allow_url_prefixes = _entries(payload.get("paper_allow_url_prefixes"))
        self.paper_allow_exact_urls = _entries(payload.get("paper_allow_exact_urls"))
        self.protected_gcs_buckets = _bucket_names(payload.get("protected_gcs_buckets")) or list(
            DEFAULT_PROTECTED_GCS_BUCKETS
        )
        self.tool_enabled = _boolean(payload.get("tool_enabled"), True)
        self.adjudicator_model = _string(payload.get("adjudicator_model"), "openai/gpt-5.5")
        self.adjudicator_reasoning_effort = _string(
            payload.get("adjudicator_reasoning_effort"),
            "low",
        )
        self.decision_timeout_seconds = _integer(payload.get("decision_timeout_seconds"), 30)
        self.max_tool_calls = _integer(payload.get("max_tool_calls"), 50)
        self.max_fetch_bytes = _integer(
            payload.get("max_fetch_bytes"),
            25 * 1024 * 1024 * 1024,
        )
        self.default_on_error = _string(payload.get("default_on_error"), "deny")
        self.notes = _string(payload.get("notes"), "")
        self._validate()

    @classmethod
    def from_path(cls, path: Path) -> NetworkPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("network policy must be a JSON object")
        return cls(payload)

    def block_reason(self, value: str) -> BlockReason | None:
        normalized = normalize_url(value)
        if self._matches_exact_answer_source(normalized):
            return "denied_protected_source"
        if self._matches_protected_gcs_bucket(normalized):
            return "denied_protected_source"
        return None

    def is_base_allowed_host(self, host: str) -> bool:
        return any(domain_matches(host, domain) for domain in self.base_allow_domains)

    def matching_initial_allow(self, url: str) -> PolicyEntry | None:
        normalized = normalize_url(url)
        if self.block_reason(normalized):
            return None
        host = host_from_url(normalized)
        for entry in self.paper_allow_exact_urls:
            if normalized == normalize_url(_entry_value(entry)):
                return entry
        for entry in self.paper_allow_url_prefixes:
            prefix = normalize_url(_entry_value(entry))
            prefix_parts = urlsplit(prefix)
            if host == (prefix_parts.hostname or "").lower().rstrip(".") and _path_prefix_matches(
                urlsplit(normalized).path or "/",
                _normalized_prefix_path(prefix_parts.path or "/"),
            ):
                return entry
        for entry in self.paper_allow_domains:
            if domain_matches(host, _entry_value(entry)):
                return entry
        return None

    def classify_scope(
        self,
        requested_url: str,
        granted_scope: str | None,
    ) -> tuple[ScopeKind, str]:
        requested = normalize_url(requested_url)
        if self.block_reason(requested):
            raise ScopeRejection("protected_source")
        requested_parts = urlsplit(requested)
        raw_scope = (granted_scope or requested).strip()
        if not raw_scope:
            raise ScopeRejection("missing_scope")
        if "://" not in raw_scope:
            return self._classify_domain_scope(raw_scope)
        scope = normalize_url(raw_scope)
        if self.block_reason(scope):
            raise ScopeRejection("protected_source")
        scope_parts = urlsplit(scope)
        if (
            scope_parts.scheme != requested_parts.scheme
            or scope_parts.netloc != requested_parts.netloc
        ):
            raise ScopeRejection("host_mismatch")
        if scope == requested:
            return "exact", scope
        if scope_parts.query:
            raise ScopeRejection("query_scope_not_exact")
        host = scope_parts.hostname or ""
        scope_path = _normalized_prefix_path(scope_parts.path or "/")
        requested_path = requested_parts.path or "/"
        if scope_path == "/":
            if self.is_permitted_domain_scope(host):
                return "domain", host
            raise ScopeRejection("root_scope")
        if not _path_prefix_matches(requested_path, scope_path):
            raise ScopeRejection("path_not_a_prefix")
        return "prefix", urlunsplit((scope_parts.scheme, scope_parts.netloc, scope_path, "", ""))

    def is_permitted_domain_scope(self, host: str) -> bool:
        return any(domain_matches(host, domain) for domain in self.permitted_domain_scopes)

    def _classify_domain_scope(self, granted_scope: str) -> tuple[ScopeKind, str]:
        host = granted_scope.strip().lower().rstrip(".")
        if not host or "/" in host or ":" in host or "?" in host or "#" in host:
            raise ScopeRejection("malformed_domain_scope")
        if any(domain_matches(host, domain) for domain in self.restricted_domain_scopes):
            raise ScopeRejection("restricted_domain_scope")
        if not self.is_permitted_domain_scope(host) and not self._is_paper_allowed_domain_scope(
            host
        ):
            raise ScopeRejection("domain_scope_not_permitted")
        return "domain", host

    def _is_paper_allowed_domain_scope(self, host: str) -> bool:
        return any(domain_matches(host, _entry_value(entry)) for entry in self.paper_allow_domains)

    def _matches_exact_answer_source(self, normalized: str) -> bool:
        requested = urlsplit(normalized)
        for source in self.answer_source_urls:
            protected = urlsplit(source)
            if (
                requested.scheme == protected.scheme
                and requested.netloc == protected.netloc
                and _repeated_unquote(requested.path) == _repeated_unquote(protected.path)
                and _query_contains(requested.query, protected.query)
            ):
                return True
        return False

    def _matches_protected_gcs_bucket(self, normalized: str) -> bool:
        parts = urlsplit(normalized)
        host = (parts.hostname or "").lower().rstrip(".")
        decoded_path = _repeated_unquote(parts.path)
        segments = [segment for segment in decoded_path.split("/") if segment]
        for bucket in self.protected_gcs_buckets:
            if host == f"{bucket}.storage.googleapis.com":
                return True
            if host in {"storage.googleapis.com", "storage.cloud.google.com"}:
                if segments and segments[0] == bucket:
                    return True
                if len(segments) >= 5 and segments[:4] == ["download", "storage", "v1", "b"]:
                    if segments[4] == bucket:
                        return True
            if host == "www.googleapis.com":
                for index, segment in enumerate(segments[:-1]):
                    if segment == "b" and segments[index + 1] == bucket:
                        return True
        return False

    def _validate(self) -> None:
        if self.mode != "allowlist_with_tool":
            raise ValueError("network policy mode must be allowlist_with_tool")
        if not self.tool_enabled:
            raise ValueError("request_web_access must remain enabled")
        if self.adjudicator_model != "openai/gpt-5.5":
            raise ValueError("adjudicator_model must be openai/gpt-5.5")
        if self.adjudicator_reasoning_effort != "low":
            raise ValueError("adjudicator_reasoning_effort must be low")
        if self.default_on_error != "deny":
            raise ValueError("default_on_error must be deny")
        if self.decision_timeout_seconds <= 0:
            raise ValueError("decision_timeout_seconds must be positive")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if self.max_fetch_bytes <= 0:
            raise ValueError("max_fetch_bytes must be positive")
        direct_domains = [
            *self.base_allow_domains,
            *self.permitted_domain_scopes,
            *[_entry_value(entry) for entry in self.paper_allow_domains],
        ]
        for configured in direct_domains:
            if any(
                domain_matches(configured, forbidden) or domain_matches(forbidden, configured)
                for forbidden in FORBIDDEN_BASE_ALLOW_DOMAINS
            ):
                raise ValueError(
                    f"object-storage domain cannot be broadly allowlisted: {configured}"
                )
        exact_answer_hosts = {host_from_url(url) for url in self.answer_source_urls}
        for configured in direct_domains:
            if any(domain_matches(host, configured) for host in exact_answer_hosts):
                raise ValueError(
                    "a domain containing an exact answer source cannot be directly "
                    f"allowlisted: {configured}"
                )


def _string(value: object, default: str) -> str:
    return value if isinstance(value, str) else default


def _boolean(value: object, default: bool) -> bool:
    return value if isinstance(value, bool) else default


def _integer(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    return value if isinstance(value, int) else default


def _domains(value: object) -> list[str]:
    return [item.strip().lower().rstrip(".") for item in _raw_strings(value)]


def _raw_strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _bucket_names(value: object) -> list[str]:
    buckets = _domains(value)
    for bucket in buckets:
        if "/" in bucket or ":" in bucket or bucket.startswith("."):
            raise ValueError(f"invalid protected GCS bucket name: {bucket}")
    return buckets


def _entries(value: object) -> list[PolicyEntry]:
    if not isinstance(value, list):
        return []
    entries: list[PolicyEntry] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            entries.append({"value": item.strip(), "reason": None})
        elif isinstance(item, dict):
            raw_value = item.get("value")
            raw_reason = item.get("reason")
            if isinstance(raw_value, str) and raw_value.strip():
                entries.append(
                    {
                        "value": raw_value.strip(),
                        "reason": raw_reason if isinstance(raw_reason, str) else None,
                    }
                )
    return entries


def _entry_value(entry: PolicyEntry) -> str:
    value = entry.get("value")
    return value if isinstance(value, str) else ""


def _normalized_prefix_path(path: str) -> str:
    if not path or path == "/":
        return "/"
    return path.rstrip("/")


def _path_prefix_matches(path: str, prefix: str) -> bool:
    if prefix == "/":
        return True
    return path == prefix or path.startswith(f"{prefix}/")


def _repeated_unquote(value: str) -> str:
    decoded = value
    for _ in range(3):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def _query_contains(requested: str, protected: str) -> bool:
    if not protected:
        return True
    requested_items = Counter(parse_qsl(requested, keep_blank_values=True))
    protected_items = Counter(parse_qsl(protected, keep_blank_values=True))
    return all(requested_items[item] >= count for item, count in protected_items.items())
