from __future__ import annotations

import hashlib
import json
import re
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .policy import NetworkPolicy, ScopeRejection, normalize_url

DEFAULT_MAX_BYTES = 25 * 1024 * 1024 * 1024


class ResponseHeaders(Protocol):
    def get(self, name: str, default: str | None = None) -> str | None: ...


class FetchResponse(Protocol):
    headers: ResponseHeaders

    def read(self, amount: int = -1) -> bytes: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class FetchResult:
    ok: bool
    local_path: Path | None = None
    manifest_path: Path | None = None
    fetch_error_code: str | None = None
    final_url: str | None = None
    content_type: str | None = None
    bytes: int = 0
    sha256: str | None = None

    def manifest_payload(
        self,
        *,
        requested_url: str,
        granted_scope: str,
        scope_kind: str,
    ) -> dict[str, object]:
        payload: dict[str, object] = asdict(self)
        payload["local_path"] = str(self.local_path) if self.local_path else None
        payload["manifest_path"] = str(self.manifest_path) if self.manifest_path else None
        payload.update(
            {
                "requested_url": requested_url,
                "granted_scope": granted_scope,
                "scope_kind": scope_kind,
                "fetched_at": datetime.now(UTC).isoformat(),
            }
        )
        return payload


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: Message,
        newurl: str,
    ) -> Request | None:
        del req, fp, code, msg, headers, newurl
        return None


class WorkspaceFetcher:
    def __init__(self, approved_web_dir: Path) -> None:
        self.approved_web_dir = approved_web_dir

    def fetch(
        self,
        *,
        requested_url: str,
        granted_scope: str,
        scope_kind: str,
        policy: NetworkPolicy,
        max_bytes: int = DEFAULT_MAX_BYTES,
        request_timeout_seconds: int = 120,
        clone_timeout_seconds: int = 300,
    ) -> FetchResult:
        return fetch_to_workspace(
            requested_url=requested_url,
            granted_scope=granted_scope,
            scope_kind=scope_kind,
            approved_web_dir=self.approved_web_dir,
            policy=policy,
            max_bytes=max_bytes,
            request_timeout_seconds=request_timeout_seconds,
            clone_timeout_seconds=clone_timeout_seconds,
        )


def fetch_to_workspace(
    *,
    requested_url: str,
    granted_scope: str,
    scope_kind: str,
    approved_web_dir: Path,
    policy: NetworkPolicy,
    max_bytes: int = DEFAULT_MAX_BYTES,
    request_timeout_seconds: int = 120,
    clone_timeout_seconds: int = 300,
) -> FetchResult:
    try:
        normalized = normalize_url(requested_url)
    except ValueError:
        return FetchResult(ok=False, fetch_error_code="unsupported_scheme")
    if policy.block_reason(normalized):
        return FetchResult(
            ok=False,
            final_url=normalized,
            fetch_error_code="protected_source",
        )
    if scope_kind not in {"exact", "prefix"}:
        return FetchResult(
            ok=False,
            final_url=normalized,
            fetch_error_code="unsupported_scope",
        )
    approved_web_dir.mkdir(parents=True, exist_ok=True)
    try:
        destination = _safe_destination(approved_web_dir, normalized)
    except OSError:
        return FetchResult(
            ok=False,
            final_url=normalized,
            fetch_error_code="disk_error",
        )
    manifest = destination.with_name(f"{destination.name}.manifest.json")
    cached = _cached_result(destination, manifest)
    if cached:
        return cached
    if _looks_like_git_repo(granted_scope):
        return _clone_repo(
            url=granted_scope,
            requested_url=normalized,
            granted_scope=granted_scope,
            scope_kind=scope_kind,
            destination=destination,
            manifest_path=manifest,
            timeout=clone_timeout_seconds,
        )
    return _download_file(
        requested_url=normalized,
        granted_scope=granted_scope,
        scope_kind=scope_kind,
        destination=destination,
        manifest_path=manifest,
        policy=policy,
        max_bytes=max_bytes,
        timeout=request_timeout_seconds,
    )


def _clone_repo(
    *,
    url: str,
    requested_url: str,
    granted_scope: str,
    scope_kind: str,
    destination: Path,
    manifest_path: Path,
    timeout: int,
) -> FetchResult:
    if not shutil.which("git"):
        return FetchResult(
            ok=False,
            final_url=url,
            fetch_error_code="git_unavailable",
        )
    lock_path = destination.with_suffix(destination.suffix + ".lock")
    try:
        lock_path.touch(exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", url, str(destination)],
            check=True,
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        result = FetchResult(
            ok=True,
            local_path=destination,
            manifest_path=manifest_path,
            final_url=url,
        )
        _write_manifest(
            result,
            requested_url=requested_url,
            granted_scope=granted_scope,
            scope_kind=scope_kind,
        )
        return result
    except subprocess.TimeoutExpired:
        return FetchResult(ok=False, final_url=url, fetch_error_code="timeout")
    except (OSError, subprocess.SubprocessError):
        return FetchResult(
            ok=False,
            final_url=url,
            fetch_error_code="git_clone_failed",
        )
    finally:
        lock_path.unlink(missing_ok=True)


def _download_file(
    *,
    requested_url: str,
    granted_scope: str,
    scope_kind: str,
    destination: Path,
    manifest_path: Path,
    policy: NetworkPolicy,
    max_bytes: int,
    timeout: int,
) -> FetchResult:
    opener = build_opener(NoRedirectHandler)
    current_url = requested_url
    redirects = 0
    while True:
        try:
            request = Request(
                current_url,
                headers={"user-agent": "bixbench3-network-gateway/1.0"},
            )
            response: FetchResponse = opener.open(request, timeout=timeout)
            break
        except HTTPError as error:
            if error.code in {301, 302, 303, 307, 308}:
                redirects += 1
                if redirects > 5:
                    return _fetch_error(current_url, "redirect_off_scope")
                location = error.headers.get("location")
                if not location:
                    return _fetch_error(current_url, "redirect_off_scope")
                current_url = urljoin(current_url, location)
                if not _redirect_allowed(current_url, granted_scope, scope_kind, policy):
                    return _fetch_error(current_url, "redirect_off_scope")
                continue
            return _fetch_error(current_url, f"http_{error.code}")
        except TimeoutError:
            return _fetch_error(current_url, "timeout")
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                return _fetch_error(current_url, "timeout")
            if isinstance(error.reason, socket.gaierror):
                return _fetch_error(current_url, "dns_error")
            return _fetch_error(current_url, "connection_error")
        except ValueError:
            return _fetch_error(current_url, "unsupported_scheme")
    return _stream_response(
        response=response,
        current_url=current_url,
        requested_url=requested_url,
        granted_scope=granted_scope,
        scope_kind=scope_kind,
        destination=destination,
        manifest_path=manifest_path,
        max_bytes=max_bytes,
    )


def _stream_response(
    *,
    response: FetchResponse,
    current_url: str,
    requested_url: str,
    granted_scope: str,
    scope_kind: str,
    destination: Path,
    manifest_path: Path,
    max_bytes: int,
) -> FetchResult:
    try:
        content_type = response.headers.get("content-type")
        digest = hashlib.sha256()
        total = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    temporary_path.unlink(missing_ok=True)
                    return FetchResult(
                        ok=False,
                        final_url=current_url,
                        content_type=content_type,
                        bytes=total,
                        fetch_error_code="too_large",
                    )
                digest.update(chunk)
                temporary.write(chunk)
        temporary_path.replace(destination)
        result = FetchResult(
            ok=True,
            local_path=destination,
            manifest_path=manifest_path,
            final_url=current_url,
            content_type=content_type,
            bytes=total,
            sha256=digest.hexdigest(),
        )
        _write_manifest(
            result,
            requested_url=requested_url,
            granted_scope=granted_scope,
            scope_kind=scope_kind,
        )
        return result
    except OSError:
        return _fetch_error(current_url, "disk_error")
    except Exception:
        return _fetch_error(current_url, "unknown_fetch_error")
    finally:
        response.close()


def _looks_like_git_repo(url: str) -> bool:
    return bool(
        re.match(
            r"https://(github|gitlab|bitbucket)\.com/[^/]+/[^/?#]+/?$",
            url.rstrip("/"),
        )
    )


def _safe_destination(root: Path, url: str) -> Path:
    digest = hashlib.sha256(url.encode()).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", url).strip("-")[:96]
    destination = (root / f"{slug or 'approved-resource'}-{digest}").resolve()
    resolved_root = root.resolve()
    if resolved_root not in destination.parents:
        raise OSError("destination escapes approved web directory")
    return destination


def _cached_result(destination: Path, manifest: Path) -> FetchResult | None:
    if not destination.exists() or not manifest.exists():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None
        return FetchResult(
            ok=True,
            local_path=destination,
            manifest_path=manifest,
            final_url=_optional_string(payload.get("final_url")),
            content_type=_optional_string(payload.get("content_type")),
            bytes=_nonnegative_int(payload.get("bytes")),
            sha256=_optional_string(payload.get("sha256")),
        )
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _write_manifest(
    result: FetchResult,
    *,
    requested_url: str,
    granted_scope: str,
    scope_kind: str,
) -> None:
    if not result.manifest_path:
        return
    payload = result.manifest_payload(
        requested_url=requested_url,
        granted_scope=granted_scope,
        scope_kind=scope_kind,
    )
    result.manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _redirect_allowed(
    url: str,
    granted_scope: str,
    scope_kind: str,
    policy: NetworkPolicy,
) -> bool:
    try:
        if policy.block_reason(url):
            return False
        if scope_kind == "exact":
            return normalize_url(url) == normalize_url(granted_scope)
        kind, _ = policy.classify_scope(url, granted_scope)
        return kind in {"exact", "prefix", "domain"}
    except (ScopeRejection, ValueError):
        return False


def _fetch_error(url: str, code: str) -> FetchResult:
    return FetchResult(ok=False, final_url=url, fetch_error_code=code)


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    return 0
