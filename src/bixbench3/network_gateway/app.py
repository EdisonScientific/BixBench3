from __future__ import annotations

import json
import os
from hmac import compare_digest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock

from .adjudicator import Adjudicator
from .allowlist_store import AllowListStore
from .audit import AuditLog
from .cache import DecisionCache
from .fetcher import WorkspaceFetcher
from .policy import NetworkPolicy, ScopeRejection, normalize_url

MAX_APPROVAL_REQUEST_BYTES = 64 * 1024
ALLOWED_DENY_CODES = {
    "denied_protected_source",
    "denied_processed_output",
    "denied_search_query",
    "denied_scope_too_broad",
    "denied_over_budget",
    "denied_other",
    "fetch_failed",
    "broker_error",
}


class ApprovalState:
    def __init__(self) -> None:
        policy_path = Path(os.environ["BIXBENCH3_BROKER_POLICY"])
        self.policy = NetworkPolicy(
            _load_policy_with_task_context(
                policy_path=policy_path,
                task_prompt_path=Path(
                    os.environ.get(
                        "BIXBENCH3_TASK_PROMPT",
                        "/task/prompt.md",
                    )
                ),
                runtime_case_path=Path(
                    os.environ.get(
                        "BIXBENCH3_RUNTIME_CASE",
                        "/task/case.json",
                    )
                ),
            )
        )
        log_dir = Path(
            os.environ.get(
                "BIXBENCH3_BROKER_LOG_DIR",
                "/tmp/bixbench3-broker-logs",
            )
        )
        cache_dir = Path(
            os.environ.get(
                "BIXBENCH3_BROKER_CACHE_DIR",
                "/tmp/bixbench3-broker-cache",
            )
        )
        self.approved_web_dir = Path(
            os.environ.get(
                "BIXBENCH3_APPROVED_WEB_DIR",
                "/workspace/approved_web",
            )
        )
        self.token = os.environ.get("BIXBENCH3_BROKER_TOKEN", "")
        if not self.token:
            raise ValueError("BIXBENCH3_BROKER_TOKEN must be set")
        self.store = AllowListStore(log_dir / "allowlist.jsonl", self.policy)
        self.audit = AuditLog(log_dir / "decisions.jsonl")
        self.cache = DecisionCache(cache_dir / "decisions.sqlite")
        self.fetcher = WorkspaceFetcher(self.approved_web_dir)
        self._adjudicator: Adjudicator | None = None
        self._budget_lock = Lock()
        self.tool_calls = 0

    def approve(self, url: str, reason: str) -> dict[str, object]:
        normalized = normalize_url(url)
        blocked = self.policy.block_reason(normalized)
        if blocked:
            return self._respond(
                agent_payload=_deny_payload(normalized, blocked),
                internal_audit={
                    "reason": "exact protected answer source",
                    "risk_type": "answer_artifact",
                },
                agent_reason=reason,
                rule="exact_answer_block",
            )
        initial = self.policy.matching_initial_allow(normalized)
        if initial:
            return self._allow(
                normalized,
                reason,
                initial.get("value"),
                "paper_allow",
                risk_type="reference_data",
                model_reason=initial.get("reason") or "approved by network policy",
            )
        cached = self.cache.get(self.policy.policy_version, normalized)
        if cached and self._cached_payload_valid(cached):
            return self._respond(
                agent_payload=cached,
                internal_audit={"reason": "cache hit", "risk_type": "unknown"},
                agent_reason=reason,
                rule="cache_hit",
            )
        if not self._consume_tool_call():
            return self._respond(
                agent_payload=_deny_payload(normalized, "denied_over_budget"),
                internal_audit={
                    "reason": "max_tool_calls exceeded",
                    "risk_type": "unknown",
                },
                agent_reason=reason,
                rule="budget",
            )
        try:
            model_decision = self.adjudicator.decide(
                url=normalized,
                reason=reason,
            ).model_dump()
        except Exception as error:
            return self._respond(
                agent_payload=_deny_payload(normalized, "broker_error"),
                internal_audit={
                    "reason": "adjudicator error",
                    "risk_type": "unknown",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                agent_reason=reason,
                rule="adjudicator_error",
            )
        if model_decision.get("decision") == "allow":
            decision = self._allow(
                normalized,
                reason,
                model_decision.get("granted_scope"),
                "adjudicator",
                risk_type=model_decision.get("risk_type"),
                model_reason=model_decision.get("reason"),
            )
        else:
            reason_code = _normalize_deny_code(model_decision.get("reason_code"))
            decision = self._respond(
                agent_payload=_deny_payload(normalized, reason_code),
                internal_audit={
                    "reason": model_decision.get("reason"),
                    "risk_type": model_decision.get("risk_type") or "unknown",
                },
                agent_reason=reason,
                rule="adjudicator",
            )
        if decision.get("decision") == "allow":
            self.cache.put(self.policy.policy_version, normalized, decision)
        return decision

    def _allow(
        self,
        requested_url: str,
        agent_reason: str,
        granted_scope: object,
        rule: str,
        *,
        risk_type: object = None,
        model_reason: object = None,
    ) -> dict[str, object]:
        scope = granted_scope if isinstance(granted_scope, str) else requested_url
        try:
            kind, value = self.policy.classify_scope(requested_url, scope)
        except ScopeRejection as error:
            reason_code = (
                "denied_protected_source"
                if error.code == "protected_source"
                else "denied_scope_too_broad"
            )
            return self._respond(
                agent_payload=_deny_payload(requested_url, reason_code),
                internal_audit={
                    "reason": f"invalid scope: {error.code}",
                    "risk_type": risk_type or "unknown",
                },
                agent_reason=agent_reason,
                rule="scope_rejected",
            )
        if kind == "domain":
            self.store.add(
                {
                    "kind": kind,
                    "value": value,
                    "source": "dynamic",
                    "reason": agent_reason,
                }
            )
            return self._respond_allow(
                requested_url=requested_url,
                granted_scope=value,
                scope_kind=kind,
                delivery="allowlist",
                risk_type=risk_type,
                model_reason=model_reason,
                agent_reason=agent_reason,
                rule=rule,
            )
        result = self.fetcher.fetch(
            requested_url=requested_url,
            granted_scope=value,
            scope_kind=kind,
            policy=self.policy,
            max_bytes=self.policy.max_fetch_bytes,
        )
        if not result.ok:
            reason_code = (
                "denied_protected_source"
                if result.fetch_error_code == "protected_source"
                else "fetch_failed"
            )
            return self._respond(
                agent_payload={
                    **_deny_payload(requested_url, reason_code),
                    "fetch_error_code": result.fetch_error_code or "unknown_fetch_error",
                },
                internal_audit={
                    "reason": model_reason,
                    "risk_type": risk_type or "unknown",
                    "fetch_error_code": result.fetch_error_code,
                    "final_url": result.final_url,
                    "bytes": result.bytes,
                },
                agent_reason=agent_reason,
                rule="fetch_failed",
            )
        local_path = _exposed_workspace_path(result.local_path, self.approved_web_dir)
        return self._respond_allow(
            requested_url=requested_url,
            granted_scope=value,
            scope_kind=kind,
            delivery="workspace_file",
            local_path=local_path,
            risk_type=risk_type,
            model_reason=model_reason,
            agent_reason=agent_reason,
            rule=rule,
            internal_extra={
                "manifest_path": _exposed_workspace_path(
                    result.manifest_path,
                    self.approved_web_dir,
                ),
                "final_url": result.final_url,
                "content_type": result.content_type,
                "bytes": result.bytes,
                "sha256": result.sha256,
            },
        )

    def _respond_allow(
        self,
        *,
        requested_url: str,
        granted_scope: str,
        scope_kind: str,
        delivery: str,
        risk_type: object,
        model_reason: object,
        agent_reason: str,
        rule: str,
        local_path: str | None = None,
        internal_extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "decision": "allow",
            "url": requested_url,
            "granted_scope": granted_scope,
            "scope_kind": scope_kind,
            "delivery": delivery,
            "reason_code": "approved",
        }
        if local_path:
            payload["local_path"] = local_path
        return self._respond(
            agent_payload=payload,
            internal_audit={
                "reason": model_reason,
                "risk_type": risk_type or "unknown",
                **(internal_extra or {}),
            },
            agent_reason=agent_reason,
            rule=rule,
        )

    def _respond(
        self,
        *,
        agent_payload: dict[str, object],
        internal_audit: dict[str, object],
        agent_reason: str,
        rule: str,
    ) -> dict[str, object]:
        agent_payload["tool_calls_remaining"] = self._remaining_tool_calls()
        record = {
            **agent_payload,
            **internal_audit,
            "requested_url": agent_payload.get("url"),
            "agent_reason": agent_reason,
            "rule": rule,
            "policy_version": self.policy.policy_version,
        }
        self.audit.write(record)
        return agent_payload

    def _consume_tool_call(self) -> bool:
        with self._budget_lock:
            if self.tool_calls >= self.policy.max_tool_calls:
                return False
            self.tool_calls += 1
            return True

    def _remaining_tool_calls(self) -> int:
        with self._budget_lock:
            return max(0, self.policy.max_tool_calls - self.tool_calls)

    def _cached_payload_valid(self, payload: dict[str, object]) -> bool:
        url = payload.get("url")
        if not isinstance(url, str) or self.policy.block_reason(url):
            return False
        if payload.get("delivery") != "workspace_file":
            return True
        local_path = payload.get("local_path")
        if not isinstance(local_path, str):
            return False
        return _actual_workspace_path(local_path, self.approved_web_dir).exists()

    @property
    def adjudicator(self) -> Adjudicator:
        if self._adjudicator is None:
            self._adjudicator = Adjudicator(self.policy)
        return self._adjudicator


class ApprovalHandler(BaseHTTPRequestHandler):
    state: ApprovalState

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok"})
            return
        if self.path == "/allowlist":
            self._json(200, {"entries": self.state.store.snapshot()})
            return
        self._json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/approve":
            self._json(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._json(403, {"error": "forbidden"})
            return
        requested_url = ""
        try:
            payload = self._read_json_body()
            raw_url = payload.get("url")
            if not isinstance(raw_url, str) or not raw_url:
                self._json(400, _deny_payload("", "broker_error"))
                return
            requested_url = raw_url
            raw_reason = payload.get("reason")
            reason = raw_reason if isinstance(raw_reason, str) else ""
            self._json(200, self.state.approve(requested_url, reason))
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as error:
            self.state.audit.write(
                {
                    **_deny_payload(requested_url, "broker_error"),
                    "requested_url": requested_url,
                    "rule": "invalid_request",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            self._json(400, _deny_payload(requested_url, "broker_error"))
        except Exception as error:
            payload = _deny_payload(requested_url, "broker_error")
            self.state.audit.write(
                {
                    **payload,
                    "requested_url": requested_url,
                    "rule": "broker_error",
                    "policy_version": self.state.policy.policy_version,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                }
            )
            self._json(200, payload)

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("content-length", "0"))
        if length <= 0 or length > MAX_APPROVAL_REQUEST_BYTES:
            raise ValueError("invalid approval request size")
        payload = json.loads(self.rfile.read(length).decode())
        if not isinstance(payload, dict):
            raise ValueError("approval request must be a JSON object")
        return payload

    def _authorized(self) -> bool:
        supplied = self.headers.get("authorization", "")
        expected = f"Bearer {self.state.token}"
        return compare_digest(supplied, expected)

    def _json(self, status: int, payload: dict[str, object]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    ApprovalHandler.state = ApprovalState()
    server = ThreadingHTTPServer((host, port), ApprovalHandler)
    server.serve_forever()


def _deny_payload(url: str, reason_code: object) -> dict[str, object]:
    return {
        "decision": "deny",
        "url": url,
        "reason_code": _normalize_deny_code(reason_code),
    }


def _normalize_deny_code(reason_code: object) -> str:
    if isinstance(reason_code, str) and reason_code in ALLOWED_DENY_CODES:
        return reason_code
    return "denied_other"


def _exposed_workspace_path(
    path: Path | None,
    approved_web_dir: Path,
) -> str | None:
    if path is None:
        return None
    try:
        relative = path.resolve().relative_to(approved_web_dir.resolve())
    except ValueError:
        return None
    return str(Path("/workspace/approved_web") / relative)


def _actual_workspace_path(path: str, approved_web_dir: Path) -> Path:
    prefix = "/workspace/approved_web/"
    if path.startswith(prefix):
        return approved_web_dir / path.removeprefix(prefix)
    return Path(path)


def _load_policy_with_task_context(
    *,
    policy_path: Path,
    task_prompt_path: Path,
    runtime_case_path: Path,
) -> dict[str, object]:
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("network policy must be a JSON object")
    enriched = dict(payload)
    if task_prompt_path.is_file():
        enriched["agent_prompt"] = task_prompt_path.read_text(encoding="utf-8")
    if runtime_case_path.is_file():
        runtime = json.loads(runtime_case_path.read_text(encoding="utf-8"))
        if not isinstance(runtime, dict):
            raise ValueError("runtime case must be a JSON object")
        task_id = runtime.get("task_id") or runtime.get("paper_id")
        if isinstance(task_id, str):
            enriched["task_id"] = task_id
    return enriched
