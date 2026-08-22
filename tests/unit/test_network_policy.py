from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from bixbench3.network_gateway import supervisor
from bixbench3.network_gateway.allowlist_store import AllowListStore
from bixbench3.network_gateway.app import ApprovalState
from bixbench3.network_gateway.policy import (
    NetworkPolicy,
    ScopeRejection,
    normalize_url,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def _policy(**overrides: object) -> NetworkPolicy:
    payload: dict[str, object] = {
        "mode": "allowlist_with_tool",
        "policy_version": "test-v1",
        "base_allow_domains": ["pypi.org"],
        "paper_allow_domains": [],
        "paper_allow_url_prefixes": [],
        "paper_allow_exact_urls": [],
        "answer_source_urls": ["https://example.org/supplement.csv"],
        "answer_source_domains": ["example.org", "ncbi.nlm.nih.gov"],
        "protected_gcs_buckets": ["bixbench3-ground-truth"],
        "tool_enabled": True,
        "adjudicator_model": "openai/gpt-5.5",
        "adjudicator_reasoning_effort": "low",
        "decision_timeout_seconds": 30,
        "max_tool_calls": 50,
        "max_fetch_bytes": 1024,
        "default_on_error": "deny",
    }
    payload.update(overrides)
    return NetworkPolicy(payload)


def test_normalize_url_removes_default_port_and_fragment() -> None:
    assert normalize_url("HTTPS://Example.Org:443/a?x=1#section") == ("https://example.org/a?x=1")


def test_squid_helper_opens_root_owned_allowlist_read_only(tmp_path: Path) -> None:
    path = tmp_path / "allowlist.jsonl"
    AllowListStore(path, _policy())
    path.chmod(0o444)

    helper_store = AllowListStore(path, _policy(), writable=False)

    assert helper_store.allows_url("https://pypi.org/simple")
    with pytest.raises(PermissionError):
        helper_store.add({"kind": "domain", "value": "example.net"})


@pytest.mark.parametrize(
    "url",
    [
        "https://storage.googleapis.com/bixbench3-ground-truth/releases/v1.0.0/task.json",
        "https://storage.cloud.google.com/bixbench3-ground-truth/releases/v1.0.0/task.json",
        "https://bixbench3-ground-truth.storage.googleapis.com/releases/v1.0.0/task.json",
        "https://www.googleapis.com/storage/v1/b/bixbench3-ground-truth/o/task.json",
        "https://storage.googleapis.com/download/storage/v1/b/bixbench3-ground-truth/o/task.json?alt=media",
        "https://storage.googleapis.com/bixbench3-ground-truth%252Freleases%252Fv1.0.0%252Ftask.json",
    ],
)
def test_blocks_all_ground_truth_bucket_url_forms(url: str) -> None:
    assert _policy().block_reason(url) == "denied_protected_source"


def test_ground_truth_bucket_match_has_path_boundary() -> None:
    policy = _policy()

    assert (
        policy.block_reason(
            "https://storage.googleapis.com/bixbench3-ground-truth-copy/file"
        )
        is None
    )


def test_blocks_exact_answer_source_with_added_query() -> None:
    policy = _policy()

    assert (
        policy.block_reason("https://example.org/supplement.csv?download=1")
        == "denied_protected_source"
    )
    assert policy.block_reason("https://example.org/other.csv") is None


def test_blocks_answer_query_with_reordered_or_added_parameters() -> None:
    policy = _policy(
        answer_source_urls=[
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=SRR1"
        ]
    )

    assert (
        policy.block_reason(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?id=SRR1&tool=x&db=sra"
        )
        == "denied_protected_source"
    )
    assert (
        policy.block_reason(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=gene&id=672"
        )
        is None
    )


def test_blocks_percent_encoded_answer_path() -> None:
    assert (
        _policy().block_reason("https://example.org/%73upplement.csv") == "denied_protected_source"
    )


def test_answer_source_domain_remains_an_adjudicator_hint() -> None:
    policy = _policy()

    assert policy.block_reason("https://ncbi.nlm.nih.gov/gene/672") is None


def test_static_allow_entry_cannot_override_exact_answer_block() -> None:
    policy = _policy(
        paper_allow_exact_urls=[
            {
                "value": "https://example.org/supplement.csv",
                "reason": "stale unsafe allow",
            }
        ]
    )

    assert policy.matching_initial_allow("https://example.org/supplement.csv") is None


def test_rejects_direct_domain_allow_for_exact_answer_host() -> None:
    with pytest.raises(ValueError, match="exact answer source"):
        _policy(paper_allow_domains=[{"value": "example.org"}])


def test_gateway_blocks_exact_answer_before_adjudication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(_policy().payload), encoding="utf-8")
    monkeypatch.setenv("BIXBENCH3_BROKER_POLICY", str(policy_path))
    monkeypatch.setenv("BIXBENCH3_BROKER_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("BIXBENCH3_BROKER_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(
        "BIXBENCH3_APPROVED_WEB_DIR",
        str(tmp_path / "workspace" / "approved_web"),
    )
    monkeypatch.setenv("BIXBENCH3_BROKER_TOKEN", "test-token")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("Compute the requested outputs.", encoding="utf-8")
    runtime_path = tmp_path / "case.json"
    runtime_path.write_text('{"task_id":"paper-1"}', encoding="utf-8")
    monkeypatch.setenv("BIXBENCH3_TASK_PROMPT", str(prompt_path))
    monkeypatch.setenv("BIXBENCH3_RUNTIME_CASE", str(runtime_path))

    state = ApprovalState()
    decision = state.approve(
        "https://example.org/supplement.csv",
        "fetch published answers",
    )

    assert decision["decision"] == "deny"
    assert decision["reason_code"] == "denied_protected_source"
    assert decision["tool_calls_remaining"] == 50
    assert state.tool_calls == 0
    assert state.policy.payload["agent_prompt"] == "Compute the requested outputs."
    assert state.policy.payload["task_id"] == "paper-1"
    records = [
        json.loads(line)
        for line in (tmp_path / "logs" / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["rule"] == "exact_answer_block"

    ground_truth_decision = state.approve(
        "https://storage.googleapis.com/bixbench3-ground-truth/releases/v1.0.0/task.json",
        "fetch benchmark ground truth",
    )
    assert ground_truth_decision["decision"] == "deny"
    assert ground_truth_decision["reason_code"] == "denied_protected_source"
    assert ground_truth_decision["tool_calls_remaining"] == 50
    assert state.tool_calls == 0


def test_scope_classification_rejects_protected_source() -> None:
    policy = _policy()

    with pytest.raises(ScopeRejection, match="protected_source"):
        policy.classify_scope(
            "https://storage.googleapis.com/bixbench3-ground-truth/file.csv",
            "https://storage.googleapis.com/bixbench3-ground-truth/file.csv",
        )


@pytest.mark.parametrize(
    "domain",
    ["storage.googleapis.com", "storage.cloud.google.com", "www.googleapis.com"],
)
def test_rejects_broad_object_storage_allowlist(domain: str) -> None:
    with pytest.raises(ValueError, match="cannot be broadly allowlisted"):
        _policy(base_allow_domains=[domain])


def test_classifies_exact_prefix_and_package_domain_scopes() -> None:
    policy = _policy()
    requested = "https://github.com/org/tool/releases/download/v1/tool.tar.gz"

    assert policy.classify_scope(requested, requested) == ("exact", requested)
    assert policy.classify_scope(requested, "https://github.com/org/tool/") == (
        "prefix",
        "https://github.com/org/tool",
    )
    assert policy.classify_scope(
        "https://bioconductor.org/packages/release/bioc/html/DESeq2.html",
        "bioconductor.org",
    ) == ("domain", "bioconductor.org")


def test_rejects_broad_github_domain_scope() -> None:
    with pytest.raises(ScopeRejection, match="restricted_domain_scope"):
        _policy().classify_scope(
            "https://github.com/org/tool/releases/download/v1/tool.tar.gz",
            "github.com",
        )


def test_gateway_image_pins_runtime_dependencies() -> None:
    dockerfile = (
        REPOSITORY_ROOT / "src" / "bixbench3" / "network_gateway" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "openai==2.36.0" in dockerfile
    assert "pydantic==2.13.4" in dockerfile
    assert "RUN pip install --no-cache-dir openai pydantic" not in dockerfile


def test_compose_requires_two_local_private_images() -> None:
    compose = yaml.safe_load(
        (REPOSITORY_ROOT / "sandbox" / "compose.yaml").read_text(encoding="utf-8")
    )
    services = compose["services"]

    assert services["network-gateway"]["image"].startswith("${BIXBENCH3_NETWORK_GATEWAY_IMAGE:")
    assert services["network-gateway"]["pull_policy"] == "never"
    assert services["default"]["image"].startswith("${BIXBENCH3_SANDBOX_IMAGE:")
    assert services["default"]["pull_policy"] == "never"


def test_supervisor_starts_squid_on_port_3128(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "etc" / "squid"
    squid_log_dir = tmp_path / "var" / "log" / "squid"
    broker_log_dir = tmp_path / "broker-logs"
    monkeypatch.setenv("BIXBENCH3_SQUID_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("BIXBENCH3_SQUID_LOG_DIR", str(squid_log_dir))
    monkeypatch.setenv("BIXBENCH3_BROKER_LOG_DIR", str(broker_log_dir))
    monkeypatch.setenv(
        "BIXBENCH3_BROKER_CACHE_DIR",
        str(tmp_path / "broker-cache"),
    )
    monkeypatch.setenv(
        "BIXBENCH3_APPROVED_WEB_DIR",
        str(tmp_path / "approved-web"),
    )
    run_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []
    thread_targets: list[Callable[[], None]] = []

    class FakeThread:
        def __init__(self, *, target: Callable[[], None], daemon: bool) -> None:
            assert daemon is True
            thread_targets.append(target)

        def start(self) -> None:
            return

    class FakeProcess:
        def wait(self) -> int:
            return 0

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        run_calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    def fake_popen(argv: list[str], **kwargs: object) -> FakeProcess:
        del kwargs
        popen_calls.append(argv)
        return FakeProcess()

    monkeypatch.setattr(supervisor.threading, "Thread", FakeThread)
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(supervisor.subprocess, "Popen", fake_popen)

    assert supervisor.main() == 0
    assert thread_targets == [supervisor.run_server]
    assert "http_port 3128" in (config_dir / "squid.conf").read_text(encoding="utf-8")
    assert run_calls[-1][:4] == ["squid", "-k", "parse", "-f"]
    assert popen_calls == [["squid", "-N", "-d", "1", "-f", str(config_dir / "squid.conf")]]
