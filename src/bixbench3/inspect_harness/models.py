from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PromptBundle(StrictModel):
    prompt_path: str

    @field_validator("prompt_path")
    @classmethod
    def validate_prompt_path(cls, value: str) -> str:
        return _validate_relative_path(value, field_name="prompt path")


class ExpectedOutput(StrictModel):
    artifact_id: str
    path: str
    format: str | None = None
    row_key: str | None = None
    description: str | None = None

    @model_validator(mode="before")
    @classmethod
    def unwrap_expected_output(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        nested = value.get("expected_output") or value.get("output_format")
        if not isinstance(nested, dict):
            return value
        artifact_id = value.get("artifact_id")
        return {"artifact_id": artifact_id, **nested}

    @field_validator("path")
    @classmethod
    def validate_output_path(cls, value: str) -> str:
        return _validate_relative_path(value, field_name="expected output path")

    @property
    def sandbox_path(self) -> str:
        path = PurePosixPath(self.path)
        if path.parts and path.parts[0] == "outputs":
            path = PurePosixPath(*path.parts[1:])
        return str(PurePosixPath("/outputs") / path)


class NetworkAllowEntry(StrictModel):
    value: str
    reason: str | None = None


class NetworkAccessPolicy(StrictModel):
    mode: str = "allowlist_with_tool"
    policy_version: str = "bixbench3-network-policy-v1"
    base_allow_domains: list[str] = Field(default_factory=list)
    paper_allow_domains: list[NetworkAllowEntry] = Field(default_factory=list)
    paper_allow_url_prefixes: list[NetworkAllowEntry] = Field(default_factory=list)
    paper_allow_exact_urls: list[NetworkAllowEntry] = Field(default_factory=list)
    answer_source_urls: list[str] = Field(default_factory=list)
    answer_source_domains: list[str] = Field(default_factory=list)
    permitted_domain_scopes: list[str] = Field(default_factory=list)
    restricted_domain_scopes: list[str] = Field(default_factory=list)
    protected_gcs_buckets: list[str] = Field(
        default_factory=lambda: ["bixbench3-ground-truth"]
    )
    tool_enabled: bool = True
    adjudicator_model: str = "openai/gpt-5.5"
    adjudicator_reasoning_effort: str = "low"
    decision_timeout_seconds: int = 30
    max_tool_calls: int = 50
    max_fetch_bytes: int = 25 * 1024 * 1024 * 1024
    default_on_error: str = "deny"
    notes: str | None = None

    @model_validator(mode="after")
    def validate_canonical_policy(self) -> Self:
        if self.mode != "allowlist_with_tool":
            raise ValueError("network policy mode must be allowlist_with_tool")
        if not self.tool_enabled:
            raise ValueError("request_web_access must remain enabled")
        if self.adjudicator_model != "openai/gpt-5.5":
            raise ValueError("network adjudicator model must be openai/gpt-5.5")
        if self.adjudicator_reasoning_effort != "low":
            raise ValueError("network adjudicator reasoning effort must be low")
        if self.default_on_error != "deny":
            raise ValueError("network policy errors must default to deny")
        if self.decision_timeout_seconds <= 0:
            raise ValueError("decision_timeout_seconds must be positive")
        if self.max_tool_calls <= 0:
            raise ValueError("max_tool_calls must be positive")
        if self.max_fetch_bytes <= 0:
            raise ValueError("max_fetch_bytes must be positive")
        return self


class RuntimeCase(StrictModel):
    task_id: str
    doi: str | None = None
    prompt: PromptBundle
    expected_outputs: list[ExpectedOutput]
    network_policy: NetworkAccessPolicy = Field(default_factory=NetworkAccessPolicy)

    @model_validator(mode="before")
    @classmethod
    def normalize_runtime_case(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "task_id" not in normalized and isinstance(normalized.get("paper_id"), str):
            normalized["task_id"] = normalized["paper_id"]
        if "prompt" not in normalized and isinstance(normalized.get("prompt_path"), str):
            normalized["prompt"] = {"prompt_path": normalized["prompt_path"]}
        if "expected_outputs" not in normalized and isinstance(normalized.get("artifacts"), list):
            normalized["expected_outputs"] = normalized["artifacts"]
        allowed = {
            "task_id",
            "doi",
            "prompt",
            "expected_outputs",
            "network_policy",
        }
        return {key: item for key, item in normalized.items() if key in allowed}


def _validate_relative_path(value: str, *, field_name: str) -> str:
    path = Path(value)
    posix = PurePosixPath(value)
    if path.is_absolute() or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"{field_name} must be a safe relative path")
    if not posix.parts:
        raise ValueError(f"{field_name} must not be empty")
    if field_name == "expected output path" and posix.parts == ("outputs",):
        raise ValueError("expected output path must name a file under outputs")
    return value
