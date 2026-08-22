"""Strict public schemas for BixBench3 releases and runs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    TypeAdapter,
    field_validator,
    model_validator,
)

RELEASE_TASK_COUNT = 20
RELEASE_ARTIFACT_COUNT = 138
EXCLUDED_PAPER_ID = "10.1101_2025.08.14.670292_v1"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ReleaseVersion = Annotated[str, Field(pattern=r"^v[0-9]+\.[0-9]+\.[0-9]+$")]


def _validate_relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value in {"", "."}:
        raise ValueError("path must be a non-empty relative POSIX path without '..'")
    return value


def _validate_iso_datetime(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("value must be an ISO 8601 datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError("datetime must include a timezone")
    return value


def _validate_public_https_url(value: str) -> str:
    if not value.startswith("https://"):
        raise ValueError("URL must use HTTPS")
    return value


class StrictModel(BaseModel):
    """Immutable model with strict input types and no unknown fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RelativeFile(StrictModel):
    """A relative file reference and its media format."""

    path: str
    format: str

    @field_validator("path")
    @classmethod
    def path_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)


class ExpectedOutput(RelativeFile):
    """An artifact output contract visible to benchmark agents."""

    artifact_id: str
    row_key: str | None = None
    description: str | None = None


class GradingCheck(StrictModel):
    """One deterministic metric check for an artifact."""

    metric: Literal["f1", "ccc", "label_f1", "interval_f1"]
    axis: Literal["rows", "columns"] | None = None
    scope: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_axis(self) -> GradingCheck:
        if self.metric in {"f1", "ccc"} and self.axis is None:
            raise ValueError(f"{self.metric} checks require an axis")
        if self.metric in {"label_f1", "interval_f1"} and self.axis is not None:
            raise ValueError(f"{self.metric} checks do not use an axis")
        return self


class HFArtifactDefinition(StrictModel):
    """Public artifact definition embedded in an HF task row."""

    artifact_id: str
    output_path: str
    output_format: str
    row_key: str | None = None
    metric_checks: list[GradingCheck]
    value_columns: list[str] = Field(default_factory=list)
    value_rows: list[str] = Field(default_factory=list)
    analysis_category: str
    depends_on: list[str] = Field(default_factory=list)
    analysis_depth: int = Field(ge=1)
    grader_package_uri: str

    @field_validator("output_path")
    @classmethod
    def output_path_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("grader_package_uri")
    @classmethod
    def grader_uri_is_public(cls, value: str) -> str:
        return _validate_public_https_url(value)


_HF_ARTIFACT_LIST = TypeAdapter(list[HFArtifactDefinition])


class HFTaskRow(StrictModel):
    """One row in the 20-row Hugging Face task table."""

    benchmark_version: ReleaseVersion
    task_number: int = Field(ge=1, le=RELEASE_TASK_COUNT)
    paper_id: str
    doi: str
    research_objective: str
    method_guidance: str
    prompt: str
    analysis_categories: list[str]
    assay_types: list[str]
    data_types: list[str]
    categories: list[str]
    modalities: list[str]
    max_analysis_depth: int = Field(ge=1)
    subject: str
    artifact_count: int = Field(ge=1)
    input_size_gb: float = Field(gt=0)
    n_raw_sources: int = Field(ge=1)
    input_manifest_url: str
    agent_package_path: str
    runtime_package_path: str
    grader_package_uri: str
    artifacts_json: str

    @field_validator(
        "input_manifest_url",
        "grader_package_uri",
    )
    @classmethod
    def public_urls_only(cls, value: str) -> str:
        return _validate_public_https_url(value)

    @field_validator("agent_package_path", "runtime_package_path")
    @classmethod
    def package_paths_are_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @model_validator(mode="after")
    def validate_artifacts(self) -> HFTaskRow:
        try:
            payload = json.loads(self.artifacts_json)
        except json.JSONDecodeError as exc:
            raise ValueError("artifacts_json must contain valid JSON") from exc
        artifacts = _HF_ARTIFACT_LIST.validate_python(payload, strict=True)
        if len(artifacts) != self.artifact_count:
            raise ValueError("artifact_count must match artifacts_json")
        if len({artifact.artifact_id for artifact in artifacts}) != len(artifacts):
            raise ValueError("artifact IDs must be unique within a task")
        max_depth = max(artifact.analysis_depth for artifact in artifacts)
        if max_depth != self.max_analysis_depth:
            raise ValueError("max_analysis_depth must match artifacts_json")
        return self

    def artifacts(self) -> list[HFArtifactDefinition]:
        """Parse and validate the nested artifact definitions."""

        return _HF_ARTIFACT_LIST.validate_python(
            json.loads(self.artifacts_json),
            strict=True,
        )


class PublicCaseManifest(StrictModel):
    """One compact public case record written to cases.jsonl."""

    benchmark_version: ReleaseVersion
    task_number: int = Field(ge=1, le=RELEASE_TASK_COUNT)
    paper_id: str
    doi: str
    artifact_count: int = Field(ge=1)
    agent_package_path: str
    runtime_package_path: str
    input_manifest_url: str
    grader_package_uri: str

    @field_validator("agent_package_path", "runtime_package_path")
    @classmethod
    def package_paths_are_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("input_manifest_url", "grader_package_uri")
    @classmethod
    def public_urls_only(cls, value: str) -> str:
        return _validate_public_https_url(value)


class TaskNumberMap(RootModel[dict[int, str]]):
    """Mapping from human task numbers to stable paper IDs."""

    model_config = ConfigDict(frozen=True, strict=True)

    @model_validator(mode="after")
    def validate_complete_map(self) -> TaskNumberMap:
        expected = set(range(1, RELEASE_TASK_COUNT + 1))
        if set(self.root) != expected:
            raise ValueError(
                f"task-number map must contain every task number from 1 to {RELEASE_TASK_COUNT}"
            )
        if len(set(self.root.values())) != RELEASE_TASK_COUNT:
            raise ValueError("task-number map paper IDs must be unique")
        if EXCLUDED_PAPER_ID in self.root.values():
            raise ValueError("excluded paper cannot appear in task-number map")
        return self


class NetworkAllowEntry(StrictModel):
    """A public network allow entry with its scientific justification."""

    value: str
    reason: str | None = None


class NetworkAccessPolicy(StrictModel):
    """Case-specific policy consumed by the controlled network gateway."""

    mode: Literal["allowlist_with_tool"]
    policy_version: str
    base_allow_domains: list[str] = Field(default_factory=list)
    paper_allow_domains: list[NetworkAllowEntry] = Field(default_factory=list)
    paper_allow_url_prefixes: list[NetworkAllowEntry] = Field(default_factory=list)
    paper_allow_exact_urls: list[NetworkAllowEntry] = Field(default_factory=list)
    answer_source_urls: list[str] = Field(default_factory=list)
    answer_source_domains: list[str] = Field(default_factory=list)
    fetch_instead_of_allow_domains: list[str] = Field(default_factory=list)
    permitted_domain_scopes: list[str] = Field(default_factory=list)
    restricted_domain_scopes: list[str] = Field(default_factory=list)
    tool_enabled: bool
    adjudicator_model: str
    adjudicator_reasoning_effort: str
    decision_timeout_seconds: int = Field(gt=0)
    max_tool_calls: int = Field(ge=0)
    max_fetch_bytes: int = Field(gt=0)
    default_on_error: Literal["deny"]
    notes: str | None = None


class LicensedRuntimeTool(StrictModel):
    """A licensed tool declaration without an internal operator-cache path."""

    name: str
    version: str
    sha256: Sha256
    license_note: str | None = None
    expose_bin: str | None = None
    expose_path: str | None = None
    wrapper_name: str | None = None

    @model_validator(mode="after")
    def validate_exposure(self) -> LicensedRuntimeTool:
        if self.expose_bin is None and self.expose_path is None:
            raise ValueError("licensed runtime tools must expose a binary or path")
        return self


class RuntimeLimits(StrictModel):
    """Fixed scientific runtime limits used for comparable benchmark runs."""

    wall_clock_seconds: int = Field(default=86_400, gt=0)
    message_limit: int = Field(default=5_000, gt=0)


class RuntimeMachine(StrictModel):
    """Fixed Compute Engine shape used by the canonical public runner."""

    zone: str = "us-central1-a"
    machine_type: str = "n2-standard-32"
    boot_disk_gb: int = Field(default=500, gt=0)


class RuntimeContract(StrictModel):
    """Score-comparability settings that are invariant across public runs."""

    inspect_agent: Literal["react"] = "react"
    compaction: Literal["auto"] = "auto"
    data_mount_read_only: Literal[True] = True
    network_mode: Literal["allowlist_with_tool"] = "allowlist_with_tool"
    limits: RuntimeLimits = Field(default_factory=RuntimeLimits)
    machine: RuntimeMachine = Field(default_factory=RuntimeMachine)


class PublicRuntimeConfig(StrictModel):
    """Per-task runtime package published on Hugging Face."""

    benchmark_version: ReleaseVersion
    task_number: int = Field(ge=1, le=RELEASE_TASK_COUNT)
    paper_id: str
    doi: str
    input_manifest_url: str
    expected_outputs: list[ExpectedOutput]
    source_urls: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    network_policy: NetworkAccessPolicy
    licensed_runtime_tools: list[LicensedRuntimeTool] = Field(default_factory=list)
    runtime_contract: RuntimeContract = Field(default_factory=RuntimeContract)

    @field_validator("input_manifest_url")
    @classmethod
    def input_url_is_public(cls, value: str) -> str:
        return _validate_public_https_url(value)


class ReleaseComponents(StrictModel):
    """Immutable component identities pinned by a coordinated release."""

    git_commit: str | None = None
    hf_revision: str | None = None
    input_release_manifest_sha256: Sha256 | None = None
    ground_truth_release_manifest_sha256: Sha256 | None = None
    runtime_image_id: str | None = None
    sandbox_sha256: Sha256 | None = None


class HFDatasetRelease(StrictModel):
    """Top-level release.json for the lightweight HF dataset."""

    benchmark_name: Literal["BixBench3"] = "BixBench3"
    benchmark_version: ReleaseVersion
    status: Literal["draft", "published"] = "draft"
    hf_repo_id: str
    paper_url: str | None = None
    created_at: str
    task_count: Literal[20] = 20
    artifact_count: Literal[138] = 138
    excluded_paper_ids: list[str]
    paper_ids: list[str]
    tasks_parquet_path: str
    cases_manifest_path: str
    task_number_map_path: str
    schema_paths: list[str]
    input_release_manifest_url: str
    ground_truth_release_manifest_url: str
    components: ReleaseComponents
    files_sha256: dict[str, Sha256]

    @field_validator("created_at")
    @classmethod
    def created_at_is_iso_datetime(cls, value: str) -> str:
        return _validate_iso_datetime(value)

    @field_validator("paper_url")
    @classmethod
    def paper_url_is_public_when_present(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_public_https_url(value)

    @field_validator(
        "tasks_parquet_path",
        "cases_manifest_path",
        "task_number_map_path",
    )
    @classmethod
    def paths_are_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("schema_paths")
    @classmethod
    def schema_paths_are_relative(cls, value: list[str]) -> list[str]:
        return [_validate_relative_path(item) for item in value]

    @field_validator(
        "input_release_manifest_url",
        "ground_truth_release_manifest_url",
    )
    @classmethod
    def release_urls_are_public(cls, value: str) -> str:
        return _validate_public_https_url(value)

    @model_validator(mode="after")
    def validate_release_contract(self) -> HFDatasetRelease:
        if self.excluded_paper_ids != [EXCLUDED_PAPER_ID]:
            raise ValueError("release must explicitly contain the one excluded paper ID")
        if len(self.paper_ids) != RELEASE_TASK_COUNT:
            raise ValueError(f"release must contain exactly {RELEASE_TASK_COUNT} paper IDs")
        if len(set(self.paper_ids)) != RELEASE_TASK_COUNT:
            raise ValueError("release paper IDs must be unique")
        if EXCLUDED_PAPER_ID in self.paper_ids:
            raise ValueError("excluded paper cannot appear in the release")
        if self.status == "published":
            missing = [
                field for field, value in self.components.model_dump().items() if value is None
            ]
            if missing:
                raise ValueError(
                    "published releases require every immutable component: " + ", ".join(missing)
                )
        return self


class GCSObjectManifest(StrictModel):
    """Integrity and provenance record for one immutable GCS object."""

    object_path: str
    public_url: str
    size_bytes: int = Field(ge=0)
    sha256: Sha256
    crc32c: str = Field(min_length=1)
    source_accession_or_url: str
    upstream_repository: str
    media_type: str
    category: Literal["raw_data", "metadata", "grading_specification", "ground_truth"]

    @field_validator("object_path")
    @classmethod
    def object_path_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("public_url")
    @classmethod
    def object_url_is_public(cls, value: str) -> str:
        return _validate_public_https_url(value)


class GCSTaskManifest(StrictModel):
    """Per-task input or grader object inventory."""

    benchmark_version: ReleaseVersion
    paper_id: str
    kind: Literal["inputs", "ground_truth"]
    bucket: str
    release_prefix: str
    created_at: str
    object_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=0)
    objects: list[GCSObjectManifest]

    @field_validator("release_prefix")
    @classmethod
    def release_prefix_is_relative(cls, value: str) -> str:
        return _validate_relative_path(value)

    @field_validator("created_at")
    @classmethod
    def created_at_is_iso_datetime(cls, value: str) -> str:
        return _validate_iso_datetime(value)

    @model_validator(mode="after")
    def validate_inventory(self) -> GCSTaskManifest:
        if self.object_count != len(self.objects):
            raise ValueError("object_count must match objects")
        if self.total_size_bytes != sum(item.size_bytes for item in self.objects):
            raise ValueError("total_size_bytes must equal the object inventory")
        expected_categories = (
            {"raw_data", "metadata"}
            if self.kind == "inputs"
            else {"grading_specification", "ground_truth"}
        )
        actual_categories = {item.category for item in self.objects}
        if not actual_categories <= expected_categories:
            raise ValueError("object category does not match manifest kind")
        return self


class GCSReleaseTask(StrictModel):
    """One task entry in a bucket-level immutable release manifest."""

    task_number: int = Field(ge=1, le=RELEASE_TASK_COUNT)
    paper_id: str
    manifest_url: str
    manifest_sha256: Sha256
    object_count: int = Field(ge=1)
    total_size_bytes: int = Field(ge=0)

    @field_validator("manifest_url")
    @classmethod
    def manifest_url_is_public(cls, value: str) -> str:
        return _validate_public_https_url(value)


class GCSReleaseManifest(StrictModel):
    """Bucket-level manifest for one immutable benchmark release."""

    benchmark_version: ReleaseVersion
    kind: Literal["inputs", "ground_truth"]
    bucket: str
    created_at: str
    task_count: Literal[20] = 20
    artifact_count: Literal[138] = 138
    tasks: list[GCSReleaseTask]
    previous_release_manifest_url: str | None = None

    @field_validator("created_at")
    @classmethod
    def created_at_is_iso_datetime(cls, value: str) -> str:
        return _validate_iso_datetime(value)

    @field_validator("previous_release_manifest_url")
    @classmethod
    def previous_release_is_public(cls, value: str | None) -> str | None:
        return None if value is None else _validate_public_https_url(value)

    @model_validator(mode="after")
    def validate_tasks(self) -> GCSReleaseManifest:
        if len(self.tasks) != RELEASE_TASK_COUNT:
            raise ValueError(
                f"GCS release must contain exactly {RELEASE_TASK_COUNT} task manifests"
            )
        if {item.task_number for item in self.tasks} != set(range(1, RELEASE_TASK_COUNT + 1)):
            raise ValueError("GCS release task numbers must be contiguous")
        if len({item.paper_id for item in self.tasks}) != RELEASE_TASK_COUNT:
            raise ValueError("GCS release paper IDs must be unique")
        if EXCLUDED_PAPER_ID in {item.paper_id for item in self.tasks}:
            raise ValueError("excluded paper cannot appear in a GCS release")
        return self


class LatestReleasePointer(StrictModel):
    """Mutable convenience pointer to an immutable GCS release."""

    benchmark_version: ReleaseVersion
    release_manifest_url: str
    release_manifest_sha256: Sha256
    updated_at: str

    @field_validator("release_manifest_url")
    @classmethod
    def manifest_url_is_public(cls, value: str) -> str:
        return _validate_public_https_url(value)

    @field_validator("updated_at")
    @classmethod
    def updated_at_is_iso_datetime(cls, value: str) -> str:
        return _validate_iso_datetime(value)


class ModelRunConfiguration(StrictModel):
    """Model identity and effort resolution recorded for a run."""

    model_id: str
    model_args: dict[str, JsonValue] = Field(default_factory=dict)
    requested_effort: str
    resolved_effort: str
    reference_model: bool
    model_profile: str | None = None


class RunDataIdentity(StrictModel):
    """Exact public inputs and grader package selected for a run."""

    input_manifest_url: str
    input_manifest_sha256: Sha256
    ground_truth_manifest_url: str
    ground_truth_manifest_sha256: Sha256
    data_disk_checksum: Sha256

    @field_validator("input_manifest_url", "ground_truth_manifest_url")
    @classmethod
    def manifest_urls_are_public(cls, value: str) -> str:
        return _validate_public_https_url(value)


class RunRetention(StrictModel):
    """Whether a complete run bundle was retained by the user."""

    mode: Literal["scores_only", "full_bundle"]
    results_bucket_uri: str | None = None

    @model_validator(mode="after")
    def validate_results_bucket(self) -> RunRetention:
        if self.mode == "full_bundle":
            if self.results_bucket_uri is None or not self.results_bucket_uri.startswith("gs://"):
                raise ValueError("full_bundle retention requires a gs:// results URI")
        elif self.results_bucket_uri is not None:
            raise ValueError("scores_only retention cannot include a results bucket")
        return self


class RunTimestamps(StrictModel):
    """Lifecycle timestamps for an ephemeral benchmark run."""

    created_at: str
    started_at: str | None = None
    agent_finished_at: str | None = None
    grading_finished_at: str | None = None
    finished_at: str | None = None

    @field_validator(
        "created_at",
        "started_at",
        "agent_finished_at",
        "grading_finished_at",
        "finished_at",
    )
    @classmethod
    def timestamps_are_iso_datetimes(cls, value: str | None) -> str | None:
        return None if value is None else _validate_iso_datetime(value)


class RunManifest(StrictModel):
    """Complete reproducibility record for one public benchmark run."""

    run_id: str
    status: Literal["created", "running", "grading", "succeeded", "failed", "cancelled"]
    phase: (
        Literal[
            "provisioning",
            "agent_running",
            "grading",
            "uploading",
            "completed",
            "failed",
            "submission_failed",
            "vm_missing",
            "collecting",
            "finalized",
            "cancelled",
        ]
        | None
    ) = None
    benchmark_version: ReleaseVersion
    package_version: str
    git_commit: str
    hf_repo_id: str
    hf_revision: str
    gcs_release_revision: ReleaseVersion
    paper_id: str
    task_number: int = Field(ge=1, le=RELEASE_TASK_COUNT)
    runtime_image_id: str
    sandbox_image_digest: str
    sandbox_sha256: Sha256
    model: ModelRunConfiguration
    limits: RuntimeLimits
    machine: RuntimeMachine
    data: RunDataIdentity
    retention: RunRetention
    timestamps: RunTimestamps
    vm_name: str | None = None
    cleanup_record_path: str | None = None
    error: str | None = None
    detached_at: str | None = None

    @field_validator("cleanup_record_path")
    @classmethod
    def cleanup_path_is_relative(cls, value: str | None) -> str | None:
        return None if value is None else _validate_relative_path(value)

    @field_validator("detached_at")
    @classmethod
    def detached_at_is_iso_datetime(cls, value: str | None) -> str | None:
        return None if value is None else _validate_iso_datetime(value)

    @model_validator(mode="after")
    def validate_status(self) -> RunManifest:
        if self.status == "failed" and not (self.error or "").strip():
            raise ValueError("failed runs must record an error")
        if self.status == "succeeded" and self.timestamps.finished_at is None:
            raise ValueError("succeeded runs must record finished_at")
        return self


def model_schema_bundle() -> dict[str, dict[str, JsonValue]]:
    """Return every public JSON schema keyed by its release filename."""

    return {
        "case.schema.json": PublicCaseManifest.model_json_schema(),
        "release.schema.json": HFDatasetRelease.model_json_schema(),
        "hf-task-row.schema.json": HFTaskRow.model_json_schema(),
        "task-number-map.schema.json": TaskNumberMap.model_json_schema(),
        "gcs-task-manifest.schema.json": GCSTaskManifest.model_json_schema(),
        "gcs-release-manifest.schema.json": GCSReleaseManifest.model_json_schema(),
        "latest-release-pointer.schema.json": LatestReleasePointer.model_json_schema(),
        "run-manifest.schema.json": RunManifest.model_json_schema(),
    }
