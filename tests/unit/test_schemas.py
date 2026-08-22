from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from bixbench3.schemas import (
    EXCLUDED_PAPER_ID,
    RELEASE_TASK_COUNT,
    GCSObjectManifest,
    GCSTaskManifest,
    GradingCheck,
    HFArtifactDefinition,
    HFDatasetRelease,
    HFTaskRow,
    ModelRunConfiguration,
    ReleaseComponents,
    RunDataIdentity,
    RunManifest,
    RunRetention,
    RuntimeLimits,
    RuntimeMachine,
    RunTimestamps,
    TaskNumberMap,
)

SHA256 = "a" * 64


def _artifact() -> HFArtifactDefinition:
    return HFArtifactDefinition(
        artifact_id="counts",
        output_path="outputs/counts.csv",
        output_format="csv",
        row_key="gene_id",
        metric_checks=[GradingCheck(metric="f1", axis="rows")],
        value_columns=[],
        value_rows=[],
        analysis_category="raw_data_to_data_matrix",
        depends_on=["raw_reads"],
        analysis_depth=1,
        grader_package_uri=(
            "https://storage.googleapis.com/bixbench3-ground-truth/"
            "releases/v1.0.0/tasks/paper/artifacts_grading.json"
        ),
    )


def _task_row() -> HFTaskRow:
    artifact = _artifact()
    return HFTaskRow(
        benchmark_version="v1.0.0",
        task_number=1,
        paper_id="paper",
        doi="10.1000/example",
        research_objective="Reconstruct the result.",
        method_guidance="Use the specified method.",
        prompt="Run the analysis.",
        analysis_categories=["raw_data_to_data_matrix"],
        assay_types=["bulk_rna_seq"],
        data_types=["bulk-transcriptomics"],
        categories=["transcriptomics"],
        modalities=["bulk"],
        max_analysis_depth=1,
        subject="genomics",
        artifact_count=1,
        input_size_gb=1.0,
        n_raw_sources=1,
        input_manifest_url=(
            "https://storage.googleapis.com/bixbench3-inputs/"
            "releases/v1.0.0/manifests/paper.json"
        ),
        agent_package_path="cases/paper/agent",
        runtime_package_path="cases/paper/runtime/runtime.json",
        grader_package_uri=artifact.grader_package_uri,
        artifacts_json=json.dumps([artifact.model_dump(mode="json")]),
    )


def test_hf_task_row_validates_nested_artifacts_json() -> None:
    row = _task_row()

    assert row.artifacts() == [_artifact()]

    invalid = row.model_copy(update={"artifact_count": 2})
    with pytest.raises(ValidationError, match="artifact_count"):
        HFTaskRow.model_validate(invalid.model_dump(), strict=True)


@pytest.mark.parametrize(
    ("metric", "axis"),
    (
        ("f1", "rows"),
        ("f1", "columns"),
        ("ccc", "rows"),
        ("ccc", "columns"),
        ("label_f1", None),
        ("interval_f1", None),
    ),
)
def test_grading_check_supports_every_release_metric_axis(
    metric: str,
    axis: str | None,
) -> None:
    check = GradingCheck.model_validate(
        {"metric": metric, "axis": axis, "scope": {}},
        strict=True,
    )

    assert check.axis == axis


def test_grading_check_rejects_axes_not_used_by_metric() -> None:
    with pytest.raises(ValidationError, match="require an axis"):
        GradingCheck.model_validate({"metric": "f1"}, strict=True)

    with pytest.raises(ValidationError, match="do not use an axis"):
        GradingCheck.model_validate(
            {"metric": "interval_f1", "axis": "rows"},
            strict=True,
        )


def test_hf_artifact_rejects_absolute_output_paths() -> None:
    with pytest.raises(ValidationError, match="relative"):
        HFArtifactDefinition.model_validate(
            _artifact().model_copy(update={"output_path": "/outputs/counts.csv"}).model_dump(),
            strict=True,
        )


def test_task_number_map_requires_complete_release_and_excludes_paper() -> None:
    mapping = {number: f"paper-{number}" for number in range(1, RELEASE_TASK_COUNT + 1)}
    assert TaskNumberMap(root=mapping).root == mapping

    mapping[RELEASE_TASK_COUNT] = EXCLUDED_PAPER_ID
    with pytest.raises(ValidationError, match="excluded"):
        TaskNumberMap(root=mapping)


def test_hf_release_requires_published_component_pins() -> None:
    payload = {
        "benchmark_version": "v1.0.0",
        "status": "published",
        "hf_repo_id": "EdisonScientific/BixBench3",
        "paper_url": "https://example.org/bixbench3-paper",
        "created_at": "2026-07-22T12:00:00+00:00",
        "excluded_paper_ids": [EXCLUDED_PAPER_ID],
        "paper_ids": [f"paper-{number}" for number in range(1, RELEASE_TASK_COUNT + 1)],
        "tasks_parquet_path": "data/tasks.parquet",
        "cases_manifest_path": "manifests/cases.jsonl",
        "task_number_map_path": "manifests/task-number-map.json",
        "schema_paths": ["schemas/release.schema.json"],
        "input_release_manifest_url": (
            "https://storage.googleapis.com/bixbench3-inputs/"
            "releases/v1.0.0/release-public.json"
        ),
        "ground_truth_release_manifest_url": (
            "https://storage.googleapis.com/bixbench3-ground-truth/"
            "releases/v1.0.0/release.json"
        ),
        "components": ReleaseComponents().model_dump(),
        "files_sha256": {},
    }

    with pytest.raises(ValidationError, match="published releases require"):
        HFDatasetRelease.model_validate(payload, strict=True)


def test_hf_release_allows_forthcoming_paper_when_published() -> None:
    payload = {
        "benchmark_version": "v1.0.0",
        "status": "published",
        "hf_repo_id": "EdisonScientific/BixBench3",
        "paper_url": None,
        "created_at": "2026-07-22T12:00:00+00:00",
        "excluded_paper_ids": [EXCLUDED_PAPER_ID],
        "paper_ids": [f"paper-{number}" for number in range(1, RELEASE_TASK_COUNT + 1)],
        "tasks_parquet_path": "data/tasks.parquet",
        "cases_manifest_path": "manifests/cases.jsonl",
        "task_number_map_path": "manifests/task-number-map.json",
        "schema_paths": ["schemas/release.schema.json"],
        "input_release_manifest_url": (
            "https://storage.googleapis.com/bixbench3-inputs/"
            "releases/v1.0.0/release-public.json"
        ),
        "ground_truth_release_manifest_url": (
            "https://storage.googleapis.com/bixbench3-ground-truth/"
            "releases/v1.0.0/release.json"
        ),
        "components": ReleaseComponents(
            git_commit="abc123",
            hf_revision="release/v1.0.0",
            input_release_manifest_sha256=SHA256,
            ground_truth_release_manifest_sha256=SHA256,
            runtime_image_id="runtime-image",
            sandbox_sha256=SHA256,
        ).model_dump(),
        "files_sha256": {},
    }

    release = HFDatasetRelease.model_validate(payload, strict=True)
    assert release.paper_url is None


def test_gcs_task_manifest_checks_inventory_totals_and_categories() -> None:
    object_record = GCSObjectManifest(
        object_path="releases/v1.0.0/tasks/paper/data/raw/reads.fastq.gz",
        public_url=(
            "https://storage.googleapis.com/bixbench3-inputs/"
            "releases/v1.0.0/tasks/paper/data/raw/reads.fastq.gz"
        ),
        size_bytes=100,
        sha256=SHA256,
        crc32c="AAAAAA==",
        source_accession_or_url="SRR000001",
        upstream_repository="NCBI SRA",
        media_type="application/gzip",
        category="raw_data",
    )
    manifest = GCSTaskManifest(
        benchmark_version="v1.0.0",
        paper_id="paper",
        kind="inputs",
        bucket="bixbench3-inputs",
        release_prefix="releases/v1.0.0",
        created_at="2026-07-22T12:00:00+00:00",
        object_count=1,
        total_size_bytes=100,
        objects=[object_record],
    )

    assert manifest.total_size_bytes == 100
    with pytest.raises(ValidationError, match="total_size_bytes"):
        GCSTaskManifest.model_validate(
            manifest.model_copy(update={"total_size_bytes": 99}).model_dump(),
            strict=True,
        )


def test_run_manifest_records_reproducibility_and_retention() -> None:
    manifest = RunManifest(
        run_id="run-1",
        status="succeeded",
        benchmark_version="v1.0.0",
        package_version="1.0.0",
        git_commit="abc123",
        hf_repo_id="EdisonScientific/BixBench3",
        hf_revision="release/v1.0.0",
        gcs_release_revision="v1.0.0",
        paper_id="paper",
        task_number=1,
        runtime_image_id="projects/example/global/machineImages/runtime",
        sandbox_image_digest="sha256:digest",
        sandbox_sha256=SHA256,
        model=ModelRunConfiguration(
            model_id="openai/gpt-5.5",
            requested_effort="max",
            resolved_effort="xhigh",
            reference_model=True,
            model_profile="openai-gpt-5.5",
        ),
        limits=RuntimeLimits(),
        machine=RuntimeMachine(),
        data=RunDataIdentity(
            input_manifest_url=(
                "https://storage.googleapis.com/bixbench3-inputs/"
                "releases/v1.0.0/manifests/paper.json"
            ),
            input_manifest_sha256=SHA256,
            ground_truth_manifest_url=(
                "https://storage.googleapis.com/bixbench3-ground-truth/"
                "releases/v1.0.0/manifests/paper.json"
            ),
            ground_truth_manifest_sha256=SHA256,
            data_disk_checksum=SHA256,
        ),
        retention=RunRetention(mode="scores_only"),
        timestamps=RunTimestamps(
            created_at="2026-07-22T12:00:00+00:00",
            finished_at="2026-07-22T13:00:00+00:00",
        ),
        cleanup_record_path="cleanup.json",
    )

    assert manifest.model.resolved_effort == "xhigh"
    with pytest.raises(ValidationError, match="results URI"):
        RunRetention(mode="full_bundle")
    cancelled = RunManifest.model_validate(
        manifest.model_copy(
            update={
                "status": "cancelled",
                "phase": "cancelled",
                "detached_at": "2026-07-22T12:01:00+00:00",
            }
        ).model_dump(),
        strict=True,
    )
    assert cancelled.status == "cancelled"
    with pytest.raises(ValidationError, match="status"):
        RunManifest.model_validate(
            manifest.model_copy(update={"status": "completed"}).model_dump(),
            strict=True,
        )
