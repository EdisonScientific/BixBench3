from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from bixbench3.hub import (
    HFApiProtocol,
    HubSourceLayout,
    PrivateUploadPlan,
    build_hf_release,
    upload_private_release,
)
from bixbench3.manifests import (
    RELEASE_PAPER_IDS,
    build_task_number_map,
    select_release_cases,
    sha256_file,
)
from bixbench3.schemas import (
    EXCLUDED_PAPER_ID,
    RELEASE_ARTIFACT_COUNT,
    JsonValue,
)

CONTRACTOR_PAPER_ORDER = (
    "10.1101_2025.06.17.659900_v1",
    "10.64898_2026.02.11.704850_v1",
    "10.1101_2025.08.16.670679_v1",
    "10.64898_2026.02.03.703548_v1",
    "10.1101_2025.07.20.665670_v1",
    "10.1101_2025.07.08.663208_v1",
    "10.1101_2025.07.28.666515_v1",
    "10.1038_s42003-022-03654-9_v1",
    "10_1186_s12915_024_01879_0_v1",
    "10.1038_s41467-023-44243-6_v1",
    "10.1016_j.crtox.2021.01.003_v1",
    "10.1101_2025.06.02.657493_v1",
    "10.64898_2026.03.10.709901_v1",
    "10.1101_2025.07.21.665972_v1",
    "10.64898_2026.01.09.698641_v1",
    "10.1101_2025.07.31.667834_v1",
    "10.1101_2025.07.18.664654_v1",
    "10.64898_2026.01.02.697332_v1",
    "10.64898_2026.01.31.702960_v1",
    "10.64898_2026.02.04.703711_v1",
)


TASK_SIX_PAPER_ID = CONTRACTOR_PAPER_ORDER[5]
ARC_SOURCE_VERSION = "Cell Ranger ARC v2.0.1"
ARC_RELEASE_VERSION = "Cell Ranger ARC v2.0.2"


def _expected_output(artifact_id: str) -> dict[str, JsonValue]:
    return {
        "artifact_id": artifact_id,
        "path": f"outputs/{artifact_id}.csv",
        "format": "csv",
        "row_key": "feature_id",
        "description": None,
    }


def _source_cases() -> list[dict[str, JsonValue]]:
    remaining_tasks = len(RELEASE_PAPER_IDS) - 1
    counts = [RELEASE_ARTIFACT_COUNT - remaining_tasks, *([1] * remaining_tasks)]
    cases: list[dict[str, JsonValue]] = []
    for paper_id, artifact_count in zip(RELEASE_PAPER_IDS, counts, strict=True):
        cases.append(
            {
                "paper_id": paper_id,
                "doi": f"10.1000/{paper_id}",
                "expected_outputs": [
                    _expected_output(f"artifact_{index}") for index in range(artifact_count)
                ],
            }
        )
    cases.append(
        {
            "paper_id": EXCLUDED_PAPER_ID,
            "doi": "10.1101/2025.08.14.670292",
            "expected_outputs": [_expected_output(f"excluded_{index}") for index in range(5)],
        }
    )
    return cases


def _network_policy() -> dict[str, JsonValue]:
    return {
        "mode": "allowlist_with_tool",
        "policy_version": "network-policy-v4",
        "base_allow_domains": ["pypi.org"],
        "paper_allow_domains": [],
        "paper_allow_url_prefixes": [],
        "paper_allow_exact_urls": [],
        "answer_source_urls": [],
        "answer_source_domains": [],
        "fetch_instead_of_allow_domains": [],
        "permitted_domain_scopes": [],
        "restricted_domain_scopes": [],
        "tool_enabled": True,
        "adjudicator_model": "openai/gpt-5.5",
        "adjudicator_reasoning_effort": "low",
        "decision_timeout_seconds": 30,
        "max_tool_calls": 50,
        "max_fetch_bytes": 1024,
        "default_on_error": "deny",
        "notes": None,
    }


def _write_json(path: Path, payload: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_source_tree(root: Path) -> HubSourceLayout:
    dataset_dir = root / "dataset"
    annotations_dir = root / "annotations"
    hf_rows_path = root / "hf-rows.jsonl"
    source_cases = _source_cases()
    complete_cases: list[dict[str, JsonValue]] = []
    hf_rows: list[dict[str, JsonValue]] = []

    for source_case in source_cases:
        paper_id = cast(str, source_case["paper_id"])
        if paper_id == EXCLUDED_PAPER_ID:
            complete_cases.append(source_case)
            hf_rows.append({"paper_id": paper_id})
            continue

        agent_relative = f"cases/{paper_id}/agent_visible"
        grader_relative = f"cases/{paper_id}/grader_only"
        agent_dir = dataset_dir / agent_relative
        grader_dir = dataset_dir / grader_relative
        artifact_outputs = cast(list[dict[str, JsonValue]], source_case["expected_outputs"])
        artifact_count = len(artifact_outputs)
        artifacts: list[dict[str, JsonValue]] = []
        annotations: list[dict[str, JsonValue]] = []
        for output in artifact_outputs:
            artifact_id = cast(str, output["artifact_id"])
            artifacts.append(
                {
                    "artifact_id": artifact_id,
                    "ground_truth": {
                        "path": f"artifacts/{artifact_id}.csv",
                        "format": "csv",
                    },
                    "expected_output": {
                        "path": output["path"],
                        "format": "csv",
                        "row_key": "feature_id",
                    },
                    "grading_method": {
                        "value_columns": [],
                        "checks": [{"metric": "f1", "axis": "rows"}],
                        "score_combination": "product",
                    },
                }
            )
            annotations.append(
                {
                    "artifact_id": artifact_id,
                    "depends_on": ["raw_reads"],
                    "analysis_category": "raw_data_to_data_matrix",
                    "annotation_rationale": "Derived directly from raw reads.",
                }
            )

        agent_dir.mkdir(parents=True, exist_ok=True)
        method_guidance = (
            f"Process the Multiome libraries with {ARC_SOURCE_VERSION}."
            if paper_id == TASK_SIX_PAPER_ID
            else "Use the specified method."
        )
        (agent_dir / "prompt.md").write_text(
            f"# Task\n\nAnalyze {paper_id}. {method_guidance}\n",
            encoding="utf-8",
        )
        _write_json(
            agent_dir / "task_body.json",
            {
                "research_objective": "Reconstruct the result.",
                "method_guidance": method_guidance,
                "manual_runtime_tools": {"operator_cached": []},
            },
        )
        _write_json(
            agent_dir / "inputs_metadata.json",
            {
                "paper_id": paper_id,
                "data_root": "/data",
                "expected_outputs": artifact_outputs,
            },
        )
        _write_json(
            grader_dir / "artifacts_grading.json",
            {
                "paper_id": paper_id,
                "doi": source_case["doi"],
                "artifact_count": artifact_count,
                "created_at": "2026-07-22T12:00:00+00:00",
                "artifacts": artifacts,
            },
        )
        _write_json(
            annotations_dir / paper_id / "annotations.json",
            {"paper_id": paper_id, "artifacts": annotations},
        )

        complete_case = {
            **source_case,
            "agent_visible_dir": agent_relative,
            "grader_only_dir": grader_relative,
            "source_urls": ["https://example.org/source"],
            "blocked_domains": [],
            "network_policy": _network_policy(),
            "manual_runtime_tools": {"operator_cached": []},
            "staged_data": {
                "gcp_project": "private-operator-project",
                "disk_name": f"private-data-disk-{paper_id}",
            },
        }
        complete_cases.append(complete_case)
        hf_rows.append(
            {
                "paper_id": paper_id,
                "doi": source_case["doi"],
                "artifact_count": artifact_count,
                "analysis_categories": ["raw_data_to_data_matrix"],
                "assay_types": ["bulk_rna_seq"],
                "data_types": ["bulk-transcriptomics"],
                "categories": ["transcriptomics"],
                "modalities": ["bulk"],
                "subject": "genomics",
                "input_size_gb": 1.0,
                "n_raw_sources": 1,
                "research_objective": "Reconstruct the result.",
                "method_guidance": method_guidance,
            }
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "cases.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in complete_cases),
        encoding="utf-8",
    )
    hf_rows_path.write_text(
        "".join(json.dumps(row) + "\n" for row in hf_rows),
        encoding="utf-8",
    )
    return HubSourceLayout(
        dataset_dir=dataset_dir,
        hf_rows_jsonl=hf_rows_path,
        annotations_dir=annotations_dir,
    )


def test_select_release_cases_uses_frozen_allowlist_and_artifact_count() -> None:
    source_cases = _source_cases()

    selected = select_release_cases(source_cases)

    assert tuple(cast(str, case["paper_id"]) for case in selected) == RELEASE_PAPER_IDS
    assert (
        sum(len(cast(list[JsonValue], case["expected_outputs"])) for case in selected)
        == RELEASE_ARTIFACT_COUNT
    )
    assert all(case["paper_id"] != EXCLUDED_PAPER_ID for case in selected)


def test_select_release_cases_fails_when_exclusion_is_implicit() -> None:
    source_cases = [case for case in _source_cases() if case["paper_id"] != EXCLUDED_PAPER_ID]

    with pytest.raises(ValueError, match="frozen source package"):
        select_release_cases(source_cases)


def test_task_number_map_is_stable_and_contiguous() -> None:
    mapping = build_task_number_map().root

    assert RELEASE_PAPER_IDS == CONTRACTOR_PAPER_ORDER
    assert list(mapping) == list(range(1, 21))
    assert tuple(mapping.values()) == CONTRACTOR_PAPER_ORDER


def test_build_hf_release_creates_exact_scrubbed_copy_only_package(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    pd = pytest.importorskip("pandas")
    sources = _write_source_tree(tmp_path / "sources")
    source_cases_path = sources.dataset_dir / "cases.jsonl"
    source_sha256 = sha256_file(source_cases_path)
    task_six_source_dir = sources.dataset_dir / f"cases/{TASK_SIX_PAPER_ID}/agent_visible"
    task_six_source_hashes = {
        filename: sha256_file(task_six_source_dir / filename)
        for filename in ("prompt.md", "task_body.json", "inputs_metadata.json")
    }
    output_dir = tmp_path / "release"

    result = build_hf_release(
        sources=sources,
        output_dir=output_dir,
        version="v1.0.0",
        created_at="2026-07-22T12:00:00+00:00",
    )

    tasks = pd.read_parquet(result.tasks_parquet)
    assert len(tasks.index) == 20
    assert int(tasks["artifact_count"].sum()) == RELEASE_ARTIFACT_COUNT
    assert tasks["task_number"].tolist() == list(range(1, 21))
    assert tuple(tasks["paper_id"]) == CONTRACTOR_PAPER_ORDER
    assert tasks.iloc[5]["paper_id"] == TASK_SIX_PAPER_ID
    assert EXCLUDED_PAPER_ID not in set(tasks["paper_id"])
    first_artifacts = json.loads(cast(str, tasks.iloc[0]["artifacts_json"]))
    assert len(first_artifacts) == RELEASE_ARTIFACT_COUNT - 19
    assert first_artifacts[0]["grader_package_uri"].startswith(
        "https://storage.googleapis.com/bixbench3-ground-truth/"
    )

    cases_text = (output_dir / "manifests/cases.jsonl").read_text(encoding="utf-8")
    runtime_text = (output_dir / f"cases/{RELEASE_PAPER_IDS[0]}/runtime/runtime.json").read_text(
        encoding="utf-8"
    )
    task_six_output_dir = output_dir / f"cases/{TASK_SIX_PAPER_ID}/agent"
    task_six_prompt = (task_six_output_dir / "prompt.md").read_text(encoding="utf-8")
    task_six_body = (task_six_output_dir / "task_body.json").read_text(encoding="utf-8")
    task_six_row = tasks.iloc[5]
    readme = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "private-operator-project" not in cases_text + runtime_text
    assert "private-data-disk-" not in cases_text + runtime_text
    assert ARC_SOURCE_VERSION not in task_six_prompt + task_six_body
    assert ARC_RELEASE_VERSION in task_six_prompt
    assert ARC_RELEASE_VERSION in task_six_body
    assert ARC_SOURCE_VERSION not in task_six_row["prompt"]
    assert ARC_SOURCE_VERSION not in task_six_row["method_guidance"]
    assert ARC_RELEASE_VERSION in task_six_row["prompt"]
    assert ARC_RELEASE_VERSION in task_six_row["method_guidance"]
    assert "20 tasks and 138 deterministically graded artifacts" in readme
    assert "`artifacts_json`" in readme
    assert "Benchmark paper: forthcoming." in readme
    assert result.release.paper_url is None
    assert "README.md" in result.release.files_sha256
    assert result.release.task_count == 20
    assert result.release.artifact_count == RELEASE_ARTIFACT_COUNT
    assert sha256_file(source_cases_path) == source_sha256
    assert task_six_source_hashes == {
        filename: sha256_file(task_six_source_dir / filename)
        for filename in ("prompt.md", "task_body.json", "inputs_metadata.json")
    }

    with pytest.raises(FileExistsError, match="destination already exists"):
        build_hf_release(
            sources=sources,
            output_dir=output_dir,
            version="v1.0.0",
        )


class _FakeHFApi(HFApiProtocol):
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        exist_ok: bool,
        token: str | None,
    ) -> object:
        self.calls.append(
            (
                "create_repo",
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "private": private,
                    "exist_ok": exist_ok,
                    "token": token,
                },
            )
        )
        return "created"

    def update_repo_settings(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        token: str | None,
    ) -> object:
        self.calls.append(
            (
                "update_repo_settings",
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "private": private,
                    "token": token,
                },
            )
        )
        return "private"

    def upload_folder(
        self,
        *,
        repo_id: str,
        repo_type: str,
        folder_path: str,
        path_in_repo: str,
        revision: str,
        commit_message: str,
        token: str | None,
    ) -> object:
        self.calls.append(
            (
                "upload_folder",
                {
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "folder_path": folder_path,
                    "path_in_repo": path_in_repo,
                    "revision": revision,
                    "commit_message": commit_message,
                    "token": token,
                },
            )
        )
        return "uploaded"


def test_private_upload_is_explicit_and_forces_private_visibility(
    tmp_path: Path,
) -> None:
    api = _FakeHFApi()
    plan = PrivateUploadPlan(
        local_dir=tmp_path,
        repo_id="EdisonScientific/BixBench3",
        revision="release/v1.0.0",
    )

    result = upload_private_release(plan, token="token", api=api)

    assert result == "uploaded"
    assert [call[0] for call in api.calls] == [
        "create_repo",
        "update_repo_settings",
        "upload_folder",
    ]
    assert api.calls[0][1]["private"] is True
    assert api.calls[1][1]["private"] is True
