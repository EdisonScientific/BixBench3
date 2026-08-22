"""Local Hugging Face release builder and explicit private upload path."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

from bixbench3.manifests import (
    HF_REPO_ID,
    RELEASE_PAPER_IDS,
    build_public_case_manifest,
    build_public_runtime_config,
    build_task_number_map,
    copy_file_copy_only,
    ensure_new_destination,
    fingerprint_files,
    grader_package_uri,
    ground_truth_release_manifest_url,
    input_manifest_url,
    input_release_manifest_url,
    load_json_object,
    load_jsonl_objects,
    select_release_cases,
    sha256_file,
    utc_now,
    verify_fingerprints,
    write_json_copy_only,
    write_jsonl_copy_only,
)
from bixbench3.schemas import (
    EXCLUDED_PAPER_ID,
    RELEASE_ARTIFACT_COUNT,
    RELEASE_TASK_COUNT,
    GradingCheck,
    HFArtifactDefinition,
    HFDatasetRelease,
    HFTaskRow,
    JsonValue,
    ReleaseComponents,
    model_schema_bundle,
)

JsonObject = dict[str, JsonValue]
_ARC_PATCH_TASK_NUMBER = 6
_ARC_PATCH_PAPER_ID = "10.1101_2025.07.08.663208_v1"
_ARC_SOURCE_VERSION = "Cell Ranger ARC v2.0.1"
_ARC_RELEASE_VERSION = "Cell Ranger ARC v2.0.2"


@dataclass(frozen=True)
class HubSourceLayout:
    """Operator-only source locations for an offline release build."""

    dataset_dir: Path
    hf_rows_jsonl: Path
    annotations_dir: Path


@dataclass(frozen=True)
class HubBuildResult:
    """Paths and validated release metadata returned by a local build."""

    output_dir: Path
    tasks_parquet: Path
    release_json: Path
    release: HFDatasetRelease


@dataclass(frozen=True)
class PrivateUploadPlan:
    """An explicit plan for the opt-in private Hub upload."""

    local_dir: Path
    repo_id: str
    revision: str


class HFApiProtocol(Protocol):
    """Subset of huggingface_hub.HfApi needed by private publication."""

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        exist_ok: bool,
        token: str | None,
    ) -> object: ...

    def update_repo_settings(
        self,
        repo_id: str,
        *,
        repo_type: str,
        private: bool,
        token: str | None,
    ) -> object: ...

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
    ) -> object: ...


def build_hf_release(
    *,
    sources: HubSourceLayout,
    output_dir: Path,
    version: str,
    status: Literal["draft", "published"] = "draft",
    hf_repo_id: str = HF_REPO_ID,
    paper_url: str | None = None,
    created_at: str | None = None,
    git_commit: str | None = None,
    hf_revision: str | None = None,
    input_release_manifest_sha256: str | None = None,
    ground_truth_release_manifest_sha256: str | None = None,
    runtime_image_id: str | None = None,
    sandbox_sha256: str | None = None,
) -> HubBuildResult:
    """Build an immutable local 20-task HF release without network access."""

    source_roots = [
        sources.dataset_dir,
        sources.hf_rows_jsonl,
        sources.annotations_dir,
    ]
    ensure_new_destination(output_dir, source_roots)
    _validate_source_layout(sources)

    source_cases = load_jsonl_objects(sources.dataset_dir / "cases.jsonl")
    selected_cases = select_release_cases(source_cases)
    hf_rows = _indexed_hf_rows(sources.hf_rows_jsonl)
    _validate_hf_source_ids(hf_rows)

    source_files = _source_files(sources, selected_cases)
    source_fingerprints = fingerprint_files(source_files)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.staging-",
            dir=output_dir.parent,
        )
    )

    try:
        result = _build_into_staging(
            sources=sources,
            selected_cases=selected_cases,
            hf_rows=hf_rows,
            staging_dir=staging_dir,
            final_output_dir=output_dir,
            version=version,
            status=status,
            hf_repo_id=hf_repo_id,
            paper_url=paper_url,
            created_at=created_at or utc_now(),
            components=ReleaseComponents(
                git_commit=git_commit,
                hf_revision=hf_revision,
                input_release_manifest_sha256=input_release_manifest_sha256,
                ground_truth_release_manifest_sha256=(ground_truth_release_manifest_sha256),
                runtime_image_id=runtime_image_id,
                sandbox_sha256=sandbox_sha256,
            ),
        )
        verify_fingerprints(source_fingerprints)
        staging_dir.replace(output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return HubBuildResult(
        output_dir=output_dir,
        tasks_parquet=output_dir / "data/tasks.parquet",
        release_json=output_dir / "release.json",
        release=result,
    )


def prepare_private_upload(
    local_dir: Path,
    *,
    repo_id: str = HF_REPO_ID,
    revision: str,
) -> PrivateUploadPlan:
    """Validate a local release and return a no-side-effect upload plan."""

    release_path = local_dir / "release.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"release.json not found under {local_dir}")
    release = HFDatasetRelease.model_validate_json(
        release_path.read_text(encoding="utf-8"),
        strict=True,
    )
    if release.hf_repo_id != repo_id:
        raise ValueError("upload repo does not match release.json")
    if not revision.strip():
        raise ValueError("HF revision must be non-empty")
    return PrivateUploadPlan(
        local_dir=local_dir.resolve(),
        repo_id=repo_id,
        revision=revision,
    )


def upload_private_release(
    plan: PrivateUploadPlan,
    *,
    token: str | None = None,
    api: HFApiProtocol | None = None,
) -> object:
    """Create or force-private a dataset repository, then upload one release folder."""

    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise RuntimeError(
                "private upload requires the optional huggingface_hub dependency"
            ) from exc
        api = cast(HFApiProtocol, HfApi())

    api.create_repo(
        plan.repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
        token=token,
    )
    api.update_repo_settings(
        plan.repo_id,
        repo_type="dataset",
        private=True,
        token=token,
    )
    return api.upload_folder(
        repo_id=plan.repo_id,
        repo_type="dataset",
        folder_path=str(plan.local_dir),
        path_in_repo="",
        revision=plan.revision,
        commit_message="Publish private BixBench3 release",
        token=token,
    )


def _build_into_staging(
    *,
    sources: HubSourceLayout,
    selected_cases: Sequence[JsonObject],
    hf_rows: Mapping[str, JsonObject],
    staging_dir: Path,
    final_output_dir: Path,
    version: str,
    status: Literal["draft", "published"],
    hf_repo_id: str,
    paper_url: str | None,
    created_at: str,
    components: ReleaseComponents,
) -> HFDatasetRelease:
    task_rows: list[HFTaskRow] = []
    public_cases: list[JsonObject] = []

    for task_number, source_case in enumerate(selected_cases, start=1):
        paper_id = _required_string(source_case, "paper_id")
        source_hf_row = hf_rows[paper_id]
        source_agent_dir = _source_agent_dir(sources.dataset_dir, source_case)
        destination_agent_dir = staging_dir / f"cases/{paper_id}/agent"
        source_agent_files = {
            filename: source_agent_dir / filename
            for filename in ("prompt.md", "task_body.json", "inputs_metadata.json")
        }
        for filename, source_path in source_agent_files.items():
            destination_path = destination_agent_dir / filename
            copy_file_copy_only(source_path, destination_path)
        _patch_task_six_agent_files(
            task_number=task_number,
            paper_id=paper_id,
            agent_dir=destination_agent_dir,
        )
        runtime = build_public_runtime_config(
            source_case,
            task_number=task_number,
            version=version,
        )
        runtime_path = staging_dir / f"cases/{paper_id}/runtime/runtime.json"
        write_json_copy_only(
            runtime_path,
            cast(JsonValue, runtime.model_dump(mode="json")),
        )

        artifacts = _build_artifact_definitions(
            paper_id=paper_id,
            version=version,
            source_hf_row=source_hf_row,
            grading_path=(
                _source_grader_dir(sources.dataset_dir, source_case) / "artifacts_grading.json"
            ),
            annotation_path=sources.annotations_dir / paper_id / "annotations.json",
        )
        prompt = (destination_agent_dir / "prompt.md").read_text(encoding="utf-8")
        task_row = _build_task_row(
            version=version,
            task_number=task_number,
            paper_id=paper_id,
            source_hf_row=source_hf_row,
            prompt=prompt,
            artifacts=artifacts,
        )
        task_rows.append(task_row)

        public_case = build_public_case_manifest(
            source_case,
            task_number=task_number,
            version=version,
        )
        public_cases.append(cast(JsonObject, public_case.model_dump(mode="json")))

    _validate_task_rows(task_rows)
    tasks_path = staging_dir / "data/tasks.parquet"
    _write_tasks_parquet(tasks_path, task_rows)
    write_jsonl_copy_only(staging_dir / "manifests/cases.jsonl", public_cases)
    _write_text_copy_only(
        staging_dir / "README.md",
        _dataset_card(version=version, paper_url=paper_url, rows=task_rows),
    )

    task_map = build_task_number_map()
    write_json_copy_only(
        staging_dir / "manifests/task-number-map.json",
        cast(JsonValue, task_map.model_dump(mode="json")),
    )

    schema_paths: list[str] = []
    for filename, schema in model_schema_bundle().items():
        relative_path = f"schemas/{filename}"
        write_json_copy_only(
            staging_dir / relative_path,
            cast(JsonValue, schema),
        )
        schema_paths.append(relative_path)

    files_sha256 = _release_file_hashes(staging_dir)
    release = HFDatasetRelease(
        benchmark_version=version,
        status=status,
        hf_repo_id=hf_repo_id,
        paper_url=paper_url,
        created_at=created_at,
        excluded_paper_ids=[EXCLUDED_PAPER_ID],
        paper_ids=list(RELEASE_PAPER_IDS),
        tasks_parquet_path="data/tasks.parquet",
        cases_manifest_path="manifests/cases.jsonl",
        task_number_map_path="manifests/task-number-map.json",
        schema_paths=schema_paths,
        input_release_manifest_url=input_release_manifest_url(version),
        ground_truth_release_manifest_url=ground_truth_release_manifest_url(version),
        components=components,
        files_sha256=files_sha256,
    )
    write_json_copy_only(
        staging_dir / "release.json",
        cast(JsonValue, release.model_dump(mode="json")),
    )
    _validate_staged_release(staging_dir, release)
    if final_output_dir.exists():
        raise FileExistsError(
            f"immutable release destination appeared during build: {final_output_dir}"
        )
    return release


def _patch_task_six_agent_files(
    *,
    task_number: int,
    paper_id: str,
    agent_dir: Path,
) -> None:
    is_patch_task = task_number == _ARC_PATCH_TASK_NUMBER
    if is_patch_task != (paper_id == _ARC_PATCH_PAPER_ID):
        raise ValueError("task 6 must map to the frozen MitoPerturb-Seq paper")
    if not is_patch_task:
        return

    replacement_count = 0
    for filename in ("prompt.md", "task_body.json", "inputs_metadata.json"):
        path = agent_dir / filename
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(_ARC_SOURCE_VERSION)
        if occurrences:
            path.write_text(
                text.replace(_ARC_SOURCE_VERSION, _ARC_RELEASE_VERSION),
                encoding="utf-8",
                newline="\n",
            )
            replacement_count += occurrences
    if replacement_count != 2:
        raise ValueError("task 6 must contain exactly two Cell Ranger ARC v2.0.1 references")


def _patch_task_six_hf_text(value: str, *, task_number: int) -> str:
    if task_number != _ARC_PATCH_TASK_NUMBER:
        return value
    return value.replace(_ARC_SOURCE_VERSION, _ARC_RELEASE_VERSION)


def _dataset_card(
    *,
    version: str,
    paper_url: str | None,
    rows: Sequence[HFTaskRow],
) -> str:
    paper_status = (
        f"Benchmark paper URL: [{paper_url}]({paper_url})"
        if paper_url is not None
        else "Benchmark paper: forthcoming."
    )
    lines = [
        "---",
        "license: cc-by-sa-4.0",
        "pretty_name: BixBench3",
        "configs:",
        "- config_name: default",
        "  data_files:",
        "  - split: train",
        "    path: data/tasks.parquet",
        "---",
        "",
        "# BixBench3",
        "",
        f"BixBench3 {version} evaluates whether AI agents can reproduce complete computational-biology analyses from public raw data. This release contains {RELEASE_TASK_COUNT} tasks and {RELEASE_ARTIFACT_COUNT} deterministically graded artifacts.",
        "",
        "The benchmark paper, “BixBench3: Benchmarking AI agents on research-study-scale computational biology tasks,” is the canonical source for the reported model results. This dataset contains benchmark tasks and release metadata, not the completed model runs used in the paper.",
        "",
        paper_status,
        "",
        "## Dataset contents",
        "",
        f"- `data/tasks.parquet`: one row per task, in the canonical `task_1` through `task_{RELEASE_TASK_COUNT}` order.",
        "- `manifests/cases.jsonl`: compact public case records used by the runner.",
        "- `manifests/task-number-map.json`: stable mapping from task number to paper ID.",
        "- `cases/{paper_id}/agent/`: prompt, structured task body, and agent-visible input metadata.",
        "- `cases/{paper_id}/runtime/runtime.json`: host-side runtime and network policy metadata. It contains no grader path or internal persistent-disk reference.",
        "- `schemas/`: JSON schemas for task rows, release metadata, GCS manifests, and run manifests.",
        "",
        "## `tasks.parquet` fields",
        "",
        "- Identity: `benchmark_version`, `task_number`, `paper_id`, and `doi`.",
        "- Task text: `research_objective`, `method_guidance`, and the complete rendered `prompt`.",
        "- Analytical metadata: `analysis_categories`, `assay_types`, `data_types`, `categories`, `modalities`, `max_analysis_depth`, and `subject`.",
        "- Scale: `artifact_count`, `input_size_gb`, and `n_raw_sources`.",
        "- Package locations: `input_manifest_url`, `agent_package_path`, `runtime_package_path`, and `grader_package_uri`.",
        "- `artifacts_json`: a JSON array containing each artifact ID, expected output path and format, row key, metric checks and scopes, value columns or rows, analysis category, dependency IDs, analysis depth, and canonical public grader-package URI.",
        "",
        "List-valued analytical fields are native Parquet lists. `artifacts_json` is intentionally serialized as JSON so its nested grading contract remains stable across Parquet readers.",
        "",
        "## Task construction",
        "",
        f"Each task is derived from a published computational-biology study with publicly retrievable raw reads and structured published result artifacts. The prompt supplies a biological research objective, method guidance that identifies consequential tools and parameters, and exact output contracts without including answer values. The release preserves the canonical `task_1` through `task_{RELEASE_TASK_COUNT}` ordering used for the benchmark paper.",
        "",
        "## Raw-data provenance",
        "",
        "Raw data and non-answer metadata are stored separately from this lightweight index. Every input object is inventoried by immutable release manifest with its public URL, byte size, SHA-256, GCS CRC32C, source accession or URL, upstream repository, media type, and classification as raw data or non-answer metadata.",
        "",
        "```text",
        f"gs://bixbench3-inputs/releases/{version}/tasks/{{paper_id}}/data/raw/",
        f"gs://bixbench3-inputs/releases/{version}/tasks/{{paper_id}}/data/metadata/",
        "```",
        "",
        "## Ground truth and grading",
        "",
        "Ground-truth artifacts and executable `artifacts_grading.json` files are published in a separate bucket. They are fetched by the VM host only after the agent sandbox exits and are never mounted into the sandbox.",
        "",
        "```text",
        f"gs://bixbench3-ground-truth/releases/{version}/tasks/{{paper_id}}/",
        "```",
        "",
        "Artifact checks use row or column F1, concordance correlation coefficient (CCC), categorical-label F1, or genomic-interval F1. Checks declared for one artifact are multiplied to form its score; missing or invalid artifacts receive zero. A task score is the arithmetic mean of its artifact scores.",
        "",
        "## Run a task",
        "",
        "```bash",
        "git clone https://github.com/EdisonScientific/BixBench3.git",
        "cd BixBench3",
        "uv sync",
        "gcloud auth login",
        "uv run bixbench3 run --task 1 --model anthropic/claude-opus-4-8 --effort max --project YOUR_PROJECT",
        "```",
        "",
        "The first run creates a checksummed reusable data disk in the user's GCP project. Benchmark runs use the paper runtime profile: an ephemeral `n2-standard-32` VM, a 500 GB boot disk, read-only inputs, Inspect ReAct with automatic compaction, a 24-hour wall-clock limit, and controlled network access.",
        "",
        "## Licensing boundaries",
        "",
        "Edison-authored benchmark materials are licensed under CC BY-SA 4.0. Public raw data and published artifact values retain their upstream terms; Edison provides a curated reproducibility mirror and does not relicense third-party scientific data. Cell Ranger 6.0.2 and Cell Ranger ARC 2.0.2 are not redistributed: users obtain them from 10x Genomics after reviewing its End User License Agreement and build a private runtime image in their own project.",
        "",
        "## Limitations",
        "",
        "Tasks are expensive, long-running reproductions rather than short analytical exercises, and some require licensed 10x software supplied by the user. Deterministic artifact metrics measure agreement with published outputs but do not capture every scientifically valid alternative workflow. Public ground truth enables independent grading, so score validity depends on the runner's sandbox isolation and network policy.",
        "",
        "## Citation",
        "",
        "The benchmark paper is forthcoming. Until it is available, cite the BixBench3 software release using the repository's `CITATION.cff`.",
        "",
        "## Task index",
        "",
        "| Task | Paper ID | DOI | Artifacts |",
        "| ---: | --- | --- | ---: |",
    ]
    lines.extend(
        f"| {row.task_number} | `{row.paper_id}` | [{row.doi}](https://doi.org/{row.doi}) | {row.artifact_count} |"
        for row in rows
    )
    lines.append("")
    return "\n".join(lines)


def _write_text_copy_only(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _build_artifact_definitions(
    *,
    paper_id: str,
    version: str,
    source_hf_row: Mapping[str, JsonValue],
    grading_path: Path,
    annotation_path: Path,
) -> list[HFArtifactDefinition]:
    grading_payload = load_json_object(grading_path)
    artifact_payloads = _required_list(grading_payload, "artifacts")

    annotations_payload = load_json_object(annotation_path)
    annotation_items = _required_list(annotations_payload, "artifacts")
    annotations = {
        _required_string(_json_object(item), "artifact_id"): _json_object(item)
        for item in annotation_items
    }
    depths = _artifact_depths(annotations)

    definitions: list[HFArtifactDefinition] = []
    for item in artifact_payloads:
        artifact = _json_object(item)
        artifact_id = _required_string(artifact, "artifact_id")
        annotation = annotations.get(artifact_id)
        if annotation is None:
            raise ValueError(f"{paper_id}/{artifact_id} has no artifact annotation")
        expected = _json_object(artifact.get("expected_output"))
        grading_method = _json_object(artifact.get("grading_method"))
        checks = [
            GradingCheck.model_validate(_json_object(check), strict=True)
            for check in _required_list(grading_method, "checks")
        ]
        definitions.append(
            HFArtifactDefinition(
                artifact_id=artifact_id,
                output_path=_required_string(expected, "path"),
                output_format=_required_string(expected, "format"),
                row_key=_optional_string(expected, "row_key"),
                metric_checks=checks,
                value_columns=_optional_string_list(
                    grading_method,
                    "value_columns",
                ),
                value_rows=_optional_string_list(grading_method, "value_rows"),
                analysis_category=_required_string(
                    annotation,
                    "analysis_category",
                ),
                depends_on=_optional_string_list(annotation, "depends_on"),
                analysis_depth=depths[artifact_id],
                grader_package_uri=grader_package_uri(version, paper_id),
            )
        )

    expected_count = _required_int(source_hf_row, "artifact_count")
    if len(definitions) != expected_count:
        raise ValueError(f"{paper_id} artifact count differs across source exports")
    if set(annotations) != {item.artifact_id for item in definitions}:
        raise ValueError(f"{paper_id} artifact annotations and grading IDs differ")
    return definitions


def _build_task_row(
    *,
    version: str,
    task_number: int,
    paper_id: str,
    source_hf_row: Mapping[str, JsonValue],
    prompt: str,
    artifacts: Sequence[HFArtifactDefinition],
) -> HFTaskRow:
    artifacts_json = json.dumps(
        [item.model_dump(mode="json") for item in artifacts],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return HFTaskRow(
        benchmark_version=version,
        task_number=task_number,
        paper_id=paper_id,
        doi=_required_string(source_hf_row, "doi"),
        research_objective=_patch_task_six_hf_text(
            _required_string(source_hf_row, "research_objective"),
            task_number=task_number,
        ),
        method_guidance=_patch_task_six_hf_text(
            _required_string(source_hf_row, "method_guidance"),
            task_number=task_number,
        ),
        prompt=_patch_task_six_hf_text(prompt, task_number=task_number),
        analysis_categories=_required_string_list(
            source_hf_row,
            "analysis_categories",
        ),
        assay_types=_required_string_list(source_hf_row, "assay_types"),
        data_types=_required_string_list(source_hf_row, "data_types"),
        categories=_required_string_list(source_hf_row, "categories"),
        modalities=_required_string_list(source_hf_row, "modalities"),
        max_analysis_depth=max(item.analysis_depth for item in artifacts),
        subject=_required_string(source_hf_row, "subject"),
        artifact_count=len(artifacts),
        input_size_gb=_required_float(source_hf_row, "input_size_gb"),
        n_raw_sources=_required_int(source_hf_row, "n_raw_sources"),
        input_manifest_url=input_manifest_url(version, paper_id),
        agent_package_path=f"cases/{paper_id}/agent",
        runtime_package_path=f"cases/{paper_id}/runtime/runtime.json",
        grader_package_uri=grader_package_uri(version, paper_id),
        artifacts_json=artifacts_json,
    )


def _artifact_depths(
    annotations: Mapping[str, Mapping[str, JsonValue]],
) -> dict[str, int]:
    artifact_ids = set(annotations)
    dependencies = {
        artifact_id: [
            parent
            for parent in _optional_string_list(annotation, "depends_on")
            if parent in artifact_ids
        ]
        for artifact_id, annotation in annotations.items()
    }
    depths: dict[str, int] = {}

    def resolve(artifact_id: str, visiting: frozenset[str]) -> int:
        if artifact_id in depths:
            return depths[artifact_id]
        if artifact_id in visiting:
            raise ValueError(f"artifact dependency cycle includes {artifact_id}")
        parents = dependencies[artifact_id]
        depth = (
            1
            if not parents
            else 1 + max(resolve(parent, visiting | {artifact_id}) for parent in parents)
        )
        depths[artifact_id] = depth
        return depth

    for artifact_id in artifact_ids:
        resolve(artifact_id, frozenset())
    return depths


def _write_tasks_parquet(path: Path, rows: Sequence[HFTaskRow]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("building tasks.parquet requires pandas and pyarrow") from exc
    records = [row.model_dump(mode="json") for row in rows]
    pd.DataFrame.from_records(records).to_parquet(path, index=False)
    check = pd.read_parquet(path)
    if len(check.index) != RELEASE_TASK_COUNT:
        raise RuntimeError("tasks.parquet row count verification failed")
    if int(check["artifact_count"].sum()) != RELEASE_ARTIFACT_COUNT:
        raise RuntimeError("tasks.parquet artifact count verification failed")


def _validate_task_rows(rows: Sequence[HFTaskRow]) -> None:
    if len(rows) != RELEASE_TASK_COUNT:
        raise ValueError(f"HF task table must contain exactly {RELEASE_TASK_COUNT} rows")
    if tuple(row.paper_id for row in rows) != RELEASE_PAPER_IDS:
        raise ValueError("HF task rows must follow the frozen release order")
    if [row.task_number for row in rows] != list(range(1, RELEASE_TASK_COUNT + 1)):
        raise ValueError("HF task numbers must be contiguous")
    if sum(row.artifact_count for row in rows) != RELEASE_ARTIFACT_COUNT:
        raise ValueError(
            f"HF task table must describe exactly {RELEASE_ARTIFACT_COUNT} artifacts"
        )


def _validate_source_layout(sources: HubSourceLayout) -> None:
    required = (
        sources.dataset_dir / "cases.jsonl",
        sources.hf_rows_jsonl,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if not sources.annotations_dir.is_dir():
        missing.append(str(sources.annotations_dir))
    if missing:
        raise FileNotFoundError("missing release sources: " + ", ".join(missing))


def _indexed_hf_rows(path: Path) -> dict[str, JsonObject]:
    rows = load_jsonl_objects(path)
    indexed: dict[str, JsonObject] = {}
    for row in rows:
        paper_id = _required_string(row, "paper_id")
        if paper_id in indexed:
            raise ValueError(f"duplicate HF source row: {paper_id}")
        indexed[paper_id] = row
    return indexed


def _validate_hf_source_ids(rows: Mapping[str, JsonObject]) -> None:
    expected = set(RELEASE_PAPER_IDS) | {EXCLUDED_PAPER_ID}
    if set(rows) != expected:
        raise ValueError("HF source rows must match the exact 28-case source package")


def _source_files(
    sources: HubSourceLayout,
    selected_cases: Sequence[JsonObject],
) -> list[Path]:
    files = [
        sources.dataset_dir / "cases.jsonl",
        sources.hf_rows_jsonl,
    ]
    for source_case in selected_cases:
        paper_id = _required_string(source_case, "paper_id")
        agent_dir = _source_agent_dir(sources.dataset_dir, source_case)
        files.extend(
            agent_dir / filename
            for filename in ("prompt.md", "task_body.json", "inputs_metadata.json")
        )
        grader_dir = _source_grader_dir(sources.dataset_dir, source_case)
        files.append(grader_dir / "artifacts_grading.json")
        files.append(sources.annotations_dir / paper_id / "annotations.json")
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing source files: " + ", ".join(missing))
    return files


def _source_agent_dir(dataset_dir: Path, source_case: Mapping[str, JsonValue]) -> Path:
    return _resolve_source_relative(
        dataset_dir,
        _required_string(source_case, "agent_visible_dir"),
    )


def _source_grader_dir(dataset_dir: Path, source_case: Mapping[str, JsonValue]) -> Path:
    return _resolve_source_relative(
        dataset_dir,
        _required_string(source_case, "grader_only_dir"),
    )


def _resolve_source_relative(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    root_resolved = root.resolve(strict=True)
    if not candidate.is_relative_to(root_resolved):
        raise ValueError(f"source path escapes dataset root: {value}")
    return candidate


def _validate_staged_release(
    staging_dir: Path,
    release: HFDatasetRelease,
) -> None:
    loaded = HFDatasetRelease.model_validate_json(
        (staging_dir / "release.json").read_text(encoding="utf-8"),
        strict=True,
    )
    if loaded != release:
        raise RuntimeError("release.json round-trip validation failed")
    for relative_path, expected_sha256 in release.files_sha256.items():
        path = staging_dir / relative_path
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"release file hash verification failed: {relative_path}")


def _release_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _required_string(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(
    payload: Mapping[str, JsonValue],
    key: str,
) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def _required_int(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _required_float(payload: Mapping[str, JsonValue], key: str) -> float:
    value = payload.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _required_list(
    payload: Mapping[str, JsonValue],
    key: str,
) -> list[JsonValue]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _required_string_list(
    payload: Mapping[str, JsonValue],
    key: str,
) -> list[str]:
    return _string_list(_required_list(payload, key), key)


def _optional_string_list(
    payload: Mapping[str, JsonValue],
    key: str,
) -> list[str]:
    value = payload.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return _string_list(value, key)


def _string_list(value: Sequence[JsonValue], key: str) -> list[str]:
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{key} must contain only strings")
    return cast(list[str], value)


def _json_object(value: JsonValue) -> JsonObject:
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value
