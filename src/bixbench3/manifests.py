"""Release allowlisting, integrity helpers, and manifest construction."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from bixbench3.schemas import (
    EXCLUDED_PAPER_ID,
    RELEASE_ARTIFACT_COUNT,
    RELEASE_TASK_COUNT,
    ExpectedOutput,
    GCSObjectManifest,
    GCSReleaseManifest,
    GCSReleaseTask,
    GCSTaskManifest,
    JsonValue,
    LicensedRuntimeTool,
    NetworkAccessPolicy,
    NetworkAllowEntry,
    PublicCaseManifest,
    PublicRuntimeConfig,
    RuntimeContract,
    TaskNumberMap,
)

RELEASE_PAPER_IDS: tuple[str, ...] = (
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

INPUT_BUCKET = "bixbench3-inputs"
GROUND_TRUTH_BUCKET = "bixbench3-ground-truth"
HF_REPO_ID = "EdisonScientific/BixBench3"

_PUBLIC_INPUT_BASE = f"https://storage.googleapis.com/{INPUT_BUCKET}"
_PUBLIC_GROUND_TRUTH_BASE = f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}"

JsonObject = dict[str, JsonValue]


def utc_now() -> str:
    """Return a timezone-aware release timestamp."""

    return datetime.now(UTC).isoformat()


def load_json_object(path: Path) -> JsonObject:
    """Load a JSON object with an explicit public JSON type."""

    payload = cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def load_jsonl_objects(path: Path) -> list[JsonObject]:
    """Load non-empty JSONL records as objects."""

    records: list[JsonObject] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        payload = cast(JsonValue, json.loads(line))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        records.append(payload)
    return records


def select_release_cases(source_cases: Sequence[JsonObject]) -> list[JsonObject]:
    """Select the checked-in 20-task allowlist from the frozen source package."""

    by_id: dict[str, JsonObject] = {}
    for case in source_cases:
        paper_id = _required_string(case, "paper_id")
        if paper_id in by_id:
            raise ValueError(f"duplicate source paper_id: {paper_id}")
        by_id[paper_id] = case

    expected_source_ids = set(RELEASE_PAPER_IDS) | {EXCLUDED_PAPER_ID}
    actual_source_ids = set(by_id)
    missing = expected_source_ids - actual_source_ids
    unexpected = actual_source_ids - expected_source_ids
    if missing or unexpected:
        raise ValueError(
            "source case set does not match the frozen source package; "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    selected = [by_id[paper_id] for paper_id in RELEASE_PAPER_IDS]
    validate_release_counts(selected)
    return selected


def validate_release_counts(cases: Sequence[Mapping[str, JsonValue]]) -> None:
    """Enforce the manuscript's exact 20-task and 138-artifact contract."""

    if len(cases) != RELEASE_TASK_COUNT:
        raise ValueError(f"release must contain exactly {RELEASE_TASK_COUNT} tasks")
    paper_ids = [_required_string(case, "paper_id") for case in cases]
    if tuple(paper_ids) != RELEASE_PAPER_IDS:
        raise ValueError("release tasks must match the frozen ordered allowlist")
    if EXCLUDED_PAPER_ID in paper_ids:
        raise ValueError("excluded paper cannot be packaged")

    artifact_total = 0
    for case in cases:
        outputs = case.get("expected_outputs")
        if not isinstance(outputs, list):
            raise ValueError(f"{_required_string(case, 'paper_id')} has no expected_outputs list")
        artifact_total += len(outputs)
    if artifact_total != RELEASE_ARTIFACT_COUNT:
        raise ValueError(
            f"release must contain exactly {RELEASE_ARTIFACT_COUNT} artifacts, "
            f"found {artifact_total}"
        )


def build_task_number_map() -> TaskNumberMap:
    """Create the stable one-based task-number map."""

    return TaskNumberMap(
        root={
            task_number: paper_id for task_number, paper_id in enumerate(RELEASE_PAPER_IDS, start=1)
        }
    )


def input_manifest_url(version: str, paper_id: str) -> str:
    """Return a task's immutable public input-manifest URL."""

    _validate_release_version(version)
    _validate_release_paper_id(paper_id)
    return f"{_PUBLIC_INPUT_BASE}/releases/{version}/manifests/{paper_id}.json"


def grader_package_uri(version: str, paper_id: str) -> str:
    """Return a task's canonical public grading specification URL."""

    _validate_release_version(version)
    _validate_release_paper_id(paper_id)
    return f"{_PUBLIC_GROUND_TRUTH_BASE}/releases/{version}/tasks/{paper_id}/artifacts_grading.json"


def input_release_manifest_url(version: str) -> str:
    """Return the immutable input-bucket release manifest URL."""

    _validate_release_version(version)
    return f"{_PUBLIC_INPUT_BASE}/releases/{version}/release-public.json"


def ground_truth_release_manifest_url(version: str) -> str:
    """Return the immutable ground-truth release manifest URL."""

    _validate_release_version(version)
    return f"{_PUBLIC_GROUND_TRUTH_BASE}/releases/{version}/release.json"


def build_public_case_manifest(
    source_case: Mapping[str, JsonValue],
    *,
    task_number: int,
    version: str,
) -> PublicCaseManifest:
    """Project one internal source record onto its public case schema."""

    paper_id = _required_string(source_case, "paper_id")
    expected_outputs = _required_list(source_case, "expected_outputs")
    return PublicCaseManifest(
        benchmark_version=version,
        task_number=task_number,
        paper_id=paper_id,
        doi=_required_string(source_case, "doi"),
        artifact_count=len(expected_outputs),
        agent_package_path=f"cases/{paper_id}/agent",
        runtime_package_path=f"cases/{paper_id}/runtime/runtime.json",
        input_manifest_url=input_manifest_url(version, paper_id),
        grader_package_uri=grader_package_uri(version, paper_id),
    )


def build_public_runtime_config(
    source_case: Mapping[str, JsonValue],
    *,
    task_number: int,
    version: str,
) -> PublicRuntimeConfig:
    """Project case runtime metadata without internal projects, disks, or cache paths."""

    paper_id = _required_string(source_case, "paper_id")
    output_payloads = _required_list(source_case, "expected_outputs")
    outputs = [
        ExpectedOutput.model_validate(_json_object(item), strict=True) for item in output_payloads
    ]

    network_payload = _json_object(source_case.get("network_policy"))
    normalized_network = _normalize_network_policy(network_payload)
    network_policy = NetworkAccessPolicy.model_validate(
        normalized_network,
        strict=True,
    )

    manual_tools = _json_object(source_case.get("manual_runtime_tools"))
    operator_cached = manual_tools.get("operator_cached", [])
    if not isinstance(operator_cached, list):
        raise ValueError("manual_runtime_tools.operator_cached must be a list")
    licensed_tools = [_public_licensed_tool(_json_object(item)) for item in operator_cached]

    source_urls = _string_list(source_case.get("source_urls", []), "source_urls")
    blocked_domains = _string_list(
        source_case.get("blocked_domains", []),
        "blocked_domains",
    )
    runtime = PublicRuntimeConfig(
        benchmark_version=version,
        task_number=task_number,
        paper_id=paper_id,
        doi=_required_string(source_case, "doi"),
        input_manifest_url=input_manifest_url(version, paper_id),
        expected_outputs=outputs,
        source_urls=source_urls,
        blocked_domains=blocked_domains,
        network_policy=network_policy,
        licensed_runtime_tools=licensed_tools,
        runtime_contract=RuntimeContract(),
    )
    return runtime


def build_gcs_task_manifest(
    *,
    version: str,
    paper_id: str,
    kind: Literal["inputs", "ground_truth"],
    bucket: str,
    release_prefix: str,
    objects: Sequence[GCSObjectManifest],
    created_at: str | None = None,
) -> GCSTaskManifest:
    """Build and validate a complete per-task object inventory."""

    return GCSTaskManifest(
        benchmark_version=version,
        paper_id=paper_id,
        kind=kind,
        bucket=bucket,
        release_prefix=release_prefix,
        created_at=created_at or utc_now(),
        object_count=len(objects),
        total_size_bytes=sum(item.size_bytes for item in objects),
        objects=list(objects),
    )


def build_gcs_release_manifest(
    *,
    version: str,
    kind: Literal["inputs", "ground_truth"],
    bucket: str,
    task_manifests: Sequence[tuple[GCSTaskManifest, str, str]],
    created_at: str | None = None,
    previous_release_manifest_url: str | None = None,
) -> GCSReleaseManifest:
    """Build a bucket-level release from task manifest, URL, and SHA triples."""

    entries: list[GCSReleaseTask] = []
    for task_number, (manifest, manifest_url, manifest_sha256) in enumerate(
        task_manifests,
        start=1,
    ):
        expected_paper_id = RELEASE_PAPER_IDS[task_number - 1]
        if manifest.paper_id != expected_paper_id:
            raise ValueError("task manifests must follow the frozen release task order")
        if manifest.kind != kind or manifest.bucket != bucket:
            raise ValueError("task manifest kind and bucket must match the release")
        entries.append(
            GCSReleaseTask(
                task_number=task_number,
                paper_id=manifest.paper_id,
                manifest_url=manifest_url,
                manifest_sha256=manifest_sha256,
                object_count=manifest.object_count,
                total_size_bytes=manifest.total_size_bytes,
            )
        )
    return GCSReleaseManifest(
        benchmark_version=version,
        kind=kind,
        bucket=bucket,
        created_at=created_at or utc_now(),
        tasks=entries,
        previous_release_manifest_url=previous_release_manifest_url,
    )


def sha256_bytes(content: bytes) -> str:
    """Return a lowercase SHA-256 digest."""

    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file without changing it."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[Path]) -> dict[Path, str]:
    """Capture immutable source fingerprints for post-build verification."""

    return {path.resolve(): sha256_file(path) for path in paths}


def verify_fingerprints(fingerprints: Mapping[Path, str]) -> None:
    """Verify that every source file remained present and byte-identical."""

    changed = [
        str(path)
        for path, expected in fingerprints.items()
        if not path.is_file() or sha256_file(path) != expected
    ]
    if changed:
        raise RuntimeError("source files changed during copy-only build: " + ", ".join(changed))


def copy_file_copy_only(source: Path, destination: Path) -> None:
    """Copy one source file while refusing overwrite and source/destination overlap."""

    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if source_resolved == destination_resolved:
        raise ValueError("source and destination must differ")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256_file(source) != sha256_file(destination):
        raise OSError(f"copy verification failed for {destination}")


def write_json_copy_only(
    path: Path,
    value: JsonValue,
    *,
    compact: bool = False,
) -> None:
    """Write JSON while refusing to overwrite any existing destination."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if compact:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    else:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def write_jsonl_copy_only(path: Path, records: Iterable[JsonObject]) -> None:
    """Write deterministic JSONL without overwriting an existing destination."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def ensure_new_destination(
    destination: Path,
    source_roots: Sequence[Path],
) -> None:
    """Require a new destination outside every source tree."""

    destination_resolved = destination.resolve(strict=False)
    if destination.exists():
        raise FileExistsError(f"immutable release destination already exists: {destination}")
    for source_root in source_roots:
        source_resolved = source_root.resolve(strict=True)
        if (
            destination_resolved == source_resolved
            or destination_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(destination_resolved)
        ):
            raise ValueError(f"destination {destination} overlaps source {source_root}")


def _normalize_network_policy(payload: JsonObject) -> JsonObject:
    normalized = dict(payload)
    for key in (
        "paper_allow_domains",
        "paper_allow_url_prefixes",
        "paper_allow_exact_urls",
    ):
        entries = normalized.get(key, [])
        if not isinstance(entries, list):
            raise ValueError(f"network_policy.{key} must be a list")
        normalized[key] = [
            (
                NetworkAllowEntry(value=item).model_dump(mode="json")
                if isinstance(item, str)
                else NetworkAllowEntry.model_validate(
                    _without_key(_json_object(item), "access_mode"),
                    strict=True,
                ).model_dump(mode="json")
            )
            for item in entries
        ]
    return normalized


def _public_licensed_tool(payload: JsonObject) -> LicensedRuntimeTool:
    public_keys = {
        "name",
        "version",
        "sha256",
        "license_note",
        "expose_bin",
        "expose_path",
        "wrapper_name",
    }
    public_payload = {key: value for key, value in payload.items() if key in public_keys}
    return LicensedRuntimeTool.model_validate(public_payload, strict=True)


def _without_key(payload: JsonObject, key: str) -> JsonObject:
    return {item_key: value for item_key, value in payload.items() if item_key != key}


def _required_string(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_list(payload: Mapping[str, JsonValue], key: str) -> list[JsonValue]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    return value


def _json_object(value: JsonValue | object) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("expected a JSON object")
    return cast(JsonObject, value)


def _string_list(value: JsonValue, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a list of strings")
    return cast(list[str], value)


def _validate_release_paper_id(paper_id: str) -> None:
    if paper_id not in RELEASE_PAPER_IDS:
        raise ValueError(f"paper is not in the frozen release: {paper_id}")


def _validate_release_version(version: str) -> None:
    if re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", version) is None:
        raise ValueError("version must use vMAJOR.MINOR.PATCH form")
