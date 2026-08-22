from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NotRequired, TypedDict

import pandas as pd

from .loaders import load_table
from .metrics import (
    MetricResult,
    ccc_score,
    f1_score,
    interval_f1_score,
    label_f1_score,
    spearman_score,
)

SELF_GRADE_TOLERANCE = 1e-9


class ErrorDetails(TypedDict):
    reason: str
    error_type: str
    message: str
    agent_artifact_path: NotRequired[str]


class CheckGrade(TypedDict):
    metric: str
    score: float
    details: dict[str, object]
    axis: NotRequired[str]


class ArtifactGrade(TypedDict):
    artifact_id: str
    score: float
    score_combination: str
    checks: list[CheckGrade]
    error: NotRequired[ErrorDetails]


class PaperGrade(TypedDict):
    paper_id: str | None
    doi: str | None
    score: float
    artifact_count: int
    artifacts: list[ArtifactGrade]


def grade(
    grading_spec_path: str | Path,
    artifacts_dir: str | Path,
    artifact_grades_dir: str | Path | None = None,
    *,
    ground_truth_root: str | Path | None = None,
) -> PaperGrade:
    spec_path = Path(grading_spec_path)
    artifacts_path = Path(artifacts_dir)
    output_path = (
        Path(artifact_grades_dir)
        if artifact_grades_dir is not None
        else artifacts_path.parent / "artifact_grades"
    )
    truth_root = Path(ground_truth_root) if ground_truth_root is not None else spec_path.parent
    with spec_path.open(encoding="utf-8") as handle:
        raw_spec: object = json.load(handle)
    if not isinstance(raw_spec, Mapping):
        raise ValueError("grading specification must be a JSON object")
    spec = _string_key_mapping(raw_spec, "grading specification")
    return grade_paper(
        spec,
        agent_artifacts_dir=artifacts_path,
        ground_truth_root=truth_root,
        artifact_grades_dir=output_path,
    )


def grade_artifact(
    spec_artifact: Mapping[str, object],
    agent_artifact_path: Path,
    *,
    repo_root: Path | None = None,
    ground_truth_root: Path | None = None,
) -> ArtifactGrade:
    normalized = normalize_grading_artifact(spec_artifact)
    repository_root = repo_root or Path.cwd()
    truth_root = ground_truth_root or repository_root
    ground_truth_spec = _required_mapping(
        normalized,
        "ground_truth",
        "grading artifact",
    )
    output_format = _required_mapping(
        normalized,
        "output_format",
        "grading artifact",
    )
    ground_truth_path = truth_root / _required_string(
        ground_truth_spec,
        "path",
        "ground_truth",
    )
    ground_truth = load_table(
        ground_truth_path,
        _optional_string(ground_truth_spec.get("format"), "ground_truth.format"),
    )
    agent_output = load_table(
        agent_artifact_path,
        _optional_string(output_format.get("format"), "output_format.format"),
    )
    return _grade_loaded_artifact(normalized, ground_truth, agent_output)


def grade_paper(
    spec: Mapping[str, object],
    run_effort_dir: Path | None = None,
    *,
    repo_root: Path | None = None,
    agent_artifacts_dir: Path | None = None,
    ground_truth_root: Path | None = None,
    artifact_grades_dir: Path | None = None,
) -> PaperGrade:
    normalized = normalize_grading_spec(spec)
    repository_root = repo_root or Path.cwd()
    if run_effort_dir is not None:
        artifact_grades_dir = artifact_grades_dir or run_effort_dir / "artifact_grades"
        agent_artifacts_dir = agent_artifacts_dir or run_effort_dir / "agent_artifacts"
    if agent_artifacts_dir is None:
        raise ValueError(
            "agent_artifacts_dir is required when run_effort_dir is not provided",
        )
    if artifact_grades_dir is None:
        raise ValueError(
            "artifact_grades_dir is required when run_effort_dir is not provided"
        )
    artifact_grades_dir.mkdir(parents=True, exist_ok=True)

    paper_id = _optional_string(normalized.get("paper_id"), "paper_id")
    artifact_results: list[ArtifactGrade] = []
    for artifact in _artifact_specs(normalized):
        output_format = _required_mapping(
            artifact,
            "output_format",
            "grading artifact",
        )
        output_path = Path(
            _required_string(output_format, "path", "output_format"),
        )
        agent_artifact_path = _resolve_agent_artifact_path(
            agent_artifacts_dir,
            output_path,
            paper_id or "",
        )
        if not agent_artifact_path.exists():
            result = _zero_artifact_result(
                artifact,
                FileNotFoundError(agent_artifact_path),
                agent_artifact_path=agent_artifact_path,
                reason="missing_agent_artifact",
            )
        else:
            try:
                result = grade_artifact(
                    artifact,
                    agent_artifact_path,
                    repo_root=repository_root,
                    ground_truth_root=ground_truth_root,
                )
            except Exception as error:
                result = _zero_artifact_result(
                    artifact,
                    error,
                    agent_artifact_path=agent_artifact_path,
                )
        artifact_results.append(result)
        _write_json(artifact_grades_dir / f"{result['artifact_id']}.json", result)

    scores = [result["score"] for result in artifact_results]
    paper_grade: PaperGrade = {
        "paper_id": paper_id,
        "doi": _optional_string(normalized.get("doi"), "doi"),
        "score": sum(scores) / len(scores) if scores else 0.0,
        "artifact_count": len(artifact_results),
        "artifacts": artifact_results,
    }
    _write_json(artifact_grades_dir.parent / "grade.json", paper_grade)
    return paper_grade


def self_grade_artifact(
    spec_artifact: Mapping[str, object],
    *,
    repo_root: Path | None = None,
    ground_truth_root: Path | None = None,
) -> ArtifactGrade:
    normalized = normalize_grading_artifact(spec_artifact)
    repository_root = repo_root or Path.cwd()
    truth_root = ground_truth_root or repository_root
    ground_truth_spec = _required_mapping(
        normalized,
        "ground_truth",
        "grading artifact",
    )
    ground_truth_path = truth_root / _required_string(
        ground_truth_spec,
        "path",
        "ground_truth",
    )
    ground_truth = load_table(
        ground_truth_path,
        _optional_string(ground_truth_spec.get("format"), "ground_truth.format"),
    )
    return _grade_loaded_artifact(normalized, ground_truth, ground_truth.copy())


def self_grade_paper(
    spec: Mapping[str, object],
    *,
    repo_root: Path | None = None,
    ground_truth_root: Path | None = None,
    tolerance: float = SELF_GRADE_TOLERANCE,
) -> list[str]:
    normalized = normalize_grading_spec(spec)
    errors: list[str] = []
    for index, artifact in enumerate(_artifact_specs(normalized)):
        try:
            result = self_grade_artifact(
                artifact,
                repo_root=repo_root,
                ground_truth_root=ground_truth_root,
            )
        except Exception as error:
            errors.append(
                f"artifacts[{index}] self-grade crashed for "
                f"{artifact.get('artifact_id', '<unknown>')}: {error}",
            )
            continue
        for check in result["checks"]:
            score = check["score"]
            if abs(score - 1.0) <= tolerance:
                continue
            axis = check.get("axis")
            errors.append(
                f"artifacts[{index}] self-grade failed for "
                f"{artifact.get('artifact_id', '<unknown>')} check "
                f"{check['metric']}{' ' + axis if axis else ''}: expected 1.0, "
                f"got {score:.6g}; check row_key/value_columns/value_rows/scope "
                f"alignment. Details: {check['details']}",
            )
    return errors


def normalize_grading_spec(spec: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(spec)
    raw_artifacts = normalized.get("artifacts", [])
    if not isinstance(raw_artifacts, Sequence) or isinstance(
        raw_artifacts,
        (str, bytes),
    ):
        raise ValueError("grading specification artifacts must be a list")
    artifacts = [
        normalize_grading_artifact(
            _string_key_mapping(artifact, "grading artifact"),
        )
        for artifact in raw_artifacts
        if isinstance(artifact, Mapping)
    ]
    normalized["artifacts"] = artifacts
    normalized["artifact_count"] = len(artifacts)
    return normalized


def normalize_grading_artifact(
    artifact: Mapping[str, object],
) -> dict[str, object]:
    normalized = dict(artifact)
    if "output_format" not in normalized and "expected_output" in normalized:
        expected_output = normalized["expected_output"]
        normalized["output_format"] = dict(
            _mapping_value(expected_output, "expected_output"),
        )
    return normalized


def _resolve_agent_artifact_path(
    agent_artifacts_dir: Path,
    output_path: Path,
    paper_id: str,
) -> Path:
    candidates = [
        agent_artifacts_dir / output_path.name,
        agent_artifacts_dir / output_path,
    ]
    if paper_id:
        candidates.extend(
            [
                agent_artifacts_dir / paper_id / output_path.name,
                agent_artifacts_dir / paper_id / output_path,
            ],
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _zero_artifact_result(
    spec_artifact: Mapping[str, object],
    error: Exception,
    *,
    agent_artifact_path: Path | None = None,
    reason: str | None = None,
) -> ArtifactGrade:
    resolved_reason = reason or _artifact_error_reason(error)
    error_details: ErrorDetails = {
        "reason": resolved_reason,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    if agent_artifact_path is not None:
        error_details["agent_artifact_path"] = str(agent_artifact_path)

    return {
        "artifact_id": _required_string(
            spec_artifact,
            "artifact_id",
            "grading artifact",
        ),
        "score": 0.0,
        "score_combination": _score_combination(spec_artifact),
        "checks": [
            {
                "metric": "artifact_error",
                "score": 0.0,
                "details": dict(error_details),
            },
        ],
        "error": error_details,
    }


def _artifact_error_reason(error: Exception) -> str:
    if isinstance(error, FileNotFoundError):
        return "missing_artifact"
    if isinstance(error, pd.errors.EmptyDataError):
        return "empty_agent_artifact"
    if isinstance(error, pd.errors.ParserError):
        return "unparseable_artifact"
    return "artifact_grading_error"


def _grade_loaded_artifact(
    spec_artifact: Mapping[str, object],
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
) -> ArtifactGrade:
    output_format = _required_mapping(
        spec_artifact,
        "output_format",
        "grading artifact",
    )
    row_key = _required_string(output_format, "row_key", "output_format")
    grading_method = _required_mapping(
        spec_artifact,
        "grading_method",
        "grading artifact",
    )
    raw_checks = grading_method.get("checks")
    if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
        raise ValueError("grading_method.checks must be a list")

    checks: list[CheckGrade] = []
    artifact_score = 1.0
    for raw_check in raw_checks:
        check = _string_key_mapping(raw_check, "grading check")
        check_result = _run_check(
            check,
            grading_method,
            ground_truth,
            agent_output,
            row_key,
        )
        checks.append(check_result)
        artifact_score *= check_result["score"]

    return {
        "artifact_id": _required_string(
            spec_artifact,
            "artifact_id",
            "grading artifact",
        ),
        "score": artifact_score,
        "score_combination": _score_combination(spec_artifact),
        "checks": checks,
    }


def _run_check(
    check: Mapping[str, object],
    grading_method: Mapping[str, object],
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    row_key: str,
) -> CheckGrade:
    metric = _required_string(check, "metric", "grading check")
    raw_scope = check.get("scope")
    scope = {} if raw_scope is None else _string_key_mapping(raw_scope, "grading check scope")
    row_filter = _optional_string(scope.get("row_filter"), "scope.row_filter")
    filtered_gt = _apply_row_filter(ground_truth, row_filter)
    filtered_agent = _apply_row_filter(agent_output, row_filter)

    if metric == "f1":
        axis = _required_string(check, "axis", "f1 check")
        if axis == "rows":
            expected = _row_values(filtered_gt, row_key, grading_method, scope)
            observed = _row_values(filtered_agent, row_key, grading_method, scope)
        elif axis == "columns":
            expected = _column_values(filtered_gt, row_key, grading_method, scope)
            observed = _observed_column_values(
                filtered_agent,
                row_key,
                grading_method,
                scope,
            )
        else:
            raise ValueError(f"unsupported f1 axis: {axis}")
        return _check_grade(metric, f1_score(expected, observed), axis=axis)

    if metric in {"ccc", "spearman"}:
        axis = _required_string(check, "axis", f"{metric} check")
        score_fn = ccc_score if metric == "ccc" else spearman_score
        result = score_fn(
            filtered_gt,
            filtered_agent,
            row_key=row_key,
            axis=axis,
            value_columns=_resolve_axis_values(grading_method, scope, "columns"),
            value_rows=_resolve_axis_values(grading_method, scope, "rows"),
            value_transform=_optional_string(
                scope.get("value_transform"),
                "scope.value_transform",
            ),
        )
        return _check_grade(metric, result, axis=axis)

    if metric == "label_f1":
        label_column = _required_string(scope, "label_column", "label_f1 scope")
        result = label_f1_score(
            filtered_gt,
            filtered_agent,
            row_key=row_key,
            label_column=label_column,
        )
        return _check_grade(metric, result)

    if metric == "interval_f1":
        interval_columns = _string_mapping(
            scope.get("interval_columns", {}),
            "scope.interval_columns",
        )
        overlap_rule = _string_key_mapping(
            scope.get("overlap_rule"),
            "scope.overlap_rule",
        )
        result = interval_f1_score(
            filtered_gt,
            filtered_agent,
            interval_columns=interval_columns,
            overlap_rule=overlap_rule,
        )
        return _check_grade(metric, result)

    raise ValueError(f"unsupported metric: {metric}")


def _check_grade(
    metric: str,
    result: MetricResult,
    *,
    axis: str | None = None,
) -> CheckGrade:
    check: CheckGrade = {
        "metric": metric,
        "score": result["score"],
        "details": result["details"],
    }
    if axis is not None:
        check["axis"] = axis
    return check


def _row_values(
    frame: pd.DataFrame,
    row_key: str,
    grading_method: Mapping[str, object],
    scope: Mapping[str, object],
) -> list[str]:
    value_rows = _resolve_axis_values(grading_method, scope, "rows")
    if value_rows is not None:
        present_rows = set(frame[row_key].astype(str))
        return [row for row in value_rows if row in present_rows]
    return frame[row_key].astype(str).tolist()


def _column_values(
    frame: pd.DataFrame,
    row_key: str,
    grading_method: Mapping[str, object],
    scope: Mapping[str, object],
) -> list[str]:
    values = _resolve_axis_values(grading_method, scope, "columns")
    if values is not None:
        return [column for column in values if column in frame.columns]
    return [str(column) for column in frame.columns if column != row_key]


def _observed_column_values(
    frame: pd.DataFrame,
    row_key: str,
    grading_method: Mapping[str, object],
    scope: Mapping[str, object],
) -> list[str]:
    expected_values = _resolve_axis_values(grading_method, scope, "columns")
    if expected_values is None:
        return [str(column) for column in frame.columns if column != row_key]

    expected_set = set(expected_values)
    numeric_columns: list[str] = []
    for raw_column in frame.columns:
        column = str(raw_column)
        if column == row_key:
            continue
        numeric = pd.to_numeric(frame[raw_column], errors="coerce")
        if column in expected_set or numeric.notna().any():
            numeric_columns.append(column)
    return numeric_columns


def _resolve_axis_values(
    grading_method: Mapping[str, object],
    scope: Mapping[str, object],
    axis_key: str,
) -> list[str] | None:
    value = scope.get(axis_key)
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value in {"value_columns", "value_rows"}:
        return _string_list(grading_method.get(value, []), f"grading_method.{value}")
    if axis_key == "columns" and "value_columns" in grading_method:
        return _string_list(
            grading_method["value_columns"],
            "grading_method.value_columns",
        )
    if axis_key == "rows" and "value_rows" in grading_method:
        return _string_list(
            grading_method["value_rows"],
            "grading_method.value_rows",
        )
    return None


def _apply_row_filter(
    frame: pd.DataFrame,
    row_filter: str | None,
) -> pd.DataFrame:
    if not row_filter:
        return frame
    row_filter = row_filter.strip()
    if not row_filter or row_filter == "all_rows":
        return frame

    comparison = _parse_simple_comparison(row_filter)
    if comparison is None:
        return frame

    column, operator, threshold = comparison
    if column not in frame.columns:
        return frame.iloc[0:0].copy()
    values = pd.to_numeric(frame[column], errors="coerce")
    if operator == "<":
        mask = values < threshold
    elif operator == "<=":
        mask = values <= threshold
    elif operator == ">":
        mask = values > threshold
    elif operator == ">=":
        mask = values >= threshold
    elif operator in {"=", "=="}:
        mask = values == threshold
    else:
        return frame
    return frame[mask.fillna(False)].copy()


def _parse_simple_comparison(
    row_filter: str,
) -> tuple[str, str, float] | None:
    for operator in ("<=", ">=", "<", ">", "==", "="):
        if operator not in row_filter:
            continue
        left, right = row_filter.split(operator, 1)
        try:
            return left.strip(), operator, float(right.strip())
        except ValueError:
            return None
    return None


def _artifact_specs(spec: Mapping[str, object]) -> list[dict[str, object]]:
    artifacts = spec.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("normalized grading specification artifacts must be a list")
    return [_string_key_mapping(artifact, "grading artifact") for artifact in artifacts]


def _score_combination(spec_artifact: Mapping[str, object]) -> str:
    grading_method = _required_mapping(
        spec_artifact,
        "grading_method",
        "grading artifact",
    )
    value = grading_method.get("score_combination", "product")
    if not isinstance(value, str):
        raise ValueError("grading_method.score_combination must be a string")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
    context: str,
) -> dict[str, object]:
    if key not in mapping:
        raise ValueError(f"{context} is missing {key!r}")
    return _string_key_mapping(mapping[key], f"{context}.{key}")


def _mapping_value(value: object, context: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _string_key_mapping(value: object, context: str) -> dict[str, object]:
    mapping = _mapping_value(value, context)
    if not all(isinstance(key, str) for key in mapping):
        raise ValueError(f"{context} keys must be strings")
    return {str(key): item for key, item in mapping.items()}


def _string_mapping(value: object, context: str) -> dict[str, str]:
    mapping = _string_key_mapping(value, context)
    if not all(isinstance(item, str) for item in mapping.values()):
        raise ValueError(f"{context} values must be strings")
    return {key: str(item) for key, item in mapping.items()}


def _required_string(
    mapping: Mapping[str, object],
    key: str,
    context: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}.{key} must be a string")
    return value


def _optional_string(value: object, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def _string_list(value: object, context: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return [str(item) for item in value]


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
