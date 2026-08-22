from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Protocol, TypedDict, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr


class MetricResult(TypedDict):
    score: float
    details: dict[str, object]


class PairScore(Protocol):
    def __call__(
        self,
        ground_truth: pd.Series[Any],
        agent_output: pd.Series[Any],
        *,
        value_transform: str | None,
    ) -> float | None: ...


Interval = tuple[str, int, int]
FloatArray = NDArray[np.float64]


def f1_score(expected: Iterable[str], observed: Iterable[str]) -> MetricResult:
    expected_set = {str(item) for item in expected}
    observed_set = {str(item) for item in observed}
    true_positive = len(expected_set & observed_set)
    false_positive = len(observed_set - expected_set)
    false_negative = len(expected_set - observed_set)
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    score = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "score": score,
        "details": {
            "expected_count": len(expected_set),
            "observed_count": len(observed_set),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
        },
    }


def spearman_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    row_key: str,
    axis: str,
    value_columns: list[str] | None = None,
    value_rows: list[str] | None = None,
    value_transform: str | None = None,
) -> MetricResult:
    return _aligned_numeric_score(
        ground_truth,
        agent_output,
        row_key=row_key,
        axis=axis,
        value_columns=value_columns,
        value_rows=value_rows,
        value_transform=value_transform,
        pair_score=_spearman_pair,
        metric_name="spearman",
    )


def ccc_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    row_key: str,
    axis: str,
    value_columns: list[str] | None = None,
    value_rows: list[str] | None = None,
    value_transform: str | None = None,
) -> MetricResult:
    return _aligned_ccc_score(
        ground_truth,
        agent_output,
        row_key=row_key,
        axis=axis,
        value_columns=value_columns,
        value_rows=value_rows,
        value_transform=value_transform,
    )


def _aligned_numeric_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    row_key: str,
    axis: str,
    value_columns: list[str] | None,
    value_rows: list[str] | None,
    value_transform: str | None,
    pair_score: PairScore,
    metric_name: str,
) -> MetricResult:
    gt_indexed = _index_by_key(ground_truth, row_key)
    agent_indexed = _index_by_key(agent_output, row_key)
    common_rows = _ordered_intersection(gt_indexed.index, agent_indexed.index)
    candidate_columns = value_columns or _non_key_common_columns(
        ground_truth,
        agent_output,
        row_key,
    )
    common_columns = [
        column
        for column in candidate_columns
        if column in gt_indexed.columns and column in agent_indexed.columns
    ]

    if value_rows is not None:
        common_row_set = set(common_rows)
        common_rows = [row for row in value_rows if row in common_row_set]

    labels: Sequence[object]
    if axis == "columns":
        labels = common_columns
        pairs = [
            pair_score(
                cast("pd.Series[Any]", gt_indexed.loc[common_rows, column]),
                cast("pd.Series[Any]", agent_indexed.loc[common_rows, column]),
                value_transform=value_transform,
            )
            for column in common_columns
        ]
    elif axis == "rows":
        labels = common_rows
        pairs = [
            pair_score(
                cast("pd.Series[Any]", gt_indexed.loc[row, common_columns]),
                cast("pd.Series[Any]", agent_indexed.loc[row, common_columns]),
                value_transform=value_transform,
            )
            for row in common_rows
        ]
    else:
        raise ValueError(f"unsupported {metric_name} axis: {axis}")

    scores = [score for score in pairs if score is not None]
    score = sum(scores) / len(scores) if scores else 0.0
    return {
        "score": score,
        "details": {
            "axis": axis,
            "item_count": len(labels),
            "scored_item_count": len(scores),
            "row_intersection_count": len(common_rows),
            "column_intersection_count": len(common_columns),
            "missing_requested_columns": [
                column for column in candidate_columns if column not in common_columns
            ],
        },
    }


def _aligned_ccc_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    row_key: str,
    axis: str,
    value_columns: list[str] | None,
    value_rows: list[str] | None,
    value_transform: str | None,
) -> MetricResult:
    gt_indexed = _index_by_key(ground_truth, row_key)
    agent_indexed = _index_by_key(agent_output, row_key)
    common_rows = _ordered_intersection(gt_indexed.index, agent_indexed.index)
    candidate_columns = value_columns or _non_key_common_columns(
        ground_truth,
        agent_output,
        row_key,
    )
    common_columns = [
        column
        for column in candidate_columns
        if column in gt_indexed.columns and column in agent_indexed.columns
    ]

    if value_rows is not None:
        common_row_set = set(common_rows)
        common_rows = [row for row in value_rows if row in common_row_set]

    labels: Sequence[object]
    if axis == "columns":
        labels = common_columns
        item_axis = 0
    elif axis == "rows":
        labels = common_rows
        item_axis = 1
    else:
        raise ValueError(f"unsupported ccc axis: {axis}")

    gt_block = gt_indexed.loc[common_rows, common_columns]
    agent_block = agent_indexed.loc[common_rows, common_columns]
    gt_numeric = _numeric_frame_to_array(gt_block)
    agent_numeric = _numeric_frame_to_array(agent_block)
    scores = _ccc_scores_from_arrays(
        gt_numeric,
        agent_numeric,
        item_axis=item_axis,
        value_transform=value_transform,
    )
    scored_item_count = int(np.count_nonzero(~np.isnan(scores)))
    score = float(np.nanmean(scores)) if scored_item_count else 0.0
    return {
        "score": score,
        "details": {
            "axis": axis,
            "item_count": len(labels),
            "scored_item_count": scored_item_count,
            "row_intersection_count": len(common_rows),
            "column_intersection_count": len(common_columns),
            "missing_requested_columns": [
                column for column in candidate_columns if column not in common_columns
            ],
        },
    }


def _numeric_frame_to_array(frame: pd.DataFrame) -> FloatArray:
    try:
        array = frame.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        array = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return np.asarray(array, dtype=np.float64)


def _ccc_scores_from_arrays(
    ground_truth: FloatArray,
    agent_output: FloatArray,
    *,
    item_axis: int,
    value_transform: str | None,
) -> FloatArray:
    if value_transform == "log1p":
        ground_truth = np.log1p(np.clip(ground_truth, 0, None))
        agent_output = np.log1p(np.clip(agent_output, 0, None))

    item_count = ground_truth.shape[1] if item_axis == 0 else ground_truth.shape[0]
    scores = np.full(item_count, np.nan, dtype=np.float64)
    for item_index in range(item_count):
        if item_axis == 0:
            gt_values = ground_truth[:, item_index]
            agent_values = agent_output[:, item_index]
        else:
            gt_values = ground_truth[item_index, :]
            agent_values = agent_output[item_index, :]
        score = _ccc_array_pair(gt_values, agent_values)
        if score is not None:
            scores[item_index] = score
    return scores


def _ccc_array_pair(
    ground_truth: FloatArray,
    agent_output: FloatArray,
) -> float | None:
    mask = ~np.isnan(ground_truth) & ~np.isnan(agent_output)
    gt_values = ground_truth[mask].astype(float)
    agent_values = agent_output[mask].astype(float)
    if len(gt_values) == 0:
        return None
    if len(gt_values) == 1:
        return 1.0 if float(gt_values[0]) == float(agent_values[0]) else 0.0
    if np.unique(gt_values).size <= 1 or np.unique(agent_values).size <= 1:
        return 1.0 if np.array_equal(gt_values, agent_values) else 0.0

    gt_mean = float(gt_values.mean())
    agent_mean = float(agent_values.mean())
    gt_var = float(gt_values.var(ddof=0))
    agent_var = float(agent_values.var(ddof=0))
    covariance = float(((gt_values - gt_mean) * (agent_values - agent_mean)).mean())
    denominator = gt_var + agent_var + (gt_mean - agent_mean) ** 2
    if denominator == 0:
        return None
    return _clip_score(2 * covariance / denominator)


def label_f1_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    row_key: str,
    label_column: str,
) -> MetricResult:
    gt_indexed = _index_by_key(ground_truth, row_key)
    agent_indexed = _index_by_key(agent_output, row_key)
    common_rows = _ordered_intersection(gt_indexed.index, agent_indexed.index)
    if label_column not in gt_indexed.columns or label_column not in agent_indexed.columns:
        return {
            "score": 0.0,
            "details": {
                "row_intersection_count": len(common_rows),
                "error": f"missing label column {label_column!r}",
            },
        }

    gt_labels = gt_indexed.loc[common_rows, label_column].astype(str)
    agent_labels = agent_indexed.loc[common_rows, label_column].astype(str)
    labels = sorted(set(gt_labels) | set(agent_labels))
    gt_counts = gt_labels.value_counts()
    agent_counts = agent_labels.value_counts()
    true_positive_counts = gt_labels[gt_labels == agent_labels].value_counts()
    per_label: dict[str, float] = {}
    for label in labels:
        true_positive = int(true_positive_counts.get(label, 0))
        false_positive = int(agent_counts.get(label, 0)) - true_positive
        false_negative = int(gt_counts.get(label, 0)) - true_positive
        precision = _safe_divide(true_positive, true_positive + false_positive)
        recall = _safe_divide(true_positive, true_positive + false_negative)
        per_label[label] = _safe_divide(2 * precision * recall, precision + recall)
    score = sum(per_label.values()) / len(per_label) if per_label else 0.0
    return {
        "score": score,
        "details": {
            "row_intersection_count": len(common_rows),
            "label_count": len(labels),
            "per_label_f1": per_label,
        },
    }


def interval_f1_score(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    *,
    interval_columns: dict[str, str],
    overlap_rule: dict[str, object],
) -> MetricResult:
    gt_intervals = _intervals_from_frame(ground_truth, interval_columns)
    agent_intervals = _intervals_from_frame(agent_output, interval_columns)
    gt_matched = _matched_interval_indices(gt_intervals, agent_intervals, overlap_rule)
    agent_matched = _matched_interval_indices(agent_intervals, gt_intervals, overlap_rule)
    precision = _safe_divide(len(agent_matched), len(agent_intervals))
    recall = _safe_divide(len(gt_matched), len(gt_intervals))
    score = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "score": score,
        "details": {
            "ground_truth_interval_count": len(gt_intervals),
            "agent_interval_count": len(agent_intervals),
            "matched_ground_truth_count": len(gt_matched),
            "matched_agent_count": len(agent_matched),
            "precision": precision,
            "recall": recall,
        },
    }


def _index_by_key(frame: pd.DataFrame, row_key: str) -> pd.DataFrame:
    if row_key not in frame.columns:
        raise ValueError(f"row key {row_key!r} not found in columns")
    return frame.drop_duplicates(subset=[row_key], keep="first").set_index(row_key)


def _non_key_common_columns(
    ground_truth: pd.DataFrame,
    agent_output: pd.DataFrame,
    row_key: str,
) -> list[str]:
    agent_columns = set(agent_output.columns)
    return [
        str(column)
        for column in ground_truth.columns
        if column != row_key and column in agent_columns
    ]


def _ordered_intersection[HashableValue: Hashable](
    left: Iterable[HashableValue],
    right: Iterable[HashableValue],
) -> list[HashableValue]:
    right_set = set(right)
    return [item for item in left if item in right_set]


def _spearman_pair(
    ground_truth: pd.Series[Any],
    agent_output: pd.Series[Any],
    *,
    value_transform: str | None,
) -> float | None:
    prepared = _prepare_numeric_pair(
        ground_truth,
        agent_output,
        value_transform=value_transform,
    )
    if prepared is None:
        return None
    gt_values, agent_values, degenerate_score = prepared
    if degenerate_score is not None:
        return degenerate_score
    result = spearmanr(gt_values, agent_values, nan_policy="omit")
    if bool(pd.isna(result.statistic)):
        return None
    return _clip_score(float(result.statistic))


def _prepare_numeric_pair(
    ground_truth: pd.Series[Any],
    agent_output: pd.Series[Any],
    *,
    value_transform: str | None,
) -> tuple[pd.Series[float], pd.Series[float], float | None] | None:
    gt_numeric = pd.to_numeric(ground_truth, errors="coerce")
    agent_numeric = pd.to_numeric(agent_output, errors="coerce")
    mask = gt_numeric.notna() & agent_numeric.notna()
    gt_values = gt_numeric[mask].astype(float)
    agent_values = agent_numeric[mask].astype(float)
    if value_transform == "log1p":
        gt_values = cast("pd.Series[float]", np.log1p(gt_values.clip(lower=0)))
        agent_values = cast(
            "pd.Series[float]",
            np.log1p(agent_values.clip(lower=0)),
        )

    if len(gt_values) == 0:
        return None
    if len(gt_values) == 1:
        score = 1.0 if float(gt_values.iloc[0]) == float(agent_values.iloc[0]) else 0.0
        return gt_values, agent_values, score
    if gt_values.nunique(dropna=True) <= 1 or agent_values.nunique(dropna=True) <= 1:
        score = (
            1.0
            if gt_values.reset_index(drop=True).equals(
                agent_values.reset_index(drop=True),
            )
            else 0.0
        )
        return gt_values, agent_values, score
    return gt_values, agent_values, None


def _clip_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _intervals_from_frame(
    frame: pd.DataFrame,
    interval_columns: dict[str, str],
) -> list[Interval]:
    chrom_col = interval_columns.get("chrom", "chrom")
    start_col = interval_columns.get("start", "start")
    end_col = interval_columns.get("end", "end")
    missing = [column for column in (chrom_col, start_col, end_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"missing interval column(s): {', '.join(missing)}")

    intervals: list[Interval] = []
    for _, row in frame.iterrows():
        intervals.append(
            (str(row[chrom_col]), int(row[start_col]), int(row[end_col])),
        )
    return intervals


def _matched_interval_indices(
    queries: list[Interval],
    candidates: list[Interval],
    overlap_rule: dict[str, object],
) -> set[int]:
    candidates_by_group: dict[str, list[tuple[int, int, Interval]]] = defaultdict(
        list,
    )
    for candidate in candidates:
        candidates_by_group[candidate[0]].append(
            (candidate[1], candidate[2], candidate),
        )

    interval_indexes: dict[
        str,
        tuple[list[int], list[int], list[Interval]],
    ] = {}
    for group, group_candidates in candidates_by_group.items():
        group_candidates.sort(key=lambda item: item[0])
        starts = [item[0] for item in group_candidates]
        intervals = [item[2] for item in group_candidates]
        prefix_max_ends: list[int] = []
        max_end: int | None = None
        for _, end, _ in group_candidates:
            max_end = end if max_end is None else max(max_end, end)
            prefix_max_ends.append(max_end)
        interval_indexes[group] = (starts, prefix_max_ends, intervals)

    matched: set[int] = set()
    for query_index, query in enumerate(queries):
        if _has_matching_interval(query, interval_indexes.get(query[0]), overlap_rule):
            matched.add(query_index)
    return matched


def _has_matching_interval(
    query: Interval,
    candidate_index: tuple[list[int], list[int], list[Interval]] | None,
    overlap_rule: dict[str, object],
) -> bool:
    if candidate_index is None:
        return False
    starts, prefix_max_ends, candidates = candidate_index
    query_start = query[1]
    query_end = query[2]
    candidate_position = bisect_left(starts, query_end) - 1
    while candidate_position >= 0 and prefix_max_ends[candidate_position] > query_start:
        candidate = candidates[candidate_position]
        if _intervals_overlap(query, candidate, overlap_rule):
            return True
        candidate_position -= 1
    return False


def _intervals_overlap(
    expected: Interval,
    observed: Interval,
    overlap_rule: dict[str, object],
) -> bool:
    if expected[0] != observed[0]:
        return False
    overlap = max(0, min(expected[2], observed[2]) - max(expected[1], observed[1]))
    if overlap <= 0:
        return False
    rule_type = overlap_rule.get("type", "any_overlap")
    if not isinstance(rule_type, str):
        raise ValueError("interval overlap rule type must be a string")
    if rule_type == "any_overlap":
        return True
    if rule_type == "reciprocal_overlap":
        minimum_fraction = _float_config(
            overlap_rule.get("minimum_fraction", 0.5),
            "minimum_fraction",
        )
        expected_length = max(1, expected[2] - expected[1])
        observed_length = max(1, observed[2] - observed[1])
        return (
            overlap / expected_length >= minimum_fraction
            and overlap / observed_length >= minimum_fraction
        )
    if rule_type == "within_bp":
        tolerance = _int_config(overlap_rule.get("tolerance_bp", 0), "tolerance_bp")
        expected_mid = (expected[1] + expected[2]) // 2
        observed_mid = (observed[1] + observed[2]) // 2
        return abs(expected_mid - observed_mid) <= tolerance
    raise ValueError(f"unsupported interval overlap rule: {rule_type}")


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _float_config(value: object, name: str) -> float:
    if not isinstance(value, str | int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _int_config(value: object, name: str) -> int:
    if not isinstance(value, str | int | float):
        raise ValueError(f"{name} must be numeric")
    return int(value)
