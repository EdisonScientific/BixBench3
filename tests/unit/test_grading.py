from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from bixbench3.grading.grader import (
    grade,
    grade_paper,
    normalize_grading_artifact,
)
from bixbench3.grading.loaders import load_table
from bixbench3.grading.metrics import (
    ccc_score,
    f1_score,
    interval_f1_score,
    label_f1_score,
    spearman_score,
)


def test_f1_uses_unique_set_membership() -> None:
    result = f1_score(["a", "a", "b", "c"], ["a", "b", "b", "d"])

    assert result["score"] == pytest.approx(2 / 3)
    assert result["details"]["true_positive"] == 2
    assert result["details"]["false_positive"] == 1
    assert result["details"]["false_negative"] == 1


def test_numeric_metrics_align_rows_and_clip_negative_correlation() -> None:
    ground_truth = pd.DataFrame(
        {"gene_id": ["a", "b", "c"], "value": [1, 2, 3]},
    )
    perfect = pd.DataFrame(
        {"gene_id": ["c", "a", "b"], "value": [3, 1, 2]},
    )
    reversed_values = pd.DataFrame(
        {"gene_id": ["a", "b", "c"], "value": [3, 2, 1]},
    )

    ccc = ccc_score(
        ground_truth,
        perfect,
        row_key="gene_id",
        axis="columns",
        value_columns=["value"],
    )
    spearman = spearman_score(
        ground_truth,
        reversed_values,
        row_key="gene_id",
        axis="columns",
        value_columns=["value"],
    )

    assert ccc["score"] == pytest.approx(1.0)
    assert spearman["score"] == 0.0


def test_label_f1_is_macro_averaged_over_observed_and_expected_labels() -> None:
    ground_truth = pd.DataFrame(
        {"gene_id": ["a", "b", "c"], "label": ["x", "y", "y"]},
    )
    agent_output = pd.DataFrame(
        {"gene_id": ["a", "b", "c"], "label": ["x", "z", "y"]},
    )

    result = label_f1_score(
        ground_truth,
        agent_output,
        row_key="gene_id",
        label_column="label",
    )

    assert result["details"]["per_label_f1"] == {
        "x": 1.0,
        "y": pytest.approx(2 / 3),
        "z": 0.0,
    }
    assert result["score"] == pytest.approx(5 / 9)


def test_interval_f1_preserves_bidirectional_match_semantics() -> None:
    ground_truth = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [0, 20], "end": [10, 30]},
    )
    agent_output = pd.DataFrame(
        {"chrom": ["chr1", "chr1"], "start": [5, 40], "end": [15, 50]},
    )

    result = interval_f1_score(
        ground_truth,
        agent_output,
        interval_columns={},
        overlap_rule={"type": "any_overlap"},
    )

    assert result["score"] == pytest.approx(0.5)
    assert result["details"]["matched_ground_truth_count"] == 1
    assert result["details"]["matched_agent_count"] == 1


def test_load_table_supports_csv_tsv_and_headerless_bed(tmp_path: Path) -> None:
    csv_path = tmp_path / "table.csv"
    tsv_path = tmp_path / "table.tsv"
    bed_path = tmp_path / "regions.bed"
    csv_path.write_text("id,value\na,1\n", encoding="utf-8")
    tsv_path.write_text("id\tvalue\na\t1\n", encoding="utf-8")
    bed_path.write_text("chr1\t0\t10\nchr2\t20\t30\n", encoding="utf-8")

    csv_frame = load_table(csv_path)
    tsv_frame = load_table(tsv_path)
    bed_frame = load_table(bed_path)

    assert csv_frame.to_dict("records") == [{"id": "a", "value": "1"}]
    assert tsv_frame.to_dict("records") == [{"id": "a", "value": "1"}]
    assert bed_frame.columns.tolist() == ["chrom", "start", "end"]
    assert bed_frame.to_dict("records") == [
        {"chrom": "chr1", "start": "0", "end": "10"},
        {"chrom": "chr2", "start": "20", "end": "30"},
    ]


def test_normalize_grading_artifact_uses_expected_output_canonically() -> None:
    expected_output = {
        "path": "outputs/result.csv",
        "format": "csv",
        "row_key": "id",
    }

    normalized = normalize_grading_artifact(
        {
            "artifact_id": "result",
            "expected_output": expected_output,
        },
    )

    assert normalized["output_format"] == expected_output
    assert normalized["output_format"] is not expected_output


def test_grade_products_checks_and_averages_all_artifacts(tmp_path: Path) -> None:
    truth_root = tmp_path / "grader"
    artifacts_dir = tmp_path / "agent_artifacts"
    artifact_grades_dir = tmp_path / "artifact_grades"
    (truth_root / "artifacts").mkdir(parents=True)
    artifacts_dir.mkdir()
    (truth_root / "artifacts" / "one.csv").write_text(
        "id,value\na,1\nb,2\nc,3\n",
        encoding="utf-8",
    )
    (truth_root / "artifacts" / "two.csv").write_text(
        "id,value\na,1\n",
        encoding="utf-8",
    )
    (artifacts_dir / "one.csv").write_text(
        "id,value\na,1\nb,2\n",
        encoding="utf-8",
    )
    spec = _paper_spec()

    result = grade_paper(
        spec,
        agent_artifacts_dir=artifacts_dir,
        ground_truth_root=truth_root,
        artifact_grades_dir=artifact_grades_dir,
    )

    assert result["artifacts"][0]["checks"][0]["score"] == pytest.approx(0.8)
    assert result["artifacts"][0]["checks"][1]["score"] == pytest.approx(1.0)
    assert result["artifacts"][0]["score"] == pytest.approx(0.8)
    assert result["artifacts"][1]["score"] == 0.0
    assert result["artifacts"][1]["error"]["reason"] == "missing_agent_artifact"
    assert result["score"] == pytest.approx(0.4)
    assert (artifact_grades_dir / "one.json").is_file()
    assert (artifact_grades_dir / "two.json").is_file()
    assert json.loads((tmp_path / "grade.json").read_text())["score"] == pytest.approx(0.4)


def test_grade_function_infers_ground_truth_root_and_writes_grades(
    tmp_path: Path,
) -> None:
    grader_root = tmp_path / "grader"
    artifacts_dir = tmp_path / "agent_artifacts"
    (grader_root / "artifacts").mkdir(parents=True)
    artifacts_dir.mkdir()
    (grader_root / "artifacts" / "one.csv").write_text(
        "id,value\na,1\nb,2\nc,3\n",
        encoding="utf-8",
    )
    (grader_root / "artifacts" / "two.csv").write_text(
        "id,value\na,1\n",
        encoding="utf-8",
    )
    (artifacts_dir / "one.csv").write_text(
        "id,value\na,1\nb,2\nc,3\n",
        encoding="utf-8",
    )
    (artifacts_dir / "two.csv").write_text(
        "id,value\na,1\n",
        encoding="utf-8",
    )
    spec_path = grader_root / "artifacts_grading.json"
    spec_path.write_text(json.dumps(_paper_spec()), encoding="utf-8")

    result = grade(spec_path, artifacts_dir)

    assert result["score"] == pytest.approx(1.0)
    assert (tmp_path / "artifact_grades" / "one.json").is_file()
    assert (tmp_path / "grade.json").is_file()


@pytest.mark.parametrize(
    ("contents", "file_format", "reason"),
    [
        ("", "csv", "empty_agent_artifact"),
        ("id,value\na,1\n", "parquet", "artifact_grading_error"),
    ],
)
def test_empty_and_grading_errors_receive_zero(
    tmp_path: Path,
    contents: str,
    file_format: str,
    reason: str,
) -> None:
    truth_root = tmp_path / "grader"
    artifacts_dir = tmp_path / "agent_artifacts"
    artifact_grades_dir = tmp_path / "artifact_grades"
    (truth_root / "artifacts").mkdir(parents=True)
    artifacts_dir.mkdir()
    (truth_root / "artifacts" / "one.csv").write_text(
        "id,value\na,1\n",
        encoding="utf-8",
    )
    (artifacts_dir / "one.csv").write_text(contents, encoding="utf-8")
    artifact = _artifact_spec("one", "one.csv", "one.csv")
    output_format = artifact["expected_output"]
    assert isinstance(output_format, dict)
    output_format["format"] = file_format
    spec: dict[str, object] = {
        "paper_id": "paper",
        "artifacts": [artifact],
    }

    result = grade_paper(
        spec,
        agent_artifacts_dir=artifacts_dir,
        ground_truth_root=truth_root,
        artifact_grades_dir=artifact_grades_dir,
    )

    assert result["score"] == 0.0
    assert result["artifacts"][0]["error"]["reason"] == reason


def _paper_spec() -> dict[str, object]:
    return {
        "paper_id": "paper",
        "doi": "10.0000/example",
        "artifacts": [
            _artifact_spec("one", "one.csv", "one.csv"),
            _artifact_spec("two", "two.csv", "two.csv"),
        ],
    }


def _artifact_spec(
    artifact_id: str,
    ground_truth_name: str,
    output_name: str,
) -> dict[str, object]:
    return {
        "artifact_id": artifact_id,
        "ground_truth": {
            "path": f"artifacts/{ground_truth_name}",
            "format": "csv",
        },
        "expected_output": {
            "path": f"outputs/{output_name}",
            "format": "csv",
            "row_key": "id",
        },
        "grading_method": {
            "value_columns": ["value"],
            "checks": [
                {"metric": "f1", "axis": "rows"},
                {
                    "metric": "ccc",
                    "axis": "columns",
                    "scope": {"columns": "value_columns"},
                },
            ],
            "score_combination": "product",
        },
    }
