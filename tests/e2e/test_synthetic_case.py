from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import bixbench3.inspect_harness.tasks as task_module
from bixbench3.grading.grader import grade
from bixbench3.inspect_harness.dataset import load_cases_dataset

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "synthetic_case"


def test_task_module_loads_from_its_file_path() -> None:
    runpy.run_path(task_module.__file__)


def test_synthetic_case_loads_and_self_grades(tmp_path: Path) -> None:
    dataset = load_cases_dataset(
        str(FIXTURE_ROOT),
        ["synthetic"],
        time_limit_seconds=300,
    )
    assert len(dataset) == 1
    assert dataset[0].id == "synthetic"
    assert "300 seconds" in dataset[0].input
    assert "cellranger 6.0.2" in dataset[0].input

    artifacts_dir = tmp_path / "agent_artifacts"
    artifacts_dir.mkdir()
    shutil.copy2(
        FIXTURE_ROOT / "grader" / "artifacts" / "counts.csv",
        artifacts_dir / "counts.csv",
    )
    result = grade(
        FIXTURE_ROOT / "grader" / "artifacts_grading.json",
        artifacts_dir,
    )
    assert result["score"] == 1.0
