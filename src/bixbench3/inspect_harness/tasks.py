from __future__ import annotations

import os
from pathlib import Path

from inspect_ai import Task, task

from bixbench3.inspect_harness.dataset import (
    DEFAULT_TIME_LIMIT_SECONDS,
    load_cases_dataset,
)
from bixbench3.inspect_harness.scorers import output_collection
from bixbench3.inspect_harness.solvers import (
    bixbench3_agent,
    prepare_case_workspace,
)

MESSAGE_LIMIT = 5_000
TIME_LIMIT_SECONDS = DEFAULT_TIME_LIMIT_SECONDS
DEFAULT_SANDBOX_COMPOSE = Path(__file__).parents[3] / "sandbox" / "compose.yaml"


@task
def bixbench3_cases(
    dataset_path: str,
    case_ids: list[str] | None = None,
    effort: str = "unspecified",
) -> Task:
    compose_path = os.environ.get(
        "BIXBENCH3_INSPECT_SANDBOX_COMPOSE",
        str(DEFAULT_SANDBOX_COMPOSE),
    )
    return Task(
        dataset=load_cases_dataset(
            dataset_path,
            case_ids,
            time_limit_seconds=TIME_LIMIT_SECONDS,
        ),
        setup=prepare_case_workspace(),
        solver=bixbench3_agent(),
        scorer=output_collection(),
        sandbox=("docker", compose_path),
        message_limit=MESSAGE_LIMIT,
        time_limit=TIME_LIMIT_SECONDS,
        token_limit=None,
        metadata={
            "effort": effort,
            "artifact_grading": "host_after_inspect_exit",
        },
    )
