from __future__ import annotations

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState
from inspect_ai.util import sandbox

from .models import RuntimeCase


@scorer(metrics=[mean(), stderr()])
def output_collection() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        del target
        case = state.metadata_as(RuntimeCase)
        artifacts: list[dict[str, object]] = []
        present = 0
        for output in case.expected_outputs:
            result = await sandbox().exec(["stat", "-c", "%s", output.sandbox_path])
            exists = result.success
            size = _parse_size(result.stdout) if exists else 0
            nonempty = exists and size > 0
            present += int(nonempty)
            artifacts.append(
                {
                    "artifact_id": output.artifact_id,
                    "expected_path": output.path,
                    "sandbox_path": output.sandbox_path,
                    "present": exists,
                    "nonempty": nonempty,
                    "bytes": size,
                }
            )
        methods_result = await sandbox().exec(["stat", "-c", "%s", "/outputs/METHODS.md"])
        total = len(case.expected_outputs)
        completeness = present / total if total else 1.0
        return Score(
            value=completeness,
            answer=f"{present}/{total} expected artifacts are present and non-empty",
            explanation=(
                "This is an output-collection health check only. Deterministic artifact "
                "grading runs on the VM host after Inspect and its sandbox have exited."
            ),
            metadata={
                "task_id": case.task_id,
                "collection_only": True,
                "present_artifacts": present,
                "expected_artifacts": total,
                "methods_present": methods_result.success,
                "artifacts": artifacts,
            },
        )

    return score


def _parse_size(value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        return 0
