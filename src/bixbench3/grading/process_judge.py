from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from inspect_ai.model import GenerateConfig, get_model

from bixbench3.inspect_harness.models import RuntimeCase

from .effort import resolve_reasoning_effort
from .judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING_EFFORT,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_MAX_TRACE_CHARS,
    parse_judge_response,
    preview_text,
    render_judge_prompt,
    report_error,
    validation_error_text,
)
from .trace_compact import compact_eval_log_data, eval_log_input_text, load_eval_log


def run_process_judge(
    *,
    run_dir: Path,
    dataset_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Run the canonical PhDval-1 process judge for one BixBench3 run."""
    return asyncio.run(
        judge_run_dir(
            run_dir=run_dir,
            dataset_path=dataset_path,
            force=force,
        )
    )


async def judge_run_dir(
    *,
    run_dir: Path,
    dataset_path: Path,
    force: bool = False,
) -> dict[str, Any]:
    output_path = run_dir / "judge.json"
    if output_path.exists() and not force:
        loaded = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"{output_path} must contain a JSON object")
        return loaded

    case = RuntimeCase.model_validate_json(dataset_path.read_text(encoding="utf-8"))
    artifact_ids = [output.artifact_id for output in case.expected_outputs]
    try:
        report = await _judge_run_inputs(
            run_dir=run_dir,
            case=case,
            artifact_ids=artifact_ids,
        )
    except Exception as error:
        report = report_error(
            paper_id=case.task_id,
            artifact_ids=artifact_ids,
            message=f"process judge failed: {error}",
            status="judge_failed",
        )
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


async def _judge_run_inputs(
    *,
    run_dir: Path,
    case: RuntimeCase,
    artifact_ids: list[str],
) -> dict[str, Any]:
    eval_log = _latest_eval_log(run_dir)
    if eval_log is None:
        return report_error(
            paper_id=case.task_id,
            artifact_ids=artifact_ids,
            message=f"missing inspect eval log under {run_dir / 'inspect_logs'}",
        )
    eval_data = load_eval_log(eval_log)
    compacted_trace = compact_eval_log_data(
        eval_data,
        max_tool_output_chars=DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        max_trace_chars=DEFAULT_MAX_TRACE_CHARS,
    )
    judge_prompt = render_judge_prompt(
        paper_id=case.task_id,
        agent_prompt=eval_log_input_text(eval_data),
        artifact_ids=artifact_ids,
        methods_md=_read_or_marker(run_dir / "outputs" / "METHODS.md"),
        numeric_grade_summary=_read_or_marker(run_dir / "grade.json"),
        artifact_previews=_artifact_previews(run_dir, case),
        compacted_trace=compacted_trace,
    )
    return await _call_judge_model(
        prompt=judge_prompt,
        paper_id=case.task_id,
        artifact_ids=artifact_ids,
    )


async def _call_judge_model(
    *,
    prompt: str,
    paper_id: str,
    artifact_ids: list[str],
) -> dict[str, Any]:
    model = get_model(
        DEFAULT_JUDGE_MODEL,
        config=GenerateConfig(
            reasoning_effort=resolve_reasoning_effort(
                DEFAULT_JUDGE_MODEL,
                DEFAULT_JUDGE_REASONING_EFFORT,
            ),  # type: ignore[arg-type]
            temperature=0,
        ),
    )
    result = await model.generate(prompt)
    try:
        report = parse_judge_response(result.completion)
    except Exception as error:
        retry = (
            prompt
            + "\n\n# Previous Response Validation Error\n\n"
            + validation_error_text(error)
            + "\n\nReturn a corrected JSON object only."
        )
        result = await model.generate(retry)
        try:
            report = parse_judge_response(result.completion)
        except Exception as retry_error:
            return report_error(
                paper_id=paper_id,
                artifact_ids=artifact_ids,
                message=f"could not parse judge response: {validation_error_text(retry_error)}",
                status=_judge_error_status(validation_error_text(retry_error)),
            )
    return report.model_dump()


def _latest_eval_log(run_dir: Path) -> Path | None:
    logs = sorted(
        (run_dir / "inspect_logs").glob("*.eval"),
        key=lambda path: path.stat().st_mtime,
    )
    return logs[-1] if logs else None


def _artifact_previews(run_dir: Path, case: RuntimeCase) -> str:
    previews: list[str] = []
    for output in case.expected_outputs:
        path = run_dir / "outputs" / Path(output.path).name
        previews.append(preview_text(output.artifact_id, _read_or_marker(path)))
    return "\n\n".join(previews)


def _read_or_marker(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[missing file: {path}]"


def _judge_error_status(message: str) -> str:
    lowered = message.lower()
    if "context window" in lowered or "model_length" in lowered or "too many tokens" in lowered:
        return "context_overflow"
    return "parse_failed"
