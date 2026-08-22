from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from bixbench3.grading import process_judge
from bixbench3.grading.judge import (
    ANALYTICAL_RIGOR_RUBRICS,
    DEFAULT_JUDGE_MODEL,
    FAILURE_MODE_TAGS,
    INSTRUCTION_FOLLOWING_RUBRICS,
    PROMPT_TEMPLATE_PATH,
    PROMPT_TEMPLATE_SHA256,
    TRANSPARENCY_RUBRICS,
    JudgeReport,
    parse_judge_response,
    render_judge_prompt,
)
from bixbench3.grading.trace_compact import compact_state_messages
from bixbench3.inspect_harness.dataset import render_agent_prompt

EXPECTED_FAILURE_MODE_TAGS = (
    "environment_setup_failure",
    "input_misinterpretation",
    "wrong_method_substitution",
    "method_misconfigured",
    "incomplete_data",
    "output_format_violation",
    "synthetic_or_placeholder_output",
    "premature_termination",
    "inefficient_analysis",
    "repetitive_retry_loop",
)


def _report_payload(paper_id: str = "paper", artifact_id: str = "counts") -> dict[str, Any]:
    score = {"score": 5, "explanation": "Supported by the trace."}
    return {
        "paper_id": paper_id,
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "instruction_following": {
                    rubric: dict(score) for rubric in INSTRUCTION_FOLLOWING_RUBRICS
                },
                "transparency": {rubric: dict(score) for rubric in TRANSPARENCY_RUBRICS},
                "analytical_rigor": {
                    rubric: dict(score) for rubric in ANALYTICAL_RIGOR_RUBRICS
                },
            }
        ],
        "failure_modes": [],
        "overall_notes": None,
    }


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_prompt_and_failure_modes_are_pinned_to_phdval_contract() -> None:
    assert FAILURE_MODE_TAGS == EXPECTED_FAILURE_MODE_TAGS
    assert hashlib.sha256(PROMPT_TEMPLATE_PATH.read_bytes()).hexdigest() == PROMPT_TEMPLATE_SHA256


def test_report_normalizes_same_eight_rubrics() -> None:
    payload = _report_payload()
    payload["artifacts"][0]["analytical_rigor"]["result_validation"]["score"] = 1

    report = JudgeReport.model_validate(payload)

    assert report.judge_score == pytest.approx(0.9)
    reparsed = parse_judge_response(f"prefix\n{json.dumps(payload)}\nsuffix")
    assert reparsed.model_dump() == report.model_dump()


def test_prompt_rendering_matches_phdval_source_when_available() -> None:
    phdval_judge = (
        Path(__file__).parents[3] / "PhDval-1" / "source" / "grading" / "judge.py"
    )
    if not phdval_judge.is_file():
        pytest.skip("sibling PhDval-1 checkout is unavailable")
    canonical = _load_module("phdval_judge_contract", phdval_judge)
    arguments = {
        "paper_id": "paper",
        "agent_prompt": "analyze the staged inputs",
        "artifact_ids": ["counts", "summary"],
        "methods_md": "used the requested method",
        "numeric_grade_summary": '{"score": 0.75}',
        "artifact_previews": "## counts",
        "compacted_trace": "# Trace Digest",
    }

    assert canonical.FAILURE_MODE_TAGS == FAILURE_MODE_TAGS
    assert canonical.render_judge_prompt(**arguments) == render_judge_prompt(**arguments)


def test_trace_compaction_matches_phdval_source_when_available() -> None:
    phdval_trace = (
        Path(__file__).parents[3] / "PhDval-1" / "source" / "grading" / "trace_compact.py"
    )
    if not phdval_trace.is_file():
        pytest.skip("sibling PhDval-1 checkout is unavailable")
    canonical = _load_module("phdval_trace_contract", phdval_trace)
    messages = [
        {"role": "user", "content": "Inspect /data/input.csv."},
        {
            "role": "assistant",
            "content": "I will inspect the input.",
            "tool_calls": [
                {
                    "function": {
                        "name": "shell",
                        "arguments": {"command": "python inspect.py /data/input.csv"},
                    }
                }
            ],
        },
        {
            "role": "tool",
            "function": "shell",
            "content": "error: first attempt failed\nvalidated rows=10\nwrote /outputs/counts.csv",
        },
    ]

    assert compact_state_messages(messages, max_tool_output_chars=30) == (
        canonical.compact_state_messages(messages, max_tool_output_chars=30)
    )


def test_judge_call_uses_gpt55_xhigh_and_retries_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _report_payload()
    prompts: list[str] = []
    observed: dict[str, Any] = {}

    class FakeModel:
        async def generate(self, prompt: str) -> SimpleNamespace:
            prompts.append(prompt)
            completion = "{}" if len(prompts) == 1 else json.dumps(payload)
            return SimpleNamespace(completion=completion)

    def fake_get_model(model: str, *, config: Any) -> FakeModel:
        observed["model"] = model
        observed["config"] = config
        return FakeModel()

    monkeypatch.setattr(process_judge, "get_model", fake_get_model)

    report = asyncio.run(
        process_judge._call_judge_model(
            prompt="judge this run",
            paper_id="paper",
            artifact_ids=["counts"],
        )
    )

    assert observed["model"] == DEFAULT_JUDGE_MODEL
    assert observed["config"].reasoning_effort == "xhigh"
    assert observed["config"].temperature == 0
    assert len(prompts) == 2
    assert "# Previous Response Validation Error" in prompts[1]
    assert report["judge_score"] == 1.0


def test_second_invalid_response_matches_phdval_parse_failed_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class InvalidModel:
        async def generate(self, prompt: str) -> SimpleNamespace:
            del prompt
            return SimpleNamespace(completion="not json")

    monkeypatch.setattr(
        process_judge,
        "get_model",
        lambda *args, **kwargs: InvalidModel(),
    )

    report = asyncio.run(
        process_judge._call_judge_model(
            prompt="judge this run",
            paper_id="paper",
            artifact_ids=["counts"],
        )
    )

    assert report["judge_status"] == "parse_failed"
    assert report["judge_score"] is None


def test_run_dir_assembles_phdval_inputs_and_writes_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "dataset"
    outputs_dir = run_dir / "outputs"
    logs_dir = run_dir / "inspect_logs"
    dataset_dir.mkdir(parents=True)
    outputs_dir.mkdir()
    logs_dir.mkdir()
    prompt_path = dataset_dir / "prompt.md"
    prompt_path.write_text("# Task\n\nAnalyze the input.\n", encoding="utf-8")
    dataset_path = dataset_dir / "case.json"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "paper",
                "prompt": {"prompt_path": "prompt.md"},
                "expected_outputs": [
                    {
                        "artifact_id": "counts",
                        "path": "outputs/counts.csv",
                        "format": "csv",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (outputs_dir / "METHODS.md").write_text("Used the requested method.\n", encoding="utf-8")
    (outputs_dir / "counts.csv").write_text("gene,count\na,1\n", encoding="utf-8")
    (run_dir / "grade.json").write_text('{"score": 0.5}\n', encoding="utf-8")
    (logs_dir / "run.eval").touch()
    captured: dict[str, Any] = {}
    exact_agent_prompt = render_agent_prompt(prompt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(
        process_judge,
        "load_eval_log",
        lambda *args, **kwargs: {
            "samples": [{"input": exact_agent_prompt, "messages": []}]
        },
    )
    monkeypatch.setattr(
        process_judge,
        "compact_eval_log_data",
        lambda *args, **kwargs: "# trace",
    )

    async def fake_call(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return JudgeReport.model_validate(_report_payload()).model_dump()

    monkeypatch.setattr(process_judge, "_call_judge_model", fake_call)

    result = asyncio.run(
        process_judge.judge_run_dir(
            run_dir=run_dir,
            dataset_path=dataset_path,
        )
    )

    expected_prompt = render_judge_prompt(
        paper_id="paper",
        agent_prompt=exact_agent_prompt,
        artifact_ids=["counts"],
        methods_md="Used the requested method.\n",
        numeric_grade_summary='{"score": 0.5}\n',
        artifact_previews="## counts\n\n```text\ngene,count\na,1\n```",
        compacted_trace="# trace",
    )
    assert captured["prompt"] == expected_prompt
    assert result["judge_score"] == 1.0
    assert json.loads((run_dir / "judge.json").read_text(encoding="utf-8")) == result


def test_missing_eval_log_writes_nonfatal_error_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir(parents=True)
    prompt_path = dataset_dir / "prompt.md"
    prompt_path.write_text("# task\n", encoding="utf-8")
    dataset_path = dataset_dir / "case.json"
    dataset_path.write_text(
        json.dumps(
            {
                "task_id": "paper",
                "prompt": {"prompt_path": "prompt.md"},
                "expected_outputs": [{"artifact_id": "counts", "path": "counts.csv"}],
            }
        ),
        encoding="utf-8",
    )

    report = asyncio.run(
        process_judge.judge_run_dir(
            run_dir=run_dir,
            dataset_path=dataset_path,
        )
    )

    assert report["judge_status"] == "judge_failed"
    assert report["judge_score"] is None
    assert "missing inspect eval log" in report["error"]
