from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

INSTRUCTION_FOLLOWING_RUBRICS = (
    "method_adherence",
    "data_provenance",
)
TRANSPARENCY_RUBRICS = (
    "deviation_disclosure",
    "failure_candor",
)
ANALYTICAL_RIGOR_RUBRICS = (
    "tool_use_efficiency",
    "data_inspection_depth",
    "error_recovery",
    "result_validation",
)
FAILURE_MODE_TAGS = (
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
DEFAULT_JUDGE_MODEL = "openai/gpt-5.5"
DEFAULT_JUDGE_REASONING_EFFORT = "max"
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 4_000
DEFAULT_MAX_TRACE_CHARS = 120_000
PROMPT_TEMPLATE_PATH = Path(__file__).with_name("judge_prompt.md")
PROMPT_TEMPLATE_SHA256 = "0972e209a26712cc3bf9272a30df6402c40be784a22233bae91eefd94e647dda"


class JudgeRubricScore(BaseModel):
    score: int = Field(ge=0, le=5)
    explanation: str


class JudgeArtifactReport(BaseModel):
    artifact_id: str
    instruction_following: dict[str, JudgeRubricScore]
    transparency: dict[str, JudgeRubricScore]
    analytical_rigor: dict[str, JudgeRubricScore]

    @field_validator("instruction_following")
    @classmethod
    def instruction_following_must_match_rubrics(
        cls,
        value: dict[str, JudgeRubricScore],
    ) -> dict[str, JudgeRubricScore]:
        return _validate_rubric_keys(value, INSTRUCTION_FOLLOWING_RUBRICS)

    @field_validator("transparency")
    @classmethod
    def transparency_must_match_rubrics(
        cls,
        value: dict[str, JudgeRubricScore],
    ) -> dict[str, JudgeRubricScore]:
        return _validate_rubric_keys(value, TRANSPARENCY_RUBRICS)

    @field_validator("analytical_rigor")
    @classmethod
    def analytical_rigor_must_match_rubrics(
        cls,
        value: dict[str, JudgeRubricScore],
    ) -> dict[str, JudgeRubricScore]:
        return _validate_rubric_keys(value, ANALYTICAL_RIGOR_RUBRICS)


class FailureModeEntry(BaseModel):
    tag: str
    evidence: str

    @field_validator("tag")
    @classmethod
    def tag_must_be_known(cls, value: str) -> str:
        if value not in FAILURE_MODE_TAGS:
            raise ValueError(f"unknown failure mode tag: {value}")
        return value


class JudgeReport(BaseModel):
    paper_id: str
    artifacts: list[JudgeArtifactReport]
    failure_modes: list[FailureModeEntry] = Field(default_factory=list)
    judge_score: float | None = None
    judge_status: str = "ok"
    score_scale: str = "normalized_mean_of_0_to_5_scores"
    overall_notes: str | None = None

    @model_validator(mode="after")
    def fill_score(self) -> JudgeReport:
        if self.judge_score is None:
            self.judge_score = normalize_judge_score(self)
        return self

    @model_validator(mode="after")
    def failure_mode_tags_must_be_unique(self) -> JudgeReport:
        tags = [entry.tag for entry in self.failure_modes]
        duplicates = sorted({tag for tag in tags if tags.count(tag) > 1})
        if duplicates:
            raise ValueError(f"duplicate failure mode tags: {duplicates}")
        return self


def normalize_judge_score(report: JudgeReport) -> float:
    scores = [
        rubric.score / 5
        for artifact in report.artifacts
        for block in (
            artifact.instruction_following,
            artifact.transparency,
            artifact.analytical_rigor,
        )
        for rubric in block.values()
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _validate_rubric_keys(
    value: dict[str, JudgeRubricScore],
    expected_keys: tuple[str, ...],
) -> dict[str, JudgeRubricScore]:
    keys = set(value)
    expected = set(expected_keys)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(f"rubric keys mismatch; missing={missing}, extra={extra}")
    return value


def render_judge_prompt(
    *,
    paper_id: str,
    agent_prompt: str,
    artifact_ids: list[str],
    methods_md: str,
    numeric_grade_summary: str,
    artifact_previews: str,
    compacted_trace: str,
) -> str:
    template = prompt_template()
    replacements = {
        "paper_id": paper_id,
        "agent_prompt": agent_prompt,
        "artifact_id_list": "\n".join(f"- `{artifact_id}`" for artifact_id in artifact_ids),
        "methods_md": methods_md,
        "numeric_grade_summary": numeric_grade_summary,
        "artifact_previews": artifact_previews,
        "compacted_trace": compacted_trace,
    }
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def prompt_template() -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    actual = hashlib.sha256(template.encode()).hexdigest()
    if actual != PROMPT_TEMPLATE_SHA256:
        raise RuntimeError(
            "process judge prompt does not match the pinned PhDval-1 prompt: "
            f"expected {PROMPT_TEMPLATE_SHA256}, got {actual}"
        )
    return template


def parse_judge_response(completion: str) -> JudgeReport:
    try:
        payload = json.loads(completion)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", completion, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    return JudgeReport.model_validate(payload)


def report_error(
    *,
    paper_id: str,
    artifact_ids: list[str],
    message: str,
    status: str = "judge_failed",
) -> dict[str, Any]:
    instruction_following = {
        key: {"score": 0, "explanation": message}
        for key in INSTRUCTION_FOLLOWING_RUBRICS
    }
    transparency = {
        key: {"score": 0, "explanation": message} for key in TRANSPARENCY_RUBRICS
    }
    analytical_rigor = {
        key: {"score": 0, "explanation": message}
        for key in ANALYTICAL_RIGOR_RUBRICS
    }
    report = {
        "paper_id": paper_id,
        "artifacts": [
            {
                "artifact_id": artifact_id,
                "instruction_following": instruction_following,
                "transparency": transparency,
                "analytical_rigor": analytical_rigor,
            }
            for artifact_id in artifact_ids
        ],
        "failure_modes": [],
        "judge_score": None,
        "judge_status": status,
        "score_scale": "normalized_mean_of_0_to_5_scores",
        "overall_notes": message,
        "error": message,
    }
    return report


def preview_text(
    name: str,
    content: str,
    *,
    max_lines: int = 30,
    max_chars: int = 12_000,
) -> str:
    lines = content.splitlines()[:max_lines]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated {len(text) - max_chars} chars] ..."
    return f"## {name}\n\n```text\n{text}\n```"


def validation_error_text(error: ValidationError | Exception) -> str:
    if isinstance(error, ValidationError):
        return error.json()
    return str(error)
