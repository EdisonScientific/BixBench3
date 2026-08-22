from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from .policy import NetworkPolicy

PROMPT_PATH = Path(__file__).parent / "prompts" / "adjudicator_prompt.md"

Decision = Literal["allow", "deny"]
RiskType = Literal[
    "method_code",
    "reference_data",
    "documentation",
    "package_index",
    "answer_artifact",
    "paper_or_supplement",
    "unknown",
]
ReasonCode = Literal[
    "approved",
    "denied_protected_source",
    "denied_processed_output",
    "denied_search_query",
    "denied_scope_too_broad",
    "denied_over_budget",
    "denied_other",
]


class AdjudicatorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Decision
    risk_type: RiskType
    reason: str
    url: str
    reason_code: ReasonCode
    granted_scope: str | None

    @model_validator(mode="after")
    def validate_decision(self) -> AdjudicatorDecision:
        if self.decision == "allow":
            if self.reason_code != "approved":
                raise ValueError("allowed decisions must use reason_code=approved")
            if not self.granted_scope:
                raise ValueError("allowed decisions must include granted_scope")
        else:
            if self.reason_code == "approved":
                raise ValueError("denied decisions cannot use reason_code=approved")
            if self.granted_scope is not None:
                raise ValueError("denied decisions must set granted_scope to null")
        return self

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"


class Adjudicator:
    def __init__(self, policy: NetworkPolicy) -> None:
        self.policy = policy
        self.client = OpenAI(timeout=float(policy.decision_timeout_seconds))

    def decide(self, *, url: str, reason: str) -> AdjudicatorDecision:
        template = PROMPT_PATH.read_text(encoding="utf-8")
        agent_prompt = self.policy.payload.get("agent_prompt")
        substitutions = {
            "{task_id}": _display_value(self.policy.payload.get("task_id")),
            "{agent_prompt}": _display_value(agent_prompt)
            or "(agent task prompt was not provided to the adjudicator)",
            "{answer_source_urls}": json.dumps(self.policy.answer_source_urls[:200], indent=2),
            "{answer_source_domains}": json.dumps(
                self.policy.answer_source_domains[:200],
                indent=2,
            ),
            "{requested_url}": url,
            "{agent_reason}": reason,
            "{policy_notes}": self.policy.notes,
        }
        prompt = template
        for placeholder, value in substitutions.items():
            prompt = prompt.replace(placeholder, value)
        return self._generate_decision(prompt)

    def _generate_decision(self, prompt: str) -> AdjudicatorDecision:
        response = self.client.responses.create(
            model=self.policy.adjudicator_model.split("/", 1)[-1],
            input=prompt,
            reasoning={"effort": "low"},
        )
        try:
            return parse_adjudicator_response(response.output_text or "")
        except (json.JSONDecodeError, ValidationError) as error:
            retry = self.client.responses.create(
                model=self.policy.adjudicator_model.split("/", 1)[-1],
                input=(
                    prompt
                    + "\n\n# Previous Response Validation Error\n\n"
                    + str(error)
                    + "\n\nReturn a corrected JSON object only."
                ),
                reasoning={"effort": "low"},
            )
            try:
                return parse_adjudicator_response(retry.output_text or "")
            except (json.JSONDecodeError, ValidationError) as retry_error:
                raise RuntimeError(f"invalid adjudicator response: {retry_error}") from retry_error


def parse_adjudicator_response(completion: str) -> AdjudicatorDecision:
    try:
        payload = json.loads(completion)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", completion, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    return AdjudicatorDecision.model_validate(payload)


def _display_value(value: object) -> str:
    return value if isinstance(value, str) else ""
