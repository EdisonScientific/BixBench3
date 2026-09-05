from __future__ import annotations

from inspect_ai.model import ModelCost, ModelInfo, set_model_info
from inspect_ai.model import _openai as openai_helpers
from inspect_ai.model._openai import is_gpt_5_model as _inspect_is_gpt_5_model
from inspect_ai.model._providers import openai as openai_provider


def register_model_info() -> None:
    set_model_info(
        "z-ai/glm-5.2",
        ModelInfo(
            organization="Z.ai",
            model="GLM 5.2",
            context_length=1_048_576,
            reasoning=True,
        ),
    )
    # inspect 0.3.220's openai.yml stops at gpt-5.5. without this, compaction
    # falls back to a 128k window and usage.total_cost stays empty
    astra = ModelInfo(
        organization="OpenAI",
        model="GPT-6 Astra",
        context_length=1_050_000,
        output_tokens=128_000,
        _input_tokens=922_000,
        reasoning=True,
        cost=ModelCost(
            input=10.0,
            output=50.0,
            input_cache_write=12.5,
            input_cache_read=1.0,
        ),
    )
    set_model_info("openai/gpt-6-astra", astra)
    set_model_info("gpt-6-astra", astra)
    _enable_gpt6_inspect_routing()


def _enable_gpt6_inspect_routing() -> None:
    # inspect only auto-enables the Responses API and reasoning.effort for
    # names containing "gpt-5". treat gpt-6 the same so max effort is sent
    if getattr(_inspect_is_gpt_5_model, "_bixbench3_gpt6", False):
        return

    def is_gpt_5_or_6(model_name: str) -> bool:
        return _inspect_is_gpt_5_model(model_name) or "gpt-6" in model_name.lower()

    is_gpt_5_or_6._bixbench3_gpt6 = True  # type: ignore[attr-defined]
    openai_helpers.is_gpt_5_model = is_gpt_5_or_6
    openai_provider.is_gpt_5_model = is_gpt_5_or_6
