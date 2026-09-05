from __future__ import annotations

from inspect_ai.model import GenerateConfig, get_model, get_model_info
from inspect_ai.model._compaction._compaction import _resolve_threshold
from inspect_ai.model._providers import openai as openai_provider
from inspect_ai.model._providers.grok import GrokAPI

from bixbench3.config import load_reference_catalog, resolve_model
from bixbench3.inspect_harness.model_info import register_model_info


def test_reference_models_have_explicit_context_and_cost_metadata() -> None:
    register_model_info()

    for model_name in load_reference_catalog().models:
        info = get_model_info(model_name)
        assert info is not None, model_name
        assert info.input_tokens is not None, model_name
        assert info.cost is not None, model_name


def test_long_context_overrides_replace_stale_inspect_metadata() -> None:
    register_model_info()

    expected = {
        "anthropic/claude-opus-4-8": (1_000_000, 128_000),
        "anthropic/claude-fable-5": (1_000_000, 128_000),
        "google/gemini-3.1-flash-lite": (1_048_576, 65_536),
        "google/gemini-3.5-flash": (1_048_576, 65_536),
        "google/gemini-3.1-pro-preview": (1_048_576, 65_536),
        "grok/grok-4.6": (500_000, None),
        "openrouter/meta/muse-spark-1.3": (1_048_576, None),
    }
    for model_name, (input_tokens, output_tokens) in expected.items():
        info = get_model_info(model_name)
        assert info is not None
        assert info.input_tokens == input_tokens
        assert info.output_tokens == output_tokens


def test_gpt6_and_grok46_effort_routing() -> None:
    register_model_info()

    assert openai_provider.is_gpt_5_model("gpt-6-astra") is True
    assert resolve_model("openai/gpt-6-astra", "max").model_args["responses_api"] is True
    grok = GrokAPI("grok-4.6", api_key="probe")
    params = grok._grok_params(GenerateConfig(reasoning_effort="xhigh"))
    assert params["reasoning_effort"] == "xhigh"


def test_compaction_uses_full_context_for_current_model_aliases() -> None:
    register_model_info()

    expected_thresholds = {
        "anthropic/claude-opus-4-8": 900_000,
        "anthropic/claude-fable-5": 900_000,
        "anthropic/claude-sonnet-4-6": 900_000,
        "google/gemini-3.5-flash": 943_718,
        "grok/grok-4.6": 450_000,
        "openrouter/meta/muse-spark-1.3": 943_718,
    }
    for model_name, expected in expected_thresholds.items():
        model = get_model(model_name, api_key="probe")
        assert _resolve_threshold(model, 0.9) == expected
