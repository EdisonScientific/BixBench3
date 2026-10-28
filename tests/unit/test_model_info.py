from __future__ import annotations

import asyncio
import itertools

import pytest
from google.genai.types import ThinkingLevel
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
    get_model_info,
)
from inspect_ai.model._compaction._compaction import _resolve_threshold
from inspect_ai.model._providers import openai as openai_provider
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.model._providers.google import GoogleGenAIAPI
from inspect_ai.model._providers.grok import GrokAPI
from inspect_ai.tool import ToolCall

from bixbench3.config import load_reference_catalog, resolve_model
from bixbench3.inspect_harness import model_info
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
        "openai/gpt-6-sol": (922_000, 128_000),
        "openai/gpt-6-luna": (922_000, 128_000),
        "anthropic/claude-opus-4-8": (1_000_000, 128_000),
        "anthropic/claude-opus-5-5": (1_000_000, 128_000),
        "anthropic/claude-fable-5-1": (1_000_000, 128_000),
        "anthropic/claude-fable-5": (1_000_000, 128_000),
        "google/gemini-3.1-flash-lite": (1_048_576, 65_536),
        "google/gemini-3.5-flash": (1_048_576, 65_536),
        "google/gemini-3.1-pro-preview": (1_048_576, 65_536),
        "google/gemini-4-argon": (1_048_576, 65_536),
        "grok/grok-4.6": (500_000, None),
        "openrouter/meta/muse-spark-1.3": (1_048_576, None),
        "openrouter/meta/muse-wonder": (1_048_576, None),
    }
    for model_name, (input_tokens, output_tokens) in expected.items():
        info = get_model_info(model_name)
        assert info is not None
        assert info.input_tokens == input_tokens
        assert info.output_tokens == output_tokens


def test_gpt_6_metadata_and_native_max_effort() -> None:
    register_model_info()

    expected_costs = {
        "openai/gpt-6-sol": (2.0, 10.0, 2.5, 0.20),
        "openai/gpt-6-luna": (0.10, 0.50, 0.125, 0.01),
    }
    for model_name, costs in expected_costs.items():
        info = get_model_info(model_name)
        assert info is not None
        assert info.context_length == 1_050_000
        assert info.cost is not None
        assert (
            info.cost.input,
            info.cost.output,
            info.cost.input_cache_write,
            info.cost.input_cache_read,
        ) == costs

        model = get_model(model_name, api_key="probe")
        assert model.api.responses_api is True
        resolved = resolve_model(model_name, "max")
        assert resolved.resolved_effort == "max"
        assert resolved.inspect_effort_args == (
            "--reasoning-effort",
            "max",
            "--verbosity",
            "high",
        )


def test_opus_5_5_metadata_and_max_effort_request() -> None:
    register_model_info()

    info = get_model_info("anthropic/claude-opus-5-5")
    assert info is not None
    assert info.cost is not None
    assert info.cost.input == 4.0
    assert info.cost.output == 20.0
    assert info.cost.input_cache_write == 5.0
    assert info.cost.input_cache_read == 0.20

    anthropic = AnthropicAPI("claude-opus-5-5", api_key="probe")
    request, _, _, _ = anthropic.completion_config(GenerateConfig(effort="max"))
    assert request["output_config"] == {"effort": "max"}


def test_muse_wonder_metadata_and_native_max_request() -> None:
    register_model_info()

    info = get_model_info("openrouter/meta/muse-wonder")
    assert info is not None
    assert info.snapshot == "20261012"
    assert info.cost is not None
    assert (
        info.cost.input,
        info.cost.output,
        info.cost.input_cache_write,
        info.cost.input_cache_read,
    ) == (4.0, 20.0, 4.0, 0.20)

    resolved = resolve_model("openrouter/meta/muse-wonder", "max")
    model = get_model(resolved.model_id, api_key="probe", **resolved.model_args)
    params = model.api.completion_params(GenerateConfig(reasoning_effort="max"), True)
    assert params["extra_body"]["reasoning"] == {"effort": "max"}
    assert params["extra_body"]["provider"] == resolved.model_args["provider"]


def test_openrouter_native_max_is_limited_to_models_that_accept_it() -> None:
    register_model_info()

    for model_name in ("openrouter/meta/muse-spark-1.3", "openrouter/moonshotai/kimi-k3"):
        api = get_model(model_name, api_key="probe").api
        params = api.completion_params(GenerateConfig(reasoning_effort="max"), True)
        assert params["extra_body"]["reasoning"] == {"effort": "max"}, model_name
    glm = get_model("openrouter/z-ai/glm-5.2", api_key="probe").api
    params = glm.completion_params(GenerateConfig(reasoning_effort="max"), True)
    assert params["extra_body"]["reasoning"] == {"effort": "xhigh"}
    muse = get_model("openrouter/meta/muse-wonder", api_key="probe").api
    params = muse.completion_params(GenerateConfig(reasoning_effort="high"), True)
    assert params["extra_body"]["reasoning"] == {"effort": "high"}


def test_gemini_4_argon_metadata_and_high_thinking_request() -> None:
    register_model_info()

    info = get_model_info("google/gemini-4-argon")
    assert info is not None
    assert info.cost is not None
    assert (
        info.cost.input,
        info.cost.output,
        info.cost.input_cache_write,
        info.cost.input_cache_read,
    ) == (2.0, 10.0, 2.0, 0.10)

    resolved = resolve_model("google/gemini-4-argon", "max")
    model = get_model(resolved.model_id, api_key="probe", **resolved.model_args)
    assert model.api.streaming is True
    thinking = model.api.chat_thinking_config(
        GenerateConfig(reasoning_effort=resolved.resolved_effort)
    )
    assert thinking is not None
    assert thinking.thinking_level == ThinkingLevel.HIGH
    assert thinking.thinking_budget is None


def test_gemini_token_counts_anchor_compaction_fragments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    register_model_info()
    counted: list[list[ChatMessage]] = []

    async def strict_count_tokens(
        self: GoogleGenAIAPI,
        input: str | list[ChatMessage],
        config: GenerateConfig | None = None,
    ) -> int:
        del self, config
        assert isinstance(input, list)
        # mirrors gemini 4: no leading model/function-response turn, no trailing model turn
        assert isinstance(input[0], ChatMessageUser)
        assert not isinstance(input[-1], ChatMessageAssistant)
        for previous, message in itertools.pairwise(input):
            if isinstance(message, ChatMessageTool):
                assert isinstance(previous, (ChatMessageAssistant, ChatMessageTool))
        counted.append(input)
        return 10 * len(input)

    monkeypatch.setattr(model_info, "_inspect_google_count_tokens", strict_count_tokens)
    monkeypatch.setattr(model_info, "_GEMINI_ANCHOR_TOKENS", {})
    gemini = get_model("google/gemini-4-argon", api_key="probe").api
    call = ToolCall(id="call_1", function="bash_session", arguments={"input": "ls"})
    assistant = ChatMessageAssistant(content="", tool_calls=[call])
    tool = ChatMessageTool(content="data", tool_call_id="call_1", function="bash_session")

    assert asyncio.run(gemini.count_tokens([ChatMessageUser(content="go"), assistant, tool])) == 30
    assert asyncio.run(gemini.count_tokens([assistant, tool])) == 20
    assert asyncio.run(gemini.count_tokens([tool])) == 20
    assert asyncio.run(gemini.count_tokens([tool, assistant])) == 40

    anchored_tool_call = counted[-1][1]
    assert isinstance(anchored_tool_call, ChatMessageAssistant)
    assert anchored_tool_call.tool_calls is not None
    assert [c.id for c in anchored_tool_call.tool_calls] == ["call_1"]
    assert getattr(GoogleGenAIAPI.count_tokens, "_bixbench3_fragments", False) is True


def test_gpt6_and_grok46_effort_routing() -> None:
    register_model_info()

    assert openai_provider.is_gpt_5_model("gpt-6-astra") is True
    assert resolve_model("openai/gpt-6-astra", "max").model_args["responses_api"] is True
    grok = GrokAPI("grok-4.6", api_key="probe")
    params = grok._grok_params(GenerateConfig(reasoning_effort="xhigh"))
    assert params["reasoning_effort"] == "xhigh"


def test_grok_token_counts_use_tiktoken_not_xai_tokenize() -> None:
    register_model_info()

    grok = GrokAPI("grok-4.6", api_key="probe")
    counted = asyncio.run(grok.count_text_tokens("hello world"))
    assert counted >= 1
    assert getattr(GrokAPI.count_text_tokens, "_bixbench3_tiktoken", False) is True


def test_compaction_uses_full_context_for_current_model_aliases() -> None:
    register_model_info()

    expected_thresholds = {
        "openai/gpt-6-sol": 829_800,
        "openai/gpt-6-luna": 829_800,
        "anthropic/claude-opus-4-8": 900_000,
        "anthropic/claude-opus-5-5": 900_000,
        "anthropic/claude-fable-5-1": 900_000,
        "anthropic/claude-fable-5": 900_000,
        "anthropic/claude-sonnet-4-6": 900_000,
        "google/gemini-3.5-flash": 943_718,
        "google/gemini-4-argon": 943_718,
        "grok/grok-4.6": 450_000,
        "openrouter/meta/muse-spark-1.3": 943_718,
        "openrouter/meta/muse-wonder": 943_718,
    }
    for model_name, expected in expected_thresholds.items():
        model = get_model(model_name, api_key="probe")
        assert _resolve_threshold(model, 0.9) == expected
