from __future__ import annotations

import base64
from typing import Any

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
    ModelCost,
    ModelInfo,
    set_model_info,
)
from inspect_ai.model import _openai as openai_helpers
from inspect_ai.model._openai import is_gpt_5_model as _inspect_is_gpt_5_model
from inspect_ai.model._providers import openai as openai_provider
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.model._providers.google import GoogleGenAIAPI
from inspect_ai.model._providers.grok import GrokAPI
from inspect_ai.model._providers.openrouter import OpenRouterAPI
from inspect_ai.model._tokens import count_text_tokens as _tiktoken_count_text_tokens
from inspect_ai.tool import ToolCall

_inspect_anthropic_input_tokens_name = AnthropicAPI.input_tokens_name
_inspect_google_count_tokens = GoogleGenAIAPI.count_tokens
_inspect_grok_params = GrokAPI._grok_params
_inspect_openrouter_completion_params = OpenRouterAPI.completion_params
_GEMINI_COUNT_ANCHOR = "."
_GEMINI_SKIP_SIGNATURE = base64.b64encode(b"skip_thought_signature_validator").decode()
_GEMINI_ANCHOR_TOKENS: dict[str, int] = {}
_OPENROUTER_NATIVE_MAX_MODELS = frozenset(
    {
        "meta/muse-wonder",
        "meta/muse-spark-1.3",
        "moonshotai/kimi-k3",
    }
)


def register_model_info() -> None:
    _register(
        ("openai/gpt-5.4-nano",),
        organization="OpenAI",
        model="GPT-5.4 Nano",
        context_length=400_000,
        output_tokens=128_000,
        input_tokens=272_000,
        cost=_cost(0.20, 1.25, 0.02, cache_write=0.25),
    )
    _register(
        ("openai/gpt-5.4-mini",),
        organization="OpenAI",
        model="GPT-5.4 Mini",
        context_length=400_000,
        output_tokens=128_000,
        input_tokens=272_000,
        cost=_cost(0.75, 4.50, 0.075, cache_write=0.9375),
    )
    _register(
        ("openai/gpt-5.5",),
        organization="OpenAI",
        model="GPT-5.5",
        context_length=1_050_000,
        output_tokens=128_000,
        input_tokens=922_000,
        cost=_cost(5.0, 30.0, 0.5, cache_write=6.25),
    )
    _register(
        ("openai/gpt-6-sol",),
        organization="OpenAI",
        model="GPT-6 Sol",
        context_length=1_050_000,
        output_tokens=128_000,
        input_tokens=922_000,
        cost=_cost(2.0, 10.0, 0.20, cache_write=2.5),
    )
    _register(
        ("openai/gpt-6-luna",),
        organization="OpenAI",
        model="GPT-6 Luna",
        context_length=1_050_000,
        output_tokens=128_000,
        input_tokens=922_000,
        cost=_cost(0.10, 0.50, 0.01, cache_write=0.125),
    )
    _register(
        ("anthropic/claude-haiku-4-5",),
        organization="Anthropic",
        model="Claude Haiku 4.5",
        context_length=200_000,
        output_tokens=64_000,
        cost=_cost(1.0, 5.0, 0.10, cache_write=1.25),
    )
    _register(
        ("anthropic/claude-sonnet-4-6",),
        organization="Anthropic",
        model="Claude Sonnet 4.6",
        context_length=1_000_000,
        output_tokens=128_000,
        cost=_cost(3.0, 15.0, 0.30, cache_write=3.75),
    )
    _register(
        ("anthropic/claude-opus-4-8",),
        organization="Anthropic",
        model="Claude Opus 4.8",
        context_length=1_000_000,
        output_tokens=128_000,
        cost=_cost(5.0, 25.0, 0.50, cache_write=6.25),
    )
    _register(
        ("anthropic/claude-opus-5-5",),
        organization="Anthropic",
        model="Claude Opus 5.5",
        context_length=1_000_000,
        output_tokens=128_000,
        cost=_cost(4.0, 20.0, 0.20, cache_write=5.0),
    )
    _register(
        ("anthropic/claude-fable-5-1",),
        organization="Anthropic",
        model="Claude Fable 5.1",
        context_length=1_000_000,
        output_tokens=128_000,
        cost=_cost(10.0, 50.0, 0.25, cache_write=12.5),
    )
    _register(
        ("anthropic/claude-fable-5",),
        organization="Anthropic",
        model="Claude Fable 5",
        context_length=1_000_000,
        output_tokens=128_000,
        cost=_cost(10.0, 50.0, 1.0, cache_write=12.5),
    )
    _register(
        ("google/gemini-3.1-flash-lite",),
        organization="Google",
        model="Gemini 3.1 Flash-Lite",
        context_length=1_048_576,
        output_tokens=65_536,
        cost=_cost(0.25, 1.50, 0.025),
    )
    _register(
        ("google/gemini-3.5-flash",),
        organization="Google",
        model="Gemini 3.5 Flash",
        context_length=1_048_576,
        output_tokens=65_536,
        cost=_cost(1.50, 9.0, 0.15),
    )
    _register(
        ("google/gemini-3.1-pro-preview",),
        organization="Google",
        model="Gemini 3.1 Pro Preview",
        context_length=1_048_576,
        output_tokens=65_536,
        cost=_cost(2.0, 12.0, 0.20),
    )
    # introductory pricing through 2026-11-30; $4 / $20 / $0.20 thereafter
    _register(
        ("google/gemini-4-argon",),
        organization="Google",
        model="Gemini 4 Argon",
        context_length=1_048_576,
        output_tokens=65_536,
        cost=_cost(2.0, 10.0, 0.10),
    )
    _register(
        ("z-ai/glm-5.2", "openrouter/z-ai/glm-5.2"),
        organization="Z.ai",
        model="GLM 5.2",
        context_length=1_048_576,
        output_tokens=131_072,
        cost=_cost(1.40, 4.40, 0.14),
    )
    _register(
        ("moonshotai/kimi-k3", "openrouter/moonshotai/kimi-k3"),
        organization="Moonshot AI",
        model="Kimi K3",
        context_length=1_048_576,
        cost=_cost(3.0, 15.0, 0.30),
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
    set_model_info(
        "grok/grok-4.6",
        ModelInfo(
            organization="xAI",
            model="Grok 4.6",
            context_length=500_000,
            reasoning=True,
            cost=ModelCost(
                input=2.0,
                output=6.0,
                input_cache_write=2.0,
                input_cache_read=0.5,
            ),
        ),
    )
    muse = ModelInfo(
        organization="Meta",
        model="Muse Spark 1.3",
        snapshot="20260902",
        context_length=1_048_576,
        reasoning=True,
        cost=ModelCost(
            input=1.25,
            output=4.25,
            input_cache_write=1.25,
            input_cache_read=0.15,
        ),
    )
    set_model_info("meta/muse-spark-1.3", muse)
    set_model_info("openrouter/meta/muse-spark-1.3", muse)
    muse_wonder = ModelInfo(
        organization="Meta",
        model="Muse Wonder",
        snapshot="20261012",
        context_length=1_048_576,
        reasoning=True,
        cost=ModelCost(
            input=4.0,
            output=20.0,
            input_cache_write=4.0,
            input_cache_read=0.20,
        ),
    )
    set_model_info("meta/muse-wonder", muse_wonder)
    set_model_info("openrouter/meta/muse-wonder", muse_wonder)
    _enable_anthropic_long_context()
    _enable_gpt6_inspect_routing()
    _enable_grok46_xhigh()
    _disable_grok_tokenize_api()
    _enable_openrouter_native_max()
    _enable_gemini_fragment_token_counts()


def _cost(
    input_price: float,
    output_price: float,
    cache_read: float,
    *,
    cache_write: float | None = None,
) -> ModelCost:
    return ModelCost(
        input=input_price,
        output=output_price,
        input_cache_write=cache_write if cache_write is not None else input_price,
        input_cache_read=cache_read,
    )


def _register(
    names: tuple[str, ...],
    *,
    organization: str,
    model: str,
    context_length: int,
    cost: ModelCost,
    output_tokens: int | None = None,
    input_tokens: int | None = None,
) -> None:
    info = ModelInfo(
        organization=organization,
        model=model,
        context_length=context_length,
        output_tokens=output_tokens,
        _input_tokens=input_tokens,
        reasoning=True,
        cost=cost,
    )
    for name in names:
        set_model_info(name, info)


def _enable_gpt6_inspect_routing() -> None:
    # inspect only auto-enables the Responses API and reasoning.effort for
    # names containing "gpt-5". treat gpt-6 the same so max effort is sent
    provider_model_check = openai_provider.is_gpt_5_model  # type: ignore[attr-defined]
    if getattr(provider_model_check, "_bixbench3_gpt6", False):
        return

    def is_gpt_5_or_6(model_name: str) -> bool:
        return _inspect_is_gpt_5_model(model_name) or "gpt-6" in model_name.lower()

    is_gpt_5_or_6._bixbench3_gpt6 = True  # type: ignore[attr-defined]
    openai_helpers.is_gpt_5_model = is_gpt_5_or_6
    openai_provider.is_gpt_5_model = is_gpt_5_or_6  # type: ignore[attr-defined]


def _enable_anthropic_long_context() -> None:
    # inspect 0.3.220 routes unknown current Claude aliases through Haiku's
    # 200k metadata. these models have a standard 1m context without a beta.
    if getattr(AnthropicAPI.input_tokens_name, "_bixbench3_claude_1m", False):
        return

    def input_tokens_name(self: AnthropicAPI) -> str:
        if self.service_model_name() in {
            "claude-sonnet-4-6",
            "claude-opus-4-8",
            "claude-fable-5",
        }:
            return self.canonical_name()
        return _inspect_anthropic_input_tokens_name(self)

    input_tokens_name._bixbench3_claude_1m = True  # type: ignore[attr-defined]
    AnthropicAPI.input_tokens_name = input_tokens_name  # type: ignore[method-assign]


def _enable_grok46_xhigh() -> None:
    # inspect 0.3.220 clamps every grok-4 xhigh request to high, but grok 4.6
    # supports xhigh natively
    if getattr(GrokAPI._grok_params, "_bixbench3_grok46", False):
        return

    def grok_params(self: GrokAPI, config: object) -> dict[str, object]:
        params = _inspect_grok_params(self, config)  # type: ignore[arg-type]
        if self.model_name == "grok-4.6" and getattr(config, "reasoning_effort", None) == "xhigh":
            params["reasoning_effort"] = "xhigh"
        return params

    grok_params._bixbench3_grok46 = True  # type: ignore[attr-defined]
    GrokAPI._grok_params = grok_params  # type: ignore[method-assign]


def _disable_grok_tokenize_api() -> None:
    # inspect's grok provider counts tokens via xAI's tokenize RPC. many keys
    # can chat but cannot tokenize; compaction only needs an estimate.
    if getattr(GrokAPI.count_text_tokens, "_bixbench3_tiktoken", False):
        return

    async def count_text_tokens(self: GrokAPI, text: str) -> int:
        del self
        return _tiktoken_count_text_tokens(text)

    count_text_tokens._bixbench3_tiktoken = True  # type: ignore[attr-defined]
    GrokAPI.count_text_tokens = count_text_tokens  # type: ignore[method-assign]


def _enable_openrouter_native_max() -> None:
    # inspect clamps every openrouter max request to xhigh, but these models
    # accept max natively (kimi-k3 does not list xhigh at all)
    if getattr(OpenRouterAPI.completion_params, "_bixbench3_native_max", False):
        return

    def completion_params(
        self: OpenRouterAPI,
        config: GenerateConfig,
        tools: bool,
    ) -> dict[str, Any]:
        params = _inspect_openrouter_completion_params(self, config, tools)
        model_name = self.model_name.removeprefix("openrouter/")
        if config.reasoning_effort == "max" and model_name in _OPENROUTER_NATIVE_MAX_MODELS:
            params["extra_body"]["reasoning"]["effort"] = "max"
        return params

    completion_params._bixbench3_native_max = True  # type: ignore[attr-defined]
    OpenRouterAPI.completion_params = completion_params  # type: ignore[method-assign]


def _enable_gemini_fragment_token_counts() -> None:
    # gemini 4 count_tokens enforces generation turn order (no leading model or
    # function response turn, no trailing model turn), but inspect compaction
    # counts the slice of messages added since its last baseline
    if getattr(GoogleGenAIAPI.count_tokens, "_bixbench3_fragments", False):
        return

    async def count_tokens(
        self: GoogleGenAIAPI,
        input: str | list[ChatMessage],
        config: GenerateConfig | None = None,
    ) -> int:
        if isinstance(input, str) or not input:
            return await _inspect_google_count_tokens(self, input, config)
        messages = _gemini_countable_messages(input)
        if len(messages) == len(input):
            return await _inspect_google_count_tokens(self, input, config)
        anchor_tokens = _GEMINI_ANCHOR_TOKENS.get(self.model_name)
        if anchor_tokens is None:
            anchor_tokens = await _inspect_google_count_tokens(
                self, [ChatMessageUser(content=_GEMINI_COUNT_ANCHOR)], config
            )
            _GEMINI_ANCHOR_TOKENS[self.model_name] = anchor_tokens
        total = await _inspect_google_count_tokens(self, messages, config)
        return max(total - anchor_tokens, 0)

    count_tokens._bixbench3_fragments = True  # type: ignore[attr-defined]
    GoogleGenAIAPI.count_tokens = count_tokens  # type: ignore[method-assign]


def _gemini_countable_messages(messages: list[ChatMessage]) -> list[ChatMessage]:
    first = messages[0]
    prefix: list[ChatMessage] = []
    if isinstance(first, ChatMessageTool):
        leading_tools: list[ChatMessageTool] = []
        for message in messages:
            if not isinstance(message, ChatMessageTool):
                break
            leading_tools.append(message)
        prefix = [
            ChatMessageUser(content=_GEMINI_COUNT_ANCHOR),
            ChatMessageAssistant(
                content=[
                    ContentReasoning(
                        reasoning=_GEMINI_SKIP_SIGNATURE if index == 0 else "",
                        redacted=True,
                        internal={"gemini_function_call_id": tool.tool_call_id or ""},
                    )
                    for index, tool in enumerate(leading_tools)
                ],
                tool_calls=[
                    ToolCall(
                        id=tool.tool_call_id or "",
                        function=tool.function or "tool",
                        arguments={},
                    )
                    for tool in leading_tools
                ],
            ),
        ]
    elif isinstance(first, ChatMessageAssistant):
        prefix = [ChatMessageUser(content=_GEMINI_COUNT_ANCHOR)]
    suffix: list[ChatMessage] = []
    if isinstance(messages[-1], ChatMessageAssistant):
        suffix = [ChatMessageUser(content=_GEMINI_COUNT_ANCHOR)]
    return [*prefix, *messages, *suffix]
