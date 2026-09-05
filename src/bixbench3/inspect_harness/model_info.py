from __future__ import annotations

from inspect_ai.model import ModelCost, ModelInfo, set_model_info
from inspect_ai.model import _openai as openai_helpers
from inspect_ai.model._openai import is_gpt_5_model as _inspect_is_gpt_5_model
from inspect_ai.model._providers import openai as openai_provider
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.model._providers.grok import GrokAPI

_inspect_anthropic_input_tokens_name = AnthropicAPI.input_tokens_name
_inspect_grok_params = GrokAPI._grok_params


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
    _enable_anthropic_long_context()
    _enable_gpt6_inspect_routing()
    _enable_grok46_xhigh()


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
