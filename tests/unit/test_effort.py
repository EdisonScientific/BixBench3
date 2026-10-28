from __future__ import annotations

import pytest

from bixbench3.grading.effort import (
    MAX_HAIKU_4_5_REASONING_TOKENS,
    normalize_effort_alias,
    resolve_reasoning_effort,
    resolve_reasoning_tokens,
)


@pytest.mark.parametrize(
    ("model", "requested", "resolved"),
    [
        ("openai/gpt-5.5", "max", "xhigh"),
        ("openai/gpt-5.6-luna", "max", "max"),
        ("gpt-5.6-sol", "maximum", "max"),
        ("openai/gpt-6-astra", "max", "max"),
        ("gpt-6-astra", "maximum", "max"),
        ("gpt-5.4-mini", "maximum", "xhigh"),
        ("anthropic/claude-opus-5", "max", "max"),
        ("anthropic/claude-opus-5", "xhigh", "xhigh"),
        ("anthropic/claude-fable-5", "max", "max"),
        ("anthropic/claude-fable-5", "xhigh", "xhigh"),
        ("google/gemini-3.5-flash", "max", "high"),
        ("google/gemini-4-argon", "max", "high"),
        ("openrouter/meta/muse-wonder", "max", "max"),
        ("openrouter/deepseek/deepseek-v4-flash", "max", "xhigh"),
        ("google/gemini-3-flash-preview", "extra-high", "high"),
        ("openrouter/z-ai/glm-5.2", "max", "xhigh"),
        ("z-ai/glm-5.2", "max", "xhigh"),
        ("xai/grok-4.5", "max", "high"),
        ("openrouter/x-ai/grok-4.5", "xhigh", "high"),
        ("grok/grok-4.6", "max", "xhigh"),
        ("grok/grok-4.6", "xhigh", "xhigh"),
        ("openrouter/moonshotai/kimi-k3", "max", "max"),
        ("anthropic/claude-opus-4-8", "max", "max"),
        ("anthropic/claude-opus-4-8", "extra-high", "xhigh"),
        ("anthropic/claude-sonnet-4-6", "max", "max"),
        ("anthropic/claude-opus-4-7", "extra-high", "high"),
        ("anthropic/claude-haiku-4-5", "max", "high"),
        ("anthropic/claude-unknown", "max", "high"),
        ("openrouter/example/model", "max", "max"),
    ],
)
def test_resolve_reasoning_effort(
    model: str,
    requested: str,
    resolved: str,
) -> None:
    assert resolve_reasoning_effort(model, requested) == resolved


def test_openai_max_resolution_drives_high_caller_verbosity() -> None:
    resolved = resolve_reasoning_effort("openai/gpt-5.5", "max")
    caller_verbosity = "high" if resolved in {"high", "xhigh"} else "medium"

    assert (resolved, caller_verbosity) == ("xhigh", "high")


def test_haiku_max_uses_fixed_reasoning_token_budget() -> None:
    assert MAX_HAIKU_4_5_REASONING_TOKENS == 63_999
    assert (
        resolve_reasoning_tokens("anthropic/claude-haiku-4-5", "max")
        == MAX_HAIKU_4_5_REASONING_TOKENS
    )
    assert resolve_reasoning_tokens("anthropic/claude-haiku-4-5", "high") is None
    assert resolve_reasoning_tokens("anthropic/claude-opus-4-8", "max") is None


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("maximum", "max"),
        ("extra-high", "xhigh"),
        ("extra high", "xhigh"),
        ("x-high", "xhigh"),
        ("medium", "medium"),
    ],
)
def test_normalize_effort_alias(raw: str, normalized: str) -> None:
    assert normalize_effort_alias(raw) == normalized
