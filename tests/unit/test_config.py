from __future__ import annotations

import json

import pytest

from bixbench3.config import (
    load_reference_catalog,
    normalize_effort,
    resolve_model,
)


def test_reference_catalog_contains_reported_models() -> None:
    catalog = load_reference_catalog()

    assert len(catalog.models) == 12
    assert "openai/gpt-5.5" in catalog.models
    assert "anthropic/claude-opus-4-8" in catalog.models
    assert "anthropic/claude-fable-5" in catalog.models
    assert "openrouter/z-ai/glm-5.2" in catalog.models
    assert "openrouter/moonshotai/kimi-k3" in catalog.models


@pytest.mark.parametrize(
    ("model_id", "resolved", "flags"),
    [
        (
            "openai/gpt-5.5",
            "xhigh",
            ("--reasoning-effort", "xhigh", "--verbosity", "high"),
        ),
        (
            "openai/gpt-5.6-luna",
            "max",
            ("--reasoning-effort", "max", "--verbosity", "high"),
        ),
        (
            "anthropic/claude-opus-5",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-fable-5",
            "max",
            ("--effort", "max"),
        ),
        (
            "xai/grok-4.5",
            "high",
            ("--reasoning-effort", "high"),
        ),
        (
            "openrouter/deepseek/deepseek-v4-flash",
            "xhigh",
            ("--reasoning-effort", "xhigh"),
        ),
        (
            "openrouter/moonshotai/kimi-k3",
            "max",
            ("--reasoning-effort", "max"),
        ),
        (
            "google/gemini-3.5-flash",
            "high",
            ("--reasoning-effort", "high"),
        ),
        (
            "openrouter/z-ai/glm-5.2",
            "xhigh",
            ("--reasoning-effort", "xhigh"),
        ),
        (
            "anthropic/claude-opus-4-8",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-sonnet-4-6",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-haiku-4-5",
            "high",
            ("--reasoning-tokens", "63999"),
        ),
    ],
)
def test_max_effort_compatibility(
    model_id: str,
    resolved: str,
    flags: tuple[str, ...],
) -> None:
    model = resolve_model(model_id, "max")

    assert model.requested_effort == "max"
    assert model.resolved_effort == resolved
    assert model.inspect_effort_args == flags


def test_non_reference_inspect_model_is_accepted() -> None:
    model = resolve_model(
        "together/deep-model",
        "maximum",
        user_model_args={"temperature": 0.2},
    )

    assert model.is_reference is False
    assert model.requested_effort == "max"
    assert model.resolved_effort == "max"
    assert model.model_args == {"temperature": 0.2}
    assert model.required_environment_keys == (
        "TOGETHER_API_KEY",
        "OPENAI_API_KEY",
    )


def test_non_openai_model_requires_agent_and_adjudicator_keys() -> None:
    model = resolve_model("anthropic/claude-opus-4-8", "high")

    assert model.required_environment_keys == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    )


def test_reference_model_args_are_applied_and_can_be_overridden() -> None:
    default = resolve_model("openrouter/z-ai/glm-5.2", "high")
    overridden = resolve_model(
        "openrouter/z-ai/glm-5.2",
        "high",
        user_model_args={"provider": {"only": ["fireworks"]}},
    )

    assert default.model_args["provider"] != overridden.model_args["provider"]
    assert overridden.model_args == {"provider": {"only": ["fireworks"]}}


def test_claude_fable_5_is_reference_without_anthropic_fallbacks() -> None:
    model = resolve_model("anthropic/claude-fable-5", "max")

    assert model.is_reference is True
    assert model.resolved_effort == "max"
    assert model.inspect_effort_args == ("--effort", "max")
    assert model.required_environment_keys == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    )
    # anthropic classifier fallbacks are opt-in via request `fallbacks`; omit them
    # so a content restriction stays a refusal instead of retrying another model
    assert model.model_args == {}


def test_kimi_k3_is_reference_and_pinned_to_moonshotai() -> None:
    model = resolve_model("openrouter/moonshotai/kimi-k3", "max")

    assert model.is_reference is True
    assert model.required_environment_keys == (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    )
    assert model.model_args == {
        "provider": {
            "order": ["moonshotai"],
            "only": ["moonshotai"],
            "allow_fallbacks": False,
        }
    }


def test_effort_must_be_explicit_and_known() -> None:
    assert normalize_effort("extra-high") == "xhigh"
    with pytest.raises(ValueError, match="unsupported effort"):
        normalize_effort("")


def test_catalog_validation_rejects_duplicate_models(tmp_path) -> None:
    catalog = load_reference_catalog()
    profile = next(iter(catalog.models.values()))
    value = {
        "schema_version": 1,
        "adjudicator": {
            "model": "openai/gpt-5.5",
            "provider_env": "OPENAI_API_KEY",
            "reasoning_effort": "low",
        },
        "models": [
            {
                "id": profile.model_id,
                "provider": profile.provider,
                "provider_env": profile.provider_env,
                "version": profile.version,
                "supported_efforts": ["high"],
                "model_args": {},
            },
            {
                "id": profile.model_id,
                "provider": profile.provider,
                "provider_env": profile.provider_env,
                "version": profile.version,
                "supported_efforts": ["high"],
                "model_args": {},
            },
        ],
    }
    path = tmp_path / "reference_models.yaml"
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate model id"):
        load_reference_catalog(path)
