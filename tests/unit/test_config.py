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

    assert len(catalog.models) == 21
    assert "google/gemini-4-argon" in catalog.models
    assert "openrouter/meta/muse-wonder" in catalog.models
    assert "openai/gpt-5.5" in catalog.models
    assert "openai/gpt-6-sol" in catalog.models
    assert "openai/gpt-6-luna" in catalog.models
    assert "openai/gpt-6-astra" in catalog.models
    assert "grok/grok-4.6" in catalog.models
    assert "openrouter/meta/muse-spark-1.3" in catalog.models
    assert "anthropic/claude-opus-4-8" in catalog.models
    assert "anthropic/claude-opus-5-5" in catalog.models
    assert "anthropic/claude-fable-5" in catalog.models
    assert "anthropic/claude-fable-5-1" in catalog.models
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
            "openai/gpt-6-luna",
            "max",
            ("--reasoning-effort", "max", "--verbosity", "high"),
        ),
        (
            "openai/gpt-6-sol",
            "max",
            ("--reasoning-effort", "max", "--verbosity", "high"),
        ),
        (
            "openai/gpt-6-astra",
            "max",
            ("--reasoning-effort", "max", "--verbosity", "high"),
        ),
        (
            "anthropic/claude-opus-5",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-opus-5-5",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-fable-5-1",
            "max",
            ("--effort", "max"),
        ),
        (
            "anthropic/claude-fable-5",
            "max",
            ("--effort", "max"),
        ),
        (
            "grok/grok-4.6",
            "xhigh",
            ("--reasoning-effort", "xhigh"),
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
            "google/gemini-4-argon",
            "high",
            ("--reasoning-effort", "high"),
        ),
        (
            "openrouter/meta/muse-wonder",
            "max",
            ("--reasoning-effort", "max"),
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


def test_claude_opus_5_5_fallback_is_opt_in() -> None:
    model = resolve_model("anthropic/claude-opus-5-5", "max")
    with_fallback = resolve_model(
        "anthropic/claude-opus-5-5",
        "max",
        fallback_model_ids=("anthropic/claude-opus-5",),
    )

    assert model.fallback_model_ids == ()
    assert with_fallback.fallback_model_ids == ("anthropic/claude-opus-5",)


def test_claude_fable_5_1_can_fallback_to_opus_5() -> None:
    model = resolve_model(
        "anthropic/claude-fable-5-1",
        "max",
        fallback_model_ids=("anthropic/claude-opus-5",),
    )

    assert model.is_reference is True
    assert model.resolved_effort == "max"
    assert model.inspect_effort_args == ("--effort", "max")
    assert model.fallback_model_ids == ("anthropic/claude-opus-5",)
    assert model.required_environment_keys == (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    )


def test_fallback_models_must_be_distinct_first_party_anthropic_models() -> None:
    with pytest.raises(ValueError, match="first-party Anthropic"):
        resolve_model(
            "openai/gpt-6-astra",
            "max",
            fallback_model_ids=("anthropic/claude-opus-5",),
        )
    with pytest.raises(ValueError, match="cannot also be"):
        resolve_model(
            "anthropic/claude-fable-5-1",
            "max",
            fallback_model_ids=("anthropic/claude-fable-5-1",),
        )


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


def test_muse_wonder_is_reference_and_pinned_to_meta() -> None:
    model = resolve_model("openrouter/meta/muse-wonder", "max")

    assert model.is_reference is True
    assert model.resolved_effort == "max"
    assert model.required_environment_keys == (
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
    )
    assert model.model_args == {
        "provider": {
            "order": ["meta"],
            "only": ["meta"],
            "allow_fallbacks": False,
        }
    }


def test_gemini_4_argon_streams_and_rejects_minimal_effort() -> None:
    model = resolve_model("google/gemini-4-argon", "max")

    assert model.is_reference is True
    assert model.required_environment_keys == ("GEMINI_API_KEY", "OPENAI_API_KEY")
    # streaming raises the per-request output cap from 32k to 64k
    assert model.model_args == {"streaming": True}
    with pytest.raises(ValueError, match="does not support effort 'minimal'"):
        resolve_model("google/gemini-4-argon", "minimal")


def test_xai_openrouter_slugs_run_direct_via_grok() -> None:
    for model_id in ("x-ai/grok-4.6", "xai/grok-4.6"):
        model = resolve_model(model_id, "max")

        assert model.model_id == "grok/grok-4.6"
        assert model.provider == "grok"
        assert model.is_reference is True
        assert model.resolved_effort == "xhigh"
        assert model.required_environment_keys == ("XAI_API_KEY", "OPENAI_API_KEY")


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
