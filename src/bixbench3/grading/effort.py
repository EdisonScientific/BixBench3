from __future__ import annotations

MAX_HAIKU_4_5_REASONING_TOKENS = 63_999


def resolve_reasoning_effort(model_name: str, effort_name: str) -> str:
    normalized = normalize_effort_alias(effort_name)
    lowered = model_name.lower()
    if lowered in {
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash",
        "z-ai/glm-5.2",
    }:
        return "xhigh" if normalized == "max" else normalized
    if "grok-4.5" in lowered:
        return "high" if normalized in {"xhigh", "max"} else normalized
    if "openai/" in lowered or lowered.startswith("gpt-"):
        if "gpt-5.6" in lowered:
            return normalized
        return "xhigh" if normalized == "max" else normalized
    if "google/" in lowered or "gemini" in lowered:
        return "high" if normalized in {"max", "xhigh"} else normalized
    if "anthropic/" in lowered or "claude" in lowered:
        if not supports_anthropic_effort(lowered):
            return "high" if normalized in {"max", "xhigh"} else normalized
        if normalized == "xhigh" and not any(
            marker in lowered
            for marker in ("claude-opus-5", "claude-fable-5", "claude-opus-4-8")
        ):
            return "high"
        if normalized == "max" and not supports_anthropic_max(lowered):
            return "high"
        return normalized
    return normalized


def resolve_reasoning_tokens(model_name: str, effort_name: str) -> int | None:
    normalized = normalize_effort_alias(effort_name)
    lowered = model_name.lower()
    if "claude-haiku-4-5" in lowered and normalized == "max":
        return MAX_HAIKU_4_5_REASONING_TOKENS
    return None


def normalize_effort_alias(effort_name: str) -> str:
    normalized = effort_name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "maximum": "max",
        "extra_high": "xhigh",
        "extra": "xhigh",
        "x_high": "xhigh",
    }
    return aliases.get(normalized, normalized)


def supports_anthropic_effort(lowered_model_name: str) -> bool:
    return any(
        marker in lowered_model_name
        for marker in (
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-opus-5",
            "claude-fable-5",
            "claude-sonnet-4-6",
            "claude-opus-4-5",
            "mythos",
        )
    )


def supports_anthropic_max(lowered_model_name: str) -> bool:
    return any(
        marker in lowered_model_name
        for marker in (
            "claude-opus-4-8",
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-opus-5",
            "claude-fable-5",
            "claude-sonnet-4-6",
            "mythos",
        )
    )
