"""Typed release and model configuration."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from . import __version__

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

BENCHMARK_VERSION = f"v{__version__}"
HF_DATASET = "EdisonScientific/BixBench3"
HF_REVISION = BENCHMARK_VERSION
INPUT_BUCKET = "bixbench3-inputs"
GROUND_TRUTH_BUCKET = "bixbench3-ground-truth"
DEFAULT_ZONE = "us-central1-a"
DEFAULT_REGION = "us-central1"
RUN_MACHINE_TYPE = "n2-standard-32"
RUN_BOOT_DISK_GB = 500
RUNTIME_IMAGE_NAME = "bixbench3-runtime-v1-0-0-r7"
RUNTIME_IMAGE_FAMILY = "bixbench3-runtime"
REMOTE_ROOT = "/var/lib/bixbench3"
MESSAGE_LIMIT = 5_000
TIME_LIMIT_SECONDS = 86_400
ADJUDICATOR_MODEL = "openai/gpt-5.5"
ADJUDICATOR_EFFORT = "low"

_EFFORT_ALIASES = {
    "maximum": "max",
    "extra": "xhigh",
    "extra_high": "xhigh",
    "x_high": "xhigh",
}
_KNOWN_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})
_PROVIDER_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "xai": "XAI_API_KEY",
}


@dataclass(frozen=True, slots=True)
class ReferenceModelProfile:
    """A model configuration used for the reported benchmark."""

    model_id: str
    provider: str
    provider_env: str
    version: str
    supported_efforts: tuple[str, ...]
    model_args: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ReferenceCatalog:
    """Validated contents of reference_models.yaml."""

    schema_version: int
    adjudicator_model: str
    adjudicator_provider_env: str
    adjudicator_effort: str
    models: dict[str, ReferenceModelProfile]


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """The exact model settings passed to Inspect."""

    model_id: str
    provider: str
    is_reference: bool
    requested_effort: str
    resolved_effort: str
    required_environment_keys: tuple[str, ...]
    model_args: dict[str, JsonValue]
    inspect_effort_args: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Fixed scientific runtime settings."""

    version: str = BENCHMARK_VERSION
    hf_dataset: str = HF_DATASET
    hf_revision: str = HF_REVISION
    input_bucket: str = INPUT_BUCKET
    ground_truth_bucket: str = GROUND_TRUTH_BUCKET
    zone: str = DEFAULT_ZONE
    region: str = DEFAULT_REGION
    machine_type: str = RUN_MACHINE_TYPE
    boot_disk_gb: int = RUN_BOOT_DISK_GB
    runtime_image_name: str = RUNTIME_IMAGE_NAME
    runtime_image_family: str = RUNTIME_IMAGE_FAMILY
    remote_root: str = REMOTE_ROOT
    message_limit: int = MESSAGE_LIMIT
    time_limit_seconds: int = TIME_LIMIT_SECONDS
    adjudicator_model: str = ADJUDICATOR_MODEL
    adjudicator_effort: str = ADJUDICATOR_EFFORT


def normalize_effort(value: str) -> str:
    """Normalize and validate a user-facing effort name."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _EFFORT_ALIASES.get(normalized, normalized)
    if normalized not in _KNOWN_EFFORTS:
        choices = ", ".join(sorted(_KNOWN_EFFORTS))
        raise ValueError(f"unsupported effort {value!r}; choose one of: {choices}")
    return normalized


def load_reference_catalog(path: Path | None = None) -> ReferenceCatalog:
    """Load the JSON-compatible YAML reference model catalog."""
    if path is None:
        text = (
            files("bixbench3")
            .joinpath("reference_models.yaml")
            .read_text(encoding="utf-8")
        )
        source = "reference_models.yaml"
    else:
        text = path.read_text(encoding="utf-8")
        source = str(path)
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} must use JSON-compatible YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"{source} must contain a mapping")
    schema_version = _required_int(raw, "schema_version", source)
    adjudicator = _required_mapping(raw, "adjudicator", source)
    model_rows = raw.get("models")
    if not isinstance(model_rows, list):
        raise ValueError(f"{source}: models must be a list")
    profiles: dict[str, ReferenceModelProfile] = {}
    for index, value in enumerate(model_rows):
        row_source = f"{source}: models[{index}]"
        if not isinstance(value, dict):
            raise ValueError(f"{row_source} must be a mapping")
        profile = _parse_reference_profile(value, row_source)
        if profile.model_id in profiles:
            raise ValueError(f"{source}: duplicate model id {profile.model_id!r}")
        profiles[profile.model_id] = profile
    if not profiles:
        raise ValueError(f"{source}: models cannot be empty")
    return ReferenceCatalog(
        schema_version=schema_version,
        adjudicator_model=_required_string(adjudicator, "model", source),
        adjudicator_provider_env=_required_env_name(adjudicator, "provider_env", source),
        adjudicator_effort=normalize_effort(
            _required_string(adjudicator, "reasoning_effort", source)
        ),
        models=profiles,
    )


def resolve_model(
    model_id: str,
    effort: str,
    *,
    user_model_args: dict[str, JsonValue] | None = None,
    catalog: ReferenceCatalog | None = None,
) -> ResolvedModel:
    """Resolve a reference or arbitrary Inspect model without restricting IDs."""
    model_id = model_id.strip()
    if not model_id or "/" not in model_id:
        raise ValueError("model must be an Inspect provider/model identifier")
    requested_effort = normalize_effort(effort)
    catalog = catalog or load_reference_catalog()
    profile = catalog.models.get(model_id)
    if profile is not None and requested_effort not in profile.supported_efforts:
        supported = ", ".join(profile.supported_efforts)
        raise ValueError(
            f"{model_id} does not support effort {requested_effort!r}; "
            f"supported values: {supported}"
        )
    provider = profile.provider if profile is not None else model_id.split("/", 1)[0].lower()
    resolved_effort = _resolve_effort(model_id, requested_effort)
    model_args = dict(profile.model_args) if profile is not None else {}
    if user_model_args:
        model_args.update(user_model_args)
    environment_keys: list[str] = []
    provider_env = (
        profile.provider_env
        if profile is not None
        else _PROVIDER_ENV.get(provider, _generic_provider_env(provider))
    )
    if provider_env:
        environment_keys.append(provider_env)
    if catalog.adjudicator_provider_env not in environment_keys:
        environment_keys.append(catalog.adjudicator_provider_env)
    return ResolvedModel(
        model_id=model_id,
        provider=provider,
        is_reference=profile is not None,
        requested_effort=requested_effort,
        resolved_effort=resolved_effort,
        required_environment_keys=tuple(environment_keys),
        model_args=model_args,
        inspect_effort_args=_inspect_effort_args(
            model_id=model_id,
            requested_effort=requested_effort,
            resolved_effort=resolved_effort,
        ),
    )


def _resolve_effort(model_id: str, requested: str) -> str:
    lowered = model_id.lower()
    if lowered in {
        "openrouter/deepseek/deepseek-v4-flash",
        "openrouter/z-ai/glm-5.2",
        "deepseek/deepseek-v4-flash",
        "z-ai/glm-5.2",
    }:
        return "xhigh" if requested == "max" else requested
    if "grok-4.5" in lowered:
        return "high" if requested in {"xhigh", "max"} else requested
    if lowered.startswith("openai/") or lowered.startswith("gpt-"):
        if "gpt-5.6" in lowered:
            return requested
        return "xhigh" if requested == "max" else requested
    if lowered.startswith("google/") or "gemini" in lowered:
        return "high" if requested in {"max", "xhigh"} else requested
    if lowered.startswith("anthropic/") or "claude" in lowered:
        if requested == "max":
            if any(
                marker in lowered
                for marker in (
                    "claude-opus-5",
                    "claude-fable-5",
                    "claude-opus-4-8",
                    "claude-opus-4-6",
                    "claude-sonnet-4-6",
                )
            ):
                return "max"
            return "high"
        if requested == "xhigh" and not any(
            marker in lowered
            for marker in ("claude-opus-5", "claude-fable-5", "claude-opus-4-8")
        ):
            return "high"
    return requested


def _inspect_effort_args(
    *,
    model_id: str,
    requested_effort: str,
    resolved_effort: str,
) -> tuple[str, ...]:
    lowered = model_id.lower()
    if "claude-haiku-4-5" in lowered and requested_effort == "max":
        return ("--reasoning-tokens", "63999")
    if lowered.startswith("anthropic/") or "claude" in lowered:
        supports_effort = any(
            marker in lowered
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
        return ("--effort", resolved_effort) if supports_effort else ()
    if lowered.startswith("openai/") or lowered.startswith("gpt-"):
        verbosity = "high" if resolved_effort in {"high", "xhigh", "max"} else "medium"
        return ("--reasoning-effort", resolved_effort, "--verbosity", verbosity)
    if lowered.startswith("google/") or "gemini" in lowered:
        return ("--reasoning-effort", resolved_effort)
    if lowered.startswith("openrouter/"):
        return ("--reasoning-effort", resolved_effort)
    return ("--reasoning-effort", resolved_effort)


def _generic_provider_env(provider: str) -> str | None:
    if provider in {"mockllm", "local"}:
        return None
    normalized = re.sub(r"[^A-Z0-9]+", "_", provider.upper()).strip("_")
    return f"{normalized}_API_KEY" if normalized else None


def _parse_reference_profile(
    row: dict[object, object],
    source: str,
) -> ReferenceModelProfile:
    model_id = _required_string(row, "id", source)
    provider = _required_string(row, "provider", source).lower()
    supported = row.get("supported_efforts")
    if not isinstance(supported, list) or not supported:
        raise ValueError(f"{source}: supported_efforts must be a non-empty list")
    efforts: list[str] = []
    for value in supported:
        if not isinstance(value, str):
            raise ValueError(f"{source}: supported_efforts values must be strings")
        effort = normalize_effort(value)
        if effort in efforts:
            raise ValueError(f"{source}: duplicate supported effort {effort!r}")
        efforts.append(effort)
    raw_model_args = row.get("model_args", {})
    if not isinstance(raw_model_args, dict):
        raise ValueError(f"{source}: model_args must be a mapping")
    model_args = _json_mapping(raw_model_args, f"{source}: model_args")
    return ReferenceModelProfile(
        model_id=model_id,
        provider=provider,
        provider_env=_required_env_name(row, "provider_env", source),
        version=_required_string(row, "version", source),
        supported_efforts=tuple(efforts),
        model_args=model_args,
    )


def _json_mapping(value: dict[object, object], source: str) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            raise ValueError(f"{source}: keys must be strings")
        result[raw_key] = _json_value(raw_value, f"{source}.{raw_key}")
    return result


def _json_value(value: object, source: str) -> JsonValue:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_json_value(item, source) for item in value]
    if isinstance(value, dict):
        return _json_mapping(value, source)
    raise ValueError(f"{source}: unsupported value type {type(value).__name__}")


def _required_mapping(
    value: dict[object, object],
    key: str,
    source: str,
) -> dict[object, object]:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{source}: {key} must be a mapping")
    return result


def _required_string(value: dict[object, object], key: str, source: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return result.strip()


def _required_int(value: dict[object, object], key: str, source: str) -> int:
    result = value.get(key)
    if not isinstance(result, int) or isinstance(result, bool):
        raise ValueError(f"{source}: {key} must be an integer")
    return result


def _required_env_name(value: dict[object, object], key: str, source: str) -> str:
    result = _required_string(value, key, source)
    if re.fullmatch(r"[A-Z][A-Z0-9_]*", result) is None:
        raise ValueError(f"{source}: {key} must be an uppercase environment variable name")
    return result
