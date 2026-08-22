from __future__ import annotations

import json
import os
import re
from pathlib import Path

import yaml
from inspect_ai.dataset import MemoryDataset, Sample

from .models import RuntimeCase

TIME_LIMIT_PLACEHOLDER = "{time_limit}"
AVAILABLE_PACKAGES_PLACEHOLDER = "{available_packages}"
EXTRA_INFO_PLACEHOLDER = "{extra_info_section}"
DEFAULT_TIME_LIMIT_SECONDS = 86_400


def load_cases_dataset(
    dataset_path: str,
    case_ids: list[str] | None = None,
    *,
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
    available_packages_path: str | Path | None = None,
) -> MemoryDataset:
    cases_path = _resolve_cases_path(Path(dataset_path))
    dataset_dir = cases_path.parent
    selected = set(case_ids or [])
    samples: list[Sample] = []
    for payload in _read_case_payloads(cases_path):
        case = RuntimeCase.model_validate(payload)
        if selected and case.task_id not in selected:
            continue
        prompt_path = _resolve_prompt_path(dataset_dir, case.prompt.prompt_path)
        prompt = render_agent_prompt(
            prompt_path.read_text(encoding="utf-8"),
            time_limit_seconds=time_limit_seconds,
            available_packages_path=available_packages_path,
        )
        samples.append(
            Sample(
                id=case.task_id,
                input=prompt,
                metadata={
                    **case.model_dump(),
                    "dataset_dir": str(dataset_dir.resolve()),
                },
            )
        )
    if selected:
        found = {str(sample.id) for sample in samples}
        missing = sorted(selected - found)
        if missing:
            raise ValueError(f"requested task IDs were not found: {', '.join(missing)}")
    if not samples:
        raise ValueError(f"no runnable tasks found in {cases_path}")
    return MemoryDataset(samples=samples)


def render_agent_prompt(
    prompt: str,
    *,
    time_limit_seconds: int = DEFAULT_TIME_LIMIT_SECONDS,
    available_packages_path: str | Path | None = None,
) -> str:
    rendered = _render_time_limit(prompt, time_limit_seconds=time_limit_seconds)
    rendered = _render_available_packages(
        rendered,
        available_packages_path=available_packages_path,
    )
    rendered = _remove_extra_info_section(rendered)
    return rendered.strip() + "\n"


def _resolve_cases_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (
        path / "cases.jsonl",
        path / "runtime.json",
        path / "case.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no cases.jsonl, runtime.json, or case.json found under {path}")


def _read_case_payloads(path: Path) -> list[object]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return list(payload["cases"])
    return [payload]


def _resolve_prompt_path(dataset_dir: Path, prompt_path: str) -> Path:
    candidate = (dataset_dir / prompt_path).resolve()
    root = dataset_dir.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"prompt path escapes dataset directory: {prompt_path}")
    if not candidate.is_file():
        raise FileNotFoundError(f"task prompt does not exist: {candidate}")
    return candidate


def _render_time_limit(prompt: str, *, time_limit_seconds: int) -> str:
    if time_limit_seconds <= 0:
        raise ValueError("time_limit_seconds must be positive")
    if TIME_LIMIT_PLACEHOLDER in prompt:
        return prompt.replace(TIME_LIMIT_PLACEHOLDER, str(time_limit_seconds))
    notice = (
        "## Runtime Limit\n\n"
        f"You have a maximum wall-clock budget of {time_limit_seconds} seconds for this task. "
        "The run may be terminated when this budget is reached, without an additional "
        "finalization turn. Write requested output artifacts and METHODS.md incrementally "
        "as you work, so partial progress can be graded if the run ends early."
    )
    return prompt.rstrip() + "\n\n" + notice + "\n"


def _render_available_packages(
    prompt: str,
    *,
    available_packages_path: str | Path | None,
) -> str:
    if AVAILABLE_PACKAGES_PLACEHOLDER not in prompt:
        return prompt
    rendered_inventory = _available_packages_markdown(available_packages_path)
    if not rendered_inventory:
        return _remove_available_packages_section(prompt)
    return prompt.replace(AVAILABLE_PACKAGES_PLACEHOLDER, rendered_inventory)


def _available_packages_markdown(
    available_packages_path: str | Path | None = None,
) -> str:
    path = (
        Path(available_packages_path)
        if available_packages_path
        else _default_available_packages_path()
    )
    if not path.is_file():
        return ""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ""
    sections = payload.get("sections")
    if not isinstance(sections, list):
        return ""
    lines = [
        "The sandbox image already includes the following general-purpose software:",
        "",
    ]
    for section in sections:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title", "")).strip()
        raw_packages = section.get("packages", [])
        if not isinstance(raw_packages, list):
            continue
        packages = [str(package).strip() for package in raw_packages if str(package).strip()]
        description = str(section.get("description", "")).strip()
        if not title or not packages:
            continue
        package_text = ", ".join(f"`{package}`" for package in packages)
        suffix = f" {description}" if description else ""
        lines.append(f"- {title}: {package_text}.{suffix}")
    if len(lines) == 2:
        return ""
    lines.extend(
        [
            "",
            "Method-specific analysis packages are not preinstalled unless listed here; "
            "install the tools and versions required by the method guidance when needed.",
        ]
    )
    return "\n".join(lines)


def _default_available_packages_path() -> Path:
    configured = os.environ.get("BIXBENCH3_AVAILABLE_PACKAGES")
    if configured:
        return Path(configured)
    return Path(__file__).parents[3] / "sandbox" / "available_packages.yaml"


def _remove_available_packages_section(prompt: str) -> str:
    return (
        re.sub(
            r"\n?## Already Available Software\n\n.*?(?=\n## |\Z)",
            "\n",
            prompt,
            count=1,
            flags=re.DOTALL,
        ).strip()
        + "\n"
    )


def _remove_extra_info_section(prompt: str) -> str:
    without_placeholder = prompt.replace(EXTRA_INFO_PLACEHOLDER, "")
    return (
        re.sub(
            r"\n?## Extra Info\n\n.*?(?=\n## |\Z)",
            "\n",
            without_placeholder,
            count=1,
            flags=re.DOTALL,
        ).strip()
        + "\n"
    )
