from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, cast

_OPAQUE_REASONING_RE = re.compile(r"gAAAAA[A-Za-z0-9_-]{80,}")


def compact_state_messages(
    messages: list[Any],
    *,
    max_tool_output_chars: int = 4000,
    max_trace_chars: int | None = 120_000,
) -> str:
    """Return a compact, budgeted markdown transcript from Inspect TaskState messages."""
    digest = _trace_digest(messages)
    blocks: list[str] = []
    for index, message in enumerate(messages):
        msg = _message_to_dict(message)
        role = str(msg.get("role") or "<unknown>")
        function_name = _function_name(msg)
        label = f"message {index} | {role}"
        if function_name:
            label += f" | {function_name}"

        parts: list[str] = []
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            parts.append("tool_calls:\n" + _safe_json(tool_calls))

        content = _content_text(msg.get("content"))
        if role == "tool":
            content = _truncate_middle(content, max_tool_output_chars)
        if content:
            parts.append(content)

        if not parts:
            continue
        body = "\n\n".join(parts)
        blocks.append(f"## {label}\n\n{body}")
    transcript = "\n\n".join(blocks).strip()
    if digest:
        transcript = f"{digest}\n\n# Budgeted Message Transcript\n\n{transcript}"
    if max_trace_chars is not None:
        transcript = _budget_text(transcript, max_trace_chars)
    return transcript


def compact_eval_log(
    eval_log_path: Path,
    *,
    max_tool_output_chars: int = 4000,
    max_trace_chars: int | None = 120_000,
) -> str:
    """Return a compact markdown transcript from a finalized Inspect .eval log."""
    return compact_eval_log_data(
        load_eval_log(eval_log_path),
        max_tool_output_chars=max_tool_output_chars,
        max_trace_chars=max_trace_chars,
    )


def load_eval_log(eval_log_path: Path) -> dict[str, Any]:
    dumped = subprocess.check_output(
        ["uv", "run", "inspect", "log", "dump", str(eval_log_path)],
        text=True,
    )
    data = json.loads(dumped)
    if not isinstance(data, dict):
        raise ValueError(f"{eval_log_path} did not dump to a JSON object")
    return cast(dict[str, Any], data)


def compact_eval_log_data(
    data: dict[str, Any],
    *,
    max_tool_output_chars: int = 4000,
    max_trace_chars: int | None = 120_000,
) -> str:
    samples = data.get("samples") or []
    if not samples:
        return ""
    return compact_state_messages(
        samples[0].get("messages") or [],
        max_tool_output_chars=max_tool_output_chars,
        max_trace_chars=max_trace_chars,
    )


def eval_log_input_text(data: dict[str, Any]) -> str:
    samples = data.get("samples") or []
    if not samples:
        return ""
    value = samples[0].get("input")
    if not isinstance(value, str):
        raise ValueError("Inspect eval sample input is not text")
    return value


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return message
    if hasattr(message, "model_dump"):
        return cast(dict[str, Any], message.model_dump())
    if hasattr(message, "__dict__"):
        return cast(dict[str, Any], vars(message))
    return {"content": str(message), "role": type(message).__name__}


def _trace_digest(messages: list[Any]) -> str:
    """Summarize process evidence before the budgeted transcript tail/head."""
    role_counts: Counter[str] = Counter()
    tool_call_counts: Counter[str] = Counter()
    shell_commands: list[str] = []
    file_mentions: set[str] = set()
    error_snippets: list[str] = []
    validation_snippets: list[str] = []
    output_snippets: list[str] = []
    last_messages: list[str] = []

    for index, message in enumerate(messages):
        msg = _message_to_dict(message)
        role = str(msg.get("role") or "<unknown>")
        role_counts[role] += 1
        content = _content_text(msg.get("content"))
        if content:
            last_messages.append(_one_line(f"message {index} | {role}: {content}", 500))
            last_messages = last_messages[-12:]
            lower = content.lower()
            if _looks_like_error(lower):
                error_snippets.append(_one_line(f"message {index}: {content}", 700))
                error_snippets = error_snippets[-30:]
            if _looks_like_validation(lower):
                validation_snippets.append(_one_line(f"message {index}: {content}", 500))
                validation_snippets = validation_snippets[-20:]
            if "/outputs" in lower or "outputs/" in lower:
                output_snippets.append(_one_line(f"message {index}: {content}", 500))
                output_snippets = output_snippets[-20:]
            file_mentions.update(_output_file_mentions(content))

        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tool_call in tool_calls:
                function_name = _tool_call_name(tool_call)
                if function_name:
                    tool_call_counts[function_name] += 1
                command = _tool_call_command(tool_call)
                if command:
                    shell_commands.append(_one_line(command, 500))
                    shell_commands = shell_commands[-80:]

    sections = ["# Trace Digest"]
    sections.append(f"- Messages: {len(messages)}")
    if role_counts:
        sections.append(
            "- Message roles: "
            + ", ".join(f"{role}={count}" for role, count in sorted(role_counts.items()))
        )
    if tool_call_counts:
        sections.append(
            "- Tool calls: "
            + ", ".join(
                f"{name}={count}" for name, count in tool_call_counts.most_common(12)
            )
        )
    if file_mentions:
        sections.append("## Output File Mentions\n" + _bullet_list(sorted(file_mentions)[:80]))
    if shell_commands:
        sections.append("## Recent Shell Commands\n" + _numbered_list(shell_commands[-40:]))
    if error_snippets:
        sections.append("## Recent Error / Warning Evidence\n" + _bullet_list(error_snippets[-20:]))
    if validation_snippets:
        sections.append("## Validation Evidence\n" + _bullet_list(validation_snippets[-15:]))
    if output_snippets:
        sections.append("## Output-Writing Evidence\n" + _bullet_list(output_snippets[-15:]))
    if last_messages:
        sections.append("## Final Messages\n" + _numbered_list(last_messages))
    return "\n\n".join(sections)


def _function_name(message: dict[str, Any]) -> str | None:
    function = message.get("function")
    if isinstance(function, str):
        return function
    if isinstance(function, dict):
        return str(function.get("name") or "") or None
    return None


def _tool_call_name(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function")
    if isinstance(function, str):
        return function
    if isinstance(function, dict):
        return str(function.get("name") or "") or None
    return None


def _tool_call_command(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function_name = _tool_call_name(tool_call)
    if function_name not in {"bash_session", "shell", "python", "R"}:
        return None
    arguments = tool_call.get("arguments")
    if isinstance(arguments, str):
        return arguments
    if isinstance(arguments, dict):
        for key in ("input", "command", "code"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "reasoning"}:
                    text = item.get("text") or item.get("reasoning") or ""
                    if text:
                        parts.append(str(text))
                elif item_type == "tool_use":
                    parts.append("tool_use:\n" + _safe_json(item))
                else:
                    parts.append(_safe_json(item))
            else:
                parts.append(str(item))
        return _sanitize_text("\n".join(parts))
    return _sanitize_text(_safe_json(content))


def _safe_json(value: Any) -> str:
    try:
        return _sanitize_text(json.dumps(value, indent=2, default=str))
    except TypeError:
        return _sanitize_text(str(value))


def _sanitize_text(text: str) -> str:
    return _OPAQUE_REASONING_RE.sub("<opaque_reasoning_blob>", text)


def _looks_like_error(lower_text: str) -> bool:
    needles = (
        "error",
        "failed",
        "traceback",
        "exception",
        "no such file",
        "not found",
        "permission denied",
        "killed",
        "timeout",
        "exited with",
    )
    return any(needle in lower_text for needle in needles)


def _looks_like_validation(lower_text: str) -> bool:
    needles = (
        "validated",
        "validation",
        "checked",
        "shape",
        "columns",
        "rows",
        "head",
        "summary",
        "qc",
        "sanity",
    )
    return any(needle in lower_text for needle in needles)


def _output_file_mentions(text: str) -> set[str]:
    pattern = r"(?:/workspace/)?outputs/[A-Za-z0-9_.=/+-]+|/outputs/[A-Za-z0-9_.=/+-]+"
    return set(re.findall(pattern, text))


def _one_line(text: str, max_chars: int) -> str:
    collapsed = " ".join(str(text).split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 20] + " ... [truncated]"


def _bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- none"


def _numbered_list(items: list[str]) -> str:
    return "\n".join(f"{index + 1}. {item}" for index, item in enumerate(items))


def _budget_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    marker = (
        f"\n\n... [trace compacted to {max_chars} chars; "
        f"omitted {len(text) - max_chars} chars from the middle] ...\n\n"
    )
    keep = max(0, (max_chars - len(marker)) // 2)
    return text[:keep] + marker + text[-keep:]


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    omitted = len(text) - (keep * 2)
    return (
        text[:keep]
        + f"\n\n... [truncated {omitted} chars] ...\n\n"
        + text[-keep:]
    )
