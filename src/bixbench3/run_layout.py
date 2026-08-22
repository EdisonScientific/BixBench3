"""Stable local and cloud resource layout for one benchmark run."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunLayout:
    """Paths and resource names for one immutable run attempt."""

    task_id: str
    paper_id: str
    model_id: str
    model_slug: str
    timestamp: str
    run_id: str
    local_dir: Path
    vm_name: str
    loader_vm_name: str
    data_disk_name: str

    @property
    def manifest_path(self) -> Path:
        return self.local_dir / "run_manifest.json"

    @property
    def artifact_grades_dir(self) -> Path:
        return self.local_dir / "artifact_grades"

    @property
    def grade_path(self) -> Path:
        return self.local_dir / "grade.json"

    @property
    def judge_path(self) -> Path:
        return self.local_dir / "judge.json"

    @property
    def cleanup_path(self) -> Path:
        return self.local_dir / "cleanup.json"

    def materialize(self) -> None:
        """Create the score-only local result directories."""
        self.artifact_grades_dir.mkdir(parents=True, exist_ok=True)


def build_run_layout(
    *,
    task_id: str,
    paper_id: str,
    model_id: str,
    manifest_sha256: str,
    runs_root: Path = Path("runs"),
    now: datetime | None = None,
) -> RunLayout:
    """Build a collision-resistant layout without touching the filesystem."""
    now = now or datetime.now(UTC)
    timestamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    model_slug = slugify(model_id)
    task_slug = slugify(paper_id)
    identity = f"{paper_id}\0{model_id}\0{timestamp}".encode()
    run_hash = hashlib.sha256(identity).hexdigest()[:10]
    run_id = f"{task_slug}-{model_slug}-{timestamp.lower()}-{run_hash}"
    release_hash = _manifest_hash(manifest_sha256)
    return RunLayout(
        task_id=task_id,
        paper_id=paper_id,
        model_id=model_id,
        model_slug=model_slug,
        timestamp=timestamp,
        run_id=run_id,
        local_dir=runs_root / paper_id / model_slug / timestamp,
        vm_name=resource_name("arb-run", task_slug, run_hash),
        loader_vm_name=resource_name("arb-loader", task_slug, release_hash),
        data_disk_name=resource_name("arb-data", task_slug, release_hash),
    )


def slugify(value: str, *, maximum: int = 48) -> str:
    """Convert a user or paper identifier to a stable path/resource slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        slug = "item"
    return slug[:maximum].rstrip("-")


def resource_name(*parts: str, maximum: int = 63) -> str:
    """Create a valid GCE resource name while preserving a hash suffix."""
    joined = "-".join(slugify(part, maximum=maximum) for part in parts if part)
    if len(joined) <= maximum:
        return joined
    suffix = slugify(parts[-1], maximum=16)
    prefix_limit = maximum - len(suffix) - 1
    return f"{joined[:prefix_limit].rstrip('-')}-{suffix}"


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted runs retain valid state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _manifest_hash(value: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{12,64}", normalized):
        return normalized[:12]
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]
