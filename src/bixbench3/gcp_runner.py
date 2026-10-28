"""Secure, single-task GCP orchestration using only the gcloud CLI."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, TextIO, cast
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

from . import __version__
from .config import (
    BENCHMARK_VERSION,
    DEFAULT_ZONE,
    GROUND_TRUTH_BUCKET,
    HF_DATASET,
    HF_REVISION,
    INPUT_BUCKET,
    JsonValue,
    ResolvedModel,
    RuntimeSettings,
)
from .grading.effort import resolve_reasoning_effort
from .grading.judge import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_JUDGE_REASONING_EFFORT,
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
    DEFAULT_MAX_TRACE_CHARS,
    PROMPT_TEMPLATE_SHA256,
)
from .manifests import input_release_manifest_url
from .run_layout import RunLayout, build_run_layout, slugify, write_json

type Argv = tuple[str, ...]

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_DEVICE_NAME = "bixbench3-data"
_DATA_MOUNT = "/mnt/bixbench3-data"
_SECRET_ENV_PATH = "/run/bixbench3/credentials.env"
_RUN_SCRIPT_PATH = "/var/lib/bixbench3/run.sh"
_RUN_STATUS_PATH = "/var/lib/bixbench3/status.json"
_RUN_FINISHED_PATH = "/var/lib/bixbench3/.finished"
_RUN_SERVICE = "bixbench3-run.service"
_RUN_SERVICE_PATH = f"/etc/systemd/system/{_RUN_SERVICE}"
_ENV_NAME = re.compile(r"[A-Z][A-Z0-9_]*")
_MODEL_ARG_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]*")
_GCS_URI = re.compile(r"gs://([a-z0-9][a-z0-9._-]{1,221}[a-z0-9])(?:/(.*))?")
_HF_CASE_FILES = (
    "agent/prompt.md",
    "agent/task_body.json",
    "agent/inputs_metadata.json",
    "runtime/runtime.json",
)
_RUNTIME_SOURCE_FILES = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
)
_RUNTIME_SOURCE_DIRECTORIES = ("src", "sandbox")
_RUNTIME_ARCHIVE_EXCLUDED_PARTS = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "runs",
    }
)
_PROCESS_JUDGE_ENVIRONMENT_KEYS = ("OPENAI_API_KEY",)


class CommandExecutor(Protocol):
    """A typed boundary around subprocess execution."""

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        """Execute one argv command without a shell."""


@dataclass(slots=True)
class SubprocessExecutor:
    """Production subprocess executor."""

    output: TextIO = sys.stdout

    def run(
        self,
        argv: Sequence[str],
        *,
        input_text: str | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        validate_safe_argv(argv)
        return subprocess.run(
            list(argv),
            input=input_text,
            text=True,
            check=check,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
        )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Public runtime metadata required to launch one task."""

    task_id: str
    paper_id: str
    input_manifest_uri: str
    input_manifest_sha256: str
    input_size_bytes: int
    grader_manifest_url: str
    grader_manifest_sha256: str
    hf_revision: str = HF_REVISION
    gcs_release: str = BENCHMARK_VERSION

    @property
    def disk_size_gb(self) -> int:
        input_gb = self.input_size_bytes / (1024**3)
        return max(10, math.ceil(input_gb * 1.10) + 1)


@dataclass(frozen=True, slots=True)
class StagedCase:
    """Locally authenticated, agent-visible one-case dataset."""

    dataset_dir: Path
    dataset_path: Path
    file_sha256: dict[str, str]


@dataclass(frozen=True, slots=True)
class RunPlan:
    """Every cloud command and immutable setting for a run."""

    layout: RunLayout
    task: TaskSpec
    model: ResolvedModel
    project: str
    zone: str
    runtime_image: str
    results_uri: str | None
    commands: dict[str, Argv]
    inspect_argv: Argv

    def public_summary(self) -> dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "paper_id": self.task.paper_id,
            "model": self.model.model_id,
            "fallback_models": list(self.model.fallback_model_ids),
            "reference_model": self.model.is_reference,
            "requested_effort": self.model.requested_effort,
            "resolved_effort": self.model.resolved_effort,
            "required_environment_keys": list(required_environment_keys(self.model)),
            "project": self.project,
            "zone": self.zone,
            "runtime_image": self.runtime_image,
            "hf_staging": {
                "authenticated_local": True,
                "revision": self.task.hf_revision,
                "files": [f"cases/{self.task.paper_id}/{relative}" for relative in _HF_CASE_FILES],
                "vm_receives_hf_credentials": False,
            },
            "data_disk": self.layout.data_disk_name,
            "data_disk_type": "pd-balanced",
            "data_disk_size_gb": self.task.disk_size_gb,
            "run_vm": self.layout.vm_name,
            "loader_vm": self.layout.loader_vm_name,
            "machine_type": RuntimeSettings().machine_type,
            "boot_disk_gb": RuntimeSettings().boot_disk_gb,
            "results_uri": self.results_uri,
            "retention": "full_bundle" if self.results_uri else "scores_only",
            "process_judge": process_judge_config(),
            "commands": {name: list(argv) for name, argv in self.commands.items()},
            "inspect_argv": list(self.inspect_argv),
        }


@dataclass(frozen=True, slots=True)
class RuntimeBuildPlan:
    """Cloud commands for a one-time private runtime image build."""

    project: str
    zone: str
    builder_vm: str
    image_name: str
    commands: dict[str, Argv]
    source_archive_sha256: str
    cellranger_sha256: str
    cellranger_arc_sha256: str

    def public_summary(self) -> dict[str, Any]:
        return {
            "project": self.project,
            "zone": self.zone,
            "builder_vm": self.builder_vm,
            "image_name": self.image_name,
            "source_archive_sha256": self.source_archive_sha256,
            "cellranger_sha256": self.cellranger_sha256,
            "cellranger_arc_sha256": self.cellranger_arc_sha256,
            "commands": {name: list(argv) for name, argv in self.commands.items()},
        }


@dataclass(slots=True)
class GCPRunner:
    """Orchestrate one benchmark task from a local workstation."""

    project: str
    executor: CommandExecutor
    settings: RuntimeSettings = field(default_factory=RuntimeSettings)
    runs_root: Path = Path("runs")
    repo_root: Path = field(default_factory=Path.cwd)

    def build_run_plan(
        self,
        *,
        task: TaskSpec,
        model: ResolvedModel,
        results_bucket: str | None,
        default_service_account: str | None = None,
        case_dataset_dir: Path | None = None,
    ) -> RunPlan:
        layout = build_run_layout(
            task_id=task.task_id,
            paper_id=task.paper_id,
            model_id=model.model_id,
            manifest_sha256=task.input_manifest_sha256,
            runs_root=self.runs_root,
        )
        results_uri = result_run_uri(results_bucket, layout) if results_bucket is not None else None
        service_account = default_service_account
        if results_uri is not None and service_account is None:
            service_account = "<PROJECT_NUMBER>-compute@developer.gserviceaccount.com"
        commands = build_run_commands(
            project=self.project,
            settings=self.settings,
            task=task,
            layout=layout,
            service_account=service_account if results_uri is not None else None,
            results_uri=results_uri,
            case_dataset_dir=case_dataset_dir,
        )
        inspect_argv = build_inspect_argv(
            settings=self.settings,
            task=task,
            model=model,
        )
        plan = RunPlan(
            layout=layout,
            task=task,
            model=model,
            project=self.project,
            zone=self.settings.zone,
            runtime_image=self.settings.runtime_image_name,
            results_uri=results_uri,
            commands=commands,
            inspect_argv=inspect_argv,
        )
        validate_plan(plan)
        return plan

    def run_task(
        self,
        *,
        task: TaskSpec,
        model: ResolvedModel,
        results_bucket: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if dry_run:
            print(
                f"planning task {task.task_id} with {model.model_id} at {model.resolved_effort} effort",
                file=sys.stderr,
                flush=True,
            )
            plan = self.build_run_plan(
                task=task,
                model=model,
                results_bucket=results_bucket,
            )
            summary = plan.public_summary()
            summary["dry_run"] = True
            return summary

        print(
            f"[1/8] validating API keys and staging public task {task.task_id}",
            file=sys.stderr,
            flush=True,
        )
        environment = selected_environment(required_environment_keys(model))
        with tempfile.TemporaryDirectory(prefix="bixbench3-case-") as temporary:
            staged_case = stage_hf_case(
                task,
                Path(temporary) / "dataset",
            )
            return self._run_staged_task(
                task=task,
                model=model,
                environment=environment,
                staged_case=staged_case,
                results_bucket=results_bucket,
            )

    def _run_staged_task(
        self,
        *,
        task: TaskSpec,
        model: ResolvedModel,
        environment: Mapping[str, str],
        staged_case: StagedCase,
        results_bucket: str | None,
    ) -> dict[str, Any]:
        print(
            f"[2/8] checking GCP project {self.project}, APIs, and result storage",
            file=sys.stderr,
            flush=True,
        )
        preflight = self._preflight(
            require_artifact_registry=False,
            results_bucket=results_bucket,
        )
        default_service_account = (
            f"{preflight['project_number']}-compute@developer.gserviceaccount.com"
            if results_bucket
            else None
        )
        plan = self.build_run_plan(
            task=task,
            model=model,
            results_bucket=results_bucket,
            default_service_account=default_service_account,
            case_dataset_dir=staged_case.dataset_dir,
        )
        print(
            f"[3/8] checking runtime image {plan.runtime_image}",
            file=sys.stderr,
            flush=True,
        )
        self._require_runtime_image()
        print(
            f"[4/8] preparing reusable input disk {plan.layout.data_disk_name}",
            file=sys.stderr,
            flush=True,
        )
        self._ensure_data_disk(plan)
        plan.layout.materialize()
        manifest = self._initial_manifest(
            plan,
            preflight,
            staged_case=staged_case,
        )
        write_json(plan.layout.manifest_path, manifest)

        vm_created = False
        try:
            print(
                f"[5/8] creating run VM {plan.layout.vm_name} in {plan.zone}",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["create_run_vm"])
            vm_created = True
            print(
                "[6/8] waiting for SSH, attaching inputs, and copying task configuration",
                file=sys.stderr,
                flush=True,
            )
            self._wait_for_ssh(plan.commands["wait_run_ssh"])
            self.executor.run(plan.commands["attach_data_disk"])
            self.executor.run(
                plan.commands["prepare_run_vm"],
                input_text=remote_prepare_script(self.settings),
            )
            self.executor.run(plan.commands["copy_run_manifest"])
            self.executor.run(plan.commands["copy_case_dataset"])
            self.executor.run(
                plan.commands["stream_environment"],
                input_text=environment_file(environment),
            )
            if plan.results_uri:
                self.executor.run(
                    plan.commands["preflight_results_write"],
                    input_text=results_write_preflight_script(plan.results_uri),
                )
            print(
                f"[7/8] starting {model.model_id} at {model.resolved_effort} effort as a detached service",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(
                plan.commands["install_run_script"],
                input_text=remote_run_script(plan, self.settings),
            )
            self.executor.run(
                plan.commands["install_run_service"],
                input_text=remote_run_service_unit(),
            )
            self.executor.run(plan.commands["start_run_service"])
        except subprocess.CalledProcessError as error:
            if vm_created:
                print(
                    "submission failed; stopping the run VM for debugging",
                    file=sys.stderr,
                    flush=True,
                )
                self.executor.run(
                    plan.commands["scrub_run_credentials"],
                    check=False,
                    capture_output=True,
                )
                cleanup = self._cleanup_run_vm(plan, succeeded=False)
                manifest["cleanup"] = cleanup
                write_json(plan.layout.cleanup_path, cleanup)
            manifest["status"] = "failed"
            manifest["phase"] = "submission_failed"
            manifest["completed_at"] = datetime.now(UTC).isoformat()
            manifest["error"] = command_error(error)
            write_json(plan.layout.manifest_path, manifest)
            raise RuntimeError(
                f"run submission failed; VM {plan.layout.vm_name} was stopped for debugging. "
                f"Boot disk charges continue until it is cancelled. {command_error(error)}"
            ) from error

        status_command = lifecycle_command("status", plan.layout.run_id, self.project)
        complete_command = lifecycle_command("complete", plan.layout.run_id, self.project)
        cancel_command = lifecycle_command("cancel", plan.layout.run_id, self.project)
        manifest["status"] = "running"
        manifest["phase"] = "agent_running"
        manifest["detached_at"] = datetime.now(UTC).isoformat()
        manifest["follow_up_commands"] = {
            "status": status_command,
            "complete": complete_command,
            "cancel": cancel_command,
        }
        write_json(plan.layout.manifest_path, manifest)
        print(
            "[8/8] run detached successfully; this VM remains billable until complete or cancel",
            file=sys.stderr,
            flush=True,
        )
        return manifest

    def status_run(self, run_id: str) -> dict[str, Any]:
        """Read local, GCP, and remote state for a detached run."""

        manifest_path, manifest = self._load_lifecycle_manifest(run_id)
        vm_name, zone = lifecycle_vm_identity(manifest)
        vm = self._describe_lifecycle_vm(vm_name, zone)
        remote = self._read_remote_status(vm_name, zone) if vm["state"] == "RUNNING" else None
        effective_status = manifest.get("status", "unknown")
        phase = manifest.get("phase", "unknown")
        health: dict[str, Any] | None = None
        if isinstance(remote, dict):
            effective_status = remote.get("status", effective_status)
            phase = remote.get("phase", phase)
        if vm["state"] == "RUNNING" and effective_status == "running":
            health = self._read_remote_health(vm_name, zone)
            if health is None:
                effective_status, phase = "issue", "ssh_unavailable"
            elif health.get("service_state") != "active":
                effective_status, phase = "issue", "service_inactive"
            elif phase == "agent_running" and not health.get("inspect_running"):
                updated_at = remote.get("updated_at") if isinstance(remote, dict) else None
                try:
                    age = (datetime.now(UTC) - datetime.fromisoformat(updated_at)).total_seconds() if isinstance(updated_at, str) else float("inf")
                except ValueError:
                    age = float("inf")
                if age > 120:
                    effective_status, phase = "issue", "inspect_not_running"
        elif vm["state"] == "MISSING" and effective_status not in {"succeeded", "failed", "cancelled"}:
            effective_status, phase = "failed", "vm_missing"
        elif vm["state"] not in {"RUNNING", "MISSING"} and effective_status not in {"succeeded", "failed", "cancelled"}:
            effective_status, phase = "issue", "vm_stopped"
        terminal = effective_status in {"succeeded", "failed", "cancelled"}
        return {
            "run_id": manifest.get("run_id", run_id),
            "project": self.project,
            "manifest_path": str(manifest_path),
            "vm_name": vm_name,
            "vm_state": vm["state"],
            "status": effective_status,
            "phase": phase,
            "terminal": terminal,
            "remote": remote,
            "health": health,
            "next_command": lifecycle_command(
                "complete" if terminal else "cancel" if effective_status == "issue" else "status",
                run_id,
                self.project,
            ),
        }

    def complete_run(self, run_id: str) -> dict[str, Any]:
        """Collect a finished detached run and delete its VM."""

        manifest_path, manifest = self._load_lifecycle_manifest(run_id)
        cleanup_path = manifest_path.parent / "cleanup.json"
        if cleanup_path.exists():
            cleanup = parse_json_object(cleanup_path.read_text(encoding="utf-8"), str(cleanup_path))
            if cleanup.get("action") == "deleted":
                return {
                    "run_id": run_id,
                    "status": manifest.get("status", "unknown"),
                    "local_dir": str(manifest_path.parent),
                    "cleanup": cleanup,
                    "already_complete": True,
                }
        current = self.status_run(run_id)
        if not current["terminal"]:
            raise RuntimeError(
                f"run {run_id} is still {current['status']} ({current['phase']}); "
                f"check it with `{lifecycle_command('status', run_id, self.project)}`"
            )
        vm_name, zone = lifecycle_vm_identity(manifest)
        manifest["phase"] = "collecting"
        write_json(manifest_path, manifest)
        local_dir = manifest_path.parent
        self._collect_lifecycle_files(vm_name, zone, local_dir)
        remote_manifest_path = local_dir / "remote_run_manifest.json"
        if remote_manifest_path.exists():
            remote_manifest = parse_json_object(
                remote_manifest_path.read_text(encoding="utf-8"),
                str(remote_manifest_path),
            )
            for key in (
                "remote_exit_status",
                "remote_completed_at",
                "status",
                "phase",
                "process_judge",
            ):
                if key in remote_manifest:
                    manifest[key] = remote_manifest[key]
        if isinstance(current.get("remote"), dict):
            remote = current["remote"]
            for key in ("remote_exit_status", "updated_at", "status", "phase"):
                if key in remote:
                    manifest[key] = remote[key]
        cleanup = self._delete_lifecycle_vm(vm_name, zone)
        manifest["status"] = (
            "succeeded" if manifest.get("remote_exit_status") == 0 else str(current["status"])
        )
        manifest["phase"] = "finalized"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        manifest["cleanup"] = cleanup
        write_json(manifest_path, manifest)
        write_json(cleanup_path, cleanup)
        return {
            "run_id": run_id,
            "status": manifest["status"],
            "local_dir": str(local_dir),
            "cleanup": cleanup,
            "already_complete": False,
        }

    def cancel_run(self, run_id: str) -> dict[str, Any]:
        """Stop and delete a detached run, collecting any available state."""

        manifest_path, manifest = self._load_lifecycle_manifest(run_id)
        cleanup_path = manifest_path.parent / "cleanup.json"
        vm_name, zone = lifecycle_vm_identity(manifest)
        vm = self._describe_lifecycle_vm(vm_name, zone)
        if vm["state"] == "RUNNING":
            self.executor.run(
                lifecycle_ssh_argv(
                    self.project,
                    vm_name,
                    zone,
                    ("sudo", "systemctl", "stop", _RUN_SERVICE),
                ),
                check=False,
                capture_output=True,
            )
            self._collect_lifecycle_files(vm_name, zone, manifest_path.parent)
        cleanup = self._delete_lifecycle_vm(vm_name, zone)
        manifest["status"] = "cancelled"
        manifest["phase"] = "cancelled"
        manifest["completed_at"] = datetime.now(UTC).isoformat()
        manifest["cleanup"] = cleanup
        write_json(manifest_path, manifest)
        write_json(cleanup_path, cleanup)
        return {
            "run_id": run_id,
            "status": "cancelled",
            "cleanup": cleanup,
        }

    def _load_lifecycle_manifest(self, run_id: str) -> tuple[Path, dict[str, Any]]:
        manifest_path = resolve_run_manifest(self.runs_root, run_id)
        manifest = parse_json_object(
            manifest_path.read_text(encoding="utf-8"),
            str(manifest_path),
        )
        gcp = manifest.get("gcp")
        manifest_project = gcp.get("project") if isinstance(gcp, dict) else None
        if manifest_project != self.project:
            raise RuntimeError(
                f"run {run_id} belongs to project {manifest_project!r}, not {self.project!r}"
            )
        return manifest_path, manifest

    def _describe_lifecycle_vm(self, vm_name: str, zone: str) -> dict[str, Any]:
        result = self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "compute",
                "instances",
                "describe",
                vm_name,
                f"--zone={zone}",
                "--format=json",
            ),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            diagnostic = "\n".join((result.stdout or "", result.stderr or ""))
            if "the resource" in diagnostic.lower() and "was not found" in diagnostic.lower():
                return {"state": "MISSING"}
            detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"cannot describe run VM {vm_name!r}; gcloud authentication or API "
                f"access may have failed: {detail or f'exit status {result.returncode}'}"
            )
        value = parse_json_object(result.stdout or "{}", "run VM metadata")
        return {"state": str(value.get("status", "UNKNOWN")), "metadata": value}

    def _read_remote_status(self, vm_name: str, zone: str) -> dict[str, Any] | None:
        result = self.executor.run(
            lifecycle_ssh_argv(
                self.project,
                vm_name,
                zone,
                ("sudo", "cat", _RUN_STATUS_PATH),
            ),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or not (result.stdout or "").strip():
            return None
        return parse_json_object(result.stdout, "remote run status")

    def _read_remote_health(self, vm_name: str, zone: str) -> dict[str, Any] | None:
        result = self.executor.run(
            lifecycle_ssh_argv(self.project, vm_name, zone, ("sudo", "python3", "-")),
            input_text=remote_health_script(),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or not (result.stdout or "").strip():
            return None
        return parse_json_object(result.stdout, "remote service health")

    def monitor_once(self) -> dict[str, Any]:
        """Check local runs and discover unmatched project run VMs."""
        results: list[dict[str, Any]] = []
        known_vms: set[tuple[str, str]] = set()
        for path in sorted(self.runs_root.glob("*/*/*/run_manifest.json")):
            try:
                manifest = parse_json_object(path.read_text(encoding="utf-8"), str(path))
                gcp = manifest.get("gcp")
                if not isinstance(gcp, dict) or gcp.get("project") != self.project:
                    continue
                known_vms.add(lifecycle_vm_identity(manifest))
                cleanup_path = path.parent / "cleanup.json"
                if cleanup_path.exists():
                    cleanup = parse_json_object(cleanup_path.read_text(encoding="utf-8"), str(cleanup_path))
                    if cleanup.get("succeeded") is True:
                        continue
                run_id = manifest.get("run_id")
                if not isinstance(run_id, str):
                    raise ValueError("missing run ID")
                result = self.status_run(run_id)
                result["flag"] = (
                    "issue" if cleanup_path.exists()
                    else "complete" if result["status"] == "succeeded"
                    else "issue" if result["status"] in {"issue", "failed"}
                    else None
                )
                results.append(result)
            except (OSError, RuntimeError, ValueError) as error:
                results.append({"manifest_path": str(path), "flag": "issue", "error": str(error)})
        try:
            for vm_name, zone, state in self._list_run_vms():
                if (vm_name, zone) in known_vms:
                    continue
                health = self._read_remote_health(vm_name, zone) if state == "RUNNING" else None
                results.append({
                    "vm_name": vm_name, "vm_state": state, "zone": zone,
                    "health": health, "phase": "orphan_run_vm", "flag": "issue",
                })
        except (OSError, RuntimeError, ValueError) as error:
            results.append({"phase": "vm_inventory_unavailable", "flag": "issue", "error": str(error)})
        return {"checked_at": datetime.now(UTC).isoformat(), "runs": results}

    def _list_run_vms(self) -> list[tuple[str, str, str]]:
        result = self.executor.run(
            ("gcloud", "--project", self.project, "compute", "instances", "list", "--format=json"),
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"cannot list run VMs: {(result.stderr or '').strip()}")
        values = json.loads(result.stdout or "[]")
        if not isinstance(values, list):
            raise ValueError("run VM inventory must be a list")
        run_vms: list[tuple[str, str, str]] = []
        for vm in values:
            if not isinstance(vm, dict):
                continue
            name, zone, labels = vm.get("name"), vm.get("zone"), vm.get("labels")
            if (isinstance(name, str) and name.startswith("arb-run-")
                    and isinstance(zone, str) and isinstance(labels, dict)
                    and labels.get("benchmark") == "bixbench3"):
                run_vms.append((name, zone.rsplit("/", 1)[-1], str(vm.get("status", "UNKNOWN"))))
        return run_vms

    def _collect_lifecycle_files(self, vm_name: str, zone: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for remote_directory in ("artifact_grades", "inspect_logs"):
            self.executor.run(
                (
                    "gcloud",
                    "--project",
                    self.project,
                    "compute",
                    "scp",
                    "--recurse",
                    f"{vm_name}:{self.settings.remote_root}/{remote_directory}",
                    str(local_dir),
                    f"--zone={zone}",
                ),
                check=False,
            )
        for filename in ("grade.json", "judge.json"):
            self.executor.run(
                (
                    "gcloud",
                    "--project",
                    self.project,
                    "compute",
                    "scp",
                    f"{vm_name}:{self.settings.remote_root}/{filename}",
                    str(local_dir / filename),
                    f"--zone={zone}",
                ),
                check=False,
            )
        self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "compute",
                "scp",
                f"{vm_name}:{self.settings.remote_root}/run_manifest.json",
                str(local_dir / "remote_run_manifest.json"),
                f"--zone={zone}",
            ),
            check=False,
        )

    def _delete_lifecycle_vm(self, vm_name: str, zone: str) -> dict[str, Any]:
        if not vm_name.startswith("arb-run-"):
            raise ValueError("only BixBench3 run VMs may be deleted")
        existing = self._describe_lifecycle_vm(vm_name, zone)
        if existing["state"] == "MISSING":
            return {
                "vm_name": vm_name,
                "action": "already_missing",
                "succeeded": True,
                "ongoing_disk_cost_warning": False,
            }
        result = self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "compute",
                "instances",
                "delete",
                vm_name,
                f"--zone={zone}",
                "--quiet",
            ),
            check=False,
            capture_output=True,
        )
        return {
            "vm_name": vm_name,
            "action": "deleted" if result.returncode == 0 else "delete_failed",
            "succeeded": result.returncode == 0,
            "ongoing_disk_cost_warning": result.returncode != 0,
        }

    def setup_runtime(
        self,
        *,
        cellranger_tarball: Path,
        cellranger_arc_tarball: Path,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        cellranger_sha = validate_vendor_tarball(
            cellranger_tarball,
            expected_name="cellranger-6.0.2",
        )
        cellranger_arc_sha = validate_vendor_tarball(
            cellranger_arc_tarball,
            expected_name="cellranger-arc-2.0.2",
        )
        with tempfile.TemporaryDirectory(prefix="bixbench3-runtime-source-") as temporary:
            source_archive = Path(temporary) / "bixbench3-source.tar.gz"
            create_runtime_source_archive(self.repo_root, source_archive)
            source_archive_sha = file_sha256(source_archive)
            plan = build_runtime_plan(
                project=self.project,
                settings=self.settings,
                source_archive=source_archive,
                source_archive_sha256=source_archive_sha,
                cellranger_tarball=cellranger_tarball,
                cellranger_arc_tarball=cellranger_arc_tarball,
                cellranger_sha256=cellranger_sha,
                cellranger_arc_sha256=cellranger_arc_sha,
            )
            return self._execute_runtime_plan(
                plan,
                dry_run=dry_run,
            )

    def _execute_runtime_plan(
        self,
        plan: RuntimeBuildPlan,
        *,
        dry_run: bool,
    ) -> dict[str, Any]:
        if dry_run:
            summary = plan.public_summary()
            summary["dry_run"] = True
            return summary
        print(
            f"[1/10] checking GCP project {self.project} and required APIs",
            file=sys.stderr,
            flush=True,
        )
        self._preflight(require_artifact_registry=False, results_bucket=None)
        print(
            f"[2/10] checking for existing custom image {plan.image_name}",
            file=sys.stderr,
            flush=True,
        )
        existing = self.executor.run(
            plan.commands["describe_image"],
            check=False,
            capture_output=True,
        )
        if existing.returncode == 0:
            print(
                f"runtime image {plan.image_name} already exists; nothing to build",
                file=sys.stderr,
                flush=True,
            )
            return {
                "status": "already_exists",
                "image_name": plan.image_name,
                "project": self.project,
            }

        builder_created = False
        image_created = False
        try:
            print(
                f"[3/10] creating temporary builder VM {plan.builder_vm} in {plan.zone}",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["create_builder"])
            builder_created = True
            print(
                "[4/10] waiting for the builder VM to accept SSH connections",
                file=sys.stderr,
                flush=True,
            )
            self._wait_for_ssh(plan.commands["wait_builder_ssh"])
            print(
                "[5/10] installing Docker and preparing the builder VM",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(
                plan.commands["prepare_builder"],
                input_text=remote_builder_prepare_script(),
            )
            print(
                "[6/10] uploading the runner source and licensed 10x archives",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["copy_source_archive"])
            self.executor.run(plan.commands["copy_cellranger"])
            self.executor.run(plan.commands["copy_cellranger_arc"])
            print(
                "[7/10] building the runtime; dependency and container downloads may take several minutes",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(
                plan.commands["build_runtime"],
                input_text=remote_runtime_build_script(
                    source_archive_sha256=plan.source_archive_sha256,
                    cellranger_sha256=plan.cellranger_sha256,
                    cellranger_arc_sha256=plan.cellranger_arc_sha256,
                    image_tag=f"bixbench3-sandbox:{__version__}",
                ),
            )
            print(
                "[8/10] smoke-testing Cell Ranger, the network gateway, and the CLI",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(
                plan.commands["smoke_test"],
                input_text=remote_runtime_smoke_script(
                    sandbox_image_tag=f"bixbench3-sandbox:{__version__}",
                    network_gateway_image_tag=(f"bixbench3-network-gateway:{__version__}"),
                ),
            )
            print(
                f"[9/10] stopping {plan.builder_vm} and saving custom image {plan.image_name}",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["stop_builder"])
            self.executor.run(plan.commands["create_image"])
            image_created = True
        finally:
            if builder_created:
                print(
                    f"[10/10] deleting temporary builder VM {plan.builder_vm}",
                    file=sys.stderr,
                    flush=True,
                )
                self.executor.run(plan.commands["delete_builder"], check=False)
        if not image_created:
            raise RuntimeError("runtime image build failed")
        print(
            f"runtime image {plan.image_name} is ready",
            file=sys.stderr,
            flush=True,
        )
        return {
            "status": "created",
            "image_name": plan.image_name,
            "project": self.project,
            "source_archive_sha256": plan.source_archive_sha256,
            "cellranger_sha256": plan.cellranger_sha256,
            "cellranger_arc_sha256": plan.cellranger_arc_sha256,
        }

    def _preflight(
        self,
        *,
        require_artifact_registry: bool,
        results_bucket: str | None,
    ) -> dict[str, str]:
        self.executor.run(("gcloud", "version"), capture_output=True)
        account_result = self.executor.run(
            (
                "gcloud",
                "auth",
                "list",
                "--filter=status:ACTIVE",
                "--format=value(account)",
            ),
            capture_output=True,
        )
        account = (account_result.stdout or "").strip()
        if not account:
            raise RuntimeError("no active gcloud account; run `gcloud auth login`")
        project_result = self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "projects",
                "describe",
                self.project,
                "--format=value(projectNumber)",
            ),
            capture_output=True,
        )
        project_number = (project_result.stdout or "").strip()
        if not project_number.isdigit():
            raise RuntimeError(
                f"cannot resolve project {self.project!r}; verify access with "
                f"`gcloud projects describe {self.project}`"
            )
        required_services = ["compute.googleapis.com"]
        if require_artifact_registry:
            required_services.append("artifactregistry.googleapis.com")
        services_result = self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "services",
                "list",
                "--enabled",
                "--format=value(config.name)",
            ),
            capture_output=True,
        )
        enabled = set((services_result.stdout or "").splitlines())
        missing = [service for service in required_services if service not in enabled]
        if missing:
            enable = " ".join(missing)
            raise RuntimeError(
                f"missing required GCP APIs: {', '.join(missing)}; run "
                f"`gcloud services enable --project {self.project} {enable}`"
            )
        if results_bucket is not None:
            bucket, _ = parse_results_bucket(results_bucket)
            self.executor.run(
                (
                    "gcloud",
                    "--project",
                    self.project,
                    "storage",
                    "buckets",
                    "describe",
                    f"gs://{bucket}",
                ),
                capture_output=True,
            )
        return {"account": account, "project_number": project_number}

    def _require_runtime_image(self) -> dict[str, JsonValue]:
        result = self.executor.run(
            (
                "gcloud",
                "--project",
                self.project,
                "compute",
                "images",
                "describe",
                self.settings.runtime_image_name,
                "--format=json",
            ),
            capture_output=True,
        )
        try:
            value = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as error:
            raise RuntimeError("gcloud returned invalid runtime image metadata") from error
        if not isinstance(value, dict) or not value.get("id"):
            raise RuntimeError(
                f"private runtime image {self.settings.runtime_image_name!r} is missing; "
                "run `bixbench3 setup-runtime ...` first"
            )
        return cast(dict[str, JsonValue], value)

    def _ensure_data_disk(self, plan: RunPlan) -> None:
        describe = self.executor.run(
            plan.commands["describe_data_disk"],
            check=False,
            capture_output=True,
        )
        if describe.returncode == 0:
            metadata = parse_json_object(describe.stdout, "data disk metadata")
            labels = metadata.get("labels")
            label_map = labels if isinstance(labels, dict) else {}
            expected_hash = plan.task.input_manifest_sha256[:12].lower()
            if label_map.get("manifest") != expected_hash:
                raise RuntimeError(
                    f"data disk {plan.layout.data_disk_name} exists but does not match "
                    "the selected release manifest"
                )
            if label_map.get("ready") == "true":
                print(
                    f"reusing verified input disk {plan.layout.data_disk_name}",
                    file=sys.stderr,
                    flush=True,
                )
                return
        else:
            print(
                f"creating {plan.task.disk_size_gb} GB input disk {plan.layout.data_disk_name}",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["create_data_disk"])

        loader_created = False
        try:
            print(
                f"creating temporary loader VM {plan.layout.loader_vm_name}",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(plan.commands["create_loader_vm"])
            loader_created = True
            self._wait_for_ssh(plan.commands["wait_loader_ssh"])
            print(
                "downloading and checksumming public task inputs; large tasks can take several minutes",
                file=sys.stderr,
                flush=True,
            )
            self.executor.run(
                plan.commands["load_data_disk"],
                input_text=loader_copy_script(plan.task),
            )
            self.executor.run(plan.commands["mark_data_disk_ready"])
        finally:
            if loader_created:
                print(
                    f"deleting temporary loader VM {plan.layout.loader_vm_name}",
                    file=sys.stderr,
                    flush=True,
                )
                self.executor.run(plan.commands["delete_loader_vm"], check=False)

    def _wait_for_ssh(self, argv: Argv, *, attempts: int = 60) -> None:
        last_error = ""
        for _ in range(attempts):
            result = self.executor.run(argv, check=False, capture_output=True)
            if result.returncode == 0:
                return
            last_error = (result.stderr or result.stdout or "").strip()
            time.sleep(5)
        raise RuntimeError(f"VM did not become reachable over gcloud SSH: {last_error}")

    def _collect_scores(self, plan: RunPlan) -> None:
        plan.layout.local_dir.mkdir(parents=True, exist_ok=True)
        self.executor.run(plan.commands["collect_scores"], check=False)

    def _cleanup_run_vm(self, plan: RunPlan, *, succeeded: bool) -> dict[str, Any]:
        if succeeded:
            result = self.executor.run(
                plan.commands["delete_run_vm"],
                check=False,
                capture_output=True,
            )
            return {
                "vm_name": plan.layout.vm_name,
                "action": "deleted" if result.returncode == 0 else "delete_failed",
                "succeeded": result.returncode == 0,
                "ongoing_disk_cost_warning": result.returncode != 0,
            }
        result = self.executor.run(
            plan.commands["stop_run_vm"],
            check=False,
            capture_output=True,
        )
        return {
            "vm_name": plan.layout.vm_name,
            "action": "stopped" if result.returncode == 0 else "stop_failed",
            "succeeded": result.returncode == 0,
            "ongoing_disk_cost_warning": True,
            "warning": "The stopped VM retains its 500 GB boot disk and continues to incur disk charges.",
        }

    def _initial_manifest(
        self,
        plan: RunPlan,
        preflight: Mapping[str, str],
        *,
        staged_case: StagedCase,
    ) -> dict[str, Any]:
        image = self._require_runtime_image()
        return {
            "schema_version": 1,
            "benchmark_version": BENCHMARK_VERSION,
            "package_version": __version__,
            "git_commit": git_commit(self.repo_root),
            "hf_dataset": HF_DATASET,
            "hf_revision": plan.task.hf_revision,
            "hf_case_files_sha256": staged_case.file_sha256,
            "gcs_release": plan.task.gcs_release,
            "input_manifest_sha256": plan.task.input_manifest_sha256,
            "ground_truth_manifest_sha256": plan.task.grader_manifest_sha256,
            "runtime_image_name": self.settings.runtime_image_name,
            "runtime_image_id": image.get("id"),
            "task_id": plan.task.task_id,
            "paper_id": plan.task.paper_id,
            "run_id": plan.layout.run_id,
            "model": {
                "id": plan.model.model_id,
                "reference_profile": plan.model.is_reference,
                "arguments": plan.model.model_args,
                "fallback_models": list(plan.model.fallback_model_ids),
            },
            "effort": {
                "requested": plan.model.requested_effort,
                "resolved": plan.model.resolved_effort,
            },
            "limits": {
                "time_seconds": self.settings.time_limit_seconds,
                "messages": self.settings.message_limit,
            },
            "gcp": {
                "project": self.project,
                "zone": self.settings.zone,
                "machine_type": self.settings.machine_type,
                "boot_disk_gb": self.settings.boot_disk_gb,
                "data_disk": plan.layout.data_disk_name,
                "data_disk_mode": "read-only",
                "vm_name": plan.layout.vm_name,
                "account": preflight["account"],
            },
            "retention": {
                "mode": "full_bundle" if plan.results_uri else "scores_only",
                "results_uri": plan.results_uri,
            },
            "process_judge": process_judge_config(),
            "started_at": datetime.now(UTC).isoformat(),
            "status": "created",
            "phase": "provisioning",
        }


def build_run_commands(
    *,
    project: str,
    settings: RuntimeSettings,
    task: TaskSpec,
    layout: RunLayout,
    service_account: str | None,
    results_uri: str | None,
    case_dataset_dir: Path | None,
) -> dict[str, Argv]:
    base = ("gcloud", "--project", project)
    zone = f"--zone={settings.zone}"
    labels = (
        f"benchmark=bixbench3,release={slugify(BENCHMARK_VERSION)},"
        f"task={slugify(task.paper_id, maximum=28)},manifest={task.input_manifest_sha256[:12].lower()}"
    )
    case_dataset_source = (
        str(case_dataset_dir.resolve())
        if case_dataset_dir is not None
        else "<LOCAL_ONE_CASE_DATASET>"
    )
    run_create = [
        *base,
        "compute",
        "instances",
        "create",
        layout.vm_name,
        zone,
        f"--machine-type={settings.machine_type}",
        f"--boot-disk-size={settings.boot_disk_gb}GB",
        "--boot-disk-type=pd-balanced",
        (
            "--image="
            f"projects/{project}/global/images/{settings.runtime_image_name}"
        ),
        f"--labels={labels},run={slugify(layout.run_id, maximum=20)}",
    ]
    if service_account is None:
        run_create.extend(("--no-service-account", "--no-scopes"))
    else:
        run_create.extend(
            (f"--service-account={service_account}", f"--scopes={_CLOUD_PLATFORM_SCOPE}")
        )
    remote_root = settings.remote_root
    commands: dict[str, Argv] = {
        "check_gcloud": ("gcloud", "version"),
        "check_active_account": (
            "gcloud",
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
        ),
        "check_project": (
            *base,
            "projects",
            "describe",
            project,
            "--format=value(projectNumber)",
        ),
        "check_compute_api": (
            *base,
            "services",
            "list",
            "--enabled",
            "--filter=config.name:compute.googleapis.com",
            "--format=value(config.name)",
        ),
        "describe_runtime_image": (
            *base,
            "compute",
            "images",
            "describe",
            settings.runtime_image_name,
            "--format=json",
        ),
        "describe_data_disk": (
            *base,
            "compute",
            "disks",
            "describe",
            layout.data_disk_name,
            zone,
            "--format=json",
        ),
        "create_data_disk": (
            *base,
            "compute",
            "disks",
            "create",
            layout.data_disk_name,
            zone,
            "--type=pd-balanced",
            f"--size={task.disk_size_gb}GB",
            f"--labels={labels},ready=false",
        ),
        "create_loader_vm": (
            *base,
            "compute",
            "instances",
            "create",
            layout.loader_vm_name,
            zone,
            "--machine-type=e2-standard-4",
            "--boot-disk-size=20GB",
            "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud",
            (f"--disk=name={layout.data_disk_name},device-name={_DEVICE_NAME},mode=rw,boot=no"),
            "--no-service-account",
            "--no-scopes",
            f"--labels={labels},role=loader",
        ),
        "wait_loader_ssh": ssh_argv(base, layout.loader_vm_name, settings.zone, ("true",)),
        "load_data_disk": ssh_argv(
            base,
            layout.loader_vm_name,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "mark_data_disk_ready": (
            *base,
            "compute",
            "disks",
            "add-labels",
            layout.data_disk_name,
            zone,
            "--labels=ready=true",
        ),
        "delete_loader_vm": (
            *base,
            "compute",
            "instances",
            "delete",
            layout.loader_vm_name,
            zone,
            "--quiet",
        ),
        "create_run_vm": tuple(run_create),
        "wait_run_ssh": ssh_argv(base, layout.vm_name, settings.zone, ("true",)),
        "attach_data_disk": (
            *base,
            "compute",
            "instances",
            "attach-disk",
            layout.vm_name,
            zone,
            f"--disk={layout.data_disk_name}",
            "--mode=ro",
            f"--device-name={_DEVICE_NAME}",
        ),
        "prepare_run_vm": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "copy_run_manifest": (
            *base,
            "compute",
            "scp",
            str(layout.manifest_path),
            f"{layout.vm_name}:{remote_root}/run_manifest.json",
            zone,
        ),
        "copy_case_dataset": (
            *base,
            "compute",
            "scp",
            "--recurse",
            case_dataset_source,
            f"{layout.vm_name}:{remote_root}/",
            zone,
        ),
        "stream_environment": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            ("sudo", "sh", "-c", f"umask 077; cat > {_SECRET_ENV_PATH}"),
        ),
        "preflight_results_write": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "install_run_script": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            (
                "sudo",
                "sh",
                "-c",
                f"umask 077; cat > {_RUN_SCRIPT_PATH}; chmod 0700 {_RUN_SCRIPT_PATH}",
            ),
        ),
        "install_run_service": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            (
                "sudo",
                "sh",
                "-c",
                f"cat > {_RUN_SERVICE_PATH}; chmod 0644 {_RUN_SERVICE_PATH}",
            ),
        ),
        "start_run_service": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            (
                "sudo",
                "sh",
                "-c",
                f"systemctl daemon-reload && systemctl enable --now {_RUN_SERVICE}",
            ),
        ),
        "scrub_run_credentials": ssh_argv(
            base,
            layout.vm_name,
            settings.zone,
            (
                "sudo",
                "sh",
                "-c",
                f"systemctl disable --now {_RUN_SERVICE} 2>/dev/null || true; "
                f"rm -f {_SECRET_ENV_PATH}; touch {_RUN_FINISHED_PATH}",
            ),
        ),
        "collect_scores": (
            *base,
            "compute",
            "scp",
            "--recurse",
            f"{layout.vm_name}:{remote_root}/artifact_grades",
            str(layout.local_dir),
            zone,
        ),
        "delete_run_vm": (
            *base,
            "compute",
            "instances",
            "delete",
            layout.vm_name,
            zone,
            "--quiet",
        ),
        "stop_run_vm": (
            *base,
            "compute",
            "instances",
            "stop",
            layout.vm_name,
            zone,
            "--quiet",
        ),
    }
    if results_uri is not None:
        bucket, _ = parse_results_bucket(results_uri)
        commands["describe_results_bucket"] = (
            *base,
            "storage",
            "buckets",
            "describe",
            f"gs://{bucket}",
        )
    return commands


def build_inspect_argv(
    *,
    settings: RuntimeSettings,
    task: TaskSpec,
    model: ResolvedModel,
) -> Argv:
    remote_root = settings.remote_root
    parts = [
        "uv",
        "run",
        "--project",
        "/opt/bixbench3",
        "inspect",
        "eval",
        (
            "/opt/bixbench3/src/bixbench3/"
            "inspect_harness/tasks.py@bixbench3_cases"
        ),
        "--model",
        model.model_id,
        "--sample-id",
        task.paper_id,
        "--log-dir",
        f"{remote_root}/inspect_logs",
        "--log-format",
        "eval",
        "--max-retries",
        "5",
        "--retry-on-error",
        "2",
        "--score-on-error",
        "-T",
        f"dataset_path={remote_root}/dataset/case.json",
        "-T",
        f"effort={model.resolved_effort}",
        "--metadata",
        f"paper_id={task.paper_id}",
        "--metadata",
        f"requested_effort={model.requested_effort}",
        "--metadata",
        f"resolved_effort={model.resolved_effort}",
    ]
    for key, value in sorted(model.model_args.items()):
        if _MODEL_ARG_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid Inspect model argument name {key!r}")
        parts.extend(("-M", f"{key}={json.dumps(value, separators=(',', ':'))}"))
    if model.fallback_model_ids:
        fallback_targets = [
            fallback_model_id.removeprefix("anthropic/")
            for fallback_model_id in model.fallback_model_ids
        ]
        parts.extend(("--fallback-models", ",".join(fallback_targets)))
    parts.extend(model.inspect_effort_args)
    return tuple(parts)


def build_runtime_plan(
    *,
    project: str,
    settings: RuntimeSettings,
    source_archive: Path,
    source_archive_sha256: str,
    cellranger_tarball: Path,
    cellranger_arc_tarball: Path,
    cellranger_sha256: str,
    cellranger_arc_sha256: str,
) -> RuntimeBuildPlan:
    build_hash = hashlib.sha256(
        (
            f"{source_archive_sha256}\0{cellranger_sha256}\0{cellranger_arc_sha256}\0{__version__}"
        ).encode()
    ).hexdigest()[:10]
    builder = f"arb-runtime-builder-{build_hash}"
    base = ("gcloud", "--project", project)
    zone = f"--zone={settings.zone}"
    commands: dict[str, Argv] = {
        "check_gcloud": ("gcloud", "version"),
        "check_active_account": (
            "gcloud",
            "auth",
            "list",
            "--filter=status:ACTIVE",
            "--format=value(account)",
        ),
        "check_project": (
            *base,
            "projects",
            "describe",
            project,
            "--format=value(projectNumber)",
        ),
        "check_compute_api": (
            *base,
            "services",
            "list",
            "--enabled",
            "--filter=config.name:compute.googleapis.com",
            "--format=value(config.name)",
        ),
        "describe_image": (
            *base,
            "compute",
            "images",
            "describe",
            settings.runtime_image_name,
            "--format=json",
        ),
        "create_builder": (
            *base,
            "compute",
            "instances",
            "create",
            builder,
            zone,
            "--machine-type=n2-standard-32",
            "--boot-disk-size=500GB",
            "--boot-disk-type=pd-balanced",
            "--image-family=ubuntu-2404-lts-amd64",
            "--image-project=ubuntu-os-cloud",
            "--no-service-account",
            "--no-scopes",
            f"--labels=benchmark=bixbench3,release={slugify(BENCHMARK_VERSION)},role=runtime-builder",
        ),
        "wait_builder_ssh": ssh_argv(base, builder, settings.zone, ("true",)),
        "prepare_builder": ssh_argv(
            base,
            builder,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "copy_source_archive": (
            *base,
            "compute",
            "scp",
            str(source_archive.resolve()),
            f"{builder}:~/bixbench3-source.tar.gz",
            zone,
            "--scp-flag=-O",
        ),
        "copy_cellranger": (
            *base,
            "compute",
            "scp",
            str(cellranger_tarball.resolve()),
            f"{builder}:~/cellranger-6.0.2.tar.gz",
            zone,
            "--scp-flag=-O",
        ),
        "copy_cellranger_arc": (
            *base,
            "compute",
            "scp",
            str(cellranger_arc_tarball.resolve()),
            f"{builder}:~/cellranger-arc-2.0.2.tar.gz",
            zone,
            "--scp-flag=-O",
        ),
        "build_runtime": ssh_argv(
            base,
            builder,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "smoke_test": ssh_argv(
            base,
            builder,
            settings.zone,
            ("sudo", "bash", "-s"),
        ),
        "stop_builder": (
            *base,
            "compute",
            "instances",
            "stop",
            builder,
            zone,
            "--quiet",
        ),
        "create_image": (
            *base,
            "compute",
            "images",
            "create",
            settings.runtime_image_name,
            f"--source-disk={builder}",
            f"--source-disk-zone={settings.zone}",
            f"--family={settings.runtime_image_family}",
            (f"--description=BixBench3 {BENCHMARK_VERSION}; runtime hash {build_hash}"),
        ),
        "delete_builder": (
            *base,
            "compute",
            "instances",
            "delete",
            builder,
            zone,
            "--quiet",
        ),
    }
    for argv in commands.values():
        validate_safe_argv(argv, allow_ephemeral_instance_delete=True)
    return RuntimeBuildPlan(
        project=project,
        zone=settings.zone,
        builder_vm=builder,
        image_name=settings.runtime_image_name,
        commands=commands,
        source_archive_sha256=source_archive_sha256,
        cellranger_sha256=cellranger_sha256,
        cellranger_arc_sha256=cellranger_arc_sha256,
    )


def ssh_argv(
    base: tuple[str, ...],
    vm_name: str,
    zone: str,
    remote_argv: Sequence[str],
) -> Argv:
    return (
        *base,
        "compute",
        "ssh",
        vm_name,
        f"--zone={zone}",
        "--command",
        shlex.join(remote_argv),
    )


def validate_plan(plan: RunPlan) -> None:
    for name, argv in plan.commands.items():
        allow_delete = name in {"delete_loader_vm", "delete_run_vm"}
        validate_safe_argv(argv, allow_ephemeral_instance_delete=allow_delete)
        joined = "\0".join(argv)
        if INPUT_BUCKET in joined or GROUND_TRUTH_BUCKET in joined:
            if name not in {"load_data_disk"}:
                raise ValueError(f"{name} unexpectedly exposes a benchmark source bucket")
    for key in plan.model.required_environment_keys:
        if any(
            key in argument and argument != key
            for argv in plan.commands.values()
            for argument in argv
        ):
            raise ValueError("environment key material cannot appear in cloud command arguments")


def validate_safe_argv(
    argv: Sequence[str],
    *,
    allow_ephemeral_instance_delete: bool = False,
) -> None:
    if not argv or argv[0] != "gcloud":
        raise ValueError("cloud commands must be non-empty gcloud argv sequences")
    lowered = [value.lower() for value in argv]
    if "mv" in lowered:
        raise ValueError("gcloud storage mv is forbidden")
    if "rsync" in lowered or any(value == "--delete" for value in lowered):
        raise ValueError("synchronizing or deleting source objects is forbidden")
    if "storage" in lowered and "rm" in lowered:
        raise ValueError("gcloud storage rm is forbidden")
    if "disks" in lowered and "delete" in lowered:
        raise ValueError("persistent disk deletion is forbidden")
    if "instances" in lowered and "delete" in lowered:
        delete_index = lowered.index("delete")
        resource = argv[delete_index + 1] if len(argv) > delete_index + 1 else ""
        if not allow_ephemeral_instance_delete and not resource.startswith("arb-"):
            raise ValueError("only BixBench3 ephemeral VMs may be deleted")


def load_public_task(task: str) -> TaskSpec:
    """Resolve a task number or paper ID from the immutable public release."""
    task = task.strip()
    if not task:
        raise ValueError("task cannot be empty")
    paper_id = task
    if task.isdigit():
        mapping = load_hf_json("manifests/task-number-map.json")
        paper_id = resolve_task_number(mapping, int(task))
    runtime = load_hf_json(f"cases/{paper_id}/runtime/runtime.json")
    input_manifest_uri = string_choice(
        runtime,
        "input_manifest_uri",
        "gcs_input_manifest_uri",
        "input_manifest_url",
        default=(f"gs://{INPUT_BUCKET}/releases/{BENCHMARK_VERSION}/manifests/{paper_id}.json"),
    )
    input_manifest_url = gs_to_https(input_manifest_uri)
    manifest_bytes = fetch_bytes(input_manifest_url)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    input_release = fetch_json(input_release_manifest_url(BENCHMARK_VERSION))
    expected_sha = release_task_manifest_sha(input_release, paper_id)
    if expected_sha.lower() != manifest_sha:
        raise RuntimeError(
            f"input manifest checksum mismatch for task {task}: "
            f"expected {expected_sha}, got {manifest_sha}"
        )
    manifest = parse_json_bytes(manifest_bytes, input_manifest_url)
    input_size = integer_choice(runtime, "input_size_bytes", "total_input_bytes")
    if input_size is None:
        input_size = manifest_total_bytes(manifest)
    grader_manifest_url = string_choice(
        runtime,
        "grader_manifest_url",
        "ground_truth_manifest_url",
        default=(
            f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}/releases/"
            f"{BENCHMARK_VERSION}/manifests/{paper_id}.json"
        ),
    )
    ground_truth_release = fetch_json(
        f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}/releases/"
        f"{BENCHMARK_VERSION}/release.json"
    )
    grader_manifest_sha = release_task_manifest_sha(
        ground_truth_release,
        paper_id,
    )
    return TaskSpec(
        task_id=task,
        paper_id=paper_id,
        input_manifest_uri=input_manifest_uri,
        input_manifest_sha256=manifest_sha,
        input_size_bytes=input_size,
        grader_manifest_url=grader_manifest_url,
        grader_manifest_sha256=grader_manifest_sha,
    )


def stage_hf_case(task: TaskSpec, destination: Path) -> StagedCase:
    """Stage exactly one authenticated HF agent/runtime package locally."""
    if destination.exists():
        raise FileExistsError(f"one-case staging destination already exists: {destination}")
    release = load_hf_json("release.json")
    release_hashes = release.get("files_sha256")
    if not isinstance(release_hashes, dict):
        raise ValueError("HF release.json must contain files_sha256")
    copied_hashes: dict[str, str] = {}
    for package_relative in _HF_CASE_FILES:
        release_relative = f"cases/{task.paper_id}/{package_relative}"
        source = hf_download(release_relative)
        expected = release_hashes.get(release_relative)
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ValueError(f"HF release has no valid checksum for {release_relative}")
        actual = file_sha256(source)
        if actual != expected.lower():
            raise RuntimeError(
                f"HF package checksum mismatch for {release_relative}: "
                f"expected {expected}, got {actual}"
            )
        target = destination / release_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_sha256(target) != actual:
            raise RuntimeError(f"local HF package copy verification failed: {release_relative}")
        copied_hashes[release_relative] = actual

    runtime_path = destination / f"cases/{task.paper_id}/runtime/runtime.json"
    runtime = parse_json_object(
        runtime_path.read_text(encoding="utf-8"),
        str(runtime_path),
    )
    network_policy = runtime.get("network_policy")
    if not isinstance(network_policy, dict):
        raise ValueError("public runtime package has no network_policy")
    runtime_policy = dict(network_policy)
    runtime_policy.pop("fetch_instead_of_allow_domains", None)
    runtime["network_policy"] = runtime_policy
    runtime["task_id"] = task.paper_id
    runtime["prompt"] = {"prompt_path": f"cases/{task.paper_id}/agent/prompt.md"}
    dataset_path = destination / "case.json"
    dataset_path.write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    copied_hashes["case.json"] = file_sha256(dataset_path)
    return StagedCase(
        dataset_dir=destination,
        dataset_path=dataset_path,
        file_sha256=copied_hashes,
    )


def hf_download(filename: str) -> Path:
    """Download one file from the pinned public Hugging Face release."""
    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(
            repo_id=HF_DATASET,
            filename=filename,
            repo_type="dataset",
            revision=HF_REVISION,
            token=None,
        )
    except Exception as error:
        raise RuntimeError(
            f"Hugging Face download failed for {filename}; verify public access to "
            f"{HF_DATASET}@{HF_REVISION}"
        ) from error
    return Path(downloaded)


def load_hf_json(filename: str) -> dict[str, Any]:
    path = hf_download(filename)
    return parse_json_object(path.read_text(encoding="utf-8"), filename)


def resolve_task_number(value: dict[str, Any], task_number: int) -> str:
    direct = value.get(str(task_number))
    if isinstance(direct, str) and direct:
        return direct
    rows = value.get("tasks")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            number = row.get("task_number", row.get("task"))
            paper_id = row.get("paper_id")
            if number == task_number and isinstance(paper_id, str):
                return paper_id
    raise ValueError(f"unknown BixBench3 task number {task_number}")


def release_task_manifest_sha(value: dict[str, Any], paper_id: str) -> str:
    tasks = value.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("GCS release manifest must contain a tasks list")
    for task in tasks:
        if not isinstance(task, dict) or task.get("paper_id") != paper_id:
            continue
        digest = task.get("manifest_sha256")
        if isinstance(digest, str) and re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            return digest.lower()
        raise ValueError(f"GCS release entry for {paper_id} has no valid manifest SHA-256")
    raise ValueError(f"GCS release manifest does not contain task {paper_id}")


def grade_existing_artifacts(
    *,
    task: TaskSpec,
    artifacts_dir: Path,
    grades_dir: Path | None = None,
    ground_truth_parent: Path | None = None,
) -> dict[str, Any]:
    """Download isolated ground truth and grade an existing artifact directory."""
    if not artifacts_dir.is_dir():
        raise ValueError(f"artifacts directory does not exist: {artifacts_dir}")
    artifact_grades_dir = grades_dir or artifacts_dir.parent / "artifact_grades"
    if ground_truth_parent is not None:
        ground_truth_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(
        prefix="bixbench3-grader-",
        dir=ground_truth_parent,
    ) as temporary:
        ground_truth_root = Path(temporary)
        download_manifest_package(
            task.grader_manifest_url,
            ground_truth_root,
            expected_manifest_sha256=task.grader_manifest_sha256,
        )
        spec_candidates = list(ground_truth_root.rglob("artifacts_grading.json"))
        if len(spec_candidates) != 1:
            raise RuntimeError(
                "ground-truth package must contain exactly one artifacts_grading.json"
            )
        spec = parse_json_object(
            spec_candidates[0].read_text(encoding="utf-8"),
            str(spec_candidates[0]),
        )
        try:
            from .grading import grade_paper
        except ImportError as error:
            raise RuntimeError("grading package is not installed") from error
        grade = grade_paper(
            spec,
            agent_artifacts_dir=artifacts_dir,
            ground_truth_root=ground_truth_root,
            artifact_grades_dir=artifact_grades_dir,
        )
    return cast(dict[str, Any], grade)


def download_manifest_package(
    manifest_url: str,
    destination: Path,
    *,
    expected_manifest_sha256: str,
) -> None:
    manifest_bytes = fetch_bytes(gs_to_https(manifest_url))
    actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha != expected_manifest_sha256.lower():
        raise RuntimeError(
            "ground-truth manifest checksum mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha}"
        )
    manifest = parse_json_bytes(manifest_bytes, manifest_url)
    entries = manifest_entries(manifest)
    destination.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        url = optional_string(entry, "public_url", "url", "https_url")
        object_path = optional_string(entry, "object_path", "path", "relative_path")
        if url is None or object_path is None:
            raise ValueError(f"{manifest_url}: each object needs a URL and path")
        relative = package_relative_path(object_path)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = fetch_bytes(gs_to_https(url))
        expected_sha = optional_string(entry, "sha256")
        actual_sha = hashlib.sha256(payload).hexdigest()
        if expected_sha is not None and actual_sha != expected_sha.lower():
            raise RuntimeError(
                f"checksum mismatch for {url}: expected {expected_sha}, got {actual_sha}"
            )
        target.write_bytes(payload)


def remote_prepare_script(settings: RuntimeSettings) -> str:
    remote_root = shlex.quote(settings.remote_root)
    mount = shlex.quote(_DATA_MOUNT)
    return f"""set -euo pipefail
mkdir -p /etc/needrestart/conf.d
cat > /etc/needrestart/conf.d/bixbench3.conf <<'EOF'
$nrconf{{restart}} = 'l';
$nrconf{{override_rc}}{{qr(^(bixbench3-run|docker|containerd)\\.service$)}} = 0;
EOF
systemctl mask --now unattended-upgrades.service apt-daily.timer apt-daily-upgrade.timer
install -d -m 0700 /run/bixbench3
mkdir -p {remote_root}/artifact_grades {remote_root}/agent_artifacts {remote_root}/inspect_logs {remote_root}/outputs {remote_root}/broker {remote_root}/broker_logs {remote_root}/broker_cache {remote_root}/workspace/approved_web {mount}
device=/dev/disk/by-id/google-{_DEVICE_NAME}
for attempt in $(seq 1 30); do
  [ -e "$device" ] && break
  sleep 2
done
test -e "$device"
mountpoint -q {mount} || mount -o ro,noload "$device" {mount}
test -d {mount}/data
remote_user="${{SUDO_USER:-root}}"
chown -R "$remote_user:$remote_user" {remote_root}
chmod 0750 {remote_root}
chmod -R a+rwX {remote_root}/outputs {remote_root}/broker_logs {remote_root}/broker_cache {remote_root}/workspace
if [ -S /var/run/docker.sock ]; then
  chmod 0666 /var/run/docker.sock
fi
"""


def loader_copy_script(task: TaskSpec) -> str:
    manifest_url = gs_to_https(task.input_manifest_uri)
    return f"""set -euo pipefail
device=/dev/disk/by-id/google-{_DEVICE_NAME}
for attempt in $(seq 1 30); do
  [ -e "$device" ] && break
  sleep 2
done
test -e "$device"
if ! blkid "$device" >/dev/null 2>&1; then
  mkfs.ext4 -F "$device"
fi
mkdir -p {_DATA_MOUNT}
mountpoint -q {_DATA_MOUNT} || mount "$device" {_DATA_MOUNT}
python3 - {shlex.quote(manifest_url)} {shlex.quote(_DATA_MOUNT)} {shlex.quote(task.input_manifest_sha256)} <<'PY'
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

manifest_url, mount_text, expected_manifest_sha = sys.argv[1:]
mount = Path(mount_text)
request = Request(manifest_url, headers={{"User-Agent": "bixbench3/{__version__}"}})
with urlopen(request, timeout=120) as response:
    manifest_bytes = response.read()
actual_manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
if actual_manifest_sha != expected_manifest_sha:
    raise RuntimeError(
        f"manifest checksum mismatch: expected {{expected_manifest_sha}}, got {{actual_manifest_sha}}"
    )
manifest = json.loads(manifest_bytes)
entries = manifest.get("objects") or manifest.get("files") or manifest.get("artifacts")
if not isinstance(entries, list):
    raise RuntimeError("input manifest has no object list")

def encoded_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        quote(parts.path, safe="/%"),
        quote(parts.query, safe="=&%"),
        parts.fragment,
    ))

def relative_path(entry: dict[str, object]) -> PurePosixPath:
    raw = entry.get("relative_path") or entry.get("object_path") or entry.get("path")
    if not isinstance(raw, str):
        raise RuntimeError("manifest object has no path")
    marker = "/data/"
    value = "data/" + raw.split(marker, 1)[1] if marker in raw else raw.lstrip("/")
    result = PurePosixPath(value)
    if result.is_absolute() or ".." in result.parts:
        raise RuntimeError(f"unsafe manifest path: {{raw}}")
    return result

for entry_value in entries:
    if not isinstance(entry_value, dict):
        raise RuntimeError("manifest entries must be mappings")
    entry = entry_value
    url = entry.get("public_url") or entry.get("url") or entry.get("https_url")
    expected_sha = entry.get("sha256")
    if not isinstance(url, str) or not isinstance(expected_sha, str):
        raise RuntimeError("manifest object needs public_url and sha256")
    destination = mount / relative_path(entry)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual == expected_sha.lower():
            continue
        raise RuntimeError(f"existing destination checksum mismatch: {{destination}}")
    partial = destination.with_name(destination.name + ".partial")
    request = Request(encoded_url(url), headers={{"User-Agent": "bixbench3/{__version__}"}})
    digest = hashlib.sha256()
    with urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != expected_sha.lower():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download checksum mismatch: {{url}}")
    partial.replace(destination)

(mount / ".bixbench3-ready.json").write_text(
    json.dumps({{
        "task_id": {task.task_id!r},
        "paper_id": {task.paper_id!r},
        "manifest_sha256": expected_manifest_sha,
    }}, indent=2) + "\\n",
    encoding="utf-8",
)
PY
sync
umount {_DATA_MOUNT}
"""


def remote_health_script() -> str:
    return """import json
import os
from pathlib import Path
import subprocess

state = subprocess.run(
    ("systemctl", "is-active", "bixbench3-run.service"),
    capture_output=True, text=True, check=False,
).stdout.strip()
main = subprocess.run(
    ("systemctl", "show", "bixbench3-run.service", "-p", "MainPID", "--value"),
    capture_output=True, text=True, check=False,
).stdout.strip()
main_pid = int(main) if main.isdigit() else 0
processes = {}
for entry in os.scandir("/proc"):
    if not entry.name.isdigit():
        continue
    try:
        stat = (Path(entry.path) / "stat").read_text().rpartition(") ")[2].split()
        args = (Path(entry.path) / "cmdline").read_bytes().replace(b"\\0", b" ").decode(errors="replace")
        processes[int(entry.name)] = (int(stat[1]), int(stat[11]) + int(stat[12]), args)
    except (OSError, IndexError, ValueError):
        continue
by_parent = {}
for pid, (parent, _, _) in processes.items():
    by_parent.setdefault(parent, []).append(pid)
children = set()
pending = [main_pid] if main_pid else []
while pending:
    pid = pending.pop()
    if pid not in children:
        children.add(pid)
        pending.extend(by_parent.get(pid, []))
inspect = [pid for pid in children if pid in processes and "inspect eval" in processes[pid][2]]
print(json.dumps({
    "service_state": state,
    "inspect_running": bool(inspect),
    "inspect_pid": inspect[0] if inspect else None,
    "cpu_ticks": sum(processes[pid][1] for pid in children if pid in processes),
}))
"""


def remote_run_service_unit() -> str:
    """Return the persistent systemd unit used to survive VM reboots."""
    return f"""[Unit]
Description=BixBench3 benchmark run
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service
ConditionPathExists={_SECRET_ENV_PATH}
ConditionPathExists=!{_RUN_FINISHED_PATH}

[Service]
Type=exec
ExecStart=/bin/bash {_RUN_SCRIPT_PATH}
ExecStopPost=/usr/bin/rm -f {_SECRET_ENV_PATH}
Restart=no

[Install]
WantedBy=multi-user.target
"""


def remote_run_script(plan: RunPlan, settings: RuntimeSettings) -> str:
    inspect_command = shlex.join(plan.inspect_argv)
    remote_root = shlex.quote(settings.remote_root)
    results_uri = shlex.quote(plan.results_uri) if plan.results_uri else ""
    upload = ""
    if plan.results_uri:
        upload = f"""
write_status uploading running "$run_status"
upload_status=0
for name in inspect_logs outputs agent_artifacts artifact_grades grade.json judge.json broker_logs run_manifest.json; do
  if [ -e {remote_root}/"$name" ]; then
    gcloud storage cp --recursive --no-clobber {remote_root}/"$name" {results_uri}/ || upload_status=$?
  fi
done
if [ "$upload_status" -ne 0 ]; then
  run_status="$upload_status"
fi
"""
    upload_if_redacted = (
        f"""if [ "$redaction_status" -eq 0 ]; then
{upload}
fi
"""
        if upload
        else ""
    )
    return f"""set -uo pipefail
write_status() {{
  python3 - "$1" "$2" "$3" {_RUN_STATUS_PATH} <<'PY'
from __future__ import annotations
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

phase, status, raw_exit_status, path_text = sys.argv[1:]
path = Path(path_text)
value = {{
    "phase": phase,
    "status": status,
    "updated_at": datetime.now(UTC).isoformat(),
    "remote_exit_status": int(raw_exit_status) if raw_exit_status else None,
    "systemd_unit": {_RUN_SERVICE!r},
}}
temporary = path.with_name("." + path.name + ".tmp")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
temporary.replace(path)
PY
}}
run_status=0
reached_end=0
finish_run() {{
  exit_status=$?
  rm -f {_SECRET_ENV_PATH}
  touch {_RUN_FINISHED_PATH}
  systemctl disable {_RUN_SERVICE} >/dev/null 2>&1 || true
  if [ "$reached_end" -ne 1 ]; then
    write_status interrupted issue "$exit_status"
  elif [ "$exit_status" -eq 0 ]; then
    write_status completed succeeded "$exit_status"
  else
    write_status failed failed "$exit_status"
  fi
}}
trap finish_run EXIT
touch {_RUN_FINISHED_PATH}

device=/dev/disk/by-id/google-{_DEVICE_NAME}
mkdir -p {_DATA_MOUNT}
mountpoint -q {_DATA_MOUNT} || mount -o ro,noload "$device" {_DATA_MOUNT} || exit 96
test -d {_DATA_MOUNT}/data || exit 96

write_status agent_running running ""
test -r {_SECRET_ENV_PATH}
export BIXBENCH3_DATA_DIR={shlex.quote(_DATA_MOUNT + "/data")}
export BIXBENCH3_OUTPUTS_DIR={settings.remote_root}/outputs
export BIXBENCH3_WORKSPACE_DIR={settings.remote_root}/workspace
export BIXBENCH3_INSPECT_SANDBOX_COMPOSE=/opt/bixbench3/sandbox/compose.yaml
export BIXBENCH3_SANDBOX_IMAGE=bixbench3-sandbox:{__version__}
export BIXBENCH3_NETWORK_GATEWAY_IMAGE=bixbench3-network-gateway:{__version__}
export BIXBENCH3_BROKER_POLICY={settings.remote_root}/broker/policy.json
export BIXBENCH3_BROKER_LOG_DIR={settings.remote_root}/broker_logs
export BIXBENCH3_BROKER_CACHE_DIR={settings.remote_root}/broker_cache
export BIXBENCH3_BROKER_HOST_PORT=18080
export BIXBENCH3_BROKER_ENDPOINT=http://127.0.0.1:18080
export BIXBENCH3_BROKER_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
cd /opt/bixbench3
python3 - {remote_root} <<'PY'
from __future__ import annotations
import json
from pathlib import Path
import sys

remote = Path(sys.argv[1])
broker = remote / "broker"
broker.mkdir(parents=True, exist_ok=True)
case = json.loads((remote / "dataset" / "case.json").read_text(encoding="utf-8"))
if not isinstance(case, dict):
    raise RuntimeError("staged one-case dataset must contain a JSON object")
network_policy = case.get("network_policy")
if not isinstance(network_policy, dict):
    raise RuntimeError("staged one-case dataset has no network policy")
network_policy.setdefault(
    "protected_gcs_buckets",
    ["bixbench3-ground-truth"],
)
(broker / "policy.json").write_text(
    json.dumps(network_policy, indent=2, sort_keys=True) + "\\n",
    encoding="utf-8",
)
PY
(
  set -a
  . {_SECRET_ENV_PATH} || exit 97
  set +a
  {inspect_command}
) || run_status=$?
cp -a {remote_root}/outputs/. {remote_root}/agent_artifacts/ 2>/dev/null || true
write_status grading grading "$run_status"
mkdir -m 0700 -p {remote_root}/host_ground_truth
uv run python - {shlex.quote(plan.task.task_id)} {shlex.quote(plan.task.paper_id)} {shlex.quote(plan.task.grader_manifest_url)} {shlex.quote(plan.task.grader_manifest_sha256)} {remote_root} <<'PY' || run_status=$?
from __future__ import annotations
from pathlib import Path
import sys

from bixbench3.gcp_runner import TaskSpec, grade_existing_artifacts

task_id, paper_id, grader_url, grader_sha256, remote_text = sys.argv[1:]
remote = Path(remote_text)
task = TaskSpec(
    task_id=task_id,
    paper_id=paper_id,
    input_manifest_uri="host-grading-only",
    input_manifest_sha256="0" * 64,
    input_size_bytes=0,
    grader_manifest_url=grader_url,
    grader_manifest_sha256=grader_sha256,
)
grade_existing_artifacts(
    task=task,
    artifacts_dir=remote / "agent_artifacts",
    grades_dir=remote / "artifact_grades",
    ground_truth_parent=remote / "host_ground_truth",
)
PY
if [ -f {remote_root}/artifact_grades/paper_grade.json ]; then
  mv {remote_root}/artifact_grades/paper_grade.json {remote_root}/grade.json
fi
write_status judging judging "$run_status"
judge_status=0
(
  set -a
  . {_SECRET_ENV_PATH} || exit 97
  set +a
  uv run python - {remote_root} {remote_root}/dataset/case.json <<'PY'
from __future__ import annotations
from pathlib import Path
import sys

from bixbench3.grading.process_judge import run_process_judge

remote_text, dataset_text = sys.argv[1:]
run_process_judge(
    run_dir=Path(remote_text),
    dataset_path=Path(dataset_text),
)
PY
) || judge_status=$?
redaction_status=0
python3 - {remote_root} {_SECRET_ENV_PATH} <<'PY' || redaction_status=$?
from __future__ import annotations
from pathlib import Path
import re
import shlex
import sys

root = Path(sys.argv[1])
env_path = Path(sys.argv[2])
secret_keys = {required_environment_keys(plan.model)!r}
secret_values: dict[str, str] = {{}}
for line in env_path.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    parsed = shlex.split(line)
    if len(parsed) != 1 or "=" not in parsed[0]:
        raise RuntimeError("invalid streamed environment file")
    key, value = parsed[0].split("=", 1)
    secret_values[key] = value
replacements = [
    (value.encode(), f"<redacted:{{key}}>".encode())
    for key in secret_keys
    if (value := secret_values.get(key))
]
patterns = (
    (re.compile(rb"sk-proj-[A-Za-z0-9_-]+"), b"<redacted:OPENAI_API_KEY>"),
    (re.compile(rb"sk-ant-[A-Za-z0-9_-]+"), b"<redacted:ANTHROPIC_API_KEY>"),
    (re.compile(rb"sk-or-[A-Za-z0-9_-]+"), b"<redacted:OPENROUTER_API_KEY>"),
    (re.compile(rb"AIza[A-Za-z0-9_-]+"), b"<redacted:GOOGLE_API_KEY>"),
)
for relative in ("inspect_logs", "outputs", "agent_artifacts", "artifact_grades", "grade.json", "judge.json", "broker_logs", "attempts"):
    base_path = root / relative
    if not base_path.exists():
        continue
    for path in base_path.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        redacted = data
        for secret, marker in replacements:
            redacted = redacted.replace(secret, marker)
        for pattern, marker in patterns:
            redacted = pattern.sub(marker, redacted)
        if redacted != data:
            path.write_bytes(redacted)
PY
if [ "$redaction_status" -ne 0 ]; then
  run_status="$redaction_status"
fi
python3 - {remote_root}/run_manifest.json "$run_status" "$judge_status" <<'PY'
from __future__ import annotations
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
status = int(sys.argv[2])
judge_exit_status = int(sys.argv[3])
value = json.loads(path.read_text(encoding="utf-8"))
value["remote_exit_status"] = status
value["remote_completed_at"] = datetime.now(UTC).isoformat()
value["status"] = "succeeded" if status == 0 else "failed"
value["phase"] = "completed" if status == 0 else "failed"
process_judge = value.get("process_judge")
if not isinstance(process_judge, dict):
    process_judge = {{}}
process_judge["exit_status"] = judge_exit_status
judge_path = path.parent / "judge.json"
if judge_path.exists():
    judge = json.loads(judge_path.read_text(encoding="utf-8"))
    process_judge["status"] = judge.get("judge_status", "unknown")
    process_judge["score"] = judge.get("judge_score")
    process_judge["failure_modes"] = judge.get("failure_modes", [])
    process_judge["report_path"] = "judge.json"
else:
    process_judge["status"] = "judge_failed"
    process_judge["score"] = None
    process_judge["failure_modes"] = []
    process_judge["report_path"] = None
value["process_judge"] = process_judge
path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
PY
{upload_if_redacted}
reached_end=1
exit "$run_status"
"""


def results_write_preflight_script(results_uri: str) -> str:
    target = f"{results_uri.rstrip('/')}/.write-preflight-{os.getpid()}"
    return f"""set -euo pipefail
printf 'bixbench3 preflight\\n' | gcloud storage cp --no-clobber - {shlex.quote(target)}
"""


def remote_builder_prepare_script() -> str:
    return """set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl docker.io docker-compose-v2
systemctl enable --now docker
"""


def remote_runtime_build_script(
    *,
    source_archive_sha256: str,
    cellranger_sha256: str,
    cellranger_arc_sha256: str,
    image_tag: str,
) -> str:
    return f"""set -euo pipefail
builder_home=/home/"${{SUDO_USER:-$USER}}"
source_archive="$builder_home/bixbench3-source.tar.gz"
source_root="$builder_home/bixbench3-source"
printf '%s  %s\\n' {shlex.quote(source_archive_sha256)} "$source_archive" | sha256sum --check -
rm -rf "$source_root"
mkdir -p "$source_root"
tar -xzf "$source_archive" -C "$source_root"
mkdir -p "$source_root/operator-cache/10x"
install -m 0600 /home/"${{SUDO_USER:-$USER}}"/cellranger-6.0.2.tar.gz "$source_root/operator-cache/10x/cellranger-6.0.2.tar.gz"
install -m 0600 /home/"${{SUDO_USER:-$USER}}"/cellranger-arc-2.0.2.tar.gz "$source_root/operator-cache/10x/cellranger-arc-2.0.2.tar.gz"
printf '%s  %s\\n' {shlex.quote(cellranger_sha256)} "$source_root/operator-cache/10x/cellranger-6.0.2.tar.gz" | sha256sum --check -
printf '%s  %s\\n' {shlex.quote(cellranger_arc_sha256)} "$source_root/operator-cache/10x/cellranger-arc-2.0.2.tar.gz" | sha256sum --check -
docker build --pull --tag {shlex.quote(image_tag)} \
  --build-arg CELLRANGER_TARBALL=operator-cache/10x/cellranger-6.0.2.tar.gz \
  --build-arg CELLRANGER_ARC_TARBALL=operator-cache/10x/cellranger-arc-2.0.2.tar.gz \
  --build-arg CELLRANGER_SHA256={shlex.quote(cellranger_sha256)} \
  --build-arg CELLRANGER_ARC_SHA256={shlex.quote(cellranger_arc_sha256)} \
  --file "$source_root/sandbox/Dockerfile" "$source_root"
docker build --pull --tag bixbench3-network-gateway:{__version__} --file "$source_root/src/bixbench3/network_gateway/Dockerfile" "$source_root"
rm -f "$source_root/operator-cache/10x/cellranger-6.0.2.tar.gz"
rm -f "$source_root/operator-cache/10x/cellranger-arc-2.0.2.tar.gz"
rm -rf /opt/bixbench3
install -d -m 0755 /opt/bixbench3
cp -a "$source_root"/. /opt/bixbench3/
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh
cd /opt/bixbench3
uv sync --frozen
rm -f "$source_archive"
cat >/etc/bixbench3-runtime.json <<'EOF'
{json.dumps({"benchmark_version": BENCHMARK_VERSION, "package_version": __version__, "source_archive_sha256": source_archive_sha256, "sandbox_image": image_tag, "network_gateway_image": f"bixbench3-network-gateway:{__version__}", "cellranger_sha256": cellranger_sha256, "cellranger_arc_sha256": cellranger_arc_sha256}, indent=2)}
EOF
chmod 0444 /etc/bixbench3-runtime.json
"""


def remote_runtime_smoke_script(
    *,
    sandbox_image_tag: str,
    network_gateway_image_tag: str,
) -> str:
    return f"""set -euo pipefail
docker run --rm {shlex.quote(sandbox_image_tag)} cellranger --version
docker run --rm {shlex.quote(sandbox_image_tag)} cellranger-arc --version
docker image inspect {shlex.quote(network_gateway_image_tag)} >/dev/null
cd /opt/bixbench3
BIXBENCH3_NETWORK_GATEWAY_IMAGE={shlex.quote(network_gateway_image_tag)} \
BIXBENCH3_BROKER_POLICY=/tmp/policy.json \
BIXBENCH3_BROKER_LOG_DIR=/tmp/broker-logs \
BIXBENCH3_BROKER_CACHE_DIR=/tmp/broker-cache \
BIXBENCH3_WORKSPACE_DIR=/tmp/workspace \
BIXBENCH3_BROKER_TOKEN=smoke-token \
OPENAI_API_KEY=smoke-key \
BIXBENCH3_SANDBOX_IMAGE={shlex.quote(sandbox_image_tag)} \
BIXBENCH3_DATA_DIR=/tmp/data \
BIXBENCH3_OUTPUTS_DIR=/tmp/outputs \
docker compose -f sandbox/compose.yaml config >/dev/null
uv run bixbench3 --help >/dev/null
"""


def selected_environment(keys: Sequence[str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in keys:
        if _ENV_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid provider environment key {key!r}")
        value = os.environ.get(key)
        if not value:
            raise RuntimeError(
                f"missing required environment variable {key}; export it before running"
            )
        if "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"{key} contains unsupported control characters")
        values[key] = value
    return values


def required_environment_keys(model: ResolvedModel) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                *model.required_environment_keys,
                *_PROCESS_JUDGE_ENVIRONMENT_KEYS,
            )
        )
    )


def process_judge_config() -> dict[str, JsonValue]:
    return {
        "model": DEFAULT_JUDGE_MODEL,
        "reasoning_effort": {
            "requested": DEFAULT_JUDGE_REASONING_EFFORT,
            "resolved": resolve_reasoning_effort(
                DEFAULT_JUDGE_MODEL,
                DEFAULT_JUDGE_REASONING_EFFORT,
            ),
        },
        "temperature": 0,
        "max_tool_output_chars": DEFAULT_MAX_TOOL_OUTPUT_CHARS,
        "max_trace_chars": DEFAULT_MAX_TRACE_CHARS,
        "prompt_sha256": PROMPT_TEMPLATE_SHA256,
        "required_environment_keys": list(_PROCESS_JUDGE_ENVIRONMENT_KEYS),
    }


def environment_file(values: Mapping[str, str]) -> str:
    lines = ["# streamed over SSH; never persisted to disk"]
    for key, value in values.items():
        if _ENV_NAME.fullmatch(key) is None:
            raise ValueError(f"invalid environment key {key!r}")
        lines.append(f"{key}={shlex.quote(value)}")
    return "\n".join(lines) + "\n"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_runtime_source_archive(repo_root: Path, destination: Path) -> None:
    """Create a deterministic allowlisted runtime source archive."""
    if destination.exists():
        raise FileExistsError(f"runtime source archive already exists: {destination}")
    required = [
        *(repo_root / name for name in _RUNTIME_SOURCE_FILES),
        *(repo_root / name for name in _RUNTIME_SOURCE_DIRECTORIES),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("runtime source inputs are missing: " + ", ".join(missing))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw_output:
        with gzip.GzipFile(
            filename="",
            fileobj=raw_output,
            mode="wb",
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for source in required:
                    candidates = (
                        [source] if source.is_file() else [source, *sorted(source.rglob("*"))]
                    )
                    for candidate in candidates:
                        relative = candidate.relative_to(repo_root)
                        if runtime_archive_path_excluded(relative):
                            continue
                        archive.add(
                            candidate,
                            arcname=relative.as_posix(),
                            recursive=False,
                            filter=normalize_tar_info,
                        )


def runtime_archive_path_excluded(path: Path) -> bool:
    return any(part in _RUNTIME_ARCHIVE_EXCLUDED_PARTS for part in path.parts) or path.suffix in {
        ".pyc",
        ".pyo",
    }


def normalize_tar_info(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.pax_headers = {}
    if info.isdir():
        info.mode = 0o755
    elif info.isfile():
        info.mode = 0o644
    return info


def validate_vendor_tarball(path: Path, *, expected_name: str) -> str:
    if not path.is_file():
        raise ValueError(f"licensed tarball does not exist: {path}")
    digest = file_sha256(path)
    try:
        with tarfile.open(path, "r:*") as archive:
            roots = {
                PurePosixPath(member.name).parts[0]
                for member in archive
                if member.name and PurePosixPath(member.name).parts
            }
    except tarfile.TarError as error:
        raise ValueError(f"invalid licensed tarball {path}: {error}") from error
    if expected_name not in roots:
        raise ValueError(
            f"{path} does not contain the expected {expected_name}/ top-level directory"
        )
    return digest


def result_run_uri(results_bucket: str, layout: RunLayout) -> str:
    bucket, prefix = parse_results_bucket(results_bucket)
    parts = [
        prefix.strip("/"),
        layout.paper_id,
        layout.model_slug,
        layout.timestamp,
    ]
    suffix = "/".join(part for part in parts if part)
    return f"gs://{bucket}/{suffix}"


def lifecycle_command(action: str, run_id: str, project: str) -> str:
    return shlex.join(
        ("uv", "run", "bixbench3", action, "--run", run_id, "--project", project)
    )


def resolve_run_manifest(runs_root: Path, run_id: str) -> Path:
    normalized = run_id.strip()
    if not normalized or "/" in normalized or normalized in {".", ".."}:
        raise ValueError("run ID must be a non-empty filename-safe identifier")
    matches: list[Path] = []
    if runs_root.exists():
        for candidate in runs_root.rglob("run_manifest.json"):
            try:
                value = parse_json_object(
                    candidate.read_text(encoding="utf-8"),
                    str(candidate),
                )
            except (OSError, RuntimeError, ValueError):
                continue
            if value.get("run_id") == normalized:
                matches.append(candidate)
    if not matches:
        raise RuntimeError(f"unknown local run ID {normalized!r} under {runs_root}")
    if len(matches) > 1:
        rendered = ", ".join(str(path) for path in matches)
        raise RuntimeError(f"run ID {normalized!r} is ambiguous: {rendered}")
    return matches[0]


def lifecycle_vm_identity(manifest: Mapping[str, Any]) -> tuple[str, str]:
    gcp = manifest.get("gcp")
    if not isinstance(gcp, dict):
        raise ValueError("run manifest has no GCP identity")
    vm_name = gcp.get("vm_name")
    zone = gcp.get("zone", DEFAULT_ZONE)
    if not isinstance(vm_name, str) or not vm_name.startswith("arb-run-"):
        raise ValueError("run manifest has an invalid run VM name")
    if not isinstance(zone, str) or not zone:
        raise ValueError("run manifest has an invalid GCP zone")
    return vm_name, zone


def lifecycle_ssh_argv(
    project: str,
    vm_name: str,
    zone: str,
    remote_argv: Sequence[str],
) -> Argv:
    return ssh_argv(("gcloud", "--project", project), vm_name, zone, remote_argv)


def parse_results_bucket(value: str) -> tuple[str, str]:
    match = _GCS_URI.fullmatch(value.strip().rstrip("/"))
    if match is None:
        raise ValueError("results bucket must be an existing gs://BUCKET[/PREFIX] URI")
    bucket = match.group(1)
    prefix = match.group(2) or ""
    if bucket in {INPUT_BUCKET, GROUND_TRUTH_BUCKET}:
        raise ValueError("benchmark input and ground-truth buckets cannot store run results")
    path = PurePosixPath(prefix)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("results bucket prefix must be a safe relative path")
    return bucket, prefix


def gs_to_https(value: str) -> str:
    if not value.startswith("gs://"):
        return value
    without_scheme = value.removeprefix("gs://")
    return f"https://storage.googleapis.com/{without_scheme}"


def fetch_json(url: str) -> dict[str, Any]:
    return parse_json_bytes(fetch_bytes(url), url)


def fetch_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": f"bixbench3/{__version__}"})
    try:
        with urlopen(request, timeout=120) as response:
            return cast(bytes, response.read())
    except HTTPError as error:
        gcs_uri = private_gcs_uri(url)
        if error.code in {401, 403} and gcs_uri is not None:
            result = subprocess.run(
                ["gcloud", "storage", "cat", gcs_uri],
                check=True,
                capture_output=True,
            )
            return result.stdout
        raise RuntimeError(f"failed to fetch public release asset {url}: {error}") from error
    except URLError as error:
        raise RuntimeError(f"failed to fetch public release asset {url}: {error}") from error


def private_gcs_uri(url: str) -> str | None:
    """Translate standard GCS HTTPS URLs for authenticated prerelease reads."""

    parsed = urlsplit(url)
    if parsed.scheme != "https" or parsed.netloc != "storage.googleapis.com":
        return None
    path = unquote(parsed.path).lstrip("/")
    bucket, separator, object_path = path.partition("/")
    if not separator or not bucket or not object_path:
        return None
    return f"gs://{bucket}/{object_path}"


def parse_json_bytes(value: bytes, source: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{source} contains invalid JSON: {error}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return cast(dict[str, Any], decoded)


def parse_json_object(value: str | None, source: str) -> dict[str, Any]:
    if value is None:
        raise ValueError(f"{source} was empty")
    return parse_json_bytes(value.encode(), source)


def manifest_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    value = manifest.get("objects") or manifest.get("files") or manifest.get("artifacts")
    if not isinstance(value, list):
        raise ValueError("release manifest must contain an objects list")
    entries: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("release manifest entries must be objects")
        entries.append(cast(dict[str, Any], item))
    return entries


def manifest_total_bytes(manifest: dict[str, Any]) -> int:
    total = 0
    for entry in manifest_entries(manifest):
        size = entry.get("size_bytes", entry.get("byte_size"))
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("manifest entries require non-negative size_bytes")
        total += size
    return total


def package_relative_path(value: str) -> PurePosixPath:
    marker = "/tasks/"
    relative = value
    if marker in value:
        after_task = value.split(marker, 1)[1]
        parts = PurePosixPath(after_task).parts
        relative = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
    path = PurePosixPath(relative.lstrip("/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe package path {value!r}")
    return path


def string_choice(
    value: Mapping[str, Any],
    *keys: str,
    default: str,
) -> str:
    selected = optional_string(value, *keys)
    return selected if selected is not None else default


def optional_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        selected = value.get(key)
        if isinstance(selected, str) and selected.strip():
            return selected.strip()
    return None


def integer_choice(value: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        selected = value.get(key)
        if isinstance(selected, int) and not isinstance(selected, bool) and selected >= 0:
            return selected
    return None


def command_error(error: subprocess.CalledProcessError) -> str:
    stderr = error.stderr if isinstance(error.stderr, str) else ""
    message = stderr.strip().splitlines()
    detail = message[-1] if message else f"exit status {error.returncode}"
    return f"{Path(str(error.cmd[0])).name} command failed: {detail}"


def git_commit(repo_root: Path) -> str | None:
    result = subprocess.run(
        ("git", "-C", str(repo_root), "rev-parse", "HEAD"),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def main_remote() -> int:
    raise RuntimeError("remote execution is performed through the streamed run script")


if __name__ == "__main__":
    raise SystemExit(main_remote())
