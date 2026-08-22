from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bixbench3.config import resolve_model
from bixbench3.gcp_runner import (
    GCPRunner,
    TaskSpec,
    build_run_commands,
    lifecycle_command,
    resolve_run_manifest,
)
from bixbench3.run_layout import build_run_layout


class LifecycleExecutor:
    def __init__(
        self,
        *,
        vm_status: str = "RUNNING",
        remote_status: dict[str, object] | None = None,
    ) -> None:
        self.vm_status = vm_status
        self.remote_status = remote_status
        self.calls: list[tuple[str, ...]] = []

    def run(
        self,
        argv: list[str] | tuple[str, ...],
        *,
        input_text: str | None = None,
        check: bool = True,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del input_text, check, capture_output
        command = tuple(argv)
        self.calls.append(command)
        if "describe" in command and "instances" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps({"status": self.vm_status}),
                stderr="",
            )
        joined = " ".join(command)
        if "cat" in joined and "/var/lib/bixbench3/status.json" in joined:
            if self.remote_status is None:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="missing")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(self.remote_status),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")


@pytest.fixture
def lifecycle_run(tmp_path: Path) -> tuple[Path, str]:
    run_id = "paper-openai-gpt-5-6-luna-20260806t120000000000z-0123456789"
    local_dir = tmp_path / "paper" / "openai-gpt-5-6-luna" / "timestamp"
    local_dir.mkdir(parents=True)
    manifest_path = local_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": "running",
                "phase": "agent_running",
                "gcp": {
                    "project": "example-project",
                    "zone": "us-central1-a",
                    "vm_name": "arb-run-paper-0123456789",
                },
            }
        ),
        encoding="utf-8",
    )
    return tmp_path, run_id


def test_run_plan_starts_detached_systemd_service(tmp_path: Path) -> None:
    task = TaskSpec(
        task_id="1",
        paper_id="paper",
        input_manifest_uri="gs://bixbench3-inputs/manifest.json",
        input_manifest_sha256="a" * 64,
        input_size_bytes=10,
        grader_manifest_url="https://storage.googleapis.com/grader/manifest.json",
        grader_manifest_sha256="b" * 64,
    )
    layout = build_run_layout(
        task_id=task.task_id,
        paper_id=task.paper_id,
        model_id="openai/gpt-5.6-luna",
        manifest_sha256=task.input_manifest_sha256,
        runs_root=tmp_path,
    )
    commands = build_run_commands(
        project="example-project",
        settings=GCPRunner(
            project="example-project",
            executor=LifecycleExecutor(),
        ).settings,
        task=task,
        layout=layout,
        service_account=None,
        results_uri=None,
        case_dataset_dir=tmp_path,
    )

    assert "run_task" not in commands
    assert "install_run_script" in commands
    start = commands["start_run_service"]
    joined = " ".join(start)
    assert "systemd-run" in joined
    assert "--property=Restart=no" in joined
    assert not any("API_KEY" in value for value in start)


def test_status_reports_remote_terminal_state(
    lifecycle_run: tuple[Path, str],
) -> None:
    runs_root, run_id = lifecycle_run
    executor = LifecycleExecutor(
        remote_status={
            "status": "succeeded",
            "phase": "completed",
            "remote_exit_status": 0,
        }
    )
    runner = GCPRunner(
        project="example-project",
        executor=executor,
        runs_root=runs_root,
    )

    result = runner.status_run(run_id)

    assert result["terminal"] is True
    assert result["status"] == "succeeded"
    assert result["next_command"] == lifecycle_command(
        "complete",
        run_id,
        "example-project",
    )


def test_complete_refuses_active_run(lifecycle_run: tuple[Path, str]) -> None:
    runs_root, run_id = lifecycle_run
    runner = GCPRunner(
        project="example-project",
        executor=LifecycleExecutor(remote_status={"status": "running", "phase": "agent_running"}),
        runs_root=runs_root,
    )

    with pytest.raises(RuntimeError, match="still running"):
        runner.complete_run(run_id)


def test_complete_collects_and_deletes_terminal_run(
    lifecycle_run: tuple[Path, str],
) -> None:
    runs_root, run_id = lifecycle_run
    executor = LifecycleExecutor(
        remote_status={
            "status": "succeeded",
            "phase": "completed",
            "remote_exit_status": 0,
        }
    )
    runner = GCPRunner(
        project="example-project",
        executor=executor,
        runs_root=runs_root,
    )

    result = runner.complete_run(run_id)

    assert result["status"] == "succeeded"
    assert result["cleanup"]["action"] == "deleted"
    assert any(
        "scp" in command and "inspect_logs" in " ".join(command)
        for command in executor.calls
    )
    assert any(
        "scp" in command and "judge.json" in " ".join(command)
        for command in executor.calls
    )
    assert any("delete" in command and "instances" in command for command in executor.calls)
    manifest = json.loads(resolve_run_manifest(runs_root, run_id).read_text(encoding="utf-8"))
    assert manifest["phase"] == "finalized"

    repeated = runner.complete_run(run_id)
    assert repeated["already_complete"] is True


def test_complete_merges_process_judge_summary_into_local_manifest(
    lifecycle_run: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root, run_id = lifecycle_run
    executor = LifecycleExecutor(
        remote_status={
            "status": "succeeded",
            "phase": "completed",
            "remote_exit_status": 0,
        }
    )
    runner = GCPRunner(
        project="example-project",
        executor=executor,
        runs_root=runs_root,
    )

    def fake_collect(
        self: GCPRunner,
        vm_name: str,
        zone: str,
        local_dir: Path,
    ) -> None:
        del self, vm_name, zone
        (local_dir / "remote_run_manifest.json").write_text(
            json.dumps(
                {
                    "remote_exit_status": 0,
                    "status": "succeeded",
                    "phase": "completed",
                    "process_judge": {
                        "model": "openai/gpt-5.5",
                        "status": "ok",
                        "score": 0.75,
                    },
                }
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(GCPRunner, "_collect_lifecycle_files", fake_collect)

    runner.complete_run(run_id)

    manifest = json.loads(resolve_run_manifest(runs_root, run_id).read_text(encoding="utf-8"))
    assert manifest["process_judge"]["status"] == "ok"
    assert manifest["process_judge"]["score"] == 0.75


def test_cancel_stops_service_and_deletes_vm(
    lifecycle_run: tuple[Path, str],
) -> None:
    runs_root, run_id = lifecycle_run
    executor = LifecycleExecutor(remote_status={"status": "running", "phase": "agent_running"})
    runner = GCPRunner(
        project="example-project",
        executor=executor,
        runs_root=runs_root,
    )

    result = runner.cancel_run(run_id)

    assert result["status"] == "cancelled"
    assert any(
        "systemctl" in " ".join(command) and " stop " in " ".join(command)
        for command in executor.calls
    )
    assert any("delete" in command and "instances" in command for command in executor.calls)


def test_run_lookup_rejects_unknown_and_ambiguous_ids(
    lifecycle_run: tuple[Path, str],
) -> None:
    runs_root, run_id = lifecycle_run
    with pytest.raises(RuntimeError, match="unknown local run ID"):
        resolve_run_manifest(runs_root, "missing")

    duplicate = runs_root / "duplicate" / "run_manifest.json"
    duplicate.parent.mkdir()
    duplicate.write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_run_manifest(runs_root, run_id)


def test_lifecycle_rejects_project_mismatch(
    lifecycle_run: tuple[Path, str],
) -> None:
    runs_root, run_id = lifecycle_run
    runner = GCPRunner(
        project="wrong-project",
        executor=LifecycleExecutor(),
        runs_root=runs_root,
    )

    with pytest.raises(RuntimeError, match="belongs to project"):
        runner.status_run(run_id)


def test_luna_resolution_remains_native_max() -> None:
    model = resolve_model("openai/gpt-5.6-luna", "max")
    assert model.resolved_effort == "max"
