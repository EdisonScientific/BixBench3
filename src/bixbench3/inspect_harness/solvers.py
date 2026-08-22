from __future__ import annotations

from inspect_ai.agent import Agent, react
from inspect_ai.model import CompactionAuto
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import bash_session, python, text_editor
from inspect_ai.util import sandbox

from .model_info import register_model_info
from .tools import request_web_access

register_model_info()


@solver
def bixbench3_agent() -> Agent:
    return react(
        tools=[
            bash_session(timeout=600),
            python(timeout=600),
            text_editor(timeout=120),
            request_web_access(),
        ],
        compaction=CompactionAuto(),
    )


@solver
def prepare_case_workspace() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        del generate
        result = await sandbox().exec(
            [
                "sh",
                "-lc",
                "mkdir -p /outputs && "
                "rm -rf /workspace/work /workspace/approved_web && "
                "find /outputs -mindepth 1 -maxdepth 1 -exec rm -rf -- {} + && "
                "ln -sfn /data /workspace/data && "
                "ln -sfn /outputs /workspace/outputs && "
                "mkdir -p /workspace/work /workspace/approved_web",
            ]
        )
        if not result.success:
            raise RuntimeError(f"workspace preparation failed: {result.stderr}")
        return state

    return solve
