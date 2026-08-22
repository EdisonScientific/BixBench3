"""Build the private BixBench3 runtime machine image."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bixbench3.gcp_runner import GCPRunner, SubprocessExecutor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the private runtime after reviewing and accepting the "
            "10x Genomics software license."
        )
    )
    parser.add_argument("--project", required=True)
    parser.add_argument("--cellranger-tarball", required=True, type=Path)
    parser.add_argument("--cellranger-arc-tarball", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    runner = GCPRunner(
        project=arguments.project,
        executor=SubprocessExecutor(),
        runs_root=repo_root / "runs",
        repo_root=repo_root,
    )
    result = runner.setup_runtime(
        cellranger_tarball=arguments.cellranger_tarball,
        cellranger_arc_tarball=arguments.cellranger_arc_tarball,
        dry_run=arguments.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
