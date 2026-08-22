"""Build the public dataset package locally and optionally upload it privately."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bixbench3.hub import (
    HubSourceLayout,
    build_hf_release,
    prepare_private_upload,
    upload_private_release,
)
from bixbench3.manifests import HF_REPO_ID


def build_parser() -> argparse.ArgumentParser:
    """Create the release publisher argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable 20-task BixBench3 HF package from "
            "operator-provided source exports."
        )
    )
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--source-hf-jsonl", type=Path, required=True)
    parser.add_argument("--source-annotations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--status", choices=("draft", "published"), default="draft")
    parser.add_argument("--hf-repo-id", default=HF_REPO_ID)
    parser.add_argument("--paper-url")
    parser.add_argument("--git-commit")
    parser.add_argument("--hf-revision")
    parser.add_argument("--input-release-manifest-sha256")
    parser.add_argument("--ground-truth-release-manifest-sha256")
    parser.add_argument("--runtime-image-id")
    parser.add_argument("--sandbox-sha256")
    parser.add_argument(
        "--upload-private",
        action="store_true",
        help="opt in to creating/forcing a private HF dataset and uploading",
    )
    parser.add_argument(
        "--upload-revision",
        help="HF branch or revision for --upload-private",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the release, with network upload disabled unless explicitly selected."""

    args = build_parser().parse_args(argv)
    result = build_hf_release(
        sources=HubSourceLayout(
            dataset_dir=args.source_dataset_dir,
            hf_rows_jsonl=args.source_hf_jsonl,
            annotations_dir=args.source_annotations_dir,
        ),
        output_dir=args.output_dir,
        version=args.version,
        status=args.status,
        hf_repo_id=args.hf_repo_id,
        paper_url=args.paper_url,
        git_commit=args.git_commit,
        hf_revision=args.hf_revision,
        input_release_manifest_sha256=args.input_release_manifest_sha256,
        ground_truth_release_manifest_sha256=(args.ground_truth_release_manifest_sha256),
        runtime_image_id=args.runtime_image_id,
        sandbox_sha256=args.sandbox_sha256,
    )
    summary: dict[str, object] = {
        "status": "built",
        "release_json": str(result.release_json),
        "tasks_parquet": str(result.tasks_parquet),
        "task_count": result.release.task_count,
        "artifact_count": result.release.artifact_count,
    }

    if args.upload_private:
        if not args.upload_revision:
            raise SystemExit("--upload-private requires --upload-revision")
        plan = prepare_private_upload(
            result.output_dir,
            repo_id=args.hf_repo_id,
            revision=args.upload_revision,
        )
        upload_result = upload_private_release(
            plan,
            token=os.environ.get("HF_TOKEN"),
        )
        summary["status"] = "uploaded_private"
        summary["upload_result"] = str(upload_result)

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
