"""Build a complete input release from reused and replacement task manifests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from bixbench3.manifests import (
    INPUT_BUCKET,
    RELEASE_PAPER_IDS,
    build_gcs_release_manifest,
    sha256_file,
    utc_now,
)
from bixbench3.schemas import GCSTaskManifest, LatestReleasePointer


def build_parser() -> argparse.ArgumentParser:
    """Create the input release builder parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-manifests-dir", type=Path, required=True)
    parser.add_argument("--replacement-manifests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--replace-paper-id", action="append", required=True)
    return parser


def main() -> int:
    """Build a validated input release without copying the data objects."""

    args = build_parser().parse_args()
    replacements = set(args.replace_paper_id)
    unknown = replacements - set(RELEASE_PAPER_IDS)
    if unknown:
        raise ValueError(f"replacement papers are outside the release: {sorted(unknown)}")
    if args.output_dir.exists():
        raise FileExistsError(f"destination already exists: {args.output_dir}")

    replacement_manifests = _replacement_manifests(args.replacement_manifests_dir)
    missing = replacements - set(replacement_manifests)
    if missing:
        raise FileNotFoundError(f"replacement manifests are missing: {sorted(missing)}")

    created_at = utc_now()
    release_root = args.output_dir / "releases" / args.version
    manifests_dir = release_root / "manifests"
    manifests_dir.mkdir(parents=True)
    manifest_triples: list[tuple[GCSTaskManifest, str, str]] = []

    for paper_id in RELEASE_PAPER_IDS:
        source_manifest = (
            replacement_manifests[paper_id]
            if paper_id in replacements
            else args.reuse_manifests_dir / f"{paper_id}.json"
        )
        manifest = GCSTaskManifest.model_validate_json(
            source_manifest.read_text(encoding="utf-8"),
            strict=True,
        )
        if manifest.paper_id != paper_id or manifest.kind != "inputs":
            raise ValueError(f"wrong input manifest selected for {paper_id}")
        destination_manifest = manifests_dir / f"{paper_id}.json"
        shutil.copy2(source_manifest, destination_manifest)
        manifest_url = (
            f"https://storage.googleapis.com/{INPUT_BUCKET}/releases/"
            f"{args.version}/manifests/{paper_id}.json"
        )
        manifest_triples.append(
            (manifest, manifest_url, sha256_file(destination_manifest))
        )

    release = build_gcs_release_manifest(
        version=args.version,
        kind="inputs",
        bucket=INPUT_BUCKET,
        task_manifests=manifest_triples,
        created_at=created_at,
    )
    release_path = release_root / "release-public.json"
    _write_json(release_path, release.model_dump(mode="json"))
    pointer = LatestReleasePointer(
        benchmark_version=args.version,
        release_manifest_url=(
            f"https://storage.googleapis.com/{INPUT_BUCKET}/releases/"
            f"{args.version}/release-public.json"
        ),
        release_manifest_sha256=sha256_file(release_path),
        updated_at=created_at,
    )
    _write_json(args.output_dir / "latest.json", pointer.model_dump(mode="json"))
    print(
        json.dumps(
            {
                "artifact_count": release.artifact_count,
                "output_dir": str(args.output_dir),
                "replacement_papers": sorted(replacements),
                "task_count": release.task_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _replacement_manifests(root: Path) -> dict[str, Path]:
    manifests: dict[str, Path] = {}
    for path in sorted(root.glob("task_*/manifest.json")):
        manifest = GCSTaskManifest.model_validate_json(
            path.read_text(encoding="utf-8"),
            strict=True,
        )
        if manifest.paper_id in manifests:
            raise ValueError(f"duplicate replacement manifest: {manifest.paper_id}")
        manifests[manifest.paper_id] = path
    return manifests


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
