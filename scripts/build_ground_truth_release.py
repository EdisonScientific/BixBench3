"""Build a selective ground-truth release from repaired Stage 7 cases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from bixbench3.manifests import (
    GROUND_TRUTH_BUCKET,
    RELEASE_PAPER_IDS,
    build_gcs_release_manifest,
    build_gcs_task_manifest,
    sha256_file,
    utc_now,
)
from bixbench3.schemas import (
    GCSObjectManifest,
    GCSTaskManifest,
    LatestReleasePointer,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the ground-truth release builder parser."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", type=Path, required=True)
    parser.add_argument("--reuse-manifests-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0.0")
    parser.add_argument("--replace-paper-id", action="append", required=True)
    return parser


def main() -> int:
    """Build replacement packages and a complete release manifest."""

    args = build_parser().parse_args()
    replacements = set(args.replace_paper_id)
    unknown = replacements - set(RELEASE_PAPER_IDS)
    if unknown:
        raise ValueError(f"replacement papers are outside the release: {sorted(unknown)}")
    if args.output_dir.exists():
        raise FileExistsError(f"destination already exists: {args.output_dir}")

    created_at = utc_now()
    release_root = args.output_dir / "releases" / args.version
    manifests_dir = release_root / "manifests"
    manifests_dir.mkdir(parents=True)
    manifest_triples: list[tuple[GCSTaskManifest, str, str]] = []

    for paper_id in RELEASE_PAPER_IDS:
        destination_manifest = manifests_dir / f"{paper_id}.json"
        if paper_id in replacements:
            manifest = _build_replacement_task(
                source_dataset_dir=args.source_dataset_dir,
                release_root=release_root,
                version=args.version,
                paper_id=paper_id,
                created_at=created_at,
            )
            _write_json(destination_manifest, manifest.model_dump(mode="json"))
        else:
            source_manifest = args.reuse_manifests_dir / f"{paper_id}.json"
            manifest = GCSTaskManifest.model_validate_json(
                source_manifest.read_text(encoding="utf-8"),
                strict=True,
            )
            shutil.copy2(source_manifest, destination_manifest)

        manifest_url = (
            f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}/releases/"
            f"{args.version}/manifests/{paper_id}.json"
        )
        manifest_triples.append(
            (manifest, manifest_url, sha256_file(destination_manifest))
        )

    release = build_gcs_release_manifest(
        version=args.version,
        kind="ground_truth",
        bucket=GROUND_TRUTH_BUCKET,
        task_manifests=manifest_triples,
        created_at=created_at,
    )
    release_path = release_root / "release.json"
    _write_json(release_path, release.model_dump(mode="json"))
    pointer = LatestReleasePointer(
        benchmark_version=args.version,
        release_manifest_url=(
            f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}/releases/"
            f"{args.version}/release.json"
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


def _build_replacement_task(
    *,
    source_dataset_dir: Path,
    release_root: Path,
    version: str,
    paper_id: str,
    created_at: str,
) -> GCSTaskManifest:
    source = source_dataset_dir / "cases" / paper_id / "grader_only"
    specification = source / "artifacts_grading.json"
    artifacts_dir = source / "artifacts"
    if not specification.is_file() or not artifacts_dir.is_dir():
        raise FileNotFoundError(f"incomplete grader package: {source}")

    task_root = release_root / "tasks" / paper_id
    task_artifacts = task_root / "artifacts"
    task_artifacts.mkdir(parents=True)
    shutil.copy2(specification, task_root / specification.name)
    for artifact in sorted(path for path in artifacts_dir.rglob("*") if path.is_file()):
        relative = artifact.relative_to(artifacts_dir)
        destination = task_artifacts / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(artifact, destination)

    specification_payload = json.loads(specification.read_text(encoding="utf-8"))
    doi = str(specification_payload["doi"])
    objects = [
        _object_manifest(
            path=path,
            output_root=release_root.parent.parent,
            version=version,
            paper_id=paper_id,
            doi=doi,
        )
        for path in sorted(task_root.rglob("*"))
        if path.is_file()
    ]
    return build_gcs_task_manifest(
        version=version,
        paper_id=paper_id,
        kind="ground_truth",
        bucket=GROUND_TRUTH_BUCKET,
        release_prefix=f"releases/{version}",
        objects=objects,
        created_at=created_at,
    )


def _object_manifest(
    *,
    path: Path,
    output_root: Path,
    version: str,
    paper_id: str,
    doi: str,
) -> GCSObjectManifest:
    object_path = path.relative_to(output_root).as_posix()
    suffix = path.suffix.lower()
    media_type = {
        ".csv": "text/csv",
        ".json": "application/json",
        ".tsv": "text/tab-separated-values",
    }.get(suffix, "application/octet-stream")
    return GCSObjectManifest(
        object_path=object_path,
        public_url=(
            f"https://storage.googleapis.com/{GROUND_TRUTH_BUCKET}/{object_path}"
        ),
        size_bytes=path.stat().st_size,
        sha256=_sha256(path),
        crc32c=_crc32c(path),
        source_accession_or_url=doi,
        upstream_repository="Published structured artifact",
        media_type=media_type,
        category=(
            "grading_specification"
            if path.name == "artifacts_grading.json"
            else "ground_truth"
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32c(path: Path) -> str:
    polynomial = 0x82F63B78
    table: list[int] = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ polynomial if value & 1 else value >> 1
        table.append(value)
    checksum = 0xFFFFFFFF
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            for byte in chunk:
                checksum = table[(checksum ^ byte) & 0xFF] ^ (checksum >> 8)
    checksum ^= 0xFFFFFFFF
    return base64.b64encode(checksum.to_bytes(4, "big")).decode("ascii")


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
