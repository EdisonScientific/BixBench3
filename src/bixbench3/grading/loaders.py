from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: Path, file_format: str | None = None) -> pd.DataFrame:
    file_format = (file_format or path.suffix.lstrip(".")).lower()
    if file_format in {"csv", "csv.gz"}:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if file_format in {"tsv", "tab", "txt", "tsv.gz"}:
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if file_format in {"bed", "bed.gz"}:
        return _load_bed(path)
    raise ValueError(f"unsupported artifact format for grading: {file_format}")


def _load_bed(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment="#",
    )
    if {"chrom", "start", "end"}.issubset(frame.columns):
        return frame

    headerless = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        comment="#",
        header=None,
    )
    if headerless.shape[1] < 3:
        raise ValueError(f"BED file must have at least three columns: {path}")

    columns = ["chrom", "start", "end"]
    columns.extend(f"field_{index}" for index in range(4, headerless.shape[1] + 1))
    headerless.columns = columns
    return headerless
