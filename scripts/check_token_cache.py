#!/usr/bin/env python3
"""Audit cached MRI token files against an expected-scan manifest.

The checker never runs an encoder. It validates existing ``.pt`` files and
writes a machine-readable JSON summary plus a per-file CSV report.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


CACHE_COLUMN_CANDIDATES = (
    "cache_file",
    "token_file",
    "token_path",
    "cache_path",
    "expected_cache",
)
SUBJECT_COLUMN_CANDIDATES = ("PTID", "subject_id", "patient_id")
VISIT_COLUMN_CANDIDATES = ("VISCODE", "visit_id", "visit", "session_id")
IMAGE_COLUMN_CANDIDATES = ("IMAGEUID", "image_id", "scan_id", "ID")
SPLIT_COLUMN_CANDIDATES = ("split", "SPLIT", "partition")
REQUIRED_CACHE_KEYS = ("embeddings", "coords", "token_count", "embedding_dim", "grid_shape")


def _first_column(fieldnames: Sequence[str], candidates: Sequence[str]) -> str | None:
    return next((name for name in candidates if name in fieldnames), None)


def _require_column(fieldnames: Sequence[str], requested: str | None, candidates: Sequence[str], label: str) -> str:
    column = requested or _first_column(fieldnames, candidates)
    if column is None:
        raise ValueError(
            f"Cannot infer {label} column. Available columns: {list(fieldnames)}. "
            f"Pass the corresponding CLI option explicitly."
        )
    if column not in fieldnames:
        raise ValueError(f"{label} column {column!r} is not in the manifest")
    return column


def _load_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {path}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
        return rows, list(reader.fieldnames)


def _issue(
    issues: list[dict[str, Any]],
    severity: str,
    code: str,
    message: str,
    *,
    cache_file: str | None = None,
    manifest_rows: Sequence[int] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {"severity": severity, "code": code, "message": message}
    if cache_file is not None:
        item["cache_file"] = cache_file
    if manifest_rows is not None:
        item["manifest_rows"] = list(manifest_rows)
    issues.append(item)
    return item


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.item()
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    try:
        if float(value) != float(number):
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _as_int_triplet(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    result = tuple(_as_int(item) for item in value)
    if any(item is None for item in result):
        return None
    return result  # type: ignore[return-value]


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _check_cache_file(
    path: Path,
    cache_dir: Path,
    issues: list[dict[str, Any]],
    *,
    check_coordinate_bounds: bool,
) -> dict[str, Any]:
    name = _display_path(path, cache_dir)
    record: dict[str, Any] = {
        "cache_file": name,
        "status": "PASS",
        "embedding_shape": None,
        "coords_shape": None,
        "token_count": None,
        "embedding_dim": None,
        "source_scan": None,
        "error_codes": [],
        "warning_codes": [],
    }

    def add(severity: str, code: str, message: str) -> None:
        _issue(issues, severity, code, message, cache_file=name)
        record["error_codes" if severity == "ERROR" else "warning_codes"].append(code)

    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # keep auditing other files
        add("ERROR", "LOAD_ERROR", f"torch.load failed: {type(exc).__name__}: {exc}")
        record["status"] = "FAIL"
        return record

    if not isinstance(obj, Mapping):
        add("ERROR", "INVALID_ROOT_TYPE", f"Expected a mapping, got {type(obj).__name__}")
        record["status"] = "FAIL"
        return record

    for key in REQUIRED_CACHE_KEYS:
        if key not in obj:
            add("ERROR", "MISSING_KEY", f"Required key is missing: {key}")

    embeddings = obj.get("embeddings")
    coords = obj.get("coords")
    n_tokens: int | None = None
    embedding_dim: int | None = None

    if embeddings is not None:
        if not isinstance(embeddings, torch.Tensor):
            add("ERROR", "EMBEDDINGS_WRONG_TYPE", f"embeddings must be a Tensor, got {type(embeddings).__name__}")
        else:
            record["embedding_shape"] = list(embeddings.shape)
            if not embeddings.dtype.is_floating_point:
                add("ERROR", "EMBEDDINGS_WRONG_DTYPE", f"embeddings must be floating point, got {embeddings.dtype}")
            if embeddings.ndim != 2:
                add("ERROR", "EMBEDDINGS_WRONG_NDIM", f"Expected embeddings [N,D], got {tuple(embeddings.shape)}")
            else:
                n_tokens, embedding_dim = map(int, embeddings.shape)
                record["token_count"] = n_tokens
                record["embedding_dim"] = embedding_dim
                if n_tokens <= 0 or embedding_dim <= 0:
                    add("ERROR", "EMPTY_EMBEDDINGS", f"Embedding dimensions must be positive, got {tuple(embeddings.shape)}")
            if embeddings.dtype.is_floating_point and embeddings.numel() and not torch.isfinite(embeddings).all().item():
                bad = int((~torch.isfinite(embeddings)).sum().item())
                add("ERROR", "NONFINITE_EMBEDDINGS", f"embeddings contains {bad} NaN/Inf value(s)")

    if coords is not None:
        if not isinstance(coords, torch.Tensor):
            add("ERROR", "COORDS_WRONG_TYPE", f"coords must be a Tensor, got {type(coords).__name__}")
        else:
            record["coords_shape"] = list(coords.shape)
            if coords.dtype == torch.bool or coords.dtype.is_complex:
                add("ERROR", "COORDS_WRONG_DTYPE", f"coords must be numeric, got {coords.dtype}")
            if coords.ndim != 2 or coords.shape[1] != 3:
                add("ERROR", "COORDS_WRONG_SHAPE", f"Expected coords [N,3], got {tuple(coords.shape)}")
            elif n_tokens is not None and int(coords.shape[0]) != n_tokens:
                add(
                    "ERROR",
                    "TOKEN_COORD_COUNT_MISMATCH",
                    f"embeddings has {n_tokens} rows but coords has {int(coords.shape[0])}",
                )
            if coords.numel() and (coords.dtype.is_floating_point or coords.dtype.is_complex):
                if not torch.isfinite(coords).all().item():
                    bad = int((~torch.isfinite(coords)).sum().item())
                    add("ERROR", "NONFINITE_COORDS", f"coords contains {bad} NaN/Inf value(s)")
            if coords.ndim == 2 and coords.shape[1] == 3 and coords.numel():
                if int(torch.unique(coords, dim=0).shape[0]) != int(coords.shape[0]):
                    add("ERROR", "DUPLICATE_COORDS", "Two or more tokens use the same coordinate")

    declared_n = _as_int(obj.get("token_count")) if "token_count" in obj else None
    declared_d = _as_int(obj.get("embedding_dim")) if "embedding_dim" in obj else None
    if "token_count" in obj and declared_n is None:
        add("ERROR", "TOKEN_COUNT_INVALID", "token_count must be one integer scalar")
    elif declared_n is not None and n_tokens is not None and declared_n != n_tokens:
        add("ERROR", "TOKEN_COUNT_MISMATCH", f"token_count={declared_n}, embeddings rows={n_tokens}")
    if "embedding_dim" in obj and declared_d is None:
        add("ERROR", "EMBEDDING_DIM_INVALID", "embedding_dim must be one integer scalar")
    elif declared_d is not None and embedding_dim is not None and declared_d != embedding_dim:
        add("ERROR", "EMBEDDING_DIM_MISMATCH", f"embedding_dim={declared_d}, embeddings columns={embedding_dim}")

    grid_shape = _as_int_triplet(obj.get("grid_shape")) if "grid_shape" in obj else None
    if "grid_shape" in obj and (grid_shape is None or any(size <= 0 for size in grid_shape)):
        add("ERROR", "GRID_SHAPE_INVALID", f"grid_shape must contain three positive integers, got {obj.get('grid_shape')!r}")
        grid_shape = None
    if grid_shape is not None and n_tokens is not None and n_tokens > math.prod(grid_shape):
        add("ERROR", "TOKEN_COUNT_EXCEEDS_GRID", f"{n_tokens} tokens cannot fit grid_shape={grid_shape}")
    if (
        check_coordinate_bounds
        and grid_shape is not None
        and isinstance(coords, torch.Tensor)
        and coords.ndim == 2
        and coords.shape[1] == 3
        and coords.numel()
        and coords.dtype != torch.bool
        and not coords.dtype.is_complex
        and (not coords.dtype.is_floating_point or torch.isfinite(coords).all().item())
    ):
        limits = torch.tensor(grid_shape, dtype=coords.dtype)
        if (coords < 0).any().item() or (coords >= limits).any().item():
            add("ERROR", "COORDS_OUT_OF_BOUNDS", f"coords must satisfy 0 <= coords < grid_shape={grid_shape}")

    brain_fraction = obj.get("brain_fraction")
    if brain_fraction is not None:
        if not isinstance(brain_fraction, torch.Tensor) or brain_fraction.ndim != 1:
            add("ERROR", "BRAIN_FRACTION_WRONG_SHAPE", "brain_fraction must be a 1D Tensor")
        else:
            if n_tokens is not None and len(brain_fraction) != n_tokens:
                add("ERROR", "BRAIN_FRACTION_COUNT_MISMATCH", "brain_fraction length must equal token_count")
            if not brain_fraction.dtype.is_floating_point:
                add("ERROR", "BRAIN_FRACTION_WRONG_DTYPE", "brain_fraction must be floating point")
            elif not torch.isfinite(brain_fraction).all().item():
                add("ERROR", "NONFINITE_BRAIN_FRACTION", "brain_fraction contains NaN/Inf")
            elif ((brain_fraction < 0) | (brain_fraction > 1)).any().item():
                add("ERROR", "BRAIN_FRACTION_OUT_OF_RANGE", "brain_fraction must stay within [0,1]")

    scan_path = obj.get("scan_path") or obj.get("volume_path")
    if scan_path is not None:
        if not isinstance(scan_path, (str, Path)):
            add("WARNING", "SOURCE_SCAN_INVALID", "scan_path/volume_path is not a string path")
        elif str(scan_path).strip():
            record["source_scan"] = str(scan_path)

    record["status"] = "FAIL" if record["error_codes"] else "PASS"
    return record


def _write_details_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "cache_file",
        "status",
        "embedding_shape",
        "coords_shape",
        "token_count",
        "embedding_dim",
        "source_scan",
        "error_codes",
        "warning_codes",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            row = dict(record)
            for key in ("embedding_shape", "coords_shape", "error_codes", "warning_codes"):
                row[key] = json.dumps(row.get(key), ensure_ascii=False)
            writer.writerow({key: row.get(key) for key in columns})


def audit_cache(
    manifest: str | Path,
    cache_dir: str | Path,
    *,
    output: str | Path | None = None,
    details_csv: str | Path | None = None,
    cache_column: str | None = None,
    key_columns: Sequence[str] | None = None,
    subject_column: str | None = None,
    split_column: str | None = None,
    recursive: bool = False,
    allow_extra_cache: bool = False,
    check_coordinate_bounds: bool = True,
) -> dict[str, Any]:
    """Validate cache files, manifest coverage, scan identity and split leakage."""
    manifest_path = Path(manifest).expanduser().resolve()
    cache_root = Path(cache_dir).expanduser().resolve()
    if not cache_root.is_dir():
        raise NotADirectoryError(f"Cache directory not found: {cache_root}")

    rows, fieldnames = _load_manifest(manifest_path)
    cache_column = _require_column(fieldnames, cache_column, CACHE_COLUMN_CANDIDATES, "cache")
    subject_column = _require_column(fieldnames, subject_column, SUBJECT_COLUMN_CANDIDATES, "subject")
    if key_columns:
        identity_columns = list(key_columns)
        missing_columns = [name for name in identity_columns if name not in fieldnames]
        if missing_columns:
            raise ValueError(f"Identity columns missing from manifest: {missing_columns}")
    else:
        identity_columns = [subject_column]
        visit_column = _first_column(fieldnames, VISIT_COLUMN_CANDIDATES)
        image_column = _first_column(fieldnames, IMAGE_COLUMN_CANDIDATES)
        if visit_column and visit_column not in identity_columns:
            identity_columns.append(visit_column)
        if image_column and image_column not in identity_columns:
            identity_columns.append(image_column)

    if split_column is None:
        split_column = _first_column(fieldnames, SPLIT_COLUMN_CANDIDATES)
    elif split_column not in fieldnames:
        raise ValueError(f"Split column {split_column!r} is not in the manifest")

    issues: list[dict[str, Any]] = []
    expected_paths: dict[Path, list[int]] = defaultdict(list)
    identity_rows: dict[tuple[str, ...], list[int]] = defaultdict(list)
    subject_expected: dict[str, set[Path]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=2):
        identity = tuple(row.get(column, "") for column in identity_columns)
        if any(not value for value in identity):
            _issue(
                issues,
                "ERROR",
                "MISSING_IDENTITY_VALUE",
                f"Identity {identity_columns} contains a blank value",
                manifest_rows=[index],
            )
        else:
            identity_rows[identity].append(index)

        subject = row.get(subject_column, "")
        split = row.get(split_column, "") if split_column else ""
        if subject and split:
            subject_splits[subject].add(split)

        cache_value = row.get(cache_column, "")
        if not cache_value:
            _issue(
                issues,
                "ERROR",
                "MISSING_CACHE_REFERENCE",
                f"Manifest column {cache_column!r} is blank",
                manifest_rows=[index],
            )
            continue
        expected = Path(cache_value).expanduser()
        expected = (expected if expected.is_absolute() else cache_root / expected).resolve()
        expected_paths[expected].append(index)
        if subject:
            subject_expected[subject].add(expected)

    duplicate_identities = []
    for identity, manifest_rows in sorted(identity_rows.items()):
        if len(manifest_rows) > 1:
            duplicate_identities.append({"identity": dict(zip(identity_columns, identity)), "manifest_rows": manifest_rows})
            _issue(
                issues,
                "ERROR",
                "DUPLICATE_MANIFEST_IDENTITY",
                f"Identity {dict(zip(identity_columns, identity))} appears {len(manifest_rows)} times",
                manifest_rows=manifest_rows,
            )

    duplicate_expected_paths = []
    for path, manifest_rows in sorted(expected_paths.items(), key=lambda item: str(item[0])):
        if len(manifest_rows) > 1:
            display = _display_path(path, cache_root)
            duplicate_expected_paths.append({"cache_file": display, "manifest_rows": manifest_rows})
            _issue(
                issues,
                "ERROR",
                "DUPLICATE_CACHE_REFERENCE",
                f"The same cache file is expected by {len(manifest_rows)} manifest rows",
                cache_file=display,
                manifest_rows=manifest_rows,
            )

    leakage = []
    for subject, splits in sorted(subject_splits.items()):
        if len(splits) > 1:
            leakage.append({"subject": subject, "splits": sorted(splits)})
            _issue(
                issues,
                "ERROR",
                "SUBJECT_SPLIT_LEAKAGE",
                f"Subject {subject!r} appears in multiple splits: {sorted(splits)}",
            )

    pattern = "**/*.pt" if recursive else "*.pt"
    actual_paths = {path.resolve() for path in cache_root.glob(pattern) if path.is_file()}
    expected_set = set(expected_paths)
    missing_paths = sorted((path for path in expected_set if not path.is_file()), key=str)
    unexpected_paths = sorted(actual_paths - expected_set, key=str)

    for path in missing_paths:
        _issue(
            issues,
            "ERROR",
            "MISSING_CACHE",
            "Expected cache file does not exist",
            cache_file=_display_path(path, cache_root),
            manifest_rows=expected_paths[path],
        )
    for path in unexpected_paths:
        _issue(
            issues,
            "WARNING" if allow_extra_cache else "ERROR",
            "UNEXPECTED_CACHE",
            "Cache file is not referenced by the manifest",
            cache_file=_display_path(path, cache_root),
        )

    file_records = [
        _check_cache_file(path, cache_root, issues, check_coordinate_bounds=check_coordinate_bounds)
        for path in sorted(actual_paths, key=str)
    ]

    source_files: dict[str, list[str]] = defaultdict(list)
    for record in file_records:
        if record["source_scan"]:
            source_files[record["source_scan"]].append(record["cache_file"])
    duplicate_sources = []
    for source, files in sorted(source_files.items()):
        if len(files) > 1:
            duplicate_sources.append({"source_scan": source, "cache_files": files})
            _issue(
                issues,
                "ERROR",
                "DUPLICATE_SOURCE_SCAN",
                f"One source scan is cached by multiple files: {files}",
            )

    missing_subjects = sorted(
        subject for subject, paths in subject_expected.items() if paths and all(not path.is_file() for path in paths)
    )
    error_count = sum(item["severity"] == "ERROR" for item in issues)
    warning_count = sum(item["severity"] == "WARNING" for item in issues)
    matched_paths = actual_paths & expected_set
    invalid_files = {record["cache_file"] for record in file_records if record["status"] == "FAIL"}
    valid_expected = sum(
        1
        for path in matched_paths
        if _display_path(path, cache_root) not in invalid_files
    )
    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL" if error_count else "PASS",
        "inputs": {
            "manifest": str(manifest_path),
            "cache_dir": str(cache_root),
            "cache_column": cache_column,
            "identity_columns": identity_columns,
            "subject_column": subject_column,
            "split_column": split_column,
            "recursive": recursive,
            "coordinate_bounds_checked": check_coordinate_bounds,
        },
        "summary": {
            "manifest_rows": len(rows),
            "expected_unique_cache_files": len(expected_set),
            "cache_files_found": len(actual_paths),
            "matched_cache_files": len(matched_paths),
            "valid_expected_cache_files": valid_expected,
            "invalid_cache_files": len(invalid_files),
            "missing_cache_files": len(missing_paths),
            "unexpected_cache_files": len(unexpected_paths),
            "missing_subjects": len(missing_subjects),
            "duplicate_manifest_identities": len(duplicate_identities),
            "duplicate_cache_references": len(duplicate_expected_paths),
            "duplicate_source_scans": len(duplicate_sources),
            "subject_split_leakage": len(leakage),
            "errors": error_count,
            "warnings": warning_count,
        },
        "missing_cache_files": [_display_path(path, cache_root) for path in missing_paths],
        "unexpected_cache_files": [_display_path(path, cache_root) for path in unexpected_paths],
        "missing_subjects": missing_subjects,
        "duplicate_manifest_identities": duplicate_identities,
        "duplicate_cache_references": duplicate_expected_paths,
        "duplicate_source_scans": duplicate_sources,
        "subject_split_leakage": leakage,
        "files": file_records,
        "issues": issues,
    }

    if output is not None:
        output_path = Path(output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if details_csv is None:
            details_csv = output_path.with_name(f"{output_path.stem}_details.csv")
    if details_csv is not None:
        _write_details_csv(Path(details_csv).expanduser(), file_records)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True, help="Directory containing cached .pt files")
    parser.add_argument("--manifest", required=True, help="CSV describing every expected scan/cache")
    parser.add_argument("--output", default="reports/cache_check.json", help="JSON report path")
    parser.add_argument("--details-csv", help="Per-file CSV path (default: next to JSON report)")
    parser.add_argument("--cache-column", help="Manifest column containing the expected cache path")
    parser.add_argument("--subject-column", help="Stable subject identifier column")
    parser.add_argument("--split-column", help="Train/validation/test column; auto-detected when omitted")
    parser.add_argument(
        "--key-columns",
        nargs="+",
        help="Columns defining one scan identity, e.g. PTID VISCODE IMAGEUID",
    )
    parser.add_argument("--recursive", action="store_true", help="Search cache subdirectories recursively")
    parser.add_argument(
        "--allow-extra-cache",
        action="store_true",
        help="Treat unreferenced .pt files as warnings instead of errors",
    )
    parser.add_argument(
        "--skip-coordinate-bounds",
        action="store_true",
        help="Do not require grid-index coordinates to fall inside grid_shape",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = audit_cache(
            manifest=args.manifest,
            cache_dir=args.cache_dir,
            output=args.output,
            details_csv=args.details_csv,
            cache_column=args.cache_column,
            key_columns=args.key_columns,
            subject_column=args.subject_column,
            split_column=args.split_column,
            recursive=args.recursive,
            allow_extra_cache=args.allow_extra_cache,
            check_coordinate_bounds=not args.skip_coordinate_bounds,
        )
    except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"CONFIG ERROR: {exc}")
        return 2

    summary = report["summary"]
    print(
        f"{report['status']}: matched {summary['matched_cache_files']}/"
        f"{summary['expected_unique_cache_files']} expected cache files; "
        f"valid={summary['valid_expected_cache_files']}, "
        f"errors={summary['errors']}, warnings={summary['warnings']}"
    )
    print(f"JSON report: {Path(args.output).expanduser().resolve()}")
    details = args.details_csv or str(Path(args.output).with_name(f"{Path(args.output).stem}_details.csv"))
    print(f"CSV details: {Path(details).expanduser().resolve()}")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
