"""Tests for the H2 cache-quality audit using synthetic token caches."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.check_token_cache import audit_cache


def _cache(n: int = 8, d: int = 16, scan_path: str = "/fake/source.nii.gz") -> dict:
    coords = torch.tensor(
        [[z, y, x] for z in range(2) for y in range(2) for x in range(2)],
        dtype=torch.int16,
    )[:n]
    return {
        "embeddings": torch.randn(n, d, generator=torch.Generator().manual_seed(n + d)).half(),
        "coords": coords,
        "token_count": n,
        "embedding_dim": d,
        "grid_shape": torch.tensor([2, 2, 2], dtype=torch.int32),
        "scan_path": scan_path,
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PTID", "VISCODE", "IMAGEUID", "split", "cache_file"])
        writer.writeheader()
        writer.writerows(rows)


class TestTokenCacheAudit(unittest.TestCase):
    def test_valid_longitudinal_subject_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "tokens"
            cache_dir.mkdir()
            rows = [
                {"PTID": "A", "VISCODE": "bl", "IMAGEUID": "I1", "split": "train", "cache_file": "A_bl.pt"},
                {"PTID": "A", "VISCODE": "m12", "IMAGEUID": "I2", "split": "train", "cache_file": "A_m12.pt"},
                {"PTID": "B", "VISCODE": "bl", "IMAGEUID": "I3", "split": "test", "cache_file": "B_bl.pt"},
            ]
            manifest = root / "manifest.csv"
            _write_manifest(manifest, rows)
            for row in rows:
                torch.save(_cache(scan_path=f"/fake/{row['IMAGEUID']}.nii.gz"), cache_dir / row["cache_file"])

            report = audit_cache(manifest, cache_dir)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["summary"]["matched_cache_files"], 3)
            self.assertEqual(report["summary"]["subject_split_leakage"], 0)
            self.assertEqual(report["summary"]["duplicate_manifest_identities"], 0)

    def test_detects_corruption_coverage_duplicates_and_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_dir = root / "tokens"
            cache_dir.mkdir()
            rows = [
                {"PTID": "A", "VISCODE": "bl", "IMAGEUID": "I1", "split": "train", "cache_file": "A_bl.pt"},
                {"PTID": "A", "VISCODE": "m12", "IMAGEUID": "I2", "split": "test", "cache_file": "A_m12.pt"},
                {"PTID": "B", "VISCODE": "bl", "IMAGEUID": "I3", "split": "train", "cache_file": "B_bl.pt"},
                {"PTID": "C", "VISCODE": "bl", "IMAGEUID": "I4", "split": "train", "cache_file": "C_bl.pt"},
                {"PTID": "D", "VISCODE": "bl", "IMAGEUID": "I5", "split": "train", "cache_file": "D_bl.pt"},
                {"PTID": "D", "VISCODE": "bl", "IMAGEUID": "I5", "split": "train", "cache_file": "D_bl.pt"},
            ]
            manifest = root / "manifest.csv"
            _write_manifest(manifest, rows)

            torch.save(_cache(scan_path="/fake/I1.nii.gz"), cache_dir / "A_bl.pt")
            torch.save(_cache(scan_path="/fake/I2.nii.gz"), cache_dir / "A_m12.pt")
            nan_cache = _cache(scan_path="/fake/I3.nii.gz")
            nan_cache["embeddings"][0, 0] = float("nan")
            torch.save(nan_cache, cache_dir / "B_bl.pt")
            wrong_shape = _cache(scan_path="/fake/I4.nii.gz")
            wrong_shape["coords"] = wrong_shape["coords"][:-1]
            torch.save(wrong_shape, cache_dir / "C_bl.pt")
            torch.save(_cache(scan_path="/fake/I1.nii.gz"), cache_dir / "unexpected_copy.pt")

            report = audit_cache(manifest, cache_dir)
            codes = {issue["code"] for issue in report["issues"]}

            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(
                {
                    "NONFINITE_EMBEDDINGS",
                    "TOKEN_COORD_COUNT_MISMATCH",
                    "MISSING_CACHE",
                    "UNEXPECTED_CACHE",
                    "DUPLICATE_MANIFEST_IDENTITY",
                    "DUPLICATE_CACHE_REFERENCE",
                    "DUPLICATE_SOURCE_SCAN",
                    "SUBJECT_SPLIT_LEAKAGE",
                }.issubset(codes)
            )
            self.assertEqual(report["missing_subjects"], ["D"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
