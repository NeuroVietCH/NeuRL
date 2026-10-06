"""MRI cleanup before NeuroVFM: N4 bias correction -> RAS 1mm -> HD-BET skull strip -> brain crop.

Ported from brainet's `preprocess_oasis_mri_for_fm.py` (shared `shared_n4_hdbet_crop` profile).
NeuroVFM's own StudyPreprocessor then handles RPI reorientation, 1x1x4mm resampling,
intensity normalization and tokenization.

Usage:
    python -m neuro_evidence.preprocess --manifest data/adnidod/manifest.csv
    python -m neuro_evidence.preprocess scan1.nii scan2.nii.gz --out-dir data/processed
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk

from .data import load_config, repo_root, resolve_path

DEFAULTS: dict[str, Any] = {
    "out_dir": "data/processed/n4_hdbet_crop",
    "orientation": "RAS",
    "spacing": [1.0, 1.0, 1.0],
    "n4": {"enabled": True, "shrink_factor": 4, "iterations": [50, 50, 30, 20], "convergence_threshold": 1.0e-7},
    "skull_strip": {"method": "hdbet", "device": "cuda", "tta": False},
}


def read_image(path: str | Path) -> sitk.Image:
    image = sitk.ReadImage(str(path), sitk.sitkFloat32)
    if image.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D image, got dimension={image.GetDimension()} from {path}")
    return image


def n4_bias_correct(image: sitk.Image, cfg: dict[str, Any]) -> sitk.Image:
    if not cfg.get("enabled", True):
        return image
    mask = sitk.OtsuThreshold(image, 0, 1, 200)
    shrink = max(1, int(cfg["shrink_factor"]))
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([int(v) for v in cfg["iterations"]])
    corrector.SetConvergenceThreshold(float(cfg["convergence_threshold"]))
    if shrink > 1:
        corrector.Execute(sitk.Shrink(image, [shrink] * 3), sitk.Shrink(mask, [shrink] * 3))
        return image / sitk.Exp(corrector.GetLogBiasFieldAsImage(image))
    return corrector.Execute(image, mask)


def resample_spacing(image: sitk.Image, spacing: tuple[float, float, float], *, is_mask: bool = False) -> sitk.Image:
    old_spacing, old_size = image.GetSpacing(), image.GetSize()
    new_size = [max(1, int(round(old_size[i] * old_spacing[i] / spacing[i]))) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline)
    return resampler.Execute(image)


def fallback_mask(image: sitk.Image) -> sitk.Image:
    """Intensity threshold + largest component. Smoke tests only: does not remove skull reliably."""
    arr = sitk.GetArrayFromImage(image)
    threshold = float(np.percentile(arr[np.isfinite(arr)], 5))
    mask = sitk.BinaryThreshold(image, lowerThreshold=threshold, upperThreshold=float(arr.max()), insideValue=1, outsideValue=0)
    mask = sitk.BinaryMorphologicalClosing(mask, [2, 2, 2])
    relabeled = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    return sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)


def hdbet_mask(image: sitk.Image, cfg: dict[str, Any]) -> sitk.Image:
    executable = Path(sys.executable).parent / "hd-bet"
    executable = str(executable) if executable.exists() else shutil.which("hd-bet")
    if executable is None:
        raise RuntimeError("HD-BET CLI not found. `pip install hd-bet` or set skull_strip.method=fallback.")
    with tempfile.TemporaryDirectory(prefix="neurl-hdbet-") as tmp:
        tmp_dir = Path(tmp)
        input_path, output_path = tmp_dir / "input.nii.gz", tmp_dir / "brain.nii.gz"
        sitk.WriteImage(image, str(input_path), True)
        cmd = [executable, "-i", str(input_path), "-o", str(output_path), "-device", str(cfg.get("device", "cuda")), "--save_bet_mask"]
        if not cfg.get("tta", False):
            cmd.append("--disable_tta")
        subprocess.run(cmd, check=True)
        masks = sorted(tmp_dir.glob("*_bet.nii*")) + sorted(tmp_dir.glob("*mask*.nii*"))
        if not masks:
            raise RuntimeError("HD-BET finished but produced no mask file.")
        mask = sitk.ReadImage(str(masks[0]), sitk.sitkUInt8)
    mask.CopyInformation(image)
    return mask


def crop_to_mask(image: sitk.Image, mask: sitk.Image) -> tuple[sitk.Image, sitk.Image]:
    coords = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
    if coords.size == 0:
        raise RuntimeError("Brain mask is empty.")
    zyx_min, zyx_max = coords.min(axis=0), coords.max(axis=0) + 1
    # SimpleITK index/size order is x, y, z.
    start = [int(v) for v in zyx_min[::-1]]
    size = [int(v) for v in (zyx_max - zyx_min)[::-1]]
    return sitk.RegionOfInterest(image, size=size, index=start), sitk.RegionOfInterest(mask, size=size, index=start)


def output_stem(path: str | Path) -> str:
    return Path(path).name.replace(".nii.gz", "").replace(".nii", "")


def preprocess_scan(path: str | Path, out_dir: str | Path, cfg: dict[str, Any] | None = None, overwrite: bool = False) -> dict[str, Any]:
    cfg = {**DEFAULTS, **(cfg or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(path)
    image_out, mask_out = out_dir / f"{stem}.nii.gz", out_dir / f"{stem}_mask.nii.gz"
    method = str(cfg["skull_strip"]["method"]).lower()
    if image_out.exists() and mask_out.exists() and not overwrite:
        return {"source_path": str(path), "processed_path": str(image_out), "mask_path": str(mask_out), "skull_strip": method, "skipped": True}

    image = read_image(path)
    image = n4_bias_correct(image, cfg["n4"])
    image = sitk.DICOMOrient(image, str(cfg["orientation"]))
    image = resample_spacing(image, tuple(float(v) for v in cfg["spacing"]))
    if method == "hdbet":
        mask = hdbet_mask(image, cfg["skull_strip"])
    elif method == "fallback":
        mask = fallback_mask(image)
    else:
        raise ValueError(f"Unknown skull_strip.method `{method}` (use hdbet or fallback)")
    image = sitk.Mask(image, sitk.Cast(mask, sitk.sitkUInt8))
    image, mask = crop_to_mask(image, mask)

    sitk.WriteImage(image, str(image_out), True)
    sitk.WriteImage(mask, str(mask_out), True)
    return {
        "source_path": str(path),
        "processed_path": str(image_out),
        "mask_path": str(mask_out),
        "skull_strip": method,
        "shape_xyz": list(image.GetSize()),
        "brain_voxels": int(sitk.GetArrayViewFromImage(mask).sum()),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans", nargs="*", help="NIfTI files to preprocess")
    parser.add_argument("--manifest", help="CSV with an image_path column; writes <manifest>_processed.csv next to it")
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML with an optional `preprocess:` section")
    parser.add_argument("--out-dir", help="Overrides preprocess.out_dir")
    parser.add_argument("--skull-strip", choices=["hdbet", "fallback"], help="Overrides preprocess.skull_strip.method")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = {**DEFAULTS, **load_config(args.config).get("preprocess", {})}
    if args.skull_strip:
        cfg["skull_strip"] = {**cfg["skull_strip"], "method": args.skull_strip}
    out_dir = resolve_path(args.out_dir or cfg["out_dir"], base=repo_root())

    manifest = None
    scans = [resolve_path(s, base=Path.cwd()) for s in args.scans]
    if args.manifest:
        import pandas as pd

        manifest = pd.read_csv(resolve_path(args.manifest, base=Path.cwd()))
        scans += [resolve_path(p, base=repo_root()) for p in manifest["image_path"]]
    if not scans:
        parser.error("pass NIfTI paths and/or --manifest")

    results = []
    for i, scan in enumerate(scans, 1):
        print(f"[{i}/{len(scans)}] {scan.name}", flush=True)
        results.append(preprocess_scan(scan, out_dir, cfg, overwrite=args.overwrite))

    if manifest is not None:
        import pandas as pd

        res = pd.DataFrame(results[len(args.scans):])
        manifest["processed_path"] = [str(Path(p).relative_to(repo_root())) if Path(p).is_relative_to(repo_root()) else p for p in res["processed_path"]]
        manifest["mask_path"] = [str(Path(p).relative_to(repo_root())) if Path(p).is_relative_to(repo_root()) else p for p in res["mask_path"]]
        out_csv = resolve_path(args.manifest, base=Path.cwd()).with_name(Path(args.manifest).stem + "_processed.csv")
        manifest.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
