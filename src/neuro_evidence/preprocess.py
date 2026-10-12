"""MRI preparation before NeuroVFM.

Default = strict: NeuroVFM's own pipeline. The scan is written unchanged (DICOM -> NIfTI, native pixel
type and pose); NeuroVFM's load_image/prepare_for_inference then do RPI reorientation, 1x1x4mm
resampling with 4mm along the acquisition axis, percentile normalization and tokenization. NeuroVFM
was trained on raw clinical scans: no N4, no registration, no skull stripping.

Optional steps (all off by default, in this order), set in the `preprocess:` config section:
  n4.enabled            N4 bias correction
  orientation           reorient, e.g. RAS
  spacing               resample, e.g. [1, 1, 1]. Isotropic input makes NeuroVFM cut 4mm slices along
                        I-S instead of the acquisition axis (sagittal for ADNI MPRAGE).
  skull_strip.method    none | hdbet | fallback. hdbet saves an HD-BET brain mask `<stem>_mask.nii.gz`
                        that encoder.py uses to add/label brain tokens (never applied to pixels);
                        fallback = threshold mask, smoke tests only.
  crop.enabled          crop the FOV to the brain mask (or head mask with method none) + margins.
An Otsu head mask (air vs head) is always saved as `<stem>_head.nii.gz` for encoder token_rule=head.
Pixels are never skull-stripped: zeroing everything outside the brain shifts NeuroVFM's intensity
normalization and breaks its background filter.

Usage (from src/):
    python -m neuro_evidence.preprocess --manifest ../data/adnidod/manifest.csv
    python -m neuro_evidence.preprocess scan1.nii scan2.nii.gz path/to/dicom_series_dir --out-dir ../data/processed
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
    "out_dir": "data/processed/strict",
    "orientation": None,  # e.g. "RAS"
    "spacing": None,  # e.g. [1.0, 1.0, 1.0]
    "n4": {"enabled": False, "shrink_factor": 4, "iterations": [50, 50, 30, 20], "convergence_threshold": 1.0e-7},
    "skull_strip": {"method": "none", "device": "cuda", "tta": False},
    # mm kept around the mask bounding box; `inferior` is below it (brainstem/neck side).
    "crop": {"enabled": False, "margin_mm": 5.0, "inferior_margin_mm": 5.0},
}


def merge_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """DEFAULTS overridden by `cfg`, one level deep (a partial `n4:` section keeps the other n4 keys)."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in DEFAULTS.items()}
    for k, v in (cfg or {}).items():
        out[k] = {**out[k], **v} if isinstance(out.get(k), dict) and isinstance(v, dict) else v
    return out


def read_image(path: str | Path) -> sitk.Image:
    """NIfTI file, or a directory holding one DICOM series (the largest series if there are several).

    Keeps the native pixel type, as NeuroVFM's own loaders do.
    """
    path = Path(path)
    if path.is_dir():
        reader = sitk.ImageSeriesReader()
        series = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(path))
        if not series:
            raise RuntimeError(f"No DICOM series in {path}")
        files = max((reader.GetGDCMSeriesFileNames(str(path), sid) for sid in series), key=len)
        reader.SetFileNames(files)
        image = reader.Execute()
    else:
        image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise RuntimeError(f"Expected 3D image, got dimension={image.GetDimension()} from {path}")
    return image


def n4_bias_correct(image: sitk.Image, cfg: dict[str, Any]) -> sitk.Image:
    if not cfg.get("enabled", True):
        return image
    image = sitk.Cast(image, sitk.sitkFloat32)
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
    image = sitk.Cast(image, sitk.sitkFloat32)
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


def head_mask(image: sitk.Image) -> sitk.Image:
    """Air vs head: Otsu on log intensity, largest connected component, holes closed. Keeps skull, scalp and neck.

    Otsu on raw T1 intensities splits tissue classes (it lands near GM/WM), not air vs head.
    """
    image = sitk.Cast(image, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(sitk.Log(sitk.Clamp(image, lowerBound=0.0) + 1.0), 0, 1, 200)
    relabeled = sitk.RelabelComponent(sitk.ConnectedComponent(mask), sortByObjectSize=True)
    mask = sitk.BinaryThreshold(relabeled, lowerThreshold=1, upperThreshold=1, insideValue=1, outsideValue=0)
    return sitk.BinaryFillhole(sitk.BinaryMorphologicalClosing(mask, [3, 3, 3]))


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
        raise RuntimeError("HD-BET CLI not found. `pip install hd-bet` or set skull_strip.method=none.")
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


def crop_to_mask(image: sitk.Image, mask: sitk.Image, margin_mm: float, inferior_margin_mm: float) -> tuple[sitk.Image, sitk.Image, list[int]]:
    """Crop image and mask to the mask's bounding box + margins. Works for any axis orientation.

    Returns the cropped pair and the crop box as [x0, y0, z0, x1, y1, z1] voxel indices of the input.
    """
    arr = sitk.GetArrayViewFromImage(mask)  # (z, y, x)
    if not arr.any():
        raise RuntimeError("Mask is empty; cannot crop")
    nz = np.nonzero(arr)
    lo = [int(nz[2].min()), int(nz[1].min()), int(nz[0].min())]
    hi = [int(nz[2].max()) + 1, int(nz[1].max()) + 1, int(nz[0].max()) + 1]
    direction = np.asarray(image.GetDirection()).reshape(3, 3)  # columns = index axes in LPS space
    si_axis = int(np.argmax(np.abs(direction[2])))  # index axis closest to physical S
    inferior_is_low = direction[2, si_axis] > 0  # index grows toward S -> inferior at low index
    size, spacing = image.GetSize(), image.GetSpacing()
    for i in range(3):
        m_lo = m_hi = margin_mm
        if i == si_axis:
            m_lo, m_hi = (inferior_margin_mm, margin_mm) if inferior_is_low else (margin_mm, inferior_margin_mm)
        lo[i] = max(0, lo[i] - int(round(m_lo / spacing[i])))
        hi[i] = min(size[i], hi[i] + int(round(m_hi / spacing[i])))
    box = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
    return image[box], mask[box], lo + hi


def output_stem(path: str | Path) -> str:
    return Path(path).name.replace(".nii.gz", "").replace(".nii", "")


def preprocess_scan(path: str | Path, out_dir: str | Path, cfg: dict[str, Any] | None = None, overwrite: bool = False) -> dict[str, Any]:
    cfg = merge_config(cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(path)
    image_out, mask_out, head_out = out_dir / f"{stem}.nii.gz", out_dir / f"{stem}_mask.nii.gz", out_dir / f"{stem}_head.nii.gz"
    method = str(cfg["skull_strip"]["method"]).lower()
    save_mask = method != "none"
    if image_out.exists() and head_out.exists() and (mask_out.exists() or not save_mask) and not overwrite:
        return {"source_path": str(path), "processed_path": str(image_out), "mask_path": str(mask_out) if save_mask else None, "head_mask_path": str(head_out), "skull_strip": method, "skipped": True}

    image = read_image(path)
    image = n4_bias_correct(image, cfg["n4"])
    if cfg["orientation"]:
        image = sitk.DICOMOrient(image, str(cfg["orientation"]))
    if cfg["spacing"]:
        image = resample_spacing(image, tuple(float(v) for v in cfg["spacing"]))
    head = head_mask(image)
    if method == "none":
        mask = head
    elif method == "hdbet":
        mask = hdbet_mask(image, cfg["skull_strip"])
    elif method == "fallback":
        mask = fallback_mask(image)
    else:
        raise ValueError(f"Unknown skull_strip.method `{method}` (use none, hdbet or fallback)")
    crop_box = None
    if cfg["crop"].get("enabled", True):
        image, mask, crop_box = crop_to_mask(image, mask, float(cfg["crop"]["margin_mm"]), float(cfg["crop"]["inferior_margin_mm"]))
        head = head[tuple(slice(lo, hi) for lo, hi in zip(crop_box[:3], crop_box[3:]))]

    sitk.WriteImage(image, str(image_out), True)
    sitk.WriteImage(head, str(head_out), True)
    if save_mask:
        sitk.WriteImage(mask, str(mask_out), True)
    return {
        "source_path": str(path),
        "processed_path": str(image_out),
        "mask_path": str(mask_out) if save_mask else None,
        "head_mask_path": str(head_out),
        "skull_strip": method,
        "n4": bool(cfg["n4"].get("enabled", True)),
        "orientation": cfg["orientation"],
        "spacing": cfg["spacing"],
        "crop_box": crop_box,
        "shape_xyz": list(image.GetSize()),
        "mask_voxels": int(sitk.GetArrayViewFromImage(mask).sum()),
        "skipped": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scans", nargs="*", help="NIfTI files or DICOM series directories to preprocess")
    parser.add_argument("--manifest", help="CSV with an image_path column; writes <manifest>_processed.csv next to it")
    parser.add_argument("--config", default="configs/baseline.yaml", help="YAML with an optional `preprocess:` section")
    parser.add_argument("--out-dir", help="Overrides preprocess.out_dir")
    parser.add_argument("--skull-strip", choices=["none", "hdbet", "fallback"], help="Overrides preprocess.skull_strip.method")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = merge_config(load_config(args.config).get("preprocess", {}))
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
        rel = lambda p: None if p is None else str(Path(p).relative_to(repo_root())) if Path(p).is_relative_to(repo_root()) else p
        manifest["processed_path"] = [rel(p) for p in res["processed_path"]]
        manifest["mask_path"] = [rel(p) for p in res["mask_path"]]
        manifest["head_mask_path"] = [rel(p) for p in res["head_mask_path"]]
        out_csv = resolve_path(args.manifest, base=Path.cwd()).with_name(Path(args.manifest).stem + "_processed.csv")
        manifest.to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
