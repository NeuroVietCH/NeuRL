"""Neuro-JEPA (NYUMedML/Neuro-JEPA-Large) encoder: MNI preprocessing + token extraction.

Neuro-JEPA was pretrained on T1s that went through registration/register_t1w.sh in its repo:
fslreorient2std -> robustfov -> N4 -> FLIRT 12-DOF affine to MNI152 1mm -> mri_synthstrip.
FSL/ANTs/FreeSurfer are not installed here, so `prep` is a Python port of those steps:
  reorient RAS -> drop neck (keep 170 mm below the top of the head, as robustfov) -> SimpleITK N4 ->
  SimpleITK similarity then 12-DOF affine (Mattes MI) to the MNI152NLin6Asym 1mm head template
  (FSL's MNI152_T1_1mm grid, 182x218x182) -> B-spline resample -> HD-BET skull strip (pixels zeroed).
`extract` then applies Neuro-JEPA's own MONAI transforms (RAS, 0.5-99.5 pct scaling, pad/crop
180x216x180, foreground crop, resize 100x120x100, centre crop 96x108x96) and the ViT backbone:
576 tokens (8x9x8 grid of 12^3 patches, C order over RAS x,y,z) x 1024 dims (ViT-Large).

  cd src && python -m neuro_evidence.neurojepa prep    --shard 0 --num-shards 4
  cd src && python -m neuro_evidence.neurojepa extract --shard 0 --num-shards 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from env.data import remap_data_path, token_name

from .data import load_config, repo_root, resolve_path
from .preprocess import hdbet_mask, head_mask, n4_bias_correct, read_image

DATA_ROOT = "/mnt/shared/datasets/NeuRL-dataset"
MANIFEST = f"{DATA_ROOT}/oasis/processed_mri/oasis_neurovfm_canonical_manifest.csv"


def _import_neurojepa(repo: str | Path) -> None:
    path = str(resolve_path(Path(repo) / "src", base=repo_root()).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def drop_neck(image: sitk.Image, keep_mm: float = 170.0) -> sitk.Image:
    """robustfov-style FOV cut on a RAS image: keep `keep_mm` below the top of the head mask."""
    mask = sitk.GetArrayFromImage(head_mask(image))  # (z, y, x), z = S axis in RAS
    rows = np.flatnonzero(mask.any(axis=(1, 2)))
    if rows.size == 0:
        return image
    top = int(rows[-1])
    bottom = max(0, top - int(round(keep_mm / image.GetSpacing()[2])))
    return image[:, :, bottom : top + 1]


def affine_to_template(moving: sitk.Image, fixed: sitk.Image, seed: int = 0) -> sitk.Transform:
    """Similarity (7 DOF) then full affine (12 DOF), Mattes MI, 3-level pyramid."""

    def run(initial: sitk.Transform, moving_initial: sitk.Transform | None, iters: int) -> sitk.Transform:
        reg = sitk.ImageRegistrationMethod()
        reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.1, seed)
        reg.SetInterpolator(sitk.sitkLinear)
        reg.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=iters, relaxationFactor=0.5)
        reg.SetOptimizerScalesFromPhysicalShift()
        reg.SetShrinkFactorsPerLevel([4, 2, 1])
        reg.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
        reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        if moving_initial is not None:
            reg.SetMovingInitialTransform(moving_initial)
        reg.SetInitialTransform(initial, inPlace=False)
        return reg.Execute(fixed, moving)

    fixed, moving = sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)
    init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Similarity3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    similarity = run(init, None, 200)
    affine = sitk.AffineTransform(3)
    affine.SetCenter(sitk.Similarity3DTransform(init).GetCenter())
    affine = run(affine, similarity, 200)
    return sitk.CompositeTransform([similarity, affine])


def prep_scan(scan_path: str | Path, out_path: str | Path, cfg: dict[str, Any]) -> dict[str, Any]:
    """Raw T1 -> skull-stripped MNI152 1mm volume (+ `<stem>_mask.nii.gz`), Neuro-JEPA style."""
    nj = cfg["neurojepa"]
    template = sitk.ReadImage(str(resolve_path(nj["template"], base=repo_root())), sitk.sitkFloat32)
    image = sitk.DICOMOrient(read_image(scan_path), "RAS")
    image = drop_neck(image, float(nj.get("fov_mm", 170.0)))
    image = n4_bias_correct(image, cfg["preprocess"]["n4"] | {"enabled": True})
    tx = affine_to_template(image, template)
    registered = sitk.Resample(image, template, tx, sitk.sitkBSpline, 0.0, sitk.sitkFloat32)
    registered = sitk.Clamp(registered, lowerBound=0.0)  # B-spline rings below 0 at edges
    mask = hdbet_mask(registered, {"device": nj.get("device", "cuda"), "tta": False})
    stripped = registered * sitk.Cast(mask, sitk.sitkFloat32)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.name.replace(".nii.gz", "")
    sitk.WriteImage(stripped, str(out_path), True)
    sitk.WriteImage(mask, str(out_path.with_name(f"{stem}_mask.nii.gz")), True)
    # Registration check: Dice of our brain mask vs the template's.
    tmask = sitk.GetArrayFromImage(sitk.ReadImage(str(resolve_path(nj["template_mask"], base=repo_root())))) > 0
    m = sitk.GetArrayFromImage(mask) > 0
    return {"dice_vs_template_brain": float(2 * (m & tmask).sum() / max(1, m.sum() + tmask.sum()))}


class NeuroJEPAEncoder:
    """Adapter around the official Neuro-JEPA backbone and MONAI transforms."""

    name = "neurojepa"

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from types import SimpleNamespace

        nj = config["neurojepa"]
        _import_neurojepa(nj["repo"])
        from neurojepa.data.transforms import loading_transforms, vit3d_transforms
        from neurojepa.utils.init_utils import load_backbone_from_hf

        self.device = nj.get("device", "cuda")
        self.img_size = tuple(int(v) for v in nj.get("img_size", (96, 108, 96)))
        self.patch_size = tuple(int(v) for v in nj.get("patch_size", (12, 12, 12)))
        weights = resolve_path(nj["checkpoint"], base=repo_root())
        if not (weights / "config.json").exists():
            raise FileNotFoundError(f"No Neuro-JEPA weights at {weights} (need config.json + model.safetensors from the gated HF repo)")
        self.backbone = load_backbone_from_hf(str(weights), device=self.device).eval()
        self.transforms = [
            loading_transforms(roi=self.img_size, spacing=(1.0, 1.0, 1.0)),
            vit3d_transforms(SimpleNamespace(data=SimpleNamespace(img_size=list(self.img_size))), mode="test"),
        ]
        self.use_amp = bool(nj.get("use_amp", True))
        self._torch = torch

    def extract(self, mni_path: str | Path) -> dict[str, Any]:
        """Tokens for one prepped (MNI, skull-stripped) volume, in the cached-token format."""
        torch = self._torch
        sample = {"image": str(mni_path)}
        for t in self.transforms:
            sample = t(sample)
        x = torch.as_tensor(sample["image"]).float().unsqueeze(0).to(self.device)  # [1, 1, 96, 108, 96]
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.use_amp):
            tokens, _moe_scores = self.backbone(x)
        grid = tuple(s // p for s, p in zip(self.img_size, self.patch_size, strict=True))
        coords = np.indices(grid).reshape(3, -1).T  # matches Conv3d(...).flatten(2)
        p = self.patch_size
        fg = (x[0, 0] > 0).float().reshape(grid[0], p[0], grid[1], p[1], grid[2], p[2]).mean(dim=(1, 3, 5)).reshape(-1)
        emb = tokens[0].float()
        return {
            "embeddings": emb.half().cpu(),
            "coords": torch.from_numpy(coords).short(),
            "token_count": int(emb.shape[0]),
            "embedding_dim": int(emb.shape[1]),
            "grid_shape": torch.tensor(grid, dtype=torch.int32),
            "brain_fraction": fg.cpu(),  # fraction of each 12^3 patch inside the stripped brain
            "scan_path": str(mni_path),
            "metadata": json.dumps({"encoder": self.name, "coords_axes": "RAS x,y,z (MNI152 after Neuro-JEPA crop/resize)"}),
        }


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prep", "extract"])
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--manifest", default=MANIFEST)
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Only the first N scans of this shard (smoke test)")
    parser.add_argument("--mni-dir", help="Overrides neurojepa.mni_dir")
    parser.add_argument("--token-dir", help="Overrides neurojepa.token_dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    nj = cfg["neurojepa"]
    mni_dir = Path(args.mni_dir or nj["mni_dir"])
    token_dir = Path(args.token_dir or nj["token_dir"])
    out_dir = mni_dir if args.stage == "prep" else token_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(args.manifest).iloc[args.shard :: args.num_shards]
    if args.limit:
        meta = meta.iloc[: args.limit]
    enc = NeuroJEPAEncoder(cfg) if args.stage == "extract" else None

    log = out_dir / f"neurojepa_{args.stage}_shard{args.shard}of{args.num_shards}.log"
    done = failed = 0
    t0 = time.time()
    with open(log, "a") as fh:
        for i, row in enumerate(meta.itertuples(index=False)):
            name = token_name(row.PTID, row.VISCODE, row.EXAMDATE).removesuffix(".pt")
            mni_path = mni_dir / f"{name}.nii.gz"
            out = mni_path if args.stage == "prep" else token_dir / f"{name}.pt"
            if out.exists():
                continue
            try:
                if args.stage == "prep":
                    info = prep_scan(remap_data_path(row.source_image_path, args.data_root), mni_path, cfg)
                    fh.write(f"OK {name} dice={info['dice_vs_template_brain']:.3f}\n")
                else:
                    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
                    torch.save(enc.extract(mni_path), tmp)
                    tmp.replace(out)
                done += 1
            except Exception:
                failed += 1
                fh.write(f"FAIL {name}\n{traceback.format_exc()}\n")
            if (i + 1) % 25 == 0:
                msg = f"[{i + 1}/{len(meta)}] done={done} failed={failed} {time.time() - t0:.0f}s"
                print(msg, flush=True)
                fh.write(msg + "\n")
            fh.flush()
        msg = f"DONE {args.stage} shard {args.shard}/{args.num_shards}: done={done} failed={failed} {time.time() - t0:.0f}s"
        print(msg, flush=True)
        fh.write(msg + "\n")


if __name__ == "__main__":
    main()
