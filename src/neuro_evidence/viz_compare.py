"""Side-by-side: which patches reach NeuroVFM for full head vs mask rule vs skull-stripped input.

Rows (same scan, same slices):
  1. full head, NeuroVFM rule only     (drop patch if ANY voxel <= 10th percentile)
  2. full head + HD-BET mask rule      (also keep every patch with >= 50% brain)
  3. skull-stripped, NeuroVFM rule only (image * HD-BET mask)

Usage (from src/):
    python -m neuro_evidence.viz_compare ../data/processed/n4_crop_1mm/<scan>.nii.gz --out ../outputs/figures/x.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import repo_root, resolve_path
from .encoder import _import_neurovfm, find_mask, patch_brain_fraction

# label -> (rgba, legend text)
LABELS = {
    1: ((0.1, 0.85, 0.3, 0.45), "brain block kept"),
    2: ((0.95, 0.15, 0.15, 0.55), "brain block DROPPED"),
    3: ((1.0, 0.6, 0.0, 0.55), "brain block added back by mask rule"),
    4: ((0.45, 0.45, 1.0, 0.45), "non-brain block kept (skull/scalp, or pure black if stripped)"),
}


def token_selection(scan_path: Path, mask_path: Path, patch_size=(4, 16, 16)):
    """NeuroVFM's volume, its own foreground rule per patch, and HD-BET brain fraction per patch."""
    from neurovfm.data.io import load_image
    from neurovfm.data.preprocess import prepare_for_inference, tokenize_volume

    img = load_image(str(scan_path), preprocess=True)
    arrs, bg, _ = prepare_for_inference(img, mode="mri")
    vol = np.asarray(arrs[0], dtype=np.float32)
    _, coords, filtered = tokenize_volume(vol, bg, patch_size=patch_size, remove_background=False)
    return vol, coords, ~filtered.astype(bool), patch_brain_fraction(mask_path, scan_path, patch_size)


def label_grid(coords, foreground, bf, mask_rule: bool, grid_shape, min_bf: float = 0.5) -> np.ndarray:
    brain = bf >= min_bf
    lab = np.zeros(len(coords), dtype=np.uint8)
    lab[brain & foreground] = 1
    lab[brain & ~foreground] = 3 if mask_rule else 2
    lab[~brain & foreground] = 4
    grid = np.zeros(grid_shape, dtype=np.uint8)
    grid[tuple(coords.T)] = lab
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan", help="full-head scan from neuro_evidence.preprocess (mask next to it)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import SimpleITK as sitk
    from matplotlib.patches import Patch

    _import_neurovfm("ext/neurovfm")
    from neurovfm.data.io import load_image

    scan = resolve_path(args.scan, base=Path.cwd()).resolve()
    mask = find_mask(scan)
    if mask is None:
        raise FileNotFoundError(f"no <stem>_mask.nii.gz next to {scan}")
    stripped = repo_root() / "data/processed/stripped_demo" / scan.name
    stripped.parent.mkdir(parents=True, exist_ok=True)
    img = sitk.ReadImage(str(scan), sitk.sitkFloat32)
    sitk.WriteImage(img * sitk.Cast(sitk.ReadImage(str(mask)), sitk.sitkFloat32), str(stripped))

    pd_, ph, pw = 4, 16, 16
    full_vol, coords, full_fg, bf = token_selection(scan, mask)
    strip_vol, coords_s, strip_fg, bf_s = token_selection(stripped, mask)
    assert np.array_equal(coords, coords_s) and np.allclose(bf, bf_s)
    grid_shape = tuple(s // p for s, p in zip(full_vol.shape, (pd_, ph, pw), strict=True))
    # HD-BET mask on NeuroVFM's grid, for the brain outline.
    mimg = load_image(str(stripped), preprocess=True)  # same geometry as the volumes
    brain_vox = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage(str(mask)), mimg, sitk.Transform(), sitk.sitkNearestNeighbor)) > 0
    if brain_vox.shape != full_vol.shape:
        brain_vox = None

    rows = [
        ("1. Full head, NeuroVFM rule", full_vol, label_grid(coords, full_fg, bf, False, grid_shape)),
        ("2. Full head + mask rule (what we use)", full_vol, label_grid(coords, full_fg, bf, True, grid_shape)),
        ("3. Skull-stripped, NeuroVFM rule", strip_vol, label_grid(coords, strip_fg, bf, False, grid_shape)),
    ]

    # Slices through the brain centre, at patch centres.
    bd, bh, bw = (np.asarray(coords[bf >= 0.5]).mean(axis=0)).round().astype(int)
    zd, zh, zw = bd * pd_ + pd_ // 2, bh * ph + ph // 2, bw * pw + pw // 2
    n_brain = int((bf >= 0.5).sum())
    edge = (bf >= 0.5) & (bf < 1.0)

    fig, axes = plt.subplots(3, 3, figsize=(14, 15), facecolor="white")
    for r, (title, vol, grid) in enumerate(rows):
        labels = grid.repeat(pd_, 0).repeat(ph, 1).repeat(pw, 2)
        rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
        for k, (c, _) in LABELS.items():
            rgba[labels == k] = c
        views = [("axial", vol[zd], rgba[zd], None if brain_vox is None else brain_vox[zd], 1),
                 ("coronal", vol[:, zh, :], rgba[:, zh, :], None if brain_vox is None else brain_vox[:, zh, :], pd_),
                 ("sagittal", vol[:, :, zw], rgba[:, :, zw], None if brain_vox is None else brain_vox[:, :, zw], pd_)]
        for c, (name, im, ov, bv, aspect) in enumerate(views):
            ax = axes[r, c]
            ax.imshow(im, cmap="gray", vmin=0, vmax=1, aspect=aspect)
            ax.imshow(ov, aspect=aspect)
            if bv is not None:
                ax.contour(bv, levels=[0.5], colors="yellow", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=12)
        g = grid[tuple(coords.T)]
        kept_brain = int(((g == 1) | (g == 3)).sum())
        kept_edge = int((((g == 1) | (g == 3)) & edge).sum())
        axes[r, 0].set_ylabel(title, fontsize=12, fontweight="bold")
        axes[r, 1].text(0.5, -0.08,
                        f"brain blocks kept {kept_brain}/{n_brain}  |  brain-edge blocks kept {kept_edge}/{int(edge.sum())}  |  "
                        f"non-brain blocks fed to model {int((g == 4).sum())}",
                        transform=axes[r, 1].transAxes, ha="center", va="top", fontsize=11)
    fig.legend(handles=[Patch(color=c[:3], label=t) for c, t in LABELS.values()] + [Patch(edgecolor="yellow", facecolor="none", label="HD-BET brain edge")],
               loc="lower center", ncol=3, fontsize=11, frameon=False)
    fig.suptitle(f"{scan.name[:40]}: which 4x16x16 mm blocks NeuroVFM sees", fontsize=14)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    print(out)


if __name__ == "__main__":
    main()
