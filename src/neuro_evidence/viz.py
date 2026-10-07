"""Show the exact volume NeuroVFM sees and which tokens reach the agent.

Recomputes NeuroVFM's preprocessing (reorient, 1x1x4mm, intensity normalization) for a scan
and overlays the cached token selection from `outputs/tokens/<sample_id>.npz`:
  green  = brain token (brain_fraction >= threshold): what the agent loads
  orange = context token: fed to NeuroVFM, dropped before the agent
  none   = patch dropped by the encoder

Usage (from src/):
    python -m neuro_evidence.viz ../data/processed/fullhead_1mm/<scan>.nii.gz --out ../outputs/figures/x.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .data import load_config, repo_root, resolve_path
from .encoder import _import_neurovfm, load_token_batch, sample_id_from_path

COLORS = {1: (1.0, 0.6, 0.0, 0.35), 2: (0.1, 0.85, 0.3, 0.35)}  # context, brain


def neurovfm_volume(scan_path: str | Path, neurovfm_repo: str | Path = "ext/neurovfm") -> np.ndarray:
    """(D, H, W) intensity-normalized array exactly as tokenized by NeuroVFM."""
    _import_neurovfm(neurovfm_repo)
    from neurovfm.data.io import load_image
    from neurovfm.data.preprocess import prepare_for_inference

    img = load_image(str(resolve_path(scan_path, base=repo_root())), preprocess=True)
    img_arrs, _bg, _view = prepare_for_inference(img, mode="mri")
    return np.asarray(img_arrs[0], dtype=np.float32)


def token_label_grid(batch, min_brain_fraction: float = 0.5) -> np.ndarray:
    """Dense (d, h, w) patch grid: 0 = dropped, 1 = context token, 2 = brain token."""
    grid = np.zeros(batch.dense_grid_shape, dtype=np.uint8)
    brain = batch.brain_fraction >= min_brain_fraction
    d, h, w = batch.coords.T
    grid[d, h, w] = np.where(brain, 2, 1)
    return grid


def plot(volume: np.ndarray, grid: np.ndarray, patch_size: tuple[int, int, int], title: str, out: str | Path, n_slices: int = 8) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pd_, ph, pw = patch_size
    # Upsample the patch grid to voxel resolution for overlays.
    labels = grid.repeat(pd_, 0).repeat(ph, 1).repeat(pw, 2)
    rgba = np.zeros(labels.shape + (4,), dtype=np.float32)
    for k, c in COLORS.items():
        rgba[labels == k] = c

    D, H, W = volume.shape
    rows = [d * pd_ + pd_ // 2 for d in np.linspace(1, grid.shape[0] - 2, n_slices).round().astype(int)]
    fig, axes = plt.subplots(3, n_slices // 2 + 1, figsize=(3.2 * (n_slices // 2 + 1), 10), facecolor="black")
    flat = axes.ravel()
    for ax, z in zip(flat, rows, strict=False):
        ax.imshow(volume[z], cmap="gray", vmin=0, vmax=1)
        ax.imshow(rgba[z])
        ax.set_title(f"slice D={z} (patch row {z // pd_})", color="white", fontsize=9)
    # Orthogonal views through the centre (4mm along D, so stretch aspect).
    for ax, (img, ov, name) in zip(
        flat[len(rows):],
        [(volume[:, H // 2, :], rgba[:, H // 2, :], f"orthogonal H={H // 2}"), (volume[:, :, W // 2], rgba[:, :, W // 2], f"orthogonal W={W // 2}")],
        strict=False,
    ):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, aspect=4)
        ax.imshow(ov, aspect=4)
        ax.set_title(name, color="white", fontsize=9)
    hist_ax = flat[len(rows) + 2]
    hist_ax.set_facecolor("black")
    hist_ax.hist(volume[volume > 0].ravel(), bins=100, color="0.8")
    hist_ax.set_title("normalized intensity (>0)", color="white", fontsize=9)
    hist_ax.tick_params(colors="white", labelsize=7)
    for ax in flat:
        if ax is not hist_ax:
            ax.axis("off")
    for ax in flat[len(rows) + 3:]:
        ax.set_visible(False)

    n_ctx, n_brain = int((grid == 1).sum()), int((grid == 2).sum())
    fig.suptitle(
        f"{title}\nNeuroVFM input volume {D}x{H}x{W} (D 4mm, H/W 1mm), patches {pd_}x{ph}x{pw}, grid {tuple(grid.shape)} = {grid.size} | "
        f"encoder tokens {n_ctx + n_brain} | agent tokens {n_brain}",
        color="white",
        fontsize=11,
    )
    fig.legend(
        handles=[Patch(color=COLORS[2][:3], label="brain token -> agent"), Patch(color=COLORS[1][:3], label="context token (encoder only)")],
        loc="lower right",
        facecolor="black",
        labelcolor="white",
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan", help="full-head scan from neuro_evidence.preprocess")
    parser.add_argument("--tokens", help="token cache (default outputs/tokens/<sample_id>.npz)")
    parser.add_argument("--min-brain-fraction", type=float, default=0.5)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    scan = resolve_path(args.scan, base=Path.cwd()).resolve()
    tokens = Path(args.tokens) if args.tokens else resolve_path(cfg["outputs_dir"], base=repo_root()) / "tokens" / f"{sample_id_from_path(scan)}.npz"
    batch = load_token_batch(tokens)
    volume = neurovfm_volume(scan, cfg["neurovfm"]["repo"])
    if tuple(volume.shape) != tuple(batch.volume_size_dhw):
        raise ValueError(f"volume {volume.shape} != cached {batch.volume_size_dhw}; token cache is stale for {scan.name}")
    print(plot(volume, token_label_grid(batch, args.min_brain_fraction), batch.patch_size, scan.name, args.out))


if __name__ == "__main__":
    main()
