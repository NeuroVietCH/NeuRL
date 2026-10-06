from __future__ import annotations

from pathlib import Path

from typing import Any

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# NeuroVFM's transpose_to_dhw: view -> axis permutation from the (z, y, x) array to (D, H, W).
_DHW_PERM = {0: (0, 1, 2), 1: (1, 0, 2), 2: (2, 1, 0)}


def _geometry_image(size_xyz, origin, spacing, direction) -> sitk.Image:
    img = sitk.Image([int(v) for v in size_xyz], sitk.sitkUInt8)
    img.SetOrigin([float(v) for v in origin])
    img.SetSpacing([float(v) for v in spacing])
    img.SetDirection([float(v) for v in direction])
    return img


def preprocessed_grid_shape(geometry: dict[str, Any], patch_size: tuple[int, int, int]) -> tuple[int, int, int]:
    """Token grid (D, H, W) implied by a preprocessed geometry."""
    zyx = list(geometry["size_xyz"])[::-1]
    dhw = [zyx[a] for a in _DHW_PERM[int(geometry["view"])]]
    return tuple(int(n // p) for n, p in zip(dhw, patch_size, strict=True))


def token_coords_to_voxels(
    token_coords: np.ndarray,
    patch_size: tuple[int, int, int],
    geometry: dict[str, Any],
    volume_path: str | Path,
) -> np.ndarray:
    """Map NeuroVFM token-grid coords (D, H, W) to voxel indices of the original scan.

    Token center -> preprocessed (D, H, W) voxel -> undo transpose -> SimpleITK
    index (x, y, z) -> physical mm -> continuous index in `volume_path`. Both ends
    use SimpleITK, so there is no LPS/RAS sign handling; the result is in the
    file's voxel order, the same order as `nib.load(volume_path).dataobj`.
    """
    coords = np.asarray(token_coords, dtype=np.float64).reshape(-1, 3)
    dhw = (coords + 0.5) * np.asarray(patch_size, dtype=np.float64) - 0.5
    zyx = np.empty_like(dhw)
    zyx[:, list(_DHW_PERM[int(geometry["view"])])] = dhw
    xyz = zyx[:, ::-1]

    pre = _geometry_image(geometry["size_xyz"], geometry["origin"], geometry["spacing"], geometry["direction"])
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(volume_path))
    reader.ReadImageInformation()
    orig = _geometry_image([1, 1, 1], reader.GetOrigin(), reader.GetSpacing(), reader.GetDirection())

    out = [orig.TransformPhysicalPointToContinuousIndex(pre.TransformContinuousIndexToPhysicalPoint(p.tolist())) for p in xyz]
    return np.asarray(out, dtype=np.float32).reshape(-1, 3)


def mri_space_coords(
    volume_path: str | Path,
    token_coords: np.ndarray,
    grid_shape: np.ndarray | tuple[int, int, int] | None = None,
    patch_size: tuple[int, int, int] = (4, 16, 16),
    geometry: dict[str, Any] | None = None,
) -> np.ndarray:
    """Token coords -> voxel coords of `volume_path`, recomputing geometry if not cached.

    `volume_path` must be the exact file the tokens were extracted from. When
    `grid_shape` is given it is checked against the geometry, which catches a
    token file paired with the wrong scan.
    """
    if geometry is None:
        from neuro_evidence.encoder import neurovfm_geometry

        geometry = neurovfm_geometry(volume_path)
    if grid_shape is not None:
        expected = preprocessed_grid_shape(geometry, patch_size)
        if tuple(int(v) for v in grid_shape) != expected:
            raise ValueError(f"token grid {tuple(grid_shape)} does not match {volume_path} preprocessed grid {expected}")
    return token_coords_to_voxels(token_coords, patch_size, geometry, volume_path)


def _normalize_slice(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    lo, hi = np.percentile(x[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def plot_mri_trajectory_3d(
    volume_path: str | Path,
    token_coords: np.ndarray,
    grid_shape: np.ndarray | tuple[int, int, int],
    anomaly_scores: np.ndarray,
    visited_indices: np.ndarray,
    inspected_indices: np.ndarray,
    title: str,
    out_path: str | Path | None = None,
    patch_size: tuple[int, int, int] = (4, 16, 16),
    geometry: dict[str, Any] | None = None,
):
    """Render MRI orthogonal slices with an overlaid token trajectory."""
    img = nib.load(str(volume_path))
    volume = np.asanyarray(img.dataobj).astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    voxel_coords = mri_space_coords(volume_path, token_coords, grid_shape, patch_size, geometry)

    visited = np.asarray(visited_indices, dtype=int)
    inspected = np.asarray(inspected_indices, dtype=int)
    path = voxel_coords[visited] if len(visited) else np.empty((0, 3), dtype=np.float32)
    inspected_xyz = voxel_coords[inspected] if len(inspected) else np.empty((0, 3), dtype=np.float32)

    if len(path):
        center = np.clip(np.round(path.mean(axis=0)).astype(int), 0, np.asarray(volume.shape) - 1)
    else:
        center = np.asarray(volume.shape) // 2
    x0, y0, z0 = [int(v) for v in center]

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    ax = fig.add_subplot(1, 2, 1, projection="3d")

    # Orthogonal grayscale MRI slices in native loaded voxel coordinates.
    y, z = np.meshgrid(np.arange(volume.shape[1]), np.arange(volume.shape[2]), indexing="ij")
    ax.plot_surface(
        np.full_like(y, x0), y, z,
        facecolors=plt.cm.gray(_normalize_slice(volume[x0, :, :])),
        rstride=2, cstride=2, shade=False, alpha=0.42, linewidth=0, antialiased=False,
    )
    x, z = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[2]), indexing="ij")
    ax.plot_surface(
        x, np.full_like(x, y0), z,
        facecolors=plt.cm.gray(_normalize_slice(volume[:, y0, :])),
        rstride=2, cstride=2, shade=False, alpha=0.36, linewidth=0, antialiased=False,
    )
    x, y = np.meshgrid(np.arange(volume.shape[0]), np.arange(volume.shape[1]), indexing="ij")
    ax.plot_surface(
        x, y, np.full_like(x, z0),
        facecolors=plt.cm.gray(_normalize_slice(volume[:, :, z0])),
        rstride=2, cstride=2, shade=False, alpha=0.30, linewidth=0, antialiased=False,
    )

    sc = None
    if len(path):
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color="red", linewidth=3.0, marker="o", markersize=5, label="visited path")
    if len(inspected_xyz):
        inspected_scores = np.asarray(anomaly_scores, dtype=np.float32)[inspected]
        sc = ax.scatter(
            inspected_xyz[:, 0], inspected_xyz[:, 1], inspected_xyz[:, 2],
            c=inspected_scores, cmap="magma", edgecolor="white", linewidth=1.2,
            s=95, depthshade=False, label="inspected evidence",
        )

    ax.set_title(title)
    ax.set_xlabel("MRI voxel axis 0")
    ax.set_ylabel("MRI voxel axis 1")
    ax.set_zlabel("MRI voxel axis 2")
    ax.set_xlim(0, volume.shape[0])
    ax.set_ylim(0, volume.shape[1])
    ax.set_zlim(0, volume.shape[2])
    ax.view_init(elev=22, azim=-55)
    ax.legend(loc="upper left")
    if sc is not None:
        fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08, label="inspected anomaly score")

    ax2 = fig.add_subplot(1, 2, 2)
    inspected_scores = np.asarray(anomaly_scores, dtype=np.float32)[inspected] if len(inspected) else np.asarray([], dtype=np.float32)
    ax2.plot(inspected_scores, marker="o")
    ax2.set_title("Inspected evidence anomaly scores")
    ax2.set_xlabel("inspection step")
    ax2.set_ylabel("score")
    ax2.grid(alpha=0.25)

    if out_path is not None:
        fig.savefig(out_path, dpi=180)
    return fig, voxel_coords



def plot_mri_trajectory_3d_interactive(
    volume_path: str | Path,
    token_coords: np.ndarray,
    grid_shape: np.ndarray | tuple[int, int, int],
    anomaly_scores: np.ndarray,
    visited_indices: np.ndarray,
    inspected_indices: np.ndarray,
    title: str,
    html_path: str | Path | None = None,
    stride: int = 3,
    patch_size: tuple[int, int, int] = (4, 16, 16),
    geometry: dict[str, Any] | None = None,
):
    """Interactive Plotly MRI-space view with rotatable orthogonal slices.

    The anatomy comes from real MRI slices in voxel coordinates. Token centers
    are mapped through NeuroVFM's preprocessing geometry (see `mri_space_coords`).
    """
    import plotly.graph_objects as go

    img = nib.load(str(volume_path))
    volume = np.asanyarray(img.dataobj).astype(np.float32)
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    voxel_coords = mri_space_coords(volume_path, token_coords, grid_shape, patch_size, geometry)

    visited = np.asarray(visited_indices, dtype=int)
    inspected = np.asarray(inspected_indices, dtype=int)
    path = voxel_coords[visited] if len(visited) else np.empty((0, 3), dtype=np.float32)
    inspected_xyz = voxel_coords[inspected] if len(inspected) else np.empty((0, 3), dtype=np.float32)
    inspected_scores = np.asarray(anomaly_scores, dtype=np.float32)[inspected] if len(inspected) else np.asarray([], dtype=np.float32)

    if len(path):
        center = np.clip(np.round(path.mean(axis=0)).astype(int), 0, np.asarray(volume.shape) - 1)
    else:
        center = np.asarray(volume.shape) // 2
    x0, y0, z0 = [int(v) for v in center]
    stride = max(int(stride), 1)

    data = []

    yy = np.arange(0, volume.shape[1], stride)
    zz = np.arange(0, volume.shape[2], stride)
    Y, Z = np.meshgrid(yy, zz, indexing="ij")
    X = np.full_like(Y, x0)
    data.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=_normalize_slice(volume[x0, ::stride, ::stride]),
        colorscale="gray", showscale=False, opacity=0.42,
        hoverinfo="skip", name=f"axis0={x0}",
    ))

    xx = np.arange(0, volume.shape[0], stride)
    zz = np.arange(0, volume.shape[2], stride)
    X, Z = np.meshgrid(xx, zz, indexing="ij")
    Y = np.full_like(X, y0)
    data.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=_normalize_slice(volume[::stride, y0, ::stride]),
        colorscale="gray", showscale=False, opacity=0.34,
        hoverinfo="skip", name=f"axis1={y0}",
    ))

    xx = np.arange(0, volume.shape[0], stride)
    yy = np.arange(0, volume.shape[1], stride)
    X, Y = np.meshgrid(xx, yy, indexing="ij")
    Z = np.full_like(X, z0)
    data.append(go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=_normalize_slice(volume[::stride, ::stride, z0]),
        colorscale="gray", showscale=False, opacity=0.28,
        hoverinfo="skip", name=f"axis2={z0}",
    ))

    if len(path):
        data.append(go.Scatter3d(
            x=path[:, 0], y=path[:, 1], z=path[:, 2],
            mode="lines+markers", name="visited path",
            line=dict(color="red", width=7),
            marker=dict(size=4, color="red"),
            hovertemplate="visited<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>",
        ))

    if len(inspected_xyz):
        data.append(go.Scatter3d(
            x=inspected_xyz[:, 0], y=inspected_xyz[:, 1], z=inspected_xyz[:, 2],
            mode="markers", name="inspected evidence",
            marker=dict(
                size=7,
                color=inspected_scores,
                colorscale="Magma",
                colorbar=dict(title="inspected anomaly"),
                line=dict(color="white", width=2),
            ),
            text=[f"score={s:.3f}" for s in inspected_scores],
            hovertemplate="inspected<br>x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<br>%{text}<extra></extra>",
        ))

    fig = go.Figure(data=data)
    fig.update_layout(
        title=title,
        width=950,
        height=760,
        scene=dict(
            xaxis_title="MRI voxel axis 0",
            yaxis_title="MRI voxel axis 1",
            zaxis_title="MRI voxel axis 2",
            xaxis=dict(range=[0, volume.shape[0]]),
            yaxis=dict(range=[0, volume.shape[1]]),
            zaxis=dict(range=[0, volume.shape[2]]),
            aspectmode="data",
            camera=dict(eye=dict(x=1.45, y=-1.55, z=1.05)),
        ),
        margin=dict(l=0, r=0, b=0, t=45),
        legend=dict(x=0.02, y=0.98),
    )
    if html_path is not None:
        fig.write_html(str(html_path), include_plotlyjs=True, full_html=True)
    return fig, voxel_coords
