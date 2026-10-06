from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .data import repo_root, resolve_path


@dataclass(frozen=True, slots=True)
class TokenBatch:
    sample_id: str
    scan_path: str
    embeddings: np.ndarray
    coords: np.ndarray
    input_shape: tuple[int, int]
    volume_size_dhw: tuple[int, int, int]
    patch_size: tuple[int, int, int]
    remove_background: bool
    metadata: dict[str, Any]

    @property
    def dense_grid_shape(self) -> tuple[int, int, int]:
        return tuple(int(v // p) for v, p in zip(self.volume_size_dhw, self.patch_size, strict=True))

    @property
    def dense_token_count(self) -> int:
        d, h, w = self.dense_grid_shape
        return int(d * h * w)

    @property
    def removed_background_count(self) -> int:
        return max(0, self.dense_token_count - int(self.coords.shape[0]))


def sample_id_from_path(path: str | Path) -> str:
    p = Path(path)
    stem = p.name.replace(".nii.gz", "").replace(".nii", "")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    digest = hashlib.sha1(str(p).encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{digest}"



def _install_flash_attn_import_stub() -> None:
    """Let official NeuroVFM import on systems without flash-attn.

    The official inference path used here passes use_flash_attn=False. These
    fallbacks use regular PyTorch layers for import/inference compatibility.
    """
    try:
        import flash_attn  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    import types
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def _missing_flash_attn(*_args, **_kwargs):
        raise RuntimeError(
            "flash_attn is not installed, and this NeuroVFM code path tried to use flash attention. "
            "Install flash-attn or keep inference on a use_flash_attn=False path."
        )

    class FusedDense(nn.Linear):
        def __init__(self, in_features, out_features, bias=True, return_residual=False, **kwargs):
            self.return_residual = return_residual
            device = kwargs.get("device", None)
            dtype = kwargs.get("dtype", None)
            super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)

        def forward(self, x):
            out = super().forward(x)
            return (out, x) if self.return_residual else out

    class FusedMLP(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, activation=F.gelu, return_residual=False, **_kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features * 4
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.activation = activation
            self.return_residual = return_residual

        def forward(self, x):
            out = self.fc2(self.activation(self.fc1(x)))
            return (out, x) if self.return_residual else out

    class RMSNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-6, **_kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = None
            self.eps = eps

        def forward(self, x):
            var = x.pow(2).mean(dim=-1, keepdim=True)
            return x * torch.rsqrt(var + self.eps) * self.weight

    def layer_norm_fn(
        x,
        weight,
        bias,
        residual=None,
        eps=1e-6,
        dropout_p=0.0,
        rowscale=None,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
        **_kwargs,
    ):
        if residual is not None:
            x = x + residual
        residual_out = x.float() if residual_in_fp32 else x
        if is_rms_norm:
            var = x.pow(2).mean(dim=-1, keepdim=True)
            out = x * torch.rsqrt(var + eps) * weight
            if bias is not None:
                out = out + bias
        else:
            out = F.layer_norm(x, x.shape[-1:], weight, bias, eps)
        if dropout_p and dropout_p > 0:
            out = F.dropout(out, p=dropout_p, training=True)
        if rowscale is not None:
            out = out * rowscale.unsqueeze(-1)
        return (out, residual_out) if prenorm else out

    flash = types.ModuleType("flash_attn")
    flash.__path__ = []
    flash.flash_attn_qkvpacked_func = _missing_flash_attn
    flash.flash_attn_varlen_qkvpacked_func = _missing_flash_attn
    flash.flash_attn_varlen_kvpacked_func = _missing_flash_attn

    layers = types.ModuleType("flash_attn.layers")
    layers.__path__ = []
    layers_patch = types.ModuleType("flash_attn.layers.patch_embed")
    layers_patch.PatchEmbed = nn.Identity

    ops = types.ModuleType("flash_attn.ops")
    ops.__path__ = []
    ops_fused = types.ModuleType("flash_attn.ops.fused_dense")
    ops_fused.FusedDense = FusedDense
    ops_fused.ColumnParallelLinear = FusedDense
    ops_fused.RowParallelLinear = FusedDense

    ops_triton = types.ModuleType("flash_attn.ops.triton")
    ops_triton.__path__ = []
    ops_ln = types.ModuleType("flash_attn.ops.triton.layer_norm")
    ops_ln.layer_norm_fn = layer_norm_fn
    ops_ln.RMSNorm = RMSNorm

    modules = types.ModuleType("flash_attn.modules")
    modules.__path__ = []
    modules_mlp = types.ModuleType("flash_attn.modules.mlp")
    modules_mlp.FusedMLP = FusedMLP

    sys.modules.update({
        "flash_attn": flash,
        "flash_attn.layers": layers,
        "flash_attn.layers.patch_embed": layers_patch,
        "flash_attn.ops": ops,
        "flash_attn.ops.fused_dense": ops_fused,
        "flash_attn.ops.triton": ops_triton,
        "flash_attn.ops.triton.layer_norm": ops_ln,
        "flash_attn.modules": modules,
        "flash_attn.modules.mlp": modules_mlp,
    })



def _install_outlines_import_stub() -> None:
    """Provide import-time VLM-only outlines symbols not needed by encoder."""
    import types

    try:
        from outlines.processors.structured import JSONLogitsProcessor  # noqa: F401
        return
    except Exception:
        pass

    def _unused(*_args, **_kwargs):
        raise RuntimeError("This minimal NeuRL path does not use NeuroVFM VLM structured decoding.")

    processors = sys.modules.get("outlines.processors") or types.ModuleType("outlines.processors")
    processors.__path__ = getattr(processors, "__path__", [])
    structured = types.ModuleType("outlines.processors.structured")
    structured.JSONLogitsProcessor = _unused
    sys.modules["outlines.processors"] = processors
    sys.modules["outlines.processors.structured"] = structured


def _install_vlm_import_stubs() -> None:
    """Provide import-time VLM-only symbols not used by encoder extraction."""
    import enum
    import types

    if "peft" not in sys.modules:
        peft = types.ModuleType("peft")

        class LoraConfig:
            def __init__(self, *_args, **_kwargs):
                pass

        class TaskType(enum.Enum):
            CAUSAL_LM = "CAUSAL_LM"

        def get_peft_model(model, *_args, **_kwargs):
            return model

        peft.LoraConfig = LoraConfig
        peft.TaskType = TaskType
        peft.get_peft_model = get_peft_model
        sys.modules["peft"] = peft

    if "transformers" not in sys.modules:
        transformers = types.ModuleType("transformers")

        class _Dummy:
            def __init__(self, *_args, **_kwargs):
                pass

            @classmethod
            def from_pretrained(cls, *_args, **_kwargs):
                raise RuntimeError("NeuRL encoder extraction does not use transformer LLM loading.")

        class GenerationConfig(_Dummy):
            pass

        class PreTrainedModel(_Dummy):
            pass

        class PreTrainedTokenizer(_Dummy):
            pass

        transformers.AutoModelForCausalLM = _Dummy
        transformers.AutoTokenizer = _Dummy
        transformers.GenerationConfig = GenerationConfig
        transformers.PreTrainedModel = PreTrainedModel
        transformers.PreTrainedTokenizer = PreTrainedTokenizer
        sys.modules["transformers"] = transformers

    if "torch_scatter" not in sys.modules:
        torch_scatter = types.ModuleType("torch_scatter")

        def segment_csr(*_args, **_kwargs):
            raise RuntimeError("NeuRL encoder extraction does not use torch_scatter MIL pooling.")

        torch_scatter.segment_csr = segment_csr
        sys.modules["torch_scatter"] = torch_scatter

def _ensure_importable(neurovfm_repo: str | Path) -> None:
    path = resolve_path(neurovfm_repo, base=repo_root()).resolve()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _import_neurovfm(neurovfm_repo: str | Path) -> None:
    _ensure_importable(neurovfm_repo)
    _install_flash_attn_import_stub()
    _install_outlines_import_stub()
    _install_vlm_import_stubs()


def image_geometry(img: Any) -> dict[str, Any]:
    """Physical geometry of a NeuroVFM-preprocessed SimpleITK image.

    `load_image` reorients, resamples and crops but keeps the image in physical
    space, so origin/spacing/direction fully describe where each voxel sits.
    `view` is how `prepare_for_inference` transposed the (z, y, x) array to
    (D, H, W); it uses the same slice-axis rule as NeuroVFM.
    """
    spacing = tuple(float(v) for v in img.GetSpacing())
    z_dim = 2 if len(set(spacing)) == 1 else int(np.argmax(spacing))
    return {
        "size_xyz": [int(v) for v in img.GetSize()],
        "origin": [float(v) for v in img.GetOrigin()],
        "spacing": list(spacing),
        "direction": [float(v) for v in img.GetDirection()],
        "view": 2 - z_dim,
    }


def neurovfm_geometry(scan_path: str | Path, neurovfm_repo: str | Path = "ext/neurovfm") -> dict[str, Any]:
    """Run NeuroVFM's own `load_image` on a scan and return its preprocessed geometry."""
    _import_neurovfm(neurovfm_repo)
    from neurovfm.data.io import load_image

    img = load_image(str(resolve_path(scan_path, base=repo_root())), preprocess=True)
    if img is None:
        raise ValueError(f"NeuroVFM could not load {scan_path}")
    return image_geometry(img)


class NeuroVFMEncoder:
    """Small adapter around the official NeuroVFM encoder/preprocessor API."""

    name = "neurovfm"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        nf = config["neurovfm"]
        _import_neurovfm(nf["repo"])
        if tuple(float(v) for v in nf.get("target_spacing", (1.0, 1.0, 4.0))) != (1.0, 1.0, 4.0):
            # NeuroVFM's load_image hardcodes 1x1x4mm; StudyPreprocessor stores but ignores target_spacing.
            raise ValueError("neurovfm.target_spacing must be [1.0, 1.0, 4.0]; NeuroVFM does not support other spacings")
        from neurovfm.pipelines import load_encoder
        from neurovfm.pipelines.preprocessor import StudyPreprocessor

        checkpoint = resolve_path(nf["checkpoint"], base=repo_root())
        self.encoder, _ = load_encoder(str(checkpoint), device=nf.get("device"))
        self.preprocessor = StudyPreprocessor(
            patch_size=tuple(nf.get("patch_size", (4, 16, 16))),
            target_spacing=tuple(nf.get("target_spacing", (1.0, 1.0, 4.0))),
            remove_background=bool(nf.get("remove_background", True)),
        )

    def extract(self, scan_path: str | Path) -> TokenBatch:
        import torch

        nf = self.config["neurovfm"]
        scan_path = resolve_path(scan_path, base=repo_root())
        batch = self.preprocessor.load_study([scan_path], modality=nf.get("modality", "mri"))
        with torch.inference_mode():
            embeddings = self.encoder.embed(batch, use_amp=bool(nf.get("use_amp", True))).detach().float().cpu().numpy()
        coords = batch["coords"].detach().cpu().numpy().astype(np.int64)
        metadata = {
            "paths": [str(p) for p in batch["path"]],
            "modes": [str(m) for m in batch["mode"]],
            "encoder": self.name,
            # Needed to map token coords back to the scan; load_study does not expose it.
            "geometry": neurovfm_geometry(scan_path, nf["repo"]),
        }
        return TokenBatch(
            sample_id=sample_id_from_path(scan_path),
            scan_path=str(scan_path),
            embeddings=embeddings.astype(np.float32),
            coords=coords,
            input_shape=tuple(int(v) for v in batch["img"].shape),
            volume_size_dhw=tuple(int(v) for v in batch["size"][0]),
            patch_size=tuple(int(v) for v in nf.get("patch_size", (4, 16, 16))),
            remove_background=bool(nf.get("remove_background", True)),
            metadata=metadata,
        )


def save_token_batch(batch: TokenBatch, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = asdict(batch)
    metadata.pop("embeddings")
    metadata.pop("coords")
    np.savez_compressed(
        path,
        embeddings=batch.embeddings.astype(np.float32),
        coords=batch.coords.astype(np.int64),
        metadata=json.dumps(metadata),
    )
    return path


def load_token_batch(path: str | Path) -> TokenBatch:
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))
    return TokenBatch(embeddings=data["embeddings"], coords=data["coords"], **metadata)
