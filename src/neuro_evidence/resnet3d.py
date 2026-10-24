"""Weighted 3D ResNet-18 encoder for layer3 region embeddings."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints/3d_neuro_simclr/simclr_3d_brain_foundation.tar"


class SimCLRResNet18Encoder(nn.Module):
    """Return layer3 spatial embeddings from a weighted 3D ResNet-18."""

    embedding_dim = 256

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def _as_batched_volume(x: torch.Tensor) -> torch.Tensor:
        """Normalize a 3D MRI tensor to ``[B, 1, D, H, W]``."""
        if x.ndim == 3:  # [D, H, W]
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:  # [B, D, H, W]
            x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ValueError(
                "Expected [D,H,W], [B,D,H,W], or [B,1,D,H,W], "
                f"got {tuple(x.shape)}"
            )
        if x.shape[1] != 1:
            raise ValueError(f"Expected one MRI channel, got shape {tuple(x.shape)}")
        if not x.is_floating_point():
            x = x.float()
        return x

    def forward_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Return the layer3 map ``[B, 256, d, h, w]``."""
        x = self._as_batched_volume(x)
        model = self.backbone

        # MONAI renamed ``relu`` to ``act`` in newer releases.
        activation = getattr(model, "relu", None) or model.act
        x = activation(model.bn1(model.conv1(x)))
        if not model.no_max_pool:
            x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        return model.layer3(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return region embeddings ``[B, num_regions, 256]``."""
        feature_map = self.forward_feature_map(x)
        return feature_map.flatten(start_dim=2).transpose(1, 2).contiguous()


def _encoder_state_dict(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Select pretrained ``encoder.*`` weights and discard the projector."""
    state = checkpoint.get("model_state_dict", checkpoint)
    weights: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key = key.removeprefix("module.")
        if key.startswith("encoder."):
            weights[key.removeprefix("encoder.")] = value
    if "conv1.weight" not in weights:
        raise ValueError("Checkpoint does not contain SimCLR encoder weights")
    return weights


def load_pretrained_simclr_resnet18(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device: str | torch.device = "cpu",
) -> SimCLRResNet18Encoder:
    """Build the matching backbone and strictly load the pretrained weights."""
    from monai.networks.nets import resnet18

    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ResNet checkpoint not found: {checkpoint_path}")

    backbone = resnet18(
        pretrained=False,
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        shortcut_type="B",
        bias_downsample=True,
        no_max_pool=False,
        feed_forward=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    weights = _encoder_state_dict(checkpoint)
    backbone.load_state_dict(weights, strict=True)

    model = SimCLRResNet18Encoder(backbone).to(device).eval()
    model.checkpoint_epoch = checkpoint.get("epoch")
    model.checkpoint_path = str(checkpoint_path.resolve())
    return model
