"""Weighted smoke test for the 3D-Neuro-SimCLR ResNet encoder.

Run from the repository root:

    PYTHONPATH=src .venv/bin/python tests/test_resnet3d_smoke.py
"""
from __future__ import annotations

import unittest
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from neuro_evidence.resnet3d import load_pretrained_simclr_resnet18


def make_fake_brain_volume(
    shape: Sequence[int] = (96, 112, 96),
    seed: int = 0,
    ventricle_scale: float = 1.0,
    asymmetry: float = 0.0,
) -> torch.Tensor:
    """Create a structured MRI-like phantom, not plain random noise."""
    if len(shape) != 3 or any(int(size) < 32 for size in shape):
        raise ValueError("shape must contain three spatial sizes, each >= 32")
    depth, height, width = (int(size) for size in shape)
    z, y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, depth),
        torch.linspace(-1.0, 1.0, height),
        torch.linspace(-1.0, 1.0, width),
        indexing="ij",
    )

    radius = torch.sqrt((z / 0.94).square() + (y / 0.98).square() + (x / 0.82).square())
    brain = radius <= 1.0
    white_matter = radius <= 0.69
    deep_tissue = ((z / 0.48).square() + (y / 0.50).square() + (x / 0.44).square()) <= 1.0

    volume = torch.zeros_like(radius)
    volume = torch.where(brain, 0.48 + 0.10 * (1.0 - radius).clamp_min(0.0), volume)
    volume = torch.where(white_matter, 0.72 + 0.08 * (1.0 - radius / 0.69), volume)
    volume = torch.where(deep_tissue, volume + 0.035, volume)

    cavity_width = 0.085 * float(ventricle_scale)
    left_cavity = (
        (z / (0.24 * ventricle_scale)).square()
        + ((y + 0.02) / (0.15 * ventricle_scale)).square()
        + ((x - 0.16 - asymmetry) / cavity_width).square()
    ) <= 1.0
    right_cavity = (
        (z / (0.24 * ventricle_scale)).square()
        + ((y + 0.02) / (0.15 * ventricle_scale)).square()
        + ((x + 0.16 - asymmetry) / cavity_width).square()
    ) <= 1.0
    volume = torch.where((left_cavity | right_cavity) & brain, 0.16, volume)

    bias = 0.92 + 0.08 * x - 0.05 * z + 0.03 * torch.sin(3.0 * y)
    texture = 0.018 * torch.sin(14.0 * x) * torch.cos(11.0 * z)
    volume = volume * bias + texture * brain

    generator = torch.Generator().manual_seed(int(seed))
    raw_noise = torch.randn((1, 1, depth, height, width), generator=generator)
    correlated_noise = F.avg_pool3d(raw_noise, kernel_size=5, stride=1, padding=2)[0, 0]
    fine_noise = torch.randn((depth, height, width), generator=generator)
    volume = volume + brain * (0.025 * correlated_noise + 0.008 * fine_noise)
    volume = volume + (~brain) * (0.003 * fine_noise.abs())
    return volume.clamp(0.0, 1.0).float()


def make_fake_brain_batch(
    batch_size: int = 2,
    shape: Sequence[int] = (96, 112, 96),
    seed: int = 7,
) -> torch.Tensor:
    """Return varied phantoms with shape ``[B,1,D,H,W]``."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    volumes = []
    for index in range(batch_size):
        offset = index - (batch_size - 1) / 2
        volumes.append(
            make_fake_brain_volume(
                shape=shape,
                seed=seed + index,
                ventricle_scale=1.0 + 0.12 * offset,
                asymmetry=0.025 * offset,
            )
        )
    return torch.stack(volumes).unsqueeze(1)


class TestWeightedResNet3DSmoke(unittest.TestCase):
    """Load real epoch-150 weights before testing any fake tensor."""

    shape = (96, 112, 96)
    batch_size = 2

    @classmethod
    def setUpClass(cls) -> None:
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cls.model = load_pretrained_simclr_resnet18(device=cls.device)
        cls.fake_mri = make_fake_brain_batch(cls.batch_size, cls.shape).to(cls.device)
        with torch.inference_mode():
            cls.feature_map = cls.model.forward_feature_map(cls.fake_mri)
            cls.embeddings = cls.model(cls.fake_mri)
            cls.repeated = cls.model(cls.fake_mri[:1])

    def test_checkpoint_is_real_epoch_150_weight(self) -> None:
        self.assertEqual(self.model.checkpoint_epoch, 150)
        self.assertTrue(self.model.checkpoint_path.endswith("simclr_3d_brain_foundation.tar"))

    def test_phantom_has_brain_like_structure(self) -> None:
        depth, height, width = self.shape
        center = self.fake_mri[
            :,
            :,
            depth // 4 : 3 * depth // 4,
            height // 4 : 3 * height // 4,
            width // 4 : 3 * width // 4,
        ]
        background = self.fake_mri[:, :, : depth // 8, : height // 8, : width // 8]
        self.assertGreater(center.mean().item(), background.mean().item() + 0.1)
        self.assertGreaterEqual(self.fake_mri.min().item(), 0.0)
        self.assertLessEqual(self.fake_mri.max().item(), 1.0)

    def test_layer3_embedding_contract(self) -> None:
        self.assertEqual(tuple(self.feature_map.shape), (2, 256, 12, 14, 12))
        self.assertEqual(tuple(self.embeddings.shape), (2, 2016, 256))
        self.assertTrue(torch.isfinite(self.embeddings).all().item())
        self.assertGreater(self.embeddings.std().item(), 0.0)

    def test_eval_is_repeatable_and_samples_are_distinct(self) -> None:
        self.assertTrue(torch.allclose(self.embeddings[:1], self.repeated, rtol=1e-5, atol=1e-6))
        self.assertFalse(torch.allclose(self.embeddings[0], self.embeddings[1]))

    @classmethod
    def tearDownClass(cls) -> None:
        pooled = cls.embeddings.mean(dim=1)
        print(f"checkpoint={cls.model.checkpoint_path}")
        print(f"checkpoint_epoch={cls.model.checkpoint_epoch} (strict weighted load: PASS)")
        print(f"fake_mri={tuple(cls.fake_mri.shape)}")
        print(f"layer3_feature_map={tuple(cls.feature_map.shape)}")
        print(f"region_embeddings={tuple(cls.embeddings.shape)}")
        print(f"optional_mean_pooled_global={tuple(pooled.shape)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
