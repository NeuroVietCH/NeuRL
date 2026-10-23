---
paths:
  - "src/neuro_evidence/resnet3d.py"
  - "tests/test_resnet3d_smoke.py"
  - "docs/encoder.md"
---

# ResNet3D rules

- Keep the production module focused on model construction, strict checkpoint
  loading and tensor-to-embedding inference. Test-only phantoms belong in
  `tests/`.
- The default checkpoint is
  `checkpoints/3d_neuro_simclr/simclr_3d_brain_foundation.tar`. It is a PyTorch
  ZIP checkpoint despite the `.tar` suffix and is intentionally gitignored.
- Load only `encoder.*` weights, exclude the SimCLR projection head and require
  `strict=True`. Never silently continue with random weights.
- The current forward contract is `[B,1,D,H,W] -> [B,N,256]`, obtained by
  flattening the layer-3 map. Derive `N` from the actual map; do not hardcode it.
- Layer-3 cells are spatial convolutional features with overlapping receptive
  fields. They do not satisfy a strict independent-local-token claim.
- A smoke test must use the real checkpoint first and then check shape,
  finiteness, non-constant output, repeatability and distinct outputs for
  distinct structured phantoms.
- Update `docs/encoder.md` when the checkpoint, architecture, preprocessing,
  output contract or test command changes.
