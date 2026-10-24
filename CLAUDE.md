# NeuRL working guide

Use this file as the short entry point for work in this repository. More focused
rules live in `.claude/rules/`.

## Sources of truth

1. Executable source, config, tests and produced artifacts describe what works
   now.
2. `docs/stage2_notes.md` contains historical decisions and measured pilot
   results; preserve its chronology.
3. `docs/architecture_v3.md` and `docs/Phân-công_v3.md` are valuable design and
   planning references, but they are not proof that every described interface,
   command or experiment is implemented. Do not force a local task to implement
   the whole planned architecture unless the request explicitly asks for it.
4. When plan and code differ, state the difference as `planned` versus
   `implemented`; do not silently claim compliance.

## Repository conventions

- Keep encoder-specific preprocessing and native output shapes explicit.
- Do not call a spatial token anatomically local or independent without a
  receptive-field/context-mixing check.
- Preserve subject-level split boundaries and never use test data for tuning.
- Large weights and data stay under ignored paths such as `checkpoints/` or an
  explicit shared-storage path; never commit them.
- Preserve user changes and unrelated files. Avoid broad refactors for a narrow
  task.
- Report tests as PASS/FAIL/BLOCKED with the exact command and key output.

## Current ResNet3D entry points

- Encoder: `src/neuro_evidence/resnet3d.py`
- Weighted tensor smoke test: `tests/test_resnet3d_smoke.py`
- Encoder comparison and usage: `docs/encoder.md`

Run the weighted smoke test from the repository root:

```bash
PYTHONPATH=src .venv/bin/python tests/test_resnet3d_smoke.py
```

The smoke test must load the real epoch-150 checkpoint successfully before
running inference. A missing/incompatible checkpoint is a failure, not a reason
to fall back to random weights.
