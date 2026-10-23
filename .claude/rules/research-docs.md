---
paths:
  - "docs/**/*.md"
---

# Research-document rules

- Treat architecture and task-assignment documents as design context unless
  current source/tests verify the described behavior.
- Use explicit labels such as `implemented`, `planned`, `measured`, and
  `unverified` when these states could be confused.
- Do not rewrite historical measurements in `stage2_notes.md`; append a dated
  correction or mark an older item superseded.
- Do not turn illustrative tensor shapes, example commands or target directory
  trees into hard constraints without checking the implementation.
- Keep preprocessing, checkpoint identity, native feature width, token count,
  coordinate semantics, local/global context and overlap risk explicit in
  encoder documentation.
- Architecture is a reference, not a mandate to expand the scope of a focused
  implementation task.
