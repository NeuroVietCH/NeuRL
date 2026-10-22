## TRACE-MRI: Student Task Assignment — v2 (22 Oct 2026)

Replaces the split in [`Phân-công.md`](Phân-công.md) from 22 Oct onward. Plan reference: [`TRACE-MRI_CVPR2027_Implementation_Plan_v2.md`](TRACE-MRI_CVPR2027_Implementation_Plan_v2.md).

**Hard deadlines (AoE):** results package to the paper writer **9 Nov** · OpenReview registration **10 Nov** · full paper **16 Nov** · supplementary **23 Nov**.

### 0. Status (22 Oct)

**Done, on OASIS-3 only (pilot, not paper results):** native NeuroVFM token extraction (global and local-window), cross-fitted Transformer reader, PPO pointer agent (STOP, position-only), random-k baseline, sweeps 1–4.

**Sweep 4 (finished, 30/30 runs, 5 folds × 3 seeds, unstripped tokens, budget 8):**

| Tokens | Test AUC | Random (8 tokens) | Gain (sd) | Wins | Bal. acc | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| `rawG` (global) | 0.882 | 0.874 | +0.009 (0.021) | 13/15 | 0.791 | 0.08 |
| `rawL` (local window) | 0.851 | 0.845 | +0.006 (0.026) | 9/15 | 0.768 | 0.36 |

Active selection still adds only ~+0.01 AUC over random, and a blind policy matched the agent in sweep 2. The A5 controls (§4) are therefore the main risk for the paper.

**Not started:** ADNI preprocessing/QC/manifest/splits and token caches (data already on CSCS), AIBL transfer, 3D CNN encoder and baseline, static Top-K, differentiable selector, REINFORCE, GRU reader, cooperative two-agent RL, the Advocate–Challenger game, CIs/ECE/significance tests.

### 1. Team and infrastructure

| Student | Access | Role |
|---|---|---|
| Anh lớn 1 | This server + CSCS | ADNI/AIBL data and NeuroVFM features |
| Anh lớn 2 | This server + CSCS | RL and the evidence game |
| Anh lớn 3 | This server + CSCS | Reader, baselines, evaluation and results |
| Em bé 4 | **CSCS only** (via the GitHub repo) | 3D CNN (ResNet) backbone |

Workflow: prototype and smoke-test on OASIS-3 on this server (24 GB RAM, at most 3 concurrent jobs), push to `NeuroVietCH/NeuRL`, run all ADNI/AIBL experiments on CSCS. **Every ADNI/AIBL number in the paper comes from CSCS.**

### 2. Main responsibilities

| Student | Assigned tasks | Due | Deliverables |
|---|---|---|---|
| Anh lớn 1 | 1. ADNI1/2/3 manifest (CN vs AD only), cross-phase subject mapping, phase-aware splits: P1 = held-out ADNI test, P2 = train ADNI1/2 → test ADNI3. No subject in more than one partition. | 24 Oct | Manifest, split files, `dataset_audit.md`, G2 memo, NeuroVFM ADNI/AIBL caches, full-volume NeuroVFM baseline, A10 and `g_free` caches, runtime/storage stats |
| | 2. **G2 memo:** global vs local tokens, based on sweep 4. Write it in `stage2_notes.md`. | 23 Oct | |
| | 3. NeuroVFM preprocessing (strict, unstripped, as in sweep 4) and QC on ADNI on CSCS. | 26 Oct | |
| | 4. ADNI NeuroVFM token caches with `extract_tokens.py`, using the token mode from the G2 memo. **Critical path.** | 27 Oct | |
| | 5. AIBL transfer, preprocessing and token cache. AIBL is never used for training or tuning. | 31 Oct | |
| | 6. Give Em bé 4 the subject list, splits and preprocessed volumes; review Em bé 4's code and cache format. | ongoing | |
| | 7. Full-volume NeuroVFM baseline (all-token probe) on ADNI. | 30 Oct | |
| | 8. Caches for ablations A10 (local tokens, separate cache) and `g_free` (no global context). | 3 Nov | |
| | 9. Runtime and storage statistics; validated data and encoder artifacts. | 9 Nov | |
| Anh lớn 2 | 1. Decide **REINFORCE vs PPO**: implement REINFORCE, or write down why PPO stays. | 24 Oct | Single, cooperative and game policies; checkpoints; trajectory logs; game ablation tables |
| | 2. Validation-calibrated confidence stopping, alongside the learned STOP. | 27 Oct | |
| | 3. Single agent (A1) on ADNI with A5a/A5b controls, budgets 2/4/6/8. Verify and document exactly which inputs each control receives. | 27 Oct (G3) | |
| | 4. Cooperative two-agent RL (A2): two simultaneous proposals per round, three rounds. | 29 Oct | |
| | 5. **Advocate–Challenger game (A0):** conditional singleton evidence scores, role-specific payoffs, alternating updates. | 2 Nov (G4) | |
| | 6. Game ablations in the priority order of §4: A5a/A5b on the game, A8, A3, A7, A6. | 6 Nov | |
| | 7. Repeat A0/A1/A2 on CNN features once Em bé 4's cache passes checks (A9). | 6 Nov | |
| | 8. Final checkpoints, game ablation tables, trajectory logs. | 9 Nov | |
| Anh lớn 3 | 1. Decide **GRU vs Transformer reader**: switch, or record why the Transformer stays. | 24 Oct | Frozen readers, non-RL baselines, evaluation tooling, all tables and figures, reproducibility checks |
| | 2. Train 5 cross-fitted readers per backbone/token mode on ADNI on CSCS (NeuroVFM first; CNN when its cache is ready; separate readers for A10 and `g_free`). | 28 Oct | |
| | 3. **Static Top-K (A4)** and **differentiable sequential attention** baselines (develop on OASIS now). | 27 Oct | |
| | 4. Random-k curves on ADNI (≥ 10 draws); subject-leakage and cache-integrity checks. | 28 Oct | |
| | 5. Evaluation tooling: bootstrap CIs, paired bootstrap differences, ECE, one results registry, per-subject prediction CSV in the plan's format. | 31 Oct | |
| | 6. P1, P2 cross-phase and AIBL external evaluation, 3 seeds for every main method. | 6 Nov | |
| | 7. Tables 1–4 (incl. the paired A0 vs A2 table and the A9 encoder table), budget curves, trajectory figures, confusion matrices. | 9 Nov | |
| | 8. Reproducibility: rerun selected runs, archive configs and checkpoints. | 9 Nov | |
| Em bé 4 | 1. ResNet-18 3D encoder exposing its spatial feature map; unit-test output shapes. | 25 Oct | ResNet encoder, full-volume CNN baseline, CNN feature caches, cache-check script, runtime profile |
| | 2. Full-volume CNN baseline for CN vs AD on Anh lớn 1's ADNI splits, 3 seeds. | 29 Oct | |
| | 3. CNN spatial-feature cache for ADNI and AIBL. **Use the same record format as the NeuroVFM cache** (`src/neuro_evidence/extract_tokens.py`, `to_cache`): `embeddings` (fp16, N×D), `coords` (int16), `token_count`, `embedding_dim`, `grid_shape`, `scan_path`, `metadata`. Then the environment and reader run unchanged. | 30 Oct | |
| | 4. Cache sanity script: NaN/Inf, wrong shapes, missing and duplicate subjects. Run it on both backbones' caches. | 31 Oct | |
| | 5. CNN runtime and memory profile. | 6 Nov | |

### 3. Deadlines and milestones

| Deadline | Anh lớn 1 | Anh lớn 2 | Anh lớn 3 | Em bé 4 |
|---|---|---|---|---|
| 24 Oct | ADNI manifest and splits; G2 memo | REINFORCE vs PPO decision | GRU vs Transformer decision | — |
| 27 Oct (G3) | ADNI NeuroVFM token cache | Single agent + A5a/A5b on ADNI | Static Top-K, differentiable baseline | ResNet encoder (25 Oct) |
| 31 Oct | AIBL cache; full-volume NeuroVFM baseline | Cooperative two-agent (29 Oct) | Readers trained; eval tooling | CNN baseline, CNN cache, cache checks |
| 2 Nov (G4) | A10 / `g_free` caches (3 Nov) | **A0 game vs A2 cooperative** | Matched A0 vs A2 comparison | — |
| 6 Nov (G5) | Runtime/storage stats | Game ablations done; CNN runs | P1/P2/AIBL eval, 3 seeds, stats | CNN runtime profile |
| 9 Nov | Validated data and encoder artifacts | Final checkpoints, ablation tables | All tables, figures, per-subject predictions | — |
| 10–16 Nov | Reproduce or correct results as requested | same | Final paper-result verification | same |

After G5 (6 Nov) the main method is frozen: no changes to the method, only reproduction, cross-cohort tests, figures and statistics.

### 4. Ablations (plan §11)

All comparisons within an encoder/token-mode setting use the same split, cache, frozen reader ensemble, proposal budget and evaluation rule. Every ablation table reports dataset, backbone, token mode, budget, AUROC, balanced accuracy and unique observed tokens. **The central game result is the paired A0 vs A2 table.**

| Priority | ID | Ablation | Exact change | Owner | OASIS pilot status |
|---|---|---|---|---|---|
| 1 | A0 | Full TRACE-MRI | Two-policy conditional evidence game | Anh lớn 2 | Not started |
| 1 | A2 | Cooperative game | `alpha = 0`; keep two policies and collision term | Anh lớn 2 | Not started |
| 1 | A1 | Single agent | One policy proposes both tokens; common reward | Anh lớn 2 | Done (PPO, sweeps 1–4) |
| 1 | A4 | Static selection | Rank tokens once from context and coordinates | Anh lớn 3 | Not started |
| 2 | A5a | Evidence-blind policy | Policy loses evidence-dependent state; reader unchanged | Anh lớn 2 | Done for single agent (sweep 2); inputs to be verified |
| 2 | A5b | Geometry-only policy | Also remove scan-specific diagnostic context; keep coordinates and mask | Anh lớn 2 | ≈ `content: false` (sweep 3); inputs to be verified |
| 2 | A9 | Encoder | CNN vs NeuroVFM, same method | Anh lớn 3 (Em bé 4, Anh lớn 2) | Blocked on CNN cache |
| 3 | A8 | Reward coefficient | `alpha` = 0, .05, .10, .20 | Anh lớn 2 | Not started |
| 3 | A3 | No collision term | `collision_weight = 0` | Anh lớn 2 | Not started |
| 3 | A7 | Role direction | Swap support/challenge signs, or same sign for both | Anh lớn 2 | Not started |
| 4 (suppl.) | A6 | Fixed vs adaptive budget | Fixed budgets vs validation-calibrated stopping | Anh lớn 2 | Learned STOP tested (sweep 2) |
| 4 (suppl.) | A10 | Context vs local tokens | Separate caches and readers | Anh lớn 1 + Anh lớn 3 | rawG vs rawL (sweep 4) |
| 4 (suppl.) | `g_free` | No global context | Learned constant $h_0$; $g$ removed from reader and policy; separate reader | Anh lớn 1 + Anh lớn 3 | Not started |
| 4 (suppl.) | — | Guarded joint reader updates | Report only as an ablation | Anh lớn 2 | Done (sweeps 2–3) |
| 4 (suppl.) | — | 64-cell grid vs native tokens | Optional granularity ablation | Anh lớn 1 | Not started |

Rules:
- A5a and A5b run on **both** the single agent and the game (A0). They are the main risk: on OASIS the blind policy matched the full agent.
- A control that still receives scan-dependent global context or coordinates is not blind. Before reusing the sweep 2 / sweep 3 controls as A5a/A5b, document exactly which inputs each policy received.
- Keep the reader unchanged for A5a and A5b.
- **Open question for the prof:** the plan lists a "memory" game ablation (§ checklist) with no row in §11. Confirm what it should be.

Compute estimate: A0–A3, A5a/b, A7 and A8 on both backbones with 3 seeds ≈ 54 policy training runs before budget sweeps — CSCS only. If time runs short, finish priorities 1–2 for the main paper and move priority 4 to the supplementary.

### 5. Coordination and dependencies

- **Anh lớn 1 → everyone.** The ADNI NeuroVFM cache (27 Oct) blocks every ADNI result. If it slips past 28 Oct, G4 and G5 are at risk. Until then, Anh lớn 2 and Anh lớn 3 keep developing on OASIS.
- **Anh lớn 3 → Anh lớn 2.** Readers are trained and frozen before any ADNI policy training.
- **Anh lớn 2 → Anh lớn 3.** RL code, checkpoints, eval configs and trajectory logs; Anh lớn 3 owns the evaluation protocol and tables.
- **Em bé 4 → Anh lớn 2, Anh lớn 3.** CNN cache in the shared format. Anh lớn 1 reviews Em bé 4's code. **Fallback:** if the CNN cache is not ready by 31 Oct, the main paper is NeuroVFM-only and A9 moves to the supplementary. No critical-path item depends only on Em bé 4.
- **Decisions this week** (token mode, REINFORCE vs PPO, GRU vs Transformer) are written up with a reason; the plan requires methodological changes to be reported.

### 6. Working rules

All four students use one repository and a shared CSCS experiment directory. MRI transfers use the approved storage location and comply with ADNI/AIBL data-access agreements.

Every experiment saves its configuration, random seed, subject split, checkpoint, predictions and evaluation metrics, and is entered in the single results registry. Each method is evaluated at budgets 2, 4, 6 and 8. Final results need three independent seeds, subject-level separation, and AIBL evaluation without retraining or tuning on AIBL.

The paper is written by the supervisor. Send the results package by **9 November**: final tables, figures, experimental settings and concise technical descriptions of what was implemented. Report any unexpected result, suspected data leakage or methodological change immediately, with the affected configuration and run IDs.
