# Stage 2: active diagnostic agent — design notes

Running log of decisions, open questions and measurements for the active agent (`src/env`, `src/agent`, `src/training`, `src/evaluation`). Add to it as things change; keep old entries and mark them superseded instead of deleting them.

## Current state (2026-10-12)

```text
cached OASIS-3 NeuroVFM tokens (skull-stripped stand-in data)
 → BrainPatchEnv: 6 one-cell moves / inspect / diagnose CN / diagnose AD
 → obs: current token embedding + position + 3 memory values
 → MLP actor-critic, PPO (src/training/train_agent.py: subject-grouped train/val, fold test split)
 → prediction from the diagnose action, or `timeout_prediction` on timeout
```

### Pointer agent (implemented 2026-10-13)

The design in §1–§4 below, next to the older one-cell-move agent (which still works via `train_agent.py`).

| File | What |
|---|---|
| `configs/pointer_agent.yaml` | All settings: budget, look cost, readers, policy, PPO, joint stage |
| `src/training/train_pointer_agent.py` | Entry point: data → cross-fitting groups → readers → PPO → test. `--joint` = stage B, `--readers` reuses pretrained readers, `--budget` |
| `src/env/token_bank.py` | Every scan's tokens in flat GPU tensors; positions normalised per scan to [-1, 1] (no brain-size info) |
| `src/env/pointer_env.py` | Batched episodes: pick a token or STOP; reward = reader loss drop − look cost |
| `src/models/set_encoder.py` | Shared set encoder (LayerNorm → d, Fourier position encoding, 2-layer attention, summary token) |
| `src/models/evidence_reader.py` | Reader + K-reader ensemble; training scans scored by their out-of-group reader, val/test by the ensemble |
| `src/agent/pointer_policy.py` | Pointer actor-critic: position-only candidate keys, STOP logit, value |
| `src/training/reader_training.py` | Random-subset pretraining (balanced, early stop on val), joint updates (picks + random mix), random-looks report |
| `src/training/pointer_ppo.py` | Batched rollouts, GAE (γ = 1), minibatch PPO; observations rebuilt from picked indices |
| `src/evaluation/pointer_eval.py` | Greedy eval vs random looks with the same readers and the same number of looks per scan (`auc_gain`) |
| `src/training/common.py` | Shared helpers (logging, CSV, subject splits, cross-fitting groups) |

Checks done (synthetic data):
- The return equals the reader's loss drop minus look costs.
- No token is picked twice.
- Replayed log-probs match the rollout (error ~5e-7).
- Reader routing by group is correct.
- On a task where 5 fixed-position tokens carry the label, the agent learns to pick them: 100% of picks by update 100, val AUC 1.000 vs 0.50–0.58 for random looks. It then learns to stop after 1 look.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-10-12 | Removed the centroid-distance anomaly score (commit `ec84817`) | It mostly encoded token region, barely separated CN/MCI, and its inspect bonus could be farmed by re-inspecting one token. |
| 2026-10-12 | OASIS-3 is the only dataset for stage 2. ADNI-DOD dropped (`load_adnidod_samples` removed) | ADNI-DOD (8 scans, CN vs MCI) was only for testing the token pipeline. |
| 2026-10-12 | Use the skull-stripped OASIS tokens (`shared_n4_hdbet_crop`) as stand-in data | The original OASIS images are not on the shared volume yet. Results are for building the pipeline only and are not comparable to strict tokens. |
| 2026-10-12 | No separate "is there signal?" probe stage | A separate repo already predicts CN/AD from these same OASIS embeddings. |
| 2026-10-12 | Adopt: classifier over inspected tokens makes the diagnosis; policy only chooses where to look and when to stop (§1) | Gives a dense reward for useful evidence and separates "where" from "what". |
| 2026-10-12 | Adopt: attention memory over the inspected-token set, not a GRU (§2) | Keeps every inspected token exactly, knows where it has been, and needs no hidden state in PPO. |
| 2026-10-12 | Adopt: option A, pick any token directly (pointer), merging move and inspect (§3) | Any token is one step away; no dead moves. One-cell moves fail ~1/3 of the time on OASIS. |
| 2026-10-12 | Train a new classifier inside this repo | Unknown whether the other repo's model accepts token subsets; treat it as no until checked (open question Q1). |

## Planned design

### 1. Evidence classifier + information-gain reward

- Evidence `E_t` = set of inspected tokens `(embedding, position)`, size 0..`max_steps`.
- Classifier `f(E_t) → P(AD)`:
  - LayerNorm, then project 768 → 256, plus a position encoding;
  - 2-layer attention over the set with a learned query, then a logit;
  - an empty set gives the class prior.
- Reward at step t: `loss(E_{t-1}) − loss(E_t) − look_cost`, with `loss` = cross-entropy of the true label, probabilities clamped to [0.01, 0.99].
  - With γ = 1 the episode return is `loss(E_0) − loss(E_T) − cost·T`. In expectation, a look's gain is the information it carries about the label.
- STOP action ends the episode. The diagnosis is `f(E_T)`. Timeout is a forced STOP.

Review findings and how the plan handles them:

1. **Classifier overfitting turns the reward into memorisation.** Train has 1,350 scans with 94 AD. A classifier trained on those scans can memorise them, so on train episodes the reward would point at memorised tokens that don't transfer.
   - Fix: cross-fitting. Train K=5 classifiers on subject-grouped inner folds of train. A scan's reward comes from the classifier that never saw it.
   - At val/test, the diagnosis is the ensemble mean. This is cheap: small models on cached embeddings.
2. **Non-stationary reward if the classifier trains during RL.** Fix: staged training.
   - (a) pretrain the classifiers on random token subsets of size 1..`max_steps`, early-stopped on val;
   - (b) freeze them and run RL;
   - (c) optional: fine-tune the classifiers on the subsets the policy picks (the policy picks non-random subsets, so the distribution shifts), then re-evaluate.
3. **Class imbalance skews the reward.** With a 7% AD prior, AD episodes start at a high loss and CN at a low one, so AD episodes would dominate the return.
   - Fix: train the classifiers with balanced sampling or class weights, so the prior is about 0.5 and `loss(E_0)` ≈ log 2 for both classes.
   - Keep balanced episode sampling in `train_agent.py`.
4. **Discounting breaks the telescoping.** Use γ = 1 (episodes are ≤ 16 steps), or accept a small bias.
5. **Reward scale.** Per-look gains are in nats (roughly 0–0.3). `look_cost` must be on that scale (start at 0.01) or the agent stops immediately. Clamping keeps a single confidently-wrong token from producing huge swings.
6. **Negative rewards are correct.** A token that pushes the classifier toward the wrong label gets a negative reward. In expectation the policy still learns label-agnostic informative locations, since it never sees the label.
7. **Brain size leaks a diagnosis.** Atrophy shrinks the brain, so the token count and extent alone predict AD.
   - Positions are normalised to the brain extent, and no global size features are given to the classifier, so the agent has to look.
   - The candidate set itself still reveals the brain's shape to the policy. That is a weak cue, but test it: compare against a size-only baseline.

### 2. Attention memory

- Each inspected token → `LN(proj(embedding)) + pos_enc(position)`. Padded to `max_steps` with a mask and recomputed fully each step (≤ 16 items, no cache).
- A learned query attends over the memory. Its output feeds the value head, the STOP logit and the pointer query `q_t`.
- The same encoder design as the classifier. Whether to share weights is decided in implementation; separate is safer for cross-fitting.

### 3. Pointer action space (option A)

- Action at step t: one of the scan's N tokens (~240 on OASIS) or STOP. Picking token j means inspecting it.
- Score: `score_j = q_t · k_j`, with `k_j = MLP(pos_enc(position_j))`, **position only**. Already-inspected tokens get −∞. Softmax over the remaining tokens plus a STOP logit.
- `max_steps` is now a budget of looks (e.g. 8–16 of ~240 tokens).
- First look: `q_0` comes from the empty memory, so the first pick is a learned prior over brain locations.
- Batched PPO: pad candidates to the max N in the batch with a mask. Envs become a batched tensor op on GPU.

### 4. Training schedule (discussed 2026-10-13)

Options considered:

- **Frozen reader (classifier trained on random subsets, then RL).** A clean measure of selection alone. Risk: the agent picks non-random, clustered sets the reader never saw, so good picks may not improve its accuracy. The agent can also only find what the reader can read.
- **Joint from scratch.** Risks:
  - a cold start, where the reward is noise, can make the chooser and reader lock into an arbitrary "private language";
  - the reward keeps moving;
  - the reader memorises the training scans.

Plan: warm start, then train together.

1. Pretrain K=5 cross-fitted readers on random subsets of size 1..`max_steps` (attention reader, balanced). Reader Ri never trains on scans from subject group Gi.
2. Stage A, frozen readers + PPO: measures selection alone. A scan in group Gi is scored by Ri.
3. Stage B, joint: readers keep training on the chooser's episodes, mixed ~50% with random subsets, updated every N policy updates at a lower learning rate. Each episode trains 4 readers and is scored by the 5th.
4. Evaluation at budgets 4/8/16/32 on held-out test: chooser + readers vs random picks + readers trained on random picks.

## Open questions

- **Q1 (2026-10-12):** Can the other repo's CN/AD model take a subset of tokens, e.g. attention with a padding mask?
  - If yes: use it as the frozen reward/diagnosis classifier, or as a teacher, and as the full-scan ceiling.
  - Currently assumed **no**.
- **Q2 (2026-10-12): Embeddings are context-mixed.** NeuroVFM encodes all kept tokens of a scan as one self-attention sequence (`encoder.py`, `series_cu_seqlens=[0, n]`). So every token embedding already contains information from the whole scan.
  - Measured on OASIS fold 0 (logistic regression, test split, 24 AD), in the table below.
  - A single random token already gets most of the way to the full scan. The useful range for the agent is about AUC 0.77–0.81 (random looks) to 0.88 (all tokens). It must beat random selection at the same budget to show value.
  - Gains may come from *which token's context* is read, not from local anatomy.
  - Option for the strict re-extraction: also cache local embeddings (each patch's neighbourhood encoded alone, or early-layer features), so "hidden until inspected" is real.
- **Q3:** When the original OASIS images arrive, re-extract strict tokens with `neuro_evidence` (`.npz`, `load_npz_token_sample`) and switch the config. Decide whether to keep `brain_fraction >= 0.5` filtering for the agent's candidate set.
- **Q4:** Fold 0 test has only 24 AD scans, so AD recall moves about ±10% from sampling alone. Report all 5 folds before trusting a number.

## Measurements

| Date | What | Result |
|---|---|---|
| 2026-10-12 | OASIS fold 0 sizes | train 1,350 (94 AD), test 337 (24 AD) |
| 2026-10-12 | OASIS tokens per scan (30 scans) | median 240 (168–772), grid ~8×10×8 |
| 2026-10-12 | Valid 6-neighbours per token | 3.97 of 6 (~1/3 of one-cell moves fail) |
| 2026-10-12 | CN/AD probe, fold 0 test: mean of all tokens | AUC 0.876, bal acc 0.729 |
| 2026-10-12 | Same, one random token | AUC 0.769, bal acc 0.617 |
| 2026-10-12 | Same, mean of 8 random tokens | AUC 0.810, bal acc 0.696 |
| 2026-10-12 | Test AUC vs k random tokens (mean-pool + logistic regression, 3 draws) | k=1 0.743, 2 0.758, 4 0.795, 8 0.810, 16 0.822, 32 0.863, 64 0.849, all 0.876. Random looks reach ~0.86 by k=32, so the agent has the most room to help at small budgets (4–16). |
| 2026-10-13 | Same probe, all 5 folds, 10 draws (`src/evaluation/random_k_baseline.py`). Mean test AUC over folds (± std across folds) | k=1 0.774±0.033, 2 0.791±0.024, 4 0.829±0.021, 8 0.848±0.023, 16 0.849±0.017, 32 0.869±0.019, 64 0.876±0.017, all 0.894±0.021. Std across draws within a fold is 0.02–0.04. Gap to all tokens: 0.065 at k=4, 0.046 at k=8–16, 0.025 at k=32. |

### First pointer-agent run (2026-10-13): `outputs/pointer_runs/stageA_fold0_b8`

Stage A (frozen readers), fold 0, budget 8, look cost 0.01, 1000 updates (2.9 min on one GPU; data loading and readers take ~1.5 min more).

- **Readers on random looks (attention readers, test AUC):** k=1 0.814, k=4 0.838, k=8 0.870, k=32 0.881. Out-of-fold on train: 0.781 / 0.835 / 0.849 / 0.847. Already about equal to the logistic probe on *all* tokens (0.876), so there is little headroom at k=8.
- **Training:** return −0.02 → +0.41, token-choice entropy 5.2 → 1.3, train acc 0.78 → 0.88. Looks per episode fell to ~3, so it learned to stop early.
- **Test (best.pt @ update 400, chosen on val AUC):** AUC 0.860 vs 0.854 for random looks with the same number of looks per scan (gain +0.006, within noise: 24 AD). Bal acc 0.813 (CN 0.834 / AD 0.792). 4.8 looks on average vs 8 allowed.
  - Looks per scan vary: 42 scans stop after 1 look, 130 use all 8. The stopping is adaptive.
  - 70 distinct first picks over 337 scans.
- **Val is too noisy to pick checkpoints:** the gain bounces between −0.017 and +0.040 across evals (~9 AD scans).
- KL spiked to 0.075 around update 700. Consider an approx-KL early stop per update.

Reading: the agent matches random looks while using ~40% fewer looks, but doesn't yet choose better tokens than random at the same count. That fits Q2: every token carries whole-scan context, so where it looks matters little.

### Budget sweep (2026-10-13): `outputs/pointer_runs/stageA_fold0_b{2,4,8}_ret`

Stage A, fold 0, readers reused from `stageA_fold0_b8`. best.pt chosen on val class-balanced mean return; PPO with target KL 0.02.

| Budget | Test AUC agent | Test AUC random (matched looks) | Test return agent / random | Looks | Val return agent / random (end) |
|---|---|---|---|---|---|
| 2 | 0.822 | 0.857 | 0.118 / 0.188 | 2.0 | 0.365 / 0.335 |
| 4 | 0.834 | 0.880 | 0.109 / 0.201 | 4.0 | 0.317 / 0.363 |
| 8 | 0.842 | 0.861 | 0.087 / 0.099 | 7.7 | 0.282 / 0.270 |

- **On test the agent is worse than random looks at every budget** (AUC −0.02 to −0.05). On val it's about even, and at budget 2 better. With 9 val AD and 24 test AD, both are noisy.
- **The random baseline itself is noisy.** At budget 2 both random numbers use exactly 2 tokens per scan, yet they came out at 0.857 and 0.802 (3 draws each). Raised `random_draws` to 10.
- **Likely cause: problem 1 in §4 (distribution shift).** The greedy agent picks similar, clustered tokens. The readers were trained only on random, spread-out sets. With context-mixed embeddings (Q2), a spread-out random sample gives the readers more independent views than a cluster.
- **What should address it:**
  - stage B (`--joint`), so the readers learn the agent's sets;
  - all 5 folds, to average out the noise;
  - a diversity term, or sampling instead of greedy at eval;
  - local embeddings (Q2) as the long-term fix.

### 5-fold sweep, stage A vs B, budgets 2 and 4 (2026-10-13): `outputs/pointer_runs/sweep_fold*`

Each fold pretrains its own readers in `A_b2` and reuses them for the other three runs; stage B fine-tunes its own copy. Summary: `outputs/pointer_runs/summary_sweep_fold.csv` (`src/evaluation/summarize_pointer_runs.py`). Test split of each fold; random = same readers, same number of looks per scan, 10 draws.

| Stage, budget | Test AUC agent | Random (matched) | AUC gain, mean (std) | Folds won | Return gain | Bal acc | Looks |
|---|---|---|---|---|---|---|---|
| A, 2 | 0.870 | 0.863 | +0.007 (0.022) | 3/5 | +0.014 | 0.792 | 2.0 |
| A, 4 | 0.862 | 0.875 | −0.013 (0.016) | 1/5 | −0.019 | 0.775 | 3.9 |
| **B, 2** | **0.882** | 0.866 | **+0.016 (0.007)** | **5/5** | +0.058 | 0.784 | 2.0 |
| B, 4 | 0.839 | 0.844 | −0.004 (0.015) | 2/5 | −0.034 | 0.766 | 2.4 |

- **Stage A** (frozen readers) gives no gain over random. That's consistent with the distribution-shift explanation.
- **Stage B, budget 2** is the first consistent gain: +0.016 AUC, positive in all 5 folds (paired t ≈ 5, df 4). Small, but in every fold.
  - Its random baseline (0.866) matches stage A's (0.863), so the joint readers did not get worse on random sets and the comparison is fair.
  - The AUC gain does not show up in balanced accuracy (0.784 vs 0.792), which depends on the 0.5 threshold.
- **Stage B, budget 4 is unstable:**
  - fold 2 collapsed (AUC 0.758; its random baseline also fell to 0.773, so the readers degraded);
  - 3 folds learned to stop after 1–1.5 looks;
  - stage B random returns are much lower than stage A's (0.066 vs 0.166), so the joint readers became less calibrated;
  - best checkpoints come from very early updates (20–220).
- **Next:**
  - stabilise joint training: lower reader lr, higher random fraction, fewer reader steps, reader early stopping on val;
  - repeat B, budget 2 with more seeds;
  - add a fixed-location baseline (the same learned tokens for every scan) to check whether the gain comes from adapting to each scan or just from a good fixed location.

### Sweep 2: guarded stage B, seeds, blind check (2026-10-13): `outputs/pointer_runs/sweep2_*`

Changes:
- Joint reader updates are gentler: lr 3e-5, 20 steps, 75% random sets.
- A guard undoes a reader's update if its val balanced BCE on fixed random sets rises by more than 0.01.
- New check, the **blind policy**: the same trained policy with the inspected embeddings zeroed and the reader belief fixed at 0.5. It can then only follow one fixed location sequence for every scan. Reported as `blind_auc` (`BlindPolicy` in `pointer_eval.py`).

Results (summary: `outputs/pointer_runs/summary_sweep2_.csv`):

| Runs | Test AUC agent | Random (matched) | Gain mean (std) | Wins | Blind | Agent − blind |
|---|---|---|---|---|---|---|
| A, budget 2, seed 7 (rerun) | 0.857 | 0.863 | −0.006 (0.020) | 1/5 | 0.862 | −0.005 |
| B, budget 2, seeds 7/8/9 (15 runs) | 0.872 | 0.862 | +0.010 (0.025) | 11/15 | 0.873 | −0.002 (0.012) |
| B, budget 4, seed 7 | 0.873 | 0.875 | −0.002 (0.024) | 2/5 | 0.880 | −0.007 |

Findings:
1. **The agent does not adapt to each scan.** The blind policy matches it in every configuration (agent − blind = −0.002 ± 0.012 over 15 runs). It has learned a fixed sequence of locations, not a search that reacts to what it sees. This fits Q2: each context-mixed token already summarises the whole scan, so seeing it doesn't change where it pays to look next.
2. **Any gain is small and noisy.** Guarded B at budget 2 gives +0.010 ± 0.025 (11/15). Unguarded B in sweep 1 gave +0.016 ± 0.007 (5/5), but on one seed.
   - The same stage A run (same fold, seed and readers) gave AUC 0.870 in sweep 1 and 0.857 in sweep 2. So the variance from one run to the next is about 0.01–0.03 per fold, as large as the gain.
3. **The guard reverts ~97% of reader updates even at 0.01 tolerance,** so guarded B is close to frozen readers. Every joint update trades random-set accuracy for the agent's sets. The budget-4 collapse is gone (no fold below 0.857), but there's no gain either.

Conclusion for the stand-in tokens: RL selection buys at most ~+0.01 AUC, and that comes from good fixed locations, not adaptive search. A direct search for the best fixed token positions would likely do as well without RL. Adaptive looking can only matter with embeddings that don't already contain the whole scan (Q2: local or early-layer embeddings).

## Baselines the agent must beat (same look budget)

- k random tokens → classifier `f`
- full scan (all tokens) → `f`: the ceiling
- brain size / token count only: checks the shape leak (§1, finding 7)
