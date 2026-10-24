---
name: run-baseline
description: Run the NeuRL standard baseline (logistic-regression random-k probe + pointer agent with frozen attention readers, seeds 0-2, budgets 4/6/8) for a dataset and an encoder cache, and report its results. Use when someone asks to "run the baseline", "run the standard baseline", "get baseline numbers", or "run the logistic regression / attention baseline" with an encoder (NeuroVFM local or global, ResNet3D, Neuro-JEPA, ...) on a dataset (OASIS, ADNI, ADNI dev subset, ...), or asks to register a new encoder cache or dataset for the baseline.
---

# Run the standard baseline

The standard baseline is fixed. The protocol lives in `configs/baselines.yaml`: the old pointer-agent architecture under the plan v3 training setup, plus the random-k logistic-regression probe. One script runs it for any registered (dataset, encoder) pair: `scripts/run_baseline.py`. Human docs: `docs/baselines.md`. Read that file if anything below is unclear.

## 1. Map the request to registry keys

Read `configs/baselines.yaml`. Map the user's words to a `--dataset` key (`oasis`, `adni`, `adni_dev`, ...) and a `--encoder` key under that dataset's `caches`:

| User says | Key |
|---|---|
| NeuroVFM, local, local context, window, rawL | `neurovfm_local` |
| NeuroVFM global, full attention, rawG | `neurovfm_global` |
| old / skull-stripped NeuroVFM | `neurovfm_stripped` |
| ResNet, CNN, SimCLR ResNet | `resnet3d` |
| JEPA, Neuro-JEPA | `neurojepa` |

If the request is ambiguous (e.g. "NeuroVFM" on OASIS, where local and global both exist), the team default is `neurovfm_local`. Say which key you chose.

If the entry is `null` or missing, the cache or scan table is not registered yet. Ask for the path; don't guess. Then either pass `--token_dir` / `--scan_table` for a one-off run, or add the path to the registry (and offer to commit it). A new dataset needs a scan table in the format given in `docs/baselines.md`. Check its columns before running.

## 2. Check before launching

```bash
.venv/bin/python scripts/run_baseline.py --dataset <ds> --encoder <enc> --dry_run --gpus <ids>
```

- It must pass the data check: both classes in every train/test split, and token files found. If it reports problems, show them to the user and stop.
- Choose GPUs. On this server, check `nvidia-smi` and pick GPUs with at least the `~X GB GPU per job` the check prints. The GPUs are shared and fill up; the runner refuses GPUs without room and lists the ones that have it.
- RAM: this server has 24 GB, so keep `--jobs 3` or fewer. Also check `free -g` and that no other heavy jobs are running (`ps aux | grep train_pointer_agent`). With 4 jobs, runs were OOM-killed.
- On CSCS/SLURM, use `--job FOLD SEED` per array task (see `docs/baselines.md`). That path has not been tested yet; say so.

## 3. Launch in the background and monitor

```bash
nohup .venv/bin/python scripts/run_baseline.py --dataset <ds> --encoder <enc> --gpus <ids> --jobs 3 \
  > outputs/baselines/<ds>__<enc>.nohup.log 2>&1 &
```

Watch `outputs/baselines/<ds>__<enc>/run.log`. It has one `start` / `ok` / `FAIL` line per job and a final `done` line. Use a background wait or monitor that also matches `FAIL` and `Traceback`, not only success. On a failure, read `outputs/baselines/<ds>__<enc>/logs/<run>.log`. Rerunning the same command resumes and skips finished runs. Rough timing: OASIS takes about 75 minutes for 45 runs with 3 jobs; one run takes about 4–5 minutes.

Do not change `configs/pointer_baseline_std.yaml` or the `protocol` block to make one run work. If a variant is needed (other seeds or budgets, a subset of folds), use `--seeds/--budgets/--folds` and label the result as non-standard.

## 4. Report the results

When `done` appears, read `outputs/baselines/<ds>__<enc>/REPORT.md` and report:

1. The **agent vs random looks** table per budget: agent AUC (std), random, gain, wins, fold p, blind, balanced accuracy.
2. **Readers alone** on k random tokens, and the **logistic-regression probe** table (k = 1…64 and all tokens).
3. A short reading:
   - Is the gain over random significant, and at which budgets?
   - Does the agent equal the blind policy (no per-scan adaptation)?
   - Comparison with other baselines in `outputs/baselines/INDEX.md`, especially the same encoder on another dataset or another encoder on the same dataset.
   - Per-fold differences under ~0.02 AUC are noise.
4. Where the files are, and whether all runs finished (`Pointer runs finished: n/n`).

The reference for this format is the OASIS standard baseline in `docs/stage2_notes.md` ("Standard baseline (2026-10-23)"). Offer to add the new table there, appended under a dated heading. Never rewrite older entries.
