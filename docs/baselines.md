# Running the standard baseline

One command runs the full standard baseline for any registered (dataset, encoder) pair:

```bash
.venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_local --gpus 1
```

It produces the same results as the OASIS standard baseline in `stage2_notes.md` (2026-10-23), for any encoder cache and any dataset. Use these numbers as the reference that new architectures must beat.

## What it runs

| Stage | What | Script |
|---|---|---|
| `check` | Scans per fold/split with a token file, both classes present, embedding width, tokens per scan, RAM/GPU estimate | `scripts/run_baseline.py` |
| `probe` | Logistic regression on the mean of k random tokens, k = 1…64 and all tokens, 10 draws per fold | `src/evaluation/random_k_baseline.py` |
| `pointer` | Pointer agent (old architecture: position-only policy, no STOP) with 3 frozen cross-fitted attention readers, every fold × seed (0, 1, 2) × budget (4, 6, 8). Per (fold, seed), budget 4 pretrains the readers and budgets 6 and 8 reuse them. | `src/training/train_pointer_agent.py` |
| `report` | `REPORT.md`, `summary.csv`, `per_run.csv`, and the cross-baseline table `outputs/baselines/INDEX.md` | `scripts/run_baseline.py` |

The protocol (training config `configs/pointer_baseline_std.yaml`, seeds, budgets, probe settings) is fixed in `configs/baselines.yaml`, so every dataset and encoder gets the same baseline. Do not change it for one run; if a run needs something different, use the override flags and say so when you report it.

Outputs go to `outputs/baselines/<dataset>__<encoder>/`:

```text
REPORT.md         the result tables (data check, agent vs random vs blind per budget, readers alone, probe)
summary.csv       one row per budget
per_run.csv       one row per (fold, seed, budget)
check.json        data check
baseline.json     what was run: token dir, scans, folds, seeds, budgets, git commit
run_config.yaml   the exact training config passed to the scripts
run.log           start / ok / FAIL per job
logs/             stdout of every job
probe/            probe per_draw.csv and summary.csv
pointer/fold{f}_s{s}_b{b}/   one run directory each (train.log, test_metrics.json, readers.pt, ...)
```

Rerunning the same command resumes: finished runs and a finished probe are skipped.

## Common commands

```bash
# What would run, and is the data OK? (seconds, runs nothing)
.venv/bin/python scripts/run_baseline.py --dataset adni --encoder resnet3d --dry_run --gpus 0

# Full baseline, 3 concurrent jobs on GPUs 2 and 3
.venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_global --gpus 2,3 --jobs 3

# Only the logistic-regression probe, or only regenerate the report
.venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_local --stages probe
.venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_local --stages report

# A cache that is not registered yet (one-off; register it once it is final)
.venv/bin/python scripts/run_baseline.py --dataset adni --encoder resnet3d --token_dir /path/to/cache --scan_table /path/to/adni_scans.csv
```

Long runs: start them with `nohup ... > baseline.log 2>&1 &` and follow `outputs/baselines/<dataset>__<encoder>/run.log`. The OASIS baseline (45 runs plus the probe) takes about 75 minutes with 3 jobs.

### This server

- 24 GB RAM: **at most 3 concurrent jobs** (`--jobs 3`, the default). Each job holds its whole token cache in RAM (`check` prints the estimate). With 4 jobs, runs were OOM-killed.
- The GPUs are shared. Before launching, the runner checks free GPU memory on `--gpus` and stops with a list of GPUs that have room if they are too full.
- Copy token caches from `/mnt/shared` to `data/` first. Loading from the mount adds about 7 minutes per run.

### CSCS / SLURM (not yet tested there)

Run one (fold, seed) chain per array task. `--job` uses the GPU the scheduler assigns, so do not pass `--gpus`:

```bash
#!/bin/bash
#SBATCH --array=0-2            # seeds 0..2 for a single-fold dataset; use 0-14 and map to (fold, seed) for 5 folds
#SBATCH --gres=gpu:1 --mem=32G --time=02:00:00
.venv/bin/python scripts/run_baseline.py --dataset adni --encoder neurovfm_local --job 0 $SLURM_ARRAY_TASK_ID
```

Then run `--stages check,probe` once (CPU is enough), and `--stages report` after all array tasks finish.

## Adding a dataset or an encoder

Everything is in `configs/baselines.yaml`.

**A new encoder cache** for an existing dataset: add one line under that dataset's `caches`:

```yaml
  adni:
    caches:
      resnet3d: /capstor/.../adni/resnet3d_layer3_tokens
```

The cache directory holds one `.pt` file per scan. Each file is a dict with the format written by `neuro_evidence.extract_tokens.to_cache` and `neurojepa.py`:

| Key | Shape / type |
|---|---|
| `embeddings` | `[N, D]` float16 (any D: 256 ResNet, 768 NeuroVFM, 1024 Neuro-JEPA) |
| `coords` | `[N, 3]` integer grid position of each token |
| `grid_shape` | `[3]` integer |
| `token_count`, `embedding_dim`, `scan_path`, `metadata` | recommended (Hưng's `scripts/check_token_cache.py` checks them) |

Positions are normalised per scan to its own token extent, so `coords` can be in any grid unit. File names must match the dataset's scan list (below).

**A new dataset**: add an entry with a `scan_table`, the folds and its caches. The scan table is a CSV with one row per (scan, fold):

| Column | Meaning |
|---|---|
| `sample_id` | unique scan id |
| `subject_id` | subject; cross-fitting and validation splits never separate a subject's scans |
| `label` | 0 = CN, 1 = AD |
| `fold` | fold index (a single train/val/test split is fold 0) |
| `split` | `train`, `val` or `test`. `val` is optional: without it, ~10% of train subjects are held out per seed. With it, the probe trains on train + val. |
| `token_file` | file name inside the encoder's cache dir (e.g. `<PTID>__<VISCODE>__<EXAMDATE>.pt`), so one table serves every encoder that uses the same names. Or `token_path`: absolute path, for a single-encoder table. |
| `class_name`, `volume_path`, `mask_path` | optional |

Scans without a token file are skipped, and `check` reports how many remain per split. OASIS keeps its original fold CSV (`folds_csv` + `metadata_csv`) instead of a scan table, so its numbers stay exactly reproducible.

## Reading the report

- **Agent vs random:** gain of the learned token choice over the same readers given the same number of random tokens. Treat per-fold differences under ~0.02 AUC as noise.
- **Blind:** the same policy with the content of seen tokens hidden. Agent = blind means the policy picks a fixed sequence of locations and does not adapt to each scan. This was true in all 90 OASIS standard-baseline runs.
- **Readers alone** and the **probe** show how much signal k random tokens carry. The probe on all tokens is the full-scan reference.
- Fold p needs at least 2 folds. With a single split (ADNI P1), compare seeds and use paired subject-level tests for claims.

Reference results: `docs/stage2_notes.md`, "Standard baseline (2026-10-23)", and `outputs/baselines/INDEX.md` on each machine.
