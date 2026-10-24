# NeuRL

Minimal research repo for atlas-free evidence extraction from 3D brain MRI using a frozen NeuroVFM encoder.

Current scope:

```text
1 MRI -> official NeuroVFM preprocessing -> frozen NeuroVFM -> token embeddings + coordinates -> inspect / visualize
```

No atlas, RL, world model, multimodal modeling, or attention baseline is implemented here yet.

## Structure

```text
notebooks/
  00_data_sanity.ipynb
  01_neurovfm_features.ipynb
  02_token_baseline.ipynb
src/neuro_evidence/
  data.py
  encoder.py
  pooling.py
  metrics.py
configs/
  baseline.yaml
outputs/
README.md
requirements.txt
```

## Setup

From this repo:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m ipykernel install --user --name neurl --display-name "Python (NeuRL)"

# Official NeuroVFM code (gitignored)
git clone --depth 1 https://github.com/MLNeurosurg/neurovfm ext/neurovfm

# NeuroVFM encoder weights (gated on Hugging Face: request access first, then log in)
.venv/bin/hf auth login
.venv/bin/hf download mlinslab/neurovfm-encoder --local-dir checkpoints/hf/mlinslab/neurovfm-encoder
```

Start Jupyter with `.venv/bin/jupyter lab` and pick the "Python (NeuRL)" kernel.

`configs/baseline.yaml` expects:

- official NeuroVFM code at `ext/neurovfm`
- encoder checkpoint at `checkpoints/hf/mlinslab/neurovfm-encoder`
- a full-head sample MRI at `sample_mri`, with its HD-BET mask next to it as `<stem>_mask.nii.gz` (currently an ADNI-DOD scan, see below)

`configs/active_neuro_agent.yaml` reads the OASIS-3 sample data from `/mnt/shared/datasets/NeuRL-dataset` (`data_root`, see below).

## Data

`data/` is gitignored. `data/adnidod/` holds 8 ADNI-DOD T1 scans (4 CN, 4 MCI, one per subject), copied from `../civil/brainet/data/adnidod`:

```text
data/adnidod/raw/*.nii                      # original N3 T1 scans
data/adnidod/manifest.csv                   # clinical rows (DX, MMSE, AGE, ...) + image_path
data/adnidod/manifest_processed.csv         # + processed_path, mask_path, head_mask_path
data/processed/strict/*.nii.gz              # scans as-is, NeuroVFM's own pipeline (+ *_head.nii.gz Otsu head masks)
data/processed/n4_headcrop_1mm/             # previous version: N4, RAS 1mm, head crop
data/processed/n4_crop_1mm/                 # older version: N4, RAS 1mm, HD-BET brain-box crop (+ *_mask.nii.gz)
data/processed/fullhead_1mm/                # oldest version: RAS 1mm, no N4, no crop
```

### OASIS-3 sample data (agent development)

The active agent (`notebooks/03_active_neuro_agent_baseline.ipynb`, `src/env`, `src/training`) currently trains on cached OASIS-3 NeuroVFM tokens from the shared dataset volume (2013 scans, 1338 subjects):

```text
/mnt/shared/datasets/NeuRL-dataset/
  oasis/processed/neurovfm_tokens/*.pt                 # {embeddings [N,768] fp16, coords [N,3], token_count, embedding_dim, grid_shape}
  oasis/processed_mri/shared_n4_hdbet_crop/            # volumes the tokens were extracted from (+ *_mask.nii.gz)
  oasis/processed_mri/brainiac_canonical/              # 96^3 BrainIAC inputs (unused here)
  oasis/processed_mri/oasis_*_manifest.csv             # clinical rows + preprocessing provenance
  postprocessed/oasis_c3/split_info/folds.csv          # 5-fold, subject-grouped, CN vs AD (CDR 0.5 dropped)
```

**These tokens are from skull-stripped input** (N4 → HD-BET skull strip → brain crop, RAS 1 mm), not the strict unstripped pipeline above, so they differ from what NeuroVFM was trained on (see "Why not skull-strip"). Treat them as stand-in data for building and debugging the agent architecture and training loop only. Results on them are not comparable to strict tokens. The original OASIS images are not on the shared volume yet. Once they are, re-extract strict tokens with `neuro_evidence` and switch the config over.

Train the agent (one fold; subject-grouped 10% validation carved from the fold's train split picks `best.pt`, which is scored once on the fold's test split at the end):

```bash
.venv/bin/python src/training/train_agent.py --config configs/train_agent.yaml [--fold 0] [--updates 500] [--name my_run]
tail -f outputs/runs/<run>/train.log    # one line per PPO update, VAL line every training.eval_every updates
```

The run directory has `train_metrics.csv`, `val_metrics.csv`, `last.pt`, `best.pt`, `test_metrics.json` and the resolved `config.yaml`.

`folds.csv` stores brainet paths (`/home/dev/brainet/data/...`, `data/...`). `build_token_manifest(..., data_root=...)` remaps them onto `data_root`. Each loaded sample carries `preprocessing: "n4_hdbet_skullstrip_crop"`.

## Preprocessing

Default is **strict**: NeuroVFM's own pipeline (Kondepudi et al., [arXiv 2511.18640](https://arxiv.org/abs/2511.18640)). NeuroVFM was trained on raw clinical scans with **no skull stripping** ("foreground masks remove air only"), no registration and no N4. Our preprocess only writes the scan unchanged as NIfTI (DICOM series are converted; pixel type and pose kept). NeuroVFM's `load_image`/`prepare_for_inference` then reorient to RPI, resample to 1x1x4 mm with 4 mm along the acquisition axis (sagittal for ADNI MPRAGE), clip to the 0.5-99.5th percentile, scale to [0,1] and cut 4x16x16 patches; `encoder.embed()` applies the training mean/std. Token caches from `data/processed/strict` are identical to feeding the raw files directly.

```bash
cd src
../.venv/bin/python -m neuro_evidence.preprocess --manifest ../data/adnidod/manifest.csv
../.venv/bin/python -m neuro_evidence.preprocess path/to/scan.nii.gz   # single scans also work
```

Optional steps, all off by default (`preprocess:` section of `configs/baseline.yaml`, applied in this order):

| Setting | Effect |
|---|---|
| `n4.enabled: true` | N4 bias correction (~no-op on ADNI's N3-corrected downloads) |
| `orientation: RAS` | reorient before NeuroVFM |
| `spacing: [1, 1, 1]` | resample; isotropic input makes NeuroVFM slice along I-S instead of the acquisition axis |
| `skull_strip.method: hdbet` | HD-BET brain mask saved as `*_mask.nii.gz` (crop + `neurovfm.min_brain_fraction`); pixels are never stripped |
| `crop.enabled: true` | crop the FOV to the brain mask (hdbet) or head mask (none) + `margin_mm` |

The previous pipeline (`n4_headcrop_1mm`) is `orientation: RAS`, `spacing: [1, 1, 1]`, `n4.enabled: true`, `crop.enabled: true`. Compared with strict on the 8 ADNI-DOD scans, its pooled embeddings have cosine 0.73-0.82 to the strict ones (0.82-0.85 over brain tokens), so the choice matters downstream. An Otsu head mask `*_head.nii.gz` is always written (for `neurovfm.token_rule: head`). Change `out_dir` when changing steps so versions do not mix. `python -m neuro_evidence.viz <scan> --out <png>` shows the exact volume NeuroVFM tokenizes, with the kept tokens overlaid.

Why not skull-strip: NeuroVFM was trained on unstripped scans. Zeroing everything outside the brain shifts its percentile intensity normalization (brain moves from ~0.3 to ~0.7, white matter saturates at 1.0) and breaks its background filter (empty patches kept, most cortex dropped).

### Token selection

`neurovfm.token_rule` (in `NeuroVFMEncoder.extract()`):

- `neurovfm` (default): NeuroVFM's released rule, patch dropped if any voxel is at/below the volume's 10th percentile. The released training code (`data/cache.py`, `datasets/dataset.py`, `pretrain.yaml`) uses the same rule, so this is what the model saw. ADNI's air is noisy (median ~120 vs a 10th percentile of ~81), so ~200 pure-air tokens per scan get through; the paper notes the same ("MRI foregrounds often include air due to lack of explicit skull stripping"). It also drops patches with a few dark voxels (sulcal CSF, bone): only 29-91% of the brain is kept.
- `head`: keep patches entirely inside our Otsu head mask `<scan stem>_head.nii.gz`. The paper's Methods mention an Otsu mask without details; this implementation (Otsu on log intensity, largest component, holes filled) is ours. On the 8 ADNI-DOD scans it keeps 715-1193 tokens per scan, covers 100% of brain patches (HD-BET brain fraction >= 0.5) and no air; ~60% of tokens are skull/scalp/face/neck. With an HD-BET mask, `min_brain_fraction` still adds/labels brain patches (`null` by default).

## Run

Note: the official NeuroVFM package imports `flash_attn` and `positional_encodings`. `src/neuro_evidence/encoder.py` installs a small `flash_attn` compatibility fallback when `flash_attn` is unavailable, because this minimal inference path uses `use_flash_attn=False`. You still need `positional-encodings` installed for the official positional encoding module. If NeuroVFM ever calls flash attention anyway, install a CUDA-compatible `flash-attn` build.

1. Open `notebooks/00_data_sanity.ipynb` and confirm the sample MRI loads.
2. Open `notebooks/01_neurovfm_features.ipynb` and extract/cache NeuroVFM token embeddings.
3. `notebooks/02_token_baseline.ipynb` is a placeholder for later simple baselines.

Expected token cache:

```text
outputs/tokens/<sample_id>.npz
```

It contains `embeddings [N, 768]`, `coords [N, 3]`, `brain_fraction [N]` (HD-BET brain fraction per token), `neurovfm_foreground [N]` (False = added by the mask rule), and JSON metadata (including `geometry` to map coords back to the scan). Downstream code should usually restrict to brain tokens, e.g. `brain_fraction >= 0.5`, while the encoder still sees the whole head as context.

## Standard baseline

Run the standard baseline (random-k logistic-regression probe + pointer agent with frozen attention readers, seeds 0–2, budgets 4/6/8) for any registered dataset and encoder cache:

```bash
.venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_local --gpus 1
```

Datasets, encoder caches and the fixed protocol are registered in `configs/baselines.yaml`; results go to `outputs/baselines/<dataset>__<encoder>/REPORT.md`. How to run it, add a cache or a dataset, and read the report: [`docs/baselines.md`](docs/baselines.md). Claude Code picks this up through the `run-baseline` skill (`.claude/skills/run-baseline/`), so asking it to "run the baseline with ResNet on ADNI" is enough.
