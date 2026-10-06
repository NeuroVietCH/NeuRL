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
- a skull-stripped sample MRI at `sample_mri` (currently an ADNI-DOD scan, see below)

`configs/active_neuro_agent.yaml` expects cached OASIS folds/manifest/token files under `../brainet/data/...`; update those paths to wherever your OASIS data lives.

## Data

`data/` is gitignored. `data/adnidod/` holds 8 ADNI-DOD T1 scans (4 CN, 4 MCI, one per subject), copied from `../civil/brainet/data/adnidod`:

```text
data/adnidod/raw/*.nii                      # original N3 T1 scans
data/adnidod/manifest.csv                   # clinical rows (DX, MMSE, AGE, ...) + image_path
data/adnidod/manifest_processed.csv         # + processed_path, mask_path
data/processed/n4_hdbet_crop/*.nii.gz       # preprocessed scans (+ *_mask.nii.gz)
```

## Preprocessing

NeuroVFM's `StudyPreprocessor` only reorients, resamples to 1x1x4mm, normalizes intensities, and drops dark background patches. It does not bias-correct or skull-strip, so clean scans first:

```bash
cd src
../.venv/bin/python -m neuro_evidence.preprocess --manifest ../data/adnidod/manifest.csv
../.venv/bin/python -m neuro_evidence.preprocess path/to/scan.nii.gz   # single scans also work
```

Steps (settings in the `preprocess:` section of `configs/baseline.yaml`): N4 bias correction -> RAS 1mm -> HD-BET skull strip (GPU) -> crop to brain. This matches the `shared_n4_hdbet_crop` profile used for the original OASIS cache. HD-BET downloads its weights to `~/hd-bet_params` on first run. Use `--skull-strip fallback` for a quick smoke test without HD-BET.

## Run

Note: the official NeuroVFM package imports `flash_attn` and `positional_encodings`. `src/neuro_evidence/encoder.py` installs a small `flash_attn` compatibility fallback when `flash_attn` is unavailable, because this minimal inference path uses `use_flash_attn=False`. You still need `positional-encodings` installed for the official positional encoding module. If NeuroVFM ever calls flash attention anyway, install a CUDA-compatible `flash-attn` build.

1. Open `notebooks/00_data_sanity.ipynb` and confirm the sample MRI loads.
2. Open `notebooks/01_neurovfm_features.ipynb` and extract/cache NeuroVFM token embeddings.
3. `notebooks/02_token_baseline.ipynb` is a placeholder for later simple baselines.

Expected token cache:

```text
outputs/tokens/<sample_id>.npz
```

It contains `embeddings [N, 768]`, `coords [N, 3]`, and JSON metadata.
