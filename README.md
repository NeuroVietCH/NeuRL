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

`configs/active_neuro_agent.yaml` expects cached OASIS folds/manifest/token files under `../brainet/data/...`; update those paths to wherever your OASIS data lives.

## Data

`data/` is gitignored. `data/adnidod/` holds 8 ADNI-DOD T1 scans (4 CN, 4 MCI, one per subject), copied from `../civil/brainet/data/adnidod`:

```text
data/adnidod/raw/*.nii                      # original N3 T1 scans
data/adnidod/manifest.csv                   # clinical rows (DX, MMSE, AGE, ...) + image_path
data/adnidod/manifest_processed.csv         # + processed_path, mask_path
data/processed/n4_crop_1mm/*.nii.gz         # N4, RAS 1mm, FOV-cropped full-head scans (+ *_mask.nii.gz HD-BET masks)
data/processed/fullhead_1mm/                # previous version: no N4, no crop
```

## Preprocessing

NeuroVFM was trained on full-head scans (only the air around the head is dropped), so the image it sees is **not** skull-stripped:

```bash
cd src
../.venv/bin/python -m neuro_evidence.preprocess --manifest ../data/adnidod/manifest.csv
../.venv/bin/python -m neuro_evidence.preprocess path/to/scan.nii.gz   # single scans also work
```

Input is a NIfTI file or a DICOM series directory (ADNI downloads either). Steps (settings in the `preprocess:` section of `configs/baseline.yaml`): N4 bias correction -> RAS 1mm -> HD-BET brain mask (GPU) -> crop the field of view to the brain bounding box + 25 mm (40 mm below the brain). The full-head image and the mask are written as separate files to `data/processed/n4_crop_1mm/`; the mask is never applied to the pixels. HD-BET downloads its weights to `~/hd-bet_params` on first run. Use `--skull-strip fallback` for a quick smoke test without HD-BET. `python -m neuro_evidence.viz <scan> --out <png>` shows the exact volume NeuroVFM tokenizes, with the kept tokens overlaid.

Why crop: NeuroVFM's background rule (patch dropped if any voxel is below the volume's 10th percentile) lets noisy air through. On an uncropped ADNI-DOD scan with a large FOV, 254 of 803 kept tokens were pure air below the head; after the crop, 11. N4 alone does not change this. No registration is done: tokens stay in each scan's native pose.

Why not skull-strip: zeroing everything outside the brain shifts NeuroVFM's percentile intensity normalization (brain moves from ~0.3 to ~0.7, white matter saturates at 1.0) and breaks its background filter (empty patches kept, most cortex dropped).

### Token selection

`NeuroVFMEncoder.extract()` keeps NeuroVFM's own foreground patches (no voxel at/below its background threshold) **plus** every patch whose HD-BET brain fraction is at least `neurovfm.min_brain_fraction` (default 0.5). NeuroVFM's rule alone drops patches that contain a few dark voxels (sulcal CSF, bone), which removes much of the cortex and varies a lot between scans. On the 8 ADNI-DOD scans, brain coverage goes from 29-91% (mean 63%) to 89-97% (mean 92%); embeddings of NeuroVFM's own brain tokens stay close (mean cosine 0.94) when the extra patches join the context. Set `min_brain_fraction: null` for NeuroVFM's rule only. Without a mask file, `extract()` warns and falls back to NeuroVFM's rule.

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
