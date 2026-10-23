# TRACE-MRI: Trajectory-based Radiological Assessment with Conditional Evidence

**Implementation plan for CVPR 2027: Version 3**  
**Issued:** 14 October 2026  
**Revised:** 21 October 2026  
**Target:** CVPR 2027 Main Conference  
**Paper registration:** 10 November 2026 (AoE)  
**Full paper deadline:** 16 November 2026 (AoE)  
**Supplementary deadline:** 23 November 2026 (AoE)  
**Project status:** Implementation underway; use the current task tracker to report completed and delayed work

## 0. Read this first

We will implement TRACE-MRI as an encoder-independent method for sequential evidence acquisition from 3D brain MRI. The main task is binary classification of cognitively normal (CN) subjects and subjects diagnosed with Alzheimer's disease (AD).

The main experiments will use **ADNI1, ADNI2/GO, and ADNI3** (collectively ADNI), with **NeuroVFM** as the primary foundation encoder, a 3D ResNet comparator, and selected **JEPA** experiments if a suitable checkpoint exposes spatial features. **AIBL data are reserved for the later external testing stage.** The main contribution is a simple two-player evidence game. One policy looks for regions supporting the current diagnosis, and the other looks for regions challenging it. Both policies receive a reward for improving the prediction of the true diagnosis. The classifier updates its belief as additional regions are inspected.

**The scientific question:** Does competitive conditional evidence acquisition improve diagnostic performance, calibration, or information efficiency over single-agent RL, cooperative two-agent selection, and non-sequential attention at the same evidence budget?

The method must work with any encoder that exposes spatial feature maps or spatial tokens. We will test this property primarily with NeuroVFM and a 3D ResNet, then with JEPA in selected feasible scenarios. Keep the policy, evidence-reader architecture, and evaluation code identical across compatible backbones.

### Non-negotiable decisions

1. Primary MRI sequence: **3D T1-weighted structural MRI**.
2. Primary task now: **CN vs AD**, with `CN = 0` and `AD = 1`. Exclude MCI from the main binary experiment. Severity estimation, severity classification/regression, and multitask learning follow after the AD classification experiments.
3. Primary evaluation: subject-level separation, pooled ADNI development and ADNI-phase-specific held-out results; cross-phase transfer is a separate protocol. **AIBL remains a completely external test set, evaluated after the primary ADNI experiments.**
4. Core spatial action space: **the selected encoder's native spatial tokens**. Scan `i` has `N_i` valid tokens; batch padding to `N_max` is masked before action selection. A fixed 64-cell grid is an optional granularity ablation, not the main action space.
5. Main trajectory: **three rounds, two agent proposals per round**. Also evaluate 2, 4, 6, and 8 proposal budgets.
6. Core architecture: one frozen image encoder per run, a backbone-specific lightweight feature adapter, a GRU partial-evidence reader, and two small selection policies. Cross-fitting creates three reader instances with the same architecture.
7. Main policy algorithm: **categorical REINFORCE with moving-average baselines**. Use deterministic greedy selection at test time.
8. Main game: the **Chooser** consists of two policies, Advocate and Challenger. They share a diagnostic objective but have opposing conditional-evidence preferences. The Chooser is not a fourth agent.
9. Train the partial-evidence readers before training the agents; freeze them throughout policy training.
10. Do not train a new MRI foundation model. Use pretrained weights or foundation models when access is approved.
11. Cache image features. Avoid re-encoding MRI volumes at each RL step.
12. Do not use clinical labels, diagnostic notes, MMSE, MoCA, clinical scores, subject identifiers, or dataset names as model inputs. These are reserved for cohort construction, targets, splits, and evaluation.
13. Keep all ADNI scans from the same person in a single partition, including people participating in multiple ADNI phases. Keep AIBL subjects outside all ADNI training and hyperparameter tuning.
14. Report the entire development process, including failed runs, leakage findings, model-selection decisions, and the final tested checkpoints.
15. Make the first working experiment small and reproducible. Scale the same implementation after its tests pass. Select classification checkpoints by validation AUROC, save periodic iteration-based checkpoints, and use consecutive seeds starting at 0.
16. Use native tokens without revisiting observed locations in the main implementation. Test local-only against global attention and keep a separate global overview from acquired local evidence where the encoder supports it.
17. Run available baselines and ablations in parallel on CSCS. Students own implementation and experiments; I write the manuscript.

### What to send me

The first deliverable is a runnable repository with a verified manifest, a full-volume CNN baseline, native-token caches with coordinates and masks, and a partial-evidence reader that accepts variable token counts. The second deliverable is a controlled comparison of random, static Top-K, single-agent, cooperative two-agent, and TRACE-MRI selection on matched caches and budgets. The final deliverable is the ADNI result package, figures, ablations, checkpoints, and reproducibility documentation, followed by AIBL external results.

Send **tables and run links**, not screenshots of terminal outputs. Every numerical result must be traceable to a saved configuration, subject list, and checkpoint.

## 1. Research questions and contribution

### 1.1 Formal problem
For subject $i$, let

$$
X_i \in \mathbb{R}^{D\times H\times W}, \qquad y_i\in\{0,1\},
$$

where $X_i$ is a T1 MRI volume and $y_i$ is the CN/AD label at the matched imaging visit.

An image encoder $f_b$ produces spatial representations:

$$
F_i^{(b)}=f_b(X_i),
$$

Here $b$ denotes the backbone. The spatial adapter retains the encoder's native spatial tokens for each scan and projects their features into a common downstream interface:

$$
Z_i^{(b)}=\{(z_{ij},c_{ij})\}_{j=1}^{N_i},\qquad z_{ij}\in\mathbb R^{128},\quad c_{ij}\in[0,1]^3.
$$

Use $d=128$ after the backbone-specific projection. The coordinate $c_{ij}$ is the token center in the registered template space. Save the native-to-template transform. Token counts and indices may differ across scans and encoders; the shared contract consists of features, coordinates, a validity mask, and global context.

Let $S_t$ be the set of regions revealed by the end of round $t$. The evidence classifier produces

$$
p_t=P_\theta(Y=1\mid S_t,g_i),
$$

where $g_i$ is a low-dimensional global context vector shared by every selection baseline. At every round, the Advocate and Challenger each propose one unobserved region. The evidence classifier updates after the chosen regions are revealed.

The full decision process must expose:

- an ordered record of the selected regions;
- the agent responsible for each proposal;
- prediction probabilities before and after each round;
- the conditional contribution of each selected region;
- a final prediction and stopping step.

### 1.2 Hypotheses
**H1: Conditional selection.** Updating selection decisions from previously observed evidence improves the performance-versus-budget curve compared with static region ranking.

**H2: Competitive selection.** Opposing evidence-seeking objectives improve diagnostic evidence acquisition compared with matched cooperative two-agent policies.

**H3: Encoder transfer.** The same reader, policy, and game architectures operate on a 3D ResNet and NeuroVFM through a masked token interface; a JEPA-based encoder is a selected additional comparison if spatial features are accessible. Native feature widths, token counts, and spatial geometry may differ. Each encoder run trains its own adapter, readers, and policies.

**H4: Cohort transfer.** The learned selection strategy retains a measurable advantage on subjects held out from a different ADNI phase, with AIBL as subsequent independent external validation.

We will test each hypothesis separately. **H2 is the central method claim.** H1, H3, and H4 support its interpretation and generality.

### 1.3 Expected contribution in the paper
- A conditional evidence game for active region acquisition in volumetric visual recognition.
- A compact, encoder-independent implementation using discrete region selection and shared partial-evidence aggregation.
- A controlled study of competitive, cooperative, single-agent, and static selection strategies at matched observation budgets.
- Evaluation across ADNI1, ADNI2/GO, and ADNI3, with later external AIBL testing, including qualitative and quantitative analysis of evidence trajectories.

The actual performance claims will follow the measured results. The scope is diagnostic classification, conditional model-evidence trajectories, encoder independence, and cross-cohort validation. Report backbone feature-extraction time and selection-layer time separately.

## 2. Project structure and responsibilities

Use this repository structure. Keep datasets and checkpoints outside Git.

```text
trace-mri/
├── README.md
├── pyproject.toml
├── requirements.lock
├── configs/
│   ├── base.yaml
│   ├── datasets/
│   │   ├── adni_pooled.yaml
│   │   ├── adni_transfer.yaml
│   │   └── aibl_external.yaml
│   ├── encoders/
│   │   ├── resnet3d18.yaml
│   │   ├── neurovfm.yaml
│   │   └── jepa.yaml
│   ├── methods/
│   │   ├── full_volume.yaml
│   │   ├── evidence_aggregator.yaml
│   │   ├── random.yaml
│   │   ├── static_topk.yaml
│   │   ├── single_rl.yaml
│   │   ├── dual_cooperative.yaml
│   │   └── trace_game.yaml
│   └── experiments/
│       ├── pooled_main.yaml
│       ├── transfer_main.yaml
│       ├── budgets.yaml
│       └── ablations.yaml
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── splits/
│   ├── qc/
│   ├── cache_index/
│   └── transfer_status/
├── trace_mri/
│   ├── data/
│   │   ├── manifest.py
│   │   ├── labels.py
│   │   ├── split.py
│   │   ├── preprocess.py
│   │   ├── datasets.py
│   │   └── quality_control.py
│   ├── encoders/
│   │   ├── base.py
│   │   ├── resnet3d.py
│   │   ├── neurovfm.py
│   │   ├── jepa.py
│   │   ├── spatial_adapter.py
│   │   └── cache.py
│   ├── models/
│   │   ├── full_volume_head.py
│   │   ├── evidence_aggregator.py
│   │   ├── policies.py
│   │   ├── game.py
│   │   ├── stopping.py
│   │   └── trajectory.py
│   ├── baselines/
│   │   ├── random.py
│   │   ├── static_topk.py
│   │   ├── single_agent.py
│   │   └── dual_cooperative.py
│   ├── training/
│   │   ├── train_encoder.py
│   │   ├── train_aggregator.py
│   │   ├── train_selector.py
│   │   ├── checkpoint_steps.py
│   │   └── loops.py
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── bootstrap.py
│   │   ├── calibration.py
│   │   ├── evidence_metrics.py
│   │   └── audit_leakage.py
│   ├── visualization/
│   │   ├── trajectories.py
│   │   ├── anatomy_overlay.py
│   │   └── paper_figures.py
│   └── utils/
│       ├── config.py
│       ├── logging.py
│       ├── seeds.py
│       └── checkpointing.py
├── scripts/
│   ├── 00_build_manifest.py
│   ├── 01_make_splits.py
│   ├── 02_preprocess.py
│   ├── 03_train_cnn.py
│   ├── 04_cache_features.py
│   ├── 05_train_aggregator.py
│   ├── 06_train_baselines.py
│   ├── 07_train_trace.py
│   ├── 08_evaluate.py
│   ├── 09_make_figures.py
│   └── 10_validate_submission.py
├── tests/
│   ├── test_manifest.py
│   ├── test_patient_splits.py
│   ├── test_preprocessing.py
│   ├── test_spatial_adapter.py
│   ├── test_action_mask.py
│   ├── test_evidence_updates.py
│   ├── test_game_rewards.py
│   ├── test_policy_gradients.py
│   ├── test_no_label_leakage.py
│   ├── test_external_holdout.py
│   └── test_reproducibility.py
├── experiments/
│   ├── registry.csv
│   └── README.md
├── reports/
│   ├── dataset_audit.md
│   ├── encoder_audit.md
│   ├── attention_locality_audit.md
│   ├── cscs_job_audit.md
│   ├── benchmark_results.md
│   ├── ablation_results.md
│   └── open_questions.md
└── paper/
    ├── figures/
    ├── tables/
    └── result_manifest.json
```

### Work allocation

Assign one accountable person to each stream:

| Stream | Responsibility | First required output |
|---|---|---|
| Data | Access, cohort definition, QC, subject splits, preprocessing | `dataset_audit.md`, manifests, split files |
| Encoders | 3D ResNet, NeuroVFM, selected JEPA feasibility, feature cache, spatial adapter, local/global attention check | Validated features `[B,N_max,128]`, coordinates `[B,N_max,3]`, masks `[B,N_max]`, and coordinate tests for available encoders |
| Method | Evidence aggregator, policies, game training, inference | Working TRACE-MRI rollout and unit tests |
| Evaluation | Baselines, repeated runs, statistics, figures, reporting | Reproducible comparison table and trajectory plots |

Assign data/CSCS/encoders to Senior 1, RL/game to Senior 2, reader/baselines/evaluation coordination to Senior 3, and bounded QC/repeat-run/visualization tasks to Junior 4. All experiments are students' responsibility; the manuscript is mine. Baselines and feasible ablations must run in parallel instead of waiting for the full game model. Rebalance tasks against the current tracker. Submit focused pull requests; every experiment must name its code commit.

## 3. Data: ADNI1, ADNI2/GO, ADNI3; AIBL external later

### 3.1 Cohort definition
Use MRI matched to diagnosis at the appropriate imaging visit. Record the actual visit date and diagnosis date; do not infer them from scan filenames.

**Include:**

- 3D structural T1 MRI, preferably a QC-approved acquisition or the dataset-recommended corrected structural image;
- a valid clinical diagnosis at the matched or pre-specified nearest visit;
- a diagnosis of CN or AD;
- one selected scan per subject in the primary experiment.

**Exclude from the primary experiment:**

- MCI and ambiguous or unsupported diagnoses;
- missing or unmatched diagnosis;
- scans failing the pre-specified quality checks;
- duplicate acquisitions after selecting the single primary scan;
- scans with invalid orientation, severe truncation, or preprocessing failures.

When multiple acceptable scans exist for one subject, choose the scan closest to the earliest eligible baseline clinical diagnosis. Resolve ties using a fixed, quality-based rule recorded in the manifest. Never select a scan using model performance.

For ADNI, consult phase-specific diagnosis tables and their data dictionaries. Relevant fields differ across phases. Explicitly map each diagnosis code to CN, MCI, or AD in `labels.py`, then validate the map against the source documentation. For AIBL, use its documented diagnosis field and validate the corresponding codebook separately.

**Do not assume that ADNI phase labels identify distinct people.** Match all available subject identifiers across ADNI phases before creating any split. The primary identity key must be the stable participant identifier.

### 3.2 Manifest schema
Create `data/manifests/all_scans.csv`. Required columns:

```csv
subject_id,source_subject_id,dataset,phase,scan_id,image_id,visit_id,scan_date,diagnosis_date,diagnosis_original,label_binary,diagnosis_source,t1_sequence,scanner_vendor,scanner_model,field_strength_t,site_id,raw_path,processed_path,qc_status,qc_reason,selected_primary,preprocessing_version,split
```

Definitions:

- `subject_id`: stable, de-identified key within the merged dataset; it must map consistently across ADNI phases.
- `dataset`: `ADNI` or `AIBL`.
- `phase`: `ADNI1`, `ADNI2/GO`, `ADNI3`, or `AIBL`. Keep source phase labels separately when ADNI2 and ADNI-GO are distinguishable.
- `label_binary`: `0` for CN, `1` for AD, and null for other diagnostic categories.
- `selected_primary`: whether the scan is the single primary scan for that subject.
- `split`: assigned after split generation; retain separate experiment-specific split files.
- `qc_reason`: a controlled exclusion reason when `qc_status` is not `pass`.

Keep paths relative to a configured data root where possible. Record the checksum or source filename mapping in a separate manifest. Do not publish subject-linked clinical metadata or original MRI files in the repository.

### 3.3 Required dataset audit
Generate `reports/dataset_audit.md` with one row per cohort:

| Dataset | Available subjects | Eligible CN | Eligible AD | Selected T1 scans | QC failures | Scanner breakdown | Data access |
|---|---:|---:|---:|---:|---:|---|---|
| ADNI1 | | | | | | | |
| ADNI2/GO | | | | | | | |
| ADNI3 | | | | | | | |
| AIBL | | | | | | | |

All cells must come from the actual downloaded manifest. The meeting estimate of roughly 8,000 available MRIs and roughly 7,000 after filtering is not a measured cohort count. Also report duplicate subject counts across ADNI phases, diagnosis time differences, scan selection reasons, and missing metadata. Verify the AIBL MRI files, matching clinical metadata, and cohort manifest; keep the planned ADNI-first execution order.

### 3.4 Two experimental protocols
#### Protocol P1: pooled ADNI development with phase-specific testing

1. Combine eligible primary scans from ADNI1, ADNI2/GO, and ADNI3.
2. Create subject-disjoint train/validation/test partitions, initially targeting 70% / 15% / 15% of subjects.
3. Stratify by diagnosis and phase using grouped splitting. If exact proportions cannot satisfy all strata, retain subject separation and report the achieved counts.
4. Freeze the split files before model development.
5. Use train for fitting and validation for hyperparameters, early stopping, temperature scaling, and confidence thresholds.
6. Evaluate on the internal ADNI test set and report metrics separately for ADNI1, ADNI2, and ADNI3 subjects within that test set.
7. Report within-phase held-out results for ADNI1, ADNI2/GO, and ADNI3 from this pooled model. These are phase-specific subgroup evaluations, not cross-phase generalization. In the later external-validation stage, evaluate the same frozen selected model on AIBL; no AIBL data may enter training, feature normalization fit, policy selection, threshold tuning, or hyperparameter search.

#### Protocol P2: cross-phase generalization

1. Train and validate using eligible subjects from ADNI1 and ADNI2/GO.
2. Define the ADNI3 test population as subjects who have no eligible scan in the ADNI1/ADNI2/GO training or validation identity set. Remove overlap at the person level.
3. Keep subject-disjoint training/validation partitions within ADNI1/ADNI2/GO.
4. Evaluate the selected frozen model on this held-out ADNI3 set. Evaluate on AIBL in the later external-validation stage.
5. Report coverage and diagnostic composition after removing cross-phase repeats.

P1 and P2 train **separate checkpoints**. Do not pool their test results into a single metric. Prioritize P1, then P2. For each protocol, select checkpoints without seeing its test labels. For transfer, ensemble the three fold-trained models using the inference rule in Section 5.2; save component results as well.

### 3.5 Data leakage checks
Before running any experiment, execute the following checks:

```python
assert train_subjects.isdisjoint(val_subjects)
assert train_subjects.isdisjoint(test_subjects)
assert val_subjects.isdisjoint(test_subjects)
assert all(subject.dataset == "AIBL" for subject in aibl_external)
assert set(aibl_external_subjects).isdisjoint(adni_subjects)
```

Also check that:

- the same scan is not represented under different filenames or corrected-image variants across partitions;
- longitudinal scans from the same participant never cross partitions;
- any data-driven intensity normalization parameters are fitted only to the training set;
- site and scanner identifiers are excluded from the model inputs;
- the target diagnosis is not indirectly included through the input metadata;
- validation and test caches preserve the original subject-to-label mapping;
- cached features are generated by a backbone trained only with the permitted training subjects.

Store the audit in `reports/leakage_audit_<protocol>.json`. Tests must stop the training pipeline if they fail.

### 3.6 Image preprocessing
Implement the following common preprocessing pipeline in `preprocess.py`:

1. Read DICOM series or NIfTI volumes; for DICOM, retain source series identifiers and conversion logs.
2. Convert DICOM to NIfTI using a fixed converter version when needed.
3. Standardize axis orientation to RAS+ and record the affine before and after conversion.
4. Check voxel spacing, affine determinant, finite intensities, and nonempty image content.
5. Perform affine alignment to a fixed anatomical reference template. Use the same registration settings for all cohorts. Store transforms and registration QC images.
6. Resample into a fixed physical field of view covering the full brain, with a common analysis array target of **96 × 112 × 96 voxels**. Treat these dimensions as the default configuration and verify anatomical coverage before batch processing.
7. Apply robust intensity normalization using the non-background voxel distribution. Use fixed, documented foreground estimation, percentile clipping, and robust scaling. Avoid fitting normalization on test-cohort statistics.
8. Retain the skull and surrounding structures in the stored preprocessed volume. Do not run skull stripping in the main pipeline.
9. Save as compressed NIfTI and write a JSON sidecar with registration, resampling, normalization, and source identifiers.
10. Generate three orthogonal mid-slice QC images for every case, plus an automated registration/coverage report.

Use a fixed field of view and correct spatial metadata. A numerical array resize without spatial alignment is not sufficient for region-level anatomical comparisons.

Apply NeuroVFM-specific preprocessing inside its encoder adapter after the common anatomical preprocessing. Record its input image, affine, crop, resampling, native token centers, template transform, and QC. Never infer token location from the compacted index after background removal.

Do not add computationally expensive preprocessing steps after the first complete working pipeline unless an experiment shows that they are necessary. Preserve one preprocessing version per run. OASIS-3 pilot tokens were extracted from skull-stripped stand-in images; their token counts, AUC values, and blind-policy findings must be tested again under the main skull-retaining ADNI pipeline before becoming main-study claims. AIBL checks follow during the external-validation stage.

### 3.7 Data QC completion criteria
- Every retained scan has a unique scan ID and a stable subject key.
- Every image has a documented label source and diagnosis-to-scan matching rule.
- All retained images have valid affine information and consistent orientation.
- The anatomy lies inside the target field of view; failed cases appear in a review list.
- Train, validation, internal test, and AIBL external populations are documented. Verify the AIBL files and associated metadata.
- Dataset counts match the actual files that can be opened by the dataloader.
- All split and manifest files have SHA-256 checksums and are versioned.

## 4. Encoder-independent representation

### 4.1 Common interface
Every encoder adapter must implement:

```python
from dataclasses import dataclass
import torch

@dataclass
class SpatialEncoding:
    tokens: torch.Tensor       # [B, N, C] spatial tokens
    coordinates: torch.Tensor  # [N, 3] or [B, N, 3], normalized RAS
    valid_mask: torch.Tensor   # [B, N] boolean
    global_feature: torch.Tensor  # [B, Cg]

class SpatialEncoder(torch.nn.Module):
    def encode_spatial(self, volume: torch.Tensor) -> SpatialEncoding:
        """volume: [B, 1, D, H, W]"""
        raise NotImplementedError
```

Keep the extraction logic separate from the selection algorithm. Do not reshape token arrays based on guessed dimensions. Obtain the actual spatial structure from the backbone and verify it with synthetic coordinate tests.

### 4.2 Backbones
**Backbone A: 3D ResNet-18**

- Implement a standard 3D ResNet-18 with four residual stages. Identify a suitable MRI-pretrained ResNet checkpoint from a documented, established source (including Hugging Face if appropriate); report architecture, pretraining population, license, and any overlap risk. Keep a train-from-scratch reference if the pretrained checkpoint is unsuitable.
- Train it on the ADNI training partition for CN/AD classification.
- Use global average pooling and a binary classification head for the full-volume CNN baseline.
- Expose the `layer3` feature map (nominal stride 16) as selection tokens. Retain `layer4` with global pooling for the full-volume CNN baseline. Verify the actual output shape from the implemented model and input volume rather than hard-coding `6 × 7 × 6`.
- After selecting the best CNN checkpoint using validation performance, freeze the feature extractor for all selection-method comparisons.
- Retain the original full-volume CNN checkpoint and predictions as the standard CNN reference.

**Backbone B: NeuroVFM (primary foundation-model encoder)**

- Request and verify access to `mlinslab/neurovfm-encoder` immediately.
- Use the official loading and preprocessing code.
- Freeze the pretrained encoder.
- Extract spatial tokens or feature maps from the visual backbone; do not use only the final pooled embedding.
- Document input voxel size, spatial token shape, any special tokens, token-to-voxel mapping, and the coordinate convention.
- Remove or separate non-spatial special tokens before region pooling.
- Train a global classification head to obtain the full-volume NeuroVFM baseline. Keep this head separate from the TRACE-MRI evidence classifier.
- Record the exact model revision, weight file checksum, and software commit.

NeuroVFM access is gated. Its model repository identifies a ViT-Base backbone with 4 × 16 × 16 input patches and non-commercial research weight licensing. Confirm the operational token-extraction API from the installed version; the model card's high-level `embed()` call alone does not establish access to individual spatial tokens.

**Backbone C: JEPA (selected additional experiments)**

- Identify the specific JEPA-based model, its MRI compatibility, pretrained checkpoint, license, and public loading code before implementation. JEPA is the intended term where the meeting transcript says "Z3" or "Zepa"; it is not a model identifier by itself.
- Verify whether the model exposes spatial tokens, the native token geometry, and whether a local/global-attention comparison is supported.
- Run direct full-representation classification and a small, matched subset of TRACE-MRI experiments only after these checks pass. Use its own adapter and readers, as for other encoders.

**Execution order:** Prioritize NeuroVFM for the full foundation-model experiment matrix, use ResNet as the CNN comparator, and evaluate JEPA on selected scenarios. Implement the method interface with available CNN features while any weight-access request is pending. Never label a substitute model as NeuroVFM or JEPA.

### 4.3 Native spatial tokens as selection units

The selection unit is one native spatial token from the chosen encoder. Scan $i$ has $N_i$ valid candidates. Token index $j$ is local to that scan and does not denote the same anatomy across scans.

**3D CNN.** Use the 3D ResNet-18 `layer3` spatial map as candidate tokens and retain locations whose centers lie within the head mask. The `layer4` map with global pooling remains the full-volume CNN classification path. On the default $96\times112\times96$ analysis array, a standard stride-16 implementation is expected to produce approximately $6\times7\times6$ layer3 locations; measure the exact shape from the implemented model.

**NeuroVFM.** Extract spatial patch tokens using the verified model API and remove non-spatial special tokens. Preserve the dense patch-grid indices before background filtering. The OASIS-3 pilot had a median of about 240 kept tokens per scan, but this is not an ADNI/AIBL specification. Audit token counts and geometry under the main skull-retaining preprocessing. Whole-scan self-attention makes ordinary NeuroVFM tokens context-mixed: each selected embedding may contain information from the entire scan. Token order is not an observation sequence; removing later-index tokens after global encoding does not remove their influence from the remaining embeddings. Inspect the encoder's attention masks and test a genuinely local-only forward pass against the standard global-attention forward pass.

**Token geometry.** For each kept token, record its original grid index, patch footprint, image origin, spacing, direction, crop, affine, and registration transform. Map the token center to physical RAS coordinates, then to the registered template and normalized $[0,1]^3$ coordinates. Verify this chain with synthetic markers and MRI overlays. Do not assign anatomy by compacted token index.

**Adapter output.** Pad each batch to its maximum token count $N_{\max}$ and mask padded positions:

```text
regions:        float32 [B, N_max, 128]
region_coords:  float32 [B, N_max, 3]
valid_regions:  bool    [B, N_max]
global_context: float32 [B, 128]
```

Cache native backbone features before the learned $C_b\rightarrow128$ projection. Train each backbone-specific projection with its partial-evidence reader, then freeze both for all selector comparisons within the same protocol, backbone, token mode, and seed. The downstream architectures are shared across encoder runs; their trained weights need not be shared.

Use `token_mode: context` for the standard globally contextualized encoder tokens. Test `token_mode: local` as a separate representation by restricting attention to each token's local neighbourhood at the feature-extraction stage, or by independently encoding crops/patches if the backbone permits. Verify the receptive field and pretrained-weight compatibility; simply masking tokens at policy time is not a local-attention experiment. Keep the **global overview** $g$ as a distinct pathway from the acquired local evidence, and report its standalone predictive strength. Give each feasible mode its own cache and reader. Use native tokens for the main experiment; investigate custom crop/window size and a fixed 64-cell grid only if time remains. Do not treat token footprints as validated anatomical explanations without coordinate tests.

Budgets of 2, 4, 6, and 8 count proposal slots. Report the actual number of unique tokens revealed, $|S_T|/N_i$, and physical footprint or coverage when comparing token modes or encoders; equal proposal counts do not imply equal spatial coverage.

### 4.4 Caching

Cache one record per scan, protocol, encoder checkpoint, preprocessing version, and token mode. Include the attention implementation and its configuration in the cache identity so local/global outputs cannot be mixed. A logical record is:

```python
{
    "scan_id": str,
    "subject_id": str,
    "dataset": str,
    "phase": str,
    "label": int,
    "features_native": FloatTensor[N_i, C_native],
    "global_native": FloatTensor[C_global],
    "token_grid_index": LongTensor[N_i, 3],
    "region_coords": FloatTensor[N_i, 3],
    "grid_shape": LongTensor[3],
    "image_geometry": dict,  # origin, spacing, direction, affine, crop, registration transform
    "token_mode": str,  # context or local
    "attention_mode": str,  # standard_global or validated_local
    "encoder_id": str,
    "encoder_checkpoint_sha256": str,
    "preprocessing_version": str,
    "token_geometry_version": str,
    "source_manifest_sha256": str,
}
```

Use `cache_root/<protocol>/<encoder>/<token_mode>/<scan_id>.pt` and an index CSV. Record the valid token count and a checksum of token ordering. The batch collator pads tokens and coordinates to $N_{\max}$, with a validity mask that is true exactly for kept tokens. Labels and identifiers may be carried as metadata for training losses and audits; they must never enter the reader or policy features.

Reject a cache if its encoder checkpoint, attention mode, preprocessing version, token geometry, source manifest, or protocol differs from the requested run. The supervised CNN checkpoint is protocol-specific. Do not mix context and local caches or their readers in a selector comparison.

### 4.5 Encoder acceptance tests
- Extract features from a batch of two scans without NaNs or infinite values.
- Check token-to-image coordinates using a synthetic volume with marked locations.
- Confirm that changing a local image region changes nearby spatial representations in a measurable way.
- Confirm that $N_i$ matches the kept spatial tokens, that padding and removed-background positions are never selectable, and that token centers map correctly to the template in synthetic and MRI QC checks.
- Confirm that full-volume predictions remain identical before and after introducing feature caching, up to numeric precision.
- Confirm that CPU/GPU, float32/mixed-precision, and single/batch inference produce compatible outputs.
- Save per-scan encoding time and peak GPU memory for each available backbone. For NeuroVFM, compare standard global attention and local-only attention on the same validation subjects and examine changes in token locality and downstream accuracy.

## 5. Partial-evidence diagnostic model

### 5.1 State and observations
The model maintains a diagnostic belief after observing selected spatial regions.

At the beginning of round $t$, define

$$
S_t=\{j_1,\ldots,j_{m_t}\},\qquad
p_t=P_\theta(Y=1\mid g,Z_{S_t}).
$$

Only features for regions in $S_t$ may enter the evidence-update operation. Every method may access the same global context $g$. The context is produced by encoding the entire volume once, so the observation budget refers to **spatial evidence readout**, not the cost of producing the encoder features.

The full global context can already contain disease signal. Measure and report the initial-context-only performance, and calculate the incremental diagnostic improvement after each selected round. Use exactly the same global context in all selection baselines.

Report initial-context-only performance $p_0$ and random-$k$ curves on validation before policy training. If they approach the full-volume ceiling at small budgets, quantify the remaining headroom. For the separate `g_free` ablation, initialize $h_0$ from a learned constant and remove $g$ from both reader and policy inputs; train a separate reader rather than reusing the context-enabled checkpoint.

Use a shared GRU hidden state $h_t\in\mathbb R^{128}$. Initialize it with a learned projection of the global context:

$$
h_0=\tanh(W_g g+b_g).
$$

For each newly revealed set $A_t$, with one or two regions, transform every region feature and coordinate:

$$
e_j=\phi([z_j,c_j]),\qquad e_j\in\mathbb R^{128}.
$$

Use a permutation-invariant operation to form the new-observation vector:

$$
o_t=\rho\left(\frac{1}{|A_t|}\sum_{j\in A_t}e_j,\ \max_{j\in A_t} e_j,\ |A_t|\right).
$$

Here $\phi$ and $\rho$ are small MLPs. Then update

$$
h_{t+1}=\mathrm{GRUCell}(o_t,h_t),\qquad
p_{t+1}=\sigma(w^\top h_{t+1}+b).
$$

This supports one-region counterfactual updates and two-region round updates with the same parameters. Region order inside a round must not change the output. Region order across rounds remains part of the trajectory.

### 5.2 Train the classifier on partial observations first
Training steps:

1. Freeze the image encoder.
2. Initialize the feature projection, $\phi$, $\rho$, GRU, and classification head.
3. Sample random valid regions without replacement for each training scan.
4. Train with 0, 1, 2, 3, and 4 rounds of observations. Each round reveals one or two regions. Randomize region order across rounds.
5. Compute classification cross-entropy after the initial context and after each update.
6. Weight each valid partial prediction equally within a sampled trajectory, then average across subjects.
7. Select the reader checkpoint by validation AUROC at six observed proposal slots, averaged over a fixed, prespecified set of random trajectories on validation subjects. Use identical trajectory draws for all candidate checkpoints of the same reader. Report negative log-likelihood and balanced accuracy alongside AUROC. Save periodic iteration checkpoints as specified in Section 9.
8. Save the frozen classifier and feature projection for all selector experiments.

**Cross-fitted reader ensemble.** Partition training subjects into $K=3$ disjoint groups $G_1,\ldots,G_K$, stratifying by class and phase where feasible. Train reader $R_k$ and its feature adapter on training subjects outside $G_k$; never use $G_k$ for that reader's fitting or checkpoint selection. Select checkpoints using the separate ADNI validation split, then freeze all readers. During policy training, a scan in $G_k$ uses only $R_k$ for its state, singleton probes, prediction, and reward. Save the subject-to-reader assignment and each checkpoint hash.

At validation and held-out test, maintain one GRU state per frozen reader for the same selected native token indices. Project the cached native features independently through all three reader-specific adapters before a policy rollout. Each reader $R_k$ uses **its own** projected region features $Z^{(k)}$, projected global context $g^{(k)}$, and hidden state $h_t^{(k)}$; never feed an adapter's projected features to another reader. For each role, evaluate its shared policy on each reader-specific state and average candidate logits before masking and action selection. The chosen pair is shared across readers, and every reader updates once using its own features. Average reader probabilities for $p_t$, calibration, stopping, and saved predictions; compute singleton counterfactuals by averaging each reader's independent probe probability. Do not average hidden states. Record per-reader and ensemble results and verify the ensemble rollout on a held-out batch. Use the same inference rule for all selectors. This implements the three-reader ensemble for phase transfer and later external-cohort evaluation; do not retrain readers on the target test cohort.

Loss:

$$
\mathcal L_{\mathrm{evidence}}
= -\frac{1}{M}\sum_{(i,t)\in\mathcal T}
\left[y_i\log p_{it}+(1-y_i)\log(1-p_{it})\right].
$$

Use class-balanced minibatch sampling for training when class imbalance is substantial. Evaluate all metrics on the original test prevalence. Record class weights if any are applied.

### 5.3 Why random subsets are necessary
The classifier must predict from the initial context, one selected region, or a partially observed sequence. The RL policy will generate region combinations the classifier did not encounter during the full-volume CNN training procedure. Training the aggregator over many random subsets provides a common, stable diagnostic reward surface.

Do not train the classifier using only trajectories from the final game policy. The random-subset model is the shared starting checkpoint for the single-agent, cooperative, and competitive experiments.

All selectors within a `(protocol, backbone, token_mode, seed)` setting use the same frozen three-reader cross-fitted ensemble and feature adapters. Different token modes have separate readers. Joint reader updates from the pointer-agent baseline are excluded from the main method; if investigated separately, guard them against degradation on fixed validation random subsets and report them as an ablation.

### 5.4 Diagnostic log-odds
For evidence scoring, use

$$
L_t=\log\frac{p_t}{1-p_t}.
$$

Clip probabilities to `[1e-4, 1 - 1e-4]` for stable logarithms. Retain unrounded probabilities for evaluation and saved predictions.

### 5.5 Aggregator tests
- A single-region update changes the state exactly once.
- A pair update is invariant to swapping the two region indices.
- An empty observation set leaves the state unchanged.
- A region already marked as observed cannot be introduced as new evidence in the next round.
- Individual evidence probes use copies of $h_t$ and do not mutate the ongoing trajectory.
- With a frozen model and fixed input, repeated inference is deterministic.
- Replacing the target label while preserving the same MRI features leaves every inference trajectory unchanged.

## 6. Selection policies and action space

### 6.1 Policy state
At each round, both policies observe:
```text
Global context                g       [128]
Current evidence state        h_t     [128]
Current diagnostic log-odds   L_t     [1]
Normalized round index        t/Tmax  [1]
Region center coordinates     c_ij    [B,N_max,3]
Visited mask                 M_t     [B,N_max]
Valid-region mask            V       [B,N_max]
```

They do **not** receive the features of unselected regions. Those features are revealed only after choosing an action. This rule is essential for a meaningful conditional-evidence experiment.

A policy computes one logit per candidate region:

$$
\ell_{j}^{(r)}=
\pi_r^{\mathrm{MLP}}([g,h_t,L_t,t/T_{\max},c_j]),
$$

where $r\in\{+, -\}$ denotes the role. Use two linear layers with hidden width 128, ReLU or GELU, and a final scalar output. Apply the visited and validity masks before softmax.

### 6.2 Actions
Scan $i$ has $N_i$ valid spatial actions, one per native token. Mask padding and previously selected tokens before the categorical softmax. The policies need no navigation direction, zoom action, or per-step image crop.

At round $t$:

```text
Advocate proposes a_t_plus among unvisited valid tokens.
Challenger proposes a_t_minus among unvisited valid tokens.
Both proposals use the same pre-round state.
The environment reveals the union of the proposed tokens.
The classifier updates once using the newly revealed set.
```

Both policies may select the same token because proposals are simultaneous. A duplicate counts as two proposals but only one unique revealed token. Record the collision and penalize it during training.

### 6.3 Rollout length and budgets
Use the main configuration:
```yaml
max_rounds: 3
proposals_per_round: 2
max_proposals: 6
num_regions: native  # N_i per scan, padded to N_max
action_mode: categorical_region
sampling_train: categorical
sampling_eval: argmax
```

For budget curves, evaluate 1, 2, 3, and 4 rounds, corresponding to 2, 4, 6, and 8 proposal slots. Report the **actual unique regions revealed** alongside proposal slots, because collisions can reduce unique coverage.

When only one valid unvisited token remains, let both policies select it, count the collision, and finish the round. Terminate when no valid unseen regions remain. This edge case must not cause a softmax over an entirely masked action set.

### 6.4 Starting-region experiment
For the initial pilot, test a uniformly random valid first region as the trajectory start and inspect the learned next-region distribution, including whether selections concentrate near the hippocampus. Also retain the V2 default learned initial GRU state without a preselected region as the matched-budget reference. Only if the pilot requires it, compare (i) a learned start-state token that is not an MRI observation and (ii) an atlas-mapped hippocampus region. These are alternative initialization experiments, not three mandatory model branches.

A randomly revealed MRI region counts as **one unique observation and one proposal slot**. When comparing fixed budgets of 2, 4, 6, and 8 proposals, include this slot in the budget and apply the remaining slots consistently across all methods; a remaining single slot uses a documented one-agent selection rule applied consistently to the compared methods. Do not append a free first observation. A learned start token contains no image evidence and consumes no region budget. Log the initial region and its selection rule. Keep the game unchanged for the main default until initialization results justify replacing it.

### 6.5 Policy gradients
During training, sample each action from its categorical softmax distribution. Use independent REINFORCE updates for the two policies, with an exponential-moving-average reward baseline for each role and round.

For player $r$:

$$
\mathcal L_{\pi_r}
=-\frac{1}{BT}\sum_{i,t}
\log\pi_r(a_{it}^{(r)}\mid s_{it})\,
\operatorname{stopgrad}(R_{it}^{(r)}-b_{t}^{(r)})
-\eta\mathcal H(\pi_r).
$$

$R_{it}^{(r)}$ is the discounted return from that round and $b_t^{(r)}$ is the moving-average return for the same role and round. Begin with discount $\gamma=0.95$, entropy coefficient $\eta=0.01$, and baseline EMA momentum `0.9`.

Keep the diagnostic aggregator frozen during this training stage. The selected actions determine its inputs, while policy gradients are computed from the log-probability of those actions. Do not backpropagate through the categorical action index.

## 7. Conditional Evidence Game

### 7.1 The two roles
The current classifier belief defines the working diagnosis:

$$
\widehat y_t=\mathbb 1[p_t\ge0.5],\qquad
s_t=2\widehat y_t-1\in\{-1,+1\}.
$$

- **Advocate:** seeks a region that increases log-odds in the direction of the current working diagnosis.
- **Challenger:** seeks a region that decreases log-odds in that direction.

The roles are defined by the **current prediction**, not by the ground-truth class. If the diagnosis changes after an update, $s_t$ changes for the next round. Use the same two policy networks throughout the trajectory.

### 7.2 Individual conditional evidence
At round $t$, before revealing either proposal, compute the singleton counterfactual log-odds:

$$
L_t(a)=L\big(\operatorname{Update}(h_t,\{a\})\big).
$$

Then define:

$$
\Delta_t(a)=L_t(a)-L_t.
$$

This is the diagnostic belief change caused by adding that region to the **current set of observed model inputs**. It can be evaluated during training because the environment has the cached full feature grid. The policies cannot access the unselected features while making their decisions.

### 7.3 Joint diagnostic reward
Let $A_t=\{a_t^+,a_t^-\}$ be the union of the proposed regions. Reveal $A_t$, update the classifier, and compute

$$
G_t
=\log P_\theta(y\mid S_t\cup A_t,g)
-\log P_\theta(y\mid S_t,g).
$$

The true label appears only in the **training reward**. It is never available to the selection policies or inference model.

Positive $G_t$ means that the selected evidence increased the probability assigned to the true diagnosis.

### 7.4 Game payoffs
Use the following two-player general-sum stage game:

$$
\boxed{
\begin{aligned}
u_t^+ &= G_t + \alpha s_t\Delta_t(a_t^+) - \lambda C_t,\\
u_t^- &= G_t - \alpha s_t\Delta_t(a_t^-) - \lambda C_t
\end{aligned}}
$$

where

$$
C_t=\mathbb 1[a_t^+=a_t^-].
$$

The shared term $G_t$ rewards both players for acquiring diagnostically useful evidence. The signed terms create opposing search preferences. The collision term discourages duplicate proposals.

The payoffs depend on both actions through the joint diagnostic gain and the collision indicator. Independent best responses are coupled through the shared environment and the other player's choice. We will train approximate policies with alternating updates; no equilibrium solver is required.

### 7.5 Reward computation
Use stable log probabilities and clamp diagnostic log-odds changes for reward stability:

```python
def binary_log_prob(p_ad, y, eps=1e-4):
    p = p_ad.clamp(eps, 1.0 - eps)
    return y * p.log() + (1.0 - y) * (1.0 - p).log()

def compute_round_rewards(
    p_before, p_after, p_after_plus, p_after_minus,
    y, action_plus, action_minus,
    alpha=0.10, collision_weight=0.05,
):
    eps = 1e-4
    def logit(p):
        p = p.clamp(eps, 1.0 - eps)
        return p.log() - (1.0 - p).log()

    s = torch.where(p_before >= 0.5, 1.0, -1.0)
    joint_gain = binary_log_prob(p_after, y) - binary_log_prob(p_before, y)
    delta_plus = (logit(p_after_plus) - logit(p_before)).clamp(-4.0, 4.0)
    delta_minus = (logit(p_after_minus) - logit(p_before)).clamp(-4.0, 4.0)
    collision = (action_plus == action_minus).float()

    reward_plus = joint_gain + alpha * s * delta_plus - collision_weight * collision
    reward_minus = joint_gain - alpha * s * delta_minus - collision_weight * collision
    return reward_plus, reward_minus, joint_gain, delta_plus, delta_minus
```

`p_after_plus` and `p_after_minus` are computed from **independent counterfactual copies of the same pre-round state**. `p_after` is computed after the union of both actions is revealed. Do not apply the singleton updates to the live recurrent state.

Start with `alpha=0.10` and `collision_weight=0.05`. Evaluate `alpha` in `{0.00, 0.05, 0.10, 0.20}` on validation only. Keep all other hyperparameters fixed when measuring this sensitivity. Log the magnitudes of the joint and signed reward terms so that one term does not dominate unnoticed.

### 7.6 Alternating policy updates
Use the following procedure:

1. Run complete trajectories with the current Advocate and Challenger policies.
2. Store each role's sampled actions, log-probabilities, entropy, rewards, and masks.
3. Compute discounted returns for both roles.
4. Update the Advocate policy from its own returns while keeping the Challenger parameters fixed during that update.
5. Update the Challenger policy from its own returns while keeping the Advocate parameters fixed during that update.
6. Update EMA reward baselines; repeat.

Keep the environment and evidence classifier fixed during the policy phase. Train with separate optimizers for each policy. Checkpoint both policies together, including their optimizer states and the shared classifier version.

### 7.7 Trajectory algorithm

The pseudocode below shows a training rollout for one subject and its assigned out-of-fold reader. At validation and test, apply the ensemble action and probability rules in Section 5.2, maintaining a separate state for every reader.

```python
# Pseudocode. All classifier state probes operate on copies.

def trace_rollout(regions, coords, global_context, valid_regions, y=None,
                  max_rounds=3, deterministic=False):
    h = aggregator.initial_state(global_context)
    visited = torch.zeros(regions.shape[0], dtype=torch.bool, device=regions.device)
    valid = valid_regions.clone()
    events = []

    for t in range(max_rounds):
        available = valid & ~visited
        if available.sum() == 0:
            break

        p_before = aggregator.predict(h)
        state = build_policy_state(h, global_context, p_before,
                                   coords, visited, t, max_rounds)

        # Both policies receive the same pre-round evidence state.
        a_plus, logp_plus, ent_plus = advocate.act(
            state, available, deterministic=deterministic
        )
        a_minus, logp_minus, ent_minus = challenger.act(
            state, available, deterministic=deterministic
        )

        # Independent probes, without modifying h.
        p_plus = aggregator.predict(aggregator.update(h.clone(), [a_plus], regions, coords))
        p_minus = aggregator.predict(aggregator.update(h.clone(), [a_minus], regions, coords))

        chosen = sorted(set([int(a_plus), int(a_minus)]))
        h_next = aggregator.update(h, chosen, regions, coords)
        p_after = aggregator.predict(h_next)

        # Rewards are computed only when training labels are available.
        if y is not None:
            rewards = compute_round_rewards(
                p_before, p_after, p_plus, p_minus, y, a_plus, a_minus
            )
        else:
            rewards = None

        events.append(record_event(t, a_plus, a_minus, p_before,
                                   p_plus, p_minus, p_after, rewards))
        visited[chosen] = True
        h = h_next

    return aggregator.predict(h), events
```

Implement the tensor-batched version for training. The pseudocode above specifies behaviour and update order; it is not a replacement for tests.

### 7.8 Early stopping
First train and evaluate all policies at fixed budgets. Then add confidence-based stopping without introducing a new action or policy network.

After each round with at least two proposed observations, stop when

$$
\max(p_t,1-p_t)\ge \tau,
$$

or when the maximum round budget is reached. Fit one temperature-scaling parameter on the ADNI validation set before choosing $\tau$. Evaluate `tau` in `{0.85, 0.90, 0.95}` on validation and report coverage, mean observed regions, accuracy, and class-specific error rates. Freeze the chosen threshold before any test-set evaluation.

If the threshold does not provide an acceptable accuracy/observation trade-off on validation, use the fixed-budget setting as the main result.

### 7.9 Game tests
Write tests verifying that:

- both actions are chosen from the same pre-round state;
- a previously visited region has zero action probability;
- identical simultaneous actions yield exactly one newly revealed region and incur the collision penalty;
- reward for each player changes when the other player's action changes the joint diagnostic outcome;
- swapping player identities and payoffs behaves as specified;
- changing `alpha` to zero removes the signed evidence terms;
- changing the target label changes training rewards but not test-time actions;
- no singleton counterfactual probe modifies the live recurrent state;
- all policy outputs and gradients remain finite;
- two repeated seeded runs yield identical sampled actions and metrics within the documented determinism settings.

## 8. Baselines and fair comparison

Run each main selection method on **NeuroVFM and ResNet**. Add a selected JEPA subset when technically feasible. Within a protocol, backbone, token mode, and seed, use the same cached token features and frozen cross-fitted feature-adapter/reader ensemble for all selectors.

### 8.1 Full-volume CNN
Standard 3D ResNet-18 with global pooling and a supervised CN/AD classification head. Use the same training partition and input preprocessing. Record all classification metrics.

### 8.2 Full-volume NeuroVFM and attention variants
Frozen NeuroVFM plus a trained global classification head. Compare a linear head and a small MLP, selecting one using validation results. Run (i) standard full/global attention and (ii) an independently validated local-only attention variant with direct classification; report the distinction in feature access and diagnostic performance. The local-only variant must modify feature extraction, not merely hide later-selected features from the policy. Do not use a public disease-specific diagnostic head trained on an unknown overlapping evaluation population as the primary reference. Run corresponding direct-classification baselines for a suitable JEPA model when its interface supports the comparison.

### 8.2.1 JEPA direct-classification baseline
Use the identified JEPA checkpoint to extract a global representation and fit a small logistic/linear or MLP classifier. Verify spatial-token extraction before running any JEPA-based trajectory experiment. Record cases where restricted-attention extraction is unsupported.

### 8.3 Random region selection
Uniformly sample valid unvisited tokens. Use the same proposal budget, reader ensemble, and deterministic seed schedule. No policy learning.

Use at least 10 independent random draws per setting. Save their seeds and report the mean and variability. Match subjects, reader ensemble, proposal slots, and, for stopping analyses, the actual number of observed tokens so that a weak random draw cannot create a spurious gain.

### 8.4 Static Top-K selection
Train a small scorer from global context and coordinates to rank all valid native tokens once per scan. Keep the ranking fixed after evidence is revealed, and select the highest-ranked available tokens up to the budget. Use policy-gradient training with the same reader-based task reward.
For a two-proposal round, take the two highest-ranked available tokens. When only one remains, both proposals select it and the collision rule applies.

### 8.5 Single-agent sequential RL
Use one categorical policy with the same state and network width as each TRACE-MRI player. For every round, sample two distinct region indices from its current score distribution, without revealing the first region before choosing the second. Reveal the pair and update the classifier. This matches the two-proposal cadence and information timing of TRACE-MRI.

Train with the common diagnostic reward $G_t$. For this baseline, no supporting/challenging role terms are used.

### 8.6 Two-agent cooperative RL
Use the same two policy networks and simultaneous selection protocol as TRACE-MRI. Give both players the common reward

$$
u_t^{\mathrm{coop}}=G_t-\lambda C_t.
$$

This is the **most important baseline for H2**. It matches the number of policies, simultaneous actions, collision handling, trajectory length, and classifier.

### 8.7 TRACE-MRI game
Use the payoffs from Section 7. Keep the policy architectures, budgets, optimizer schedules, and classifier identical to the cooperative setting.

### 8.8 Strong non-RL comparison
After the core methods work, implement a differentiable attention or Gumbel-Top-K selector with the same feature-access constraints. Use it to assess whether categorical RL contributes beyond a learned, gradient-based region-ranking mechanism.

Also evaluate fixed-location search, a size-only baseline, an evidence-blind policy, and a geometry-only policy. Find fixed locations using training and validation subjects only. The size-only baseline uses token count and coordinate extent without embeddings. The evidence-blind control retains scan-specific $g$, coordinates, and masks but withholds updates from revealed evidence; the geometry-only control also removes scan-specific diagnostic state and context while retaining candidate geometry. These controls use the same reader ensemble and proposal budget as the corresponding selector. Geometry-only selections may still vary across scans; only fixed-location search directly tests one location sequence.

### 8.9 Comparison rules
- Use identical subject splits and labels for every method within a protocol.
- Use the same frozen encoder cache for all selection methods using that backbone.
- Use the same frozen cross-fitted reader ensemble for the policy comparisons.
- Match the number of proposal slots and report actual unique observations.
- Use the same validation metric and comparable training budgets for policy selection.
- Tune hyperparameters exclusively on ADNI validation subjects.
- Report parameter count and training/inference time for the selection and aggregator layers separately from backbone encoding.
- Report full-volume baselines in separate rows because they use all image features.
- Report performance with and without stopping as separate experiments.

## 9. Training configuration

Use PyTorch, MONAI where appropriate for volumetric medical imaging utilities, NiBabel, NumPy, pandas, scikit-learn, and a standard experiment logger. Pin the working versions in `requirements.lock` after testing the environment. Use a single main configuration file to override experiment-specific values.

### 9.1 Initial configuration
```yaml
project:
  name: TRACE-MRI
  seed: 0
  output_root: outputs
  device: cuda
  precision: mixed

experiment:
  protocol: adni_pooled  # also: adni_transfer
  task: CN_vs_AD
  num_classes: 2
  external_test: AIBL  # evaluate after primary ADNI experiments

data:
  raw_root: /path/to/authorized/data
  processed_root: /path/to/processed/data
  cache_root: /path/to/feature/cache
  canonical_orientation: RAS
  analysis_shape: [96, 112, 96]
  skull_stripping: false
  primary_scan_per_subject: 1

encoder:
  name: neurovfm  # also resnet3d18; JEPA for selected feasible runs
  frozen_for_selector: true
  spatial_units: native_tokens
  token_mode: context  # compare with validated local-only attention
  global_context_mode: enabled  # g_free is a separate ablation
  common_feature_dim: 128
  global_feature_dim: 128

aggregator:
  crossfit_folds: 3
  ensemble_action_rule: mean_policy_logits
  ensemble_probability_rule: mean_reader_probabilities
  hidden_dim: 128
  region_mlp_hidden: 128
  initial_observation_rounds: 0
  max_training_rounds: 4
  partial_observation_loss: cross_entropy

agent:
  num_agents: 2
  policy_hidden_dim: 128
  action_count: native  # N_i per scan
  max_rounds: 3
  simultaneous_proposals: true
  train_sampling: categorical
  eval_sampling: argmax
  discount: 0.95
  entropy_coefficient: 0.01
  reward_baseline_momentum: 0.90

reward:
  alpha: 0.10
  collision_weight: 0.05
  max_abs_logit_change: 4.0

training:
  cnn:
    epochs: 80
    batch_size: 8
    lr: 0.0003
    weight_decay: 0.0001
    patience: 12
  aggregator:
    epochs: 50
    batch_size: 64
    lr: 0.001
    weight_decay: 0.0001
    patience: 10
  selector:
    epochs: 80
    batch_size: 64
    lr: 0.0003
    weight_decay: 0.0001
    patience: 15
    grad_clip_norm: 1.0

selection:
  checkpoint_metric: validation_auc_budget6
  random_draws: 10
  repeat_seeds: [0, 1, 2, 3, 4, 5]  # start with 0,1,2; extend in order
  budgets: [2, 4, 6, 8]
  early_stopping_thresholds: [0.85, 0.90, 0.95]

checkpointing:
  interval_iterations: 500  # configurable; measure throughput before fixing interval
  save_last: true
  save_best_auc: true
  save_optimizer_and_rng: true

initialization:
  main: learned_gru_state
  pilot: random_first_region  # count it in the proposal budget

compute:
  backend: cscs_slurm
  permit_independent_jobs_in_parallel: true
```

These are initial settings for implementation and controlled tuning, not established optimal hyperparameters. Keep the same selected settings for the matched method comparison whenever possible. Document any backbone-specific change.

The cross-fitting count, token mode, global-context mode, ensemble action rule, and random-draw count are explicit configuration fields. Use `attention_mode: standard_global` for the main encoder features, and `attention_mode: validated_local` only after the restricted-attention forward pass and spatial tests work. Save cache and reader-assignment hashes with each run. Select classification policy checkpoints by **validation AUROC at six proposal slots**, applying the same prespecified metric to all learned selectors; also record validation NLL, balanced accuracy, common diagnostic return, and role-specific returns. Save checkpoints after configurable numbers of optimizer iterations and at the best validation AUROC, including optimizer, scheduler, step, RNG state, reader hashes, and configurations. Never choose a metric after inspecting test results.

### 9.2 Training phases
**Phase A: Supervised CNN backbone**

- Train the 3D ResNet-18 CN/AD baseline.
- Use AdamW, early stopping, mixed precision, and validation monitoring.
- Use moderate training-only intensity perturbations and small spatial rotations if they preserve valid anatomy. Do not apply arbitrary left-right flipping in the default protocol.
- Save the best backbone and head, learning curves, and per-subject validation predictions.

**Phase B: Feature caching**

- Freeze the selected ResNet, NeuroVFM, and, where feasible, JEPA encoder features.
- Extract native spatial features and global features for every available scan.
- Validate feature geometry, model checkpoint hash, and sample association.
- Record throughput, peak GPU RAM, output tensor shape, and cache size.

**Phase C: Cross-fitted evidence readers**
- Train three out-of-fold feature-adapter and GRU-reader pairs on random partial-observation sequences for each `(protocol, backbone, token_mode)` setting. Audit subject-to-reader routing.
- Freeze all readers and projections for selectors within the setting.
- Report $p_0$, random-$k$ curves, token counts, and spatial variation in single-token signal before policy training. Compare NeuroVFM standard global attention with a validated local-only extraction; record feasibility and the initial random-start pilot.

**Phase D: Selection baselines**

- Run random region selection first.
- Train static Top-K.
- Train single-agent RL.
- Train cooperative two-agent RL.
- Run every method at budgets 2, 4, 6, and 8.

**Phase E: TRACE-MRI**

- Initialize two policies with independent parameters and the same network structure as cooperative RL.
- Train with the game payoffs and alternating updates.
- Monitor the common diagnostic reward, two role-specific reward terms, collisions, trajectory diversity, and diagnostic validation metrics.
- Save the full training history, including degenerate or collapsed policies.

**Phase F: Calibration and final evaluation**

- Select models from the validation set.
- Calibrate probabilities and select optional stopping thresholds using validation data only.
- Freeze every selected checkpoint and configuration.
- Generate predictions on internal ADNI test, phase-specific subsets, and cross-phase ADNI3 as required by protocol; evaluate AIBL in the subsequent external-validation stage.
- Generate statistical analyses and paper tables from saved prediction files.

### 9.3 Experiment tracking
Every run must save:

```text
run_id
created_at
code_commit
config_yaml
protocol
train_manifest_hash
validation_manifest_hash
test_manifest_hash
backbone_name
backbone_checkpoint_hash
reader_ensemble_hash
reader_fold_map_hash
cache_index_hash
token_mode
global_context_mode
policy_checkpoint_hash
attention_mode
initialization_mode
checkpoint_iteration
checkpoint_metric_value
seed
training_log
validation_metrics
best_epoch
checkpoint_path
predictions_path
trajectory_path
```

The registry must also record `spatial_unit`, `reader_ensemble_rule`, `checkpoint_selection_metric`, `checkpoint_iteration`, `random_draws`, `attention_mode`, `initialization_mode`, CSCS job ID, and whether a result is a pilot or main-study result.

The `experiments/registry.csv` should contain a row for every run. Track successive iteration checkpoints under the parent run ID; never mix a checkpoint from another fold or attention mode. Do not overwrite a failed run with a new configuration under the same ID.

## 10. Evaluation metrics and analysis

### 10.1 Classification metrics
Primary metrics:

- AUROC;
- balanced accuracy;
- sensitivity and specificity at a threshold selected from ADNI validation;
- AUPRC with the positive class fixed as AD;
- negative log-likelihood and Brier score;
- expected calibration error (ECE) using a declared binning procedure.

Report the number of CN and AD subjects next to each test result. Use subject-level predictions. For every model, save the raw probability of AD, true label, and selected threshold.

### 10.2 Evidence-budget metrics
For each method and budget, report:

- performance at the stated proposal budget;
- mean and standard deviation of unique observed regions;
- area under the AUROC-versus-budget curve, with the numerical integration rule stated;
- mean absolute and signed conditional log-odds change per round;
- changes in true-class log-probability with each additional round;
- median and mean number of observed regions before confidence stopping;
- total test-time feature extraction and policy/aggregator runtime separately.

The main figure will plot **AUROC and balanced accuracy versus proposal budget** for the matched selection methods.

Plot $p_0$ alongside random-$k$ and feasible context/local conditions. Report paired A0-minus-A2, agent-minus-evidence-blind, and agent-minus-geometry-only comparisons. Across encoders or token modes, add unique-token fraction and physical footprint to the proposal-slot budget.

### 10.3 Game-specific metrics
Compute:

**Collision rate**

$$
\operatorname{CollisionRate}=
\frac{\#\{(i,t):a_{it}^+=a_{it}^-\}}
{\#\text{ evaluated rounds}}.
$$

**Role-specific signed evidence**

$$
E^+=\mathbb E[s_t\Delta_t(a_t^+)],\qquad
E^-=\mathbb E[-s_t\Delta_t(a_t^-)].
$$

**Unique-region coverage**
$\operatorname{Coverage}_i=|S_{T,i}|/N_i$.
**Region-selection entropy and frequency:** bin token coordinates into fixed template-space bins by class and phase. Do not aggregate by token index, which does not denote the same anatomy across scans. State the bin size and measure coordinate-based overlap.

**Evidence contribution:** recompute the final prediction after removing one selected region from the recorded sequence and replaying the remaining observations in their original order. Compare the final log-odds to the original trajectory. This is a model sensitivity measure. Report its relationship to the original signed step contribution.

**Empirical one-step best-response check:** for a small validation subset, hold one agent's proposed action fixed and enumerate the other agent's valid alternatives using the frozen aggregator. Measure the difference between the chosen payoff and the best attainable payoff at that state. This describes local game behaviour; do not claim an equilibrium from low one-step regret alone.

### 10.4 Statistical analysis
- Use consecutive seeds `0, 1, 2, ...` for repeated runs. Start with `0, 1, 2`; extend to `3, 4, 5` as resources permit, prioritizing full TRACE-MRI, cooperative two-agent, and single-agent RL. Do not infer that the six listed seeds require six runs per ablation.
- Report the mean and standard deviation across seeds, and individual seed results in supplementary material.
- Generate 95% subject-level bootstrap confidence intervals with at least 2,000 resamples for AUROC and balanced accuracy. Resample within the evaluated cohort; preserve class representation by using a stratified subject bootstrap when needed.
- For paired comparisons, resample the **same subjects** across methods and compute the paired difference in metrics.
- Use test-set predictions from models fixed in advance. Do not select the best random seed using test results.
- Report actual improvements and uncertainty, including nonsignificant or negative differences.

### 10.5 Cross-cohort evaluation
Use the exact same trained model for every target cohort within a protocol. On AIBL, during the subsequent external-validation stage:

- Apply the fixed preprocessing procedure and feature extraction.
- Use the already fitted feature adapter, aggregator, and policies.
- Apply the validation-selected classification threshold and stopping threshold.
- Do not adjust normalization using AIBL class labels or diagnostic outcomes.
- Report results for the full eligible external cohort and note scanner/protocol differences when available.

If one cohort has a substantially different class distribution, retain AUROC and balanced accuracy as main comparisons and report AUPRC with class prevalence.

## 11. Required ablations

Complete these in the listed order. All comparisons within an encoder/token-mode setting use the same split, cache, frozen reader ensemble, proposal budget, and evaluation rule.

| ID | Ablation | Exact change | Question |
|---|---|---|---|
| A0 | Full TRACE-MRI | Two-policy conditional evidence game | Main method |
| A1 | Single agent | One policy proposes both tokens; common reward | Value of two policies |
| A2 | Cooperative game | Set `alpha = 0`; retain two policies and collision term | Value of opposing evidence preferences |
| A3 | No collision term | Set `collision_weight = 0` | Value of discouraging overlap |
| A4 | Static selection | Rank valid tokens once from context and coordinates | Value of conditioning on acquired evidence |
| A5a | Evidence-blind policy | Retain $g$, coordinates, and masks; replace evidence-dependent state by its pre-observation state in the policy only | Value of reacting to revealed evidence |
| A5b | Geometry-only policy | Also remove scan-specific diagnostic context/state from the policy; retain coordinates and valid mask | Value of diagnostic conditioning beyond geometry |
| A6 | Fixed vs adaptive budget | Compare fixed budgets with validation-calibrated stopping | Reliability of early stopping |
| A7 | Role direction | Swap support/challenge signs or give both the same sign | Sensitivity to role objectives |
| A8 | Reward coefficient | `alpha = 0, .05, .10, .20` | Stability of the game effect |
| A9 | Encoder | ResNet vs NeuroVFM, with selected JEPA comparisons if feasible | Encoder independence |
| A10 | Context vs local tokens | NeuroVFM standard global vs validated local-only attention, with separate caches/readers on matched splits and budgets | Effect of context mixing |
| A11 | Initial state | Default state versus random first observed token; test learned start token or hippocampus start only if needed | Starting-region effects |
| A12 | Region granularity | Native tokens vs optional custom crops or fixed 64-cell grid if feasible | Choice of observation scale |

Keep the reader unchanged for A5a and A5b so the contrast isolates information available to the policy. Neither control receives labels at inference. Candidate coordinates and masks may remain scan-specific, so a geometry-only policy is not necessarily a fixed sequence; use the separate fixed-location baseline for that claim. For A7, keep all other training settings fixed.

Ablation tables must include dataset, backbone, token mode, budget, AUROC, balanced accuracy, and unique observed tokens. Save a paired A0-versus-A2 table as the central game result.

## 12. Qualitative evidence trajectories

### 12.1 Required examples
Select examples using a declared, label-independent or error-category-based rule after evaluation, without changing the model:

1. Correct AD classification with increasing confidence.
2. Correct CN classification with competing evidence and confidence stabilization.
3. Case where the Challenger finds evidence that changes the working diagnosis.
4. Incorrect prediction or unstable trajectory.
5. A held-out ADNI phase case; add an AIBL case from the later external evaluation.

Use actual selected coordinates, predictions, and evidence values. Do not redraw approximate paths by hand.

### 12.2 Figure contents
Each trajectory figure must contain:

- MRI anatomical slices in the fixed coordinate system;
- region boxes or centers projected into axial, coronal, and sagittal views as appropriate;
- distinct role colors with accessible legend labels;
- round numbers and direction of movement in evidence space;
- probability of AD after each round;
- positive and negative conditional log-odds changes;
- true label and final predicted class;
- backbone and dataset identifier without personally identifying data.

If anatomical region names are displayed, derive them from a documented atlas overlay and registration transform. Anatomical overlays are explanatory visualization metadata and must not provide the policies with predetermined region labels.

### 12.3 Figure acceptance checks
- All region centers map correctly to the MRI volume.
- The plotted trajectory exactly matches the saved `trajectory.json`.
- Re-running figure generation does not change selected examples or results.
- The role colors and evidence signs are consistent between panels.
- Captions distinguish learned model evidence from clinical pathology or causal attribution.

## 13. Output files and reporting format

### 13.1 Per-subject predictions
For every method, backbone, seed, dataset split, and budget, save:

```csv
run_id,protocol,backbone,token_mode,attention_mode,global_context_mode,initialization_mode,method,policy_control,seed,dataset,phase,subject_id,scan_id,true_label,p_ad,predicted_label,decision_threshold,proposal_budget,proposals_used,unique_regions_used,n_valid_tokens,coverage_fraction,cache_hash,reader_ensemble_hash,checkpoint_iteration,stopped_early,stop_round,correct
```

Preserve the existing column names and meanings for compatibility with earlier result files. Add the new columns through a documented migration script. `reader_ensemble_hash` identifies the ordered set of reader checkpoints and its action/probability aggregation rules; for a single-reader pilot it identifies that reader and rule.

### 13.2 Per-step trajectories
Use JSONL, one record per scan:

```json
{
  "run_id": "example-run",
  "subject_id": "deidentified-subject",
  "scan_id": "deidentified-scan",
  "true_label": 1,
  "token_mode": "context",
  "attention_mode": "standard_global",
  "initialization_mode": "learned_gru_state",
  "n_valid_tokens": 42,
  "reader_ensemble_rule": "mean_policy_logits_then_mean_reader_probabilities",
  "initial_p_ad": null,
  "events": [
    {
      "round": 0,
      "advocate_region": 12,
      "challenger_region": 33,
      "advocate_token_grid_index": [1, 2, 3],
      "challenger_token_grid_index": [2, 3, 4],
      "advocate_center_ras_mm": [12.0, -15.0, 24.0],
      "challenger_center_ras_mm": [18.0, -9.0, 30.0],
      "advocate_center_template": [0.4, 0.5, 0.6],
      "challenger_center_template": [0.5, 0.6, 0.7],
      "proposal_slots_used": 2,
      "selected_unique_regions": [12, 33],
      "p_before": null,
      "p_after_advocate_singleton": null,
      "p_after_challenger_singleton": null,
      "p_after_joint": null,
      "advocate_conditional_delta_logit": null,
      "challenger_conditional_delta_logit": null,
      "collision": false
    }
  ],
  "final_p_ad": null,
  "stopped_early": false
}
```

The `null` values above are schema examples. Replace them with measured floating-point values in actual output. Inference records do not require the true class except for the evaluation export; keep inference-only APIs label-free.

The `*_region` indices above are native token indices in the valid-token list; keep these existing field names for compatibility. Record each token's native grid index, physical RAS center, and registered template center. For training rollouts, also record the assigned out-of-fold reader and its fold. For validation and test, record the reader ensemble rule and checkpoint hashes. Count both proposal slots even when the two roles select the same token; store each singleton probe and the prediction after the joint update. The inference API takes no label; label-dependent rewards belong in separate training logs.

### 13.3 Required result tables
**Table 1:** Dataset and cohort characteristics, including subject counts, labels, scanner/protocol distribution, and split sizes.

**Table 2:** Main P1 comparisons, prioritizing NeuroVFM and reporting ResNet separately; include ADNI internal test and phase-specific test results. Add AIBL external results from the subsequent evaluation stage.

**Table 3:** P2 cross-phase transfer, including ADNI3 held-out results for frozen models; add AIBL results from the subsequent evaluation stage. Report three-reader ensembles and component-reader results.

**Table 4:** Game ablations, especially full TRACE-MRI versus the matched cooperative two-agent control, with validation AUROC used for checkpoint selection.

Tables 2–4 must identify `token_mode`, `attention_mode`, and `initialization_mode` and include initial-context-only ($p_0$), random-$k$, evidence-blind, and position-only controls where applicable. Base the main A0-versus-A2 claim on paired subject differences across repeated seeds. Report any OASIS pilot separately from the ADNI/AIBL main study.

**Supplementary:** Seed-level results, hyperparameters, budget curves, stopping thresholds, calibration, data QC, and preprocessing details.

### 13.4 Required figures
1. Architecture and conditional evidence game, showing exactly which features are visible to the policies.
2. AUROC/balanced accuracy versus selected-region budget, prioritizing NeuroVFM and showing ResNet and feasible JEPA results separately.
3. Representative trajectories over real MRI slices, including at least one challenge-induced prediction change if observed.
4. Supplemental heatmaps of region-selection frequency by diagnosis and cohort.
5. Supplemental learning curves and ablation plots.

## 14. Quality control, reproducibility, and unit tests

### 14.1 Unit tests
**Data and subjects**

- Stable subject identifiers across phases.
- Exact CN/AD label mapping from documented fields.
- Scan-to-visit matching and unique primary scan selection.
- Disjoint train/validation/test identities.
- AIBL entirely outside development.

**Imaging and spatial coordinates**

- RAS orientation and affine consistency.
- Reproducible image resampling and voxel intensities.
- Verify the native grid index → physical RAS → template center mapping with synthetic markers and image QC.
- Verify token indices lie in `[0, N_i-1]` and that padding or removed-background tokens cannot be selected.
- Check for label leakage, correct scan-to-out-of-fold-reader routing during policy training, and the specified reader-ensemble rule at validation and test.

**Evidence model**

- Pair-update permutation invariance.
- Correct singleton evidence change.
- No update when nothing new is revealed.
- Finite log-probabilities for all valid predictions.

**Policies and game**

- Action masks remove all previously revealed native tokens and all invalid or padded positions.
- No test-time access to ground-truth labels.
- Both policies act from the same pre-round state.
- Collisions are counted and penalized once for each role.
- Game rewards match direct manual calculations on small synthetic examples.
- Gradients affect policy parameters and do not modify the frozen encoder or classifier.
- Greedy evaluation is deterministic.

**Evaluation**

- AUROC, AUPRC, and sensitivity/specificity calculations match scikit-learn on a reference array.
- Per-subject metric calculations use one prediction per subject in the primary analysis.
- Bootstrap samples and paired differences use matched subject indices.
- Temperature scaling and thresholds are fitted only on validation.
- No output aggregation accidentally averages probabilities across different subjects.

### 14.2 Required automated checks
Run the test suite before merging:

```bash
python3 -m pytest tests/ -q
python3 scripts/10_validate_submission.py \
  --experiment-registry experiments/registry.csv \
  --check-splits --check-checkpoints --check-predictions
```

Configure continuous integration for fast unit tests using synthetic arrays. Data-access-dependent integration tests should run on the research server under the approved storage location.

### 14.3 Reproducibility manifest
At the end of each working day, update:

```text
Current commit:
Data manifest and split hashes:
Completed experiments:
Failed experiments and causes:
Latest reproducible metrics:
Unresolved blockers:
Next experiment:
```

Keep one canonical results registry. Figures and manuscript tables must be generated directly from saved predictions and registry entries.

## 15. Schedule and weekly deliverables

The fixed submission deadline is 16 November 2026 AoE. The 14–20 October items are retrospective milestones: mark each as completed, delayed, or blocked in the task tracker. From 21 October onward, submit every independent feasible CSCS job without waiting for the complete architecture.

### 14–17 October: data and encoder feasibility

- [ ] Verify ADNI1/2/3 files and clinical data; check the AIBL MRI files and match the clinical metadata.
- [ ] Build the common metadata/diagnosis manifest.
- [ ] Audit cross-phase repeated subjects and create P1/P2 split files.
- [ ] Run preprocessing and anatomical QC on a small stratified sample.
- [ ] Implement and unit-test the 3D CNN feature interface.
- [ ] Request/verify NeuroVFM weights and inspect spatial-token extraction.
- [ ] Finalize the native-token action space, geometry and crop transforms, `token_mode`, and cache schema; verify two scans with different valid token counts. Record this item as complete or overdue against the actual work.

**Delivery by 17 October:** `dataset_audit.md`, split manifests, spatial-adapter tests, NeuroVFM access report, and a working two-scan encoding demonstration.

### 18–22 October: supervised baselines and diagnostic aggregator

- [ ] Train the full-volume CNN baseline and evaluate available MRI-pretrained ResNet weights.
- [ ] Extract CNN spatial features for the development set.
- [ ] Extract NeuroVFM features if weights are operational. In parallel, test standard/global versus local-only attention and the JEPA token interface; do not block the NeuroVFM main run on JEPA.
- [ ] Train and audit all three cross-fitted GRU readers for each primary backbone and feasible token mode. Prioritize the NeuroVFM configuration.
- [ ] Produce random-region performance curves by observation budget.
- [ ] Test all subject-level leakage and cache integrity checks.

**Delivery by 22 October (update actual completion status):** reproducible full-volume and random-evidence baseline metrics; three-reader checkpoints where completed; complete cache schema. Report any outstanding tasks in the tracker rather than silently changing the deadline.

At Gate 2, report $p_0$, random-$k$ with at least 10 draws, spatial variation in single-token signal, valid token counts per scan, validation AUROC and NLL, and geometry-test results. If local-token mode lacks spatially varying signal or leaves insufficient headroom, document the decision to proceed or redesign it before policy training.

### 23–27 October: simple selectors

- [ ] Implement static Top-K scoring.
- [ ] Implement single-agent discrete RL.
- [ ] Match two proposal slots per round and the same frozen classifier.
- [ ] Check action masking, rollouts, finite rewards, and gradient flow.
- [ ] Generate the first genuine trajectories over MRI slices.

**Delivery by 27 October:** single-agent and static-selection comparisons at budgets 2, 4, 6, and 8; trajectory visualization from saved outputs.

### 28 October–2 November: two-agent evidence game

- [ ] Implement two-player simultaneous selection.
- [ ] Train the cooperative two-agent control; run independent encoder/classifier baseline and token-mode ablations concurrently on CSCS.
- [ ] Implement conditional singleton evidence scores.
- [ ] Train TRACE-MRI with the two role-specific payoffs.
- [ ] Run collision, role, and `alpha` ablations.
- [ ] Compare game and cooperative performance on the same validation subjects.

**Delivery by 2 November:** verified game implementation, game-specific metrics, validation comparisons, and the full configuration history.

### 3–6 November: final evaluation

- [ ] Freeze selected models and selection rules.
- [ ] Run P1 ADNI held-out test with ADNI1, ADNI2/GO, and ADNI3 subgroup results; schedule AIBL external testing after the primary ADNI evaluation.
- [ ] Run P2 ADNI1/2-to-ADNI3 cross-phase evaluation.
- [ ] Complete at least the initial three consecutive seeds `0,1,2` for the essential method comparison; expand in order if resources permit.
- [ ] Bootstrap paired metric differences.
- [ ] Generate Tables 1–4, budget curves, and trajectory figures.

**Delivery by 6 November:** final candidate result package with exact subject counts and uncertainties.

### 7–10 November: first manuscript and registration

- [ ] Deliver the final implementation equations, method details, and verified results to me for paper writing.
- [ ] Generate camera-ready-quality diagrams and tables.
- [ ] Deliver the verified related-work comparison and experimental details; I write the introduction and manuscript.
- [ ] Check the current CVPR 2027 author kit and formatting requirements.
- [ ] Register the paper through OpenReview by 10 November AoE.
- [ ] Verify every co-author's OpenReview account and registration requirements.

**Delivery by 10 November:** a complete first manuscript draft and verified paper registration.

### 11–16 November: submission

- [ ] Run a final independent check of the split lists, code hashes, metrics, and captions.
- [ ] Confirm that all main results refer to the selected frozen checkpoints.
- [ ] Finalize cross-cohort and game ablation discussion.
- [ ] Check anonymization, references, paper length, and submission policies.
- [ ] Submit by 16 November AoE.

### 17–23 November: supplementary

- [ ] Add reproducibility details, training configurations, and seed-level tables.
- [ ] Include the complete ablations, extra trajectories, QC methodology, and failure cases.
- [ ] Upload supplementary material by 23 November AoE.

## 16. Decision gates and escalation

### Gate 1: 17 October

We need a valid cohort manifest, cross-phase identity mapping, working CNN feature extraction, and an explicit status for NeuroVFM access. Any unresolved access issue must be logged with the blocking action and owner.

### Gate 2: 22 October

The partial-evidence classifier must produce finite, reproducible predictions from randomly selected regions. The CNN and frozen-feature baselines must be evaluated on validation subjects. Fix the classifier or data pipeline before policy training if these checks fail.

### Gate 3: 27 October

The single-agent policy must execute reproducible trajectories, respect its region budget, and be comparable with static selection. If the policy repeatedly selects a small set of fixed regions independent of evidence state, inspect the observation features and test the state-dependence control before proceeding.

Run evidence-blind and position-only controls, and verify precisely which inputs each receives. A control that still receives scan-dependent global context or coordinates does not establish that its selection sequence is fixed across scans.

### Gate 4: 2 November

We need a matched comparison of TRACE-MRI and cooperative two-agent RL, with identical reader ensembles, attention modes, and AUROC-based checkpoint selection. The outcome determines the main experimental story. Record gains, losses, uncertainty, and evidence-role behaviour. If the game provides no improvement, complete the agreed ablations and report the result directly.

Match A0 and A2 on cache, reader ensemble, split, seed, and budget. Interpret context mixing or conditional-selection benefits only after blind, random, and static controls run on the same data. Present OASIS results as a pilot and reserve ADNI/AIBL claims until those tests are complete.

### Gate 5: 6 November

Stop changing the main method. Freeze evaluated checkpoints and focus on reproducing tables, cross-cohort tests, manuscript figures, and statistical analysis.

### Required escalation information

When an experiment is blocked, report:

```text
Blocking issue:
Exact command and error:
Affected run ID:
Current evidence:
Attempted fix:
Required access or decision:
Impact on next deliverable:
```

Do not silently modify the dataset, labels, architecture, or evaluation protocol to make a run complete.

## 17. Manuscript and results handoff

I will write the CVPR manuscript. Students will supply verified results, model definitions, implementation notes, references and figures. Target an eight-page main-paper draft excluding references, subject to final CVPR 2027 formatting instructions.

### Intended section allocation

| Section | Approximate main-paper space | Content |
|---|---:|---|
| 1. Introduction | 1.0 page | Problem, evidence acquisition, game formulation, contribution |
| 2. Related work | 0.75 page | Active visual attention, RL region selection, multi-agent medical vision |
| 3. Method | 2.25 pages | Encoder interface, evidence update, two-player payoffs, training |
| 4. Experimental protocol | 1.25 pages | ADNI/AIBL, baselines, splits, training, metrics |
| 5. Main results | 1.25 pages | Budget curves, game comparisons, external tests |
| 6. Analysis | 1.0 page | Ablations, evidence trajectories, generalization |
| 7. Conclusion | 0.5 page | Findings supported by final results |

### Main figures

1. **TRACE-MRI architecture:** show ResNet and NeuroVFM as alternative encoders, with exactly one selected per run and a shared downstream method after the cache interface. Distinguish context and local token modes. Show valid-token masks and $N_i$, policy inputs, region revelation, GRU update, singleton counterfactuals from the same pre-round state, joint belief update, and payoff computation.

2. **Conditional evidence acquisition:** show AUROC and balanced accuracy against region proposal budget, with methods and backbones separated clearly.
3. **Trajectory analysis:** show actual region selections and belief updates on a real MRI, including support and challenge actions.

### Main tables

- Main classification comparison for P1, including internal held-out and ADNI phase-specific results; include AIBL in the later external-validation stage.
- Cross-phase transfer for P2, with component and ensemble models.
- Cooperative-versus-competitive game ablations.

### Contribution-writing rules

- Explain the two-player game mathematically and connect every term to an experiment.
- Specify exactly what each policy observes before choosing an action.
- Define the evidence budget as the number of spatial feature regions revealed to the reasoning layer.
- Report complete encoder computation separately from policy computation.
- Distinguish model-based evidence changes from validated clinical explanations.
- Report the outcome of the game-vs-cooperative comparison irrespective of its direction.
- Cite the actual prior papers supporting claims about RL attention, medical image localization, and agentic visual reasoning.

## 18. References and required technical reading

Read these resources before implementing their relevant modules. Record any version-specific change in the project notes.

### Conference

- CVPR 2027 Call for Papers: https://cvpr.thecvf.com/Conferences/2027/CallForPapers
- CVPR 2027 dates: https://cvpr.thecvf.com/Conferences/2027/Dates
- CVPR author kit: https://github.com/cvpr-org/author-kit
- CVPR 2027 OpenReview group: https://openreview.net/group?id=thecvf.com/CVPR/2027/Conference

### Datasets and cohort definitions

- ADNI MRI data and imaging documentation: https://adni.loni.usc.edu/data-samples/adni-data/neuroimaging/mri/
- ADNI MRI metadata and acquisition tables: https://adni.loni.usc.edu/quick-start-guide-asset/MRI_tables.html
- ADNI FAQ, diagnosis mapping and image-selection guidance: https://adni.loni.usc.edu/help-faqs/faqs/
- AIBL study: https://aibl.org.au/
- Clinica AIBL conversion documentation: https://aramislab.paris.inria.fr/clinica/docs/public/dev/Converters/AIBL2BIDS/

### Backbone

- NeuroVFM encoder repository: https://huggingface.co/mlinslab/neurovfm-encoder
- NeuroVFM implementation: https://github.com/MLNeurosurg/neurovfm
- NeuroVFM weight license and access conditions: https://huggingface.co/mlinslab/neurovfm-encoder/tree/main
- JEPA model selection: document the exact chosen implementation, checkpoint, spatial-token API, and license in `reports/encoder_audit.md` before running it; do not treat an automatic "Z3" transcription as a model name.

### Related work to examine

- Mnih et al., *Recurrent Models of Visual Attention*, NeurIPS 2014: https://papers.neurips.cc/paper_files/paper/2014/hash/3e456b31302cf8210edd4029292a40ad-Abstract.html
- Wang et al., *Adaptive Focus for Efficient Video Recognition*, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Wang_Adaptive_Focus_for_Efficient_Video_Recognition_ICCV_2021_paper.html
- Wang et al., *AdaFocus V2: End-to-End Training of Spatial Dynamic Networks for Video Recognition*, CVPR 2022: https://openaccess.thecvf.com/content/CVPR2022/html/Wang_AdaFocus_V2_End-to-End_Training_of_Spatial_Dynamic_Networks_for_Video_CVPR_2022_paper.html

Complete a short related-work matrix with these columns: task, data modality, observation/action space, RL algorithm, number of agents, payoff/reward, dynamic stopping, encoder requirements, and differences from TRACE-MRI. Add the closest 2025–2026 papers after checking the original methods. Document the precise technical difference from each close method in the related-work matrix. Support every novelty claim with that comparison.

## 19. Command-line execution contract

Implement the commands below with the named flags. They define the expected entry points and output locations. All path arguments must accept explicit values from the configuration. The commands shown here are the target interface for the repository.

### 19.1 Prepare a 20-scan smoke-test cohort
Select 20 QC-passed, labeled ADNI scans from the development partition, balanced by class where possible, and create a **development-only smoke split**. Use training and validation subjects exclusively. This is for software validation, not reported scientific results.

```bash
python3 scripts/00_build_manifest.py \
  --config configs/datasets/adni_pooled.yaml \
  --output data/manifests/all_scans.csv

python3 scripts/01_make_splits.py \
  --config configs/datasets/adni_pooled.yaml \
  --manifest data/manifests/all_scans.csv \
  --output-dir data/splits/p1

python3 scripts/02_preprocess.py \
  --config configs/base.yaml \
  --manifest data/manifests/all_scans.csv \
  --split data/splits/p1/train.csv \
  --max-scans 20 \
  --output-dir /path/to/processed/smoke20
```

Record excluded scans in the QC report. A command must print the number of successful, failed, and skipped scans and return a nonzero exit code on a fatal configuration error.

### 19.2 CNN baseline and cache
```bash
python3 scripts/03_train_cnn.py \
  --config configs/encoders/resnet3d18.yaml \
  --protocol adni_pooled \
  --seed 0 \
  --output-dir outputs/cnn_p1_seed0

python3 scripts/04_cache_features.py \
  --config configs/encoders/resnet3d18.yaml \
  --checkpoint outputs/cnn_p1_seed0/best.ckpt \
  --token-mode context \
  --manifest data/manifests/all_scans.csv \
  --split-dir data/splits/p1 \
  --output-dir /path/to/feature/cache/p1/resnet3d18
```

Implement a `--max-scans 20` option for fast encoder checks. For real experiments omit that flag and verify that every required scan has a valid cache record.

### 19.3 Evidence readers and policies
```bash
python3 scripts/05_train_aggregator.py \
  --config configs/methods/evidence_aggregator.yaml \
  --protocol adni_pooled \
  --encoder resnet3d18 \
  --token-mode context \
  --crossfit-folds 3 \
  --seed 0 \
  --output-dir outputs/readers_p1_cnn_context_seed0

python3 scripts/06_train_baselines.py \
  --config configs/methods/dual_cooperative.yaml \
  --protocol adni_pooled \
  --encoder resnet3d18 \
  --token-mode context \
  --reader-ensemble outputs/readers_p1_cnn_context_seed0/ensemble_manifest.json \
  --seed 0 \
  --output-dir outputs/cooperative_p1_cnn_seed0

python3 scripts/07_train_trace.py \
  --config configs/methods/trace_game.yaml \
  --protocol adni_pooled \
  --encoder resnet3d18 \
  --token-mode context \
  --reader-ensemble outputs/readers_p1_cnn_context_seed0/ensemble_manifest.json \
  --seed 0 \
  --output-dir outputs/trace_p1_cnn_seed0
```

The reader ensemble manifest must list the three checkpoints, their hashes, the subject-to-fold map, feature adapter versions, token mode, cache hash, and the action/probability aggregation rules. Train and evaluate the remaining selectors against this same manifest. A single-reader pilot uses a separate manifest and cannot be mixed into the three-reader main comparison.

Implement `--smoke-test` in each training command. In smoke mode, train for two minibatches, perform one validation batch, save a checkpoint, reload it, and confirm that inference outputs match. This mode must complete without needing the full dataset.

### 19.4 Fixed test evaluation
```bash
python3 scripts/08_evaluate.py \
  --config configs/experiments/pooled_main.yaml \
  --run-dir outputs/trace_p1_cnn_seed0 \
  --dataset adni_internal_test \
  --budgets 2 4 6 8 \
  --output-dir outputs/eval_trace_p1_cnn

# Run this external evaluation in the planned AIBL stage.
python3 scripts/08_evaluate.py \
  --config configs/experiments/pooled_main.yaml \
  --run-dir outputs/trace_p1_cnn_seed0 \
  --dataset aibl_external \
  --budgets 2 4 6 8 \
  --output-dir outputs/eval_trace_p1_cnn_aibl

python3 scripts/09_make_figures.py \
  --registry experiments/registry.csv \
  --output-dir paper/figures
```

The evaluation command must save `predictions.csv`, `trajectories.jsonl`, `metrics.json`, `run_manifest.json`, and `runtime.csv` in its output directory. All evaluation code must execute under `torch.no_grad()` and model `eval()` mode.

### 19.5 Required Python interfaces
Define the following public interfaces and use them consistently across modules:

```python
class RegionAdapter(torch.nn.Module):
    def forward(self, encoding: SpatialEncoding) -> tuple:
        """Return regions [B,N_max,128], coords [B,N_max,3], valid [B,N_max], global [B,128]."""

class EvidenceAggregator(torch.nn.Module):
    def initial_state(self, global_context): ...
    def update(self, state, selected_indices, regions, coords): ...
    def predict(self, state): ...  # probability of AD

class ReaderEnsemble:
    def project_all(self, native_features, native_global, coords, valid):
        """Return per-reader projected regions and global contexts."""
    def initial_states(self, global_context_by_reader): ...
    def mean_action_logits(self, policy, states, global_context_by_reader,
                           coords, visited, round_id, allowed_mask): ...
    def update_all(self, states, selected_indices, regions_by_reader, coords): ...
    def mean_probability(self, states): ...
    def singleton_probe(self, states, selected_index, regions_by_reader, coords): ...

class RegionPolicy(torch.nn.Module):
    def logits(self, global_context, state, p_ad, coords, visited, round_id): ...
    def act(self, policy_state, allowed_mask, deterministic=False): ...

class EvidenceGame:
    def rollout(self, batch, budget, deterministic=False): ...
    def compute_rewards(self, round_records, labels): ...

class Evaluator:
    def predict_dataset(self, model, dataloader, budgets): ...
    def save_predictions(self, predictions, output_dir): ...
    def compute_metrics(self, predictions): ...
```

Use batch-first tensors consistently. Route each training subject to its out-of-fold reader; at validation and test, `EvidenceGame` uses `ReaderEnsemble` as specified in Section 5.2. For any function accepting a list of selected tokens, preserve a stable, documented output when it is given zero, one, or two unique tokens. Never let Python iteration order over a set decide model input order.

### 19.6 CSCS job execution
Transfer ADNI data from lab storage to the approved CSCS project storage using authorized transfer paths and checksums. Keep identifiable metadata and restricted raw imaging data outside Git. Prepare a reproducible environment and Slurm batch scripts for preprocessing, encoding, reader training, selection baselines, TRACE training, and evaluation. Confirm the account's simultaneous GPU-job limits, memory limits, storage quotas, and job-array support with CSCS documentation and the assigned allocation. Use queued dependencies only for jobs requiring predecessor outputs; submit independent backbone baselines, locality tests, and ablations concurrently when resources permit. Log job IDs, array indices, resources, failure status, and output/checkpoint locations.

### 19.7 Smoke-test success criteria
The smoke-test batch must include scans with different $N_i$ values. Add a NeuroVFM global-attention/local-only feasibility smoke check and record whether a selected JEPA model provides usable spatial tokens. Run mask and geometry tests, train two minibatches, save and reload checkpoints, verify out-of-fold routing for the cross-fitted path or the documented single-reader pilot path, execute a two-role rollout, and export trajectories with token coordinates.

On the research server, the same pipeline must run the 20-scan smoke cohort end to end before full ADNI feature extraction begins.

## 20. Final delivery checklist

### Code and data

- [ ] Repository with pinned environment and documented commands.
- [ ] Data manifest, cohort audit, image QC and immutable subject splits.
- [ ] Working 3D CNN encoder and full-volume CNN baseline; documented MRI-pretrained ResNet feasibility and checkpoint provenance.
- [ ] Working NeuroVFM spatial encoder and full-volume foundation-model baseline, subject to approved weight access; NeuroVFM global/local attention comparison and local-feature feasibility record.
- [ ] Native-token adapter, validated geometry and padding masks, and a `token_mode`-scoped cache; if implemented, treat a fixed 64-cell version as a separate ablation.
- [ ] Reproducible per-scan feature caches.
- [ ] Three cross-fitted partial-evidence readers and saved frozen checkpoints, with per-reader adapters and AUROC-based selection.
- [ ] Random, static Top-K, single-agent, cooperative, and TRACE-MRI implementations.
- [ ] Game reward unit tests and policy-gradient checks.
- [ ] Full configuration and experiment registry, including iteration checkpoint IDs and CSCS job IDs.
- [ ] JEPA spatial-token feasibility and selected direct-prediction/TRACE runs if compatible.
- [ ] Student task tracker updated with completed, overdue, and reassigned tasks.

### Experiments

- [ ] P1 pooled ADNI with phase-specific results.
- [ ] P2 cross-phase ADNI1/2-to-ADNI3 transfer.
- [ ] AIBL external evaluation with frozen checkpoints; prioritize ADNI deliverables first.
- [ ] ResNet and NeuroVFM results shown separately; include selected JEPA results when feasible.
- [ ] Budgets 2, 4, 6, and 8.
- [ ] Matched cooperative-vs-competitive comparison.

- [ ] NeuroVFM local-only versus global attention comparison, context/local mode feasibility results, $p_0$ and random-$k$ headroom, evidence-blind, position-only, size-only, and fixed-location controls; paired A0–A2 comparisons for every feasible backbone and token mode.
- [ ] Required game and memory ablations.
- [ ] Calibrated stopping experiment if validation supports it.
- [ ] Repeated seeds and subject-level paired uncertainty estimates.

### Paper material

- [ ] Dataset statistics with exact subject and scan counts.
- [ ] Main results and game ablation tables.
- [ ] Performance-versus-budget plots.
- [ ] Real MRI evidence trajectories with exact log-odds changes.
- [ ] Verified manuscript claims and references.
- [ ] Supplementary reproducibility document.
- [ ] Saved checkpoints, metrics, predictions, and figures matched to commit hashes.

**Prioritize the pooled ADNI manifest, subject-disjoint splits, NeuroVFM/ResNet baselines, three-reader interface, and CSCS execution now.** Keep the first execution end to end and reproducible. Every subsequent experiment should reuse the same cohort definitions, spatial interface, and evaluation pipeline.
