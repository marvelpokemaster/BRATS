# QoS-HRGN: Heterogeneous Adaptive Graph Learning for BraTS 3D Segmentation

QoS-HRGN is a multimodal 3D brain-tumor segmentation research pipeline. It represents MRI volumes as heterogeneous supervoxel graphs, applies a Heterogeneous Graph Transformer (HGT) with node-wise adaptive propagation, and optionally refines the graph predictions at voxel resolution.

> **Dataset configuration:** The current notebooks target the Kaggle mirror `dschettler8845/brats-2021-task1` (BraTS 2021 Task 1). Dataset discovery in the notebook reported **1,252 complete labeled cases** in the downloaded package. This is the count for that mirror/package, not a claim about every BraTS 2021 challenge release or split.

## Pipeline at a glance

```text
BraTS 2021 MRI: T1, T1ce, T2, FLAIR + segmentation
                         |
               patient-level data split
                         |
        independent 3D SLIC on T1ce and FLAIR
                         |
         heterogeneous supervoxel graph
       (within-modality spatial + cross-modal
                correspondence edges)
                         |
             HGT (2 layers, 4 heads)
                         |
       uncertainty-aware adaptive propagation
             (candidate depths k = 0..4)
                         |
           node-wise class prediction
                         |
       project graph probabilities to voxels
                         |
      optional 3D voxel refinement network
                         |
              WT / TC / ET Dice
                         |
        diagnostic visualizations / XAI
```

## Dataset and split

The current data loader uses:

- **Dataset:** `dschettler8845/brats-2021-task1` via `kagglehub`
- **Modalities:** T1, T1ce (post-contrast T1), T2, and FLAIR
- **Labels:** raw BraTS labels 0, 1, 2, and 4, internally mapped to 0, 1, 2, and 3
- **Case discovery:** recursively locates NIfTI files and retains cases with all four modalities and a segmentation
- **Case limit:** `MAX_CASES = None`, meaning all complete discovered cases are selected

The split is patient-level and shuffled with `SEED = 42`. The notebook applies a 70% / 15% / remainder split using integer truncation:

| Partition | Cases (when 1,252 complete cases are discovered) |
|---|---:|
| Train | 876 |
| Validation | 187 |
| Test | 189 |
| **Total** | **1,252** |

The code asserts that patient IDs do not overlap between partitions. If the number of complete cases differs in another environment or dataset revision, the split counts will differ; use the notebook's printed counts as the run-specific record. The Kaggle mirror is a labeled dataset package; these are notebook-created partitions, not necessarily the official challenge partitions.

## Graph representation

- **Node types:** `t1ce` and `flair`, each produced by its own 3D SLIC partition.
- **SLIC resolution:** `N_SEGMENTS = 15000` per modality; `COMPACTNESS = 0.3`; `SLIC_ITERS = 10`.
- **Node features:** 32 values: 28 appearance statistics (mean, standard deviation, and five quantiles for each of four MRI modalities) plus four geometric features (normalized centroid coordinates and log-relative volume).
- **Spatial edges:** connect face-adjacent supervoxels within a modality.
- **Correspondence edges:** connect supervoxels from the two modalities that overlap in voxel space.
- **Fractional labels:** node targets preserve the class proportions within each supervoxel, rather than representing each node only by its majority class.

## Model

The graph model uses a two-layer PyTorch Geometric `HGTConv` backbone (hidden dimension 128, four attention heads), followed by uncertainty-aware adaptive propagation. The adaptive component considers candidate propagation depths from 0 through 4 and learns node-specific mixtures rather than imposing one fixed receptive-field depth on every node.

An adaptive edge gate conditions message weighting on node representations, edge information, similarity, relation type, and prediction uncertainty. HGT supplies relation-aware heterogeneous attention; the adaptive propagation/gating module is the project's proposed adaptation.

## Voxel refinement

The pipeline has two training stages:

1. **Stage 1 — graph/node model:** train the heterogeneous graph model and evaluate graph-derived voxel predictions.
2. **Stage 2 — voxel refinement:** freeze the graph model, project node probabilities into voxel space, combine them with the four MRI channels, and train the voxel refinement head on cropped volumes.

Current notebook configuration:
- Graph training: `EPOCHS = 100` maximum
- Voxel refinement: `VOXEL_EPOCHS = 40` maximum
- Graph early-stopping patience: `EARLY_STOP_PATIENCE = 15`
- Stage 2 can pause at the configured time budget and resume from its checkpoint.

These are configured maxima, not a guarantee that every epoch ran. Report the actual completed/best epoch from the run logs or checkpoints. Voxel-refinement results should only be reported when Stage 2 completed and its held-out evaluation was produced.

## Conventional 3D U-Net baseline

`notebook33_part2.ipynb` includes a selectable independent 3D U-Net baseline
for the same persisted patient-level split used by the graph baselines. It is a
lightweight two-level 3D encoder-decoder with max-pooling, transposed-convolution
upsampling, skip connections, four MRI input channels (T1, T1ce, T2, FLAIR),
and four output classes using the internal label mapping above. Crop bounds are
derived from MRI foreground only; segmentation labels are used only as training
targets and evaluation ground truth.

### Training protocol and gradient accumulation

Training uses **patient-weighted gradient accumulation**:
- Each microbatch scales its backward pass by its actual patient count $n_i$.
- At the optimizer step, accumulated gradients are normalized by the total number
  of patients $N_{\text{total}} = \sum n_i$ in that accumulation group.
- This ensures that incomplete final batches or unequal microbatch sizes receive
  exact per-patient weighting equivalent to a single combined batch.
- **Loss formulation:** `voxel_loss` supports decoupled per-patient Dice loss
  (`per_patient_dice=True`), computing spatial intersections and unions per
  volume before averaging across the batch. For $B=1$ (default), per-patient
  Dice and global batch Dice are identical; for $B>1$, per-patient reduction
  ensures exact mathematical linear separability across microbatches.
- For mixed precision (FP16), gradients are unscaled before normalization and
  gradient clipping ($L_2 \le 2.0$), preventing scaled-gradient distortion.

The baseline registry keeps `RUN_BASELINES = False` by default. To include the
U-Net in a baseline run, set `RUN_BASELINES = True` and leave
`RUN_3D_UNET_BASELINE = True`. Comparative reporting rejects
`BASELINE_MAX_CASES != 0`; limited-case comparisons require retraining every
compared model on the same selected subjects.

### Checkpoints and protocol metadata

The best validation-Dice checkpoint is saved as `baseline_3d_unet_best.pt` under
`PERSISTENT_BASE`. The checkpoint file contains:
- `state_dict`: model weights.
- `protocol`: complete experiment metadata including parameter count, optimizer,
  learning rate, early stopping patience, best epoch, loss description, device,
  precision requested and effective, gradient accumulation steps, gradient weighting
  semantics, split fingerprint, and per-epoch timing/memory metrics.

Its path plus protocol metadata are recorded in `fair_baseline_registry.json`.
The registry reports WT, TC, and ET Dice, HD95 in voxel units, sensitivity,
precision, and IoU with mean, standard deviation, and valid/total counts.

U-Net resource settings are configurable in the baseline cell:
`UNET_BASE_CHANNELS = 8`, `UNET_BATCH_SIZE = 1`,
`UNET_GRADIENT_ACCUMULATION_STEPS = 1`, `UNET_PRECISION = "auto"`, and
`UNET_CHANNELS_LAST_3D = True`. `auto` uses BF16 autocast when the selected
CUDA device supports it and otherwise uses FP32; set it to `"off"` for an
explicit FP32 run or `"fp16"` when BF16 is unavailable but FP16 is desired.
An explicit `"bf16"` request fails on a CUDA device without BF16 support rather
than silently changing precision.

### Verification and tests

For unit tests and gradient accumulation checks:

```bash
python -m unittest test_3d_unet_baseline -v
```

For a target-GPU smoke test in Molab:

```bash
python -m unittest test_3d_unet_baseline.ThreeDUNetBaselineTest.test_cuda_synthetic_smoke
```

For an opt-in integration test on a genuine BraTS case (requires an authorized
environment with downloaded BraTS data; never executed on local development machines):

```bash
BRATS_REAL_CASE_PATH=/path/to/BraTS2021_00000 \
BRATS_UNET_CHECKPOINT=/path/to/baseline_3d_unet_best.pt \
python -m unittest test_3d_unet_baseline.ThreeDUNetBaselineTest.test_opt_in_real_brats_case_pipeline -v
```

The implemented comparison set is CNN-only, 3D U-Net, GraphSAGE or GAT,
HGT graph-only, and HGT graph+CNN. nnU-Net and SegResNet remain future-work
placeholders; no baseline performance is implied until the corresponding
training run has completed.

## Evaluation

The notebook evaluates segmentation at voxel level using the standard BraTS regions:

- **WT (Whole Tumor):** NCR/NET + ED + ET
- **TC (Tumor Core):** NCR/NET + ET
- **ET (Enhancing Tumor):** ET

| Region | Definition |
|---|---|
| WT | Internal classes 1, 2, 3 |
| TC | Internal classes 1, 3 |
| ET | Internal class 3 |

Use the held-out **test** partition for final reported performance. Validation metrics are for model selection and should not be presented as test performance. This README does not hard-code Dice values because they must correspond to the exact completed run, model checkpoint, post-processing configuration, and test split. Include per-region Dice (WT, TC, ET), and clarify whether values are raw or post-processed.

## Running the notebooks

The repository separates the work into two notebooks to support the staged training/checkpoint workflow:

1. **`notebook34_part1.ipynb` — Stage 1**
   - Installs/imports dependencies through the notebook environment.
   - Downloads/discovers the BraTS 2021 Kaggle dataset.
   - Builds or loads the graph cache.
   - Creates the patient-level train/validation/test split.
   - Trains and evaluates the graph model.
   - Saves and uploads the Stage 1 checkpoint and run metadata.

2. **`notebook33_part2.ipynb` — Stage 2**
   - Loads and validates the Stage 1 checkpoint and matching split/configuration.
   - Prepares projected graph-probability maps and MRI crops.
   - Trains/resumes voxel refinement.
   - Evaluates graph-only versus graph-plus-voxel-refinement predictions and runs diagnostics.

Run Stage 1 first and wait for the notebook's checkpoint upload/verification to finish before starting Stage 2. Both notebooks require access to the expected cache/checkpoint locations and compatible configuration. A checkpoint from a different split or configuration is intentionally rejected.

### Environment and access

The notebooks are designed for a Python environment with PyTorch, CUDA, PyTorch Geometric, and the listed scientific packages. They also use:

- `kagglehub` for dataset download (configure Kaggle credentials/access as required by Kaggle)
- Hugging Face Hub for graph-cache/checkpoint persistence when enabled (configure a token with appropriate repository permissions)

Store credentials in environment variables or the runtime's secret manager. **Do not commit Kaggle or Hugging Face tokens to the repository or notebook.**

A full 1,252-case run can be compute-, memory-, storage-, and time-intensive. First validate the workflow with a small `MAX_CASES` value, then restore `MAX_CASES = None` for the full experiment. Make sure the resulting split and checkpoint are from the same run.

## Main configuration

```python
# Dataset
DATASET_SLUG = "dschettler8845/brats-2021-task1"
MAX_CASES = None

# Graph
N_SEGMENTS = 15000
COMPACTNESS = 0.3
SLIC_ITERS = 10
NODE_TYPES = ("t1ce", "flair")
NODE_FEAT_DIM = 32

# Graph model
HIDDEN_DIM = 128
HEADS = 4
HGT_LAYERS = 2
USE_ADAPTIVE_HOPS = True
K_MAX = 4
EPOCHS = 100                 # maximum graph epochs

# Voxel refinement
USE_VOXEL_REFINEMENT = True
VOXEL_EPOCHS = 40            # maximum refinement epochs
VOXEL_BASE_CHANNELS = 32
```

See the notebook cells for the complete configuration, optional structural refinement and masked-reconstruction settings, checkpoint paths, and diagnostic switches.

## Label mapping

| Raw label | Internal label | Meaning |
|---:|---:|---|
| 0 | 0 | Background |
| 1 | 1 | NCR/NET |
| 2 | 2 | Edema (ED) |
| 4 | 3 | Enhancing Tumor (ET) |

## Repository contents

- `notebook34_part1.ipynb`: Stage 1 graph construction, training, and evaluation.
- `notebook33_part2.ipynb`: Stage 2 voxel refinement and downstream diagnostics.
- Other repository files may contain intermediate experiments or artifacts; use the two staged notebooks above as the current workflow.

## References

- Saueressig, C., Berkley, A., Kang, E., Munbodh, R., & Singh, R. (2020). *Exploring Graph-Based Neural Networks for Automatic Brain Tumor Segmentation.* DataMod 2020.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). *Heterogeneous Graph Transformer.* Proceedings of The Web Conference (WWW 2020).
- BraTS 2021 Task 1 data mirror: `dschettler8845/brats-2021-task1` on Kaggle.
