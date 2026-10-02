# BraTS - consolidated research revision v5

This bundle includes the advisor-concern repairs, GPU calibration, verified
Hugging Face handoff, and a new controlled research experiment runner. Original
uploads and synced files were not edited. No real BraTS training, GPU benchmark
or live Hugging Face upload was performed in this review.

## Start here

1. Extract the ZIP. Keep **all five helper modules** beside the notebooks.
2. Use `.py` notebooks in marimo/molab; matching `.ipynb` copies are included.
3. Fill the blank `HF_TOKEN_PLACEHOLDER` near the beginning of each notebook
   you run. Confirm the dataset/model repository IDs and token write access.
4. Select your RTX Pro 6000 runtime. The configuration checks CUDA and a small
   3D convolution before the long work. Missing credentials/CUDA stop early.
5. Run Part 1, then Part 2 for the main model. For the research comparisons,
   read [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) and use the independent
   experiments notebook in additional sessions.

| File | Purpose |
|---|---|
| `part1_corrected.py` / `.ipynb` | Dataset audit, graph training, verified HF handoff |
| `part2_corrected.py` / `.ipynb` | Restore handoff, train voxel refinement, main evaluation |
| `research_experiments.py` / `.ipynb` | Controlled component experiments, resumable across sessions |
| `brats_protocol.py` | Dataset audit, label-independent inference, metrics |
| `brats_gpu.py` | GPU calibration, mixed precision, preparation and OOM retries |
| `brats_transfer.py` | Verified uploads/downloads and ZIP extraction |
| `brats_experiments.py` | Frozen experiment plans, training/evaluation continuation, reports |
| `baseline_3d_unet.py` | Existing optional baseline helper |

The independent research notebook restores Part 1's audited data/split but
does not execute the main Part 2 refinement run. Enable `RUN_ABLATIONS` in its
last configuration cell. The default study has **18 configurations x 3 seeds**;
use `RESEARCH_JOBS` and `RESEARCH_ACTIVE_SEEDS` to schedule individual sessions.
Changing scheduling does not change the frozen study. This complete study
requires many sessions; it is not expected to finish inside one 12-hour run.

## What the final research revision fixes

- The full ablation row includes structural refinement, matching the declared
  proposed model. There is also a descriptor-free refinement control.
- Fixed-depth and uniform-mixture controls use the proposed model's same
  shared propagation operator. Maximum-depth and no-hop-penalty rows help
  interpret the mechanism separately from its regularizer.
- SLIC resolution runs preserve exactly the same patient IDs/subsets.
- Graph and voxel training match the relevant optimizer, loss, schedule,
  stopping and inference settings, except for each declared intervention.
- Graph-only and no-boundary comparisons reuse identical graph weights.
- Research training, SLIC preparation and per-patient evaluation resume across
  sessions and use verified HF receipts. Test evaluation waits for the entire
  frozen training plan to finish.
- Reports include patient metrics, seed variability, paired confidence
  intervals, corrected comparison p-values, curves, resource measurements,
  and up to ten paired graph/hybrid/error examples with disclosed selection.
- Optional `RUN_REGION_XAI=True` computes direct WT/TC/ET probability
  attributions for the graph branch. Its fixed topology and exclusion of the
  CNN are explicit. These extra analysis cells are disabled by default.

The protocol document explains each row, how both uploaded papers informed
the design, how to report results, and what evidence is still needed before
journal submission. None of the exported software files contain real results.

## GPU behavior

Stage 1 retains its graph architecture/batch size and overlaps bounded batch
preparation with GPU work. Stage 2 calibrates training microbatches 1/2/4 and
inference batches 1/2/4/8 on the detected device. It prefers BF16 where supported,
otherwise FP16, and recalibrates in FP32 if a fresh run has non-finite probes.
Recorded precision is preserved when resuming a trained checkpoint.

Training keeps a fixed effective batch of four patients through gradient
accumulation. Weighted CE is averaged per patient. OOM retries discard partial
gradients and replay the same group at a smaller microbatch. Inference retries
without duplicating coverage. Probe weights and Torch RNG are restored before
training. Bounded CPU workers prepare pinned patches and never execute the
graph model on the GPU. `gpu_runtime.json` and research checkpoints record the
hardware, selected settings, timings and allocated memory.

For troubleshooting a fresh run, `GPU_AUTOTUNE=False`,
`VOXEL_MAX_MICROBATCH=1`, `INFERENCE_MAX_BATCH=1` and
`VOXEL_PREFETCH_WORKERS=0` provide a conservative path. Precision/effective-batch
changes intentionally invalidate an incompatible Stage 2 resume. No custom
CUDA kernels or model compilation were added.

CPU graph construction, metadata reads, ZIP creation and uploads can still
limit throughput. Available VRAM does not establish sufficient RAM/disk or
guarantee the session limit. Budget checks occur between operations; actual
GPU speed, storage requirements and the longest operations need a molab run.

## Handoff and compatibility

Part 1 uploads a complete graph-cache ZIP, saved graph checkpoint and handoff
manifest. Hash/size checks and recorded immutable HF revisions let Part 2
restore those exact artifacts even if repository files later change. Complete
matching graph ZIPs are reused. Transfer failures preserve local artifacts and
are not reported as success. Use the same audited data/settings in both parts.

Repaired **Stage 1 review-v3** checkpoints remain compatible; rerun the updated
save/handoff cells if the earlier manifest lacks pinned ZIP receipts. The
main Stage 2 **GPU-v4** checkpoint protocol remains compatible with this v5
research update. Older pre-GPU-v4 refinement checkpoints require fresh training.
The new research runner uses a separate `research_v5/<study-id>/` namespace and
retrains controlled models rather than reusing unmatched research results.

Main training saves completed epochs and restores model/optimizer/scheduler/
scaler/RNG/history. Partial epochs replay. Later main diagnostics wait for
Stage 2 completion. Keep the legacy `RUN_BASELINES=False` for the matched
research study; use the new runner's CNN/GraphSAGE rows. The old small U-Net
driver remains exploratory and lacks complete rolling resume.

## Dataset and evaluation essentials

The expected cohort is **1251**, giving **875/187/189** under the existing split
rounding. A 1252-case cohort gives 876/187/189. The audit stops on unexplained
counts, duplicate MRI content, inconsistent modalities or geometry. It does
not discard an arbitrary patient. Use documented `CASE_EXCLUSIONS` and, when
available, `OFFICIAL_CASE_IDS_PATH`. Inspect `audit/dataset_audit.json`; matching
the count alone does not prove official membership. The pipeline currently
requires matching 1 mm geometry and uses an internal patient split.

Validation/model selection use full-volume postprocessed WT/TC/ET Dice.
Inference never reads segmentation labels to choose a crop. Labels may guide
training patch sampling and post-test figure selection. Training patch Dice,
node proxy Dice and full-volume validation Dice measure different supports.

HD95 is in mm with a documented custom empty-region convention: both-empty
gets HD95=0/Dice=1; one-empty gets Dice=0 and a physical volume-diagonal HD95
penalty. Patients are retained in mean/SD. Do not assume official-evaluator
parity or compare with papers using another convention.

## Verification and limits

**35 synthetic check groups passed:** 12 regression, 12 GPU/transfer and
11 research integration groups. The integration checks exercise real tiny
PyG HGT, structural refinement, shared-hop and GraphSAGE models, gradients,
label-free inference, CNN training, exact CPU resume with dropout, interrupted
SLIC/evaluation, paired statistics, saved figures, mocked research HF receipts
and direct region Shapley efficiency. Real HF client signatures were checked.

All three notebooks pass marimo 0.24.0 scope/dependency analysis and native
`marimo check`. Public names have one defining cell; leading-underscore names
are cell-local. The loaded graph model and downstream completion dependencies
are explicit. See `validation_results.json`, `gpu_transfer_validation.json`,
`research_validation.json`, `marimo_validation.json` and `source_provenance.json`.

These checks do not certify actual CUDA behavior, full-cohort runtime/memory,
live Hub reliability, research novelty or journal acceptance. Real results,
failure analysis and a justified manuscript claim are still required.
