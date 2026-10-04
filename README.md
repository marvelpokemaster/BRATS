# BraTS practical v8: five Molab sessions

This replaces the expansive v7 experiment schedule. Start with RUN_NEXT.md.
The target is five sessions, each at most 12 hours. Work budgets are 9.5 hours
per session, reserving 2.5 hours for ZIP creation/transfer. These are stop rules,
not measured RTX PRO 6000 completion guarantees. Long individual operations and
slow uploads can exceed the reserve. Actual validation quality remains unknown.

## Main method

- 15,000 SLIC regions per T1ce/FLAIR partition; all four MRI modalities in node features.
- HGT with two layers and the existing 128 hidden channels.
- Learned additional depth mixture over k=0,1,2; fixed shared two-hop control.
- Structural refinement and masked feature/correspondence reconstruction OFF.
- Dice + weighted cross entropy, without focal/boundary auxiliary loss.
- Effective graph batch 8; actual physical batch calibrated up to 8, with memory
  headroom, propagation checkpointing and safe forward/backward OOM retries.
- Full audited patient cohort and persisted independent main splits.
- One predeclared seed (42); no three-seed sweep or multi-seed robustness claim.

The hop selector still computes all its candidate depths. It is a learned mixture,
not adaptive early-exit compute. A small learned mixture retains the advisor's
requested effective-hop analysis without keeping the expensive k=0..4 + SSL system.
Disabled module definitions remain in the helpers for compatible notebook wiring;
they create no structural/reconstruction model or extra training pass in v8.

## Work retained

Six main report rows: hybrid, graph-only, CNN-only, GraphSAGE graph-only,
fixed-hop HGT graph-only, and MONAI SegResNet. These need six distinct training
phases in total: three graph models and three voxel models. Graph-only reuses the
main graph exactly. GraphSAGE and fixed-hop effects are compared with graph-only
HGT, rather than attributing an absent CNN's effect to the graph architecture.

Each main phase has a predeclared 135-minute allowance including preparation
inside that phase and calibration, a 100-epoch maximum and patience 15. Epochs
can differ across models. This is a comparison under a stated compute budget;
it is not equal-epoch training or a claim that every baseline has converged.
The best complete validation epoch is selected. A partial epoch is discarded.
`stopping_reason`, actual epoch counts and elapsed time are saved. At least one
validated epoch is needed for a selectable checkpoint; meeting that minimum does
not by itself make a model suitable for publication.

The SLIC comparison is a separate validation-only graph pilot: 64 randomly selected
training patients and 16 validation patients, frozen once with a fixed seed,
at 5k/10k/15k/20k regions. Every resolution, including 15k, gets fresh graph builds
and fresh graph training on the same subset. Each has a 45-minute/20-epoch cap.
No pilot model is tested on held-out subjects or substituted for the main model.
This is preliminary performance/cost evidence, not a full-cohort optimum claim.

Test evaluation retains per-patient raw/postprocessed WT/TC/ET Dice, HD95,
sensitivity, precision and IoU, mean/SD, paired patient uncertainty estimates,
5–10 varied qualitative examples, modality Shapley on ten fixed test patients,
region hop distributions and gate/local-graph explanations. Single-seed confidence
intervals do not establish variation across training seeds. Explanations cover
the graph at fixed topology, not the entire graph+CNN pipeline.

## Transfers and compatibility

Use all matching v8 files. A fresh Part 1 is required; v7 model weights are
incompatible with the reduced architecture/training identity. Compatible verified
15k graph caches can be reused. The token placeholder stays blank.

Part 1 final handoff: `stage1_practical_v8/final/catalog.zip` in the configured
model repository. Later stores: `continuation_v8/<Part-1-hash>/...`.
Training files/figures are coalesced in ZIP commits about every two hours and at
stage boundaries. Final reports also have a downloadable ZIP. Graph caches keep
their existing ZIP plus small verification receipts. No per-epoch per-file spam.

If a required workload does not finish within its session budget, its status is
incomplete. The package does not silently add more sessions or fabricate the
missing comparisons. Saved ZIPs remain recoverable after a failure, but extending
the five-session study would be a separate budget decision.

## Dependencies and checks

Keep all ten helper Python files beside the selected notebook. Retain Molab's
working CUDA PyTorch and PyG installation. SegResNet adds MONAI, tested here at
1.5.1. With the existing PyTorch/NumPy environment, install that package using
`python -m pip install --no-deps monai==1.5.1` in both the Part 3 and Part 5
sessions (fresh sessions may not retain installations). Other packages are
the existing marimo, torch-geometric, numpy, scipy, nibabel, scikit-image,
matplotlib, kagglehub, joblib and huggingface-hub dependencies.

Local checks use tiny synthetic CPU graphs/volumes and a fake Hub. They cover
actual HGT/GraphSAGE/CNN/SegResNet training, test gating, SLIC separation, budget
stops, the actual Part 1/2 adapters, ZIP handoff and marimo scope. No full patient
training, actual GPU timing, live Hub writes or journal-readiness certification
was performed. See the validation JSON files and ADVISOR_SCOPE.md.

## Reference scope

The supplied Saueressig et al. paper already uses graph segmentation followed by
CNN refinement: https://arxiv.org/abs/2109.05580 . Do not claim that sequence alone
is novel. Its reported 15k resolution motivates a reference point, not proof that
15k is optimal for these two partitions or this internal split.

The supplied 2025.I2.050 paper uses a different transformer/DeepLab-based method
and dataset/evaluation setup. Its published scores are background, not a matched
baseline or a basis for claiming superiority. The narrower candidate contribution
is the typed multimodal HGT + bounded learned-hop mixture and its measured
cost/segmentation/explanation behavior under this stated protocol.

MONAI SegResNet API: https://monai.readthedocs.io/en/1.5.1/networks.html#segresnet
