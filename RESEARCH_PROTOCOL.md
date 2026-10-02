# BraTS research protocol - consolidated revision v5

This revision prepares controlled, repeatable experiments. It does not certify
journal acceptance, establish clinical usefulness, or supply results that have
not been measured. Use the experiment outputs to decide which claims the data
support. Freeze the protocol before examining the final test results.

## Which notebook to run

1. `part1_corrected.py`: audit the dataset, train/select the main graph model,
   then upload the verified graph ZIP, checkpoint and handoff manifest.
2. `part2_corrected.py`: restore the handoff and train/evaluate the main voxel
   refinement model. This retains the GPU-v4 training/checkpoint protocol.
3. `research_experiments.py`: independent additional molab sessions for the
   controlled experiments below. It restores the audited Part 1 data and split,
   then trains its own declared models. It does not execute main Stage 2 first.

All three have `.ipynb` equivalents. Keep all five helper modules alongside
the notebooks. Fill the blank `HF_TOKEN_PLACEHOLDER` in each notebook you run.
The research notebook uses the same HF repositories and verifies checkpoint
receipts against their recorded immutable revisions.

Main training still uses two stages. The experiments require additional
sessions: 18 configurations x 3 seeds by default, with identical graph weights
reused where the change affects only the voxel stage. This is a study, not
18 free evaluations of the already-trained model. Its total runtime cannot be
inferred from VRAM or promised to fit in one 12-hour session.

## Research controls (final cell of the experiments notebook)

| Setting | Default / meaning |
|---|---|
| `RUN_ABLATIONS` | False; set True to execute the research runner |
| `RESEARCH_MODE` | `train`, then `validation`, then `test` |
| `RESEARCH_SEEDS` | (42, 43, 44); frozen repeated-training seeds |
| `RESEARCH_PLAN_NAMES` | Empty = every declared row; a predeclared subset must include `full` |
| `RESEARCH_JOBS` | Empty = all planned rows; select row names to schedule a session |
| `RESEARCH_ACTIVE_SEEDS` | Empty = all planned seeds; select seeds to schedule a session |
| `RESEARCH_MAX_CASES_PER_SPLIT` | 0 = full cohort; nonzero marks a debug study |
| `RESEARCH_BUDGET_HOURS` | The notebook training budget, measured from session start |

Changing jobs or active seeds only schedules work. Changing the frozen plan,
seed set, selected cohort, training settings, source data or implementation
creates a different study identity. The model, data and split settings must
match the Part 1 handoff. The declared full model has adaptive hops and
structural refinement enabled, and masked reconstruction disabled.

For example, schedule `RESEARCH_JOBS = ("full", "cnn_only")` and
`RESEARCH_ACTIVE_SEEDS = (42,)` first while leaving the complete plan unchanged.
Later select the remaining jobs/seeds. This is configuration, not a model-code
edit. A debug subset is for feasibility checks and cannot export a final test
report. Do not mix debug results with full-cohort results.

After every planned training run finishes, use `validation` to inspect
validation outputs. Choose and freeze any development changes using validation
only, then use `test` for final reporting. The test phase refuses to start if
any run in the frozen plan is unfinished. This guard cannot undo test results
already inspected during earlier development: disclose that history and use
a new untouched cohort for confirmatory claims if needed.

## Declared comparison matrix

Except where explicitly changed below, each row includes the same HGT,
structural wrapper, voxel head, soft labels, losses and inference protocol.

| Row | Question / exact change |
|---|---|
| `full` | Complete proposed graph + structural refinement + voxel model |
| `no_structural` | Remove the structural module and its associated auxiliary loss |
| `no_descriptors` | Keep the refinement/auxiliary-loss branch but remove its structural-descriptor inputs |
| `fixed_shared_hops` | Use fixed depth HOPS with the same shared propagation operator |
| `fixed_shared_max` | Use fixed depth K_MAX with the same shared propagation operator |
| `uniform_shared_hops` | Average the shared-depth states uniformly instead of learning their weights |
| `no_hop_penalty` | Keep learned hop selection, set its expected-depth regularizer to zero |
| `hgt_only` | Remove propagation after HGT; retain structural and voxel refinement |
| `no_cross_edges` | Remove both directed correspondence relations; original graphs stay intact |
| `hard_labels` | Replace graph fractional targets with majority one-hot targets |
| `no_boundary` | Zero the voxel boundary auxiliary loss; this tests boundary supervision |
| `no_voxel` | Evaluate the exact same full graph checkpoint without voxel refinement |
| `cnn_only` | Four MRI channels into the same voxel-head family, without graph priors |
| `graphsage_backbone` | Merge typed relations and use max-pool GraphSAGE, with later refinement retained |
| `dice_ce` | Loss-family comparison: weighted CE + Dice; remove focal and boundary terms |
| `slic_5000`, `slic_10000`, `slic_20000` | Complete pipeline at changed SLIC resolution; `full` covers default 15000 |

If the active SLIC resolution differs, the available rows cover the other
members of 5000/10000/15000/20000. This is the requested resolution **per
partition**, not the total retained node count. Both modality partitions are
present; report measured nodes/edges as well as the requested SLIC setting.

Fixed and uniform controls do not train the unused selector. Their active
parameter counts are recorded. The expected-hop penalty is relevant only to
learned selection; `no_hop_penalty` separates its contribution. Compare the
fixed/mean controls as well as full versus no-penalty when interpreting the
adaptive mechanism. The `dice_ce` and GraphSAGE rows are deliberately broader
loss/backbone comparisons, not claims that only one scalar changed.

The CNN control uses four input channels; the hybrid uses eight. Report that
and their actual parameter counts. The boundary head supplies an auxiliary
training signal, not a GSCNN reproduction or an independent anatomical prior.

## Matching and model selection

- Every row and SLIC rebuild uses the same persisted patient IDs. Debug
  selection is performed within each split and reused across resolutions.
- All trainable models start fresh for their declared seed. Validation selects
  the best full-volume postprocessed mean WT/TC/ET Dice. Train/test labels never
  select the inference crop. Structural evaluation uses predicted classes.
- Graph training matches the proposed loss, structural auxiliary loss,
  teacher schedule, hop penalty, AdamW, cosine schedule, epoch cap and patience
  unless that component is explicitly removed. Graph shuffle order has its own
  seed per epoch, independent of model initialization.
- Voxel models share patch policy, effective patient batch, AMP calibration,
  optimizer, schedule, epoch cap and patience. All use full-volume sliding
  windows and the same postprocessing. Different patch shapes are not given
  artificial background padding to form a batch.
- The main Part 1 checkpoint supplies the audited split/data identity and
  training-derived class weights. Research models are retrained so comparisons
  use one common runner; the main model's previously observed test score is
  not substituted as the full row.

Default graph and voxel caps are 100 epochs with patience 15. Class weights
are fixed from the common training cohort across controls. This isolates
supervision/architecture choices while keeping weighting consistent.

## Saving, continuation and resources

The runner saves model, optimizer, scheduler, scaler, RNG, histories and best
validation weights after each completed epoch. It uploads periodically and on
pause/completion. An interrupted epoch replays from its preceding checkpoint.
Completed experiments are skipped. Evaluation saves per-patient progress and
does not repeat already-completed patients when resumed.

Separate SLIC caches include the study identity. A budget pause stores their
partial ZIP; a subsequent session builds missing cases. Completed SLIC ZIPs
are reused. Frozen graph probabilities use a model-weight digest and are
regenerated when needed in a fresh runtime. They are not confused across rows.
Checkpoints and metadata use verified HF receipts with exact revisions.

Storage must accommodate original data, graph caches and an additional ZIP
copy. The SLIC builder/packer checks free disk space. A GPU with 96 GB VRAM
does not establish sufficient system RAM, disk or network throughput. Budget
checks occur between batches/cases; an individual long graph build, ZIP,
upload or inference call can still run beyond the remaining session allowance.
Do a small GPU feasibility run before committing to the full study.

## Outputs and statistical interpretation

Results are under `PERSISTENT_BASE/research_v5/<study-id>/` locally and
`research_v5/<study-id>/` in the configured model repository. Main folders:

- `protocol.json`: fixed patient IDs, seeds, settings and implementation hashes.
- `models/` and `runs/`: resumable states, best epochs, histories and completion records.
- `evaluation/val/` and `evaluation/test/`: per-patient raw and postprocessed metrics.
- `reports/*_patients.csv`: patient-level metrics and cached-graph inference timings.
- `reports/*_summary.csv`: per-run mean/SD, parameter counts, best epochs,
  node/edge sizes, recorded epoch time and peak CUDA allocations.
- `reports/*_report.json`: raw/processed summaries and paired comparisons.
- `reports/curves/`: graph/voxel training objectives and validation Dice.
- `reports/*_slic_performance_cost.png`: resolution versus Dice and cached-graph inference time.
- `reports/qualitative/`: up to ten paired reference/graph/hybrid/error panels
  with a saved selection manifest. The first declared seed is used. Examples
  cover observed poor/good Dice, small ET, large WT, a surface-to-volume proxy,
  then deterministic random cases. These are disclosed descriptive selections,
  not a representative performance sample. The reference mask chooses only
  the displayed slice after full-volume inference, never the inference crop.

Graph-construction seconds are recorded separately for rebuilt SLIC cases.
The inference timer includes cached metadata loading, model inference and
postprocessing, and excludes SLIC construction and metric computation. Recorded
training seconds cover completed epochs; they do not count discarded partial
epochs, startup, cache preparation, calibration or uploads as training time.
Do not label these fields end-to-end deployment latency or total rental time.

The comparison reports average seeds within each patient before a paired
patient bootstrap (5000 draws). They also report seed-level variability and
paired sign-flip p-values, with Holm correction across reported Dice endpoints
and comparisons. Seed-patient pairs are not treated as independent patients.
The intervals are conditional on this cohort, protocol and fixed seed set.
All compared rows must have identical patients and seeds; missing/duplicate
pairs are rejected. Patient SD, seed SD and confidence intervals mean different
things and must be labelled separately. Effect size and uncertainty matter
even when a p-value is small.

HD95 retains the previously documented custom empty-region convention. It is
not automatically equivalent to the official BraTS evaluator or a paper using
another convention. Report the convention, units and raw/processed results.

## Explanation scope

`RUN_REGION_XAI=True` enables the optional Part 2 graph-analysis cells in an
additional session. The cohort output now attributes p(WT), p(TC), and p(ET)
directly, instead of relabelling predicted-class attributions. It integrates
all nodes using fractional region voxel weights within each patient and
reports patient means/SD and present-region counts. Empty regions are explicit.

All 16 modality coalitions are evaluated using training-mean feature
replacement. Efficiency is checked for the three region outputs. Graph
topology is fixed: this explains the graph branch's features, not the effects
of SLIC construction or the complete CNN pipeline. Predicted-class plots are
separate exploratory diagnostics. No uncorrected directional significance
claim is made for the graph-region cohort summary.

## How the supplied papers were used

### Saueressig et al., 2109.05580v2.pdf

The graph-plus-CNN pipeline, sequential training, supervoxel projection and
need to test the CNN's contribution already appear in this paper (methods
2.1-2.5, Figure 2). Its CNN crop comes from the GNN prediction, never the true
test mask. That supports removing the earlier label-dependent crop and the
graph-only/CNN-only/hybrid comparisons.

Its construction stacks four modalities into one SLIC partition, uses
GraphSAGE-pool and a two-layer CNN with projected logits. This implementation
uses two single-modality partitions, all four modalities as features, typed
HGT relations, shared-depth mixing, structural refinement, projected
probabilities and a different voxel head. The GraphSAGE row is a controlled
backbone comparison on this pipeline, **not an exact reproduction** of that
paper. Its official challenge cohorts, model selection and mean/median
reporting differ from this internal split; do not use unmatched scores to
claim superiority. The 15000 setting from its construction does not establish
the optimum for these two partitions.

### Abd-Elhafiez, 2025.I2.050.pdf

This paper describes GSCNN/DeepLabv3+ with ResNet50 on BraTS2020. Its use of
shape/boundary information motivates testing boundary supervision rather than
assuming it helps. It does not establish the novelty of a heterogeneous graph
or adaptive-hop model. Its title's heterogeneous tumours are not typed graph
relations. Its reported F1/accuracy percentages are not interchangeable with
3D patient-level WT/TC/ET Dice. Table 2's region Dice and Table 1's headline
metrics must remain distinct. Neither paper supplies a directly comparable
internal-test baseline for this pipeline without a matched reimplementation.

## Manuscript claims and remaining evidence

A defensible candidate contribution is an experimentally supported combination
of typed cross-partition context, shared-depth selection and predicted
structural descriptors for this segmentation pipeline. Generic GNN+CNN and
adaptive-hop concepts are prior ideas; do not call them individually new.
Computing all candidate depths does not prove an early-exit speedup. Keep any
claim about efficiency tied to recorded measurements and their stated scope.

The implementation now supports the component study, but publication still
depends on real results, a justified novelty comparison, dataset provenance,
representative paired figures, and the target journal's requirements. A
strong external baseline such as a properly configured nnU-Net/SegResNet and
external-site testing may be needed for stronger performance/generalization
claims. The legacy small independent U-Net driver is not a substitute for
those claims and is not part of the new matched research table.

Before submission, report cohort inclusion/exclusion, patient-level separation,
preprocessing, model-selection rule, complete hyperparameters, seeds, uncertainty,
failures and code/data availability. These documentation priorities are
consistent with [CLAIM 2024](https://pubs.rsna.org/doi/10.1148/ryai.240300).
Do not conceal negative ablations or select seeds based on test performance.
Code inspection and synthetic checks cannot identify the real extra patient,
prove the learned mechanism helps, establish external generalization or
guarantee acceptance.
