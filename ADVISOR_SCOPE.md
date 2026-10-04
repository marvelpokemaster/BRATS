# Advisor requirements versus the practical budget

The original advisor text and two supplied PDFs were reviewed as evidence. This
document distinguishes retained requirements from deliberately narrowed evidence.
It must not be presented as proof that all empirical concerns are resolved.

| Concern | Practical implementation and limits |
|---|---|
| 1,252 vs 1,251 patients, duplicates, independent splits | Existing patient/data audit and saved split verification retained. No main-cohort truncation to save training time. Actual audit results must still be checked. |
| Graph-only vs CNN-only vs graph+CNN | All three retained, same full subjects. Graph-only reuses the exact selected main graph. CNN-only uses the same refinement architecture without graph channels. |
| Ordinary graph baseline | GraphSAGE retained. Compare it directly against graph-only HGT; a separate paired table makes this comparison explicit. |
| Strong voxel reference | MONAI SegResNet retained as the independent voxel baseline. The extra lightweight U-Net and nnU-Net training are omitted. This meets the advisor's minimum of one voxel reference but does not reproduce every model in the example list. Adequacy depends on its actual training curves/results. |
| Typed graph + adaptive learnable hops | Retained with depths 0–2. Fixed shared two-hop graph control retained. Structural refinement and masked reconstruction were additions, not requirements in the supplied advisor remarks, and are removed from the active method. |
| Up to 100 graph / 100–150 CNN epochs; patience 15–20 | Maximum 100 and patience 15 retained, but a 135-minute phase cap can stop training earlier. This is an explicit compute-limited study, not proof of convergence or an equivalent fully trained comparison. |
| Best validation model and learning curves | Retained; incomplete epochs never become selectable checkpoints. Main train loss, validation loss, train diagnostic Dice and full-volume validation Dice are saved/plotted with clear scope labels. |
| Dice + weighted CE | Used as the main graph and voxel objective. Additional focal/boundary loss sweeps omitted. |
| 5k/10k/15k/20k Dice vs computation | Retained only as a preliminary graph study on the same 64 training / 16 validation patients, one seed, maximum 20 epochs / 45 min each. Includes fresh 15k training/builds. It does not establish full-cohort hybrid performance or a definitive optimum. This is the largest reduction in the advisor evidence. |
| Parameters, VRAM, construction/epoch/inference time, nodes/edges | Instrumentation retained. Pilot graph builds timed for all resolutions. Cached-graph inference is labelled; it excludes SLIC. Report incomplete stages and phase termination reasons. |
| 5–10 varied qualitative test cases | Retained, with disclosed post-test example-selection rules. They do not affect model selection. |
| WT/TC/ET modality explanations and hop distributions | Retained. Exact 16-coalition feature-replacement Shapley on ten fixed test patients; effective-hop and gate diagnostics. Graph/fixed-topology scope only. |
| WT/TC/ET Dice, HD95, sensitivity, precision, IoU; mean ± SD | Retained on the full main test cohort, plus raw/processed scores and patient-paired intervals. No seed-variation claim from one seed. Custom empty-region HD95 convention remains documented in code. |
| All four modalities; explain T1ce/FLAIR SLIC | Four modalities remain node features; T1ce/FLAIR determine partitions. Explain that design and its limitations. |
| Appropriate novelty and literature comparisons | Both supplied papers remain prior context. No numerical superiority claim against different published partitions; no claim that graph+CNN, supervoxels or explainability alone are novel. |

What was cut: structural refinement, masked reconstruction, three seeds per row,
separate descriptor/correspondence/soft-label/loss-family/hop-penalty/uniform-hop
sweeps, duplicate full-pipeline CNN retraining for graph controls, and full-cohort
multi-seed retraining at every SLIC resolution. Claims about those removed
mechanisms are also removed.

The 60-hour constraint and a fully converged, exhaustive advisor study cannot both
be guaranteed without timings. This package implements a smaller, transparent
study within declared limits. A journal submission still depends on adequate
learning curves, meaningful measured effects, honest limitations and the advisor's
assessment. It does not certify publication readiness.
