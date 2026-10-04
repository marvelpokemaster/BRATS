# Frozen practical study

Main seed: 42. Main subjects: the complete audited Part 1 train/validation/test
split. Main models: 100-epoch maximum, patience 15, 135-minute phase allowance,
best completed full-volume validation Dice checkpoint. All comparisons use the
same declared allowance; achieved epochs and preparation costs can differ.

| Row | Training work | Scheduled session |
|---|---|---|
| full | Main graph then hybrid voxel CNN | 1 and 2 |
| no_voxel | Reuse exact main graph; no training | 2 |
| cnn_only | MRI-only version of the same voxel refinement head | 2 |
| graphsage_backbone | Graph-only GraphSAGE; compare with no_voxel | 3 |
| fixed_shared_hops | Graph-only shared two-hop HGT control; compare with no_voxel | 3 |
| segresnet | Independent MONAI SegResNet voxel baseline | 3 |

The full/graph-only pair tests adding the CNN; full/CNN-only tests the hybrid
against the same CNN family without graph inputs. GraphSAGE/fixed-hop controls
test graph-branch behavior, not fully retrained hybrid architectures. The report
contains a separate no_voxel-reference comparison table for those controls.

Session 4 has its own experiment identity and validation-only data: 64 training
subjects and 16 validation subjects sampled once with the fixed seed, fresh
5k/10k/15k/20k graph builds/training, graph-only, maximum 20 epochs and 45 minutes
per resolution. No main test subjects, no imported 15k trained weights, and no
automatic change to the 15k main method based on this pilot. It provides
preliminary cost/accuracy evidence only. Unequal achieved epochs are disclosed.

Session 5 evaluates the six main rows and reports the pilot separately. Do not
interpret single-seed patient intervals as training-seed robustness. Do not call
a budget-stopped model converged or use test results to choose the architecture.
