# Frozen continuation experiment plan

Default: 17 configurations, three seeds (Part 1 seed and the next two integers),
the exact saved full train/validation/test split, full-volume validation selection,
100-epoch caps and patience 15. The main seed reuses the actual selected Part 1 and
Part 2 checkpoints. Other seeds and changed components train independently.

| Default row | Question addressed | Part |
|---|---|---|
| full | Complete proposed system, including active masked reconstruction | 3 |
| no_structural | Does structural refinement and its auxiliary supervision help? | 3 |
| no_descriptors | Do the predicted structural descriptors help beyond the refinement module? | 3 |
| fixed_shared_hops | Does learned depth mixing help versus the same shared operator at fixed depth? | 3 |
| hgt_only | What does extra propagation add after HGT? | 3 |
| no_cross_edges | Do cross-partition correspondence edges help? | 3 |
| hard_labels | Do fractional supervoxel targets help? | 3 |
| no_boundary | Does voxel boundary supervision help? | 3 |
| no_voxel | Graph-only versus graph-plus-CNN, with identical graph weights | 3 |
| cnn_only | Does the graph add value beyond an MRI-only refinement CNN? | 3 |
| graphsage_backbone | Ordinary homogeneous GraphSAGE versus typed HGT in the matched pipeline | 3 |
| dice_ce | Dice + weighted CE versus the declared focal/boundary loss family | 3 |
| no_masked_reconstruction | Does the active masked feature/link objective help? | 3 |
| unet_3d | Independent lightweight 3D U-Net, same patients and full-volume evaluation | 3 |
| slic_5000 | Segmentation performance versus SLIC resolution and cost | 4 |
| slic_10000 | Same comparison | 4 |
| slic_20000 | Same comparison; 15k is the full row | 4 |

The U-Net has its own architecture and no boundary head; it is an independent
baseline, not a single-component ablation or a substitute for a tuned nnU-Net.
The GraphSAGE row keeps the applicable auxiliary objectives; it is a backbone
control, not an exact reproduction of the reference paper. Removing correspondence
edges also removes held-out correspondence reconstruction targets from that graph;
describe that row as removing the correspondence information pathway.

Optional extended rows are `fixed_shared_max`, `uniform_shared_hops` and
`no_hop_penalty`. They distinguish maximum depth, learned mixture and regularization
effects more finely. They are not essential to the advisor's requested fixed/adaptive
comparison. Without these rows, avoid claims that separately establish the benefit
of the hop penalty or superiority over a uniform mixture. Set the extended switch
identically in Parts 3–5 before starting Part 3 if those claims are central.

## Advisor evidence retained

- The inherited audited 1251-case cohort and disjoint saved patient IDs remain
  authoritative. No arbitrary case is dropped to meet the count.
- Main Part 1/2 histories and curves retain the best validation epoch. Research
  histories/curves are exported separately; node/patch diagnostics are not labelled
  as full-volume Dice.
- Graph-only/CNN-only/hybrid, HGT/GraphSAGE, independent voxel, loss, structural,
  reconstruction and trained 5k/10k/15k/20k comparisons remain enabled.
- Raw and consistently postprocessed patient Dice, HD95, sensitivity, precision
  and IoU are retained, with patient means/SD, paired confidence intervals, seed
  variability and multiplicity-corrected comparison tests.
- Parameter counts, training/inference measurements, GPU allocation and graph
  sizes are recorded. Cached-graph inference excludes SLIC and is labelled as such.
- Up to ten paired graph/CNN/error examples have disclosed post-test selection rules.
- Direct WT/TC/ET probability Shapley uses all 16 modality coalitions, a training
  reference and efficiency checks. It explains graph features at fixed topology,
  not the full CNN pipeline or modality effects on SLIC.
- Hop distributions by node region, gate boundary/interior statistics, local
  explanation graphs and reconstruction-versus-simple-baseline diagnostics are
  saved in Part 5. These are descriptive, not causal proof.

The code preserves the custom finite empty-region HD95 policy. It does not claim
official BraTS evaluator parity. Report the policy and internal cohort explicitly.
