# Run convergence protocol v9 (retrain all six rows)

v9 replaces the 135-minute training allowance. Every main row now trains to an
epoch cap with early stopping on validation Dice and keeps the best-validation
checkpoint. Nothing stops training because of a time budget.

| Setting | Graph stage | CNN stage |
|---|---|---|
| Epoch cap | 100 | 100 |
| Early stopping | patience 15 on full-volume validation Dice | patience 15 |
| Checkpoint kept | best validation mean WT/TC/ET Dice | same |
| Training-time cap | none | none |

Recorded `stopping_reason` is `early_stopping` or `epoch_cap` only. `compute_budget`
now exists only for the separate SLIC pilot (Part 4), which keeps its declared
45-minute allowance and is carried over unchanged.

## Sessions pause and resume

Molab sessions still end. Each part saves after every completed epoch (pushed to
the Hub every 5 epochs and on pause). Before starting an epoch it checks whether
that epoch fits in the session's remaining 9.5-hour work budget; if not, it stops
with a message that says so. **Rerun the same notebook in a new session and it
resumes from the last completed epoch**, with the same optimizer, scheduler and
RNG state. A pause never marks training complete.

Every epoch prints its duration and the most time left to the epoch cap, so you
can see how many sessions a part needs.

## Order

| Step | Notebook | Trains | Rerun until it prints |
|---|---|---|---|
| 0 | `preflight_v9` | nothing (tests, ~2 min) | `PREFLIGHT PASSED` |
| 1 | `part1_corrected` | main HGT graph stage (`full`, `no_voxel` share it) | `Part1 handoff ready` |
| 2 | `part2_refinement` | `full` CNN stage, `cnn_only` | `PART 2 COMPLETE` |
| 3 | `part3_experiments` | `graphsage_backbone`, `fixed_shared_hops`, `segresnet` | `THIS PART COMPLETE` |
| 4 | (skip) | the SLIC pilot from v8 is reused as-is | |
| 5 | `part5_reports` | nothing; held-out test evaluation, XAI, reports | final export receipt |

Keep `monai==1.5.1` installed for Parts 3 and 5. Fill `HF_TOKEN_PLACEHOLDER` locally
and clear it before exporting any notebook to HTML.

## What is reused from the Hub

* The audited 15k graph cache (keyed by graph identity; unchanged).
* The v8 SLIC pilot receipt and its research store under
  `continuation_v8/670ec244f0e27740def0/` (Part 5 finds it automatically; set
  `PILOT_SOURCE_PREFIX_OVERRIDE` to choose explicitly).

## What is new on the Hub

* Part 1 trains under `stage1_convergence_v9/runs/...` and publishes its handoff
  to `stage1_convergence_v9/final`. The v8 handoff under `stage1_practical_v8/`
  is left untouched and is no longer read by Parts 2-5.
* Parts 2-5 write to `continuation_v8/<new Part 1 model sha>/...`, a new folder.

## GPU use

The CNN stage now projects graph priors to voxels, accumulates sliding-window
probabilities and takes the arg-max on the GPU (same arithmetic as before; see
`test_gpu_equivalence.py`). The effective batch of 16 patches can run in one
GPU pass (`VOXEL_MAX_MICROBATCH = 16`), inference batches up to 16 windows, and
calibration may use up to 85% of free VRAM. The number of optimizer updates per
epoch is unchanged. Validation post-processing and Dice run in CPU threads while
the GPU moves on to the next patient.

## Reading results

Read `stopping_reason`, the best epoch and the four curves for every model:
train/validation loss and train/validation Dice (graph stage: `train_loss`,
`val_total`, `train_node_proxy`, `val_dice`; CNN stage: `train_loss`, `val_loss`,
`train_patch_dice`, `val_dice`). `epoch_cap` means validation Dice was still
improving within the last 15 epochs at epoch 100; report it as such.

Do not tune on test results. Single-seed limitations still apply.
