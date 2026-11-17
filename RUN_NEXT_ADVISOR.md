# Advisor follow-up (Advisor.md, 7 Nov): what to run, in what order

**Status: written and re-verified statically (12 Nov), NOT executed.** The authoring environment had no PyTorch, PyG, marimo,
GPU or Hugging Face access, so nothing could be run there. Checked: valid Python; marimo DAG rules (AST checker, same result as the
working Part 5 notebook); the prefix cells match the working Part 5 notebook (52) except for the intended changes; every call into
the v9 helpers (`ResearchRunner`, `fit_graph`, `fit_voxel`, `train_run`, `evaluate_run`, store `read/save/flush`, metric keys)
was read against the v9 source. Numpy helpers were unit-tested. Do a smoke test first (below).

**Revision 15 Nov (second Molab run)**
The 14 Nov handoff fix worked: Part 2 handoff passed (only the provenance labels differ: the saved Part 2 config says
`compute_protocol: practical-v8`, `phase_seconds_limit: 8100`; every training hyper-parameter matches). The run then stopped with
`ImportError: cannot import name 'discover_main_plan_key' from 'brats_workflow' (/marimo/brats_workflow.py)`: the helper files on
that Molab session were the OLD set (BRATS_Part5_Molab_v11, 4.5 KB brats_workflow.py). Upload ALL NINE helpers from
`Downloads\BRATS_v9` (brats_bundles, experiments, gpu, graph, practical, protocol, reporting, transfer, workflow), overwrite the old ones,
restart the kernel. Both notebooks now run a preflight that names any old helper and compares each file with the v9 SHA-256 values.

**Revision 14 Nov (after the first Molab run)**
The first run of `part3b_graphsage_hybrid.py` stopped at `Part 2 handoff mismatch: stage2_config` (the saved Part 2 CNN config differs
from the one in the notebook); every later cell failed as "ancestor raised". That run used the 12 Nov file's predecessor, so it did not
say which entry differs. Both notebooks now:
1. use the Hub copy if a stale local copy of the Part 2 state is the cause (the local file is kept as `.stale_local_backup`);
2. tolerate a difference ONLY in provenance labels (`compute_protocol`, `phase_seconds_limit`, `gpu_training_protocol`, `precision_policy`),
   print a WARNING and record it in the saved JSON (`part2_handoff_notes`);
3. stop with the exact differing entries if a training hyper-parameter differs (epochs, lr, channels, loss weights, batch size, ROI, overlap).
   In that case do NOT work around it: the new GraphSAGE+CNN row must be trained with the same CNN settings as `full`.
4. if the plan check (`Running configuration differs from plan ...`) fails afterwards, print the differing `run_config` entries.

**Changes in this revision (12 Nov)**
1. **Part 2 handoff cell** (in both notebooks): the saved Part 2 state is compared by content (tuple = list, numpy scalar = Python
   scalar) and, if it still differs, the error now names the field, both types and the differing entries instead of only
   "Part 2 handoff mismatch: <field>". If you still see it, the printed entries are the real cause (e.g. a Part 2 run saved with a
   different config) - send that line.
2. **Results were not being committed to the Hub.** `store.save(..., push=True)` only commits every `ZIP_SYNC_MINUTES` (120), so a
   Molab session ending earlier would have lost the new checkpoints/results. Both notebooks now call `store.flush()` after training,
   after test evaluation, after every saved JSON and at the end.

## Files (put beside the existing helper `.py` files; Molab cannot upload folders)

| File | Answers | Trains? | Rough GPU time (estimate) |
|---|---|---|---|
| `part3b_graphsage_hybrid.py` | Q1: typed HGT vs homogeneous GraphSAGE **with the same CNN** | CNN stage only (GraphSAGE graph weights reused from the Hub) | ~5-9 h training, then ~10-20 min test inference; resumable across sessions |
| `part6_advisor_tests.py` | Q2 (which CNN errors the graph fixes, esp. ET) and Q3a/3b/3c (gate, Shapley and hop faithfulness) | No, inference only | ~2-5 h depending on `ADV_N_PATIENTS` |

Both start with the same cells as Part 5 (they discover and adopt the recorded v9 plan key, so every checkpoint already on
Hugging Face is reused). Neither changes any existing row, split, or result.

## Run order
1. Fresh Molab session, GPU, helper files beside the notebooks, HF token entered only in the session (never saved in the notebook).
2. **Smoke test Part 6 first:** set `ADV_N_PATIENTS = 3` in the settings cell, run end to end (~20-30 min), fix any error, then set it back to `60` and rerun. Keep the final value fixed before looking at any result.
3. Run `part3b_graphsage_hybrid.py` until the training cell prints "COMPLETE" (if it prints "PAUSED", open a new session and rerun; it resumes from the last epoch). The comparison cell then prints the table.
4. Run `part6_advisor_tests.py`.
5. Outputs are pushed to the research store on the Hub under `advisor_followup/`: `hgt_vs_sage_with_cnn.json`, `cnn_error_correction.json`, `edge_perturbation.json`, `modality_intervention.json`, `hop_sensitivity.json`, `et_corrections.png`, `index.json`.
6. Before exporting any notebook to HTML: clear the token field and rerun nothing after clearing.

## What each result means
- **Q1** (`hgt_vs_sage_with_cnn.json`): paired differences full - graphsage_hybrid on the same 189 test patients, 95% patient bootstrap CI, Holm within this one comparison (HD95 difference is a - b, so a *negative* value favours `a`). If the CI includes 0, report "no consistent advantage of the typed graph once the CNN is added", as for the graph-only comparison.
- **Q2** (`cnn_error_correction.json`, `et_corrections.png`): voxels/islands the CNN-only model gets wrong that the full model gets right (and the reverse), ET false negatives recovered and false positives removed, per-patient ET Dice gain with CI, and whether the corrections disappear when the cross-modality edges are removed.
- **Q3a**: removing the top-gate edges vs the lowest-gate edges vs random edges, same number removed in every relation. Gates are faithful only if high-gate removal hurts clearly more than low-gate and random removal.
- **Q3b**: Shapley modality rankings vs the Dice change under dropout, noise and bias-field perturbations of each modality (Kendall tau, top-1 agreement).
- **Q3c**: forcing 0/1/2 hops and dropping each hop's messages; relates prediction change to the learned per-node hop weights.
- Wording until these run: call the gates, Shapley values and hop weights "diagnostics", not "faithful explanations".

## Verified against the source (12 Nov) vs still only provable by running
- Verified: `meta["vis"]` is the cropped float16 modality volumes (`build_and_save`); `QoSHRGN.last_edge_gates` is a list with one dict per hop and `last_hop_info[nt]` has `weights` / `effective_hop` / `selected_hop`; `hop_selector[nt]` is called in hop order 0..K; `graphsage_hybrid` reuses the saved GraphSAGE checkpoint (graph identity ignores name / voxel / voxel_kind); metric keys are `Dice_WT`, `HD95_ET`, ...
- Still only provable by running: the brain mask is `node_map >= 0` (a rebuild-check cell asserts mean abs feature difference < 0.05); `fit_voxel(spec, seed, groups, None)` returns the saved `cnn_only` checkpoint without retraining; plan-key adoption writes `amendments/<key>.json` to the Hub store (expected; it records that helper sources changed since training).

## Housekeeping flagged earlier
- Revoke the Hugging Face token that appears in the exported notebook (52); notebooks must be exported with the token field blank.
- Abstract: voxel-stage validation Dice should read 87.66% (0.876649); confirm the quoted test numbers are the post-processed ones from the report JSON.
- `graphsage_hybrid` is a post-hoc, advisor-requested row with its own Holm family; say so in the paper. Its checkpoint is selected on validation only.
- The BRATS clone still has 5 unpushed commits; nothing was pushed.
