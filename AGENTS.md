# AGENTS.md: QoS-HRGN (BraTS 2021) project state, rules and roadmap

**Last verified: 24 Nov 2026 (re-checked in the evening against GitHub and the laptop).** Settings below were checked against the code in this folder, not copied from older notes.
Read this file first, then `AGENT_GUIDELINES.md` (marimo rules). Do not revert anything in section 3 without the owner's permission.

Project: QoS-HRGN, brain-tumour segmentation on BraTS 2021 (1,251 patients; split 875 train / 187 val / 189 test, seed 42).
Pipeline: SLIC supervoxels (15,000 per modality, T1ce + FLAIR) -> typed hetero-graph (3 relations) -> 2 HGT layers (4 heads, hidden 128)
-> gated adaptive propagation (K_MAX = 2, learned per-node hop mixture) -> voxel projection -> 3D CNN refinement -> ET island cut (< 300 voxels).
Team: Adarsh Binu, Pranav Pradeep, Sharoon Joseph, Sourav Sasiraj. Guide: Ms. Asha Ashok. Stage: S7 final review was completed on 18 Nov 2026 (report and deck are done and out of scope for further work);
code of record is `marvelpokemaster/BRATS` commit `5476be2` (merged to `main` as `551b1ed`).

---

## 1. Critical rules

* **No heavy compute on the laptop.** Training and preprocessing run on Molab (RTX PRO 6000 Blackwell). Laptop work is editing, checking and packaging only.
* **Network commands use `proxychains`** (git push, pip install, etc.). Owner's standing rule.
* **Never commit or export secrets.** Hugging Face tokens must be blank in every notebook before committing or exporting to HTML (see roadmap item A3).
* **Reuse what has already been run.** Finished runs are on the Hugging Face repo. Assume they are there and reuse them; do not retrain to "regenerate" code.
* **Correctness over speed.** Never state a number that was not read from a saved result. Do not tune on test results. Single-seed limits apply to every claim.
* **Marimo rules for any notebook edit:** names starting with `_` are private to their cell; every other global is defined in exactly one cell;
  cross-cell names must be cell parameters and returned by the defining cell. `.ipynb` twins use 0-space indentation.
  Run `python tools/check_notebook_parity.py <part>` for each part you touch (pass part names one at a time; running it with no argument fails).
* **Commit or push only when the owner asks.** Do not edit the original `qos-hrgn-dashboard` folder; all dashboard work goes in `qos-hrgn-dashboard-v2`.
* **Never mention the AI assistant** in the report, slides, deliverable files or their metadata (owner's standing rule).

## 2. Current configuration (verified in code, 24 Nov 2026)

| Setting | Value | Where |
|---|---|---|
| Cohort | `MAX_CASES = None` (full 1,251; split 875/187/189) | parts 1-3 |
| Research cap | `RESEARCH_MAX_CASES_PER_SPLIT = 0` (full saved cohort; the old 300-case cap is gone) | part 2 |
| Masked reconstruction | `USE_MASKED_RECONSTRUCTION = False` (removed from the active method) | parts 1-3 |
| Baselines / ablations / XAI | `RUN_BASELINES = True`, `RUN_ABLATIONS = True`, `RUN_REGION_XAI = True` | part 2 |
| Epochs and stopping | cap 100 (graph) and 100 (CNN); `EARLY_STOP_PATIENCE = 15`; best-validation checkpoint | parts 1-3 |
| Checkpoint push | every 5 epochs (`CKPT_PUSH_EVERY_EPOCHS = 5`) and on pause | parts 1-2 |
| Batch sizes | `BATCH_SIZE = 8`; `VOXEL_EFFECTIVE_BATCH_SIZE = 4` in part 1, `16` in parts 2-3 | parts 1-3 |
| Prefetch workers | `VOXEL_PREFETCH_WORKERS = 2` in part 1, `12` in parts 2-3 | parts 1-3 |
| GPU memory | `GPU_MEMORY_FRACTION = 0.85`, `GRAPH_MEMORY_FRACTION = 0.85` | part 2 |
| Advisor tests | `ADV_BOOT_DRAWS = 5000`; `ADV_N_PATIENTS = 60` in the repo script, but the executed run and the reported results used 30 (see A-done-9) | part 6 |
| Statistics | patient-paired bootstrap (5,000 draws, seed 1942), sign-flip p, Holm | `brats_experiments.py` |

The previous version of this file listed masked reconstruction as True, batch size 64 and a 300-case cap. Those were wrong for the final runs and have been corrected.

## 3. Fixes and decisions that must not be reverted (all present in current code)

* **Kaggle duplicate bug.** The Kaggle mirror's `BraTS2021_00495.tar` and `BraTS2021_00621.tar` contain loose files identical to copies in the main tar.
  All parts use `select_canonical_nifti` (keeps the foldered copy) and filename-based `case_id_from_path` from `brats_protocol.py`; orphan cache cleanup uses
  `DATASET_AUDIT["cases"]`. Without this the pipeline builds 1,250 graphs instead of 1,251. Regression test: `tests/test_case_ids.py`.
* **HF naming.** `STAGE1_REMOTE_NAME = "stage1_graph_model.pt"`, matching the files actually on the owner's Hugging Face repo. Do not restore the old `_review_v3` suffixes.
* **SegResNet odd-size crash.** Inputs are padded to a multiple of 8 before the forward pass and cropped back afterwards (training in part 3, inference in part 5).
* **Convergence protocol v9.** No training-time cap. `stopping_reason` is `early_stopping` or `epoch_cap`; `convergence_demonstrated` is true only for `early_stopping`.
  A session deadline (`BudgetPause`) saves the last completed epoch and never finalizes a phase; rerunning the notebook resumes with optimizer, scheduler and RNG state.
  The SLIC pilot keeps its 45-minute allowance (`PILOT_SECONDS`). Tests: `tests/test_convergence_protocol.py`.
* **Hub namespace.** Part 1 trains under `stage1_convergence_v9/runs/...` and publishes to `stage1_convergence_v9/final` (`brats_graph.PART1_FINAL_PREFIX`).
  The v8 handoff (`stage1_practical_v8/`) is left intact. Parts 2-5 write to `continuation_v8/<Part 1 model sha>/...`.
* **GPU data path.** CNN training sends compact patches and `brats_protocol.materialize_examples` builds batches on the device; sliding-window inference is device-resident.
  Both must stay bit-identical to `make_training_patch` and `sliding_window_predict_reference` (`tests/test_gpu_equivalence.py`).
  `VOXEL_MAX_MICROBATCH = 16`, `INFERENCE_MAX_BATCH = 16`.
* **Plan adoption.** Follow-up notebooks adopt the recorded v9 plan key (`brats_workflow.discover_main_plan_key`) so existing checkpoints are reused (`tests/test_plan_adoption.py`).
  Upload all nine helper `.py` files to Molab; an old helper set causes `ImportError: cannot import name 'discover_main_plan_key'`.
* **SLIC pilot carry-over.** Part 5 reads the v8 pilot receipt via `brats_workflow.find_pilot_receipt` if the new store has none.
* **Results are committed to the Hub with `store.flush()`**; `store.save(..., push=True)` alone only commits every 120 minutes.
* **HD95 convention** (custom, not the official evaluator): 95th percentile of pooled bidirectional surface distances in voxels; both masks empty -> 0; one empty -> volume diagonal (about 371 for 240x240x155).

Older v5 notes (a `checkpoint_revision` manifest fallback and a flexible state-dict loader in `part2_corrected.py`) could not be found in the current code and have been dropped.
If a legacy checkpoint fails to load, investigate rather than assuming those patches exist.

## 4. Repository map

| Path | Purpose |
|---|---|
| `part1_corrected` .py/.ipynb | graph stage: SLIC graphs, HGT training, Part 1 handoff |
| `part2_refinement` | `full` CNN stage and `cnn_only` |
| `part3_experiments` | `graphsage_backbone`, `fixed_shared_hops`, `segresnet` |
| `part4_slic_study` | 5k/10k/15k/20k SLIC pilot (64 train / 16 val patients, one seed, max 20 epochs) |
| `part5_reports` | held-out test evaluation, Shapley XAI, final reports export |
| `part3b_graphsage_hybrid.py` | follow-up: GraphSAGE + CNN row (post-hoc, advisor-requested; own Holm family). **No `.ipynb` twin.** |
| `part6_advisor_tests.py` | follow-up: CNN-error correction (Q2) and faithfulness tests 3a/3b/3c. **No `.ipynb` twin.** |
| `verify_hub_results_v4.ipynb` | verifies the six rows on the Hub (checks 875/187/189 split, 189 patients per row, no duplicates) |
| `preflight_v9.ipynb` | tests and helper-file check before any run (`PREFLIGHT PASSED`) |
| `brats_*.py` (9 helpers), `baseline_3d_unet.py` | shared code; upload all helpers together |
| `tests/`, `tools/` | regression tests; notebook parity checker (supports parts 1-5 only) |
| `RUN_NEXT.md`, `RUN_NEXT_ADVISOR.md`, `RESEARCH_PLAN.md`, `RESEARCH_PROTOCOL.md`, `ADVISOR_SCOPE.md`, `Advisor.md` | run order, study design, advisor requirements |
| `SHA256SUMS.json`, `source_provenance.json` | integrity and provenance records |

Outside this folder: Hugging Face model repo `marvelpokemaster/brats-qos-hrgn-model` (final reports at `continuation_v8/b36d43872a3781687ac7/final_reports.zip`);
dashboard v2 at `Brats new\qos-hrgn-dashboard-v2` and on GitHub `marvelpokemaster/qos-hrgn-dashboard` `main` (PR #2 at `25b8884`, then the 24 Nov per-patient GraphSAGE+CNN and training-curve update, pushed, merged and checked identical to local); Hub extraction helper `pull_followup_from_hub.py` and its output `hub_extract\` in `Brats new`; report, deck and Word/zip sources in `Downloads\Brats_review_final`.

## 5. Results of record (test n = 189, seed 42; from the report)

| Model | Dice WT / TC / ET | Mean | HD95 WT / TC / ET |
|---|---|---|---|
| QoS-HRGN (full) | 92.26 / 89.79 / 85.84 | 89.30 | 5.46 / 4.51 / 11.61 |
| GraphSAGE + CNN | 92.14 / 89.65 / 85.73 | 89.18 | 5.01 / 4.66 / 13.57 |
| SegResNet | | 89.15 | |
| CNN only | | 88.15 | |
| Graph only | | 80.22 | |

Paired (patient bootstrap, 95% CI; Holm):
full - CNN-only +1.15 (Holm 0.027; ET +1.78, WT +0.55, TC +1.12);
HGT - GraphSAGE+CNN +0.12 [-0.28, +0.57] (Holm 1.0, tie);
GraphSAGE+CNN - CNN-only +1.03 [+0.28, +1.83] (Holm 0.035).
The last two rows come from the follow-up run; their Holm adjustment is within that run (7 metrics per comparison), not the 20-comparison main family.
Per-patient test scores for GraphSAGE+CNN were pulled from the Hub on 24 Nov (`evaluation/test/graphsage_hybrid/seed_42.json`: 189 unique patients, complete, no missing values) and re-checked.
Recomputed means match the report (Dice 92.14 / 89.65 / 85.73, mean 89.177; HD95 5.01 / 4.66 / 13.57), and the follow-up file reproduces the paired results above (+0.123 [-0.280, +0.572] Holm 1.0; +1.028 [+0.283, +1.827] Holm 0.0348).
In the same follow-up family, full - CNN-only is +1.151 [+0.343, +2.022] with Holm 0.038; the report cites the main-family value (Holm 0.027, interval +0.36 to +2.02). Both are consistent; different bootstrap draws and Holm families.
ET gain, +1.78 vs +1.28: +1.776 points over all 189 patients vs +1.281 over the 184 with enhancing tumour. Five test patients have no ET in the ground truth; four tie, and one (BraTS2021_01666) is Dice 1.0 for the full model vs 0.0 for CNN-only (+0.53 points).

Faithfulness (report Table "faith"): 3a gates supported (high-gate edge removal flips 0.169 of tumour nodes vs 0.001 low-gate, 0.013 random; Holm 0.0096);
3b partly supported (Shapley tracks Dice damage under dropout, tau +0.43 to +0.53; weak under noise and bias field);
3c weak (two-thirds of nodes have effective hop >= 1.90 of 2, so little variation to explain). Dice-damage effects use a 30-patient subset.

## 6. Roadmap

Status key: DONE / OPEN. Priorities: P0 do first, P1 soon, P2 science that strengthens the paper, P3 hygiene.

### A. Done

* A-done-1: Data audit and fixed split (1,251 patients, 875/187/189) with duplicate fix and regression test.
* A-done-2: Convergence protocol v9 runs for all six rows (Parts 1, 2, 3, 5); final reports exported and on the Hub.
* A-done-3: SLIC resolution pilot (Part 4, carried over from v8), reported separately as preliminary evidence.
* A-done-4: Advisor follow-up: GraphSAGE+CNN row (`part3b`) and Q2/Q3 tests (`part6`); results are in the report.
* A-done-5: Verification notebook, parity checks (5/5 parts, 0 mismatches), regression tests present.
* A-done-6: S7 report (45 pages: PDF, Word, source zip) and deck v7; literature table cross-checked against the papers (HGT range corrected to 9-21%, nnU-Net batch-size claim removed, GraphSAGE metric set to micro-F1).
* A-done-7: Code pushed and merged: `marvelpokemaster/BRATS` `5476be2` -> `main` `551b1ed`.
* A-done-9: P1 doc and test items cleared on 24 Nov: stale status text fixed in `README.md` and `RUN_NEXT_ADVISOR.md` (and the `README.md` entry in `SHA256SUMS.json` refreshed, 32/32 checksums match); the 30-vs-60 patient setting confirmed from the executed notebook (the run used 30, the repo script still says 60; reported results use 30, so the script was left unchanged to keep it identical to commit `5476be2`); and a fresh clone of the dashboard `main` loaded all 8 tabs with 0 failed requests, 0 console errors and no broken images (headless browser test).
* A-done-10: GraphSAGE+CNN per-patient data added to dashboard v2 (local, not yet pushed). Files pulled from the Hub with `pull_followup_from_hub.py` (in `Brats new`; output in `Brats new\hub_extract`, no token inside).
  New script `add_graphsage_cnn_patients.py` adds per-patient rows, exact means and the full paired comparison; `extract_and_prepare.py` now runs it after the summary-row step. The Benchmark view shows GraphSAGE+CNN in the patient-level plots and table (mean Dice 89.18, was 89.17); it stays out of the cost table (no params/timing record).
  Headless-browser test of all 8 tabs: 0 failed requests, 0 console errors, no broken images. Backups of the edited files: `Brats new\v2_backup_24Nov`. The ET 184-vs-189 difference is explained (section 5).
* A-done-8: Dashboard v2 built (Validate tab, GraphSAGE+CNN benchmark row). Pushed as `5ae8bac` and merged via PR #2 into `main` (`25b8884`, 24 Nov). On `main`: `static/js/validate.js`, `data/validation.json`, and all 76 manifest image paths. The report and deck links already point to this repo.

### B. Open

**P0: do first**
* **A3. Revoke exposed Hugging Face tokens.** All 13 notebook HTML exports in `Brats new` contain token-like strings. None are in `Brats_final` or the dashboard repo (both scanned). Revoke the token(s) on Hugging Face, create a new one, and keep it out of every file.
  Do not push the HTML exports anywhere. The owner is handling this; it cannot be verified from here.

**P1: soon**
* B7. Done (24 Nov 2026): dashboard v2 per-patient GraphSAGE+CNN update and training curve tile pushed, merged, and local files checked identical to GitHub `main`.

**P2: strengthens the paper**
* C1. Repeat main rows with at least 3 seeds (current claims are single-seed, one split).
* C2. External validation on another cohort or BraTS year; compare HD95 with the official BraTS evaluator (current convention is custom).
* C3. Run the faithfulness Dice-damage tests on more than 30 patients (all 189 where affordable).
* C4. Hop mixture: most nodes sit near the cap (effective hop >= 1.90), so test three and four hops (K_MAX > 2) and a hop regularizer; the current 3c result is weak.
* C5. GraphSAGE+CNN with multiple seeds before any claim about HGT vs GraphSAGE beyond "no consistent difference".
* C6. Journal manuscript drafting; keep "diagnostics" wording where faithfulness is partial.
* C7. Boundary and false enhancing tumour (report Future work 1): switch on the boundary loss and head, which are off in the final runs (`boundary_weight = 0.0` by default), and tune the ET post-processing (`ET_MIN_VOXELS = 300` in part 2).
  Judge it on ET HD95 (11.61 for ET against 4.51 for TC) and false-ET voxels.
* C8. Stronger graph (report Future work 2): richer node features such as pooled CNN features, and joint graph-CNN training instead of two separate stages.
* C9. Converge the CNN stage (report Future work 4 and Limitations): the CNN ended at the epoch cap, so convergence is not shown. Extend training past 100 epochs and report again.
* C10. Baselines and literature (report Future work 5): train nnU-Net on the same 875/187/189 split (only SegResNet was trained on it), and run a systematic literature search to confirm the novelty statement.
* C11. SLIC resolution study at full scale (report Future work 6): the 64/16-patient pilot is inconclusive, so repeat 5k/10k/15k/20k on the full cohort before changing the 15k default.
* C12. Explanation and timing gaps (report Limitations): Shapley is feature-level and does not cover the effect of the T1ce/FLAIR supervoxel partition; inference time excludes graph construction (measure end to end);
  the +1.28 vs +1.78 ET gain difference is now explained (see section 5).

**P3: repo hygiene**
* D1. Add `.ipynb` twins for `part3b` and `part6`, and extend `tools/check_notebook_parity.py` (it only accepts parts 1-5; it also fails when run with no arguments).
* D2. `Aczr005/BRATS` was the fork used for PR #6 (`final-submission` branch), which is merged into `marvelpokemaster/BRATS` `main`. Confirm the fork holds nothing that `main` lacks; the report cites only the marvelpokemaster repo.
* D3. Regenerate `SHA256SUMS.json` and `source_provenance.json` after any code change.
* D4. Dashboard repo: `data/extracted` is already tracked on `main` (277 files, 26 PNGs) even though v2's `.gitignore` lists it, so the ignore rule does nothing for those files.
  Decide whether to keep them tracked (fresh clones work without extraction) or untrack them (`git rm -r --cached data/extracted`); the README run instructions must match the choice.
* D5. Delete or archive the local clone `qos-hrgn-dashboard-v2\dash_push` once it is no longer needed (it sits inside the v2 folder, which is easy to copy by mistake).
* D6. Optional: host the dashboard (`vercel.json` exists) so there is a live demo link as well as the repo.

## 7. Claim discipline (what the evidence supports)

* Supported: the graph prior adds +1.15 mean Dice over the same CNN (Holm 0.027), mostly on ET; the model matches SegResNet on overlap; edge gates are faithful (test 3a).
* Not supported: an accuracy advantage of HGT over GraphSAGE (+0.12, CI includes 0), or over SegResNet or fixed-hop controls.
* Weak or partial: Shapley agreement (3b) and hop weights (3c). Call these diagnostics, not faithful explanations.
* Limits to state every time: one seed, one split, custom HD95, a 30-patient subset for Dice-damage tests, and a separate Holm family for the GraphSAGE+CNN rows.

## 8. Change log

* 24 Nov 2026 (B7 done): pushed and merged the dashboard v2 update (per-patient GraphSAGE+CNN rows, paired stats, ET note, curve tile); confirmed local files identical to the repo. P1 is now empty.
* 24 Nov 2026 (B8 done): the Hub `runs/graphsage_hybrid/seed_42/trained.json` holds per-epoch history (graph 88 epochs, early stopping, best epoch 73; voxel 100 epochs, epoch cap, best epoch 86). Drew `graphsage_cnn.png` in the existing tile style (training objective and full-volume validation Dice, both stages), registered it under `images.curves` (idempotent step `register_curve` in `add_graphsage_cnn_patients.py`) and verified it is served. Local in dashboard v2 only; goes out with B7.
* 24 Nov 2026 (curves): added `--find` and `--get` options to `pull_followup_from_hub.py` and opened item B8 (GraphSAGE+CNN training curve).
* 24 Nov 2026 (B2 pass): pulled and verified the GraphSAGE+CNN per-patient files, added them to dashboard v2 locally, explained the ET 184-vs-189 difference, and added open item B7 (push). Edited dashboard files only; no repo code changed.
* 24 Nov 2026 (P1 pass): cleared B1, B4 and B6; B2 stays open because the Hub files are inside ZIP archives and need an authenticated extraction. A3 is owner-handled. Edited `README.md` and `RUN_NEXT_ADVISOR.md` status lines only (backups in `Downloads\docs_backup_24Nov`).
* 24 Nov 2026 (later): added C7-C12 so the roadmap covers every item in the report's Future work and Limitations sections; extended C4 to three and four hops.
* 24 Nov 2026 (evening): dashboard v2 pushed and merged (PR #2, `main` `25b8884`); moved those items to done, added B6 and D4-D6, and re-verified GitHub state, manifest image paths and token scans.
* 24 Nov 2026: rewrote from the v5 "post-v5" changelog to a current state, rules, verified configuration, results and roadmap. Corrected wrong settings (masked reconstruction, batch sizes, research cap),
  dropped unverifiable v5 patches, and added the follow-up scripts, dashboard v2 status and open items. The previous file is saved outside the repo as `Downloads\AGENTS_before_24Nov_backup.md`.
