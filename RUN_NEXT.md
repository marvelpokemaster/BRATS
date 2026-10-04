# Run practical v8 in five sessions

Extract this whole folder. Use the `.py` marimo entry point OR its matching
`.ipynb`, with all helpers alongside it. Keep a copy of the original ZIP.
Start fresh Part 1, fill `HF_TOKEN_PLACEHOLDER` locally, and use the configured
graph-cache/model repository IDs consistently in every part. Do not mix v7 files.

| Session | Notebook | Scheduled work | Declared training allowance |
|---|---|---|---|
| 1 | part1_corrected | Audit/reuse 15k graphs; main HGT with learned 0–2 hops; handoff | One phase, 135 min |
| 2 | part2_refinement | Main hybrid CNN and matched MRI-only CNN | Two phases, 270 min total |
| 3 | part3_experiments | GraphSAGE, fixed shared two-hop HGT, independent SegResNet | Three phases, 405 min total |
| 4 | part4_slic_study | Separate 64-train/16-validation SLIC pilot at four resolutions | Four phases, at most 180 min total, plus fresh graph builds |
| 5 | part5_reports | Main held-out evaluation, metrics, plots and graph explanations | No new training |

The remaining time is for downloads, graph preparation, validation/report work
outside training phases and uploads. Training phases include their own validation,
calibration and prior preparation. These allocations are not predicted runtimes.
Every session has a 9.5-hour work deadline, with the final 2.5 hours reserved for
transfer. A slow data build or transfer is still possible; the code cannot enforce
the platform's hard limit inside a single blocking call.

1. Part 1: leave `N_SEGMENTS=15000`, `K_MAX=2`, `BATCH_SIZE=8`, physical-batch cap
   8, structural refinement OFF and reconstruction OFF. Calibration can select
   less than eight. Wait for the verified final Part 1 ZIP handoff before Part 2.
2. Part 2: keep the same settings/token. It selects the hybrid model and trains
   the CNN-only control with the same voxel loss/architecture and phase allowance.
   Continue only after `PART 2 COMPLETE` and successful ZIP synchronization.
3. Part 3: ensure `monai==1.5.1` is installed. Keep `RUN_ABLATIONS=True` and the
   frozen single-seed plan. Continue after `THIS PART COMPLETE` and verified ZIPs.
4. Part 4: this session needs raw MRI to build its small, frozen pilot. It retrains
   all four graph resolutions, including 15k. Read the printed `SLIC pilot`
   receipt: `complete=True` means all pilot validation results exist. An incomplete
   receipt is preserved and disclosed in Part 5; it cannot support a SLIC claim.
5. Part 5: ensure `monai==1.5.1` is also installed in this fresh session. It evaluates the six completed main rows on the saved full test split,
   then adds Shapley/hop/gate diagnostics and the separate pilot results. It will
   not silently test missing/untrained main baselines. Read the final status and
   follow the printed `continuation_v8/.../final_reports.zip` location.

Read `stopping_reason`, completed epochs and curves for every model. `compute_budget`
means training ended because of the declared allowance, not because it converged.
Low-quality or still-improving curves must be reported as limitations. This plan
does not promise publishable scores from 135 minutes of training.

Do not tune on test results or spend extra sessions without declaring a revised
budget/protocol. If a required stage cannot produce a completed checkpoint in its
allowance, preserve the ZIP and inspect the recorded timings before deciding what
to change. Resume support is for recovery; repeated sessions are not the default
completion strategy for this five-session plan.
