# Continue after your running Part 1

**Leave the running `part1_corrected.ipynb` unchanged.** This package contains only
Parts 2–5 and their helpers. Do not run the old all-in-one Part 2 alongside these.

1. Wait for Part 1 to print **“Part1 handoff ready: verified graph ZIP + checkpoint + persisted split + manifest”**.
   If its time budget expires before that message, resume that same Part 1.
2. Extract this package. In a **fresh Molab session**, upload the chosen notebook
   and **all eight helper `.py` files** to the same directory. Use the native
   marimo `.py` notebook, or its matching `.ipynb` copy if that is your usual import.
   Do not upload both formats as separate experiments.
3. Select the RTX PRO 6000 runtime. Fill `HF_TOKEN_PLACEHOLDER` in the first
   configuration cell with your token. It needs read access to the graph dataset
   and read/write access to the model repository. Keep the existing repository IDs:
   `marvelpokemaster/brats-qos-hrgn-graphs` and `marvelpokemaster/brats-qos-hrgn-model`.
4. Run these in order. Each row may require **several sessions**, not one session.

| Notebook | Work | When to advance |
|---|---|---|
| `part2_refinement.py` | Restore Part 1; train/resume the refinement CNN; save curves | `PART 2 COMPLETE` after verified ZIP handoff |
| `part3_experiments.py` | Component comparisons, CNN-only, GraphSAGE and independent 3D U-Net | `THIS PART COMPLETE` |
| `part4_slic_study.py` | Train 5k, 10k and 20k SLIC models; 15k is the full row from Part 3 | `THIS PART COMPLETE` |
| `part5_reports.py` | Held-out comparisons, figures, node/hop/gate analysis, reconstruction diagnostics and region Shapley | Status contains `complete: True` and final report ZIP location |

5. When a part says **PAUSED**, or stops with a session-budget message, wait for
   **VERIFIED ZIP HANDOFF**. Start a fresh session with the **same part and settings**.
   It restores completed epochs, cases and results from Hugging Face automatically.
   Do not advance simply because a notebook reached the bottom of its page.
6. After Part 5, download `final_reports.zip` from the model repository path printed
   by the notebook: `continuation_v6/<Part-1-checkpoint-hash>/final_reports.zip`.

## Important settings

- `RUN_ABLATIONS=True`, `RUN_BASELINES=True` and `RUN_REGION_XAI=True` remain enabled.
  Each task executes only in its designated part.
- The default plan has **17 configurations × 3 seeds**. The matching full-model
  seed reuses Part 1 and Part 2; identical graph components also share checkpoints.
- Keep the frozen plan, seeds and training settings identical in Parts 3, 4 and 5.
  Leave `RESEARCH_ACTIVE_SEEDS=()` to run remaining work automatically.
- Three additional hop-mechanism controls remain available through
  `RUN_EXTENDED_HOP_CONTROLS=True`. If wanted, set it identically in Parts 3–5
  **before starting Part 3**. It expands the plan to 20 configurations.
- Do not reintroduce the old 300-case debug cap for final results.

## Transfers and interruptions

Work stops at a **9.5-hour budget**, leaving time before Molab's 12-hour limit
for ZIP creation and transfer. This is a reserve, not a guarantee for very large
archives or slow connections. Keep enough disk space for source data, extracted
graphs, the Hub download cache and a temporary upload ZIP.

Checkpoints save locally every completed epoch. Remote saves combine changed
files into ZIPs about every two hours, and at a clean pause/exit. Small files are
not uploaded separately. Unchanged graphs and reports are reused. A sudden runtime
kill can lose work since the last successful remote save; an incomplete epoch is
replayed. Do not run two continuation sessions against the same study concurrently.

If an upload fails while the session is still alive, rerun the failed cell in that
same session so its local checkpoints can be uploaded. Do not close the session
until the handoff succeeds. Authentication, checksum, missing-data or configuration
errors are not signs to proceed to the next part.

No change to your running Part 1's upload behavior is made by this package.
