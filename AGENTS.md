# Agent Instructions & Post-v5 Changelog

**ATTENTION FUTURE AI ASSISTANTS (Codex, Claude, Gemini, etc.):**
Read this document before modifying the codebase. This repository contains the `v5` state-of-the-art revision of the BraTS QoS-HRGN pipeline, but several critical modifications have been made *on top of v5* to ensure it runs correctly on the user's specific infrastructure and meets strict journal submission requirements. 

**Do NOT revert these changes without explicit user permission.**

## 1. Critical Environment Rules
* **NO LOCAL EXECUTION**: The user's laptop is NOT a training workstation. Do not execute computationally intensive tasks (training, heavy preprocessing) locally. Code runs on "Molab" (an RTX PRO 6000 Blackwell Server).
* **PROXY REQUIRED**: All network commands (git push, pip install, etc.) MUST use `proxychains` (e.g., `proxychains git push`).
* **HUGGING FACE TOKENS**: HF tokens must be `sed`-scrubbed before committing code.

## 2. Post-v5 Bug Fixes (Do Not Revert)
* **Kaggle Duplicate Bug**: The Kaggle mirror's `BraTS2021_00495.tar` and `BraTS2021_00621.tar` contain root-level files md5-identical to copies in the main tar. Extracting all archives into shared `_extracted/` leaves those patch copies loose; the old filter kept the loose copies and dropped the foldered copies, so folder-based `get_case_id` assigned both to `_extracted` and produced 1,250 graphs. All five parts now use `select_canonical_nifti` to keep foldered copies and filename-based `case_id_from_path` from `brats_protocol.py`; orphan cache cleanup uses `DATASET_AUDIT["cases"]`. Regression coverage is in `tests/test_case_ids.py`, and notebook parity is checked with `tools/check_notebook_parity.py`. *Note: In `.ipynb` files, injected lines MUST have 0 spaces of indentation because Marimo cells have no outer `def` wrapper.*
* **HF Naming Compatibility**: The `v5` code expected `_review_v3` suffixes for HF files. We reverted these constants (`STAGE1_REMOTE_NAME = "stage1_graph_model.pt"`) to match the files actually hosted on the user's Hugging Face repo.
* **Stage 1 Checkpoint Manifest**: Pre-v5 checkpoints lack a `checkpoint_revision` key in their manifest. We added a fallback: `_rev = _manifest.get("checkpoint_revision") or "main"`.
* **Flexible State-Dict Loading**: Old Stage 1 checkpoints have a different `config_hash` (because v5 added new flags to the identity dictionary) and lack the `base_model.` prefix in their `state_dict` keys. We patched the Stage 1 loading cell in `part2_corrected.py` to print a soft warning instead of hard-crashing, and flexibly map the keys so legacy checkpoints load successfully into the new `StructuralQoSHRGN` wrapper.

## 3. Paper Submission Configuration
To satisfy rigorous journal requirements (baselines, ablations, XAI, SSL), the following flags are permanently enabled in `part1_corrected` and `part2_corrected`:
* `MAX_CASES = None` (Ensures training runs on the full 1,251 cohort).
* `USE_MASKED_RECONSTRUCTION = True` (Enables the SSL objective in Stage 1).
* `RUN_REGION_XAI = True` (Computes Exact Modality Shapley Values).
* `RUN_BASELINES = True` (Trains the 3D U-Net baseline on the exact same split).
* `RUN_ABLATIONS = True` (Executes `brats_experiments.py` for the ablation grid).

## 4. Hardware Optimizations (RTX 6000 Blackwell 95GB)
To prevent the GPU from idling and to speed up the massive baseline/ablation workload, batch limits have been cranked up:
* `BATCH_SIZE = 64`
* `VOXEL_EFFECTIVE_BATCH_SIZE = 16`
* `UNET_BATCH_SIZE = 4`
* `VOXEL_PREFETCH_WORKERS = 12`
* `RESEARCH_MAX_CASES_PER_SPLIT = 300` (Restricts the 14 ablation experiments to a 300-patient subset to cut computation time from days to hours).
