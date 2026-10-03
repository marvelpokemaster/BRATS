# BraTS continuation v6 — Parts 2, 3, 4 and 5

Start with [RUN_NEXT.md](RUN_NEXT.md). This package continues the GitHub
`part1_corrected.ipynb` that was already running. It does not contain a replacement
Part 1. The fetched source commit is recorded in `source_provenance.json`.

## What changed

The GitHub Part 2 combined refinement, a large experiment grid, independent
baselines, diagnostics and Shapley work in one reactive notebook. Enabling its
switches did not solve the runtime limit: its research runner rejected the now
enabled reconstruction objective, and its 300-case debug cap prevented final test
reporting. Per-artifact checkpoint and receipt uploads also caused many Hub writes.

The continuation copies separate these tasks into four entry points with a common
frozen study. They support the active reconstruction objective, preserve the full
saved patient cohort, and aggregate writes into incremental ZIP commits. The
independent 3D U-Net now uses the resumable experiment machinery instead of its
separate non-resumable driver. The graph/CNN architecture of the main model remains
unchanged. Selected main-model weights are imported exactly for the matching
default-seed full research row; they are not retrained or randomly extended.

## Files

Each numbered notebook is supplied as native marimo `.py` and matching `.ipynb`.
Keep these eight helpers beside whichever notebook you run:

`brats_protocol.py`, `brats_gpu.py`, `brats_transfer.py`, `brats_experiments.py`,
`brats_bundles.py`, `brats_workflow.py`, `brats_reporting.py`, `baseline_3d_unet.py`.

Use Molab's CUDA-compatible PyTorch environment. Required Python packages are
marimo, torch, torch-geometric, numpy, scipy, nibabel, scikit-image, matplotlib,
kagglehub, joblib and huggingface-hub. Do not replace Molab's working CUDA Torch
build with a CPU build. CUDA availability and a small convolution are checked
before the long work. No custom kernel or compilation dependency was added.

## Part 1 compatibility

The importer reads the completed checkpoint, its saved architecture/configuration,
patient split, dataset fingerprint and graph archive receipt. It supports both
the current `stage1_graph_model.*` and earlier `stage1_graph_model_review_v3.*`
handoff names. The current name takes priority; an incomplete current manifest
does not silently fall back to an older run. A user-specified manifest override
is available if there are intentionally multiple runs.

Checkpoint and graph ZIP downloads use saved immutable revisions and SHA256
checks. When an older manifest omits the checkpoint revision but has the required
checksum, the loader pins the current repository revision and checks those bytes.
It never loads unmatched model weights with `strict=False`, discards structural
weights, or leaves an untrained refinement wrapper active. Incompatible legacy
artifacts stop with an explanation instead of producing misleading predictions.

Parts 2, 3 and 5 use the MRI tensors/targets already in Part 1's verified graph
metadata. They inherit its audited cohort; they do not claim to repeat the raw-data
audit. Part 4 downloads and audits the raw MRI to build other SLIC partitions,
retains GitHub's patched-case path filtering, and requires the same source fingerprint.
Every part checks the persisted patient split and selected graph weights.

## Scheduling and computation

The main epoch limits, patience, graph batch setting and training objectives are
retained. Research uses the saved Part 1 graph configuration and the continuation's
declared voxel configuration. GPU microbatch calibration, mixed precision,
patient-weighted accumulation and OOM retries for the voxel head are retained.
No claim of measured RTX PRO 6000 utilization is made without a real run.

The default grid is reduced to 17 rows by making three extra mechanism probes
optional. No advisor-required comparison is removed. See [RESEARCH_PLAN.md](RESEARCH_PLAN.md).
The full model at the original seed reuses the completed Part 1/2 weights. Graph-only
and no-boundary rows share the appropriate identical full graph checkpoint. Completed
jobs skip their SLIC-cache reconstruction and training. Completed evaluation rows skip
model loading/inference. Fixed random samples and per-patient records let explanations
and evaluation resume after a pause.

Seventeen rows across three seeds still require many sessions. Dividing the notebook
does not make the total training cost disappear. The optional extended controls must
be selected before the study starts. Do not change the frozen protocol after looking
at test results. Part 5 refuses test evaluation until its whole frozen plan completes.
Part 1 already computes its own test diagnostics, so this guard cannot undo earlier
test exposure; disclose that research history in the manuscript.

## ZIP transfer design

Local completed-epoch checkpoints are frequent. Remote synchronization is limited
to approximately two-hour intervals and clean pause/completion/error boundaries.
All updated artifacts enter ZIPs; one atomic Hub commit contains the changed-data
ZIP(s) and `catalog.zip`. Files are split into roughly 2 GiB archives where possible;
an individual larger artifact remains a larger archive. The catalog records each
file's checksum and which ZIP contains its current version. New sessions pin one
catalog commit and restore only needed bundles. Unchanged files are not reuploaded.
The original Part 1 graph repository is read-only to these continuation notebooks.

The final human-readable report ZIP is also published once at the printed model
repository path. Small receipt files remain inside the catalog bundles. Authentication
errors propagate; rate-limit/service-busy responses get bounded retries. Failed
uploads preserve local files. A stale concurrent writer is rejected. Do not delete
older bundle ZIPs manually: the latest catalog may still reference them.

Full graph/SLIC metadata are large. Disk must accommodate raw data where needed,
downloaded archives, extracted metadata and temporary ZIP copies. Free-space checks
stop rather than report a failed save as successful. Available GPU VRAM alone does
not establish enough host RAM or disk. A hard kill loses unflushed work and a
9.5-hour work budget cannot guarantee upload completion before 12 hours.

## Evidence and limitations

See `marimo_validation.json`, `research_validation.json` and
`continuation_validation.json` for the checks actually performed. Tests use tiny
synthetic CPU graphs/volumes and a mocked Hub, including fresh-session restoration,
corruption, failed uploads, rate limiting, SSL gradients, exact selected-weight reuse,
U-Net training, reporting and interruption/resume. These are not BraTS results.

The two supplied reference papers continue to inform the component comparisons:
Saueressig et al. (2109.05580v2) already introduced a graph-plus-CNN pipeline;
Abd-Elhafiez (2025.I2.050) motivates testing boundary supervision. The GraphSAGE
control is not an exact reimplementation of the first paper, and neither paper's
unmatched published metrics establish superiority on this internal split.

Actual CUDA behavior, full-cohort runtime/storage, live transfers and learned
performance remain to be measured. This code cannot guarantee a bug-free Molab run
or journal acceptance. A strong independently configured baseline/external cohort
may still be needed for stronger claims than this component study supports.

Implementation references: [Hugging Face uploads](https://huggingface.co/docs/huggingface_hub/guides/upload)
and [marimo variable definitions](https://docs.marimo.io/guides/understanding_errors/multiple_definitions/).
