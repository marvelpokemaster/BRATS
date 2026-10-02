# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kagglehub==1.0.2",
#     "nibabel==5.4.2",
#     "torch-geometric==2.8.0.post1",
#     "huggingface-hub==1.32.0",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QoS-HRGN v2 -- PART 2 of 2: voxel refinement, evaluation, XAI

    Continuation of Part 1 in a **separate Colab session** (fresh 12h runtime).
    Rebuilds the same config/graph pipeline (fast: graphs are pulled from the
    Sec. 2b Hugging Face graph-cache instead of rebuilt), downloads Part 1's
    trained graph model from Hugging Face Hub, then trains the voxel refinement
    head and runs the rest of the original notebook (evaluation, diagnostics,
    Shapley/XAI).

    Graph-cache transfer uses a single ZIP archive. Set `HF_TOKEN` in the environment (or use the stored Hugging Face login); voxel-head training resumes from `stage2_latest.pt` when available.

    **GPU update v4:** keep `brats_protocol.py`, `brats_gpu.py`, `brats_transfer.py` and `baseline_3d_unet.py` alongside this notebook. Fill the blank `HF_TOKEN_PLACEHOLDER` in the configuration cell before molab. Repaired Stage1 v3 checkpoints remain compatible; Stage2 uses fresh GPU v4 checkpoints. Read README.md before a full run.

    **Research revision v5:** read RESEARCH_PROTOCOL.md. Use research_experiments.py for resumable component experiments. Keep brats_experiments.py with the other helpers.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # QoS-HRGN v2: Heterogeneous Adaptive Graph Learning for BraTS 3D Segmentation

    **Pipeline**

    `BraTS MRI → modality-specific 3D SLIC (T1ce / FLAIR partitions) → heterogeneous supervoxel graph → HGT → adaptive multi-hop propagation → pathology-aware prediction → voxel reconstruction`

    **What changed from v1 (400-segment notebook) and why**

    The v1 run showed node-level Dice ≈ 0.78 but voxel-level Dice of only WT 0.45 / TC 0.33 / ET 0.24.
    That gap is a *graph-construction* problem, not a graph-learning problem. v2 fixes the construction:

    | # | Change | Reason |
    |---|---|---|
    | 1 | `N_SEGMENTS` 400 → **15 000** (Saueressig et al. optimum on BraTS) | 400 supervoxels ≈ 15 mm bricks; ET is a thin rim of a few thousand voxels and cannot be drawn at that resolution. |
    | 2 | Hard priority labels → **fractional (soft) labels** per node | Priority labelling made ET 3.7× more frequent than NCR at node level (opposite of reality) and painted whole bricks ET at reconstruction → low ET precision. |
    | 3 | 6 single-modality features → **32 features from all four modalities** (mean, std, 5 quantiles each + position + size) | A T1ce node could not see FLAIR (edema) and vice-versa; T1/T2 were loaded and discarded. |
    | 4 | Centroid-matched cross edges → **voxel-overlap cross edges** with overlap fraction as edge attribute | Overlap is the natural correspondence between two partitions of the same volume; gives 3–8 typed edges per node instead of 1. |
    | 5 | **Oracle / ASA cells** | Reconstructing from ground-truth node labels shows the ceiling of a partition; if the ceiling is low the model cannot be blamed. |
    | 6 | Model selection on **per-patient volume-weighted Dice** (= voxel Dice under node-constant predictions), ET small-volume post-processing, hidden 64→128, residual HGT, cosine LR, 80 epochs | v1 selected on batched node Dice which does not track voxel Dice. |

    BraTS 2020 facts used here: 369 training cases (293 HGG, 76 LGG), 240×240×155 at 1 mm³, co-registered and skull-stripped; labels 0 = background, 1 = NCR/NET, 2 = ED, 4 = ET; evaluated on ET, TC = NCR/NET ∪ ET and WT = TC ∪ ED. Many LGG cases have **no ET at all**, which is why ET post-processing matters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Imports and configuration

    Every tunable lives in this cell. `N_SEGMENTS` and `COMPACTNESS` define the graph cache; changing either builds a new cache directory.
    """)
    return


@app.cell
def _():
    # packages added via marimo's package management: torch-geometric nibabel scikit-image matplotlib kagglehub scipy joblib
    import os, glob, re, random, time, math, datetime, json, hashlib
    import numpy as np
    import matplotlib.pyplot as plt
    import nibabel as nib
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from skimage.segmentation import slic
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HGTConv, GATConv, SAGEConv, HeteroConv
    from torch_geometric.loader import DataLoader
    from joblib import Parallel, delayed
    import kagglehub

    print("PyTorch:", torch.__version__)
    try:
        import torch_geometric
        print("PyG:", torch_geometric.__version__)
    except Exception:
        pass

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # ── Graph construction ───────────────────────────────────────
    N_SEGMENTS = 15000
    # N_SEGMENTS is the active resolution; this tuple is the ablation sweep.
    SLIC_ABLATION_SEGMENTS = (5000, 10000, 15000, 20000)
    COMPACTNESS = 0.3
    SLIC_ITERS = 10
    MIN_OVERLAP = 0.05
    NODE_TYPES = ("t1ce", "flair")
    SLIC_MODALITIES = ("t1ce", "flair")
    MODALITIES = ("t1", "t1ce", "t2", "flair")
    NODE_FEATURE_MODALITIES = MODALITIES
    QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
    NODE_FEAT_DIM = len(MODALITIES) * (2 + len(QUANTILES)) + 4
    GRAPH_VERSION = 4

    # ── Model / training ─────────────────────────────────────────
    HIDDEN_DIM = 128
    HEADS = 4
    HGT_LAYERS = 2
    HOPS = 2                    # fixed-hop baseline when USE_ADAPTIVE_HOPS=False
    USE_ADAPTIVE_HOPS = True    # proposed node-wise learned receptive-field depth
    K_MAX = 4                   # candidate depths are k=0,...,K_MAX
    HOP_REG_WEIGHT = 0.002      # small expected-hop cost; discourages always choosing K_MAX
    HOP_TEMPERATURE = 1.0       # softmax temperature; keep 1.0 for the main experiment
    DROPOUT = 0.1
    EPOCHS = 100
    BATCH_SIZE = 8
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    DICE_WEIGHT = 1.0

    # ── Evaluation ───────────────────────────────────────────────
    ET_MIN_VOXELS = 300

    # ── Oracle sweep ─────────────────────────────────────────────
    ORACLE_CASES = 4
    ORACLE_SWEEP = (400, 4000, 15000)

    # ── Voxel refinement head ─────────────────────────────────────
    USE_VOXEL_REFINEMENT = True
    VOXEL_EPOCHS = 100
    VOXEL_LR = 1e-3
    VOXEL_BASE_CHANNELS = 32
    VOXEL_DICE_WEIGHT = 1.0
    VOXEL_FOCAL_WEIGHT = 0.5
    VOXEL_FOCAL_GAMMA = 2.0
    VOXEL_CE_WEIGHT = 1.0
    VOXEL_BOUNDARY_WEIGHT = 0.5
    VOXEL_MARGIN = 10
    VOXEL_MAX_SIZE = 128

    # ── Focal loss for the graph model ────────────────────────────
    ET_FOCAL_WEIGHT = 0.5
    FOCAL_GAMMA = 2.0

    # ── Connected-component post-processing ──────────────────────
    USE_CC_POSTPROCESS = True

    # ── Fair baseline experiments (disabled by default) ─────────
    RUN_BASELINES = False
    BASELINE_EPOCHS = EPOCHS
    BASELINE_VOXEL_EPOCHS = VOXEL_EPOCHS
    BASELINE_MAX_CASES = 0          # 0 = use every case in the persisted split

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    USE_AMP = device.type == "cuda"
    if USE_AMP:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Device:", device)
    if USE_AMP:
        print("GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"\n{'='*60}")
    print(f"  QoS-HRGN Config: N_SEGMENTS={N_SEGMENTS}  FEATURES={NODE_FEAT_DIM}")
    print(f"  Adaptive hops={USE_ADAPTIVE_HOPS}  candidates=0..{K_MAX if USE_ADAPTIVE_HOPS else HOPS}")
    print(f"  Voxel CE weight={VOXEL_CE_WEIGHT}  boundary weight={VOXEL_BOUNDARY_WEIGHT}")
    print(f"{'='*60}\n")
    INFERENCE_OVERLAP = 0.25
    EXPECTED_CASES = 1249
    CASE_EXCLUSIONS = {"BraTS2021_00495": "Missing/duplicate modalities", "BraTS2021_00621": "Missing/duplicate modalities"}
    OFFICIAL_CASE_IDS_PATH = None  # Optional JSON list of official training IDs.
    HF_TOKEN_PLACEHOLDER = ""  # Fill this before uploading/running in molab.
    HF_REQUIRED = True
    REQUIRE_CUDA = True
    GPU_PRECISION = "auto"  # BF16 on supported CUDA GPUs; FP32 on CPU helpers.
    GPU_AUTOTUNE = True
    GPU_MEMORY_FRACTION = 0.70
    VOXEL_EFFECTIVE_BATCH_SIZE = 4
    VOXEL_MAX_MICROBATCH = 4
    INFERENCE_MAX_BATCH = 8
    VOXEL_PREFETCH_WORKERS = 2
    from huggingface_hub import get_token as _get_hf_token_early
    if HF_REQUIRED and not ((os.environ.get("HF_TOKEN") or "").strip() or (_get_hf_token_early() or "").strip() or HF_TOKEN_PLACEHOLDER.strip()):
        raise RuntimeError("Fill HF_TOKEN_PLACEHOLDER in this configuration cell before running molab")
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise RuntimeError("Select the molab GPU runtime and a CUDA-compatible PyTorch build before training")
    if torch.cuda.is_available():
        try:
            _gpu_check = torch.ones((1, 4, 8, 8, 8), device=device)
            _gpu_kernel = torch.ones((4, 4, 3, 3, 3), device=device)
            _gpu_result = F.conv3d(_gpu_check, _gpu_kernel)
            torch.cuda.synchronize(device)
            del _gpu_check, _gpu_kernel, _gpu_result
        except RuntimeError as _gpu_error:
            raise RuntimeError("The molab PyTorch/CUDA build cannot execute GPU convolutions; use its CUDA-compatible runtime") from _gpu_error

    RUN_REGION_XAI = False  # Additional analysis session; saves direct graph-region attribution.
    return (
        BASELINE_EPOCHS,
        BASELINE_MAX_CASES,
        BASELINE_VOXEL_EPOCHS,
        BATCH_SIZE,
        CASE_EXCLUSIONS,
        COMPACTNESS,
        DICE_WEIGHT,
        DROPOUT,
        DataLoader,
        EPOCHS,
        ET_FOCAL_WEIGHT,
        ET_MIN_VOXELS,
        EXPECTED_CASES,
        F,
        FOCAL_GAMMA,
        GATConv,
        GPU_AUTOTUNE,
        GPU_MEMORY_FRACTION,
        GPU_PRECISION,
        GRAPH_VERSION,
        HEADS,
        HF_REQUIRED,
        HF_TOKEN_PLACEHOLDER,
        HGTConv,
        HGT_LAYERS,
        HIDDEN_DIM,
        HOPS,
        HOP_REG_WEIGHT,
        HOP_TEMPERATURE,
        HeteroConv,
        HeteroData,
        INFERENCE_MAX_BATCH,
        INFERENCE_OVERLAP,
        K_MAX,
        LR,
        MIN_OVERLAP,
        MODALITIES,
        NODE_FEATURE_MODALITIES,
        NODE_FEAT_DIM,
        NODE_TYPES,
        N_SEGMENTS,
        OFFICIAL_CASE_IDS_PATH,
        ORACLE_CASES,
        ORACLE_SWEEP,
        Parallel,
        QUANTILES,
        RUN_BASELINES,
        RUN_REGION_XAI,
        SAGEConv,
        SEED,
        SLIC_ITERS,
        SLIC_MODALITIES,
        USE_ADAPTIVE_HOPS,
        USE_AMP,
        USE_CC_POSTPROCESS,
        USE_VOXEL_REFINEMENT,
        VOXEL_BASE_CHANNELS,
        VOXEL_BOUNDARY_WEIGHT,
        VOXEL_CE_WEIGHT,
        VOXEL_DICE_WEIGHT,
        VOXEL_EFFECTIVE_BATCH_SIZE,
        VOXEL_EPOCHS,
        VOXEL_FOCAL_GAMMA,
        VOXEL_FOCAL_WEIGHT,
        VOXEL_LR,
        VOXEL_MARGIN,
        VOXEL_MAX_MICROBATCH,
        VOXEL_MAX_SIZE,
        VOXEL_PREFETCH_WORKERS,
        WEIGHT_DECAY,
        delayed,
        device,
        glob,
        hashlib,
        json,
        kagglehub,
        math,
        nib,
        nn,
        np,
        os,
        plt,
        random,
        re,
        slic,
        time,
        torch,
    )


@app.cell
def _():
    import time as _time_mod
    PART2_START_TIME = _time_mod.time()
    SESSION_HARD_CAP_HOURS = 12.0
    BUILD_BUDGET_HOURS = 3.0
    TRAIN_BUDGET_HOURS = 10.0
    CKPT_PUSH_EVERY_EPOCHS = 5
    RESUME_STAGE2 = True
    EARLY_STOP_PATIENCE = 15

    print(f"[Part 2] Started. Build budget {BUILD_BUDGET_HOURS:.1f}h, train budget {TRAIN_BUDGET_HOURS:.1f}h (hard cap {SESSION_HARD_CAP_HOURS:.0f}h).")
    return (
        BUILD_BUDGET_HOURS,
        CKPT_PUSH_EVERY_EPOCHS,
        EARLY_STOP_PATIENCE,
        PART2_START_TIME,
        RESUME_STAGE2,
        SESSION_HARD_CAP_HOURS,
        TRAIN_BUDGET_HOURS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1b. Option B configuration
    """)
    return


@app.cell
def _(EPOCHS):
    # ============================================================
    # Option B: Structure-Aware Graph Refinement — configuration
    # ------------------------------------------------------------
    # Research ablation on top of the existing QoS-HRGN backbone. Set
    # USE_STRUCTURAL_REFINEMENT = False to run the exact original baseline
    # unchanged (nothing below in this cell affects that path). When True, an
    # additional lightweight module analyses the STRUCTURE of the model's own
    # predicted tumour graph -- not just node/edge features -- and uses it to
    # refine the final classification (see the model cell below).
    #
    # We do NOT assume any structural relationship (e.g. "ET encloses NCR") is
    # a universal biological rule. These are measurable graph properties the
    # refinement module is free to learn to use, or to ignore.
    USE_STRUCTURAL_REFINEMENT = True
    STRUCTURAL_USE_NEIGHBOR_STATS = True
    # Individual descriptor toggles, so ablations can show WHICH structural
    # signal (if any) actually helps.
    STRUCTURAL_USE_DISTANCE = True  # tumour-neighbour / ET-neighbour fraction
    STRUCTURAL_USE_COMPONENT_SIZE = True  # hop-distance to nearest predicted ET / nearest predicted BG
    STRUCTURAL_USE_ENCLOSURE = True  # size of the predicted-tumour connected component
    STRUCTURAL_USE_CROSS_MODAL = True  # ET-neighbour fraction minus BG-neighbour fraction
    STRUCTURAL_HIDDEN = 32  # T1ce<->FLAIR correspondence fan-out
    STRUCTURAL_DROPOUT = 0.1
    STRUCTURAL_AUX_WEIGHT = 0.3  # width of the structural MLP -- kept small on purpose
    STRUCTURAL_TEACHER_PROB_START = 1.0
    STRUCTURAL_TEACHER_PROB_END = 0.0  # loss = hierarchy_loss(final) + STRUCTURAL_AUX_WEIGHT * hierarchy_loss(initial)
      # keeps the base model's OWN initial predictions well-calibrated,
    def structural_teacher_prob(epoch, total_epochs=EPOCHS):  # since the structural descriptors are derived from them, and avoids
        """Linear decay from STRUCTURAL_TEACHER_PROB_START to _END over training."""  # turning this into "a second, independent classifier."
        if total_epochs <= 1:
    # Scheduled sampling for structural context (the exposure-bias fix). Early in
    # training, structural descriptors are computed mostly from ground-truth
    # labels (a clean signal to bootstrap the refinement module). As training
    # progresses we shift towards the model's OWN predicted labels, so the module
    # learns to correct genuinely noisy structure rather than just to trust
    # already-correct GT structure. GT is NEVER used for structural context
    # outside of training -- validation/test always call with teacher_prob=0.0.
            return STRUCTURAL_TEACHER_PROB_END
        frac = min(max((epoch - 1) / (total_epochs - 1), 0.0), 1.0)
        return STRUCTURAL_TEACHER_PROB_START + (STRUCTURAL_TEACHER_PROB_END - STRUCTURAL_TEACHER_PROB_START) * frac
    STRUCTURAL_FLAGS = dict(neighbor_stats=STRUCTURAL_USE_NEIGHBOR_STATS, distance=STRUCTURAL_USE_DISTANCE, component_size=STRUCTURAL_USE_COMPONENT_SIZE, enclosure=STRUCTURAL_USE_ENCLOSURE, cross_modal=STRUCTURAL_USE_CROSS_MODAL)

    def structural_descriptor_dim(flags):
        dim = 0
        if flags['neighbor_stats']:
            dim = dim + 2
        if flags['distance']:
            dim = dim + 2
        if flags['component_size']:
            dim = dim + 1
        if flags['enclosure']:
            dim = dim + 1
        if flags['cross_modal']:
            dim = dim + 1
        return dim
    print(f'[Option B] USE_STRUCTURAL_REFINEMENT={USE_STRUCTURAL_REFINEMENT}  structural_dim={structural_descriptor_dim(STRUCTURAL_FLAGS)}  flags={STRUCTURAL_FLAGS}')  # tumour-neighbour fraction, ET-neighbour fraction  # hop-distance to nearest ET, hop-distance to nearest BG  # log1p(tumour connected-component size), normalised  # ET-neighbour fraction minus BG-neighbour fraction  # normalised cross-modal correspondence fan-out
    return (
        STRUCTURAL_AUX_WEIGHT,
        STRUCTURAL_DROPOUT,
        STRUCTURAL_FLAGS,
        STRUCTURAL_HIDDEN,
        STRUCTURAL_TEACHER_PROB_END,
        STRUCTURAL_TEACHER_PROB_START,
        USE_STRUCTURAL_REFINEMENT,
        structural_descriptor_dim,
        structural_teacher_prob,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1c. Option C configuration - heterogeneous masked graph reconstruction

    Auxiliary self-supervised objective added on top of the existing segmentation task:

    `L_total = L_segmentation + lambda_rec * L_reconstruction`

    Set `USE_MASKED_RECONSTRUCTION = False` (default) to reproduce the exact baseline run.
    """)
    return


@app.cell
def _(MODALITIES, NODE_FEAT_DIM, QUANTILES):
    # ============================================================
    # Option C: Heterogeneous Masked Graph Reconstruction (auxiliary task)
    # ------------------------------------------------------------
    #   L_total = L_segmentation + lambda_rec * L_reconstruction
    #
    # Segmentation stays the PRIMARY objective. With REC_SEPARATE_CLEAN_PASS=True
    # the segmentation loss is computed on the CLEAN graph exactly as in the
    # baseline, and the reconstruction loss on a separately corrupted graph, so the
    # primary objective is mathematically unchanged and the ablation is clean.
    #
    # Set USE_MASKED_RECONSTRUCTION = False to run the exact original baseline
    # (nothing in this cell affects that path).
    USE_MASKED_RECONSTRUCTION = False

    LAMBDA_REC = 0.3            # weight of the auxiliary reconstruction loss
    REC_WARMUP_EPOCHS = 5       # linear ramp 0 -> LAMBDA_REC, so the aux task cannot
                                # dominate before the backbone produces useful embeddings

    # ---- what gets masked -------------------------------------------------------
    REC_MASK_RATE = 0.30        # fraction of nodes whose APPEARANCE features are masked
    REC_EDGE_MASK_RATE = 0.15   # fraction of t1ce<->flair correspondence edges held out

    # ---- which reconstruction targets are enabled -------------------------------
    REC_USE_FEATURE = True          # (1) masked node appearance reconstruction
    REC_USE_CORRESPONDENCE = True   # (2) T1ce <-> FLAIR correspondence relation prediction
    REC_USE_SPATIAL_EDGE = False    # (3) spatial edge existence -- OFF BY DEFAULT, see note
    #
    # WHY SPATIAL EDGE RECONSTRUCTION IS OFF BY DEFAULT
    # Spatial adjacency is a DETERMINISTIC function of the SLIC tessellation, and the
    # normalised centroid (feature dims 28..30) is already part of every node's input.
    # Face-adjacent supervoxels are exactly those within ~one supervoxel diameter, so
    # with randomly sampled negatives the task is solved by centroid distance alone
    # (AUC ~= 1.0 at initialisation): it produces an impressive-looking number, no
    # gradient, and no representation learning. Enabling it uses distance-matched hard
    # negatives (2-hop nodes), which makes it non-trivial but largely redundant with the
    # correspondence task. Kept behind a flag so the ablation can REPORT this as a
    # negative result rather than omitting it silently.

    REC_NEG_PER_POS = 1         # hard negatives sampled per held-out positive edge
    REC_DECODER_HIDDEN = 64     # reconstruction heads are deliberately lightweight
    REC_FEAT_LOSS = "smooth_l1" # "smooth_l1" | "mse" | "sce" (GraphMAE scaled cosine error)
    REC_SCE_GAMMA = 2.0         # only used when REC_FEAT_LOSS == "sce"
    REC_SEPARATE_CLEAN_PASS = True
    #   True  -> 2 forward passes/step: clean (segmentation) + corrupted (reconstruction).
    #            Primary objective identical to baseline. ~2x step time. RECOMMENDED.
    #   False -> 1 forward pass: segmentation AND reconstruction from the corrupted graph.
    #            Cheaper, but masked nodes have no appearance, so their segmentation
    #            targets are partly unanswerable and the primary objective changes.

    # ---- appearance / geometry split of the 32-dim node feature vector ----------
    # node_table() builds x as: [per-modality (mean, std, 5 quantiles)] + [centroid xyz] + [log size]
    STATS_PER_MODALITY = 2 + len(QUANTILES)                       # 7
    APPEARANCE_DIM = len(MODALITIES) * STATS_PER_MODALITY          # 28  (dims 0..27)
    GEOMETRY_DIM = NODE_FEAT_DIM - APPEARANCE_DIM                  # 4   (dims 28..31)
    MODALITY_SLICES = {m: slice(i * STATS_PER_MODALITY, (i + 1) * STATS_PER_MODALITY)
                       for i, m in enumerate(MODALITIES)}
    assert APPEARANCE_DIM == 28 and GEOMETRY_DIM == 4, "feature layout changed; update Option C config"
    #
    # ONLY dims 0..27 (appearance) are ever masked. The geometry dims are left intact:
    #  * a supervoxel centroid is recoverable almost exactly from its face-adjacent
    #    neighbours, so reconstructing it is a degenerate target, and
    #  * the segmentation head legitimately needs position.
    # Masking the FULL 28-dim appearance block (never a single modality) is essential:
    # if only the T1ce stats were masked while T1/T2/FLAIR remained, the model would
    # solve the task by intra-node cross-modality correlation and never consult the
    # graph at all -- the opposite of what this auxiliary task is for.

    def rec_lambda(epoch):
        """lambda_rec with linear warm-up (epoch is 1-based)."""
        if not USE_MASKED_RECONSTRUCTION:
            return 0.0
        if REC_WARMUP_EPOCHS <= 0:
            return LAMBDA_REC
        return LAMBDA_REC * min(1.0, epoch / float(REC_WARMUP_EPOCHS))

    REC_FLAGS = dict(feature=REC_USE_FEATURE, correspondence=REC_USE_CORRESPONDENCE,
                     spatial_edge=REC_USE_SPATIAL_EDGE)

    print(f"[Option C] USE_MASKED_RECONSTRUCTION={USE_MASKED_RECONSTRUCTION}  lambda_rec={LAMBDA_REC}  "
          f"mask_rate={REC_MASK_RATE}  edge_mask_rate={REC_EDGE_MASK_RATE}")
    print(f"[Option C] targets={[k for k, v in REC_FLAGS.items() if v]}  "
          f"appearance_dim={APPEARANCE_DIM} (masked)  geometry_dim={GEOMETRY_DIM} (kept)  "
          f"separate_clean_pass={REC_SEPARATE_CLEAN_PASS}")
    return (
        APPEARANCE_DIM,
        LAMBDA_REC,
        MODALITY_SLICES,
        REC_DECODER_HIDDEN,
        REC_EDGE_MASK_RATE,
        REC_FEAT_LOSS,
        REC_FLAGS,
        REC_MASK_RATE,
        REC_NEG_PER_POS,
        REC_SCE_GAMMA,
        USE_MASKED_RECONSTRUCTION,
    )


@app.cell
def _(
    APPEARANCE_DIM,
    BATCH_SIZE,
    COMPACTNESS,
    DICE_WEIGHT,
    DROPOUT,
    EPOCHS,
    ET_FOCAL_WEIGHT,
    FOCAL_GAMMA,
    GPU_PRECISION,
    GRAPH_VERSION,
    HEADS,
    HGT_LAYERS,
    HIDDEN_DIM,
    HOPS,
    HOP_REG_WEIGHT,
    HOP_TEMPERATURE,
    INFERENCE_OVERLAP,
    K_MAX,
    LAMBDA_REC,
    LR,
    MIN_OVERLAP,
    MODALITIES,
    NODE_FEATURE_MODALITIES,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NUM_CLASSES,
    N_SEGMENTS,
    PROTOCOL_VERSION,
    QUANTILES,
    REC_DECODER_HIDDEN,
    REC_FLAGS,
    SEED,
    SLIC_ITERS,
    SLIC_MODALITIES,
    STRUCTURAL_AUX_WEIGHT,
    STRUCTURAL_DROPOUT,
    STRUCTURAL_FLAGS,
    STRUCTURAL_HIDDEN,
    USE_ADAPTIVE_HOPS,
    USE_AMP,
    USE_MASKED_RECONSTRUCTION,
    USE_STRUCTURAL_REFINEMENT,
    VOXEL_BASE_CHANNELS,
    VOXEL_BOUNDARY_WEIGHT,
    VOXEL_CE_WEIGHT,
    VOXEL_DICE_WEIGHT,
    VOXEL_EFFECTIVE_BATCH_SIZE,
    VOXEL_EPOCHS,
    VOXEL_FOCAL_GAMMA,
    VOXEL_FOCAL_WEIGHT,
    VOXEL_LR,
    VOXEL_MARGIN,
    VOXEL_MAX_SIZE,
    WEIGHT_DECAY,
    hashlib,
    json,
):
    IDENTITY_CONFIG = {
        "protocol": PROTOCOL_VERSION,
        "seed": SEED,
        "graph": {
            "n_segments": N_SEGMENTS, "compactness": COMPACTNESS, "slic_iters": SLIC_ITERS,
            "min_overlap": MIN_OVERLAP, "node_types": list(NODE_TYPES), "slic_modalities": list(SLIC_MODALITIES),
            "modalities": list(MODALITIES), "node_feature_modalities": list(NODE_FEATURE_MODALITIES),
            "quantiles": list(QUANTILES), "node_feat_dim": NODE_FEAT_DIM, "graph_version": GRAPH_VERSION,
        },
        "model": {
            "hidden_dim": HIDDEN_DIM, "heads": HEADS, "hgt_layers": HGT_LAYERS, "hops": HOPS,
            "use_adaptive_hops": USE_ADAPTIVE_HOPS, "k_max": K_MAX,
            "hop_temperature": HOP_TEMPERATURE, "dropout": DROPOUT, "num_classes": NUM_CLASSES,
        },
        "structural": {
            "use_structural_refinement": USE_STRUCTURAL_REFINEMENT,
            "structural_flags": dict(STRUCTURAL_FLAGS), "structural_hidden": STRUCTURAL_HIDDEN,
            "structural_dropout": STRUCTURAL_DROPOUT,
        },
        "rec": {
            "use_masked_reconstruction": USE_MASKED_RECONSTRUCTION,
            **({"appearance_dim": APPEARANCE_DIM, "rec_decoder_hidden": REC_DECODER_HIDDEN,
                "rec_flags": dict(REC_FLAGS)} if USE_MASKED_RECONSTRUCTION else {}),
        },
    }
    RUN_CONFIG = {
        "identity": IDENTITY_CONFIG,
        "train": {
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dice_weight": DICE_WEIGHT, "et_focal_weight": ET_FOCAL_WEIGHT, "focal_gamma": FOCAL_GAMMA,
            "amp": USE_AMP, "hop_reg_weight": HOP_REG_WEIGHT, "structural_aux_weight": STRUCTURAL_AUX_WEIGHT,
            "lambda_rec": LAMBDA_REC,
        },
        "voxel": {
            "gpu_training_protocol": "v4-patient-mean-ce", "effective_batch_size": VOXEL_EFFECTIVE_BATCH_SIZE,
            "precision_policy": GPU_PRECISION,
            "voxel_epochs": VOXEL_EPOCHS, "voxel_lr": VOXEL_LR,
            "voxel_base_channels": VOXEL_BASE_CHANNELS, "voxel_dice_weight": VOXEL_DICE_WEIGHT,
            "voxel_focal_weight": VOXEL_FOCAL_WEIGHT, "voxel_focal_gamma": VOXEL_FOCAL_GAMMA,
            "ce_weight": VOXEL_CE_WEIGHT,
            "voxel_boundary_weight": VOXEL_BOUNDARY_WEIGHT, "voxel_margin": VOXEL_MARGIN,
            "voxel_max_size": VOXEL_MAX_SIZE, "inference_overlap": INFERENCE_OVERLAP,
            "training_patch": "50pct-tumour-50pct-uniform", "selection": "full-volume-pp-Dice",
        },
    }


    def config_hash(cfg):
        return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


    def config_diff(a, b):
        _out = []
        def _walk(x, y, prefix=""):
            for _key in sorted(set(x) | set(y)):
                _path = f"{prefix}{_key}"
                if _key not in x or _key not in y:
                    _out.append(f"{_path}: {x.get(_key, '<absent>')} != {y.get(_key, '<absent>')}")
                elif isinstance(x[_key], dict) and isinstance(y[_key], dict):
                    _walk(x[_key], y[_key], _path + ".")
                elif x[_key] != y[_key]:
                    _out.append(f"{_path}: {x[_key]} != {y[_key]}")
        _walk(a, b)
        return _out


    CONFIG_HASH = config_hash(IDENTITY_CONFIG)
    print(f"[Run identity] CONFIG_HASH={CONFIG_HASH}")
    return CONFIG_HASH, IDENTITY_CONFIG, RUN_CONFIG, config_hash


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Local graph cache storage

    Graphs (`*.graph.pt`, small, kept in RAM) and reconstruction metadata (`*.meta.pt`, large, loaded lazily) are stored separately.
    """)
    return


@app.cell
def _(
    COMPACTNESS,
    GRAPH_VERSION,
    IDENTITY_CONFIG,
    N_SEGMENTS,
    config_hash,
    os,
):
    PERSISTENT_BASE = os.path.join(os.getcwd(), "brats_qos_hrgn_data_review_v3")
    GRAPH_CACHE_BASE_PATH = os.path.join(PERSISTENT_BASE, "graphs", f"slic_{N_SEGMENTS}_c{COMPACTNESS}_v{GRAPH_VERSION}_{config_hash(IDENTITY_CONFIG['graph'])}")
    os.makedirs(GRAPH_CACHE_BASE_PATH, exist_ok=True)
    print(f"Graph cache directory: {GRAPH_CACHE_BASE_PATH}")
    return GRAPH_CACHE_BASE_PATH, PERSISTENT_BASE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Download / discover BraTS 2021 Task 1 data

    Switched from the BraTS 2020 Kaggle mirror (369 cases) to
    [`dschettler8845/brats-2021-task1`](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)
    (~1,251 training cases, ~12.3 GB compressed). File naming
    (`BraTS2021_XXXXX_{t1,t1ce,t2,flair,seg}.nii.gz`, all under a
    `BraTS2021_Training_Data/` folder) follows the same underscore-delimited
    convention the modality-detection regex below already handles, so no
    parsing logic changed -- only the dataset slug.

    The much larger case count means: (a) SLIC graph construction and the
    graph cache scale up roughly 3.4x versus BraTS 2020, and (b) raw dataset +
    graph cache can meaningfully strain local disk on constrained environments
    (Colab, etc.) -- see §2b for an optional Hugging Face Hub remote cache
    that addresses this. Set `MAX_CASES` in §1 while debugging to avoid
    downloading/processing the full set.
    """)
    return


@app.cell
def _(
    CASE_EXCLUSIONS,
    EXPECTED_CASES,
    GRAPH_CACHE_BASE_PATH,
    OFFICIAL_CASE_IDS_PATH,
    PERSISTENT_BASE,
    audit_dataset,
    glob,
    json,
    kagglehub,
    os,
    re,
):
    import tarfile

    DATASET_SLUG = "dschettler8845/brats-2021-task1"
    dataset_root = kagglehub.dataset_download(DATASET_SLUG)
    print("Dataset:", dataset_root)

    def _extract_tars(root):
        """The Kaggle mirror ships as .tar archives (BraTS2021_Training_Data.tar
        plus a couple of individually-added case tars), not pre-extracted NIfTI
        files -- kagglehub's own "Extracting files..." step only unpacks the
        outer download, not these. Extract once, cache with marker files so
        reruns don't re-extract 12+ GB every time."""
        tar_paths = glob.glob(os.path.join(root, "*.tar"))
        if not tar_paths:
            return root
        extract_dir = os.path.join(root, "_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        for tp in tar_paths:
            marker = extract_dir + "/" + os.path.basename(tp) + ".done"
            if os.path.exists(marker):
                continue
            print(f"Extracting {os.path.basename(tp)} ({os.path.getsize(tp) / 1e9:.1f} GB) ...")
            with tarfile.open(tp) as tf:
                tf.extractall(extract_dir, filter="data")  # "data" filter: safe extraction (PEP 706)
            open(marker, "w").close()
        return extract_dir

    dataset_root = _extract_tars(dataset_root)
    print("Extracted to:", dataset_root)

    def modality_kind(path):
        name = os.path.basename(path).lower()
        if re.search(r"(^|[_-])t1ce([_.-]|$)", name):
            return "t1ce"
        if re.search(r"(^|[_-])flair([_.-]|$)", name):
            return "flair"
        if re.search(r"(^|[_-])t2([_.-]|$)", name):
            return "t2"
        if re.search(r"(^|[_-])t1([_.-]|$)", name):
            return "t1"
        if re.search(r"(^|[_-])(seg|segm|mask)([_.-]|$)", name):
            return "seg"
        return None

    all_nii = glob.glob(os.path.join(dataset_root, "**", "*.nii"), recursive=True)
    all_nii = all_nii + glob.glob(os.path.join(dataset_root, "**", "*.nii.gz"), recursive=True)
    case_map = {}
    for _p in all_nii:
        _k = modality_kind(_p)
        if _k:
            case_map.setdefault(os.path.dirname(_p), {})[_k] = _p

    _official_ids = None
    if OFFICIAL_CASE_IDS_PATH:
        with open(OFFICIAL_CASE_IDS_PATH) as _fh:
            _official_ids = json.load(_fh)
    valid_cases_all, DATASET_AUDIT = audit_dataset(all_nii, modality_kind,
        os.path.join(PERSISTENT_BASE, "audit"), expected_count=EXPECTED_CASES,
        exclusions=CASE_EXCLUSIONS, official_ids=_official_ids)
    DATASET_FINGERPRINT = DATASET_AUDIT["source_fingerprint"]
    if DATASET_AUDIT["errors"]:
        raise RuntimeError("Dataset audit failed; inspect audit/dataset_audit.json:\n" +
                           "\n".join(DATASET_AUDIT["errors"]))
    MAX_CASES = None  # Set only for smoke tests; subset scores are not thesis results.
    valid_cases = valid_cases_all if MAX_CASES is None else valid_cases_all[:MAX_CASES]
    print("Audited patients:", len(valid_cases_all), "| selected:", len(valid_cases))
    GRAPH_CACHE_PATH = os.path.join(GRAPH_CACHE_BASE_PATH, DATASET_FINGERPRINT[:16])
    os.makedirs(GRAPH_CACHE_PATH, exist_ok=True)
    return DATASET_AUDIT, DATASET_FINGERPRINT, GRAPH_CACHE_PATH, valid_cases


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2b. Optional: Hugging Face Hub remote graph cache

    BraTS 2021's ~1,251 cases (vs. 369 for BraTS 2020) make the local SLIC
    graph cache substantially larger. Hugging Face Hub gives free accounts
    generous storage for public dataset repos, so this optionally uses a HF
    dataset repo as a REMOTE cache for the built graphs (`*.graph.pt` /
    `*.meta.pt`): before building a case locally, the cache-build cell (§10)
    checks the the single ZIP archive first and skips local SLIC entirely if it's
    already cached there; after the complete cache is built, it uploads one `ZIP_STORED` archive;
    later runs can pull and resume extraction instead of rebuilding.

    Set `HF_TOKEN` in the environment (or use the stored Hugging Face login);
    the notebook contains no credential. With no token, the cache remains
    local-only. Training resumes from `stage1_latest.pt` or
    `stage2_latest.pt` when those rolling checkpoints are available.
    """)
    return


@app.cell
def _(
    DATASET_FINGERPRINT,
    GRAPH_CACHE_PATH,
    HF_REQUIRED,
    HF_TOKEN_PLACEHOLDER,
    IDENTITY_CONFIG,
    PERSISTENT_BASE,
    config_hash,
    hashlib,
    json,
    os,
    upload_verified,
    write_json_atomic,
):
    from huggingface_hub import HfApi, hf_hub_download, get_token
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    HF_TOKEN = (HF_TOKEN_PLACEHOLDER.strip() or (os.environ.get("HF_TOKEN") or "").strip()
                or (get_token() or "").strip())
    HF_ENABLED = bool(HF_TOKEN)
    HF_REPO_ID = "marvelpokemaster/brats-qos-hrgn-graphs"
    HF_REPO_TYPE = "dataset"
    HF_MODEL_REPO_ID = "marvelpokemaster/brats-qos-hrgn-model"
    HF_MODEL_REPO_TYPE = "model"
    HF_CACHE_SUBDIR = f"review_v3_{config_hash(IDENTITY_CONFIG['graph'])}_{DATASET_FINGERPRINT[:16]}"
    HF_CACHE_MANIFEST_NAME = f"{HF_CACHE_SUBDIR}.zip.manifest.json"
    HF_CACHE_ZIP_NAME = f"{HF_CACHE_SUBDIR}.zip"
    GRAPH_CACHE_RECEIPT_PATH = os.path.join(PERSISTENT_BASE, "graph_cache_receipt.json")
    GRAPH_CACHE_ZIP_PATH = GRAPH_CACHE_PATH + ".zip"
    STAGE1_REMOTE_NAME = "stage1_graph_model_review_v3.pt"
    STAGE1_MANIFEST_NAME = "stage1_graph_model_review_v3.manifest.json"
    STAGE1_LATEST_NAME = "stage1_latest_review_v3.pt"
    STAGE2_LATEST_NAME = "stage2_latest_gpu_v4.pt"
    STAGE1_CKPT_PATH = os.path.join(PERSISTENT_BASE, STAGE1_REMOTE_NAME)
    STAGE1_LATEST_PATH = os.path.join(PERSISTENT_BASE, STAGE1_LATEST_NAME)
    STAGE2_LATEST_PATH = os.path.join(PERSISTENT_BASE, STAGE2_LATEST_NAME)

    hf_api = HfApi(token=HF_TOKEN) if HF_ENABLED else None
    if HF_ENABLED:
        hf_api.create_repo(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, private=False, exist_ok=True)
        hf_api.create_repo(repo_id=HF_MODEL_REPO_ID, repo_type=HF_MODEL_REPO_TYPE, private=True, exist_ok=True)
        print("HF enabled")
    else:
        print("HF disabled")


    def sha256_file(path):
        _hash = hashlib.sha256()
        with open(path, "rb") as _fh:
            while True:
                _chunk = _fh.read(1 << 20)
                if not _chunk:
                    break
                _hash.update(_chunk)
        return _hash.hexdigest()


    def hf_upload_file_verified(local_path, remote_name, repo_id, repo_type, commit_message):
        if not HF_ENABLED or hf_api is None:
            raise RuntimeError("Hugging Face token is required; fill HF_TOKEN_PLACEHOLDER before running")
        _receipt = upload_verified(hf_api, lambda **kw: hf_hub_download(token=HF_TOKEN, **kw),
            local_path, remote_name, repo_id, repo_type, commit_message)
        _receipt_path = os.path.join(PERSISTENT_BASE, "hf_upload_receipts.json")
        _receipts = {}
        if os.path.exists(_receipt_path):
            with open(_receipt_path) as _fh:
                _receipts = json.load(_fh)
        _receipts[repo_id + "/" + remote_name] = _receipt
        write_json_atomic(_receipt_path, _receipts)
        return _receipt["sha256"]


    def hf_upload_receipt(repo_id, remote_name):
        with open(os.path.join(PERSISTENT_BASE, "hf_upload_receipts.json")) as _fh:
            return json.load(_fh)[repo_id + "/" + remote_name]


    def hf_try_download(remote_name, repo_id, repo_type, revision="main"):
        try:
            return hf_hub_download(repo_id=repo_id, repo_type=repo_type,
                                   filename=remote_name, revision=revision,
                                   force_download=True, token=HF_TOKEN)
        except EntryNotFoundError:
            return None
        except RepositoryNotFoundError:
            if HF_REQUIRED:
                raise RuntimeError("Hub repository unavailable: check token access and repository IDs")
            return None

    return (
        GRAPH_CACHE_RECEIPT_PATH,
        GRAPH_CACHE_ZIP_PATH,
        HF_CACHE_MANIFEST_NAME,
        HF_CACHE_SUBDIR,
        HF_CACHE_ZIP_NAME,
        HF_ENABLED,
        HF_MODEL_REPO_ID,
        HF_MODEL_REPO_TYPE,
        HF_REPO_ID,
        HF_REPO_TYPE,
        HF_TOKEN,
        STAGE1_CKPT_PATH,
        STAGE1_MANIFEST_NAME,
        STAGE1_REMOTE_NAME,
        STAGE2_LATEST_NAME,
        STAGE2_LATEST_PATH,
        hf_api,
        hf_hub_download,
        hf_try_download,
        hf_upload_file_verified,
        hf_upload_receipt,
        sha256_file,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Utility functions

    Each modality is z-scored inside the brain mask. SLIC is only a **regionalisation step**; labels are voxel fractions per node.
    """)
    return


@app.cell
def _(
    ET_MIN_VOXELS,
    MODALITIES,
    NODE_FEATURE_MODALITIES,
    USE_CC_POSTPROCESS,
    VOXEL_MARGIN,
    VOXEL_MAX_SIZE,
    nib,
    np,
    torch,
):
    def load_nii(path):
        return nib.load(path).get_fdata().astype(np.float32)

    def normalize_brain(vol):
        mask = vol > 0
        out = np.zeros_like(vol, dtype=np.float32)
        if mask.any():
            vals = vol[mask]
            out[mask] = (vals - vals.mean()) / (vals.std() + 1e-8)
        return out

    CLASS_NAMES = ["BG", "NCR/NET", "ED", "ET"]
    NUM_CLASSES = 4

    def remap_labels(seg):
        # BraTS raw labels 0/1/2/4 -> 0/1/2/3
        out = np.zeros(seg.shape, dtype=np.uint8)
        out[seg == 1] = 1
        out[seg == 2] = 2
        out[seg == 4] = 3
        return out

    def load_case(files):
        vols = {m: load_nii(files[m]) for m in MODALITIES}
        brain = np.zeros(vols["t1"].shape, dtype=bool)
        for m in NODE_FEATURE_MODALITIES:
            brain |= vols[m] > 0
        vols = {m: normalize_brain(v) for m, v in vols.items()}
        seg = remap_labels(load_nii(files["seg"]).round().astype(np.int64))
        return vols, seg, brain

    def crop_bounds(brain):
        coords = np.argwhere(brain)
        return coords.min(axis=0), coords.max(axis=0) + 1

    def crop(vol, lo, hi):
        return vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    def dice_binary(pred, target, eps=1e-6):
        pred = pred.astype(bool)
        target = target.astype(bool)
        inter = np.logical_and(pred, target).sum()
        return float((2 * inter + eps) / (pred.sum() + target.sum() + eps))

    # Metric implementations are shared by both stages.

    def postprocess_prediction(pred, et_min_voxels=ET_MIN_VOXELS):
        """Standard BraTS trick: a tiny predicted ET is almost always a false positive
        (LGG cases often have no ET). Relabel it as NCR/NET so TC is preserved."""
        et = pred == 3
        n_et = int(et.sum())
        if 0 < n_et < et_min_voxels:
            pred = pred.copy()
            pred[et] = 1
        return pred

    def fmt_metrics(m):
        return f"WT={m['Dice_WT']:.4f} | TC={m['Dice_TC']:.4f} | ET={m['Dice_ET']:.4f}"

    # ============================================================
    # Voxel refinement helpers
    # ============================================================
    def tumour_crop_bounds(seg_c, margin=VOXEL_MARGIN, max_size=VOXEL_MAX_SIZE):
        """Crop to the tumour bounding box + margin for the voxel refinement head.

        Returns (lo, hi) in the cropped-brain coordinate system, or (None, None) if
        the case has no tumour. If the crop exceeds max_size, it is centred and
        truncated to keep the U-Net memory-bounded.
        """
        coords = np.argwhere(seg_c > 0)
        if len(coords) == 0:
            return None, None
        lo = np.maximum(coords.min(axis=0) - margin, 0)
        hi = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(seg_c.shape))
        # Clamp to max_size (centred on the tumour)
        sz = hi - lo
        for ax in range(3):
            if sz[ax] > max_size:
                ctr = (lo[ax] + hi[ax]) // 2
                lo[ax] = max(0, ctr - max_size // 2)
                hi[ax] = lo[ax] + max_size
        return lo, hi


    def project_node_probs_cropped(node_probs, meta):
        """Return (X, Y, Z, NUM_CLASSES) probability maps in the cropped brain region."""
        crop_shape = tuple(meta["crop_shape"])
        acc = np.zeros(crop_shape + (NUM_CLASSES,), dtype=np.float32)
        count = np.zeros(crop_shape, dtype=np.float32)
        for nt, probs in node_probs.items():
            nm = meta["node_map"][nt]
            valid = nm >= 0
            acc[valid] += np.asarray(probs, dtype=np.float32)[nm[valid]]
            count[valid] += 1.0
        valid = count > 0
        acc[valid] /= count[valid, None]
        return acc


    def postprocess_prediction_cc(pred, et_min_voxels=ET_MIN_VOXELS):
        """Connected-component post-processing: remove small ET islands individually.

        More precise than the global-volume-threshold approach: a large total ET
        volume with one big island and several tiny false-positive islands will
        keep the big island and remove only the spurious ones.
        """
        from scipy.ndimage import label as _cc_label
        pred = pred.copy()
        et = pred == 3
        if et.any():
            labeled, n_cc = _cc_label(et)
            for _i in range(1, n_cc + 1):
                comp = labeled == _i
                if int(comp.sum()) < et_min_voxels:
                    pred[comp] = 1   # relabel as NCR/NET (preserves TC)
        return pred


    def postprocess_prediction_auto(pred, et_min_voxels=ET_MIN_VOXELS):
        """Use connected-component post-processing when enabled, else the original."""
        if USE_CC_POSTPROCESS:
            return postprocess_prediction_cc(pred, et_min_voxels)
        return postprocess_prediction(pred, et_min_voxels)


    def load_meta(mpath):
        return torch.load(mpath, weights_only=False)
    import sys, os
    _nb_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if _nb_dir not in sys.path: sys.path.insert(0, _nb_dir)
    from brats_protocol import (PROTOCOL_VERSION, segmentation_metrics, summarize_metric_rows,
                                mean_region_dice, sliding_window_predict, make_training_patch, audit_dataset)
    from brats_gpu import (bounded_prefetch, patient_groups, configure_gpu, amp_context, train_patient_group, train_voxel_experiment)
    from brats_transfer import (upload_verified, verify_remote, extract_verified_zip, write_json_atomic)

    return (
        CLASS_NAMES,
        NUM_CLASSES,
        PROTOCOL_VERSION,
        amp_context,
        audit_dataset,
        bounded_prefetch,
        configure_gpu,
        crop,
        crop_bounds,
        extract_verified_zip,
        fmt_metrics,
        load_case,
        load_meta,
        make_training_patch,
        mean_region_dice,
        patient_groups,
        postprocess_prediction,
        postprocess_prediction_auto,
        segmentation_metrics,
        sliding_window_predict,
        summarize_metric_rows,
        train_patient_group,
        train_voxel_experiment,
        upload_verified,
        verify_remote,
        write_json_atomic,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Build the heterogeneous supervoxel graph (vectorised)

    Two node types, each a separate SLIC partition of the *same* cropped brain:

    - `t1ce`: supervoxels whose boundaries follow T1ce contrast (ET / NCR boundaries).
    - `flair`: supervoxels whose boundaries follow FLAIR contrast (edema boundary).

    Relations:

    - `t1ce --spatial--> t1ce`, `flair --spatial--> flair`: face-adjacent supervoxels. Edge attributes `[normalised centroid distance, normalised shared-face count]`.
    - `t1ce --corresponds--> flair` and back: supervoxels that **overlap in voxel space**. Edge attributes `[normalised centroid distance, overlap / min(volume)]`.

    Per node (32 features): for each of T1, T1ce, T2, FLAIR → mean, std and the 10/25/50/75/90 % quantiles of the voxels inside the supervoxel; plus normalised centroid (3) and log relative volume (1).

    Per node targets: `y_frac` (fraction of voxels in each of BG / NCR-NET / ED / ET), `y` (majority class, for reporting only), `vol` (fraction of brain voxels, used to volume-weight the Dice loss so it equals voxel Dice).

    Everything is `bincount`/`lexsort` based — no per-supervoxel Python loops — so 15 000 nodes cost about the same as 400 did in v1.
    """)
    return


@app.cell
def _(
    COMPACTNESS,
    HeteroData,
    MIN_OVERLAP,
    MODALITIES,
    NODE_FEATURE_MODALITIES,
    NODE_TYPES,
    NUM_CLASSES,
    N_SEGMENTS,
    QUANTILES,
    SLIC_ITERS,
    SLIC_MODALITIES,
    crop,
    crop_bounds,
    load_case,
    np,
    slic,
    torch,
):
    def _label_mean_std(lab, vals, K):
        cnt = np.bincount(lab, minlength=K).astype(np.float64)
        s1 = np.bincount(lab, weights=vals, minlength=K)
        s2 = np.bincount(lab, weights=vals * vals, minlength=K)
        mean = s1 / np.maximum(cnt, 1)
        var = np.maximum(s2 / np.maximum(cnt, 1) - mean * mean, 0.0)
        return (mean, np.sqrt(var))

    def _label_quantiles(lab, vals, K, qs):
        order = np.lexsort((vals, lab))  # sort by label, then value
        lab_s, vals_s = (lab[order], vals[order])
        cnt = np.bincount(lab_s, minlength=K)
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        out = np.empty((K, len(qs)), dtype=np.float64)
        for qi, q in enumerate(qs):
            pos = starts + np.floor(q * np.maximum(cnt - 1, 0)).astype(np.int64)
            out[:, qi] = vals_s[np.minimum(pos, len(vals_s) - 1)]
        return out

    def slic_node_map(vol_c, brain_c, n_segments, compactness, max_iter):
        """Run SLIC inside the brain mask; return an int32 array of node indices (-1 outside the brain)."""
        segments = slic(vol_c, n_segments=n_segments, compactness=compactness, max_num_iter=max_iter, start_label=1, mask=brain_c, channel_axis=None)
        ids = np.unique(segments[brain_c])
        lut = np.full(int(segments.max()) + 1, -1, dtype=np.int32)
        lut[ids] = np.arange(len(ids), dtype=np.int32)
        node_map = lut[segments]
        node_map[~brain_c] = -1
        return (node_map, int(len(ids)))

    def node_table(node_map, brain_c, vols_c, seg_c):
        lab = node_map[brain_c].astype(np.int64)
        K = int(lab.max()) + 1
        cnt = np.bincount(lab, minlength=K).astype(np.float64)
        feats = []
        for m in NODE_FEATURE_MODALITIES:
            v = vols_c[m][brain_c].astype(np.float64)
            mean, std = _label_mean_std(lab, v, K)
            feats = feats + [mean[:, None], std[:, None], _label_quantiles(lab, v, K, QUANTILES)]
        xyz = np.argwhere(brain_c).astype(np.float64)
        cent = np.stack([np.bincount(lab, weights=xyz[:, a], minlength=K) / cnt for a in range(3)], axis=1)
        cent_n = cent / np.asarray(brain_c.shape, dtype=np.float64)
        vol_frac = cnt / cnt.sum()
        feats = feats + [cent_n, np.log(vol_frac * K)[:, None]]
        x = np.concatenate(feats, axis=1).astype(np.float32)  # same C-order as node_map[brain_c]
        cls = seg_c[brain_c].astype(np.int64)
        y_frac = np.bincount(lab * NUM_CLASSES + cls, minlength=K * NUM_CLASSES).reshape(K, NUM_CLASSES)
        y_frac = (y_frac / cnt[:, None]).astype(np.float32)
        return dict(x=x, y_frac=y_frac, vol=vol_frac.astype(np.float32), cent=cent.astype(np.float32), cnt=cnt, K=K)  # log relative size (0 = average node)

    def spatial_edges(node_map, cent, K):
        keys = []
        for axis in range(3):
            a_sl = [slice(None)] * 3
            b_sl = [slice(None)] * 3
            a_sl[axis] = slice(0, -1)
            b_sl[axis] = slice(1, None)
            a, b = (node_map[tuple(a_sl)], node_map[tuple(b_sl)])
            valid = (a >= 0) & (b >= 0) & (a != b)
            u, v = (a[valid].astype(np.int64), b[valid].astype(np.int64))
            keys.append(u * K + v)
            keys.append(v * K + u)
        keys, faces = np.unique(np.concatenate(keys), return_counts=True)
        u, v = (keys // K, keys % K)
        dist = np.linalg.norm(cent[u] - cent[v], axis=1)
        attr = np.stack([dist / (dist.mean() + 1e-08), faces / (faces.mean() + 1e-08)], axis=1).astype(np.float32)
        return (np.stack([u, v]).astype(np.int64), attr)

    def overlap_edges(map_a, map_b, brain_c, tab_a, tab_b, min_overlap=MIN_OVERLAP):
        la = map_a[brain_c].astype(np.int64)
        lb = map_b[brain_c].astype(np.int64)
        Kb = tab_b['K']
        keys, n_shared = np.unique(la * Kb + lb, return_counts=True)
        i, j = (keys // Kb, keys % Kb)
        overlap = n_shared / np.minimum(tab_a['cnt'][i], tab_b['cnt'][j])

        def _best(group):
            order = np.lexsort((-n_shared, group))
            first = np.ones(len(group), dtype=bool)
            first[1:] = group[order][1:] != group[order][:-1]
            mark = np.zeros(len(group), dtype=bool)
            mark[order[first]] = True
            return mark  # mark the strongest partner of every node in `group`
        keep = (overlap >= min_overlap) | _best(i) | _best(j)
        i, j, overlap = (i[keep], j[keep], overlap[keep])
        dist = np.linalg.norm(tab_a['cent'][i] - tab_b['cent'][j], axis=1)
        attr = np.stack([dist / (dist.mean() + 1e-08), overlap], axis=1).astype(np.float32)
        return (np.stack([i, j]).astype(np.int64), attr)

    assert tuple(NODE_TYPES) == tuple(SLIC_MODALITIES)
    assert tuple(NODE_FEATURE_MODALITIES) == tuple(MODALITIES)

    def build_hetero_case(files, n_segments=N_SEGMENTS, compactness=COMPACTNESS, max_iter=SLIC_ITERS):
        vols, seg, brain = load_case(files)
        lo, hi = crop_bounds(brain)
        brain_c = crop(brain, lo, hi)
        seg_c = crop(seg, lo, hi)
        vols_c = {m: crop(v, lo, hi) for m, v in vols.items()}
        parts = {}
        for nt in SLIC_MODALITIES:
            node_map, K = slic_node_map(vols_c[nt], brain_c, n_segments, compactness, max_iter)
            tab = node_table(node_map, brain_c, vols_c, seg_c)
            ei, ea = spatial_edges(node_map, tab['cent'], K)
            parts[nt] = dict(node_map=node_map, tab=tab, ei=ei, ea=ea)
        cross_ei, cross_ea = overlap_edges(parts['t1ce']['node_map'], parts['flair']['node_map'], brain_c, parts['t1ce']['tab'], parts['flair']['tab'])
        data = HeteroData()
        for nt in NODE_TYPES:
            tab = parts[nt]['tab']
            data[nt].x = torch.from_numpy(tab['x'])
            data[nt].y_frac = torch.from_numpy(tab['y_frac'])
            data[nt].y = torch.from_numpy(tab['y_frac'].argmax(axis=1).astype(np.int64))
            data[nt].vol = torch.from_numpy(tab['vol'])
            data[nt, 'spatial', nt].edge_index = torch.from_numpy(parts[nt]['ei'])
            data[nt, 'spatial', nt].edge_attr = torch.from_numpy(parts[nt]['ea'])
        data['t1ce', 'corresponds', 'flair'].edge_index = torch.from_numpy(cross_ei)
        data['t1ce', 'corresponds', 'flair'].edge_attr = torch.from_numpy(cross_ea)
        data['flair', 'corresponds', 't1ce'].edge_index = torch.from_numpy(cross_ei[::-1].copy())
        data['flair', 'corresponds', 't1ce'].edge_attr = torch.from_numpy(cross_ea.copy())
        meta = {'seg': seg.astype(np.uint8), 'lo': lo, 'hi': hi, 'crop_shape': tuple(brain_c.shape), 'original_shape': tuple(seg.shape), 'node_map': {nt: parts[nt]['node_map'].astype(np.int32) for nt in NODE_TYPES}, 'vis': {m: vols_c[m].astype(np.float16) for m in MODALITIES}, 'slic_modalities': list(SLIC_MODALITIES), 'node_feature_modalities': list(NODE_FEATURE_MODALITIES)}
        return (data, meta)  # all 4 modalities for voxel head

    return (build_hetero_case,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Voxel reconstruction and the oracle (achievable segmentation accuracy)

    `project_node_probs` paints each node's 4-class probability vector onto its supervoxel for both partitions, averages where both cover a voxel, and takes the arg-max.

    The **oracle** feeds the *ground-truth* node fractions through the same projection. Its Dice is the upper bound any node classifier can reach on this partition. If the oracle is low, the fix is graph construction, not the model.
    """)
    return


@app.cell
def _(
    NODE_TYPES,
    NUM_CLASSES,
    USE_STRUCTURAL_REFINEMENT,
    np,
    postprocess_prediction_auto,
    segmentation_metrics,
    torch,
):
    def project_node_probs(node_probs, meta):
        crop_shape = tuple(meta['crop_shape'])
        acc = np.zeros(crop_shape + (NUM_CLASSES,), dtype=np.float32)
        count = np.zeros(crop_shape, dtype=np.float32)
        for nt, probs in node_probs.items():
            nm = meta['node_map'][nt]
            valid = nm >= 0
            acc[valid] = acc[valid] + np.asarray(probs, dtype=np.float32)[nm[valid]]
            count[valid] = count[valid] + 1.0
        valid = count > 0
        acc[valid] = acc[valid] / count[valid, None]
        pred_c = acc.argmax(axis=-1).astype(np.uint8)
        pred_c[~valid] = 0
        pred = np.zeros(tuple(meta['original_shape']), dtype=np.uint8)
        lo, hi = (meta['lo'], meta['hi'])
        pred[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = pred_c
        return pred

    def oracle_prediction(data, meta, soft=False):
        """Reconstruction from ground-truth node labels. soft=True averages the class
        fractions; soft=False uses one-hot majority labels (what a perfect hard classifier gives)."""
        node_probs = {}
        for nt in NODE_TYPES:
            yf = data[nt].y_frac.cpu().numpy()
            node_probs[nt] = yf if soft else np.eye(NUM_CLASSES, dtype=np.float32)[yf.argmax(axis=1)]
        return project_node_probs(node_probs, meta)

    def oracle_metrics(data, meta, soft=False):
        return segmentation_metrics(oracle_prediction(data, meta, soft=soft), meta['seg'])

    @torch.no_grad()
    def predict_node_probs(model, data):
        """Baseline-compatible: returns the FINAL prediction (identical to the
        plain model output when USE_STRUCTURAL_REFINEMENT=False; the refined
        prediction otherwise). Used by reconstruct_case/evaluate_items below,
        which therefore keeps reporting the same WT/TC/ET metrics either way.
        See predict_node_probs_both (Option B cell) for initial-vs-refined."""
        model.eval()
        dev = next(model.parameters()).device
        if USE_STRUCTURAL_REFINEMENT:
            _, logits, _ = model(data.clone().to(dev), teacher_prob=0.0)  # never GT at eval
        else:
            logits, _ = model(data.clone().to(dev))
        return {nt: torch.softmax(logits[nt].float(), dim=-1).cpu().numpy() for nt in NODE_TYPES}

    def reconstruct_case(model, data, meta, postprocess=True):
        pred = project_node_probs(predict_node_probs(model, data), meta)
        return postprocess_prediction_auto(pred) if postprocess else pred

    return (
        oracle_metrics,
        predict_node_probs,
        project_node_probs,
        reconstruct_case,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Oracle sweep over the number of supervoxels

    Runs SLIC at several `n_segments` on a handful of cases and reports the ceiling Dice, node count and build time. Use this to justify `N_SEGMENTS` (Saueressig et al. tuned exactly this quantity, calling it ASA). Set `ORACLE_CASES = 0` in the config cell to skip.
    """)
    return


@app.cell
def _(
    COMPACTNESS,
    ORACLE_CASES,
    ORACLE_SWEEP,
    Parallel,
    build_hetero_case,
    delayed,
    np,
    oracle_metrics,
    os,
    time,
    valid_cases,
):
    def partition_ceiling(files, n_segments, compactness):
        t0 = time.time()
        d, m = build_hetero_case(files, n_segments=n_segments, compactness=compactness)
        row = dict(case=os.path.basename(os.path.dirname(files["t1"])), k=n_segments,
                   nodes=int(d["t1ce"].x.size(0)), secs=time.time() - t0)
        row.update({f"hard_{k[5:]}": v for k, v in oracle_metrics(d, m, soft=False).items()})
        row.update({f"soft_{k[5:]}": v for k, v in oracle_metrics(d, m, soft=True).items()})
        return row

    oracle_rows = []
    if ORACLE_CASES and ORACLE_SWEEP:
        _step = max(1, len(valid_cases) // ORACLE_CASES)
        _sweep_cases = [files for _, files in valid_cases[::_step][:ORACLE_CASES]]
        _tasks = [(f, k) for f in _sweep_cases for k in ORACLE_SWEEP]
        print(f"Oracle sweep: {len(_sweep_cases)} cases x {list(ORACLE_SWEEP)} segments ...")
        oracle_rows = Parallel(n_jobs=min(len(_tasks), os.cpu_count() or 4), backend="loky")(
            delayed(partition_ceiling)(f, k, COMPACTNESS) for f, k in _tasks
        )
        print(f"\n{'k':>7} | {'nodes':>6} | {'sec':>5} | {'oracle hard WT/TC/ET':^26} | {'oracle soft WT/TC/ET':^26}")
        print("-" * 85)
        for _k in ORACLE_SWEEP:
            _rs = [r for r in oracle_rows if r["k"] == _k]
            _mean = lambda key: np.mean([r[key] for r in _rs])
            print(f"{_k:>7} | {_mean('nodes'):>6.0f} | {_mean('secs'):>5.0f} | "
                  f"{_mean('hard_WT'):.3f} / {_mean('hard_TC'):.3f} / {_mean('hard_ET'):.3f}       | "
                  f"{_mean('soft_WT'):.3f} / {_mean('soft_TC'):.3f} / {_mean('soft_ET'):.3f}")
        print("\nThe oracle is the ceiling for a perfect node classifier on that partition.")
    else:
        print("Oracle sweep skipped (ORACLE_CASES=0).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Inspect one heterogeneous graph
    """)
    return


@app.cell
def _(build_hetero_case, fmt_metrics, oracle_metrics, valid_cases):
    sample_data, sample_meta = build_hetero_case(valid_cases[0][1])
    print(sample_data)
    print("\nMetadata:", sample_data.metadata())
    for _nt in sample_data.node_types:
        _yf = sample_data[_nt].y_frac
        print(f"{_nt}: nodes={_yf.size(0)}  pure nodes={(float((_yf.max(dim=1).values > 0.999).float().mean()) * 100):.1f}%  "
              f"class voxel-fraction={[round(float(v), 4) for v in (sample_data[_nt].vol[:, None] * _yf).sum(0)]}")
    for _et in sample_data.edge_types:
        print(_et, "edges:", sample_data[_et].edge_index.shape[1])
    print("\nOracle on this case (hard):", fmt_metrics(oracle_metrics(sample_data, sample_meta)))
    print("Oracle on this case (soft):", fmt_metrics(oracle_metrics(sample_data, sample_meta, soft=True)))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. HGT + adaptive multi-hop propagation

    ### HGT
    `HGTConv` is the heterogeneous backbone: separate node types, typed relations. v2 wraps each layer with a residual connection and LayerNorm so two layers train stably at `hidden_dim=128`.

    ### Adaptive propagation
    After HGT, an adaptive message weight is computed for every edge from

    - source / destination embeddings
    - the edge attributes (distance and contact/overlap strength)
    - feature cosine similarity
    - cross-modal relation indicator
    - current prediction uncertainty (normalised entropy)

    Messages are gate-normalised (attention-style) rather than summed raw. Applied `HOPS` times, so information can travel beyond immediate neighbours. Set `HOPS = 0` for the *HGT-only* ablation.
    """)
    return


@app.cell
def _(
    F,
    HGTConv,
    HOP_TEMPERATURE,
    K_MAX,
    NODE_TYPES,
    USE_ADAPTIVE_HOPS,
    math,
    nn,
    torch,
):
    class AdaptiveHeteroPropagation(nn.Module):
        def __init__(self, hidden_dim, edge_types, node_types, edge_attr_dim=2, dropout=0.1):
            super().__init__()
            self.edge_types = edge_types
            self.msg = nn.ModuleDict()
            self.gate = nn.ModuleDict()
            for et in edge_types:
                key = "__".join(et)
                self.msg[key] = nn.Linear(hidden_dim, hidden_dim)
                self.gate[key] = nn.Sequential(
                    nn.Linear(hidden_dim * 2 + edge_attr_dim + 3, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1),
                )
            self.norm = nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in node_types})
            self.drop = nn.Dropout(dropout)

        def forward(self, x_dict, edge_index_dict, edge_attr_dict, logits_dict, return_gates=False):
            out = {nt: torch.zeros(x.size(0), x.size(1), device=x.device, dtype=torch.float32) for nt, x in x_dict.items()}
            denom = {nt: torch.zeros(x.size(0), 1, device=x.device, dtype=torch.float32) for nt, x in x_dict.items()}
            edge_gates = {}

            unc = {}
            for nt, logits in logits_dict.items():
                p = torch.softmax(logits.float(), dim=-1).clamp_min(1e-8)
                unc[nt] = -(p * p.log()).sum(dim=-1) / math.log(p.size(-1))

            for et in self.edge_types:
                src_t, rel, dst_t = et
                key = "__".join(et)
                ei = edge_index_dict[et]
                if ei.numel() == 0:
                    if return_gates:
                        edge_gates[et] = torch.empty(0, device=x_dict[src_t].device)
                    continue
                src, dst = ei
                hs = x_dict[src_t][src].float()
                hd = x_dict[dst_t][dst].float()
                ea = edge_attr_dict[et].float()
                sim = F.cosine_similarity(hs, hd, dim=-1).unsqueeze(1)
                cross = torch.full((ea.size(0), 1), 1.0 if rel == "corresponds" else 0.0, device=hs.device)
                u = ((unc[src_t][src] + unc[dst_t][dst]) / 2).unsqueeze(1)

                alpha = torch.sigmoid(self.gate[key](torch.cat([hs, hd, ea, sim, cross, u], dim=1))).float()
                out[dst_t].index_add_(0, dst, self.msg[key](hs).float() * alpha)
                denom[dst_t].index_add_(0, dst, alpha)
                if return_gates:
                    edge_gates[et] = alpha.squeeze(1)

            updated = {
                nt: self.norm[nt](x_dict[nt].float() + self.drop(F.relu(out[nt] / (denom[nt] + 1.0))))
                for nt in x_dict
            }
            return (updated, edge_gates) if return_gates else updated


    class QoSHRGN(nn.Module):
        def __init__(self, in_dim, hidden_dim=128, heads=4, num_classes=4, hops=2, hgt_layers=2,
                     dropout=0.1, adaptive_hops=USE_ADAPTIVE_HOPS, k_max=K_MAX,
                     hop_temperature=HOP_TEMPERATURE):
            super().__init__()
            self.metadata = (
                list(NODE_TYPES),
                [
                    ("t1ce", "spatial", "t1ce"),
                    ("flair", "spatial", "flair"),
                    ("t1ce", "corresponds", "flair"),
                    ("flair", "corresponds", "t1ce"),
                ],
            )
            self.adaptive_hops = adaptive_hops
            self.k_max = k_max
            self.hop_temperature = hop_temperature
            self.input_proj = nn.ModuleDict({
                nt: nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
                for nt in NODE_TYPES
            })
            self.hgt = nn.ModuleList([
                HGTConv(in_channels=hidden_dim, out_channels=hidden_dim, metadata=self.metadata, heads=heads)
                for _ in range(hgt_layers)
            ])
            self.hgt_norm = nn.ModuleList([
                nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in NODE_TYPES}) for _ in range(hgt_layers)
            ])
            self.pre_head = nn.ModuleDict({nt: nn.Linear(hidden_dim, num_classes) for nt in NODE_TYPES})
            if adaptive_hops:
                self.adaptive_prop = AdaptiveHeteroPropagation(
                    hidden_dim, self.metadata[1], self.metadata[0], edge_attr_dim=2, dropout=dropout)
                self.hop_selector = nn.ModuleDict({
                    nt: nn.Sequential(
                        nn.Linear(hidden_dim + 2, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                        nn.Linear(hidden_dim // 2, 1),
                    ) for nt in NODE_TYPES
                })
            else:
                self.adaptive = nn.ModuleList([
                    AdaptiveHeteroPropagation(hidden_dim, self.metadata[1], self.metadata[0], edge_attr_dim=2, dropout=dropout)
                    for _ in range(hops)
                ])
            self.final_head = nn.ModuleDict({nt: nn.Linear(hidden_dim, num_classes) for nt in NODE_TYPES})
            self.drop = nn.Dropout(dropout)
            self.hop_regularization = torch.tensor(0.0)
            self.last_hop_info = {}
            self.last_edge_gates = {}

        @staticmethod
        def _uncertainty(logits):
            p = torch.softmax(logits.float(), dim=-1).clamp_min(1e-8)
            return (-(p * p.log()).sum(dim=-1) / math.log(p.size(-1))).unsqueeze(1)

        def _adaptive_hop_mix(self, x_dict, edge_index_dict, edge_attr_dict):
            states = [{nt: x_dict[nt] for nt in NODE_TYPES}]
            logits_states = [{nt: self.pre_head[nt](x_dict[nt]) for nt in NODE_TYPES}]
            gate_history = []
            current = x_dict
            for _ in range(self.k_max):
                current, gates = self.adaptive_prop(
                    current, edge_index_dict, edge_attr_dict, logits_states[-1], return_gates=True)
                states.append({nt: current[nt] for nt in NODE_TYPES})
                logits_states.append({nt: self.pre_head[nt](current[nt]) for nt in NODE_TYPES})
                gate_history.append({et: alpha.detach() for et, alpha in gates.items()})

            mixed = {}
            hop_info = {}
            expected_terms = []
            for nt in NODE_TYPES:
                scores = []
                previous = states[0][nt]
                for k, (state, logits) in enumerate(zip(states, logits_states)):
                    h = state[nt]
                    delta = torch.zeros(h.size(0), 1, device=h.device) if k == 0 else (
                        1.0 - F.cosine_similarity(h.float(), previous.float(), dim=-1)).unsqueeze(1)
                    scores.append(self.hop_selector[nt](torch.cat([h, self._uncertainty(logits[nt]), delta], dim=1)))
                    previous = h
                beta = torch.softmax(torch.cat(scores, dim=1) / self.hop_temperature, dim=1)
                mixed[nt] = sum(beta[:, k:k + 1] * states[k][nt] for k in range(self.k_max + 1))
                hop_values = torch.arange(self.k_max + 1, device=beta.device, dtype=beta.dtype)
                effective = (beta * hop_values.unsqueeze(0)).sum(dim=1)
                expected_terms.append(effective.mean())
                hop_info[nt] = {
                    "weights": beta.detach(),
                    "effective_hop": effective.detach(),
                    "selected_hop": beta.detach().argmax(dim=1),
                }
            self.hop_regularization = torch.stack(expected_terms).mean() / max(self.k_max, 1)
            self.last_hop_info = hop_info
            self.last_edge_gates = gate_history
            return mixed

        def forward(self, data):
            x_dict = {nt: self.input_proj[nt](data[nt].x) for nt in NODE_TYPES}
            edge_index_dict = {et: data[et].edge_index for et in self.metadata[1]}
            edge_attr_dict = {et: data[et].edge_attr for et in self.metadata[1]}

            for conv, norms in zip(self.hgt, self.hgt_norm):
                h = conv(x_dict, edge_index_dict)
                x_dict = {nt: norms[nt](x_dict[nt] + self.drop(F.relu(h[nt]))) for nt in x_dict}

            if self.adaptive_hops:
                x_dict = self._adaptive_hop_mix(x_dict, edge_index_dict, edge_attr_dict)
            else:
                logits_dict = {nt: self.pre_head[nt](x_dict[nt]) for nt in x_dict}
                for layer in self.adaptive:
                    x_dict = layer(x_dict, edge_index_dict, edge_attr_dict, logits_dict)
                    logits_dict = {nt: self.pre_head[nt](x_dict[nt]) for nt in x_dict}
                self.hop_regularization = torch.zeros((), device=next(iter(x_dict.values())).device)
                self.last_hop_info = {}
                self.last_edge_gates = {}

            final_logits = {nt: self.final_head[nt](x_dict[nt]) for nt in x_dict}
            return final_logits, x_dict

    return (QoSHRGN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8d. Voxel refinement head (3D U-Net)

    The supervoxel oracle ceiling for ET is 0.7243 at 15k segments — **even a perfect
    node classifier cannot reach 0.8** because ET and NCR/NET are mixed within the same
    supervoxels. A voxel-level refinement head breaks this ceiling by discriminating
    *within* supervoxels using the original MRI intensity at full voxel resolution.

    **Input:** 4 MRI modalities + 4 projected node probability maps = 8 channels.
    **Output:** 4-class voxel-level logits.
    **Architecture:** lightweight 3D U-Net (2 downsample levels, ~1M params).
    **Training:** two-stage — graph model is frozen, only the voxel head is trained.
    """)
    return


@app.cell
def _(
    F,
    GPU_AUTOTUNE,
    GPU_MEMORY_FRACTION,
    GPU_PRECISION,
    INFERENCE_MAX_BATCH,
    INFERENCE_OVERLAP,
    VOXEL_MAX_SIZE,
    nn,
    postprocess_prediction_auto,
    predict_node_probs,
    sliding_window_predict,
    torch,
):
    # ============================================================
    # VoxelRefinementHead: 3D U-Net that refines node predictions at voxel resolution
    # ============================================================
    class VoxelRefinementHead(nn.Module):
        """Lightweight 3D U-Net for voxel-level refinement.

        Input: (B, 8, X, Y, Z) — 4 MRI modalities + 4 projected node probability maps
        Output: ((B, 4, X, Y, Z), (B, 1, X, Y, Z)) — 4-class voxel-level logits and
        a boundary-map logit predicted from the same full-resolution decoder
        features, so the shared decoder is explicitly supervised on where label
        transitions are (the hardest voxels for a supervoxel-projected prior).

        Can exceed the supervoxel oracle ceiling because it operates at voxel
        resolution and can split ET from NCR/NET within the same supervoxel using
        T1ce intensity patterns.
        """

        def __init__(self, in_channels=8, num_classes=4, base_channels=32):
            super().__init__()
            c = base_channels
            self.brats_gpu_options = dict(precision=GPU_PRECISION, autotune=GPU_AUTOTUNE,
                max_inference_batch=INFERENCE_MAX_BATCH, memory_fraction=GPU_MEMORY_FRACTION)
            # Encoder
            self.enc1 = self._block(in_channels, c)       # full res
            self.enc2 = self._block(c, c * 2)              # half res
            # Bottleneck
            self.bot = self._block(c * 2, c * 4)           # quarter res
            # Decoder
            self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
            self.dec2 = self._block(c * 4, c * 2)          # half res (with skip)
            self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
            self.dec1 = self._block(c * 2, c)             # full res (with skip)
            # Output
            self.out = nn.Conv3d(c, num_classes, 1)
            self.boundary_out = nn.Conv3d(c, 1, 1)

        def _block(self, in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1), nn.GroupNorm(4, out_c), nn.ReLU(),
                nn.Conv3d(out_c, out_c, 3, padding=1), nn.GroupNorm(4, out_c), nn.ReLU(),
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(F.max_pool3d(e1, 2))
            b = self.bot(F.max_pool3d(e2, 2))
            d2 = self.up2(b)
            if d2.shape[2:] != e2.shape[2:]:
                d2 = F.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[2:] != e1.shape[2:]:
                d1 = F.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1), self.boundary_out(d1)


    def seg_boundary_map(seg, num_classes=4):
        """Ground-truth boundary map from a label volume via morphological gradient.

        One-hot the labels, then dilate and erode each class channel with a 3x3x3
        structuring element (`F.max_pool3d`, erosion = -max_pool3d(-x)). A voxel
        where dilation and erosion disagree for any class sits on a label
        transition. Returns (B, 1, X, Y, Z) floats in {0, 1}.
        """
        oh = F.one_hot(seg, num_classes).permute(0, 4, 1, 2, 3).float()
        dil = F.max_pool3d(oh, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool3d(-oh, kernel_size=3, stride=1, padding=1)
        return (dil - ero).amax(dim=1, keepdim=True).clamp(0, 1)


    def voxel_loss(logits, seg, class_weights=None, dice_weight=1.0,
                   ce_weight=1.0, focal_weight=0.0, focal_gamma=2.0,
                   boundary_logits=None, boundary_weight=0.0,
                   per_patient_dice=True):
        """Dice + ce_weight*CE (+ focal_weight*Focal) (+ boundary_weight*BoundaryBCE) over the 4-class voxel problem.

        logits: (B, 4, X, Y, Z), seg: (B, X, Y, Z) with values 0-3,
        boundary_logits: (B, 1, X, Y, Z) raw logits of the boundary head.

        When per_patient_dice=True (default), Dice is computed per volume and averaged
        across the batch. This ensures consistent per-patient gradient weighting under
        gradient accumulation and unequal microbatch sizes. For B=1, per-patient Dice
        and global batch Dice are identical.
        """
        ce = torch.stack([F.cross_entropy(logits[i:i+1], seg[i:i+1], weight=class_weights)
                              for i in range(logits.size(0))]).mean()
        probs = F.softmax(logits.float(), dim=1)          # (B, 4, X, Y, Z)
        # Region probabilities (same definitions as the graph-level loss)
        p_wt = probs[:, 1:].sum(dim=1)                      # BG vs tumour
        g_wt = (seg > 0).float()
        p_tc = probs[:, 1] + probs[:, 3]                    # NCR/NET + ET
        g_tc = ((seg == 1) | (seg == 3)).float()
        p_et = probs[:, 3]                                  # ET only
        g_et = (seg == 3).float()
        eps = 1e-7
        if per_patient_dice and logits.dim() == 5:
            dims = (-3, -2, -1)
            d_wt = 1.0 - (2.0 * (p_wt * g_wt).sum(dim=dims) + eps) / (p_wt.sum(dim=dims) + g_wt.sum(dim=dims) + eps)
            d_tc = 1.0 - (2.0 * (p_tc * g_tc).sum(dim=dims) + eps) / (p_tc.sum(dim=dims) + g_tc.sum(dim=dims) + eps)
            d_et = 1.0 - (2.0 * (p_et * g_et).sum(dim=dims) + eps) / (p_et.sum(dim=dims) + g_et.sum(dim=dims) + eps)
            dice = ((d_wt + d_tc + d_et) / 3.0).mean()
        else:
            dice = ((1 - (2 * (p_wt * g_wt).sum() + eps) / ((p_wt + g_wt).sum() + eps)) +
                     1 - (2 * (p_tc * g_tc).sum() + eps) / ((p_tc + g_tc).sum() + eps)) / 3.0
            # ET Dice is the hardest and most important — weight it explicitly
            dice = dice + (1 - (2 * (p_et * g_et).sum() + eps) / ((p_et + g_et).sum() + eps)) / 3.0
        loss = dice_weight * dice + ce_weight * ce
        if focal_weight > 0:
            logp = F.log_softmax(logits.float(), dim=1)
            p = torch.exp(logp)
            seg_oh = F.one_hot(seg, 4).float().permute(0, 4, 1, 2, 3)
            fl = -(seg_oh * (1 - p) ** focal_gamma * logp).sum(dim=1).mean()
            loss = loss + focal_weight * fl
        if boundary_logits is not None and boundary_weight > 0:
            gt_boundary = seg_boundary_map(seg, logits.size(1))
            loss = loss + boundary_weight * F.binary_cross_entropy_with_logits(
                boundary_logits.float(), gt_boundary)
        return loss



    @torch.no_grad()
    def reconstruct_case_voxel(model, voxel_head, data, meta, postprocess=True):
        node_probs = predict_node_probs(model, data)
        pred = sliding_window_predict(voxel_head, meta, node_probs,
                                      roi=VOXEL_MAX_SIZE, overlap=INFERENCE_OVERLAP)
        return postprocess_prediction_auto(pred) if postprocess else pred

    return VoxelRefinementHead, reconstruct_case_voxel, voxel_loss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8c. Option C: masking operators and heterogeneous reconstruction heads

    **What is masked.** A fraction `REC_MASK_RATE` of nodes (independently in each partition) has its
    entire 28-dim *appearance* block replaced by a learnable per-node-type `[MASK]` token; the 4 geometry
    dims are kept. A fraction `REC_EDGE_MASK_RATE` of `t1ce <-> flair` correspondence edges is removed
    from message passing (in **both** mirrored directions) and held out as positives.

    **What is reconstructed.**

    1. *Masked node appearance* - per-node-type MLP decoder on the final embedding. The node's own
       appearance never entered the encoder, so the only information routes are spatial neighbours and
       the cross-modal correspondence partner. No leakage, hence no re-masking trick needed.
    2. *T1ce <-> FLAIR correspondence* - relation-specific scorer on `(h_t1ce, h_flair)`. Negatives are
       **distance-matched hard negatives**: for a held-out positive `(i, j)` we sample `(i, j')` where
       `j'` is a spatial neighbour of the true partner `j`. `j'` lies in the same neighbourhood at
       comparable distance from `i`, which removes the geometric shortcut and forces the model to judge
       whether the two partitions actually *align* there.

    Heterogeneity is respected throughout: separate mask tokens, separate feature decoders per node
    type, and separate scorers per relation.
    """)
    return


@app.cell
def _(
    APPEARANCE_DIM,
    F,
    NODE_TYPES,
    REC_EDGE_MASK_RATE,
    REC_FEAT_LOSS,
    REC_FLAGS,
    REC_MASK_RATE,
    REC_NEG_PER_POS,
    REC_SCE_GAMMA,
    nn,
    torch,
):
    # ============================================================
    # Option C: masking operators + heterogeneous reconstruction heads
    # ============================================================
    CORR_ET_FT = ("t1ce", "corresponds", "flair")
    CORR_ET_TF = ("flair", "corresponds", "t1ce")


    def _rand(n, device, gen):
        return torch.rand(n, device=device, generator=gen) if gen is not None else torch.rand(n, device=device)


    def _randperm(n, device, gen):
        return torch.randperm(n, device=device, generator=gen) if gen is not None else torch.randperm(n, device=device)


    class HeteroMaskedReconstructor(nn.Module):
        """Lightweight, type-aware reconstruction heads for Option C.

        Holds (a) a learnable [MASK] token per NODE TYPE, (b) a feature decoder per
        NODE TYPE, and (c) a pair scorer per RELATION -- so the heterogeneous graph is
        not collapsed into a homogeneous one. Attaches to the embeddings the EXISTING
        QoSHRGN already returns (`x_dict`); the backbone is not modified.
        """

        def __init__(self, hidden_dim, appearance_dim, node_types, decoder_hidden=64, dropout=0.1):
            super().__init__()
            self.appearance_dim = appearance_dim
            self.mask_token = nn.ParameterDict({
                nt: nn.Parameter(torch.zeros(appearance_dim)) for nt in node_types
            })
            self.feat_decoder = nn.ModuleDict({
                nt: nn.Sequential(
                    nn.Linear(hidden_dim, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(decoder_hidden, appearance_dim),
                ) for nt in node_types
            })
            # One scorer per relation being reconstructed (relation-aware, not shared).
            self.link_decoder = nn.ModuleDict({
                rel: nn.Sequential(
                    nn.Linear(hidden_dim * 3, decoder_hidden), nn.ReLU(), nn.Dropout(dropout),
                    nn.Linear(decoder_hidden, 1),
                ) for rel in ("corresponds", "spatial__t1ce", "spatial__flair")
            })

        def decode_features(self, h, nt):
            return self.feat_decoder[nt](h)

        def score_pairs(self, h_src, h_dst, rel):
            """Symmetric-ish pair score from concat + Hadamard product."""
            if h_src.size(0) == 0:
                return h_src.new_zeros(0)
            z = torch.cat([h_src, h_dst, h_src * h_dst], dim=-1)
            return self.link_decoder[rel](z).squeeze(-1)


    def _random_neighbour_map(spatial_ei, n_nodes, device, gen):
        """For every node, ONE pseudo-random spatial neighbour (-1 if isolated).

        Scatter-with-random-permutation: last write wins, and the permutation makes
        which neighbour wins random. O(E) and needs no CSR build or distance matrix.
        """
        nbr = torch.full((n_nodes,), -1, dtype=torch.long, device=device)
        if spatial_ei.numel() == 0:
            return nbr
        s, d = spatial_ei[0].long(), spatial_ei[1].long()
        perm = _randperm(s.size(0), device, gen)
        nbr[s[perm]] = d[perm]
        return nbr


    def sample_hard_negatives(pos, dst_spatial_ei, n_dst, true_ei, neg_per_pos, device, gen,
                              exclude_self=False):
        """Distance-matched negatives for link reconstruction.

        For a positive (i, j) we return (i, j') with j' a spatial neighbour of j. j' sits
        in the same neighbourhood as the true partner, so centroid distance carries almost
        no signal and the geometric shortcut is removed. Any (i, j') that is actually a
        true edge (including a held-out one) is discarded, so no false negatives.
        """
        empty = pos.new_zeros((2, 0))
        if pos.numel() == 0 or n_dst == 0:
            return empty
        src, dst = pos[0].long(), pos[1].long()
        true_keys = true_ei[0].long() * n_dst + true_ei[1].long()
        outs = []
        for _ in range(max(1, int(neg_per_pos))):
            nbr = _random_neighbour_map(dst_spatial_ei, n_dst, device, gen)
            cand = nbr[dst]
            ok = cand >= 0
            if exclude_self:
                ok = ok & (cand != src)
            if not bool(ok.any()):
                continue
            s_ok, c_ok = src[ok], cand[ok]
            keep = ~torch.isin(s_ok * n_dst + c_ok, true_keys)
            if bool(keep.any()):
                outs.append(torch.stack([s_ok[keep], c_ok[keep]]))
        return torch.cat(outs, dim=1) if outs else empty


    def mask_hetero_graph(data, rec, mask_rate=REC_MASK_RATE, edge_mask_rate=REC_EDGE_MASK_RATE,
                          flags=None, gen=None):
        """Return (corrupted_graph, info) without ever mutating the cached graph.

        info holds the reconstruction targets: masked node indices + their original
        appearance vectors, and the held-out positive edges per relation.
        """
        flags = REC_FLAGS if flags is None else flags
        corrupt = data.clone()
        info = {"masked_idx": {}, "feat_target": {}, "pos": {}, "n_nodes": {}}

        # ---- (1) node appearance masking -------------------------------------
        for nt in NODE_TYPES:
            x = data[nt].x
            n = x.size(0)
            info["n_nodes"][nt] = n
            if not flags.get("feature", False) or mask_rate <= 0 or n == 0:
                info["masked_idx"][nt] = torch.zeros(0, dtype=torch.long, device=x.device)
                continue
            m = _rand(n, x.device, gen) < mask_rate
            idx = m.nonzero(as_tuple=True)[0]
            app, geo = x[:, :APPEARANCE_DIM], x[:, APPEARANCE_DIM:]
            mf = m.unsqueeze(1).to(app.dtype)
            token = rec.mask_token[nt].to(app.dtype).unsqueeze(0)
            # Differentiable wrt the mask token; no in-place writes into a leaf tensor.
            corrupt[nt].x = torch.cat([app * (1.0 - mf) + token * mf, geo], dim=1)
            info["masked_idx"][nt] = idx
            info["feat_target"][nt] = app[idx].detach()

        # ---- (2) correspondence edge masking ---------------------------------
        if flags.get("correspondence", False) and edge_mask_rate > 0:
            ei_ft, ea_ft = data[CORR_ET_FT].edge_index, data[CORR_ET_FT].edge_attr
            ei_tf, ea_tf = data[CORR_ET_TF].edge_index, data[CORR_ET_TF].edge_attr
            e = ei_ft.size(1)
            # build_hetero_case() mirrors the SAME pair list into both directions, and PyG
            # collates each relation in the same per-graph order, so column k refers to the
            # same underlying pair in both. Assert rather than assume.
            assert ei_tf.size(1) == e, "correspondence relations are not mirrored 1:1"
            if e > 0:
                keep = _rand(e, ei_ft.device, gen) >= edge_mask_rate
                corrupt[CORR_ET_FT].edge_index = ei_ft[:, keep]
                corrupt[CORR_ET_FT].edge_attr = ea_ft[keep]
                corrupt[CORR_ET_TF].edge_index = ei_tf[:, keep]
                corrupt[CORR_ET_TF].edge_attr = ea_tf[keep]
                info["pos"]["corresponds"] = ei_ft[:, ~keep]

        # ---- (3) optional spatial edge masking -------------------------------
        if flags.get("spatial_edge", False) and edge_mask_rate > 0:
            for nt in NODE_TYPES:
                et = (nt, "spatial", nt)
                ei, ea = data[et].edge_index, data[et].edge_attr
                e = ei.size(1)
                if e == 0:
                    continue
                keep = _rand(e, ei.device, gen) >= edge_mask_rate
                corrupt[et].edge_index = ei[:, keep]
                corrupt[et].edge_attr = ea[keep]
                info["pos"][f"spatial__{nt}"] = ei[:, ~keep]

        return corrupt, info


    def _feature_recon_loss(pred, target, kind=None, gamma=None):
        kind = REC_FEAT_LOSS if kind is None else kind
        gamma = REC_SCE_GAMMA if gamma is None else gamma
        pred, target = pred.float(), target.float()
        if kind == "mse":
            return F.mse_loss(pred, target)
        if kind == "sce":   # GraphMAE scaled cosine error
            cos = F.cosine_similarity(pred, target, dim=-1).clamp(-1.0, 1.0)
            return ((1.0 - cos).pow(gamma)).mean()
        return F.smooth_l1_loss(pred, target)   # default: robust to quantile outliers


    def reconstruction_loss(rec, h_dict, info, data, flags=None, gen=None, device=None):
        """L_reconstruction = mean over ENABLED components (so lambda_rec stays interpretable).

        Returns (total_loss, parts_dict) where parts_dict logs each component separately.
        """
        flags = REC_FLAGS if flags is None else flags
        device = h_dict[NODE_TYPES[0]].device if device is None else device
        parts, terms = {}, []

        # ---- masked node appearance ------------------------------------------
        if flags.get("feature", False):
            feat_terms = []
            for nt in NODE_TYPES:
                idx = info["masked_idx"].get(nt)
                if idx is None or idx.numel() == 0:
                    continue
                pred = rec.decode_features(h_dict[nt][idx], nt)
                l = _feature_recon_loss(pred, info["feat_target"][nt])
                parts[f"feat_{nt}"] = float(l.detach())
                feat_terms.append(l)
            if feat_terms:
                l_feat = torch.stack(feat_terms).mean()
                parts["feat"] = float(l_feat.detach())
                terms.append(l_feat)

        # ---- link reconstruction ---------------------------------------------
        link_specs = []
        if flags.get("correspondence", False):
            link_specs.append(("corresponds", "t1ce", "flair", CORR_ET_FT, False))
        if flags.get("spatial_edge", False):
            for nt in NODE_TYPES:
                link_specs.append((f"spatial__{nt}", nt, nt, (nt, "spatial", nt), True))

        for rel, src_t, dst_t, et, same_type in link_specs:
            pos = info["pos"].get(rel)
            if pos is None or pos.numel() == 0:
                continue
            n_dst = info["n_nodes"][dst_t]
            neg = sample_hard_negatives(
                pos, data[(dst_t, "spatial", dst_t)].edge_index, n_dst,
                data[et].edge_index, REC_NEG_PER_POS, device, gen, exclude_self=same_type,
            )
            if neg.numel() == 0:
                continue
            h_src, h_dst = h_dict[src_t], h_dict[dst_t]
            s_pos = rec.score_pairs(h_src[pos[0].long()], h_dst[pos[1].long()], rel)
            s_neg = rec.score_pairs(h_src[neg[0].long()], h_dst[neg[1].long()], rel)
            scores = torch.cat([s_pos, s_neg]).float()
            labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)]).float()
            l = F.binary_cross_entropy_with_logits(scores, labels)
            with torch.no_grad():
                acc = ((scores > 0).float() == labels).float().mean()
            parts[rel] = float(l.detach())
            parts[f"{rel}_acc"] = float(acc)
            terms.append(l)

        if not terms:
            return torch.zeros((), device=device), parts
        total = torch.stack(terms).mean()
        parts["total"] = float(total.detach())
        return total, parts

    return (
        CORR_ET_FT,
        HeteroMaskedReconstructor,
        mask_hetero_graph,
        sample_hard_negatives,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9b. Option B: structure-aware graph refinement (research ablation)

    After the base model's HGT + adaptive propagation produce initial node predictions, this optional module analyses the STRUCTURE of that predicted tumour graph (neighbour-class fractions, hop-distance to nearest ET/BG, tumour connected-component size, enclosure score, cross-modal fan-out) and uses it to refine the final classification. Controlled by `USE_STRUCTURAL_REFINEMENT` (True in this bundle); the baseline model/training/eval below is completely unaffected when disabled.
    """)
    return


@app.cell
def _(
    NODE_TYPES,
    STRUCTURAL_DROPOUT,
    STRUCTURAL_FLAGS,
    STRUCTURAL_HIDDEN,
    nn,
    np,
    structural_descriptor_dim,
    torch,
):
    # ============================================================
    # Option B: structural descriptor computation (graph algorithms)
    # ------------------------------------------------------------
    # Descriptors are computed on the SPATIAL graph of a single node type
    # (t1ce-spatial-t1ce or flair-spatial-flair) using a TEMPORARY discrete
    # predicted class per node (mixed with GT via scheduled sampling during
    # training only -- see StructuralQoSHRGN.forward). These are graph-algorithm
    # operations (BFS, connected components) on a discrete label graph; they run
    # on detached tensors via scipy sparse routines and are NOT part of the
    # autograd graph. Gradients still reach the refinement module through the
    # continuous embeddings/probabilities (h, p) that accompany the descriptors.
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path, connected_components
    _STRUCT_DIST_SENTINEL = 16.0
      # finite "far / not present in this graph" hop-distance
    def edge_index_to_csr(edge_index, num_nodes):
        if edge_index.numel() == 0:
            return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
        ei = edge_index.detach().cpu().numpy()
        data = np.ones(ei.shape[1], dtype=np.float32)
        return sp.csr_matrix((data, (ei[0], ei[1])), shape=(num_nodes, num_nodes))

    def _multi_source_hop_distance(csr, seed_mask_np, sentinel=_STRUCT_DIST_SENTINEL):
        """Hop-distance from the nearest True node in seed_mask_np to every node,
        via a virtual super-source connected to all seeds (standard multi-source
        BFS trick): ONE sparse shortest-path call regardless of how many seed
        nodes there are -- no all-pairs shortest paths, no dense NxN matrix.
        Nodes with no reachable seed (e.g. no ET anywhere in this patient's
        graph) get the finite `sentinel` value, never inf/NaN."""
        n = csr.shape[0]
        seed_idx = np.nonzero(seed_mask_np)[0]
        if seed_idx.size == 0:
            return np.full(n, sentinel, dtype=np.float32)
        src_row = sp.csr_matrix((np.ones(seed_idx.size, dtype=np.float32), (np.zeros(seed_idx.size, dtype=np.int64), seed_idx)), shape=(1, n))
        aug = sp.vstack([sp.hstack([csr, src_row.T]), sp.hstack([src_row, sp.csr_matrix((1, 1), dtype=np.float32)])]).tocsr()
        d = shortest_path(aug, method='D', directed=False, unweighted=True, indices=n)
        d = d[:n] - 1.0
        d[~np.isfinite(d)] = sentinel
        d = np.clip(d, 0.0, sentinel)
        return d.astype(np.float32)

    def _tumour_component_sizes(csr, tumour_mask_np):
        """Size of the connected component (within the SPATIAL graph, restricted
        to predicted-tumour nodes) that each tumour node belongs to. Non-tumour  # subtract the virtual source -> seed hop
        nodes get 0. Uses the spatial graph only, per spec."""
        n = csr.shape[0]
        idx = np.nonzero(tumour_mask_np)[0]
        sizes = np.zeros(n, dtype=np.float32)
        if idx.size == 0:
            return sizes
        sub = csr[idx][:, idx]
        n_comp, labels = connected_components(sub, directed=False)
        comp_sizes = np.bincount(labels, minlength=n_comp)
        sizes[idx] = comp_sizes[labels]
        return sizes

    def compute_structural_descriptors(edge_index, num_nodes, pred_class, cross_src_index, flags, device):
        """Build the [num_nodes, D] structural descriptor tensor for one node type.

        pred_class      : LongTensor [num_nodes], the TEMPORARY discrete predicted
                           class defining "the tumour graph" for this call (may be
                           a scheduled-sampling mix of GT and the model's own
                           prediction during training -- see the wrapper).
        cross_src_index : data[nt, 'corresponds', other_nt].edge_index, i.e. the
                           correspondence edges with THIS node type as source --
                           used only for the cross-modal fan-out descriptor.
        """
        pred_np = pred_class.detach().cpu().numpy()
        tumour_mask = pred_np != 0
        et_mask = pred_np == 3
        bg_mask = pred_np == 0
        csr = edge_index_to_csr(edge_index, num_nodes)
        deg = np.asarray(csr.sum(axis=1)).reshape(-1)
        parts = []
        et_nbr = None
        if flags['neighbor_stats']:
            tumour_nbr = np.asarray(csr @ tumour_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            et_nbr = np.asarray(csr @ et_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            parts = parts + [tumour_nbr, et_nbr]
        if flags['distance']:
            dist_et = _multi_source_hop_distance(csr, et_mask) / _STRUCT_DIST_SENTINEL
            dist_bg = _multi_source_hop_distance(csr, bg_mask) / _STRUCT_DIST_SENTINEL
            parts = parts + [dist_et, dist_bg]
        if flags['component_size']:
            comp = _tumour_component_sizes(csr, tumour_mask)
            parts = parts + [np.log1p(comp) / np.log1p(max(num_nodes, 2))]
        if flags['enclosure']:
            et_nbr_e = et_nbr if et_nbr is not None else np.asarray(csr @ et_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            bg_nbr_e = np.asarray(csr @ bg_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            parts = parts + [et_nbr_e - bg_nbr_e]
        if flags['cross_modal']:
            fanout = np.bincount(cross_src_index[0].detach().cpu().numpy(), minlength=num_nodes).astype(np.float32)
            fanout = fanout[:num_nodes]
            fanout = fanout / max(float(fanout.mean()), 1e-06)
            parts = parts + [fanout]
        if not parts:  # normalised, roughly in [0, 1]
            return torch.zeros(num_nodes, 0, device=device)
        s = np.stack(parts, axis=1).astype(np.float32)
        return torch.from_numpy(s).to(device)  # Local enclosure/boundary score, EXPLICITLY defined as:
      #   ET-neighbour fraction minus BG-neighbour fraction, in [-1, 1].
    class StructuralRefinementHead(nn.Module):  #   +1 => every spatial neighbour is predicted ET  (deep interior of an ET region)
        """Per-node-type refinement head for Option B.  #   -1 => every spatial neighbour is predicted BG  (isolated / far from any tumour)
      #    0 => mixed / boundary neighbourhood, or ambiguous
        Combines the base model's node embedding (h), its initial class  # This is a MEASURED graph property, not an assumption that ET always
        probabilities (p), and the structural descriptor vector (s) computed  # encloses anything -- the refinement module decides whether/how to use it.
        from the temporary predicted tumour graph. Outputs a RESIDUAL correction
        to the initial logits -- the final linear layer is zero-initialised, so
        at the start of training final_logits == initial_logits exactly, and the
        module only learns to deviate where structural evidence helps.
        """

        def __init__(self, hidden_dim, num_classes, structural_dim, refine_hidden=32, dropout=0.1):
            super().__init__()
            self.structural_dim = structural_dim
            if structural_dim > 0:
                self.structural_mlp = nn.Sequential(nn.Linear(structural_dim, refine_hidden), nn.ReLU(), nn.Linear(refine_hidden, refine_hidden), nn.ReLU())
                combine_in = hidden_dim + num_classes + refine_hidden
            else:
                self.structural_mlp = None
                combine_in = hidden_dim + num_classes
            self.combine = nn.Sequential(nn.Linear(combine_in, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout))
            self.out = nn.Linear(hidden_dim // 2, num_classes)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

        def forward(self, h, p, s, initial_logits):
            feats = [h, p]
            if self.structural_mlp is not None and s.size(1) > 0:
                feats.append(self.structural_mlp(s))
            z = self.combine(torch.cat(feats, dim=-1))
            delta = self.out(z)
            return initial_logits + delta

    class StructuralQoSHRGN(nn.Module):
        """QoS-HRGN + Option B: structure-aware graph refinement.

        Wraps an EXISTING, UNCHANGED QoSHRGN instance (`base_model`) -- does not
        redesign the backbone. Reuses its HGT + adaptive propagation output
        (embeddings `x_dict` and initial per-node logits) and adds a second,
        lightweight, per-modality refinement stage conditioned on the structure
        of the model's own current tumour-graph prediction.

        This is a hypothesis-driven ablation: we do NOT assume enclosure, or any
        other structural pattern, is a universal biological rule. Whether these
        descriptors help is exactly what the STRUCTURAL_USE_* flags let you
        measure (see the diagnostics cell after evaluation).
        """

        def __init__(self, base_model, hidden_dim, num_classes, flags=STRUCTURAL_FLAGS, refine_hidden=STRUCTURAL_HIDDEN, dropout=STRUCTURAL_DROPOUT):
            super().__init__()
            self.base_model = base_model
            self.flags = flags
            self.structural_dim = structural_descriptor_dim(flags)
            self.refine = nn.ModuleDict({nt: StructuralRefinementHead(hidden_dim, num_classes, self.structural_dim, refine_hidden, dropout) for nt in NODE_TYPES})

        def forward(self, data, teacher_prob=0.0):
            initial_logits, x_dict = self.base_model(data)
            final_logits = {}
            for nt in NODE_TYPES:
                other_nt = [t for t in NODE_TYPES if t != nt][0]
                edge_index = data[nt, 'spatial', nt].edge_index
                num_nodes = data[nt].x.size(0)
                dev = data[nt].x.device
                init_prob = torch.softmax(initial_logits[nt].float(), dim=-1)
                pred_class = init_prob.detach().argmax(dim=-1)
                if self.training and teacher_prob > 0:
                    gt_class = data[nt].y
                    use_gt = torch.rand(num_nodes, device=dev) < teacher_prob
                    struct_class = torch.where(use_gt, gt_class, pred_class)
                else:
                    struct_class = pred_class
                cross_src_index = data[nt, 'corresponds', other_nt].edge_index
                s = compute_structural_descriptors(edge_index, num_nodes, struct_class, cross_src_index, self.flags, dev)
                h = x_dict[nt]
                final_logits[nt] = self.refine[nt](h, init_prob, s, initial_logits[nt])
            return (initial_logits, final_logits, x_dict)  # Scheduled sampling: mix GT-derived and model-predicted  # structure during TRAINING ONLY. Never at eval/test.  # eval/test: predictions only -- no GT leakage.

    return StructuralQoSHRGN, connected_components, edge_index_to_csr


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Pathology-aware loss on fractional labels

    - **Soft cross-entropy**: `-Σ_c w_c · y_frac_c · log p_c`, so a node that is 60 % ED / 40 % ET is not forced to pick one.
    - **Volume-weighted region Dice** for WT / TC / ET computed *per patient*. Because every voxel in a supervoxel receives the node's prediction, `Σ_nodes vol·p·g` is exactly the voxel-level soft Dice — the loss now optimises the metric BraTS reports.

    `ET = class 3`, `TC = NCR/NET ∪ ET`, `WT = NCR/NET ∪ ED ∪ ET`, as in the BraTS definition.
    """)
    return


@app.cell
def _(
    DICE_WEIGHT,
    ET_FOCAL_WEIGHT,
    F,
    FOCAL_GAMMA,
    NODE_TYPES,
    NUM_CLASSES,
    device,
    torch,
):
    def soft_cross_entropy(logits, y_frac, class_weights=None):
        logp = F.log_softmax(logits.float(), dim=-1)
        w = class_weights.view(1, -1).float() if class_weights is not None else 1.0
        return -(w * y_frac.float() * logp).sum(dim=-1).mean()

    def _region_pairs(probs, y_frac):
        return (
            (1.0 - probs[:, 0], 1.0 - y_frac[:, 0]),                      # WT
            (probs[:, 1] + probs[:, 3], y_frac[:, 1] + y_frac[:, 3]),      # TC
            (probs[:, 3], y_frac[:, 3]),                                   # ET
        )

    def region_dice_loss(probs, y_frac, vol, batch, num_graphs, eps=1e-7):
        """Per-patient, volume-weighted soft Dice averaged over WT/TC/ET (== voxel soft Dice)."""
        probs, y_frac, vol = probs.float(), y_frac.float(), vol.float()
        loss = 0.0
        for p, g in _region_pairs(probs, y_frac):
            inter = torch.zeros(num_graphs, device=p.device).index_add_(0, batch, vol * p * g)
            den = torch.zeros(num_graphs, device=p.device).index_add_(0, batch, vol * (p + g))
            loss = loss + (1.0 - (2.0 * inter + eps) / (den + eps)).mean()
        return loss / 3.0

    def focal_loss_soft(logits, y_frac, gamma=2.0):
        """Soft focal loss for fractional labels. Emphasises hard (low-confidence) examples."""
        logp = F.log_softmax(logits.float(), dim=-1)
        p = torch.exp(logp)
        return -(y_frac.float() * (1 - p) ** gamma * logp).sum(dim=-1).mean()

    def hierarchy_loss(logits, data_nt, num_graphs, class_weights=None, dice_weight=DICE_WEIGHT):
        ce = soft_cross_entropy(logits, data_nt.y_frac, class_weights)
        probs = torch.softmax(logits.float(), dim=-1)
        batch = data_nt.batch if hasattr(data_nt, "batch") and data_nt.batch is not None else torch.zeros(
            probs.size(0), dtype=torch.long, device=probs.device)
        d = region_dice_loss(probs, data_nt.y_frac, data_nt.vol, batch, num_graphs)
        loss = ce + dice_weight * d
        if ET_FOCAL_WEIGHT > 0:
            loss = loss + ET_FOCAL_WEIGHT * focal_loss_soft(logits, data_nt.y_frac, gamma=FOCAL_GAMMA)
        return loss

    @torch.no_grad()
    def node_region_dice(logits_dict, data, eps=1e-7):
        """Per-patient volume-weighted hard Dice (mean of WT/TC/ET), pooled over both node types.
        This is a training diagnostic; selection uses fused full-volume validation Dice."""
        num_graphs = data.num_graphs if hasattr(data, "num_graphs") else 1
        inter = torch.zeros(3, num_graphs, device=device)
        den = torch.zeros(3, num_graphs, device=device)
        for nt in NODE_TYPES:
            pred = F.one_hot(logits_dict[nt].argmax(dim=-1), NUM_CLASSES).float()
            yf, vol = data[nt].y_frac.float(), data[nt].vol.float()
            batch = data[nt].batch if hasattr(data[nt], "batch") and data[nt].batch is not None else torch.zeros(
                vol.size(0), dtype=torch.long, device=vol.device)
            for r, (p, g) in enumerate(_region_pairs(pred, yf)):
                inter[r].index_add_(0, batch, vol * p * g)
                den[r].index_add_(0, batch, vol * (p + g))
        dice = (2.0 * inter + eps) / (den + eps)             # [3, num_graphs]
        return dice.mean(dim=0)                               # per patient

    return (
        focal_loss_soft,
        hierarchy_loss,
        region_dice_loss,
        soft_cross_entropy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10. Build the dataset graphs

    Graphs are built in parallel and cached. Each worker saves its own files (`*.graph.pt` ≈ a few MB, `*.meta.pt` ≈ 50 MB) so nothing large is shipped back to the main process. Only the graphs are kept in memory; metadata is loaded lazily during evaluation.
    """)
    return


@app.cell
def _(
    BUILD_BUDGET_HOURS,
    COMPACTNESS,
    DATASET_AUDIT,
    DATASET_FINGERPRINT,
    GRAPH_CACHE_PATH,
    GRAPH_CACHE_RECEIPT_PATH,
    GRAPH_CACHE_ZIP_PATH,
    GRAPH_VERSION,
    HF_CACHE_MANIFEST_NAME,
    HF_CACHE_SUBDIR,
    HF_CACHE_ZIP_NAME,
    HF_ENABLED,
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_REQUIRED,
    HF_TOKEN,
    IDENTITY_CONFIG,
    N_SEGMENTS,
    PART2_START_TIME,
    Parallel,
    SLIC_ITERS,
    STAGE1_HANDOFF_MANIFEST,
    build_hetero_case,
    config_hash,
    delayed,
    extract_verified_zip,
    hf_api,
    hf_hub_download,
    hf_try_download,
    hf_upload_file_verified,
    hf_upload_receipt,
    json,
    np,
    os,
    time,
    torch,
    valid_cases,
    verify_remote,
    write_json_atomic,
):
    GRAPH_PULL_RECEIPT = STAGE1_HANDOFF_MANIFEST.get("graph_cache")
    assert not DATASET_AUDIT["errors"]
    import zipfile

    print(f"[Cache] N_SEGMENTS={N_SEGMENTS}  COMPACTNESS={COMPACTNESS}  version={GRAPH_VERSION}")
    print(f"[Cache] {GRAPH_CACHE_PATH}")

    GRAPH_BUILD_STATS_PATH = os.path.join(GRAPH_CACHE_PATH, "build_stats.json")


    def get_case_id(files):
        return os.path.basename(os.path.dirname(files["t1"]))


    def cache_paths(case_id):
        return (os.path.join(GRAPH_CACHE_PATH, f"{case_id}.graph.pt"),
                os.path.join(GRAPH_CACHE_PATH, f"{case_id}.meta.pt"))


    def build_and_save(files, n_segments, compactness, max_iter, gpath, mpath):
        _data, _meta = build_hetero_case(files, n_segments=n_segments,
                                          compactness=compactness, max_iter=max_iter)
        torch.save((_data.cpu(), {"n_segments": n_segments, "compactness": compactness,
                                  "version": GRAPH_VERSION}), gpath + ".tmp")
        torch.save(_meta, mpath + ".tmp")
        os.replace(gpath + ".tmp", gpath)
        os.replace(mpath + ".tmp", mpath)
        return gpath


    def missing_cases():
        return [(_files, _g, _m) for _, _files in valid_cases
                for _g, _m in [cache_paths(get_case_id(_files))]
                if not (os.path.exists(_g) and os.path.exists(_m))]


    def hf_pull_graph_cache_zip():
        if not missing_cases():
            return
        if not HF_ENABLED:
            if HF_REQUIRED:
                raise RuntimeError("Hugging Face is required for graph-cache transfer")
            return
        _receipt = GRAPH_PULL_RECEIPT
        if _receipt is None:
            _manifest_path = hf_try_download(HF_CACHE_MANIFEST_NAME, HF_REPO_ID, HF_REPO_TYPE)
            if _manifest_path is None:
                return  # Fresh Part 1 has no remote cache yet.
            with open(_manifest_path) as _fh:
                _receipt = json.load(_fh)
        if (_receipt.get("dataset_fingerprint") != DATASET_FINGERPRINT
                or _receipt.get("graph_settings_hash") != config_hash(IDENTITY_CONFIG["graph"])
                or _receipt.get("cache_subdir") != HF_CACHE_SUBDIR):
            raise RuntimeError("Graph archive manifest identity mismatch")
        _zip_path = hf_try_download(_receipt["remote_name"], _receipt["repo_id"],
                                    _receipt["repo_type"], revision=_receipt["revision"])
        if _zip_path is None:
            raise RuntimeError("Manifest references a missing graph archive")
        _count = extract_verified_zip(_zip_path, GRAPH_CACHE_PATH, HF_CACHE_SUBDIR, _receipt["sha256"])
        write_json_atomic(GRAPH_CACHE_RECEIPT_PATH, _receipt)
        print(f"[HF] Verified and extracted {_count} graph-cache files")

    def hf_push_graph_cache_zip():
        if not HF_ENABLED:
            if HF_REQUIRED:
                raise RuntimeError("Hugging Face is required for graph-cache upload")
            return None
        _files = []
        for _root, _dirs, _names in os.walk(GRAPH_CACHE_PATH):
            for _name in sorted(_names):
                if _name.endswith((".graph.pt", ".meta.pt")) or _name == "build_stats.json":
                    _full = os.path.join(_root, _name)
                    _files.append((_full, os.path.relpath(_full, GRAPH_CACHE_PATH)))
        if not _files:
            raise RuntimeError("No graph-cache files to upload")
        _tmp = GRAPH_CACHE_ZIP_PATH + ".tmp"
        _total_bytes = sum(os.path.getsize(_full) for _full, _rel in _files)
        _written_bytes = 0
        _last_report = time.time()
        with zipfile.ZipFile(_tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as _zf:
            for _index, (_full, _rel) in enumerate(_files, 1):
                _zf.write(_full, arcname=f"{HF_CACHE_SUBDIR}/{_rel.replace(os.sep, '/')}")
                _written_bytes += os.path.getsize(_full)
                if time.time() - _last_report >= 5 or _index == len(_files):
                    print(f"[HF] ZIP {_index}/{len(_files)} files: {_written_bytes/1024**3:.2f}/{_total_bytes/1024**3:.2f} GB", flush=True)
                    _last_report = time.time()
        os.replace(_tmp, GRAPH_CACHE_ZIP_PATH)
        hf_upload_file_verified(GRAPH_CACHE_ZIP_PATH, HF_CACHE_ZIP_NAME, HF_REPO_ID, HF_REPO_TYPE,
                                f"verified graph cache {HF_CACHE_SUBDIR}")
        _receipt = hf_upload_receipt(HF_REPO_ID, HF_CACHE_ZIP_NAME)
        _receipt.update(dataset_fingerprint=DATASET_FINGERPRINT,
                        graph_settings_hash=config_hash(IDENTITY_CONFIG["graph"]),
                        cache_subdir=HF_CACHE_SUBDIR, complete=not missing_cases())
        write_json_atomic(GRAPH_CACHE_RECEIPT_PATH, _receipt)
        hf_upload_file_verified(GRAPH_CACHE_RECEIPT_PATH, HF_CACHE_MANIFEST_NAME, HF_REPO_ID, HF_REPO_TYPE,
                                "graph cache checksum and pinned revision")
        os.remove(GRAPH_CACHE_ZIP_PATH)
        return _receipt


    def ensure_graph_cache_handoff():
        if not HF_ENABLED:
            if HF_REQUIRED:
                raise RuntimeError("Cannot complete Part1 without the required HF graph archive")
            return None
        if missing_cases():
            raise RuntimeError("Cannot hand off an incomplete graph cache")
        if os.path.exists(GRAPH_CACHE_RECEIPT_PATH):
            with open(GRAPH_CACHE_RECEIPT_PATH) as _fh:
                _receipt = json.load(_fh)
            if (_receipt.get("complete") and _receipt.get("dataset_fingerprint") == DATASET_FINGERPRINT
                    and _receipt.get("graph_settings_hash") == config_hash(IDENTITY_CONFIG["graph"])):
                verify_remote(hf_api, _receipt["repo_id"], _receipt["repo_type"], _receipt["remote_name"],
                              _receipt["revision"], _receipt["sha256"], _receipt["size_bytes"],
                              lambda **kw: hf_hub_download(token=HF_TOKEN, **kw))
                return _receipt
        return hf_push_graph_cache_zip()

    hf_pull_graph_cache_zip()
    if missing_cases():
        raise RuntimeError("Part2 requires the complete verified Part1 graph ZIP; cache rebuild is disabled")
    _todo = missing_cases()
    _built_any = False
    if _todo:
        _cpu_budget = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        _n_jobs = min(_cpu_budget, len(_todo))
        _chunk_size = max(1, _n_jobs * 4)
        _last_chunk_hours = 0.0
        for _chunk_start in range(0, len(_todo), _chunk_size):
            _elapsed_h = (time.time() - PART2_START_TIME) / 3600
            if _elapsed_h + _last_chunk_hours * 1.1 > BUILD_BUDGET_HOURS:
                print(f"build budget reached; {len(_todo) - _chunk_start} cases remain")
                break
            _chunk = _todo[_chunk_start:_chunk_start + _chunk_size]
            _chunk_t0 = time.time()
            Parallel(n_jobs=_n_jobs, backend="loky", verbose=0)(
                delayed(build_and_save)(f, N_SEGMENTS, COMPACTNESS, SLIC_ITERS, g, m)
                for f, g, m in _chunk)
            _last_chunk_hours = (time.time() - _chunk_t0) / 3600
            _stats = []
            if os.path.exists(GRAPH_BUILD_STATS_PATH):
                with open(GRAPH_BUILD_STATS_PATH) as _stats_fh:
                    _stats = json.load(_stats_fh)
            _stats.append({"cases": len(_chunk), "seconds": _last_chunk_hours * 3600.0, "n_jobs": _n_jobs})
            with open(GRAPH_BUILD_STATS_PATH + ".tmp", "w") as _stats_fh:
                json.dump(_stats, _stats_fh, indent=2)
            os.replace(GRAPH_BUILD_STATS_PATH + ".tmp", GRAPH_BUILD_STATS_PATH)
            _built_any = True

    graph_build_complete = not missing_cases()
    if _built_any and graph_build_complete:
        hf_push_graph_cache_zip()
    if not graph_build_complete:
        print(f"re-run Part 2 to finish the remaining {len(missing_cases())} graphs; nothing pushed")

    graph_items = []
    for _, _files in sorted(valid_cases, key=lambda case: get_case_id(case[1])):
        _gpath, _mpath = cache_paths(get_case_id(_files))
        if not (os.path.exists(_gpath) and os.path.exists(_mpath)):
            continue
        _data, _cache_meta = torch.load(_gpath, weights_only=False, map_location="cpu")
        assert _cache_meta.get("version") == GRAPH_VERSION and _cache_meta.get("n_segments") == N_SEGMENTS, (
            f"CACHE MISMATCH in {_gpath}: {_cache_meta}")
        graph_items.append((_data, _mpath))
    graph_items_by_case = {os.path.basename(_mpath).replace(".meta.pt", ""): (_data, _mpath)
                           for _data, _mpath in graph_items}
    print("Usable graphs:", len(graph_items))
    if graph_items:
        _nodes = [_data["t1ce"].x.size(0) + _data["flair"].x.size(0) for _data, _ in graph_items]
        print(f"Nodes per graph (both types): mean={np.mean(_nodes):.0f}  min={np.min(_nodes)}  max={np.max(_nodes)}")
    if not graph_build_complete:
        raise RuntimeError("Graph cache is incomplete; refusing to train on a non-reproducible partial set.")
    return (
        GRAPH_BUILD_STATS_PATH,
        build_and_save,
        get_case_id,
        graph_items,
        graph_items_by_case,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Patient-level train / validation / test split and class weights

    Split by **patient**. Class weights come from *voxel* fractions (`Σ vol · y_frac`) rather than node counts, so the priority-labelling distortion of v1 cannot happen. Square-root inverse frequency keeps them moderate; the Dice term handles the rest of the imbalance.
    """)
    return


@app.cell
def _(device, graph_items_by_case, hashlib, json, stage1_checkpoint):
    SPLIT_CASE_IDS = stage1_checkpoint["split"]
    _missing = [c for _key in SPLIT_CASE_IDS for c in SPLIT_CASE_IDS[_key] if c not in graph_items_by_case]
    assert not _missing, f"Graphs missing from persisted Part-1 split: {_missing[:10]}"
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["val"]))
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["test"]))
    assert not (set(SPLIT_CASE_IDS["val"]) & set(SPLIT_CASE_IDS["test"]))
    assert len(set().union(*[set(v) for v in SPLIT_CASE_IDS.values()])) == sum(len(v) for v in SPLIT_CASE_IDS.values())
    SPLIT_FINGERPRINT = hashlib.sha256(json.dumps(SPLIT_CASE_IDS, sort_keys=True).encode()).hexdigest()
    if stage1_checkpoint.get("split_fingerprint") != SPLIT_FINGERPRINT:
        raise RuntimeError("Stage-1 split fingerprint mismatch; refusing to evaluate mixed subjects")
    print("Split fingerprint:", SPLIT_FINGERPRINT)
    train_items = [graph_items_by_case[c] for c in SPLIT_CASE_IDS["train"]]
    val_items = [graph_items_by_case[c] for c in SPLIT_CASE_IDS["val"]]
    test_items = [graph_items_by_case[c] for c in SPLIT_CASE_IDS["test"]]
    class_weights = stage1_checkpoint["class_weights"]
    class_weights_dev = class_weights.to(device)
    print("Split restored from Stage-1 checkpoint: train:", len(train_items), "| Val:", len(val_items), "| Test:", len(test_items))
    return (
        SPLIT_CASE_IDS,
        SPLIT_FINGERPRINT,
        class_weights,
        test_items,
        train_items,
        val_items,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12. Load the Stage-1 graph model (trained in Part 1)

    Part 1 already trained and validated the graph model (QoS-HRGN + Option B/C
    modules) and pushed a checkpoint to the Hub. This session reconstructs the
    *same* architecture from the config cells above, then loads those weights --
    it does not retrain the graph model.
    """)
    return


@app.cell
def _(
    APPEARANCE_DIM,
    DROPOUT,
    HEADS,
    HGT_LAYERS,
    HIDDEN_DIM,
    HOPS,
    HOP_TEMPERATURE,
    HeteroMaskedReconstructor,
    K_MAX,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NUM_CLASSES,
    QoSHRGN,
    REC_DECODER_HIDDEN,
    STRUCTURAL_DROPOUT,
    STRUCTURAL_FLAGS,
    STRUCTURAL_HIDDEN,
    StructuralQoSHRGN,
    USE_ADAPTIVE_HOPS,
    USE_MASKED_RECONSTRUCTION,
    USE_STRUCTURAL_REFINEMENT,
    device,
):
    untrained_base_model = QoSHRGN(
        in_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, heads=HEADS, num_classes=NUM_CLASSES,
        hops=HOPS, hgt_layers=HGT_LAYERS, dropout=DROPOUT,
        adaptive_hops=USE_ADAPTIVE_HOPS, k_max=K_MAX, hop_temperature=HOP_TEMPERATURE,
    ).to(device)
    if USE_STRUCTURAL_REFINEMENT:
        untrained_model = StructuralQoSHRGN(untrained_base_model, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                                   flags=STRUCTURAL_FLAGS, refine_hidden=STRUCTURAL_HIDDEN,
                                   dropout=STRUCTURAL_DROPOUT).to(device)
    else:
        untrained_model = untrained_base_model

    rec_module = None
    if USE_MASKED_RECONSTRUCTION:
        rec_module = HeteroMaskedReconstructor(
            hidden_dim=HIDDEN_DIM, appearance_dim=APPEARANCE_DIM, node_types=NODE_TYPES,
            decoder_hidden=REC_DECODER_HIDDEN, dropout=DROPOUT,
        ).to(device)

    print(f"Architecture reconstructed (untrained). Parameters: "
          f"{sum(p.numel() for p in untrained_model.parameters()) / 1e6:.2f} M"
          + (f" + rec heads {sum(p.numel() for p in rec_module.parameters()) / 1e3:.1f} k"
             if rec_module is not None else ""))
    def forward_model(untrained_model, data, teacher_prob=0.0):
        """Uniform call across baseline and Option-B modes.
        Returns (logits_for_main_loss, aux_initial_logits_or_None, x_dict)."""
        if USE_STRUCTURAL_REFINEMENT:
            initial_logits, final_logits, x_dict = untrained_model(data, teacher_prob=teacher_prob)
            return final_logits, initial_logits, x_dict
        else:
            final_logits, x_dict = untrained_model(data)
            return final_logits, None, x_dict

    return forward_model, rec_module, untrained_base_model, untrained_model


@app.cell
def _(
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    HF_REQUIRED,
    PERSISTENT_BASE,
    STAGE1_CKPT_PATH,
    STAGE1_MANIFEST_NAME,
    STAGE1_REMOTE_NAME,
    device,
    hf_try_download,
    json,
    os,
    rec_module,
    sha256_file,
    torch,
    untrained_base_model,
    untrained_model,
):
    model = untrained_model
    base_model = untrained_base_model
    _local_manifest = os.path.join(PERSISTENT_BASE, STAGE1_MANIFEST_NAME)
    if HF_ENABLED:
        _manifest_path = hf_try_download(STAGE1_MANIFEST_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE)
    else:
        _manifest_path = _local_manifest if os.path.exists(_local_manifest) else None
    if _manifest_path is None:
        raise RuntimeError("Part1 handoff manifest is missing; complete Part1 upload first")
    with open(_manifest_path) as _fh:
        _manifest = json.load(_fh)
    _ckpt_path = STAGE1_CKPT_PATH if os.path.exists(STAGE1_CKPT_PATH) and sha256_file(STAGE1_CKPT_PATH) == _manifest["sha256"] else None
    if _ckpt_path is None and HF_ENABLED:
        if not _manifest.get("checkpoint_revision"):
            raise RuntimeError("Re-run updated Part1 handoff cell to record the checkpoint revision; no graph retraining needed")
        _ckpt_path = hf_try_download(STAGE1_REMOTE_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                    revision=_manifest["checkpoint_revision"])
    if _ckpt_path is None or sha256_file(_ckpt_path) != _manifest["sha256"]:
        raise RuntimeError("Stage1 checkpoint missing or checksum mismatch")
    if HF_REQUIRED and not (_manifest.get("graph_cache") or {}).get("complete"):
        raise RuntimeError("Complete HF graph ZIP receipt missing; re-run updated Part1 handoff cell")
    stage1_checkpoint = torch.load(_ckpt_path, weights_only=False, map_location="cpu")
    for _key in ("config_hash", "split", "training_complete"):
        if _key not in stage1_checkpoint:
            raise RuntimeError("Legacy Stage1 checkpoint format; fresh Part1 training required")
    if stage1_checkpoint["config_hash"] != CONFIG_HASH:
        raise RuntimeError("Stage1 architecture/config mismatch")
    if stage1_checkpoint.get("dataset_fingerprint") != DATASET_FINGERPRINT:
        raise RuntimeError("Stage1 source dataset mismatch")
    if not stage1_checkpoint["training_complete"]:
        raise RuntimeError("Part1 graph training did not finish; resume Part1")
    model.load_state_dict(stage1_checkpoint["model_state_dict"])
    model.to(device).eval()
    if rec_module is not None and stage1_checkpoint.get("rec_state_dict") is not None:
        rec_module.load_state_dict(stage1_checkpoint["rec_state_dict"])
        rec_module.to(device).eval()
    best_val_dice, best_epoch = stage1_checkpoint["best_val_dice"], stage1_checkpoint["best_epoch"]
    val_metrics, test_metrics = stage1_checkpoint["val_metrics"], stage1_checkpoint["test_metrics"]
    STAGE1_MODEL_SHA256 = sha256_file(_ckpt_path)
    STAGE1_HANDOFF_MANIFEST = _manifest
    print(f"Verified Stage1 checkpoint: best validation Dice={best_val_dice:.4f}, epoch={best_epoch}")
    return (
        STAGE1_HANDOFF_MANIFEST,
        STAGE1_MODEL_SHA256,
        base_model,
        best_epoch,
        best_val_dice,
        model,
        stage1_checkpoint,
        test_metrics,
        val_metrics,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 12b. Voxel refinement head training (Stage 2)

    Two-stage training: the graph model (QoS-HRGN) is already trained and frozen.
    Only the voxel refinement head is trained here. It learns to correct the graph
    model's supervoxel-constant predictions at voxel resolution, using the original
    MRI intensity to split mixed supervoxels (especially ET vs NCR/NET).

    Model selection is on the mean of WT/TC/ET validation Dice.
    """)
    return


@app.cell
def _(
    CKPT_PUSH_EVERY_EPOCHS,
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    EARLY_STOP_PATIENCE,
    GPU_AUTOTUNE,
    GPU_MEMORY_FRACTION,
    GPU_PRECISION,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    INFERENCE_MAX_BATCH,
    INFERENCE_OVERLAP,
    MODALITIES,
    NUM_CLASSES,
    PART2_START_TIME,
    PERSISTENT_BASE,
    PROTOCOL_VERSION,
    RESUME_STAGE2,
    RUN_CONFIG,
    SEED,
    SPLIT_CASE_IDS,
    SPLIT_FINGERPRINT,
    STAGE1_MODEL_SHA256,
    STAGE2_LATEST_NAME,
    STAGE2_LATEST_PATH,
    TRAIN_BUDGET_HOURS,
    USE_AMP,
    USE_VOXEL_REFINEMENT,
    VOXEL_BASE_CHANNELS,
    VOXEL_BOUNDARY_WEIGHT,
    VOXEL_CE_WEIGHT,
    VOXEL_DICE_WEIGHT,
    VOXEL_EFFECTIVE_BATCH_SIZE,
    VOXEL_EPOCHS,
    VOXEL_FOCAL_GAMMA,
    VOXEL_FOCAL_WEIGHT,
    VOXEL_LR,
    VOXEL_MAX_MICROBATCH,
    VOXEL_MAX_SIZE,
    VOXEL_PREFETCH_WORKERS,
    VoxelRefinementHead,
    WEIGHT_DECAY,
    amp_context,
    bounded_prefetch,
    class_weights,
    configure_gpu,
    device,
    fmt_metrics,
    hashlib,
    hf_try_download,
    hf_upload_file_verified,
    json,
    load_meta,
    make_training_patch,
    mean_region_dice,
    mo,
    model,
    np,
    os,
    patient_groups,
    postprocess_prediction_auto,
    predict_node_probs,
    random,
    reconstruct_case,
    segmentation_metrics,
    sliding_window_predict,
    summarize_metric_rows,
    test_items,
    time,
    torch,
    train_items,
    train_patient_group,
    val_items,
    voxel_loss,
    write_json_atomic,
):
    # Stage 2: frozen graph prior, on-demand disk cache, full-volume selection.
    def stage2_node_probs(data, meta_path, cached_only=False):
        _key = hashlib.sha256((STAGE1_MODEL_SHA256 + DATASET_FINGERPRINT + meta_path).encode()).hexdigest()
        _cache_dir = os.path.join(PERSISTENT_BASE, "node_probs", STAGE1_MODEL_SHA256[:16])
        os.makedirs(_cache_dir, exist_ok=True)
        _path = os.path.join(_cache_dir, _key + ".npz")
        if not os.path.exists(_path):
            if cached_only:
                raise RuntimeError("Frozen node cache is missing; resume the cache warmup before threaded preparation")
            _probs = predict_node_probs(model, data)
            with open(_path + ".tmp", "wb") as _fh:
                np.savez_compressed(_fh, **_probs)
            os.replace(_path + ".tmp", _path)
        with np.load(_path, allow_pickle=False) as _z:
            return {k: _z[k].copy() for k in _z.files}


    def stage2_predict(data, meta_path, meta):
        _pred = sliding_window_predict(voxel_head, meta, stage2_node_probs(data, meta_path),
                                       roi=VOXEL_MAX_SIZE, overlap=INFERENCE_OVERLAP)
        return postprocess_prediction_auto(_pred)


    def evaluate_items_voxel(items, name="Set"):
        _rows = {"graph": [], "voxel": []}
        _patient_rows = []
        for _data, _mpath in items:
            _meta = load_meta(_mpath)
            _graph = reconstruct_case(model, _data, _meta)
            _hybrid = stage2_predict(_data, _mpath, _meta)
            _mg = segmentation_metrics(_graph, _meta["seg"])
            _mv = segmentation_metrics(_hybrid, _meta["seg"])
            _rows["graph"].append(_mg); _rows["voxel"].append(_mv)
            _patient_rows.append({"case_id": os.path.basename(_mpath).replace(".meta.pt", ""),
                                  "graph": _mg, "voxel": _mv})
            print(f"{name}: graph {fmt_metrics(_mg)} | hybrid {fmt_metrics(_mv)}")
        _summary = {k: summarize_metric_rows(v) for k, v in _rows.items()}
        with open(os.path.join(PERSISTENT_BASE, name.lower() + "_patient_metrics.json"), "w") as _fh:
            json.dump({"summary": _summary, "patients": _patient_rows,
                       "protocol": PROTOCOL_VERSION, "stage1_sha256": STAGE1_MODEL_SHA256}, _fh, indent=2)
        return _summary


    voxel_head = None
    vox_history = []
    best_vox_dice = -1.0
    best_vox_epoch = 0
    best_vox_state = None
    vox_epochs_since_improve = 0
    stage2_training_complete = not USE_VOXEL_REFINEMENT
    val_vox_metrics = None
    test_vox_metrics = None

    if USE_VOXEL_REFINEMENT:
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
        if USE_AMP:
            torch.cuda.manual_seed_all(SEED)
        model.eval()
        for _parameter in model.parameters():
            _parameter.requires_grad_(False)
        voxel_head = VoxelRefinementHead(len(MODALITIES) + NUM_CLASSES, NUM_CLASSES, VOXEL_BASE_CHANNELS).to(device)
        voxel_optimizer = torch.optim.AdamW(voxel_head.parameters(), lr=VOXEL_LR, weight_decay=WEIGHT_DECAY)
        voxel_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(voxel_optimizer, T_max=VOXEL_EPOCHS, eta_min=VOXEL_LR * .02)
        _gpu_runtime = configure_gpu(voxel_head, np.zeros((len(MODALITIES) + NUM_CLASSES, 4, 4, 4), np.float32),
                                     precision=GPU_PRECISION, autotune=False)
        _gpu_runtime["precision_validated"] = False
        voxel_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and _gpu_runtime["precision"] == "fp16")
        _class_weights_dev = class_weights.to(device)

        def stage2_state(epoch, complete):
            return {"epoch": epoch, "training_complete": complete, "protocol": PROTOCOL_VERSION,
                    "config_hash": CONFIG_HASH, "split": SPLIT_CASE_IDS, "split_fingerprint": SPLIT_FINGERPRINT,
                    "dataset_fingerprint": DATASET_FINGERPRINT, "stage1_sha256": STAGE1_MODEL_SHA256,
                    "stage2_config": RUN_CONFIG["voxel"],
                    "gpu_runtime": _gpu_runtime,
                    "voxel_state_dict": {k: v.detach().cpu().clone() for k, v in voxel_head.state_dict().items()},
                    "optimizer_state_dict": voxel_optimizer.state_dict(), "scheduler_state_dict": voxel_scheduler.state_dict(),
                    "scaler_state_dict": voxel_scaler.state_dict(),
                    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                            "python": random.getstate(), "cuda": torch.cuda.get_rng_state_all() if USE_AMP else None},
                    "vox_history": vox_history, "best_vox_dice": best_vox_dice, "best_vox_epoch": best_vox_epoch,
                    "best_vox_state": best_vox_state, "vox_epochs_since_improve": vox_epochs_since_improve}

        def save_stage2(epoch, complete, push=False):
            torch.save(stage2_state(epoch, complete), STAGE2_LATEST_PATH + ".tmp")
            os.replace(STAGE2_LATEST_PATH + ".tmp", STAGE2_LATEST_PATH)
            if push and HF_ENABLED:
                hf_upload_file_verified(STAGE2_LATEST_PATH, STAGE2_LATEST_NAME, HF_MODEL_REPO_ID,
                                        HF_MODEL_REPO_TYPE, f"review stage2 epoch {epoch}")

        _start_epoch = 1
        _resume_path = None
        if RESUME_STAGE2:
            _resume_path = STAGE2_LATEST_PATH if os.path.exists(STAGE2_LATEST_PATH) else (
                hf_try_download(STAGE2_LATEST_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE) if HF_ENABLED else None)
        if _resume_path:
            _latest = torch.load(_resume_path, map_location="cpu", weights_only=False)
            for _key, _expected in {"protocol": PROTOCOL_VERSION, "config_hash": CONFIG_HASH,
                                    "split": SPLIT_CASE_IDS, "split_fingerprint": SPLIT_FINGERPRINT,
                                    "dataset_fingerprint": DATASET_FINGERPRINT, "stage1_sha256": STAGE1_MODEL_SHA256,
                                    "stage2_config": RUN_CONFIG["voxel"]}.items():
                if _latest.get(_key) != _expected:
                    raise RuntimeError(f"Stage2 resume mismatch: {_key}; use a fresh checkpoint")
            # A pause during frozen-cache warmup has not trained this head yet.
            # Permit the first real calibration to choose FP32 if AMP is unstable.
            _saved_precision = (_latest.get("gpu_runtime", {}).get("precision")
                if _latest["epoch"] > 0 or _latest.get("gpu_runtime", {}).get("precision_validated", False) else None)
            _gpu_runtime = configure_gpu(voxel_head, np.zeros((len(MODALITIES) + NUM_CLASSES, 4, 4, 4), np.float32),
                                         precision=GPU_PRECISION, autotune=False, resume_precision=_saved_precision)
            if _saved_precision is not None and _saved_precision != _gpu_runtime["precision"]:
                raise RuntimeError("Resume requires the saved numeric precision on a compatible GPU")
            _gpu_runtime["precision_validated"] = _saved_precision is not None
            voxel_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and _gpu_runtime["precision"] == "fp16")
            voxel_head.load_state_dict(_latest["voxel_state_dict"])
            voxel_optimizer.load_state_dict(_latest["optimizer_state_dict"])
            voxel_scheduler.load_state_dict(_latest["scheduler_state_dict"])
            voxel_scaler.load_state_dict(_latest["scaler_state_dict"])
            torch.set_rng_state(_latest["rng"]["torch"])
            np.random.set_state(_latest["rng"]["numpy"])
            random.setstate(_latest["rng"]["python"])
            if USE_AMP and _latest["rng"]["cuda"] is not None:
                torch.cuda.set_rng_state_all(_latest["rng"]["cuda"])
            vox_history = _latest["vox_history"]
            best_vox_dice, best_vox_epoch = _latest["best_vox_dice"], _latest["best_vox_epoch"]
            best_vox_state = _latest["best_vox_state"]
            vox_epochs_since_improve = _latest["vox_epochs_since_improve"]
            stage2_training_complete = _latest["training_complete"]
            _start_epoch = _latest["epoch"] + 1
        else:
            save_stage2(0, False, push=True)

        def _stage2_loss(output, target):
            _logits, _boundary = output
            return voxel_loss(_logits, target, _class_weights_dev, dice_weight=VOXEL_DICE_WEIGHT,
                              ce_weight=VOXEL_CE_WEIGHT, focal_weight=VOXEL_FOCAL_WEIGHT,
                              focal_gamma=VOXEL_FOCAL_GAMMA, boundary_logits=_boundary,
                              boundary_weight=VOXEL_BOUNDARY_WEIGHT)

        # Build only compact frozen probabilities before CPU workers start. Workers
        # never invoke CUDA or touch the graph model, and never retain full cohorts.
        if not stage2_training_complete:
            for _data, _mpath in train_items + val_items:
                if (time.time() - PART2_START_TIME) / 3600 >= TRAIN_BUDGET_HOURS:
                    save_stage2(_start_epoch - 1, False, push=True)
                    mo.stop(True, mo.md("Node cache partly built. Resume Part 2; completed files are retained."))
                stage2_node_probs(_data, _mpath)
        _first_data, _first_path = train_items[0]
        _probe_meta = load_meta(_first_path)
        _probe_x, _probe_y = make_training_patch(_probe_meta, VOXEL_MAX_SIZE,
            stage2_node_probs(_first_data, _first_path), rng=np.random.RandomState(SEED))
        _gpu_runtime = configure_gpu(voxel_head, _probe_x, _probe_y,
            _stage2_loss if not stage2_training_complete else None,
            roi=VOXEL_MAX_SIZE, precision=GPU_PRECISION, max_microbatch=VOXEL_MAX_MICROBATCH,
            max_inference_batch=INFERENCE_MAX_BATCH, memory_fraction=GPU_MEMORY_FRACTION,
            autotune=GPU_AUTOTUNE, resume_precision=(_gpu_runtime["precision"]
                if _resume_path and _gpu_runtime.get("precision_validated", False) else None))
        _gpu_runtime["precision_validated"] = True
        voxel_scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and _gpu_runtime["precision"] == "fp16")
        if _resume_path:
            voxel_scaler.load_state_dict(_latest["scaler_state_dict"])
        write_json_atomic(os.path.join(PERSISTENT_BASE, "gpu_runtime.json"), _gpu_runtime)
        if not _resume_path:
            save_stage2(0, False, push=True)
        del _probe_meta, _probe_x, _probe_y

        def _prepare_stage2_case(value):
            (_data, _mpath), _seed = value
            _meta = load_meta(_mpath)
            _x, _y = make_training_patch(_meta, VOXEL_MAX_SIZE, stage2_node_probs(_data, _mpath, cached_only=True),
                                         rng=np.random.RandomState(int(_seed)))
            _tx = torch.from_numpy(_x).float()
            _ty = torch.from_numpy(_y).long()
            if USE_AMP:
                _tx, _ty = _tx.pin_memory(), _ty.pin_memory()
            return _tx, _ty

        _last_epoch = _start_epoch - 1
        _last_epoch_hours = 0.
        _paused = False
        for _epoch in range(_start_epoch, VOXEL_EPOCHS + 1):
            if stage2_training_complete:
                break
            if (time.time() - PART2_START_TIME) / 3600 + 1.1 * _last_epoch_hours >= TRAIN_BUDGET_HOURS:
                _paused = True
                break
            _t0 = time.time()
            voxel_head.train()
            _train_losses, _train_dices = [], []
            _epoch_items = sorted(train_items, key=lambda item: os.path.basename(item[1]))
            random.shuffle(_epoch_items)
            _seeds = np.random.randint(0, 2**31 - 1, size=len(_epoch_items))
            _prepared = bounded_prefetch(zip(_epoch_items, _seeds), _prepare_stage2_case,
                                         workers=VOXEL_PREFETCH_WORKERS if USE_AMP else 0,
                                         depth=max(2, VOXEL_EFFECTIVE_BATCH_SIZE))
            try:
                for _group in patient_groups(_prepared, VOXEL_EFFECTIVE_BATCH_SIZE):
                    if (time.time() - PART2_START_TIME) / 3600 >= TRAIN_BUDGET_HOURS:
                        _paused = True
                        break
                    _group_loss, _group_dices = train_patient_group(voxel_head, _group, voxel_optimizer,
                        voxel_scaler, _stage2_loss, _gpu_runtime)
                    _train_losses.extend([_group_loss] * len(_group))
                    _train_dices.extend(_group_dices)
            finally:
                _prepared.close()
            if _paused:
                break
            voxel_head.eval()
            _val_dices, _val_losses = [], []
            for _data, _mpath in val_items:
                if (time.time() - PART2_START_TIME) / 3600 >= TRAIN_BUDGET_HOURS:
                    _paused = True
                    break
                _meta = load_meta(_mpath)
                # Selection uses ALL voxels. The plotted loss uses a fixed UNIFORM
                # validation patch and is labelled as a patch loss in the run notes.
                _val_dices.append(mean_region_dice(stage2_predict(_data, _mpath, _meta), _meta["seg"]))
                _rng_seed = int(hashlib.sha256(os.path.basename(_mpath).encode()).hexdigest()[:8], 16)
                _x_np, _y_np = make_training_patch(_meta, VOXEL_MAX_SIZE, stage2_node_probs(_data, _mpath),
                                                   rng=np.random.RandomState(_rng_seed), tumour_probability=0.)
                with torch.no_grad(), amp_context(_gpu_runtime, device):
                    _logits, _boundary = voxel_head(torch.from_numpy(_x_np).unsqueeze(0).float().to(device))
                    _vloss = voxel_loss(_logits, torch.from_numpy(_y_np).unsqueeze(0).long().to(device),
                                        _class_weights_dev, dice_weight=VOXEL_DICE_WEIGHT, ce_weight=VOXEL_CE_WEIGHT,
                                        focal_weight=VOXEL_FOCAL_WEIGHT, focal_gamma=VOXEL_FOCAL_GAMMA,
                                        boundary_logits=_boundary, boundary_weight=VOXEL_BOUNDARY_WEIGHT)
                _val_losses.append(float(_vloss))
            if _paused:
                break
            voxel_scheduler.step()
            _vd = float(np.mean(_val_dices))
            _seconds = time.time() - _t0
            vox_history.append(dict(epoch=_epoch, train_loss=float(np.mean(_train_losses)),
                                    val_loss=float(np.mean(_val_losses)), train_patch_dice=float(np.mean(_train_dices)),
                                    val_dice=_vd, epoch_seconds=_seconds))
            if _vd > best_vox_dice:
                best_vox_dice, best_vox_epoch = _vd, _epoch
                best_vox_state = {k: v.detach().cpu().clone() for k, v in voxel_head.state_dict().items()}
                vox_epochs_since_improve = 0
            else:
                vox_epochs_since_improve += 1
            _last_epoch, _last_epoch_hours = _epoch, _seconds / 3600
            stage2_training_complete = _epoch == VOXEL_EPOCHS or vox_epochs_since_improve >= EARLY_STOP_PATIENCE
            save_stage2(_epoch, stage2_training_complete,
                        push=stage2_training_complete or _epoch % CKPT_PUSH_EVERY_EPOCHS == 0)
            write_json_atomic(os.path.join(PERSISTENT_BASE, "gpu_runtime.json"), _gpu_runtime)
            print(f"Epoch {_epoch}: training patch loss={np.mean(_train_losses):.4f}; full-volume val Dice={_vd:.4f}")
            if stage2_training_complete:
                break
        if _paused:
            # Retain the last COMPLETED epoch; partial epochs are replayed on resume.
            if HF_ENABLED:
                hf_upload_file_verified(STAGE2_LATEST_PATH, STAGE2_LATEST_NAME, HF_MODEL_REPO_ID,
                                        HF_MODEL_REPO_TYPE, f"review stage2 paused after epoch {_last_epoch}")
        mo.stop(not stage2_training_complete, mo.md("Session budget reached. Resume Part 2; the last completed epoch is saved."))
        if best_vox_state is not None:
            voxel_head.load_state_dict(best_vox_state)
        val_vox_metrics = evaluate_items_voxel(val_items, "VAL")
        test_vox_metrics = evaluate_items_voxel(test_items, "TEST")
    return (
        best_vox_epoch,
        stage2_training_complete,
        test_vox_metrics,
        val_vox_metrics,
        vox_history,
        voxel_head,
    )


@app.cell
def _(
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    PERSISTENT_BASE,
    best_vox_epoch,
    hf_upload_file_verified,
    mo,
    np,
    os,
    plt,
    stage1_checkpoint,
    stage2_training_complete,
    vox_history,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    import csv as _csv


    def export_training_curves(hist, stem, loss_keys, dice_keys, best_epoch):
        _fig_dir = os.path.join(PERSISTENT_BASE, "figures")
        os.makedirs(_fig_dir, exist_ok=True)
        _csv_path = os.path.join(_fig_dir, f"{stem}.csv")
        _png_path = os.path.join(_fig_dir, f"{stem}.png")
        _scalar_keys = []
        for _row in hist:
            for _key, _value in _row.items():
                if isinstance(_value, (int, float, np.integer, np.floating)) and _key not in _scalar_keys:
                    _scalar_keys.append(_key)
        with open(_csv_path, "w", newline="") as _fh:
            _writer = _csv.DictWriter(_fh, fieldnames=_scalar_keys)
            _writer.writeheader()
            for _row in hist:
                _writer.writerow({k: _row.get(k, "") for k in _scalar_keys})
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
        _epochs = [row.get("epoch", i + 1) for i, row in enumerate(hist)]
        for _key in loss_keys:
            if _key in _scalar_keys:
                _axes[0].plot(_epochs, [row.get(_key, np.nan) for row in hist], label=_key)
        for _key in dice_keys:
            if _key in _scalar_keys:
                _axes[1].plot(_epochs, [row.get(_key, np.nan) for row in hist], label=_key)
        _axes[0].set_title("Loss"); _axes[0].set_xlabel("epoch"); _axes[0].legend()
        _axes[1].set_title("Dice"); _axes[1].set_xlabel("epoch"); _axes[1].legend()
        for _axis in _axes:
            _axis.axvline(best_epoch, color="black", linestyle="--", alpha=0.7)
            _axis.grid(alpha=0.2)
        _fig.tight_layout()
        _fig.savefig(_png_path, dpi=150)
        plt.close(_fig)
        return _csv_path, _png_path


    STAGE2_CURVE_FILES = export_training_curves(
        vox_history, "stage2_curves", ("train_loss", "val_loss"),
        ("train_patch_dice", "val_dice"), best_vox_epoch)
    STAGE1_CURVE_FILES = export_training_curves(
        stage1_checkpoint.get("history") or [], "stage1_curves",
        ("train_total", "val_total"), ("train_node_proxy", "val_dice"),
        stage1_checkpoint.get("best_epoch", 0))
    if HF_ENABLED and stage1_checkpoint.get("training_complete"):
        for _path in STAGE1_CURVE_FILES:
            hf_upload_file_verified(_path, os.path.basename(_path), HF_MODEL_REPO_ID,
                                    HF_MODEL_REPO_TYPE, "stage1 training curves")
    if HF_ENABLED and stage2_training_complete:
        for _path in STAGE2_CURVE_FILES:
            hf_upload_file_verified(_path, os.path.basename(_path), HF_MODEL_REPO_ID,
                                    HF_MODEL_REPO_TYPE, "stage2 training curves")
    return


@app.cell
def _(
    NODE_TYPES,
    USE_STRUCTURAL_REFINEMENT,
    mo,
    postprocess_prediction,
    project_node_probs,
    stage2_training_complete,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # ============================================================
    # Option B: reconstruction helpers (initial vs refined)
    # ------------------------------------------------------------
    # Generalises predict_node_probs/reconstruct_case to return BOTH the base
    # model's initial prediction and the (possibly refined) final prediction, so
    # we can measure whether refinement actually helps. In baseline mode
    # (USE_STRUCTURAL_REFINEMENT=False) initial and final are identical by
    # construction -- the plain model output, unchanged.
    @torch.no_grad()
    def predict_node_probs_both(model, data):
        model.eval()
        dev = next(model.parameters()).device
        if USE_STRUCTURAL_REFINEMENT:
            initial_logits, final_logits, _ = model(data.to(dev), teacher_prob=0.0)  # never GT at eval
        else:
            final_logits, _ = model(data.to(dev))
            initial_logits = final_logits
        init_p = {nt: torch.softmax(initial_logits[nt].float(), dim=-1).cpu().numpy() for nt in NODE_TYPES}
        fin_p = {nt: torch.softmax(final_logits[nt].float(), dim=-1).cpu().numpy() for nt in NODE_TYPES}
        return init_p, fin_p

    def reconstruct_case_both(model, data, meta, postprocess=True):
        init_p, fin_p = predict_node_probs_both(model, data)
        pred_init = project_node_probs(init_p, meta)
        pred_fin = project_node_probs(fin_p, meta)
        if postprocess:
            pred_init = postprocess_prediction(pred_init)
            pred_fin = postprocess_prediction(pred_fin)
        return pred_init, pred_fin

    return (reconstruct_case_both,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13a. Learned-hop diagnostics and graph-node visualization

    When adaptive hops are enabled, every node receives a probability distribution over propagation depths $k=0,\ldots,K_{max}$. The expected depth $k_{eff}=\sum_k k\beta_k$ is a differentiable, interpretable receptive-field size. The plots below report the learned distribution by modality and class, color graph nodes by effective hop, and summarize learned edge-gate strengths at every propagation step.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    K_MAX,
    NODE_TYPES,
    USE_ADAPTIVE_HOPS,
    base_model,
    load_meta,
    mo,
    model,
    np,
    plt,
    predict_node_probs,
    stage2_training_complete,
    test_items,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    @torch.no_grad()
    def adaptive_hop_diagnostics(item, title="Learned adaptive propagation depth"):
        if not USE_ADAPTIVE_HOPS:
            print("Adaptive hops disabled; set USE_ADAPTIVE_HOPS=True to generate these diagnostics.")
            return None
        data, mpath = item
        meta = load_meta(mpath)
        predict_node_probs(model, data)
        hop_info = base_model.last_hop_info
        gate_history = base_model.last_edge_gates
        if not hop_info:
            print("No hop diagnostics available; run a model forward pass first.")
            return None

        fig, axes = plt.subplots(2, len(NODE_TYPES), figsize=(7 * len(NODE_TYPES), 10))
        class_colors = np.asarray(["#808080", "#4169e1", "#e53935", "#fdd835"])
        summaries = {}
        for col, nt in enumerate(NODE_TYPES):
            effective = hop_info[nt]["effective_hop"].cpu().numpy()
            selected = hop_info[nt]["selected_hop"].cpu().numpy()
            weights = hop_info[nt]["weights"].cpu().numpy()
            labels = data[nt].y.cpu().numpy()
            centroids = data[nt].x[:, 28:31].cpu().numpy()
            summaries[nt] = {
                "mean_effective_hop": float(effective.mean()),
                "mean_hop_weights": weights.mean(axis=0).tolist(),
                "selected_hop_counts": np.bincount(selected, minlength=K_MAX + 1).tolist(),
            }

            axes[0, col].hist(effective, bins=np.linspace(0, K_MAX, 21), color="#5b8ff9", alpha=0.85)
            axes[0, col].axvline(effective.mean(), color="black", linestyle="--",
                                 label=f"mean={effective.mean():.2f}")
            axes[0, col].set(title=f"{nt}: effective-hop distribution", xlabel="effective hop", ylabel="nodes")
            axes[0, col].legend()

            scatter = axes[1, col].scatter(centroids[:, 0], centroids[:, 1], c=effective,
                                           cmap="viridis", vmin=0, vmax=K_MAX, s=5, alpha=0.7)
            tumour = labels > 0
            axes[1, col].scatter(centroids[tumour, 0], centroids[tumour, 1],
                                 facecolors="none", edgecolors=class_colors[labels[tumour]], s=18, linewidths=0.5)
            axes[1, col].set(title=f"{nt}: nodes colored by learned k\nring=GT tumour class",
                             xlabel="normalized x", ylabel="normalized y", aspect="equal")
            fig.colorbar(scatter, ax=axes[1, col], label="effective hop")

            print(f"{nt}: mean effective hop={effective.mean():.3f}")
            print("  mean beta:", "  ".join(f"k={k}:{v:.3f}" for k, v in enumerate(weights.mean(axis=0))))
            for cls, name in enumerate(CLASS_NAMES):
                mask = labels == cls
                if mask.any():
                    print(f"  {name}: n={mask.sum()}  mean k_eff={effective[mask].mean():.3f}")

        if gate_history:
            print("\nLearned adaptive edge-gate alpha by propagation step:")
            for step, gates in enumerate(gate_history, start=1):
                print(f"  hop {step}: " + "  ".join(
                    f"{et[0]}-{et[1]}-{et[2]}={float(alpha.mean()):.3f}"
                    for et, alpha in gates.items() if alpha.numel()))
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        return summaries

    hop_diagnostics = adaptive_hop_diagnostics(test_items[0]) if test_items else None
    return


@app.cell
def _(
    NODE_TYPES,
    USE_ADAPTIVE_HOPS,
    base_model,
    mo,
    model,
    np,
    predict_node_probs,
    stage2_training_complete,
    test_items,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    @torch.no_grad()
    def effective_khop_by_region(items):
        if not USE_ADAPTIVE_HOPS:
            return None
        _regions = {"BG": lambda y: y == 0, "WT": lambda y: y > 0,
                    "TC": lambda y: (y == 1) | (y == 3), "ET": lambda y: y == 3}
        _per_node = {nt: {region: [] for region in _regions} for nt in NODE_TYPES}
        _pooled = {region: [] for region in _regions}
        for _data, _mpath in items:
            predict_node_probs(model, _data)
            _hop_info = base_model.last_hop_info
            _patient_pooled = {region: [] for region in _regions}
            for _nt in NODE_TYPES:
                _effective = _hop_info[_nt]["effective_hop"].cpu().numpy()
                _labels = _data[_nt].y.cpu().numpy()
                for _region, _mask_fn in _regions.items():
                    _mask = _mask_fn(_labels)
                    if _mask.any():
                        _value = float(_effective[_mask].mean())
                        _per_node[_nt][_region].append(_value)
                        _patient_pooled[_region].extend(_effective[_mask].tolist())
            for _region, _values in _patient_pooled.items():
                if _values:
                    _pooled[_region].append(float(np.mean(_values)))
        def _summary(_values):
            return {"mean": float(np.mean(_values)) if _values else None,
                    "std": float(np.std(_values)) if _values else None,
                    "n_patients": len(_values)}
        _result = {"per_node_type": {nt: {region: _summary(_per_node[nt][region])
                                            for region in _regions} for nt in NODE_TYPES},
                   "pooled": {region: _summary(_pooled[region]) for region in _regions}}
        print("Region".ljust(10) + "".join(nt.rjust(14) for nt in NODE_TYPES) + "  " + "Pooled".rjust(14))
        for _region in _regions:
            _columns = [
                f"{_result['per_node_type'][_nt][_region]['mean']:.3f}"
                if _result['per_node_type'][_nt][_region]['mean'] is not None else "n/a"
                for _nt in NODE_TYPES]
            _pooled_mean = _result["pooled"][_region]["mean"]
            print(_region.ljust(10) + "".join(_value.rjust(14) for _value in _columns) +
                  "  " + (f"{_pooled_mean:.3f}" if _pooled_mean is not None else "n/a").rjust(14))
        return _result


    KHOP_BY_REGION = effective_khop_by_region(test_items)
    return (KHOP_BY_REGION,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 13c. Learned gate α: boundary vs interior edges
    """)
    return


@app.cell
def _(
    NODE_TYPES,
    USE_ADAPTIVE_HOPS,
    base_model,
    mo,
    model,
    np,
    plt,
    predict_node_probs,
    stage2_training_complete,
    test_items,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    @torch.no_grad()
    def edge_alpha_boundary_analysis(item, title="Learned edge gate α: boundary vs interior edges"):
        """Are the learned adaptive gates α selective about tumour boundaries?

        For every edge, α is averaged over the propagation hops (`base_model.
        last_edge_gates` holds one α vector per hop per edge type), then the edges
        of each relation are split into

        * **boundary edges** — the two endpoints differ in tumour/normal status
          (`data[nt].y > 0` differs), i.e. the edge crosses the tumour margin;
        * **interior/similar edges** — both endpoints share the same tumour status.

        A gate that is systematically *lower* on boundary edges means the model
        learned to stop mixing features across the margin (sharper boundaries); a
        systematically *higher* gate means it deliberately pulls context across it.
        """
        if not USE_ADAPTIVE_HOPS:
            print("Adaptive hops disabled; set USE_ADAPTIVE_HOPS=True to generate these diagnostics.")
            return None
        data, _ = item
        predict_node_probs(model, data)                 # populates base_model.last_edge_gates
        gate_history = base_model.last_edge_gates
        if not gate_history:
            print("No edge gates recorded; run a model forward pass first.")
            return None

        truth = {nt: data[nt].y.cpu().numpy() for nt in NODE_TYPES}
        groups = {}
        for edge_type in base_model.metadata[1]:
            steps = [step[edge_type].cpu().numpy() for step in gate_history
                     if edge_type in step and step[edge_type].numel()]
            if not steps:
                continue
            alpha = np.mean(np.stack(steps), axis=0)     # mean α per edge, averaged over hops
            src, dst = data[edge_type].edge_index.cpu().numpy()
            is_boundary = (truth[edge_type[0]][src] > 0) != (truth[edge_type[2]][dst] > 0)
            groups["-".join(edge_type)] = dict(alpha=alpha, boundary=is_boundary)
        if not groups:
            print("No edges with recorded gates.")
            return None

        summary = {}
        print("Mean learned gate α, boundary (tumour status differs) vs interior edges:")
        print(f"{'relation':<26}{'n_bnd':>8}{'α_bnd':>9}{'n_int':>9}{'α_int':>9}{'Δ':>9}")
        for name, g in groups.items():
            a_b, a_i = g["alpha"][g["boundary"]], g["alpha"][~g["boundary"]]
            row = {
                "n_boundary": int(a_b.size), "n_interior": int(a_i.size),
                "mean_alpha_boundary": float(a_b.mean()) if a_b.size else float("nan"),
                "mean_alpha_interior": float(a_i.mean()) if a_i.size else float("nan"),
                "std_alpha_boundary": float(a_b.std()) if a_b.size else float("nan"),
                "std_alpha_interior": float(a_i.std()) if a_i.size else float("nan"),
            }
            row["delta"] = row["mean_alpha_boundary"] - row["mean_alpha_interior"]
            summary[name] = row
            print(f"{name:<26}{row['n_boundary']:>8}{row['mean_alpha_boundary']:>9.3f}"
                  f"{row['n_interior']:>9}{row['mean_alpha_interior']:>9.3f}{row['delta']:>+9.3f}")

        names = list(groups)
        fig, axes = plt.subplots(1, len(names) + 1, figsize=(4.5 * (len(names) + 1), 4.2))
        axes = np.atleast_1d(axes)
        for col, name in enumerate(names):
            g = groups[name]
            bins = np.linspace(0.0, 1.0, 41)
            axes[col].hist(g["alpha"][~g["boundary"]], bins=bins, density=True, alpha=0.55,
                           color="#5b8ff9", label="interior/similar")
            axes[col].hist(g["alpha"][g["boundary"]], bins=bins, density=True, alpha=0.55,
                           color="#e53935", label="boundary")
            axes[col].set(title=name, xlabel="gate α (mean over hops)", ylabel="density")
            axes[col].legend(fontsize=8)

        width = 0.38
        positions = np.arange(len(names))
        axes[-1].bar(positions - width / 2, [summary[n]["mean_alpha_interior"] for n in names],
                     width, yerr=[summary[n]["std_alpha_interior"] for n in names],
                     color="#5b8ff9", capsize=3, label="interior/similar")
        axes[-1].bar(positions + width / 2, [summary[n]["mean_alpha_boundary"] for n in names],
                     width, yerr=[summary[n]["std_alpha_boundary"] for n in names],
                     color="#e53935", capsize=3, label="boundary")
        axes[-1].set(xticks=positions, ylabel="mean α", title="Mean gate α per relation")
        axes[-1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        axes[-1].legend(fontsize=8)
        fig.suptitle(title)
        fig.tight_layout()
        plt.show()
        return summary


    edge_alpha_boundary_summary = edge_alpha_boundary_analysis(test_items[0]) if test_items else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13b. Option B diagnostics: initial vs refined, structural statistics
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    NODE_TYPES,
    NUM_CLASSES,
    STRUCTURAL_FLAGS,
    USE_STRUCTURAL_REFINEMENT,
    connected_components,
    device,
    edge_index_to_csr,
    fmt_metrics,
    load_meta,
    mo,
    model,
    np,
    reconstruct_case_both,
    segmentation_metrics,
    stage2_training_complete,
    test_items,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # ============================================================
    # Option B diagnostics: does structural refinement actually help?
    # ------------------------------------------------------------
    # Reports initial (pre-refinement) vs final (post-refinement) Dice, plus
    # structural diagnostics. In baseline mode (USE_STRUCTURAL_REFINEMENT=False)
    # initial == final by construction, so this section mainly confirms that and
    # skips the structure-specific diagnostics.
    import time as _optb_time

    def _class_neighbor_transition_matrix(items, n_cases=8):
        """[NUM_CLASSES, NUM_CLASSES] matrix: rows = a node's predicted class,
        cols = the class distribution among its spatial neighbours. Row-normalised.
        Computed on the FINAL prediction, on up to n_cases test graphs (a
        diagnostic, not a training-time metric -- kept small on purpose)."""
        trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)
        for _d, _mpath in items[:n_cases]:
            _dev_d = _d.to(device)
            with torch.no_grad():
                if USE_STRUCTURAL_REFINEMENT:
                    _, _final_logits, _ = model(_dev_d, teacher_prob=0.0)
                else:
                    _final_logits, _ = model(_dev_d)
            for nt in NODE_TYPES:
                pc = _final_logits[nt].argmax(dim=-1).cpu().numpy()
                ei = _d[nt, 'spatial', nt].edge_index.cpu().numpy()
                if ei.size == 0:
                    continue
                src, dst = ei
                for c in range(NUM_CLASSES):
                    mask = pc[src] == c
                    if not mask.any():
                        continue
                    nb = pc[dst[mask]]
                    trans[c] = trans[c] + np.bincount(nb, minlength=NUM_CLASSES)
        row_sums = trans.sum(axis=1, keepdims=True)
        return trans / np.maximum(row_sums, 1.0)

    def _tumour_component_diagnostics(items, n_cases=8):
        """Average/median predicted-tumour connected-component size and the
        number of isolated (size==1) tumour components, on the FINAL prediction."""
        sizes_all = []
        for _d, _mpath in items[:n_cases]:
            _dev_d = _d.to(device)
            with torch.no_grad():
                if USE_STRUCTURAL_REFINEMENT:
                    _, _final_logits, _ = model(_dev_d, teacher_prob=0.0)
                else:
                    _final_logits, _ = model(_dev_d)
            for nt in NODE_TYPES:
                pc = _final_logits[nt].argmax(dim=-1).cpu().numpy()
                tumour_mask = pc != 0
                idx = np.nonzero(tumour_mask)[0]
                if idx.size == 0:
                    continue
                csr = edge_index_to_csr(_d[nt, 'spatial', nt].edge_index, pc.shape[0])
                sub = csr[idx][:, idx]
                _, labels = connected_components(sub, directed=False)
                sizes_all.extend(np.bincount(labels).tolist())
        sizes_all = np.asarray(sizes_all, dtype=np.float64)
        if sizes_all.size == 0:
            return dict(mean=0.0, median=0.0, isolated=0, n_components=0)
        return dict(mean=float(sizes_all.mean()), median=float(np.median(sizes_all)), isolated=int((sizes_all == 1).sum()), n_components=int(sizes_all.size))
    print('=' * 60)
    print('Option B diagnostics')
    print('=' * 60)
    print(f'USE_STRUCTURAL_REFINEMENT = {USE_STRUCTURAL_REFINEMENT}')
    if USE_STRUCTURAL_REFINEMENT:
        print(f'Enabled descriptors: {[k for k, v in STRUCTURAL_FLAGS.items() if v]}')
    _optb_t0 = _optb_time.time()
    _init_rows, _fin_rows = ([], [])
    for _d, _mpath in test_items:
        _meta = load_meta(_mpath)
        _pred_init, _pred_fin = reconstruct_case_both(model, _d, _meta, postprocess=True)
        _init_rows.append(segmentation_metrics(_pred_init, _meta['seg']))
        _fin_rows.append(segmentation_metrics(_pred_fin, _meta['seg']))
    init_mean = {k: float(np.mean([r[k] for r in _init_rows])) for k in _init_rows[0]}
    fin_mean = {k: float(np.mean([r[k] for r in _fin_rows])) for k in _fin_rows[0]}
    print(f'\nInitial (pre-refinement) : {fmt_metrics(init_mean)}')
    print(f'Refined (post-refinement): {fmt_metrics(fin_mean)}')
    print('Improvement (Refined - Initial):', {k: f'{fin_mean[k] - init_mean[k]:+.4f}' for k in init_mean})
    if USE_STRUCTURAL_REFINEMENT:
        _comp_diag = _tumour_component_diagnostics(test_items)
        print(f'\nTumour connected components (final prediction, first {min(8, len(test_items))} test cases):')
        print(f"  count={_comp_diag['n_components']}  mean_size={_comp_diag['mean']:.1f}  median_size={_comp_diag['median']:.1f}  isolated(size=1)={_comp_diag['isolated']}")
        _trans = _class_neighbor_transition_matrix(test_items)
        print('\nClass-neighbour transition matrix (row = node class, col = neighbour-class fraction):')
        print('        ' + ''.join((f'{c:>10s}' for c in CLASS_NAMES)))
        for _i, _c in enumerate(CLASS_NAMES):
            print(f'{_c:>8s}' + ''.join((f'{_trans[_i, _j]:10.3f}' for _j in range(NUM_CLASSES))))
    else:
        print('\n(Structure-specific diagnostics skipped -- baseline mode.)')
    print(f'\nDiagnostics computed in {(_optb_time.time() - _optb_t0) / 60:.1f} min.')
    return fin_mean, init_mean


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14. Visualise one segmentation result

    The slice with the largest ground-truth tumour area is shown.
    """)
    return


@app.cell
def _(
    crop,
    fmt_metrics,
    load_meta,
    mo,
    model,
    np,
    plt,
    reconstruct_case,
    segmentation_metrics,
    stage2_training_complete,
    test_items,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    def show_prediction(item, title="QoS-HRGN v2 prediction"):
        d, mpath = item
        meta = load_meta(mpath)
        pred = reconstruct_case(model, d, meta)
        lo, hi = meta["lo"], meta["hi"]
        seg_c = crop(meta["seg"], lo, hi)
        pred_c = crop(pred, lo, hi)
        z = int(np.argmax((seg_c > 0).sum(axis=(0, 1))))
        fig, ax = plt.subplots(1, 4, figsize=(16, 4))
        ax[0].imshow(meta["vis"]["t1ce"][:, :, z].astype(np.float32).T, cmap="gray", origin="lower"); ax[0].set_title("T1ce")
        ax[1].imshow(meta["vis"]["flair"][:, :, z].astype(np.float32).T, cmap="gray", origin="lower"); ax[1].set_title("FLAIR")
        ax[2].imshow(seg_c[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=3); ax[2].set_title("Ground truth")
        ax[3].imshow(pred_c[:, :, z].T, cmap="viridis", origin="lower", vmin=0, vmax=3)
        ax[3].set_title(f"{title}\n{fmt_metrics(segmentation_metrics(pred, meta['seg']))}", fontsize=9)
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()

    show_prediction(test_items[0])
    return


@app.cell
def _(
    PERSISTENT_BASE,
    SEED,
    USE_VOXEL_REFINEMENT,
    crop,
    fmt_metrics,
    load_meta,
    mo,
    model,
    np,
    os,
    plt,
    random,
    reconstruct_case,
    reconstruct_case_voxel,
    segmentation_metrics,
    stage2_training_complete,
    test_items,
    voxel_head,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    def error_overlay(pred_mask, gt_mask):
        _rgba = np.zeros(pred_mask.shape + (4,), dtype=np.float32)
        _tp = pred_mask & gt_mask
        _fp = pred_mask & ~gt_mask
        _fn = ~pred_mask & gt_mask
        _rgba[_tp] = (0.0, 1.0, 0.0, 0.6)
        _rgba[_fp] = (1.0, 0.0, 0.0, 0.6)
        _rgba[_fn] = (0.0, 0.0, 1.0, 0.6)
        return _rgba


    def show_error_map(item, use_voxel=None, save=True):
        if use_voxel is None:
            use_voxel = USE_VOXEL_REFINEMENT and voxel_head is not None
        _data, _mpath = item
        _meta = load_meta(_mpath)
        _pred = (reconstruct_case_voxel(model, voxel_head, _data, _meta, postprocess=True)
                 if use_voxel else reconstruct_case(model, _data, _meta, postprocess=True))
        _lo, _hi = _meta["lo"], _meta["hi"]
        _gt_c = crop(_meta["seg"], _lo, _hi)
        _pred_c = crop(_pred, _lo, _hi)
        _t1ce_c = _meta["vis"]["t1ce"]
        _z = int(np.argmax((_gt_c > 0).sum(axis=(0, 1))))
        _regions = {
            "WT": (lambda x: x > 0),
            "TC": (lambda x: (x == 1) | (x == 3)),
            "ET": (lambda x: x == 3),
        }
        _fig, _axes = plt.subplots(1, 6, figsize=(24, 4))
        _axes[0].imshow(_t1ce_c[:, :, _z].T, cmap="gray", origin="lower"); _axes[0].set_title("T1ce")
        _axes[1].imshow(_gt_c[:, :, _z].T, cmap="viridis", origin="lower", vmin=0, vmax=3); _axes[1].set_title("GT")
        _axes[2].imshow(_pred_c[:, :, _z].T, cmap="viridis", origin="lower", vmin=0, vmax=3); _axes[2].set_title("Pred")
        for _axis, (_region, _mask_fn) in zip(_axes[3:], _regions.items()):
            _gt_region = _mask_fn(_meta["seg"])
            _pred_region = _mask_fn(_pred)
            _fp = int(np.logical_and(_pred_region, ~_gt_region).sum())
            _fn = int(np.logical_and(~_pred_region, _gt_region).sum())
            _axis.imshow(_t1ce_c[:, :, _z].T, cmap="gray", origin="lower")
            _axis.imshow(error_overlay(_pred_region[_lo[0]:_hi[0], _lo[1]:_hi[1], _lo[2]:_hi[2]],
                                       _gt_region[_lo[0]:_hi[0], _lo[1]:_hi[1], _lo[2]:_hi[2]])[:, :, _z].transpose(1, 0, 2))
            _axis.set_title(f"{_region} err  FP={_fp} FN={_fn}")
        for _axis in _axes:
            _axis.axis("off")
        _case_id = os.path.basename(_mpath).replace(".meta.pt", "")
        _fig.suptitle(f"{_case_id} | {fmt_metrics(segmentation_metrics(_pred, _meta['seg']))}")
        _fig.tight_layout()
        _path = None
        if save:
            os.makedirs(os.path.join(PERSISTENT_BASE, "figures"), exist_ok=True)
            _path = os.path.join(PERSISTENT_BASE, "figures", f"error_map_{_case_id}.png")
            _fig.savefig(_path, dpi=150)
        plt.show()
        plt.close(_fig)
        return _path


    for _item in random.Random(SEED).sample(test_items, min(8, len(test_items))):
        show_error_map(_item)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14b. Option B: visualise initial vs refined prediction
    """)
    return


@app.cell
def _(
    NUM_CLASSES,
    USE_STRUCTURAL_REFINEMENT,
    load_meta,
    mo,
    model,
    np,
    plt,
    reconstruct_case_both,
    stage2_training_complete,
    test_items,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # ============================================================
    # Option B: optional visualisation (not required for training)
    # ============================================================
    def show_structural_refinement(item, title="Option B: ground truth vs initial vs refined"):
        if not USE_STRUCTURAL_REFINEMENT:
            print("Structural refinement is disabled (USE_STRUCTURAL_REFINEMENT=False); nothing to visualise.")
            return
        d, mpath = item
        meta = load_meta(mpath)
        pred_init, pred_fin = reconstruct_case_both(model, d, meta, postprocess=True)
        seg = meta["seg"]
        z = int(np.argmax((seg > 0).sum(axis=(0, 1))))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, img, name in zip(
            axes, [seg[:, :, z], pred_init[:, :, z], pred_fin[:, :, z]],
            ["Ground truth", "Initial (pre-refinement)", "Refined (post-refinement)"],
        ):
            ax.imshow(img.T, cmap="viridis", vmin=0, vmax=NUM_CLASSES - 1, origin="lower")
            ax.set_title(name)
            ax.axis("off")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    if USE_STRUCTURAL_REFINEMENT and test_items:
        show_structural_refinement(test_items[0])
    else:
        print("Skipping visualisation (baseline mode, or no test cases available).")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 14c. Option C: can the model actually recover masked graph information?

    Three checks, each against a baseline that must be beaten for the claim to mean anything:

    | Check | Model | Baseline it must beat |
    |---|---|---|
    | Masked appearance MAE | decoder on the embedding | (a) global mean of unmasked nodes, (b) mean of unmasked **spatial neighbours** |
    | Correspondence AUC | relation scorer | centroid-distance-only scorer (pure geometry) |
    | Visual | original -> masked -> reconstructed on one slice | - |

    The *neighbour-mean* baseline is the important one: beating the global mean only shows the model
    learned the dataset's average intensity, whereas beating the neighbour mean shows it learned
    something from the graph beyond naive local smoothing. The *distance-only* AUC baseline is what
    makes the correspondence number defensible -- if the model cannot beat geometry, the relation task
    is not measuring relational learning.
    """)
    return


@app.cell
def _(
    APPEARANCE_DIM,
    CORR_ET_FT,
    MODALITIES,
    MODALITY_SLICES,
    NODE_TYPES,
    REC_NEG_PER_POS,
    USE_MASKED_RECONSTRUCTION,
    crop,
    device,
    forward_model,
    load_meta,
    mask_hetero_graph,
    mo,
    model,
    np,
    plt,
    rec_module,
    sample_hard_negatives,
    stage2_training_complete,
    test_items,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # ============================================================
    # Option C: recovery evaluation + visualisation
    # ============================================================
    def _auc(scores, labels):
        """Rank-based ROC-AUC (no sklearn dependency)."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        order = np.argsort(scores, kind="mergesort")
        ranks = np.empty(len(scores), dtype=np.float64)
        ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
        return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


    @torch.no_grad()
    def rec_recover_case(data, seed=0):
        """Mask one graph, run the encoder, decode. Returns everything the diagnostics need."""
        model.eval(); rec_module.eval()
        gen = torch.Generator(device=device); gen.manual_seed(seed)
        d = data.to(device)
        corrupt, info = mask_hetero_graph(d, rec_module, gen=gen)
        _, _, h_dict = forward_model(model, corrupt, teacher_prob=0.0)
        out = {"info": info, "pred": {}, "target": {}, "idx": {}}
        for nt in NODE_TYPES:
            idx = info["masked_idx"].get(nt)
            if idx is None or idx.numel() == 0:
                continue
            out["idx"][nt] = idx.cpu().numpy()
            out["pred"][nt] = rec_module.decode_features(h_dict[nt][idx], nt).float().cpu().numpy()
            out["target"][nt] = info["feat_target"][nt].float().cpu().numpy()
        # correspondence scoring (model vs geometry-only baseline)
        pos = info["pos"].get("corresponds")
        if pos is not None and pos.numel() > 0:
            neg = sample_hard_negatives(
                pos, d[("flair", "spatial", "flair")].edge_index, info["n_nodes"]["flair"],
                d[CORR_ET_FT].edge_index, REC_NEG_PER_POS, device, gen, exclude_self=False)
            if neg.numel() > 0:
                s_pos = rec_module.score_pairs(h_dict["t1ce"][pos[0].long()], h_dict["flair"][pos[1].long()], "corresponds")
                s_neg = rec_module.score_pairs(h_dict["t1ce"][neg[0].long()], h_dict["flair"][neg[1].long()], "corresponds")
                # geometry-only baseline: negative centroid distance from feature dims 28..30
                c_t1 = d["t1ce"].x[:, APPEARANCE_DIM:APPEARANCE_DIM + 3].float()
                c_fl = d["flair"].x[:, APPEARANCE_DIM:APPEARANCE_DIM + 3].float()
                g_pos = -(c_t1[pos[0].long()] - c_fl[pos[1].long()]).norm(dim=1)
                g_neg = -(c_t1[neg[0].long()] - c_fl[neg[1].long()]).norm(dim=1)
                out["link"] = dict(
                    s_pos=s_pos.float().cpu().numpy(), s_neg=s_neg.float().cpu().numpy(),
                    g_pos=g_pos.cpu().numpy(), g_neg=g_neg.cpu().numpy(),
                )
        return out


    @torch.no_grad()
    def rec_evaluate(items, n_cases=8, seed=0):
        """Masked-feature MAE vs trivial baselines, and correspondence AUC vs geometry."""
        if not USE_MASKED_RECONSTRUCTION or rec_module is None:
            print("Option C disabled (USE_MASKED_RECONSTRUCTION=False); nothing to evaluate.")
            return None
        mae_model, mae_global, mae_nbr = [], [], []
        per_mod = {m: [] for m in MODALITIES}
        s_pos_all, s_neg_all, g_pos_all, g_neg_all = [], [], [], []

        for _i, (_d, _mpath) in enumerate(items[:n_cases]):
            _r = rec_recover_case(_d, seed=seed + _i)
            _dev_d = _d.to(device)
            for nt in NODE_TYPES:
                if nt not in _r["pred"]:
                    continue
                idx = _r["idx"][nt]
                pred, tgt = _r["pred"][nt], _r["target"][nt]
                mae_model.append(np.abs(pred - tgt).mean())
                for m in MODALITIES:
                    sl = MODALITY_SLICES[m]
                    per_mod[m].append(np.abs(pred[:, sl] - tgt[:, sl]).mean())

                app = _dev_d[nt].x[:, :APPEARANCE_DIM].float().cpu().numpy()
                unmasked = np.ones(app.shape[0], dtype=bool)
                unmasked[idx] = False
                # baseline (a): global mean of unmasked nodes
                gmean = app[unmasked].mean(axis=0) if unmasked.any() else np.zeros(APPEARANCE_DIM, np.float32)
                mae_global.append(np.abs(gmean[None, :] - tgt).mean())
                # baseline (b): mean over UNMASKED spatial neighbours (falls back to global mean)
                ei = _dev_d[(nt, "spatial", nt)].edge_index.cpu().numpy()
                nbr_pred = np.repeat(gmean[None, :], len(idx), axis=0)
                if ei.size:
                    src, dst = ei
                    keep = unmasked[src]
                    src_k, dst_k = src[keep], dst[keep]
                    acc = np.zeros_like(app); cnt = np.zeros(app.shape[0], np.float32)
                    np.add.at(acc, dst_k, app[src_k])
                    np.add.at(cnt, dst_k, 1.0)
                    has = cnt[idx] > 0
                    nbr_pred[has] = acc[idx][has] / cnt[idx][has, None]
                mae_nbr.append(np.abs(nbr_pred - tgt).mean())

            if "link" in _r:
                s_pos_all.append(_r["link"]["s_pos"]); s_neg_all.append(_r["link"]["s_neg"])
                g_pos_all.append(_r["link"]["g_pos"]); g_neg_all.append(_r["link"]["g_neg"])

        print("=" * 66)
        print(f"Option C recovery evaluation ({min(n_cases, len(items))} cases)")
        print("=" * 66)
        if mae_model:
            print("Masked node APPEARANCE reconstruction (mean absolute error, lower is better)")
            print(f"  model (embedding decoder)      : {np.mean(mae_model):.4f}")
            print(f"  baseline (a) global mean       : {np.mean(mae_global):.4f}")
            print(f"  baseline (b) neighbour mean    : {np.mean(mae_nbr):.4f}")
            _best_base = min(np.mean(mae_global), np.mean(mae_nbr))
            print(f"  -> model beats best baseline by: {(_best_base - np.mean(mae_model)) / max(_best_base, 1e-8) * 100:+.1f} %")
            print("  per-modality MAE: " + "  ".join(f"{m}={np.mean(v):.4f}" for m, v in per_mod.items() if v))
        if s_pos_all:
            s = np.concatenate(s_pos_all + s_neg_all)
            g = np.concatenate(g_pos_all + g_neg_all)
            lab = np.concatenate([np.ones(sum(len(a) for a in s_pos_all)),
                                  np.zeros(sum(len(a) for a in s_neg_all))])
            print("\nT1ce <-> FLAIR CORRESPONDENCE reconstruction (distance-matched hard negatives)")
            print(f"  model scorer AUC               : {_auc(s, lab):.4f}")
            print(f"  geometry-only (centroid dist)  : {_auc(g, lab):.4f}   <- must be beaten")
            print(f"  model accuracy @ logit>0       : {float(((s > 0).astype(int) == lab).mean()):.4f}")
            print(f"  positives={int(lab.sum())}  hard negatives={int((lab == 0).sum())}")
        return dict(mae_model=float(np.mean(mae_model)) if mae_model else None,
                    mae_global=float(np.mean(mae_global)) if mae_global else None,
                    mae_nbr=float(np.mean(mae_nbr)) if mae_nbr else None)


    def project_node_scalar(values, meta, nt, fill=np.nan):
        """Paint one scalar per node onto its supervoxel (reuses the cached node_map)."""
        nm = meta["node_map"][nt]
        out = np.full(nm.shape, fill, dtype=np.float32)
        valid = nm >= 0
        out[valid] = np.asarray(values, dtype=np.float32)[nm[valid]]
        return out


    def rec_visualize_recovery(item, nt="t1ce", modality="t1ce", seed=0):
        """original -> masked -> reconstructed, for one appearance channel on one slice."""
        if not USE_MASKED_RECONSTRUCTION or rec_module is None:
            print("Option C disabled (USE_MASKED_RECONSTRUCTION=False); nothing to visualise.")
            return
        d, mpath = item
        meta = load_meta(mpath)
        r = rec_recover_case(d, seed=seed)
        if nt not in r["pred"]:
            print(f"No masked {nt} nodes in this case.")
            return
        ch = MODALITY_SLICES[modality].start          # that modality's MEAN intensity channel
        idx = r["idx"][nt]
        orig = d[nt].x[:, ch].float().cpu().numpy().copy()
        masked = orig.copy(); masked[idx] = np.nan    # what the encoder was actually given
        recon = orig.copy(); recon[idx] = r["pred"][nt][:, ch]   # same channel as orig/masked

        seg_c = crop(meta["seg"], meta["lo"], meta["hi"])
        z = int(np.argmax((seg_c > 0).sum(axis=(0, 1))))
        panels = [("Original graph", orig), ("Masked graph (input)", masked), ("Reconstructed", recon)]
        vmin = float(np.nanmin(orig)); vmax = float(np.nanmax(orig))

        fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
        for ax, (name, vals) in zip(axes, panels):
            img = project_node_scalar(vals, meta, nt)[:, :, z]
            ax.imshow(img.T, cmap="magma", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\n({nt} nodes, {modality} mean)", fontsize=9)
            ax.axis("off")
        # true vs predicted for the masked nodes
        tgt = r["target"][nt][:, ch]; prd = r["pred"][nt][:, ch]
        axes[3].scatter(tgt, prd, s=3, alpha=0.25, edgecolors="none")
        _lim = [min(tgt.min(), prd.min()), max(tgt.max(), prd.max())]
        axes[3].plot(_lim, _lim, "r--", lw=1)
        _corr = float(np.corrcoef(tgt, prd)[0, 1]) if len(tgt) > 2 else float("nan")
        axes[3].set_title(f"Masked nodes: true vs reconstructed\nn={len(tgt)}  r={_corr:.3f}", fontsize=9)
        axes[3].set_xlabel("true"); axes[3].set_ylabel("reconstructed")
        fig.suptitle(f"Option C: masked graph recovery  |  {nt} partition  |  slice z={z}"
                     f"  |  {len(idx)}/{len(orig)} nodes masked", fontsize=11)
        plt.tight_layout()
        plt.show()

        if "link" in r:
            fig2, ax2 = plt.subplots(figsize=(5.2, 3.4))
            ax2.hist(r["link"]["s_neg"], bins=40, alpha=0.6, label="hard negatives", density=True)
            ax2.hist(r["link"]["s_pos"], bins=40, alpha=0.6, label="held-out positives", density=True)
            ax2.axvline(0.0, color="k", lw=1, ls="--")
            ax2.set_xlabel("correspondence score (logit)"); ax2.set_ylabel("density")
            ax2.set_title("Recovering masked T1ce <-> FLAIR relations", fontsize=10)
            ax2.legend(fontsize=8)
            plt.tight_layout()
            plt.show()


    rec_eval_summary = rec_evaluate(test_items, n_cases=8)
    if USE_MASKED_RECONSTRUCTION and rec_module is not None and test_items:
        rec_visualize_recovery(test_items[0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15. Save the trained model
    """)
    return


@app.cell
def _(
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    N_SEGMENTS,
    PERSISTENT_BASE,
    PROTOCOL_VERSION,
    RUN_CONFIG,
    SPLIT_CASE_IDS,
    SPLIT_FINGERPRINT,
    STAGE1_MODEL_SHA256,
    USE_VOXEL_REFINEMENT,
    VOXEL_BOUNDARY_WEIGHT,
    best_epoch,
    best_val_dice,
    class_weights,
    mo,
    model,
    os,
    rec_module,
    stage1_checkpoint,
    stage2_training_complete,
    test_metrics,
    test_vox_metrics,
    torch,
    val_metrics,
    val_vox_metrics,
    vox_history,
    voxel_head,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    checkpoint = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "class_weights": class_weights.detach().cpu(), "best_val_dice": best_val_dice,
        "best_epoch": best_epoch, "graph_training_history": stage1_checkpoint.get("history"),
        "voxel_training_history": vox_history,
        "voxel_head_state_dict": ({k: v.detach().cpu().clone() for k, v in voxel_head.state_dict().items()}
                                  if voxel_head is not None else None),
        "use_voxel_refinement": USE_VOXEL_REFINEMENT,
        "rec_state_dict": ({k: v.detach().cpu() for k, v in rec_module.state_dict().items()}
                           if rec_module is not None else None),
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "val_vox_metrics": val_vox_metrics if USE_VOXEL_REFINEMENT else None,
        "test_vox_metrics": test_vox_metrics if USE_VOXEL_REFINEMENT else None,
        "config_hash": CONFIG_HASH, "split": SPLIT_CASE_IDS, "run_config": RUN_CONFIG,
        "split_fingerprint": SPLIT_FINGERPRINT, "dataset_fingerprint": DATASET_FINGERPRINT,
        "stage1_sha256": STAGE1_MODEL_SHA256, "protocol": PROTOCOL_VERSION,
        "voxel_boundary_weight": VOXEL_BOUNDARY_WEIGHT,
        "gpu_training_protocol": RUN_CONFIG["voxel"],
        "gpu_runtime": getattr(voxel_head, "brats_gpu_runtime", None),
    }
    final_model_path = os.path.join(PERSISTENT_BASE, f"qos_hrgn_gpu_v4_slic{N_SEGMENTS}.pt")
    torch.save(checkpoint, final_model_path)
    print(f"Saved: {final_model_path}")
    return (final_model_path,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 16f. Ablation table driver
    """)
    return


@app.cell
def _(
    BATCH_SIZE,
    CKPT_PUSH_EVERY_EPOCHS,
    COMPACTNESS,
    DATASET_FINGERPRINT,
    DICE_WEIGHT,
    DROPOUT,
    DataLoader,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    ET_FOCAL_WEIGHT,
    ET_MIN_VOXELS,
    FOCAL_GAMMA,
    GPU_AUTOTUNE,
    GPU_MEMORY_FRACTION,
    GPU_PRECISION,
    HEADS,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    HGT_LAYERS,
    HIDDEN_DIM,
    HOPS,
    HOP_REG_WEIGHT,
    HOP_TEMPERATURE,
    INFERENCE_MAX_BATCH,
    INFERENCE_OVERLAP,
    K_MAX,
    LR,
    MODALITIES,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NUM_CLASSES,
    N_SEGMENTS,
    PART2_START_TIME,
    PERSISTENT_BASE,
    QoSHRGN,
    RUN_CONFIG,
    SEED,
    SLIC_ITERS,
    STRUCTURAL_AUX_WEIGHT,
    STRUCTURAL_DROPOUT,
    STRUCTURAL_FLAGS,
    STRUCTURAL_HIDDEN,
    STRUCTURAL_TEACHER_PROB_END,
    STRUCTURAL_TEACHER_PROB_START,
    StructuralQoSHRGN,
    TRAIN_BUDGET_HOURS,
    USE_ADAPTIVE_HOPS,
    USE_CC_POSTPROCESS,
    USE_MASKED_RECONSTRUCTION,
    USE_STRUCTURAL_REFINEMENT,
    VOXEL_BASE_CHANNELS,
    VOXEL_BOUNDARY_WEIGHT,
    VOXEL_CE_WEIGHT,
    VOXEL_DICE_WEIGHT,
    VOXEL_EFFECTIVE_BATCH_SIZE,
    VOXEL_EPOCHS,
    VOXEL_FOCAL_GAMMA,
    VOXEL_FOCAL_WEIGHT,
    VOXEL_LR,
    VOXEL_MAX_MICROBATCH,
    VOXEL_MAX_SIZE,
    VOXEL_PREFETCH_WORKERS,
    VoxelRefinementHead,
    WEIGHT_DECAY,
    build_and_save,
    class_weights,
    device,
    focal_loss_soft,
    get_case_id,
    hf_try_download,
    hf_upload_file_verified,
    hf_upload_receipt,
    load_meta,
    mo,
    postprocess_prediction_auto,
    project_node_probs,
    region_dice_loss,
    soft_cross_entropy,
    stage2_training_complete,
    structural_teacher_prob,
    test_items,
    train_items,
    val_items,
    valid_cases,
    voxel_loss,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before research experiments."))
    # Research experiments run in additional resumable molab sessions.
    RUN_ABLATIONS = False
    RESEARCH_MODE = "train"  # train, validation, then test after the frozen plan completes
    RESEARCH_SEEDS = (42, 43, 44)
    RESEARCH_PLAN_NAMES = ()  # Empty = all declared rows. Freeze BEFORE training.
    RESEARCH_JOBS = ()  # Empty = all planned rows; choose names to split work across sessions.
    RESEARCH_ACTIVE_SEEDS = ()  # Empty = all planned seeds; scheduling does not change the plan.
    RESEARCH_MAX_CASES_PER_SPLIT = 0  # Nonzero = debugging only, never a final test report.
    RESEARCH_BUDGET_HOURS = TRAIN_BUDGET_HOURS
    RESEARCH_PIPELINE_SHA256 = "20b86fe1ec92af535468743985ee759273c97c89de484a484c81262f6c5b38c2"
    from brats_experiments import run_research, experiment_plan
    RESEARCH_AVAILABLE_ROWS = experiment_plan(N_SEGMENTS, HOPS, K_MAX)
    print("Available research jobs:", [row["name"] for row in RESEARCH_AVAILABLE_ROWS])
    RESEARCH_CONTEXT = dict(
        train_items=train_items,
        val_items=val_items,
        test_items=test_items,
        SEED=SEED,
        N_SEGMENTS=N_SEGMENTS,
        HOPS=HOPS,
        DATASET_FINGERPRINT=DATASET_FINGERPRINT,
        RUN_CONFIG=RUN_CONFIG,
        class_weights=class_weights,
        EPOCHS=EPOCHS,
        VOXEL_EPOCHS=VOXEL_EPOCHS,
        EARLY_STOP_PATIENCE=EARLY_STOP_PATIENCE,
        USE_MASKED_RECONSTRUCTION=USE_MASKED_RECONSTRUCTION,
        USE_STRUCTURAL_REFINEMENT=USE_STRUCTURAL_REFINEMENT,
        USE_ADAPTIVE_HOPS=USE_ADAPTIVE_HOPS,
        HF_ENABLED=HF_ENABLED,
        HF_MODEL_REPO_ID=HF_MODEL_REPO_ID,
        HF_MODEL_REPO_TYPE=HF_MODEL_REPO_TYPE,
        hf_upload_file_verified=hf_upload_file_verified,
        hf_upload_receipt=hf_upload_receipt,
        hf_try_download=hf_try_download,
        PERSISTENT_BASE=PERSISTENT_BASE,
        NODE_TYPES=NODE_TYPES,
        valid_cases=valid_cases,
        get_case_id=get_case_id,
        build_and_save=build_and_save,
        COMPACTNESS=COMPACTNESS,
        SLIC_ITERS=SLIC_ITERS,
        NODE_FEAT_DIM=NODE_FEAT_DIM,
        HIDDEN_DIM=HIDDEN_DIM,
        NUM_CLASSES=NUM_CLASSES,
        HEADS=HEADS,
        HGT_LAYERS=HGT_LAYERS,
        DROPOUT=DROPOUT,
        K_MAX=K_MAX,
        HOP_TEMPERATURE=HOP_TEMPERATURE,
        QoSHRGN=QoSHRGN,
        STRUCTURAL_FLAGS=STRUCTURAL_FLAGS,
        StructuralQoSHRGN=StructuralQoSHRGN,
        STRUCTURAL_HIDDEN=STRUCTURAL_HIDDEN,
        STRUCTURAL_DROPOUT=STRUCTURAL_DROPOUT,
        device=device,
        soft_cross_entropy=soft_cross_entropy,
        DICE_WEIGHT=DICE_WEIGHT,
        region_dice_loss=region_dice_loss,
        ET_FOCAL_WEIGHT=ET_FOCAL_WEIGHT,
        focal_loss_soft=focal_loss_soft,
        FOCAL_GAMMA=FOCAL_GAMMA,
        DataLoader=DataLoader,
        BATCH_SIZE=BATCH_SIZE,
        structural_teacher_prob=structural_teacher_prob,
        STRUCTURAL_AUX_WEIGHT=STRUCTURAL_AUX_WEIGHT,
        HOP_REG_WEIGHT=HOP_REG_WEIGHT,
        load_meta=load_meta,
        postprocess_prediction_auto=postprocess_prediction_auto,
        project_node_probs=project_node_probs,
        LR=LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        CKPT_PUSH_EVERY_EPOCHS=CKPT_PUSH_EVERY_EPOCHS,
        VoxelRefinementHead=VoxelRefinementHead,
        MODALITIES=MODALITIES,
        VOXEL_BASE_CHANNELS=VOXEL_BASE_CHANNELS,
        VOXEL_MAX_SIZE=VOXEL_MAX_SIZE,
        voxel_loss=voxel_loss,
        VOXEL_DICE_WEIGHT=VOXEL_DICE_WEIGHT,
        VOXEL_CE_WEIGHT=VOXEL_CE_WEIGHT,
        VOXEL_FOCAL_WEIGHT=VOXEL_FOCAL_WEIGHT,
        VOXEL_FOCAL_GAMMA=VOXEL_FOCAL_GAMMA,
        VOXEL_BOUNDARY_WEIGHT=VOXEL_BOUNDARY_WEIGHT,
        GPU_PRECISION=GPU_PRECISION,
        VOXEL_MAX_MICROBATCH=VOXEL_MAX_MICROBATCH,
        INFERENCE_MAX_BATCH=INFERENCE_MAX_BATCH,
        GPU_MEMORY_FRACTION=GPU_MEMORY_FRACTION,
        GPU_AUTOTUNE=GPU_AUTOTUNE,
        VOXEL_PREFETCH_WORKERS=VOXEL_PREFETCH_WORKERS,
        VOXEL_EFFECTIVE_BATCH_SIZE=VOXEL_EFFECTIVE_BATCH_SIZE,
        INFERENCE_OVERLAP=INFERENCE_OVERLAP,
        VOXEL_LR=VOXEL_LR,
        USE_CC_POSTPROCESS=USE_CC_POSTPROCESS,
        ET_MIN_VOXELS=ET_MIN_VOXELS,
        STRUCTURAL_TEACHER_PROB_START=STRUCTURAL_TEACHER_PROB_START,
        STRUCTURAL_TEACHER_PROB_END=STRUCTURAL_TEACHER_PROB_END,
        RESEARCH_PIPELINE_SHA256=RESEARCH_PIPELINE_SHA256,
    )

    RESEARCH_SETTINGS = dict(mode=RESEARCH_MODE,seeds=RESEARCH_SEEDS,plan_names=RESEARCH_PLAN_NAMES,
        jobs=RESEARCH_JOBS,active_seeds=RESEARCH_ACTIVE_SEEDS,max_cases_per_split=RESEARCH_MAX_CASES_PER_SPLIT,
        budget_hours=RESEARCH_BUDGET_HOURS,session_start=PART2_START_TIME)
    research_status = run_research(RESEARCH_CONTEXT, RESEARCH_SETTINGS) if RUN_ABLATIONS else None
    if not RUN_ABLATIONS:
        print("Research driver idle. Use research_experiments.py for independent sessions; read RESEARCH_PROTOCOL.md.")
    return


@app.cell
def _(
    BASELINE_EPOCHS,
    BASELINE_MAX_CASES,
    BASELINE_VOXEL_EPOCHS,
    EARLY_STOP_PATIENCE,
    F,
    GATConv,
    GPU_AUTOTUNE,
    GPU_MEMORY_FRACTION,
    GPU_PRECISION,
    HIDDEN_DIM,
    HeteroConv,
    INFERENCE_MAX_BATCH,
    INFERENCE_OVERLAP,
    LR,
    MODALITIES,
    NODE_FEATURE_MODALITIES,
    NODE_TYPES,
    NUM_CLASSES,
    PERSISTENT_BASE,
    RUN_BASELINES,
    SAGEConv,
    SEED,
    SLIC_MODALITIES,
    SPLIT_CASE_IDS,
    SPLIT_FINGERPRINT,
    STAGE1_CKPT_PATH,
    STAGE2_LATEST_PATH,
    USE_AMP,
    VOXEL_BASE_CHANNELS,
    VOXEL_BOUNDARY_WEIGHT,
    VOXEL_CE_WEIGHT,
    VOXEL_DICE_WEIGHT,
    VOXEL_EFFECTIVE_BATCH_SIZE,
    VOXEL_FOCAL_GAMMA,
    VOXEL_FOCAL_WEIGHT,
    VOXEL_LR,
    VOXEL_MARGIN,
    VOXEL_MAX_MICROBATCH,
    VOXEL_MAX_SIZE,
    VOXEL_PREFETCH_WORKERS,
    VoxelRefinementHead,
    WEIGHT_DECAY,
    class_weights,
    device,
    hierarchy_loss,
    json,
    load_meta,
    make_training_patch,
    mean_region_dice,
    mo,
    model,
    nn,
    np,
    os,
    postprocess_prediction_auto,
    project_node_probs,
    random,
    reconstruct_case,
    reconstruct_case_voxel,
    segmentation_metrics,
    sliding_window_predict,
    stage2_training_complete,
    test_items,
    torch,
    train_items,
    train_voxel_experiment,
    val_items,
    voxel_head,
    voxel_loss,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # Fair baseline registry. Disabled by default because these are full retraining runs.
    from baseline_3d_unet import train_3d_unet
    BASELINE_GRAPH_KIND = "GraphSAGE"  # change to "GAT" for the alternative ordinary graph baseline
    RUN_3D_UNET_BASELINE = True
    UNET_BASE_CHANNELS = 8
    UNET_BATCH_SIZE = 1
    UNET_GRADIENT_ACCUMULATION_STEPS = VOXEL_EFFECTIVE_BATCH_SIZE
    UNET_PRECISION = "auto"  # auto prefers BF16 on supported CUDA GPUs, otherwise FP32
    UNET_CHANNELS_LAST_3D = True
    BASELINE_SUBSET_SEED = SEED + 1701
    BASELINE_EDGE_TYPES = (
        ("t1ce", "spatial", "t1ce"),
        ("flair", "spatial", "flair"),
        ("t1ce", "corresponds", "flair"),
        ("flair", "corresponds", "t1ce"),
    )

    def baseline_metric_summary(rows):
        if not rows:
            return {}
        summary = {}
        for key in rows[0]:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            finite = np.isfinite(values)
            summary[key] = {
                "mean": float(values[finite].mean()) if finite.any() else None,
                "std": float(values[finite].std()) if finite.any() else None,
                "n_valid": int(finite.sum()),
                "n_total": int(values.size),
            }
        return summary

    def baseline_mri_crop_bounds(meta, margin=VOXEL_MARGIN, max_size=VOXEL_MAX_SIZE):
        """Deterministic image-derived crop; never reads `meta['seg']`."""
        mri = np.stack([meta["vis"][m].astype(np.float32) for m in MODALITIES])
        foreground = np.any(np.abs(mri) > 1e-6, axis=0)
        coords = np.argwhere(foreground)
        if len(coords) == 0:
            return tuple(0 for _ in mri.shape[1:]), tuple(mri.shape[1:])
        lo = np.maximum(coords.min(axis=0) - margin, 0)
        hi = np.minimum(coords.max(axis=0) + margin + 1, np.asarray(foreground.shape))
        for axis in range(3):
            if hi[axis] - lo[axis] > max_size:
                center = int((lo[axis] + hi[axis]) // 2)
                lo[axis] = max(0, center - max_size // 2)
                hi[axis] = min(foreground.shape[axis], lo[axis] + max_size)
                lo[axis] = max(0, hi[axis] - max_size)
        return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


    def baseline_case_input(meta):
        x, y = make_training_patch(meta, VOXEL_MAX_SIZE)
        return torch.from_numpy(x).unsqueeze(0).float(), torch.from_numpy(y).unsqueeze(0).long(), None, None, tuple(y.shape)


    def assert_baseline_crop_label_invariance(meta):
        # Exercise the actual inference path with a small pointwise CNN.
        stripped = {k: v for k, v in meta.items() if k != "seg"}
        rng_state = torch.get_rng_state()
        try:
            probe = nn.Conv3d(len(MODALITIES), NUM_CLASSES, 1).to(device)
            original_prediction = baseline_cnn_predict(probe, meta)
            assert np.array_equal(original_prediction, baseline_cnn_predict(probe, stripped))
            assert np.array_equal(original_prediction, baseline_cnn_predict(probe, dict(meta, seg=np.zeros_like(meta["seg"]))))
        finally:
            torch.set_rng_state(rng_state)

    def run_baseline_input_smoke_test(items):
        """Execute the target-invariance check on a persisted case when available."""
        if not items:
            print("Baseline crop smoke test skipped: no persisted cases available")
            return False
        assert_baseline_crop_label_invariance(load_meta(items[0][1]))
        print("Baseline crop smoke test passed: MRI input/crop is label-invariant")
        return True

    class ThreeDUNetBaseline(nn.Module):
        """Lightweight independent 3D U-Net for four-channel BraTS volumes."""
        def __init__(self, in_channels=len(MODALITIES), num_classes=NUM_CLASSES, base_channels=8):
            super().__init__()
            self.channels_last_3d = False
            self.amp_enabled = False
            self.amp_dtype = None
            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv3d(cin, cout, 3, padding=1, bias=False),
                    nn.InstanceNorm3d(cout, affine=True),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                    nn.InstanceNorm3d(cout, affine=True),
                    nn.LeakyReLU(inplace=True),
                )
            self.enc1 = block(in_channels, base_channels)
            self.enc2 = block(base_channels, base_channels * 2)
            self.bottleneck = block(base_channels * 2, base_channels * 4)
            self.pool = nn.MaxPool3d(2)
            self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 2, stride=2)
            self.dec2 = block(base_channels * 4, base_channels * 2)
            self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, 2, stride=2)
            self.dec1 = block(base_channels * 2, base_channels)
            self.out = nn.Conv3d(base_channels, num_classes, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            b = self.bottleneck(self.pool(e2))
            d2 = self.up2(b)
            if d2.shape[2:] != e2.shape[2:]:
                d2 = F.interpolate(d2, size=e2.shape[2:], mode="trilinear", align_corners=False)
            d2 = self.dec2(torch.cat((d2, e2), dim=1))
            d1 = self.up1(d2)
            if d1.shape[2:] != e1.shape[2:]:
                d1 = F.interpolate(d1, size=e1.shape[2:], mode="trilinear", align_corners=False)
            d1 = self.dec1(torch.cat((d1, e1), dim=1))
            return self.out(d1)


    def baseline_cnn_predict(head, meta):
        pred = sliding_window_predict(head, meta, roi=VOXEL_MAX_SIZE, overlap=INFERENCE_OVERLAP)
        return postprocess_prediction_auto(pred)

    def train_cnn_only_baseline(train_set, val_set):
        torch.manual_seed(SEED)
        head = VoxelRefinementHead(len(MODALITIES), NUM_CLASSES, VOXEL_BASE_CHANNELS).to(device)
        def _prepare(value):
            (_, mpath), seed = value
            x, y = make_training_patch(load_meta(mpath), VOXEL_MAX_SIZE, rng=np.random.RandomState(int(seed)))
            tx, ty = torch.from_numpy(x).float(), torch.from_numpy(y).long()
            return (tx.pin_memory(), ty.pin_memory()) if USE_AMP else (tx, ty)
        def _loss(output, targets):
            logits, boundary = output
            return voxel_loss(logits, targets, class_weights.to(device), dice_weight=VOXEL_DICE_WEIGHT,
                              ce_weight=VOXEL_CE_WEIGHT, focal_weight=VOXEL_FOCAL_WEIGHT,
                              focal_gamma=VOXEL_FOCAL_GAMMA, boundary_logits=boundary,
                              boundary_weight=VOXEL_BOUNDARY_WEIGHT)
        def _validate(trained_head, item):
            meta = load_meta(item[1])
            return mean_region_dice(baseline_cnn_predict(trained_head, meta), meta["seg"])
        head, protocol = train_voxel_experiment(head, train_set, val_set, _prepare, _validate, _loss,
            epochs=BASELINE_VOXEL_EPOCHS, seed=SEED, learning_rate=VOXEL_LR, weight_decay=WEIGHT_DECAY,
            effective_batch=VOXEL_EFFECTIVE_BATCH_SIZE, max_microbatch=VOXEL_MAX_MICROBATCH,
            max_inference_batch=INFERENCE_MAX_BATCH, roi=VOXEL_MAX_SIZE, precision=GPU_PRECISION,
            memory_fraction=GPU_MEMORY_FRACTION, autotune=GPU_AUTOTUNE, workers=VOXEL_PREFETCH_WORKERS,
            patience=EARLY_STOP_PATIENCE)
        protocol.update(architecture="VoxelRefinementHead CNN-only", loss="patient-mean weighted CE + Dice + focal + boundary BCE",
                        parameters=sum(p.numel() for p in head.parameters()), split_fingerprint=SPLIT_FINGERPRINT)
        checkpoint_path = os.path.join(PERSISTENT_BASE, "baseline_cnn_only_gpu_v4.pt")
        protocol["checkpoint"] = checkpoint_path
        head.baseline_protocol = protocol
        torch.save({"state_dict":head.state_dict(), "protocol":protocol}, checkpoint_path)
        return head

    class OrdinaryGraphBaseline(nn.Module):
        def __init__(self, hidden_dim=HIDDEN_DIM, kind=BASELINE_GRAPH_KIND):
            super().__init__()
            conv = GATConv if kind.upper() == "GAT" else SAGEConv
            if conv is GATConv:
                first = {et: conv((-1, -1), hidden_dim, add_self_loops=False)
                         for et in BASELINE_EDGE_TYPES}
                second = {et: conv((hidden_dim, hidden_dim), NUM_CLASSES, add_self_loops=False)
                          for et in BASELINE_EDGE_TYPES}
            else:
                first = {et: conv((-1, -1), hidden_dim) for et in BASELINE_EDGE_TYPES}
                second = {et: conv((hidden_dim, hidden_dim), NUM_CLASSES)
                          for et in BASELINE_EDGE_TYPES}
            self.conv1 = HeteroConv(first, aggr="sum")
            self.conv2 = HeteroConv(second, aggr="sum")

        def forward(self, data):
            features = {nt: data[nt].x for nt in NODE_TYPES}
            features = {nt: F.relu(value)
                        for nt, value in self.conv1(features, data.edge_index_dict).items()}
            return self.conv2(features, data.edge_index_dict)

    def train_graph_baseline(train_set, val_set):
        torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
        graph_model = OrdinaryGraphBaseline().to(device)
        opt = torch.optim.AdamW(graph_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        best_state, best_score, stale, best_epoch = None, -float("inf"), 0, 0
        for epoch in range(1, BASELINE_EPOCHS + 1):
            graph_model.train()
            for data, _ in train_set:
                batch = data.clone().to(device)
                out = graph_model(batch)
                loss = sum(hierarchy_loss(out[nt], batch[nt], 1, class_weights.to(device)) for nt in NODE_TYPES) / len(NODE_TYPES)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            graph_model.eval()
            rows = []
            with torch.no_grad():
                for data, mpath in val_set:
                    meta = load_meta(mpath)
                    batch = data.clone().to(device)
                    out = graph_model(batch)
                    probs = {nt: torch.softmax(out[nt], -1).cpu().numpy() for nt in NODE_TYPES}
                    rows.append(segmentation_metrics(postprocess_prediction_auto(project_node_probs(probs, meta)), meta["seg"]))
            score = (float(np.mean([
                (row["Dice_WT"] + row["Dice_TC"] + row["Dice_ET"]) / 3
                for row in rows])) if rows else 0.0)
            if score > best_score:
                best_score, stale, best_epoch = score, 0, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in graph_model.state_dict().items()}
            else:
                stale += 1
                if stale >= 15:
                    break
        if best_state is not None:
            graph_model.load_state_dict(best_state)
        graph_model.baseline_protocol = {
            "architecture": BASELINE_GRAPH_KIND,
            "parameters": int(sum(p.numel() for p in graph_model.parameters())),
            "optimizer": "AdamW",
            "learning_rate": float(LR),
            "epoch_cap": int(BASELINE_EPOCHS),
            "early_stopping_patience": 15,
            "best_epoch": int(best_epoch),
            "loss": "weighted soft CE + region Dice + focal",
            "seed": int(SEED),
        }
        checkpoint_path = os.path.join(
            PERSISTENT_BASE, f"baseline_{BASELINE_GRAPH_KIND.lower()}_best.pt")
        graph_model.baseline_protocol["checkpoint"] = checkpoint_path
        torch.save({"state_dict": graph_model.state_dict(), "protocol": graph_model.baseline_protocol}, checkpoint_path)
        return graph_model

    def evaluate_graph_baseline(graph_model, items):
        graph_model.eval()
        rows = []
        with torch.no_grad():
            for data, mpath in items:
                meta = load_meta(mpath)
                batch = data.clone().to(device)
                out = graph_model(batch)
                probs = {nt: torch.softmax(out[nt], -1).cpu().numpy() for nt in NODE_TYPES}
                rows.append(segmentation_metrics(postprocess_prediction_auto(project_node_probs(probs, meta)), meta["seg"]))
        return rows

    def run_fair_baselines():
        if BASELINE_MAX_CASES:
            raise ValueError(
                "BASELINE_MAX_CASES is disabled for comparative reporting: "
                "retrain every compared model on the same selected split first."
            )
        parent_split = (train_items, val_items, test_items)
        baseline_train = sorted(train_items, key=lambda item: os.path.basename(item[1]))
        baseline_val = sorted(val_items, key=lambda item: os.path.basename(item[1]))
        baseline_test = sorted(test_items, key=lambda item: os.path.basename(item[1]))
        if BASELINE_MAX_CASES:
            baseline_train = sorted(random.Random(BASELINE_SUBSET_SEED).sample(baseline_train, min(BASELINE_MAX_CASES, len(baseline_train))), key=lambda item: os.path.basename(item[1]))
            baseline_val = sorted(random.Random(BASELINE_SUBSET_SEED + 1).sample(baseline_val, min(BASELINE_MAX_CASES, len(baseline_val))), key=lambda item: os.path.basename(item[1]))
            baseline_test = sorted(random.Random(BASELINE_SUBSET_SEED + 2).sample(baseline_test, min(BASELINE_MAX_CASES, len(baseline_test))), key=lambda item: os.path.basename(item[1]))
        _selected_sets = [set(os.path.basename(item[1]).replace(".meta.pt", "") for item in group)
                          for group in (baseline_train, baseline_val, baseline_test)]
        _parent_sets = [set(os.path.basename(item[1]).replace(".meta.pt", "") for item in group)
                        for group in parent_split]
        assert not (_selected_sets[0] & _selected_sets[1])
        assert not (_selected_sets[0] & _selected_sets[2])
        assert not (_selected_sets[1] & _selected_sets[2])
        assert _selected_sets[0] <= _parent_sets[0]
        assert _selected_sets[1] <= _parent_sets[1]
        assert _selected_sets[2] <= _parent_sets[2]
        run_baseline_input_smoke_test(baseline_train)
        cnn_head = train_cnn_only_baseline(baseline_train, baseline_val)
        cnn_rows = []
        for _, mpath in baseline_test:
            meta = load_meta(mpath)
            cnn_rows.append(segmentation_metrics(baseline_cnn_predict(cnn_head, meta), meta["seg"]))
        unet_head = None
        unet_protocol = {}
        unet_rows = []
        if RUN_3D_UNET_BASELINE:
            def _unet_batch(meta):
                x, y, *_ = baseline_case_input(meta)
                return x, y
            def _unet_eval(head, meta):
                return segmentation_metrics(baseline_cnn_predict(head, meta), meta["seg"])
            unet_head, unet_protocol = train_3d_unet(
                ThreeDUNetBaseline(base_channels=UNET_BASE_CHANNELS), baseline_train, baseline_val,
                load_meta=load_meta, make_batch=_unet_batch, evaluate=_unet_eval,
                loss_fn=lambda logits, targets: voxel_loss(logits, targets, class_weights.to(device),
                    dice_weight=VOXEL_DICE_WEIGHT, ce_weight=VOXEL_CE_WEIGHT, focal_weight=VOXEL_FOCAL_WEIGHT,
                    focal_gamma=VOXEL_FOCAL_GAMMA), device=device, epochs=BASELINE_VOXEL_EPOCHS,
                loss_name="Dice + weighted CE + focal (no boundary head)",
                learning_rate=VOXEL_LR, weight_decay=WEIGHT_DECAY, patience=15,
                checkpoint_path=os.path.join(PERSISTENT_BASE, "baseline_3d_unet_best.pt"),
                seed=SEED, precision=UNET_PRECISION,
                channels_last_3d=UNET_CHANNELS_LAST_3D,
                batch_size=UNET_BATCH_SIZE,
                gradient_accumulation_steps=UNET_GRADIENT_ACCUMULATION_STEPS)
            unet_protocol.update({
                "split_fingerprint": SPLIT_FINGERPRINT,
                "selected_case_ids": {
                    "train": sorted(_selected_sets[0]),
                    "val": sorted(_selected_sets[1]),
                    "test": sorted(_selected_sets[2]),
                },
                "hd95_units": "mm",
            })
            for _, mpath in baseline_test:
                meta = load_meta(mpath)
                unet_rows.append(_unet_eval(unet_head, meta))
        graph_baseline = train_graph_baseline(baseline_train, baseline_val)
        graph_rows = evaluate_graph_baseline(graph_baseline, baseline_test)
        hgt_graph_rows, hgt_voxel_rows = [], []
        for data, mpath in baseline_test:
            meta = load_meta(mpath)
            hgt_graph_rows.append(segmentation_metrics(reconstruct_case(model, data, meta), meta["seg"]))
            hgt_voxel_rows.append(segmentation_metrics(
                reconstruct_case_voxel(model, voxel_head, data, meta), meta["seg"]))
        registry = {
            "CNN only": baseline_metric_summary(cnn_rows),
            BASELINE_GRAPH_KIND: baseline_metric_summary(graph_rows),
            "HGT graph only": baseline_metric_summary(hgt_graph_rows),
            "HGT graph + CNN": baseline_metric_summary(hgt_voxel_rows),
            "model_protocols": {
                "CNN only": getattr(cnn_head, "baseline_protocol", {}),
                BASELINE_GRAPH_KIND: getattr(graph_baseline, "baseline_protocol", {}),
                "HGT graph only": {"architecture": "QoS-HRGN HGT", "checkpoint": STAGE1_CKPT_PATH},
                "HGT graph + CNN": {"architecture": "QoS-HRGN HGT + CNN", "checkpoint": STAGE2_LATEST_PATH},
            },
            "protocol": {
                "split_fingerprint": SPLIT_FINGERPRINT,
                "case_ids": SPLIT_CASE_IDS,
                "selected_case_ids": {
                    "train": sorted(_selected_sets[0]),
                    "val": sorted(_selected_sets[1]),
                    "test": sorted(_selected_sets[2]),
                },
                "parent_split_fingerprint": SPLIT_FINGERPRINT,
                "subset_seed": BASELINE_SUBSET_SEED,
                "hd95_units": "mm",
                "slic_modalities": list(SLIC_MODALITIES),
                "node_feature_modalities": list(NODE_FEATURE_MODALITIES),
                "external_slots": ["3D U-Net", "nnU-Net", "SegResNet"],
                "implemented_baselines": ["CNN only", BASELINE_GRAPH_KIND, "HGT graph only", "HGT graph + CNN"],
                "external_slot_note": "3D U-Net, nnU-Net, and SegResNet are future work; populate only with runs on this exact split and protocol.",
            },
        }
        if RUN_3D_UNET_BASELINE:
            registry["3D U-Net"] = baseline_metric_summary(unet_rows)
            registry["model_protocols"]["3D U-Net"] = unet_protocol
            registry["protocol"]["external_slots"].remove("3D U-Net")
            registry["protocol"]["implemented_baselines"].insert(1, "3D U-Net")
            registry["protocol"]["external_slot_note"] = "nnU-Net and SegResNet are future work; populate only with runs on this exact split and protocol."
        with open(os.path.join(PERSISTENT_BASE, "fair_baseline_registry.json"), "w") as fh:
            json.dump(registry, fh, indent=2)
        baseline_names = ["CNN only"]
        if RUN_3D_UNET_BASELINE:
            baseline_names.append("3D U-Net")
        baseline_names.extend([BASELINE_GRAPH_KIND, "HGT graph only", "HGT graph + CNN"])
        for name in baseline_names:
            summary = registry[name]
            print(name)
            for metric, values in summary.items():
                mean = "n/a" if values["mean"] is None else f"{values['mean']:.4f}"
                std = "n/a" if values["std"] is None else f"{values['std']:.4f}"
                print(f"  {metric}: {mean} +/- {std} (n={values['n_valid']}/{values['n_total']})")
        return registry

    fair_baseline_results = run_fair_baselines() if RUN_BASELINES else None
    if not RUN_BASELINES:
        print("Fair baselines idle (RUN_BASELINES=False); no empirical scores are claimed.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 15b. Persist the final checkpoint

    Colab's local disk doesn't survive the session ending. The Part-1 handoff
    checkpoint only had the graph model; this uploads the FINAL checkpoint
    (graph model + trained voxel head) to the same HF model repo, under its own
    filename, so the end-to-end result of both notebooks isn't lost when this
    session closes.
    """)
    return


@app.cell
def _(
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    PART2_START_TIME,
    SESSION_HARD_CAP_HOURS,
    TRAIN_BUDGET_HOURS,
    VOXEL_BOUNDARY_WEIGHT,
    final_model_path,
    hf_upload_file_verified,
    mo,
    os,
    stage2_training_complete,
    time,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    FINAL_REMOTE_NAME = os.path.basename(final_model_path)
    if not stage2_training_complete:
        print("Stage 2 is paused; final checkpoint upload skipped")
    elif HF_ENABLED:
        hf_upload_file_verified(final_model_path, FINAL_REMOTE_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                f"final checkpoint, boundary_weight={VOXEL_BOUNDARY_WEIGHT}")
    else:
        print(f"HF disabled -- final checkpoint stays local-disk only ({final_model_path})")
    _elapsed_h = (time.time() - PART2_START_TIME) / 3600
    print(f"Part 2 total elapsed: {_elapsed_h:.2f}h (train budget {TRAIN_BUDGET_HOURS:.1f}h, hard cap {SESSION_HARD_CAP_HOURS:.1f}h)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16. What is implemented, and how to run the ablations

    ### Backbone (established)
    - 3D SLIC supervoxels, two modality-specific partitions of the same brain.
    - PyG `HeteroData` with `t1ce` / `flair` node types, spatial and overlap-based cross-modal relations.
    - PyG `HGTConv` with residual + LayerNorm.

    ### Proposed component
    - Adaptive propagation weights conditioned on node embeddings, edge attributes (distance, contact/overlap), feature similarity, cross-modal relation and prediction uncertainty; gate-normalised aggregation; `HOPS` stages.
    - Pathology-aware prediction on fractional labels with a per-patient volume-weighted Dice term that equals voxel Dice.

    ### Ablations to report (one config change each, same cache)
    | Question | Change |
    |---|---|
    | Does adaptive propagation help over HGT alone? | `HOPS = 0` |
    | Does the heterogeneous cross-modal relation help? | build with `MIN_OVERLAP = 1.01` (no cross edges) |
    | Does resolution matter? | oracle sweep (Section 6) and/or `N_SEGMENTS = 4000` |
    | Do the soft labels matter? | replace `y_frac` by one-hot `y` in `hierarchy_loss` |

    ### Remaining known limitations
    - Node-constant predictions cannot beat the oracle ceiling; Saueressig et al. gained ~2 % with a shallow voxel CNN refinement on top of the projected logits — a natural next step.
    - Per-case z-scoring; dataset-level normalisation may help cross-site consistency.\n\nThe voxel table is extended by the + boundary refinement (`VOXEL_BOUNDARY_WEIGHT`) row; run the ablation driver cell for the complete comparison.\n
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16b. Option B: experiment summary
    """)
    return


@app.cell
def _(
    STRUCTURAL_FLAGS,
    USE_STRUCTURAL_REFINEMENT,
    best_epoch,
    best_val_dice,
    device,
    fin_mean,
    init_mean,
    mo,
    model,
    stage2_training_complete,
    torch,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    # ============================================================
    # Option B: experiment summary
    # ------------------------------------------------------------
    # RESEARCH INTEGRITY: this reflects one run with this seed/config. Do not
    # treat a positive delta here as a proven result -- compare against the
    # HOPS=0 / no-cross-modal-edge / one-hot-label ablations already documented
    # in the "What is implemented" section, and ideally repeat with a different
    # seed, before claiming Option B improves performance.
    _optb_param_count = sum(p.numel() for p in model.parameters()) / 1e6
    _optb_gpu_peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0

    print("=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Structural refinement enabled : {USE_STRUCTURAL_REFINEMENT}")
    if USE_STRUCTURAL_REFINEMENT:
        print(f"Enabled descriptors           : {[k for k, v in STRUCTURAL_FLAGS.items() if v]}")
    print(f"Parameters                    : {_optb_param_count:.2f} M")
    print(f"GPU peak memory                : {_optb_gpu_peak:.2f} GB")
    print(f"Best epoch / val patient-Dice  : {best_epoch} / {best_val_dice:.4f}")
    print()
    print(f"{'Region':<8}{'Initial':>10}{'Refined':>10}{'Delta':>10}")
    for _k, _name in zip(["Dice_WT", "Dice_TC", "Dice_ET"], ["WT", "TC", "ET"]):
        print(f"{_name:<8}{init_mean[_k]:>10.4f}{fin_mean[_k]:>10.4f}{fin_mean[_k] - init_mean[_k]:>+10.4f}")
    print()
    print("This is a single run -- not yet a validated result. Compare against the")
    print("existing ablation table in the final markdown section before drawing conclusions.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16c. Option C: the ablation to report

    | Run | `USE_MASKED_RECONSTRUCTION` | `LAMBDA_REC` | What it isolates |
    |---|---|---|---|
    | Baseline | `False` | - | QoS-HRGN + segmentation only |
    | + masked reconstruction | `True` | `0.3` | the proposed auxiliary objective |
    | lambda sweep | `True` | `0.1 / 0.3 / 1.0` | sensitivity; large lambda should start hurting Dice |
    | feature target only | `True`, `REC_USE_CORRESPONDENCE=False` | `0.3` | value of node-feature masking alone |
    | relation target only | `True`, `REC_USE_FEATURE=False` | `0.3` | value of correspondence masking alone |
    | + spatial edges | `True`, `REC_USE_SPATIAL_EDGE=True` | `0.3` | expected to add ~nothing (report as negative result) |

    Report WT / TC / ET test Dice for each, plus the recovery numbers from section 14c.

    **What can honestly be claimed.** The contribution is an auxiliary *heterogeneous* masked-graph
    reconstruction objective: appearance masking that can only be solved through the graph, and a
    correspondence-relation task made non-trivial by distance-matched hard negatives. Whether it
    improves Dice is an empirical question this ablation answers -- a null result is still a legitimate
    finding, and the recovery diagnostics in 14c stand on their own as evidence that the embeddings
    encode recoverable heterogeneous structure.

    **What must not be claimed.** That reconstruction improves segmentation, before the baseline and the
    `LAMBDA_REC` sweep have been run with the same seed and split. Masked autoencoding is a
    *regulariser*; on 258 training patients with a strong supervised signal it may well be neutral.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16d. Voxel refinement head: the ablation to report

    | Run | `USE_VOXEL_REFINEMENT` | `ET_FOCAL_WEIGHT` | `USE_CC_POSTPROCESS` | What it isolates |
    |---|---|---|---|---|
    | Baseline (v3) | `False` | 0 | `False` | QoS-HRGN + segmentation only (original) |
    | + focal loss | `False` | 0.5 | `False` | focal loss helps the graph model close the gap to oracle |
    | + CC post-proc | `False` | 0 | `True` | connected-component ET removal |
    | + voxel head | `True` | 0.5 | `True` | full pipeline: graph + voxel refinement |\n| + boundary refinement | `True` | 0.5 | `True` | + boundary refinement (`VOXEL_BOUNDARY_WEIGHT`) |

    **Why ET 0.8 is achievable now.** The 15k supervoxel oracle ceiling for ET is
    0.7243 — a perfect node classifier cannot reach 0.8 because ET and NCR/NET are
    mixed within supervoxels. The voxel refinement head breaks this ceiling by
    discriminating at voxel resolution using T1ce intensity. Within a mixed
    supervoxel, the enhancing tumour region has higher T1ce intensity than necrosis,
    and the 3D U-Net can learn this local pattern to split them.

    **What to report.**
    - Graph-only vs graph+voxel WT/TC/ET test Dice (the `evaluate_items_voxel` output).
    - Per-patient ET delta to show which cases benefit most.
    - The oracle ceiling alongside both, to show the voxel head exceeds it.
    - The focal loss ablation (graph-only with and without focal) to isolate that contribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 16e. Adaptive-hop propagation ablation

    The proposed model uses one shared recurrent propagation layer and learns a node-wise soft distribution over $k=0,\ldots,K_{max}$. This isolates propagation distance from layer-specific parameter count.

    | Run | `USE_ADAPTIVE_HOPS` | `HOPS` / `K_MAX` | Purpose |
    |---|---:|---:|---|
    | HGT only | `False` | `HOPS=0` | no adaptive propagation |
    | Fixed propagation | `False` | `HOPS=2` | original QoS-HRGN baseline |
    | Adaptive depth | `True` | `K_MAX=4` | node-wise learned receptive field |
    | No hop cost | `True` | `K_MAX=4`, `HOP_REG_WEIGHT=0` | tests whether the expected-hop penalty matters |
    | Full model | `True` | `K_MAX=4`, `HOP_REG_WEIGHT=0.002` | uncertainty-conditioned adaptive depth |

    Report WT/TC/ET Dice, parameter count, runtime, mean effective hop, mean $\beta_k$, and effective-hop distributions by node type and tissue class. Adaptive hop selection learns propagation depth over a fixed graph; it should not be described as learning or rewiring graph topology.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 17. Exact modality Shapley explanations

    Saueressig et al. used DeepSHAP to study how T1, T1CE, T2, and FLAIR contribute to each tissue prediction. Here the four modalities are treated as four cooperative-game players, making it practical to calculate **exact grouped Shapley values over all 16 modality coalitions** rather than a neural approximation. Missing modality groups are replaced by a 500-node training-background mean; geometry and graph topology remain unchanged. All nodes in a patient graph are perturbed simultaneously because a GNN node cannot be explained independently of its neighbors.

    The resulting plots follow the paper's class × modality × bright/dark organization. Positive values support the full-model predicted class; negative values oppose it. This explains modality evidence, while the learned edge gates and adaptive hops in the next section explain graph-structural evidence.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    MODALITIES,
    NODE_TYPES,
    NUM_CLASSES,
    QUANTILES,
    RUN_REGION_XAI,
    SEED,
    USE_STRUCTURAL_REFINEMENT,
    device,
    math,
    mo,
    model,
    np,
    plt,
    stage2_training_complete,
    test_items,
    torch,
    train_items,
):
    mo.stop(not RUN_REGION_XAI, mo.md("Optional graph-region XAI is disabled; enable RUN_REGION_XAI in its own session."))
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    from itertools import combinations as _xai_combinations
    _XAI_GROUP_WIDTH = 2 + len(QUANTILES)
    _XAI_MODALITY_SLICES = {modality: slice(i * _XAI_GROUP_WIDTH, (i + 1) * _XAI_GROUP_WIDTH) for i, modality in enumerate(MODALITIES)}

    def build_xai_background(items, sample_nodes=500, seed=SEED + 701):
        """Randomly sample training nodes while preserving empirical label prevalence."""
        rng = np.random.default_rng(seed)
        background = {}
        for nt in NODE_TYPES:
            sizes = np.asarray([data[nt].x.size(0) for data, _ in items], dtype=np.int64)
            cumulative = np.cumsum(sizes)
            total = int(cumulative[-1])
            chosen = rng.choice(total, size=min(sample_nodes, total), replace=False)
            vectors, labels = ([], [])
            for global_index in chosen:
                graph_index = int(np.searchsorted(cumulative, global_index, side='right'))
                previous = 0 if graph_index == 0 else int(cumulative[graph_index - 1])
                local_index = int(global_index - previous)
                vectors.append(items[graph_index][0][nt].x[local_index].float())
                labels.append(int(items[graph_index][0][nt].y[local_index]))
            samples = torch.stack(vectors)
            background[nt] = {'mean': samples.mean(dim=0), 'samples': samples, 'labels': np.asarray(labels), 'label_fraction': np.bincount(labels, minlength=NUM_CLASSES) / len(labels)}
        return background

    @torch.no_grad()
    def xai_predict_probabilities(graph):
        model.eval()
        graph = graph.to(device)
        if USE_STRUCTURAL_REFINEMENT:
            _, logits, _ = model(graph, teacher_prob=0.0)
        else:
            logits, _ = model(graph)
        return {nt: torch.softmax(logits[nt].float(), dim=-1).cpu() for nt in NODE_TYPES}

    @torch.no_grad()
    def exact_modality_shapley(item, background, max_nodes_per_class=150, seed=SEED + 702):
        """Exact four-player Shapley attribution for each node's full-model predicted class."""
        data, _ = item
        full_prob = xai_predict_probabilities(data.clone())
        full_class = {nt: full_prob[nt].argmax(dim=1) for nt in NODE_TYPES}
        coalition_outputs = {}
        region_coalitions = {}
        modality_count = len(MODALITIES)
        for mask in range(1 << modality_count):
            perturbed = data.clone()
            for nt in NODE_TYPES:
                x = perturbed[nt].x.clone()
                baseline = background[nt]['mean'].to(x.device, dtype=x.dtype)
                for modality_index, modality in enumerate(MODALITIES):
                    if not mask & 1 << modality_index:
                        feature_slice = _XAI_MODALITY_SLICES[modality]
                        x[:, feature_slice] = baseline[feature_slice]
                perturbed[nt].x = x
            probabilities = xai_predict_probabilities(perturbed)
            region_coalitions[mask] = {nt: torch.stack((1-probabilities[nt][:,0],
                probabilities[nt][:,1]+probabilities[nt][:,3], probabilities[nt][:,3]),dim=1) for nt in NODE_TYPES}
            coalition_outputs[mask] = {nt: probabilities[nt].gather(1, full_class[nt].unsqueeze(1)).squeeze(1) for nt in NODE_TYPES}
        factorial = math.factorial
        results = {}
        rng = np.random.default_rng(seed)
        for nt in NODE_TYPES:
            shapley = torch.zeros(data[nt].x.size(0), modality_count)
            region_phi = torch.zeros(data[nt].x.size(0), 3, modality_count)
            for modality_index in range(modality_count):
                others = [j for j in range(modality_count) if j != modality_index]
                for subset_size in range(modality_count):
                    coefficient = factorial(subset_size) * factorial(modality_count - subset_size - 1) / factorial(modality_count)
                    for subset in _xai_combinations(others, subset_size):
                        mask = sum((1 << j for j in subset))
                        region_phi[:,:,modality_index] += coefficient * (region_coalitions[mask | 1 << modality_index][nt] - region_coalitions[mask][nt])
                        shapley[:, modality_index] = shapley[:, modality_index] + coefficient * (coalition_outputs[mask | 1 << modality_index][nt] - coalition_outputs[mask][nt])
            selected = []
            predicted = full_class[nt].numpy()
            for cls in range(NUM_CLASSES):
                candidates = np.flatnonzero(predicted == cls)
                if len(candidates) > max_nodes_per_class:
                    candidates = rng.choice(candidates, max_nodes_per_class, replace=False)
                selected.extend(candidates.tolist())
            selected = np.asarray(sorted(selected), dtype=np.int64)
            node_x = data[nt].x.float().cpu().numpy()
            brightness = np.zeros((len(selected), modality_count), dtype=bool)
            thresholds = []
            for modality_index, modality in enumerate(MODALITIES):
                first_feature = _XAI_MODALITY_SLICES[modality].start
                threshold = float(np.quantile(background[nt]['samples'][:, first_feature].cpu().numpy(), 0.85))
                thresholds.append(threshold)
                brightness[:, modality_index] = node_x[selected, first_feature] >= threshold
            results[nt] = {'node_index': selected, 'predicted_class': predicted[selected], 'true_class': data[nt].y.cpu().numpy()[selected], 'confidence': full_prob[nt].max(dim=1).values.numpy()[selected], 'shapley': shapley.numpy()[selected], 'bright': brightness, 'brightness_threshold': np.asarray(thresholds), 'efficiency_error': float((shapley.sum(dim=1) - (coalition_outputs[(1 << modality_count) - 1][nt] - coalition_outputs[0][nt])).abs().max())}
            region_weights = data[nt].vol.detach().cpu().float()[:,None] * torch.stack((
                1-data[nt].y_frac[:,0], data[nt].y_frac[:,1]+data[nt].y_frac[:,3], data[nt].y_frac[:,3]),dim=1).cpu()
            region_sums = (region_phi*region_weights[:,:,None]).sum(dim=0)
            results[nt]["region_attribution_sum"] = region_sums.numpy()
            results[nt]["region_voxel_weight"] = region_weights.sum(0).numpy()
            results[nt]["region_efficiency_error"] = float((region_phi.sum(-1) - (region_coalitions[15][nt]-region_coalitions[0][nt])).abs().max())
        xai_predict_probabilities(data.clone())
        return results

    def plot_modality_shapley(results, title='Exact modality Shapley values across heterogeneous nodes'):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
        axes = axes.ravel()
        dark_color, bright_color = ('#1769aa', '#d9eaf7')
        for cls, axis in enumerate(axes):
            pooled_values = [[[], []] for _ in MODALITIES]
            for result in results.values():
                class_mask = result['predicted_class'] == cls
                for modality_index in range(len(MODALITIES)):
                    values = result['shapley'][class_mask, modality_index]
                    bright = result['bright'][class_mask, modality_index]
                    pooled_values[modality_index][0].extend(values[~bright].tolist())
                    pooled_values[modality_index][1].extend(values[bright].tolist())
            for modality_index, (dark_values, bright_values) in enumerate(pooled_values):
                for offset, values, color, name in [(-0.16, dark_values, dark_color, 'Dark'), (0.16, bright_values, bright_color, 'Bright')]:
                    if len(values) >= 2:
                        violin = axis.violinplot(values, positions=[modality_index + offset], widths=0.28, showmeans=False, showmedians=True, showextrema=False)
                        for body in violin['bodies']:
                            body.set_facecolor(color)
                            body.set_edgecolor('black')
                            body.set_alpha(0.85)
                        violin['cmedians'].set_color('black')
                    elif values:
                        axis.scatter([modality_index + offset], values, color=color, edgecolor='black', s=18)
            axis.axhline(0, color='gray', linewidth=0.8)
            axis.set_xticks(range(len(MODALITIES)), [m.upper() for m in MODALITIES])
            axis.set_title(f'{CLASS_NAMES[cls]} (predicted label {cls})')
            axis.set_ylabel('Shapley contribution to predicted-class probability')
            axis.grid(axis='y', alpha=0.2)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=dark_color, markeredgecolor='black', label='Dark', markersize=8), plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=bright_color, markeredgecolor='black', label='Bright (top 15%)', markersize=8)]
        fig.legend(handles=handles, loc='upper right', title='Image intensity')
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, 0, 0.96, 0.96))
        plt.show()
        print('Mean absolute modality contribution:')
        for nt, result in results.items():
            means = np.abs(result['shapley']).mean(axis=0)
            print(f'  {nt}: ' + '  '.join((f'{m}={v:.5f}' for m, v in zip(MODALITIES, means))))
            print(f"       Shapley efficiency max error={result['efficiency_error']:.2e}")
    xai_background = build_xai_background(train_items, sample_nodes=500)
    print('XAI background label fractions:')
    for _xai_nt in NODE_TYPES:
        print(f'  {_xai_nt}: ' + '  '.join((f'{name}={fraction:.3f}' for name, fraction in zip(CLASS_NAMES, xai_background[_xai_nt]['label_fraction']))))
    modality_shapley = exact_modality_shapley(test_items[0], xai_background) if test_items else None
    if modality_shapley is not None:
        plot_modality_shapley(modality_shapley)
    return exact_modality_shapley, xai_background, xai_predict_probabilities


@app.cell
def _(
    MODALITIES,
    PERSISTENT_BASE,
    SEED,
    exact_modality_shapley,
    mo,
    np,
    os,
    random,
    stage2_training_complete,
    test_items,
    write_json_atomic,
    xai_background,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics."))
    SHAPLEY_COHORT_MAX_PATIENTS = 20
    _shapley_cohort = random.Random(SEED + 703).sample(test_items, min(SHAPLEY_COHORT_MAX_PATIENTS,len(test_items)))
    _region_rows = {name:[] for name in ("WT","TC","ET")}
    _region_patient_rows = []
    for _item in _shapley_cohort:
        _results = exact_modality_shapley(_item,xai_background)
        _patient = {"case_id":os.path.basename(_item[1]).replace(".meta.pt","")}
        for _i,_region in enumerate(("WT","TC","ET")):
            _mass = sum(r["region_voxel_weight"][_i] for r in _results.values())
            _values = sum(r["region_attribution_sum"][_i] for r in _results.values())/_mass if _mass>0 else None
            _patient[_region] = _values.tolist() if _values is not None else None
            if _values is not None:
                _region_rows[_region].append(_values)
        _patient["efficiency_error"] = max(r["region_efficiency_error"] for r in _results.values())
        _region_patient_rows.append(_patient)
    SHAPLEY_MODALITY_IMPORTANCE = {
        "scope":"Graph branch only; fixed graph topology and training-mean feature replacement; excludes CNN and SLIC effects",
        "target":"WT/TC/ET probability, weighted by fractional target-region voxel mass within each patient",
        "n_patients":len(_shapley_cohort),"patients":_region_patient_rows,
        "per_region":{region:{"n_present":len(rows),"mean":np.mean(rows,axis=0).tolist() if rows else None,
                             "std":np.std(rows,axis=0).tolist() if rows else None} for region,rows in _region_rows.items()},
        "modality_order":list(MODALITIES),"inference":"descriptive; no uncorrected directional significance claims"}
    SHAPLEY_MODALITY_IMPORTANCE_PATH = os.path.join(PERSISTENT_BASE,"shapley_modality_importance.json")
    write_json_atomic(SHAPLEY_MODALITY_IMPORTANCE_PATH,SHAPLEY_MODALITY_IMPORTANCE)
    print(SHAPLEY_MODALITY_IMPORTANCE["scope"])
    print(SHAPLEY_MODALITY_IMPORTANCE["per_region"])
    return (SHAPLEY_MODALITY_IMPORTANCE,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 18. Labelled heterogeneous explanation graph

    A full patient graph contains roughly 30,000 nodes and is not visually interpretable. This view therefore selects a clinically relevant target node and displays its bounded heterogeneous neighborhood. Node color is predicted tissue, marker shape is modality-specific node type, node size is confidence, each label contains node ID/class/probability/effective hop, edge color is relation type, and edge opacity/width/numeric labels show the model's learned adaptive gate $\alpha$. Gate weights are averaged over recurrent propagation steps. This is a faithful visualization of signals actually used by adaptive propagation, not a force-directed decorative graph.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    HOPS,
    K_MAX,
    NODE_TYPES,
    NUM_CLASSES,
    base_model,
    mo,
    np,
    plt,
    stage2_training_complete,
    test_items,
    torch,
    xai_predict_probabilities,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    from matplotlib.collections import LineCollection as _XAILineCollection
    from matplotlib.lines import Line2D as _XAILine2D
    _XAI_CLASS_COLORS = np.asarray(['#8b8b8b', '#355cce', '#e53935', '#f5c400'])
    _XAI_RELATION_COLORS = {'t1ce-spatial-t1ce': '#8e44ad', 'flair-spatial-flair': '#16a085', 't1ce-corresponds-flair': '#e67e22'}
    _XAI_DISPLAY_EDGE_TYPES = (('t1ce', 'spatial', 't1ce'), ('flair', 'spatial', 'flair'), ('t1ce', 'corresponds', 'flair'))

    def _xai_relation_name(edge_type):
        return '-'.join(edge_type)

    @torch.no_grad()
    def explain_local_hetero_graph(item, target_type='t1ce', target_class=3, neighborhood_hops=2, max_nodes=90, label_nodes=24, display_axes=(0, 1)):
        """Draw a bounded, labelled heterogeneous neighborhood around one target node."""
        data, _ = item
        probabilities = xai_predict_probabilities(data.clone())
        hop_info = base_model.last_hop_info
        gate_history = base_model.last_edge_gates
        predicted = {nt: probabilities[nt].argmax(dim=1).numpy() for nt in NODE_TYPES}
        confidence = {nt: probabilities[nt].max(dim=1).values.numpy() for nt in NODE_TYPES}
        candidates = np.flatnonzero(predicted[target_type] == target_class)
        if len(candidates):
            target_index = int(candidates[np.argmax(confidence[target_type][candidates])])
        else:
            target_index = int(probabilities[target_type][:, target_class].argmax())
            print(f'No predicted class-{target_class} {target_type} node; using highest class probability instead.')
        target = (target_type, target_index)
        edge_arrays = {edge_type: data[edge_type].edge_index.cpu().numpy() for edge_type in _XAI_DISPLAY_EDGE_TYPES}
        distance = {target: 0}
        frontier = {target}
        for depth in range(1, neighborhood_hops + 1):
            discovered = set()
            for edge_type, edge_index in edge_arrays.items():
                src_type, _, dst_type = edge_type
                src, dst = edge_index
                frontier_src = np.asarray([index for nt, index in frontier if nt == src_type], dtype=np.int64)
                frontier_dst = np.asarray([index for nt, index in frontier if nt == dst_type], dtype=np.int64)
                if len(frontier_src):
                    for index in dst[np.isin(src, frontier_src)]:
                        discovered.add((dst_type, int(index)))
                if len(frontier_dst):
                    for index in src[np.isin(dst, frontier_dst)]:
                        discovered.add((src_type, int(index)))
            discovered = discovered - set(distance)
            for node in discovered:
                distance[node] = depth
            frontier = discovered
            if not frontier:
                break
        nodes = list(distance)
        if len(nodes) > max_nodes:
            nodes = sorted(nodes, key=lambda node: (distance[node], -float(confidence[node[0]][node[1]]), node[0], node[1]))[:max_nodes]
        node_set = set(nodes)
        positions = {}
        for nt, index in nodes:
            centroid = data[nt].x[index, 28:31].cpu().numpy()
            offset = np.asarray([0.004, -0.004]) if nt == 'flair' else np.asarray([-0.004, 0.004])
            positions[nt, index] = centroid[list(display_axes)] + offset
        averaged_gates = {}
        if gate_history:
            for edge_type in _XAI_DISPLAY_EDGE_TYPES:
                available = [step[edge_type].cpu().numpy() for step in gate_history if edge_type in step]
                if available:
                    averaged_gates[edge_type] = np.mean(np.stack(available), axis=0)
        visible_edges = []
        for edge_type, edge_index in edge_arrays.items():
            src_type, _, dst_type = edge_type
            weights = averaged_gates.get(edge_type)
            for edge_id, (src, dst) in enumerate(edge_index.T):
                source, destination = ((src_type, int(src)), (dst_type, int(dst)))
                if source in node_set and destination in node_set and (source != destination):
                    weight = float(weights[edge_id]) if weights is not None else 0.5
                    visible_edges.append((source, destination, edge_type, edge_id, weight))
        deduplicated = {}
        for source, destination, edge_type, edge_id, weight in visible_edges:
            canonical = tuple(sorted((source, destination)))
            key = (edge_type, canonical)
            deduplicated.setdefault(key, []).append((edge_id, weight))
        visible_edges = [(canonical[0], canonical[1], edge_type, values[0][0], float(np.mean([weight for _, weight in values]))) for (edge_type, canonical), values in deduplicated.items()]
        fig, axis = plt.subplots(figsize=(15, 12))
        for edge_type in _XAI_DISPLAY_EDGE_TYPES:
            relation_edges = [edge for edge in visible_edges if edge[2] == edge_type]
            if not relation_edges:
                continue
            segments = [[positions[edge[0]], positions[edge[1]]] for edge in relation_edges]
            weights = np.asarray([edge[4] for edge in relation_edges])
            collection = _XAILineCollection(segments, colors=_XAI_RELATION_COLORS[_xai_relation_name(edge_type)], linewidths=0.4 + 3.0 * weights, alpha=float(np.clip(0.18 + 0.65 * weights.mean(), 0.2, 0.85)), zorder=1)
            axis.add_collection(collection)
        for nt, marker in [('t1ce', 'o'), ('flair', 's')]:
            typed_nodes = [node for node in nodes if node[0] == nt]
            if not typed_nodes:  # Spatial edge_index commonly stores both directions. Collapse each displayed
                continue  # anatomical connection and average its directional gates for a clean figure.
            xy = np.stack([positions[node] for node in typed_nodes])
            cls = np.asarray([predicted[nt][node[1]] for node in typed_nodes])
            conf = np.asarray([confidence[nt][node[1]] for node in typed_nodes])
            effective = np.asarray([float(hop_info[nt]['effective_hop'][node[1]].cpu()) if hop_info else float(HOPS) for node in typed_nodes])
            axis.scatter(xy[:, 0], xy[:, 1], c=_XAI_CLASS_COLORS[cls], marker=marker, s=35 + 120 * conf, edgecolors=plt.cm.viridis(effective / max(K_MAX, 1)), linewidths=1.8, alpha=0.95, zorder=3)
        ranked_nodes = sorted(nodes, key=lambda node: (node != target, distance[node], -float(confidence[node[0]][node[1]])))[:label_nodes]
        for nt, index in ranked_nodes:
            cls = int(predicted[nt][index])
            eff = float(hop_info[nt]['effective_hop'][index].cpu()) if hop_info else float(HOPS)
            text = f'{nt[0].upper()}{index}\n{CLASS_NAMES[cls]} p={confidence[nt][index]:.2f}\nk={eff:.2f}'
            axis.annotate(text, positions[nt, index], xytext=(4, 4), textcoords='offset points', fontsize=6.5, bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='#555', alpha=0.78), zorder=5)
        for source, destination, edge_type, _, weight in sorted(visible_edges, key=lambda edge: edge[4], reverse=True)[:min(20, len(visible_edges))]:
            midpoint = (positions[source] + positions[destination]) / 2
            axis.text(midpoint[0], midpoint[1], f'{weight:.2f}', fontsize=6, color='#222', bbox=dict(boxstyle='round,pad=0.08', fc='white', ec='none', alpha=0.72), zorder=4)
        target_xy = positions[target]
        axis.scatter([target_xy[0]], [target_xy[1]], s=420, facecolors='none', edgecolors='black', linewidths=3, zorder=6)
        class_handles = [_XAILine2D([0], [0], marker='o', linestyle='', markerfacecolor=color, markeredgecolor='black', label=name, markersize=9) for color, name in zip(_XAI_CLASS_COLORS, CLASS_NAMES)]
        type_handles = [_XAILine2D([0], [0], marker='o', linestyle='', color='black', label='T1ce node'), _XAILine2D([0], [0], marker='s', linestyle='', color='black', label='FLAIR node')]
        relation_handles = [_XAILine2D([0], [0], color=color, linewidth=2.5, label=name) for name, color in _XAI_RELATION_COLORS.items()]
        legend_one = axis.legend(handles=class_handles + type_handles, loc='upper left', title='Nodes')
        axis.add_artist(legend_one)
        axis.legend(handles=relation_handles, loc='upper right', title='Relations')
        axis.set(title=f'Heterogeneous explanation graph: target {target_type}[{target_index}] → {CLASS_NAMES[predicted[target_type][target_index]]}\nnode fill=prediction | border=effective hop | edge width/label=learned gate α', xlabel=f'centroid axis {display_axes[0]}', ylabel=f'centroid axis {display_axes[1]}', aspect='equal')
        axis.grid(alpha=0.12)
        fig.tight_layout()
        plt.show()
        relation_summary = {}
        print(f'Target: {target_type}[{target_index}]  predicted={CLASS_NAMES[predicted[target_type][target_index]]} confidence={confidence[target_type][target_index]:.3f}')
        print(f'Displayed {len(nodes)} nodes and {len(visible_edges)} edges ({neighborhood_hops}-hop neighborhood).')
        for edge_type in _XAI_DISPLAY_EDGE_TYPES:
            values = np.asarray([edge[4] for edge in visible_edges if edge[2] == edge_type])
            if len(values):
                relation_summary[_xai_relation_name(edge_type)] = {'count': int(len(values)), 'mean_gate': float(values.mean()), 'max_gate': float(values.max()), 'std_gate': float(values.std())}
                print(f'  {_xai_relation_name(edge_type)}: edges={len(values)}  mean α={values.mean():.3f}  max α={values.max():.3f}')
        return {'target': target, 'nodes': nodes, 'edges': visible_edges, 'relation_summary': relation_summary}

    @torch.no_grad()
    def global_graph_explanation_summary(item):
        """Aggregate learned relation gates and effective hops by class and correctness."""
        data, _ = item
        probabilities = xai_predict_probabilities(data.clone())
        predicted = {nt: probabilities[nt].argmax(dim=1).numpy() for nt in NODE_TYPES}
        truth = {nt: data[nt].y.cpu().numpy() for nt in NODE_TYPES}
        gates = base_model.last_edge_gates
        hop_info = base_model.last_hop_info
        relation_names = [_xai_relation_name(et) for et in base_model.metadata[1]]
        gate_matrix = np.full((len(relation_names), NUM_CLASSES), np.nan, dtype=np.float32)
        for relation_index, edge_type in enumerate(base_model.metadata[1]):
            if not gates:
                continue
            step_values = [step[edge_type].cpu().numpy() for step in gates if edge_type in step]
            if not step_values:
                continue
            alpha = np.mean(np.stack(step_values), axis=0)
            destinations = data[edge_type].edge_index[1].cpu().numpy()
            destination_class = predicted[edge_type[2]][destinations]
            for cls in range(NUM_CLASSES):
                mask = destination_class == cls
                if mask.any():
                    gate_matrix[relation_index, cls] = float(alpha[mask].mean())
        hop_matrix = np.full((len(NODE_TYPES) * 2, NUM_CLASSES), np.nan, dtype=np.float32)
        hop_rows = []
        for node_type_index, nt in enumerate(NODE_TYPES):
            effective = hop_info[nt]['effective_hop'].cpu().numpy() if hop_info else np.full(len(predicted[nt]), HOPS)
            correct = predicted[nt] == truth[nt]
            for correctness_index, (name, correctness) in enumerate([('correct', True), ('incorrect', False)]):
                row = node_type_index * 2 + correctness_index
                hop_rows.append(f'{nt} {name}')
                for cls in range(NUM_CLASSES):
                    mask = (predicted[nt] == cls) & (correct == correctness)
                    if mask.any():
                        hop_matrix[row, cls] = float(effective[mask].mean())
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        gate_image = axes[0].imshow(gate_matrix, cmap='magma', aspect='auto', vmin=0, vmax=1)
        axes[0].set(xticks=range(NUM_CLASSES), xticklabels=CLASS_NAMES, yticks=range(len(relation_names)), yticklabels=relation_names, title='Mean learned edge gate α by destination class')
        for row in range(gate_matrix.shape[0]):
            for col in range(gate_matrix.shape[1]):
                if np.isfinite(gate_matrix[row, col]):
                    axes[0].text(col, row, f'{gate_matrix[row, col]:.2f}', ha='center', va='center', color='white')
        fig.colorbar(gate_image, ax=axes[0], label='mean α')
        hop_image = axes[1].imshow(hop_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=K_MAX)
        axes[1].set(xticks=range(NUM_CLASSES), xticklabels=CLASS_NAMES, yticks=range(len(hop_rows)), yticklabels=hop_rows, title='Mean effective hop by class and correctness')
        for row in range(hop_matrix.shape[0]):
            for col in range(hop_matrix.shape[1]):
                if np.isfinite(hop_matrix[row, col]):
                    axes[1].text(col, row, f'{hop_matrix[row, col]:.2f}', ha='center', va='center', color='white')
        fig.colorbar(hop_image, ax=axes[1], label='effective hop')
        fig.tight_layout()
        plt.show()
        return {'relation_names': relation_names, 'gate_by_class': gate_matrix, 'hop_rows': hop_rows, 'effective_hop_by_class': hop_matrix}
    local_graph_explanation = explain_local_hetero_graph(test_items[0], target_type='t1ce', target_class=3, neighborhood_hops=2, max_nodes=90, label_nodes=24) if test_items else None
    graph_explanation_summary = global_graph_explanation_summary(test_items[0]) if test_items else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 19. How to interpret and report the explanations

    The three explanation channels answer different questions and should be reported together:

    1. **Exact modality Shapley:** which MRI modality supports or opposes a node's predicted tissue class? This is a coalition-based perturbation explanation adapted from Saueressig et al. Exactness applies to the four modality groups and the chosen background-replacement game—not to every individual intensity statistic.
    2. **Adaptive edge gate $\alpha$:** which spatial or cross-modal messages did the propagation module permit strongly? This is a native model quantity, but a high gate is not by itself causal proof; it should be interpreted alongside modality ablation and prediction changes.
    3. **Effective hop $k_{eff}$:** how much graph context did a node select? This measures learned receptive-field depth over the fixed graph, not learned topology.

    Recommended figures and tables:

    - Four-panel Shapley violin plot by predicted class, modality, and bright/dark intensity, matching the paper's presentation.
    - One correctly predicted ET, one incorrect ET, and one edema local explanation graph per test cohort.
    - Mean $|\phi|$ by modality and class.
    - Mean gate $\alpha$ by relation, class, and correct/incorrect prediction.
    - Effective-hop distributions by node type, class, uncertainty quartile, and correctness.
    - Sanity check: randomize model weights or labels and verify that explanations change materially.

    The local graph is deliberately bounded. Rendering the full approximately 30k-node heterogeneous graph would hide rather than explain its structure. Numeric edge labels are therefore restricted to the 20 strongest displayed edges.
    """)
    return


@app.cell
def _(
    GRAPH_BUILD_STATS_PATH,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    KHOP_BY_REGION,
    NODE_TYPES,
    PERSISTENT_BASE,
    SHAPLEY_MODALITY_IMPORTANCE,
    device,
    graph_items,
    hf_upload_file_verified,
    json,
    load_meta,
    mo,
    model,
    np,
    os,
    rec_module,
    reconstruct_case,
    reconstruct_case_voxel,
    stage1_checkpoint,
    stage2_training_complete,
    test_items,
    time,
    torch,
    vox_history,
    voxel_head,
):
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 before diagnostics or final export."))
    def _benchmark_summary(values):
        return {"mean": float(np.mean(values)) if values else None,
                "std": float(np.std(values)) if values else None,
                "n": len(values)}


    def _benchmark_epoch_mean(hist):
        _values = [row["epoch_seconds"] for row in hist
                   if isinstance(row, dict) and isinstance(row.get("epoch_seconds"), (int, float))]
        return float(np.mean(_values)) if _values else None

    _graph_params = sum(param.numel() for param in model.parameters())
    if rec_module is not None:
        _graph_params += sum(param.numel() for param in rec_module.parameters())
    _voxel_params = sum(param.numel() for param in voxel_head.parameters()) if voxel_head is not None else 0
    _graph_stats = "n/a (cache pulled from Hub; no local build this run)"
    if os.path.exists(GRAPH_BUILD_STATS_PATH):
        with open(GRAPH_BUILD_STATS_PATH) as _fh:
            _build_rows = json.load(_fh)
        _case_total = sum(row["cases"] for row in _build_rows)
        _seconds_total = sum(row["seconds"] for row in _build_rows)
        _graph_stats = {
            "total_cases": _case_total,
            "total_wall_seconds": _seconds_total,
            "wall_seconds_per_case": _seconds_total / _case_total if _case_total else None,
            "worker_seconds_per_case": (sum(row["seconds"] * row["n_jobs"] for row in _build_rows) / _case_total
                                        if _case_total else None),
        }
    _graph_only_times, _graph_voxel_times = [], []
    for _data, _mpath in test_items:
        _meta = load_meta(_mpath)
        if device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = time.perf_counter()
        reconstruct_case(model, _data, _meta, postprocess=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        _graph_only_times.append(time.perf_counter() - _t0)
        if voxel_head is not None:
            if device.type == "cuda":
                torch.cuda.synchronize()
            _t0 = time.perf_counter()
            reconstruct_case_voxel(model, voxel_head, _data, _meta, postprocess=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _graph_voxel_times.append(time.perf_counter() - _t0)
    _node_values = {nt: [] for nt in NODE_TYPES}
    _node_values["total"] = []
    _edge_values = {}
    _edge_values["total"] = []
    for _data, _ in graph_items:
        _node_total = 0
        for _nt in NODE_TYPES:
            _count = int(_data[_nt].x.size(0))
            _node_values[_nt].append(_count)
            _node_total += _count
        _node_values["total"].append(_node_total)
        _edge_total = 0
        for _edge_type in _data.edge_types:
            _edge_name = "__".join(_edge_type)
            _count = int(_data[_edge_type].edge_index.size(1))
            _edge_values.setdefault(_edge_name, []).append(_count)
            _edge_total += _count
        _edge_values["total"].append(_edge_total)
    BENCHMARKS = {
        "params_graph_model_M": _graph_params / 1e6,
        "params_voxel_head_M": _voxel_params / 1e6,
        "params_total_M": (_graph_params + _voxel_params) / 1e6,
        "peak_gpu_gb_part2": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "peak_gpu_gb_part1": stage1_checkpoint.get("peak_gpu_gb"),
        "graph_build": _graph_stats,
        "stage1_seconds_per_epoch_mean": _benchmark_epoch_mean(stage1_checkpoint.get("history") or []),
        "stage2_seconds_per_epoch_mean": _benchmark_epoch_mean(vox_history),
        "inference_seconds_per_patient": {
            "graph_only": _benchmark_summary(_graph_only_times),
            "graph_plus_voxel": _benchmark_summary(_graph_voxel_times),
        },
        "graph_size": {
            "nodes": {key: _benchmark_summary(value) for key, value in _node_values.items()},
            "edges": {key: _benchmark_summary(value) for key, value in _edge_values.items()},
        },
        "khop_by_region": KHOP_BY_REGION,
        "shapley_modality_importance": SHAPLEY_MODALITY_IMPORTANCE,
    }
    BENCHMARKS_PATH = os.path.join(PERSISTENT_BASE, "benchmarks.json")
    with open(BENCHMARKS_PATH, "w") as _fh:
        json.dump(BENCHMARKS, _fh, indent=2)
    print("Benchmark".ljust(42) + "Value")
    for _key, _value in BENCHMARKS.items():
        print(f"{_key.ljust(42)}{str(_value)}")
    if HF_ENABLED and stage2_training_complete:
        hf_upload_file_verified(BENCHMARKS_PATH, os.path.basename(BENCHMARKS_PATH),
                                HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE, "Part 2 benchmarks")
    return


if __name__ == "__main__":
    app.run()
