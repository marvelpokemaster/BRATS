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
    # QoS-HRGN v2 -- PART 1 of 2: graph model training

    Split into two Colab sessions because a single 12-hour Colab runtime is not
    enough headroom for graph-model training + voxel refinement + evaluation +
    XAI back to back.

    **This notebook (Part 1):** data/graph-cache setup through graph-model
    training and its evaluation (original sections 1-13's graph-level metrics),
    then a checkpoint is pushed to Hugging Face Hub for Part 2 to pick up.
    Target: well under 6h on an RTX Pro 6000 session.

    **Part 2 (separate notebook/session):** downloads that checkpoint, trains the
    voxel refinement head, runs the rest of the evaluation, diagnostics and XAI.

    Graph-cache transfer uses a single ZIP archive. Set `HF_TOKEN` in the environment (or use the stored Hugging Face login); rolling training checkpoints are saved as `stage1_latest.pt` for resumable sessions.

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
    from torch_geometric.nn import HGTConv
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
    EXPECTED_CASES = 1251
    CASE_EXCLUSIONS = {}  # Add only verified canonical IDs with documented reasons.
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
    return (
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
        SEED,
        SLIC_ITERS,
        SLIC_MODALITIES,
        USE_ADAPTIVE_HOPS,
        USE_AMP,
        USE_CC_POSTPROCESS,
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
        datetime,
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
    PART1_START_TIME = _time_mod.time()
    SESSION_HARD_CAP_HOURS = 12.0
    BUILD_BUDGET_HOURS = 7.0
    TRAIN_BUDGET_HOURS = 10.5
    CKPT_PUSH_EVERY_EPOCHS = 5
    RESUME_STAGE1 = True
    EARLY_STOP_PATIENCE = 15
    print(f"[Part 1] Started. Build budget {BUILD_BUDGET_HOURS:.1f}h, train budget {TRAIN_BUDGET_HOURS:.1f}h (hard cap {SESSION_HARD_CAP_HOURS:.0f}h).")
    return (
        BUILD_BUDGET_HOURS,
        CKPT_PUSH_EVERY_EPOCHS,
        EARLY_STOP_PATIENCE,
        PART1_START_TIME,
        RESUME_STAGE1,
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

    # Individual descriptor toggles, so ablations can show WHICH structural
    # signal (if any) actually helps.
    STRUCTURAL_USE_NEIGHBOR_STATS = True    # tumour-neighbour / ET-neighbour fraction
    STRUCTURAL_USE_DISTANCE = True          # hop-distance to nearest predicted ET / nearest predicted BG
    STRUCTURAL_USE_COMPONENT_SIZE = True    # size of the predicted-tumour connected component
    STRUCTURAL_USE_ENCLOSURE = True         # ET-neighbour fraction minus BG-neighbour fraction
    STRUCTURAL_USE_CROSS_MODAL = True       # T1ce<->FLAIR correspondence fan-out

    STRUCTURAL_HIDDEN = 32     # width of the structural MLP -- kept small on purpose
    STRUCTURAL_DROPOUT = 0.1
    STRUCTURAL_AUX_WEIGHT = 0.3   # loss = hierarchy_loss(final) + STRUCTURAL_AUX_WEIGHT * hierarchy_loss(initial)
                                   # keeps the base model's OWN initial predictions well-calibrated,
                                   # since the structural descriptors are derived from them, and avoids
                                   # turning this into "a second, independent classifier."

    # Scheduled sampling for structural context (the exposure-bias fix). Early in
    # training, structural descriptors are computed mostly from ground-truth
    # labels (a clean signal to bootstrap the refinement module). As training
    # progresses we shift towards the model's OWN predicted labels, so the module
    # learns to correct genuinely noisy structure rather than just to trust
    # already-correct GT structure. GT is NEVER used for structural context
    # outside of training -- validation/test always call with teacher_prob=0.0.
    STRUCTURAL_TEACHER_PROB_START = 1.0
    STRUCTURAL_TEACHER_PROB_END = 0.0

    def structural_teacher_prob(epoch, total_epochs=EPOCHS):
        """Linear decay from STRUCTURAL_TEACHER_PROB_START to _END over training."""
        if total_epochs <= 1:
            return STRUCTURAL_TEACHER_PROB_END
        frac = min(max((epoch - 1) / (total_epochs - 1), 0.0), 1.0)
        return STRUCTURAL_TEACHER_PROB_START + (STRUCTURAL_TEACHER_PROB_END - STRUCTURAL_TEACHER_PROB_START) * frac

    STRUCTURAL_FLAGS = dict(
        neighbor_stats=STRUCTURAL_USE_NEIGHBOR_STATS,
        distance=STRUCTURAL_USE_DISTANCE,
        component_size=STRUCTURAL_USE_COMPONENT_SIZE,
        enclosure=STRUCTURAL_USE_ENCLOSURE,
        cross_modal=STRUCTURAL_USE_CROSS_MODAL,
    )

    def structural_descriptor_dim(flags):
        dim = 0
        if flags["neighbor_stats"]:
            dim += 2   # tumour-neighbour fraction, ET-neighbour fraction
        if flags["distance"]:
            dim += 2   # hop-distance to nearest ET, hop-distance to nearest BG
        if flags["component_size"]:
            dim += 1   # log1p(tumour connected-component size), normalised
        if flags["enclosure"]:
            dim += 1   # ET-neighbour fraction minus BG-neighbour fraction
        if flags["cross_modal"]:
            dim += 1   # normalised cross-modal correspondence fan-out
        return dim

    print(f"[Option B] USE_STRUCTURAL_REFINEMENT={USE_STRUCTURAL_REFINEMENT}  "
          f"structural_dim={structural_descriptor_dim(STRUCTURAL_FLAGS)}  flags={STRUCTURAL_FLAGS}")
    return (
        STRUCTURAL_AUX_WEIGHT,
        STRUCTURAL_DROPOUT,
        STRUCTURAL_FLAGS,
        STRUCTURAL_HIDDEN,
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
        REC_DECODER_HIDDEN,
        REC_EDGE_MASK_RATE,
        REC_FEAT_LOSS,
        REC_FLAGS,
        REC_MASK_RATE,
        REC_NEG_PER_POS,
        REC_SCE_GAMMA,
        REC_SEPARATE_CLEAN_PASS,
        REC_WARMUP_EPOCHS,
        USE_MASKED_RECONSTRUCTION,
        rec_lambda,
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
    return CONFIG_HASH, IDENTITY_CONFIG, RUN_CONFIG, config_diff, config_hash


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
    # Remove unpatched duplicates from the main tar so the official patches are used
    all_nii = [p for p in all_nii if not ("/BraTS2021_00495/BraTS2021_00495_" in p.replace("\\", "/")) and not ("/BraTS2021_00621/BraTS2021_00621_" in p.replace("\\", "/"))]
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
        STAGE1_LATEST_NAME,
        STAGE1_LATEST_PATH,
        STAGE1_MANIFEST_NAME,
        STAGE1_REMOTE_NAME,
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
        audit_dataset,
        bounded_prefetch,
        crop,
        crop_bounds,
        extract_verified_zip,
        fmt_metrics,
        load_case,
        load_meta,
        mean_region_dice,
        postprocess_prediction_auto,
        segmentation_metrics,
        sliding_window_predict,
        summarize_metric_rows,
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
        return mean, np.sqrt(var)

    def _label_quantiles(lab, vals, K, qs):
        order = np.lexsort((vals, lab))            # sort by label, then value
        lab_s, vals_s = lab[order], vals[order]
        cnt = np.bincount(lab_s, minlength=K)
        starts = np.concatenate(([0], np.cumsum(cnt)[:-1]))
        out = np.empty((K, len(qs)), dtype=np.float64)
        for qi, q in enumerate(qs):
            pos = starts + np.floor(q * np.maximum(cnt - 1, 0)).astype(np.int64)
            out[:, qi] = vals_s[np.minimum(pos, len(vals_s) - 1)]
        return out

    def slic_node_map(vol_c, brain_c, n_segments, compactness, max_iter):
        """Run SLIC inside the brain mask; return an int32 array of node indices (-1 outside the brain)."""
        segments = slic(
            vol_c, n_segments=n_segments, compactness=compactness, max_num_iter=max_iter,
            start_label=1, mask=brain_c, channel_axis=None,
        )
        ids = np.unique(segments[brain_c])
        lut = np.full(int(segments.max()) + 1, -1, dtype=np.int32)
        lut[ids] = np.arange(len(ids), dtype=np.int32)
        node_map = lut[segments]
        node_map[~brain_c] = -1
        return node_map, int(len(ids))

    def node_table(node_map, brain_c, vols_c, seg_c):
        lab = node_map[brain_c].astype(np.int64)
        K = int(lab.max()) + 1
        cnt = np.bincount(lab, minlength=K).astype(np.float64)

        feats = []
        for m in NODE_FEATURE_MODALITIES:
            v = vols_c[m][brain_c].astype(np.float64)
            mean, std = _label_mean_std(lab, v, K)
            feats += [mean[:, None], std[:, None], _label_quantiles(lab, v, K, QUANTILES)]

        xyz = np.argwhere(brain_c).astype(np.float64)          # same C-order as node_map[brain_c]
        cent = np.stack([np.bincount(lab, weights=xyz[:, a], minlength=K) / cnt for a in range(3)], axis=1)
        cent_n = cent / np.asarray(brain_c.shape, dtype=np.float64)
        vol_frac = cnt / cnt.sum()
        feats += [cent_n, np.log(vol_frac * K)[:, None]]        # log relative size (0 = average node)
        x = np.concatenate(feats, axis=1).astype(np.float32)

        cls = seg_c[brain_c].astype(np.int64)
        y_frac = np.bincount(lab * NUM_CLASSES + cls, minlength=K * NUM_CLASSES).reshape(K, NUM_CLASSES)
        y_frac = (y_frac / cnt[:, None]).astype(np.float32)
        return dict(x=x, y_frac=y_frac, vol=vol_frac.astype(np.float32), cent=cent.astype(np.float32), cnt=cnt, K=K)

    def spatial_edges(node_map, cent, K):
        keys = []
        for axis in range(3):
            a_sl = [slice(None)] * 3
            b_sl = [slice(None)] * 3
            a_sl[axis] = slice(0, -1)
            b_sl[axis] = slice(1, None)
            a, b = node_map[tuple(a_sl)], node_map[tuple(b_sl)]
            valid = (a >= 0) & (b >= 0) & (a != b)
            u, v = a[valid].astype(np.int64), b[valid].astype(np.int64)
            keys.append(u * K + v)
            keys.append(v * K + u)
        keys, faces = np.unique(np.concatenate(keys), return_counts=True)
        u, v = keys // K, keys % K
        dist = np.linalg.norm(cent[u] - cent[v], axis=1)
        attr = np.stack([dist / (dist.mean() + 1e-8), faces / (faces.mean() + 1e-8)], axis=1).astype(np.float32)
        return np.stack([u, v]).astype(np.int64), attr

    def overlap_edges(map_a, map_b, brain_c, tab_a, tab_b, min_overlap=MIN_OVERLAP):
        la = map_a[brain_c].astype(np.int64)
        lb = map_b[brain_c].astype(np.int64)
        Kb = tab_b["K"]
        keys, n_shared = np.unique(la * Kb + lb, return_counts=True)
        i, j = keys // Kb, keys % Kb
        overlap = n_shared / np.minimum(tab_a["cnt"][i], tab_b["cnt"][j])

        def _best(group):                       # mark the strongest partner of every node in `group`
            order = np.lexsort((-n_shared, group))
            first = np.ones(len(group), dtype=bool)
            first[1:] = group[order][1:] != group[order][:-1]
            mark = np.zeros(len(group), dtype=bool)
            mark[order[first]] = True
            return mark

        keep = (overlap >= min_overlap) | _best(i) | _best(j)
        i, j, overlap = i[keep], j[keep], overlap[keep]
        dist = np.linalg.norm(tab_a["cent"][i] - tab_b["cent"][j], axis=1)
        attr = np.stack([dist / (dist.mean() + 1e-8), overlap], axis=1).astype(np.float32)
        return np.stack([i, j]).astype(np.int64), attr

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
            ei, ea = spatial_edges(node_map, tab["cent"], K)
            parts[nt] = dict(node_map=node_map, tab=tab, ei=ei, ea=ea)

        cross_ei, cross_ea = overlap_edges(
            parts["t1ce"]["node_map"], parts["flair"]["node_map"], brain_c,
            parts["t1ce"]["tab"], parts["flair"]["tab"],
        )

        data = HeteroData()
        for nt in NODE_TYPES:
            tab = parts[nt]["tab"]
            data[nt].x = torch.from_numpy(tab["x"])
            data[nt].y_frac = torch.from_numpy(tab["y_frac"])
            data[nt].y = torch.from_numpy(tab["y_frac"].argmax(axis=1).astype(np.int64))
            data[nt].vol = torch.from_numpy(tab["vol"])
            data[nt, "spatial", nt].edge_index = torch.from_numpy(parts[nt]["ei"])
            data[nt, "spatial", nt].edge_attr = torch.from_numpy(parts[nt]["ea"])
        data["t1ce", "corresponds", "flair"].edge_index = torch.from_numpy(cross_ei)
        data["t1ce", "corresponds", "flair"].edge_attr = torch.from_numpy(cross_ea)
        data["flair", "corresponds", "t1ce"].edge_index = torch.from_numpy(cross_ei[::-1].copy())
        data["flair", "corresponds", "t1ce"].edge_attr = torch.from_numpy(cross_ea.copy())

        meta = {
            "seg": seg.astype(np.uint8),
            "lo": lo, "hi": hi,
            "crop_shape": tuple(brain_c.shape),
            "original_shape": tuple(seg.shape),
            "node_map": {nt: parts[nt]["node_map"].astype(np.int32) for nt in NODE_TYPES},
            "vis": {m: vols_c[m].astype(np.float16) for m in MODALITIES},  # all 4 modalities for voxel head
            "slic_modalities": list(SLIC_MODALITIES),
            "node_feature_modalities": list(NODE_FEATURE_MODALITIES),
        }
        return data, meta

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
    load_meta,
    mean_region_dice,
    np,
    postprocess_prediction_auto,
    segmentation_metrics,
    torch,
):
    def project_node_probs(node_probs, meta):
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
        pred_c = acc.argmax(axis=-1).astype(np.uint8)
        pred_c[~valid] = 0
        pred = np.zeros(tuple(meta["original_shape"]), dtype=np.uint8)
        lo, hi = meta["lo"], meta["hi"]
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
        return segmentation_metrics(oracle_prediction(data, meta, soft=soft), meta["seg"])

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
    def graph_validation_dice(graph_model, items):
        values = []
        for data, meta_path in items:
            meta = load_meta(meta_path)
            values.append(mean_region_dice(reconstruct_case(graph_model, data, meta, postprocess=True), meta["seg"]))
        if not values:
            raise RuntimeError("Validation split is empty")
        return float(np.mean(values))

    return (
        graph_validation_dice,
        oracle_metrics,
        predict_node_probs,
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
                   boundary_logits=None, boundary_weight=0.0):
        """Dice + ce_weight*CE (+ focal_weight*Focal) (+ boundary_weight*BoundaryBCE) over the 4-class voxel problem.

        logits: (B, 4, X, Y, Z), seg: (B, X, Y, Z) with values 0-3,
        boundary_logits: (B, 1, X, Y, Z) raw logits of the boundary head.
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

    return


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

    return HeteroMaskedReconstructor, mask_hetero_graph, reconstruction_loss


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

    _STRUCT_DIST_SENTINEL = 16.0   # finite "far / not present in this graph" hop-distance

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
        src_row = sp.csr_matrix(
            (np.ones(seed_idx.size, dtype=np.float32), (np.zeros(seed_idx.size, dtype=np.int64), seed_idx)),
            shape=(1, n),
        )
        aug = sp.vstack([
            sp.hstack([csr, src_row.T]),
            sp.hstack([src_row, sp.csr_matrix((1, 1), dtype=np.float32)]),
        ]).tocsr()
        d = shortest_path(aug, method="D", directed=False, unweighted=True, indices=n)
        d = d[:n] - 1.0                     # subtract the virtual source -> seed hop
        d[~np.isfinite(d)] = sentinel
        d = np.clip(d, 0.0, sentinel)
        return d.astype(np.float32)

    def _tumour_component_sizes(csr, tumour_mask_np):
        """Size of the connected component (within the SPATIAL graph, restricted
        to predicted-tumour nodes) that each tumour node belongs to. Non-tumour
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
        if flags["neighbor_stats"]:
            tumour_nbr = np.asarray(csr @ tumour_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            et_nbr = np.asarray(csr @ et_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            parts += [tumour_nbr, et_nbr]

        if flags["distance"]:
            dist_et = _multi_source_hop_distance(csr, et_mask) / _STRUCT_DIST_SENTINEL
            dist_bg = _multi_source_hop_distance(csr, bg_mask) / _STRUCT_DIST_SENTINEL
            parts += [dist_et, dist_bg]

        if flags["component_size"]:
            comp = _tumour_component_sizes(csr, tumour_mask)
            parts += [np.log1p(comp) / np.log1p(max(num_nodes, 2))]   # normalised, roughly in [0, 1]

        if flags["enclosure"]:
            # Local enclosure/boundary score, EXPLICITLY defined as:
            #   ET-neighbour fraction minus BG-neighbour fraction, in [-1, 1].
            #   +1 => every spatial neighbour is predicted ET  (deep interior of an ET region)
            #   -1 => every spatial neighbour is predicted BG  (isolated / far from any tumour)
            #    0 => mixed / boundary neighbourhood, or ambiguous
            # This is a MEASURED graph property, not an assumption that ET always
            # encloses anything -- the refinement module decides whether/how to use it.
            et_nbr_e = et_nbr if et_nbr is not None else (
                np.asarray(csr @ et_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0))
            bg_nbr_e = np.asarray(csr @ bg_mask.astype(np.float32)).reshape(-1) / np.maximum(deg, 1.0)
            parts += [et_nbr_e - bg_nbr_e]

        if flags["cross_modal"]:
            fanout = np.bincount(cross_src_index[0].detach().cpu().numpy(), minlength=num_nodes).astype(np.float32)
            fanout = fanout[:num_nodes]
            fanout = fanout / max(float(fanout.mean()), 1e-6)
            parts += [fanout]

        if not parts:
            return torch.zeros(num_nodes, 0, device=device)
        s = np.stack(parts, axis=1).astype(np.float32)
        return torch.from_numpy(s).to(device)


    class StructuralRefinementHead(nn.Module):
        """Per-node-type refinement head for Option B.

        Combines the base model's node embedding (h), its initial class
        probabilities (p), and the structural descriptor vector (s) computed
        from the temporary predicted tumour graph. Outputs a RESIDUAL correction
        to the initial logits -- the final linear layer is zero-initialised, so
        at the start of training final_logits == initial_logits exactly, and the
        module only learns to deviate where structural evidence helps.
        """
        def __init__(self, hidden_dim, num_classes, structural_dim, refine_hidden=32, dropout=0.1):
            super().__init__()
            self.structural_dim = structural_dim
            if structural_dim > 0:
                self.structural_mlp = nn.Sequential(
                    nn.Linear(structural_dim, refine_hidden), nn.ReLU(),
                    nn.Linear(refine_hidden, refine_hidden), nn.ReLU(),
                )
                combine_in = hidden_dim + num_classes + refine_hidden
            else:
                self.structural_mlp = None
                combine_in = hidden_dim + num_classes
            self.combine = nn.Sequential(
                nn.Linear(combine_in, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            )
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
        def __init__(self, base_model, hidden_dim, num_classes, flags=STRUCTURAL_FLAGS,
                     refine_hidden=STRUCTURAL_HIDDEN, dropout=STRUCTURAL_DROPOUT):
            super().__init__()
            self.base_model = base_model
            self.flags = flags
            self.structural_dim = structural_descriptor_dim(flags)
            self.refine = nn.ModuleDict({
                nt: StructuralRefinementHead(hidden_dim, num_classes, self.structural_dim, refine_hidden, dropout)
                for nt in NODE_TYPES
            })

        def forward(self, data, teacher_prob=0.0):
            initial_logits, x_dict = self.base_model(data)
            final_logits = {}
            for nt in NODE_TYPES:
                other_nt = [t for t in NODE_TYPES if t != nt][0]
                edge_index = data[nt, "spatial", nt].edge_index
                num_nodes = data[nt].x.size(0)
                dev = data[nt].x.device

                init_prob = torch.softmax(initial_logits[nt].float(), dim=-1)
                pred_class = init_prob.detach().argmax(dim=-1)

                if self.training and teacher_prob > 0:
                    # Scheduled sampling: mix GT-derived and model-predicted
                    # structure during TRAINING ONLY. Never at eval/test.
                    gt_class = data[nt].y
                    use_gt = torch.rand(num_nodes, device=dev) < teacher_prob
                    struct_class = torch.where(use_gt, gt_class, pred_class)
                else:
                    struct_class = pred_class   # eval/test: predictions only -- no GT leakage.

                cross_src_index = data[nt, "corresponds", other_nt].edge_index
                s = compute_structural_descriptors(edge_index, num_nodes, struct_class, cross_src_index, self.flags, dev)
                h = x_dict[nt]
                final_logits[nt] = self.refine[nt](h, init_prob, s, initial_logits[nt])
            return initial_logits, final_logits, x_dict

    return (StructuralQoSHRGN,)


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

    return hierarchy_loss, node_region_dice


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
    PART1_START_TIME,
    Parallel,
    SLIC_ITERS,
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
    GRAPH_PULL_RECEIPT = None
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
    _todo = missing_cases()
    _built_any = False
    if _todo:
        _cpu_budget = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 4)
        _n_jobs = min(_cpu_budget, len(_todo))
        _chunk_size = max(1, _n_jobs * 4)
        _last_chunk_hours = 0.0
        for _chunk_start in range(0, len(_todo), _chunk_size):
            _elapsed_h = (time.time() - PART1_START_TIME) / 3600
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
    if _built_any:
        hf_push_graph_cache_zip()
    if not graph_build_complete:
        if _built_any:
            print(f"partial archive pushed ({len(missing_cases())} graphs still missing); re-run Part 1 to finish")
        else:
            print(f"re-run Part 1 to finish the remaining {len(missing_cases())} graphs; nothing pushed")

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
    return ensure_graph_cache_handoff, graph_items


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 11. Patient-level train / validation / test split and class weights

    Split by **patient**. Class weights come from *voxel* fractions (`Σ vol · y_frac`) rather than node counts, so the priority-labelling distortion of v1 cannot happen. Square-root inverse frequency keeps them moderate; the Dice term handles the rest of the imbalance.
    """)
    return


@app.cell
def _(
    CLASS_NAMES,
    NODE_TYPES,
    NUM_CLASSES,
    SEED,
    graph_items,
    hashlib,
    json,
    os,
    random,
    torch,
    valid_cases,
):
    if len(graph_items) != len(valid_cases):
        raise RuntimeError("Graph build is incomplete; resume cache construction before splitting")
    _ordered = sorted(graph_items, key=lambda item: os.path.basename(item[1]).replace(".meta.pt", ""))
    random.Random(SEED).shuffle(_ordered)
    n_items = len(_ordered)
    n_train = max(1, int(0.7 * n_items))
    n_val = max(1, int(0.15 * n_items)) if n_items >= 7 else 1
    train_items = _ordered[:n_train]
    val_items = _ordered[n_train:n_train + n_val]
    test_items = _ordered[n_train + n_val:]
    if not all((train_items, val_items, test_items)):
        raise RuntimeError("Need at least three disjoint, nonempty splits")
    SPLIT_CASE_IDS = {
        "train": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in train_items],
        "val": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in val_items],
        "test": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in test_items],
    }
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["val"]))
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["test"]))
    assert not (set(SPLIT_CASE_IDS["val"]) & set(SPLIT_CASE_IDS["test"]))
    assert len(set().union(*[set(v) for v in SPLIT_CASE_IDS.values()])) == sum(len(v) for v in SPLIT_CASE_IDS.values())
    SPLIT_FINGERPRINT = hashlib.sha256(json.dumps(SPLIT_CASE_IDS, sort_keys=True).encode()).hexdigest()
    print("Split fingerprint:", SPLIT_FINGERPRINT)
    print("Train:", len(train_items), "| Val:", len(val_items), "| Test:", len(test_items))
    voxel_fractions = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for _d, _ in train_items:
        for _nt in NODE_TYPES:
            voxel_fractions = voxel_fractions + (_d[_nt].vol[:, None].double() * _d[_nt].y_frac.double()).sum(0)
    voxel_fractions = voxel_fractions / voxel_fractions.sum()
    class_weights = (1.0 / (NUM_CLASSES * voxel_fractions.clamp_min(1e-6))).sqrt()
    class_weights = (class_weights / class_weights.mean()).float()
    print("Training voxel fraction per class:", [f"{c}={v:.4f}" for c, v in zip(CLASS_NAMES, voxel_fractions.tolist())])
    print("Class weights (sqrt inverse freq):", [f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, class_weights.tolist())])
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
    ## 12. Train QoS-HRGN

    `HGT → uncertainty-aware adaptive propagation × HOPS → pathology prediction`, trained with soft-CE + per-patient volume-weighted Dice, AdamW + cosine schedule, AMP. The checkpoint is selected on **full-volume post-processed patient Dice**, computed after the two graph partitions are fused and projected.
    """)
    return


@app.cell
def _(
    APPEARANCE_DIM,
    BATCH_SIZE,
    CKPT_PUSH_EVERY_EPOCHS,
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    DROPOUT,
    DataLoader,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    HEADS,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    HGT_LAYERS,
    HIDDEN_DIM,
    HOPS,
    HOP_REG_WEIGHT,
    HOP_TEMPERATURE,
    HeteroMaskedReconstructor,
    IDENTITY_CONFIG,
    K_MAX,
    LAMBDA_REC,
    LR,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NUM_CLASSES,
    PART1_START_TIME,
    QoSHRGN,
    REC_DECODER_HIDDEN,
    REC_FLAGS,
    REC_SEPARATE_CLEAN_PASS,
    REC_WARMUP_EPOCHS,
    RESUME_STAGE1,
    RUN_CONFIG,
    SEED,
    SPLIT_CASE_IDS,
    SPLIT_FINGERPRINT,
    STAGE1_LATEST_NAME,
    STAGE1_LATEST_PATH,
    STRUCTURAL_AUX_WEIGHT,
    STRUCTURAL_DROPOUT,
    STRUCTURAL_FLAGS,
    STRUCTURAL_HIDDEN,
    StructuralQoSHRGN,
    TRAIN_BUDGET_HOURS,
    USE_ADAPTIVE_HOPS,
    USE_AMP,
    USE_MASKED_RECONSTRUCTION,
    USE_STRUCTURAL_REFINEMENT,
    WEIGHT_DECAY,
    bounded_prefetch,
    class_weights,
    config_diff,
    device,
    graph_validation_dice,
    hf_try_download,
    hf_upload_file_verified,
    hierarchy_loss,
    mask_hetero_graph,
    mo,
    node_region_dice,
    np,
    os,
    random,
    rec_lambda,
    reconstruction_loss,
    structural_teacher_prob,
    time,
    torch,
    train_items,
    val_items,
):
    base_model = QoSHRGN(
        in_dim=NODE_FEAT_DIM, hidden_dim=HIDDEN_DIM, heads=HEADS, num_classes=NUM_CLASSES,
        hops=HOPS, hgt_layers=HGT_LAYERS, dropout=DROPOUT,
        adaptive_hops=USE_ADAPTIVE_HOPS, k_max=K_MAX, hop_temperature=HOP_TEMPERATURE,
    ).to(device)
    # Option B: wrap the UNCHANGED base model with structural refinement when enabled.
    # Baseline mode (USE_STRUCTURAL_REFINEMENT=False) is untouched: model is base_model.
    if USE_STRUCTURAL_REFINEMENT:
        model = StructuralQoSHRGN(base_model, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                                   flags=STRUCTURAL_FLAGS, refine_hidden=STRUCTURAL_HIDDEN,
                                   dropout=STRUCTURAL_DROPOUT).to(device)
    else:
        model = base_model

    # Option C: reconstruction heads live OUTSIDE the backbone and are trained jointly.
    # When disabled, rec_module is None and the optimizer/loss are identical to baseline.
    rec_module = None
    if USE_MASKED_RECONSTRUCTION:
        rec_module = HeteroMaskedReconstructor(
            hidden_dim=HIDDEN_DIM, appearance_dim=APPEARANCE_DIM, node_types=NODE_TYPES,
            decoder_hidden=REC_DECODER_HIDDEN, dropout=DROPOUT,
        ).to(device)

    _params = list(model.parameters()) + (list(rec_module.parameters()) if rec_module is not None else [])
    optimizer = torch.optim.AdamW(_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    class_weights_dev = class_weights.to(device)

    train_graphs = [d for d, _ in train_items]
    val_graphs = [d for d, _ in val_items]

    # Reproducible masking: train generator advances, val generator is reseeded every
    # epoch so the validation reconstruction loss is comparable across epochs.
    rec_gen = torch.Generator(device=device)
    rec_gen.manual_seed(SEED)
    REC_VAL_SEED = SEED + 1234

    def make_loader(graphs, shuffle):
        return bounded_prefetch(iter(DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=shuffle,
                pin_memory=device.type == "cuda", num_workers=0)), workers=1 if device.type == "cuda" else 0, depth=2)

    def forward_model(model, data, teacher_prob=0.0):
        """Uniform call across baseline and Option-B modes.
        Returns (logits_for_main_loss, aux_initial_logits_or_None, x_dict)."""
        if USE_STRUCTURAL_REFINEMENT:
            initial_logits, final_logits, x_dict = model(data, teacher_prob=teacher_prob)
            return final_logits, initial_logits, x_dict
        else:
            final_logits, x_dict = model(data)
            return final_logits, None, x_dict

    def batch_loss(logits, data, aux_logits=None):
        main = sum(hierarchy_loss(logits[nt], data[nt], data.num_graphs, class_weights_dev) for nt in NODE_TYPES) / len(NODE_TYPES)
        if aux_logits is not None:
            # Option B auxiliary term: keeps the base model's OWN initial predictions
            # well-calibrated, since structural descriptors are derived from them.
            aux = sum(hierarchy_loss(aux_logits[nt], data[nt], data.num_graphs, class_weights_dev) for nt in NODE_TYPES) / len(NODE_TYPES)
            main = main + STRUCTURAL_AUX_WEIGHT * aux
        # Charged exactly once per clean segmentation forward pass. The normalized
        # expected-hop cost discourages the selector from always choosing K_MAX.
        if USE_ADAPTIVE_HOPS and HOP_REG_WEIGHT > 0:
            main = main + HOP_REG_WEIGHT * base_model.hop_regularization
        return main

    def step_losses(data, teacher_prob=0.0, lam=0.0, gen=None):
        """One optimisation step's losses: (total, seg_detached, rec_detached, rec_parts, logits).

        Option C off  -> single clean pass, total == seg (bit-identical to baseline).
        Option C on   -> segmentation from the CLEAN graph (primary objective unchanged)
                         and reconstruction from a separately corrupted graph.
        """
        if rec_module is None or lam <= 0.0:
            logits, aux_logits, _ = forward_model(model, data, teacher_prob=teacher_prob)
            seg = batch_loss(logits, data, aux_logits=aux_logits)
            return seg, float(seg.detach()), 0.0, {}, logits

        if REC_SEPARATE_CLEAN_PASS:
            logits, aux_logits, _ = forward_model(model, data, teacher_prob=teacher_prob)
            seg = batch_loss(logits, data, aux_logits=aux_logits)
            corrupt, info = mask_hetero_graph(data, rec_module, gen=gen)
            _, _, h_dict = forward_model(model, corrupt, teacher_prob=0.0)
        else:
            corrupt, info = mask_hetero_graph(data, rec_module, gen=gen)
            logits, aux_logits, h_dict = forward_model(model, corrupt, teacher_prob=teacher_prob)
            seg = batch_loss(logits, data, aux_logits=aux_logits)

        rec, parts = reconstruction_loss(rec_module, h_dict, info, data, gen=gen, device=device)
        return seg + lam * rec, float(seg.detach()), float(rec.detach()), parts, logits

    print(f"Training on {device} | patients={len(train_graphs)} | batch={BATCH_SIZE} | epochs={EPOCHS} | AMP={USE_AMP}")
    print(f"Parameters: {sum(p.numel() for p in _params) / 1e6:.2f} M"
          + (f"  (backbone {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M"
             f" + rec heads {sum(p.numel() for p in rec_module.parameters()) / 1e3:.1f} k)"
             if rec_module is not None else ""))
    if USE_ADAPTIVE_HOPS:
        print(f"Adaptive-hop propagation ACTIVE: k=0..{K_MAX}, shared propagation weights, "
              f"expected-hop penalty={HOP_REG_WEIGHT}, temperature={HOP_TEMPERATURE}")
    if rec_module is not None:
        print(f"Option C ACTIVE: lambda_rec={LAMBDA_REC} (warmup {REC_WARMUP_EPOCHS} ep), "
              f"targets={[k for k, v in REC_FLAGS.items() if v]}, "
              f"separate_clean_pass={REC_SEPARATE_CLEAN_PASS}")

    def stage1_state(epoch, training_complete):
        return {
            "stage": "1-latest", "epoch": epoch, "epochs_total": EPOCHS,
            "config_hash": CONFIG_HASH, "run_config": RUN_CONFIG, "split": SPLIT_CASE_IDS,
            "split_fingerprint": SPLIT_FINGERPRINT, "dataset_fingerprint": DATASET_FINGERPRINT,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "rec_state_dict": ({k: v.detach().cpu() for k, v in rec_module.state_dict().items()}
                               if rec_module is not None else None),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(), "python": random.getstate(),
                "rec_gen": rec_gen.get_state(),
            },
            "history": history, "best_val_dice": best_val_dice, "best_epoch": best_epoch,
            "best_state": best_state, "best_rec_state": best_rec_state,
            "epochs_since_improve": epochs_since_improve,
            "training_complete": training_complete,
        }


    def save_and_push_latest(epoch, training_complete):
        torch.save(stage1_state(epoch, training_complete), STAGE1_LATEST_PATH + ".tmp")
        os.replace(STAGE1_LATEST_PATH + ".tmp", STAGE1_LATEST_PATH)
        if HF_ENABLED:
            hf_upload_file_verified(STAGE1_LATEST_PATH, STAGE1_LATEST_NAME,
                                    HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                    f"stage1 latest epoch {epoch}")


    history = []
    best_val_dice = -1.0
    best_state = None
    best_rec_state = None
    best_epoch = 0
    epochs_since_improve = 0
    _start_epoch = 1
    _latest = None
    if RESUME_STAGE1:
        _latest_path = STAGE1_LATEST_PATH if os.path.exists(STAGE1_LATEST_PATH) else (
            hf_try_download(STAGE1_LATEST_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE) if HF_ENABLED else None)
        if _latest_path is not None:
            _latest = torch.load(_latest_path, weights_only=False, map_location="cpu")
            if _latest.get("config_hash") != CONFIG_HASH:
                raise RuntimeError("Stage-1 latest checkpoint config mismatch:\n" +
                                   "\n".join(config_diff(_latest.get("run_config", {}).get("identity", {}),
                                                          IDENTITY_CONFIG)))
            if _latest.get("split") != SPLIT_CASE_IDS or _latest.get("split_fingerprint") != SPLIT_FINGERPRINT:
                raise RuntimeError("Stage-1 latest checkpoint split mismatch")
            if _latest.get("dataset_fingerprint") != DATASET_FINGERPRINT:
                raise RuntimeError("Stage1 source dataset mismatch")
            if _latest.get("run_config", {}).get("train") != RUN_CONFIG["train"]:
                raise RuntimeError("Stage1 training configuration mismatch")
            model.load_state_dict(_latest["model_state_dict"])
            if rec_module is not None and _latest["rec_state_dict"] is not None:
                rec_module.load_state_dict(_latest["rec_state_dict"])
            optimizer.load_state_dict(_latest["optimizer_state_dict"])
            scheduler.load_state_dict(_latest["scheduler_state_dict"])
            scaler.load_state_dict(_latest["scaler_state_dict"])
            torch.set_rng_state(_latest["rng"]["torch"])
            if _latest["rng"]["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(_latest["rng"]["cuda"])
            np.random.set_state(_latest["rng"]["numpy"])
            random.setstate(_latest["rng"]["python"])
            rec_gen.set_state(_latest["rng"]["rec_gen"])
            history = _latest["history"]
            best_val_dice = _latest["best_val_dice"]
            best_epoch = _latest["best_epoch"]
            best_state = _latest["best_state"]
            best_rec_state = _latest["best_rec_state"]
            epochs_since_improve = _latest["epochs_since_improve"]
            _start_epoch = _latest["epoch"] + 1

    stage1_training_complete = bool(_latest and _latest.get("training_complete", False))
    _last_epoch = _start_epoch - 1
    _last_epoch_h = 0.0
    for _epoch in range(_start_epoch, EPOCHS + 1):
        if stage1_training_complete:
            break
        _elapsed_h = (time.time() - PART1_START_TIME) / 3600
        if _elapsed_h + _last_epoch_h * 1.1 > TRAIN_BUDGET_HOURS:
            save_and_push_latest(_epoch - 1, False)
            print("budget reached — re-run Part 1 to resume")
            stage1_training_complete = False
            break
        _epoch_t0 = time.time()
        _lam = rec_lambda(_epoch)
        model.train()
        if rec_module is not None:
            rec_module.train()
        _tl = _ts = _tr = 0.0
        _nb = 0
        _dices_tr = []
        _tparts = {}
        for _data in make_loader(train_graphs, shuffle=True):
            _data = _data.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _teacher_prob = structural_teacher_prob(_epoch) if USE_STRUCTURAL_REFINEMENT else 0.0
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                _loss, _seg_v, _rec_v, _parts, _logits = step_losses(_data, teacher_prob=_teacher_prob, lam=_lam, gen=rec_gen)
            _dices_tr.append(node_region_dice({k: v.detach() for k, v in _logits.items()}, _data).cpu())
            scaler.scale(_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(_params, 2.0)
            scaler.step(optimizer)
            scaler.update()
            _tl += float(_loss.detach()); _ts += _seg_v; _tr += _rec_v
            for _key, _value in _parts.items():
                _tparts[_key] = _tparts.get(_key, 0.0) + _value
            _nb += 1
        scheduler.step()
        model.eval()
        if rec_module is not None:
            rec_module.eval()
            rec_gen.manual_seed(REC_VAL_SEED)
        _vl = _vs = _vr = 0.0
        _vb = 0
        _vparts = {}
        _dices = []
        with torch.no_grad():
            for _data in make_loader(val_graphs, shuffle=False):
                _data = _data.to(device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP):
                    _loss, _seg_v, _rec_v, _parts, _logits = step_losses(_data, teacher_prob=0.0, lam=_lam, gen=rec_gen)
                _vl += float(_loss); _vs += _seg_v; _vr += _rec_v
                for _key, _value in _parts.items():
                    _vparts[_key] = _vparts.get(_key, 0.0) + _value
                _vb += 1
                _dices.append(node_region_dice(_logits, _data).cpu())
        if rec_module is not None:
            rec_gen.manual_seed(SEED + _epoch)
        _val_node_proxy = float(torch.cat(_dices).mean())
        _val_dice = graph_validation_dice(model, val_items)
        _nb_s, _vb_s = max(1, _nb), max(1, _vb)
        history.append(dict(epoch=_epoch, lam=_lam, train_total=_tl / _nb_s,
                            train_seg=_ts / _nb_s, train_rec=_tr / _nb_s,
                            val_total=_vl / _vb_s, val_seg=_vs / _vb_s, val_rec=_vr / _vb_s,
                            train_node_proxy=float(torch.cat(_dices_tr).mean()) if _dices_tr else 0.0,
                            val_dice=_val_dice, val_node_proxy=_val_node_proxy, train_rec_parts={k: v / _nb_s for k, v in _tparts.items()},
                            val_rec_parts={k: v / _vb_s for k, v in _vparts.items()},
                            epoch_seconds=time.time() - _epoch_t0))
        if _val_dice > best_val_dice:
            best_val_dice, best_epoch = _val_dice, _epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_rec_state = ({k: v.detach().cpu().clone() for k, v in rec_module.state_dict().items()}
                              if rec_module is not None else None)
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        _last_epoch = _epoch
        _last_epoch_h = (time.time() - _epoch_t0) / 3600
        if _epoch == 1 or _epoch % 5 == 0:
            _h = history[-1]
            print(f"Epoch {_epoch:03d} | Train {_h['train_total']:.4f} | Val patient-Dice {_val_dice:.4f} | lr {scheduler.get_last_lr()[0]:.2e}")
        if _epoch % CKPT_PUSH_EVERY_EPOCHS == 0:
            save_and_push_latest(_epoch, False)
        if epochs_since_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping after {_epoch} epochs without improvement")
            stage1_training_complete = True
            break
    else:
        stage1_training_complete = True

    if stage1_training_complete and not (_latest and _latest.get("training_complete")):
        save_and_push_latest(_last_epoch, True)
    mo.stop(not stage1_training_complete, mo.md("Session budget reached. Resume Part 1 before evaluation or handoff."))
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    if rec_module is not None and best_rec_state is not None:
        rec_module.load_state_dict(best_rec_state)
        rec_module.to(device)
    print(f"Restored best model from epoch {best_epoch} (val patient-Dice {best_val_dice:.4f}).")
    return (
        best_epoch,
        best_val_dice,
        history,
        model,
        rec_module,
        stage1_training_complete,
    )


@app.cell
def _(
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    PERSISTENT_BASE,
    best_epoch,
    hf_upload_file_verified,
    history,
    np,
    os,
    plt,
    stage1_training_complete,
):
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


    STAGE1_CURVE_FILES = export_training_curves(
        history, "stage1_curves", ("train_total", "val_total"),
        ("train_node_proxy", "val_dice"), best_epoch)
    if HF_ENABLED and stage1_training_complete:
        for _path in STAGE1_CURVE_FILES:
            hf_upload_file_verified(_path, os.path.basename(_path), HF_MODEL_REPO_ID,
                                    HF_MODEL_REPO_TYPE, "stage1 training curves")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 13. Validation / test metrics (voxel level)

    For every case we report the model's Dice **raw**, **post-processed** (small-ET removal) and the **oracle** ceiling of the partition. The gap `oracle − post-processed` is what better graph learning can still gain; the gap `1 − oracle` can only be closed by a finer partition or a voxel-level refinement head.
    """)
    return


@app.cell
def _(
    fmt_metrics,
    load_meta,
    model,
    oracle_metrics,
    postprocess_prediction_auto,
    reconstruct_case,
    segmentation_metrics,
    summarize_metric_rows,
    test_items,
    val_items,
):
    def evaluate_items(items, name="Set", verbose=True):
        rows = {"raw": [], "pp": [], "oracle": []}
        for _i, (_d, _mpath) in enumerate(items):
            _meta = load_meta(_mpath)
            _pred_raw = reconstruct_case(model, _d, _meta, postprocess=False)
            _m_raw = segmentation_metrics(_pred_raw, _meta["seg"])
            _m_pp = segmentation_metrics(postprocess_prediction_auto(_pred_raw), _meta["seg"])
            _m_or = oracle_metrics(_d, _meta, soft=False)
            rows["raw"].append(_m_raw); rows["pp"].append(_m_pp); rows["oracle"].append(_m_or)
            if verbose:
                print(f"{name} {_i + 1:02d}: {fmt_metrics(_m_pp)}   [oracle {fmt_metrics(_m_or)}]")
        summary = {}
        for _kind, _rs in rows.items():
            summary[_kind] = summarize_metric_rows(_rs)
        print(f"\n{name} mean  raw          : {fmt_metrics(summary['raw'])}")
        print(f"{name} mean  post-processed: {fmt_metrics(summary['pp'])}")
        print(f"{name} mean  oracle ceiling: {fmt_metrics(summary['oracle'])}")
        return summary

    val_metrics = evaluate_items(val_items, "VAL")
    test_metrics = evaluate_items(test_items, "TEST")
    return test_metrics, val_metrics


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1 complete -- save graph-model checkpoint and hand off to Part 2

    Graph-model training (Sec. 12) and its evaluation (Sec. 13) are done. Everything
    below in the original notebook -- voxel refinement, diagnostics, XAI -- now
    runs in **Part 2**, a separate Colab session, so neither notebook risks the
    12-hour cap on its own.

    The handoff is a checkpoint (`model` + `rec_module` weights, class weights,
    config, and the Sec. 13 metrics) pushed to a dedicated Hugging Face **model**
    repo (`marvelpokemaster/brats-qos-hrgn-model` -- distinct from the graph-cache **dataset** repo in Sec. 2b;
    do not mix the two). Part 2 downloads it fresh, verifies it against a
    checksum manifest, and picks up from there.

    **Before starting Part 2:** wait for the "Upload verified" line below. The
    upload can take a few minutes on Colab's outbound bandwidth -- starting
    Part 2 before it prints means Part 2 will either fail its checksum check or,
    worse, silently load a stale checkpoint from an earlier run.
    """)
    return


@app.cell
def _(
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    HF_CACHE_SUBDIR,
    HF_REPO_ID,
    RUN_CONFIG,
    SPLIT_CASE_IDS,
    SPLIT_FINGERPRINT,
    STAGE1_CKPT_PATH,
    best_epoch,
    best_val_dice,
    class_weights,
    device,
    graph_items,
    history,
    model,
    os,
    rec_module,
    sha256_file,
    stage1_training_complete,
    test_metrics,
    torch,
    val_metrics,
):
    stage1_checkpoint = {
        "stage": 1,
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "rec_state_dict": ({k: v.detach().cpu() for k, v in rec_module.state_dict().items()}
                           if rec_module is not None else None),
        "class_weights": class_weights.detach().cpu(),
        "best_val_dice": best_val_dice, "best_epoch": best_epoch, "epochs_run": len(history),
        "training_complete": stage1_training_complete, "history": history,
        "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0,
        "val_metrics": val_metrics, "test_metrics": test_metrics,
        "config_hash": CONFIG_HASH, "run_config": RUN_CONFIG, "split": SPLIT_CASE_IDS,
        "split_fingerprint": SPLIT_FINGERPRINT, "dataset_fingerprint": DATASET_FINGERPRINT,
        "graph_cache": {"repo_id": HF_REPO_ID, "subdir": HF_CACHE_SUBDIR, "n_graphs": len(graph_items)},
    }
    torch.save(stage1_checkpoint, STAGE1_CKPT_PATH + ".tmp")
    os.replace(STAGE1_CKPT_PATH + ".tmp", STAGE1_CKPT_PATH)
    print(f"Saved Stage-1 checkpoint: {STAGE1_CKPT_PATH}")
    STAGE1_SAVED_PATH = STAGE1_CKPT_PATH
    STAGE1_SAVED_SHA256 = sha256_file(STAGE1_SAVED_PATH)
    return STAGE1_SAVED_PATH, STAGE1_SAVED_SHA256


@app.cell
def _(
    CONFIG_HASH,
    DATASET_FINGERPRINT,
    GRAPH_VERSION,
    HF_ENABLED,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    PERSISTENT_BASE,
    SPLIT_CASE_IDS,
    STAGE1_MANIFEST_NAME,
    STAGE1_REMOTE_NAME,
    STAGE1_SAVED_PATH,
    STAGE1_SAVED_SHA256,
    best_epoch,
    best_val_dice,
    datetime,
    ensure_graph_cache_handoff,
    hf_try_download,
    hf_upload_file_verified,
    hf_upload_receipt,
    history,
    os,
    sha256_file,
    stage1_training_complete,
    write_json_atomic,
):
    if not stage1_training_complete:
        raise RuntimeError("Resume graph training before the Part1 handoff")
    GRAPH_CACHE_HANDOFF = ensure_graph_cache_handoff()
    _checksum = STAGE1_SAVED_SHA256
    _checkpoint_revision = None
    if HF_ENABLED:
        hf_upload_file_verified(STAGE1_SAVED_PATH, STAGE1_REMOTE_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                f"stage1 verified checkpoint, epoch={best_epoch}")
        _checkpoint_revision = hf_upload_receipt(HF_MODEL_REPO_ID, STAGE1_REMOTE_NAME)["revision"]
    _manifest = {
        "sha256": _checksum, "size_bytes": os.path.getsize(STAGE1_SAVED_PATH),
        "checkpoint_revision": _checkpoint_revision, "graph_cache": GRAPH_CACHE_HANDOFF,
        "uploaded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "best_val_dice": best_val_dice, "best_epoch": best_epoch, "epochs_run": len(history),
        "training_complete": stage1_training_complete, "graph_version": GRAPH_VERSION,
        "config_hash": CONFIG_HASH, "dataset_fingerprint": DATASET_FINGERPRINT,
        "split_sizes": {k: len(v) for k, v in SPLIT_CASE_IDS.items()},
    }
    _manifest_path = os.path.join(PERSISTENT_BASE, STAGE1_MANIFEST_NAME)
    write_json_atomic(_manifest_path, _manifest)
    if HF_ENABLED:
        hf_upload_file_verified(_manifest_path, STAGE1_MANIFEST_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                "complete Part1 handoff manifest")
        _remote_path = hf_try_download(STAGE1_REMOTE_NAME, HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE,
                                       revision=_checkpoint_revision)
        if _remote_path is None or sha256_file(_remote_path) != _checksum:
            raise RuntimeError("Stage1 read-after-write checksum mismatch")
    print("Part1 handoff ready: verified graph ZIP + checkpoint + persisted split + manifest")
    return


if __name__ == "__main__":
    app.run()
