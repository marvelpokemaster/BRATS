# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "kagglehub==1.0.2",
#     "nibabel==5.4.2",
#     "torch-geometric==2.8.0.post1",
#     "huggingface-hub==1.32.0",
#     "monai==1.5.1",
# ]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App(auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Part 6 - advisor follow-up tests (inference only)

    Answers the advisor's new questions (`Advisor.md`) using the trained v9 checkpoints already on Hugging Face.
    **Nothing is trained or changed.** Run it after Parts 1-3 and 5 are complete, in a fresh Molab session
    with the helper `.py` files beside it (Molab cannot upload folders).

    | Test | Advisor question |
    |---|---|
    | 2 | Which wrong CNN predictions does the graph correct, especially enhancing tumour? |
    | 3a | Edge perturbation: remove high-gate vs low-gate edges with matched budgets |
    | 3b | Modality intervention: do Shapley rankings agree with realistic modality perturbations? |
    | 3c | Hop sensitivity: restrict or perturb information from different graph distances |

    Clear the blank Hugging Face token field before exporting this notebook to HTML.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


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
    K_MAX = 2
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
    VOXEL_FOCAL_WEIGHT = 0.0
    VOXEL_FOCAL_GAMMA = 2.0
    VOXEL_CE_WEIGHT = 1.0
    VOXEL_BOUNDARY_WEIGHT = 0.0
    VOXEL_MARGIN = 10
    VOXEL_MAX_SIZE = 128

    # ── Focal loss for the graph model ────────────────────────────
    ET_FOCAL_WEIGHT = 0.0
    FOCAL_GAMMA = 2.0

    # ── Connected-component post-processing ──────────────────────
    USE_CC_POSTPROCESS = True

    # ── Baselines run in Part 3 ─────────
    RUN_BASELINES = True
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
    EXPECTED_CASES = 1251
    CASE_EXCLUSIONS = {}  # Add only verified canonical IDs with documented reasons.
    OFFICIAL_CASE_IDS_PATH = None  # Optional JSON list of official training IDs.
    HF_TOKEN_PLACEHOLDER = ""  # Fill this before uploading/running in molab.
    HF_REQUIRED = True
    REQUIRE_CUDA = True
    GPU_PRECISION = "auto"  # BF16 on supported CUDA GPUs; FP32 on CPU helpers.
    GPU_AUTOTUNE = True
    GPU_MEMORY_FRACTION = 0.85
    VOXEL_EFFECTIVE_BATCH_SIZE = 16
    VOXEL_MAX_MICROBATCH = 16  # whole effective batch in one GPU pass when it fits (same optimizer updates)
    INFERENCE_MAX_BATCH = 16
    VOXEL_PREFETCH_WORKERS = 12
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

    RUN_REGION_XAI = True  # Additional analysis session; saves direct graph-region attribution.

    # Read the final handoff from the running Part 1, without writing to its files.
    from brats_workflow import load_part1, load_exact_model, make_store, require_main_complete, restore_main, sync_main
    HF_MODEL_REPO_ID = "marvelpokemaster/brats-qos-hrgn-model"
    HF_MODEL_REPO_TYPE = "model"
    STAGE1_MANIFEST_OVERRIDE = ""  # Normally leave blank; detects current and review-v3 names.
    PART1_INPUT = load_part1(HF_TOKEN_PLACEHOLDER.strip() or (os.environ.get("HF_TOKEN") or "").strip() or (_get_hf_token_early() or "").strip(),
                            HF_MODEL_REPO_ID, HF_MODEL_REPO_TYPE, STAGE1_MANIFEST_OVERRIDE)
    stage1_checkpoint = PART1_INPUT["checkpoint"]
    STAGE1_HANDOFF_MANIFEST = PART1_INPUT["manifest"]
    STAGE1_MODEL_SHA256 = PART1_INPUT["sha256"]
    ZIP_SYNC_MINUTES = 120
    _saved_graph = stage1_checkpoint["run_config"]["identity"]["graph"]
    _saved_model = stage1_checkpoint["run_config"]["identity"]["model"]
    _saved_train = stage1_checkpoint["run_config"]["train"]
    N_SEGMENTS = _saved_graph["n_segments"]
    COMPACTNESS = _saved_graph["compactness"]
    SLIC_ITERS = _saved_graph["slic_iters"]
    MIN_OVERLAP = _saved_graph["min_overlap"]
    NODE_TYPES = _saved_graph["node_types"]
    SLIC_MODALITIES = _saved_graph["slic_modalities"]
    MODALITIES = _saved_graph["modalities"]
    NODE_FEATURE_MODALITIES = _saved_graph["node_feature_modalities"]
    QUANTILES = _saved_graph["quantiles"]
    NODE_FEAT_DIM = _saved_graph["node_feat_dim"]
    GRAPH_VERSION = _saved_graph["graph_version"]
    HIDDEN_DIM = _saved_model["hidden_dim"]
    HEADS = _saved_model["heads"]
    HGT_LAYERS = _saved_model["hgt_layers"]
    HOPS = _saved_model["hops"]
    USE_ADAPTIVE_HOPS = _saved_model["use_adaptive_hops"]
    K_MAX = 2
    HOP_TEMPERATURE = _saved_model["hop_temperature"]
    DROPOUT = _saved_model["dropout"]
    EPOCHS = _saved_train["epochs"]
    BATCH_SIZE = 8
    LR = _saved_train["lr"]
    WEIGHT_DECAY = _saved_train["weight_decay"]
    DICE_WEIGHT = _saved_train["dice_weight"]
    ET_FOCAL_WEIGHT = 0.0
    FOCAL_GAMMA = _saved_train["focal_gamma"]
    HOP_REG_WEIGHT = _saved_train["hop_reg_weight"]
    SEED = stage1_checkpoint["run_config"]["identity"]["seed"]

    GRAPH_MAX_MICROBATCH = 8
    GRAPH_MEMORY_FRACTION = 0.85

    import monai as _monai  # Required here and again in a fresh Part 5 session; tested with MONAI 1.5.1.

    return (
        BATCH_SIZE,
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
        HF_MODEL_REPO_ID,
        HF_MODEL_REPO_TYPE,
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
        PART1_INPUT,
        QUANTILES,
        RUN_REGION_XAI,
        SEED,
        SLIC_ITERS,
        SLIC_MODALITIES,
        STAGE1_HANDOFF_MANIFEST,
        STAGE1_MODEL_SHA256,
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
        VOXEL_MAX_MICROBATCH,
        VOXEL_MAX_SIZE,
        VOXEL_PREFETCH_WORKERS,
        WEIGHT_DECAY,
        ZIP_SYNC_MINUTES,
        device,
        hashlib,
        json,
        load_exact_model,
        make_store,
        math,
        nib,
        nn,
        np,
        os,
        plt,
        require_main_complete,
        slic,
        stage1_checkpoint,
        time,
        torch,
    )


@app.cell
def _():
    import time as _time_mod
    PART2_START_TIME = _time_mod.time()
    SESSION_HARD_CAP_HOURS = 12.0
    BUILD_BUDGET_HOURS = 3.0
    TRAIN_BUDGET_HOURS = 9.5
    CKPT_PUSH_EVERY_EPOCHS = 5
    RESUME_STAGE2 = True
    EARLY_STOP_PATIENCE = 15
    SESSION_DEADLINE = PART2_START_TIME + TRAIN_BUDGET_HOURS * 3600
    print("Continuation session: 9.5-hour work budget, then verified ZIP handoff.")
    return (
        CKPT_PUSH_EVERY_EPOCHS,
        EARLY_STOP_PATIENCE,
        PART2_START_TIME,
        SESSION_DEADLINE,
        TRAIN_BUDGET_HOURS,
    )


@app.cell
def _(EPOCHS, stage1_checkpoint):
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
    USE_STRUCTURAL_REFINEMENT = False
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
    _saved_struct = stage1_checkpoint["run_config"]["identity"]["structural"]
    USE_STRUCTURAL_REFINEMENT = False
    STRUCTURAL_FLAGS = dict(_saved_struct["structural_flags"])
    STRUCTURAL_HIDDEN = _saved_struct["structural_hidden"]
    STRUCTURAL_DROPOUT = _saved_struct["structural_dropout"]
    STRUCTURAL_AUX_WEIGHT = stage1_checkpoint["run_config"]["train"]["structural_aux_weight"]
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


@app.cell
def _(MODALITIES, NODE_FEAT_DIM, QUANTILES, stage1_checkpoint):
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
    # Set USE_MASKED_RECONSTRUCTION = True to run the exact original baseline
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
    LAMBDA_REC = stage1_checkpoint["run_config"]["train"]["lambda_rec"]
    if USE_MASKED_RECONSTRUCTION:
        REC_FLAGS = dict(stage1_checkpoint["run_config"]["identity"]["rec"]["rec_flags"])
        REC_DECODER_HIDDEN = stage1_checkpoint["run_config"]["identity"]["rec"]["rec_decoder_hidden"]
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
            "normalization": "per-patient-v7",
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
            "phase_seconds_limit": None, "compute_protocol": "convergence-v9",
            "graph_training_protocol": "practical-v8-patient-mean",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dice_weight": DICE_WEIGHT, "et_focal_weight": ET_FOCAL_WEIGHT, "focal_gamma": FOCAL_GAMMA,
            "amp": USE_AMP, "hop_reg_weight": HOP_REG_WEIGHT, "structural_aux_weight": STRUCTURAL_AUX_WEIGHT,
            "lambda_rec": LAMBDA_REC,
        },
        "voxel": {
            "phase_seconds_limit": None, "compute_protocol": "convergence-v9",
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
    return IDENTITY_CONFIG, RUN_CONFIG, config_hash


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


@app.cell
def _(EXPECTED_CASES, GRAPH_CACHE_BASE_PATH, os, stage1_checkpoint):
    # Part 1 already audited the data. Its verified graph ZIP contains the MRI
    # tensors and targets needed for refinement/evaluation. Only Part 4 needs raw MRI.
    RAW_SOURCE_AVAILABLE = False
    DATASET_FINGERPRINT = stage1_checkpoint["dataset_fingerprint"]
    DATASET_AUDIT = dict(errors=[],source_fingerprint=DATASET_FINGERPRINT,
        scope="Inherited from the completed, checksum-verified Part 1 handoff; no new raw-data audit in this session")
    GRAPH_CACHE_PATH = os.path.join(GRAPH_CACHE_BASE_PATH,DATASET_FINGERPRINT[:16])
    os.makedirs(GRAPH_CACHE_PATH,exist_ok=True)
    _saved_ids = [cid for group in stage1_checkpoint["split"].values() for cid in group]
    if len(_saved_ids)!=len(set(_saved_ids)) or len(_saved_ids)!=EXPECTED_CASES:
        raise RuntimeError("Saved Part 1 cohort is duplicate or does not match the required patient count")
    valid_cases = [(cid,{"t1":os.path.join(GRAPH_CACHE_PATH,cid,cid+"_t1.nii.gz")}) for cid in sorted(_saved_ids)]
    MAX_CASES = None
    print("Using Part 1's audited cohort and verified graph archive:",len(valid_cases),"patients")
    return (
        DATASET_AUDIT,
        DATASET_FINGERPRINT,
        GRAPH_CACHE_PATH,
        RAW_SOURCE_AVAILABLE,
        valid_cases,
    )


@app.cell
def _(
    GRAPH_CACHE_PATH,
    HF_MODEL_REPO_ID,
    HF_MODEL_REPO_TYPE,
    HF_TOKEN_PLACEHOLDER,
    PART1_INPUT,
    PERSISTENT_BASE,
    STAGE1_HANDOFF_MANIFEST,
    STAGE1_MODEL_SHA256,
    ZIP_SYNC_MINUTES,
    make_store,
    os,
):
    from huggingface_hub import HfApi, hf_hub_download, get_token
    from huggingface_hub.utils import EntryNotFoundError
    HF_TOKEN = HF_TOKEN_PLACEHOLDER.strip() or (os.environ.get("HF_TOKEN") or "").strip() or (get_token() or "").strip()
    HF_ENABLED = bool(HF_TOKEN)
    hf_api = PART1_INPUT["api"]
    HF_REPO_ID = STAGE1_HANDOFF_MANIFEST["graph_cache"]["repo_id"]
    HF_REPO_TYPE = STAGE1_HANDOFF_MANIFEST["graph_cache"]["repo_type"]
    HF_CACHE_SUBDIR = STAGE1_HANDOFF_MANIFEST["graph_cache"]["cache_subdir"]
    HF_CACHE_MANIFEST_NAME = HF_CACHE_SUBDIR + ".zip.manifest.json"
    HF_CACHE_ZIP_NAME = HF_CACHE_SUBDIR + ".zip"
    GRAPH_CACHE_RECEIPT_PATH = os.path.join(PERSISTENT_BASE,"graph_cache_receipt.json")
    GRAPH_CACHE_ZIP_PATH = GRAPH_CACHE_PATH + ".zip"
    STAGE1_REMOTE_NAME = PART1_INPUT["checkpoint_name"]
    STAGE1_MANIFEST_NAME = PART1_INPUT["manifest_name"]
    STAGE1_LATEST_NAME = "stage1_latest.pt"
    STAGE2_LATEST_NAME = "stage2_latest_gpu_v4.pt"
    STAGE1_CKPT_PATH = PART1_INPUT["path"]
    STAGE1_LATEST_PATH = os.path.join(PERSISTENT_BASE,STAGE1_LATEST_NAME)
    STAGE2_LATEST_PATH = os.path.join(PERSISTENT_BASE,STAGE2_LATEST_NAME)

    def sha256_file(path):
        from brats_transfer import sha256_path
        return sha256_path(path)

    def hf_try_download(remote_name, repo_id, repo_type, revision="main"):
        try:
            return hf_hub_download(repo_id=repo_id,repo_type=repo_type,filename=remote_name,revision=revision,token=HF_TOKEN)
        except EntryNotFoundError:
            return None

    ZIP_CONTEXT = dict(PERSISTENT_BASE=PERSISTENT_BASE,STAGE1_MODEL_SHA256=STAGE1_MODEL_SHA256,
        HF_ENABLED=HF_ENABLED,hf_api=hf_api,hf_try_download=hf_try_download,
        HF_MODEL_REPO_ID=HF_MODEL_REPO_ID,HF_MODEL_REPO_TYPE=HF_MODEL_REPO_TYPE,ZIP_SYNC_MINUTES=ZIP_SYNC_MINUTES)
    MAIN_STORE = make_store(ZIP_CONTEXT,"main")

    def hf_upload_file_verified(local_path,remote_name,repo_id,repo_type,commit_message):
        # Compatibility wrapper queues bytes locally. The ZIP store performs the
        # actual verified atomic commit at an interval or explicit final flush.
        if repo_id != HF_MODEL_REPO_ID:
            raise RuntimeError("Continuation must not overwrite Part 1's graph repository")
        MAIN_STORE.add_file(local_path,remote_name)
        return sha256_file(local_path)

    def hf_upload_receipt(repo_id,remote_name):
        raise RuntimeError("Individual-file receipts are replaced by the pinned ZIP catalog")

    def research_store_factory(key):
        return make_store(ZIP_CONTEXT,"research/"+key)

    return (
        GRAPH_CACHE_RECEIPT_PATH,
        HF_CACHE_MANIFEST_NAME,
        HF_CACHE_SUBDIR,
        HF_ENABLED,
        HF_REPO_ID,
        HF_REPO_TYPE,
        MAIN_STORE,
        STAGE2_LATEST_NAME,
        ZIP_CONTEXT,
        hf_try_download,
        hf_upload_file_verified,
        hf_upload_receipt,
        research_store_factory,
    )


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
    import sys as _sys, os as _os
    _nb_dir = _os.path.dirname(_os.path.abspath(globals().get("__file__", "notebook.py"))) if '__file__' in globals() else _os.getcwd()
    if _nb_dir not in _sys.path: _sys.path.insert(0, _nb_dir)
    from brats_protocol import (PROTOCOL_VERSION, segmentation_metrics, summarize_metric_rows,
                                mean_region_dice, sliding_window_predict, make_training_patch, audit_dataset,
                                case_id_from_path)
    from brats_gpu import (bounded_prefetch, patient_groups, configure_gpu, amp_context, train_patient_group, train_voxel_experiment)
    from brats_transfer import (upload_verified, verify_remote, extract_verified_zip, write_json_atomic)

    return (
        CLASS_NAMES,
        NUM_CLASSES,
        PROTOCOL_VERSION,
        case_id_from_path,
        crop,
        crop_bounds,
        extract_verified_zip,
        load_case,
        load_meta,
        postprocess_prediction_auto,
        segmentation_metrics,
        sliding_window_predict,
        write_json_atomic,
    )


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

    return predict_node_probs, project_node_probs


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
                from torch.utils.checkpoint import checkpoint as _checkpoint
                _propagate = (lambda *args, **kwargs: _checkpoint(self.adaptive_prop, *args, use_reentrant=False, **kwargs)) if self.training and torch.is_grad_enabled() else self.adaptive_prop
                current, gates = _propagate(
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
                self.hop_regularization_by_node[nt] = effective
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
            self.hop_regularization_by_node = {}
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

    return VoxelRefinementHead, voxel_loss


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
        reconstruction_loss,
        sample_hard_negatives,
    )


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
                from brats_graph import patient_descriptors as _patient_descriptors
                s = _patient_descriptors(data, nt, struct_class, cross_src_index, self.flags, compute_structural_descriptors)
                h = x_dict[nt]
                final_logits[nt] = self.refine[nt](h, init_prob, s, initial_logits[nt])
            return (initial_logits, final_logits, x_dict)  # Scheduled sampling: mix GT-derived and model-predicted  # structure during TRAINING ONLY. Never at eval/test.  # eval/test: predictions only -- no GT leakage.

    return (StructuralQoSHRGN,)


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
        from brats_graph import patient_hierarchy_loss as _patient_hierarchy_loss
        return _patient_hierarchy_loss(logits, data_nt, num_graphs, class_weights, dice_weight,
            soft_cross_entropy, region_dice_loss, focal_loss_soft, ET_FOCAL_WEIGHT, FOCAL_GAMMA)

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

    return focal_loss_soft, node_region_dice, region_dice_loss, soft_cross_entropy


@app.cell
def _(
    COMPACTNESS,
    DATASET_AUDIT,
    DATASET_FINGERPRINT,
    GRAPH_CACHE_PATH,
    GRAPH_CACHE_RECEIPT_PATH,
    GRAPH_VERSION,
    HF_CACHE_MANIFEST_NAME,
    HF_CACHE_SUBDIR,
    HF_ENABLED,
    HF_REPO_ID,
    HF_REPO_TYPE,
    HF_REQUIRED,
    IDENTITY_CONFIG,
    N_SEGMENTS,
    STAGE1_HANDOFF_MANIFEST,
    build_hetero_case,
    case_id_from_path,
    config_hash,
    extract_verified_zip,
    hf_try_download,
    json,
    np,
    os,
    torch,
    valid_cases,
    write_json_atomic,
):
    GRAPH_PULL_RECEIPT = STAGE1_HANDOFF_MANIFEST.get("graph_cache")
    assert not DATASET_AUDIT["errors"]
    import zipfile

    print(f"[Cache] N_SEGMENTS={N_SEGMENTS}  COMPACTNESS={COMPACTNESS}  version={GRAPH_VERSION}")
    print(f"[Cache] {GRAPH_CACHE_PATH}")

    GRAPH_BUILD_STATS_PATH = os.path.join(GRAPH_CACHE_PATH, "build_stats.json")


    def get_case_id(files):
        return case_id_from_path(files["t1"])


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

    hf_pull_graph_cache_zip()
    if missing_cases():
        raise RuntimeError("Part2 requires the complete verified Part1 graph ZIP; cache rebuild is disabled")
    graph_build_complete = True

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
    return build_and_save, get_case_id, graph_items_by_case


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
    return SPLIT_CASE_IDS, class_weights, test_items, train_items, val_items


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
    DATASET_FINGERPRINT,
    IDENTITY_CONFIG,
    STAGE1_MODEL_SHA256,
    device,
    load_exact_model,
    rec_module,
    stage1_checkpoint,
    untrained_base_model,
    untrained_model,
):
    model = untrained_model
    base_model = untrained_base_model
    load_exact_model(model,base_model,stage1_checkpoint,IDENTITY_CONFIG,DATASET_FINGERPRINT,rec_module)
    model.to(device).eval()
    if rec_module is not None: rec_module.to(device).eval()
    best_val_dice = stage1_checkpoint["best_val_dice"]
    best_epoch = stage1_checkpoint["best_epoch"]
    val_metrics = stage1_checkpoint.get("val_metrics",{})
    test_metrics = stage1_checkpoint.get("test_metrics",{})
    print("Verified unchanged Part 1 selected model:",STAGE1_MODEL_SHA256[:16])
    STAGE1_LOADED = True
    return STAGE1_LOADED, base_model, model


@app.cell
def _(
    DATASET_FINGERPRINT,
    MAIN_STORE,
    RUN_CONFIG,
    SPLIT_CASE_IDS,
    STAGE1_LOADED,
    STAGE1_MODEL_SHA256,
    STAGE2_LATEST_NAME,
    mo,
    require_main_complete,
):
    mo.stop(not STAGE1_LOADED,mo.md("Wait for the verified Part 1 model."))
    MAIN_STAGE2_STATE = require_main_complete(MAIN_STORE,STAGE2_LATEST_NAME)

    def _canon(_x):
        # The saved Part 2 state went through torch.save / torch.load (and the Hub); the current values are fresh
        # Python objects. Compare CONTENT, not container type: tuple == list, numpy scalar == Python scalar.
        if isinstance(_x, dict):
            return {str(_k): _canon(_v) for _k, _v in sorted(_x.items(), key=lambda _kv: str(_kv[0]))}
        if isinstance(_x, (list, tuple)):
            return [_canon(_v) for _v in _x]
        if hasattr(_x, "item") and getattr(_x, "shape", None) == ():
            return _x.item()
        return _x

    def _entries(_stored, _expected):
        _a, _b = _canon(_stored), _canon(_expected)
        _keys = sorted(k for k in set(_a) | set(_b) if _a.get(k, "<absent>") != _b.get(k, "<absent>"))
        return _keys, {k: (_a.get(k, "<absent>"), _b.get(k, "<absent>")) for k in _keys}

    def _local_copy_is_not_the_hub_copy():
        # require_main_complete() returns the local file first when it is marked "pending upload". A leftover local file
        # from an earlier session would then be compared instead of the Hub's Part 2 state.
        _rec = MAIN_STORE.catalog["files"].get(STAGE2_LATEST_NAME)
        _path = MAIN_STORE.path(STAGE2_LATEST_NAME)
        if _rec is None or not _path.exists():
            return False
        from brats_transfer import sha256_path as _sha256_path
        return _sha256_path(_path) != _rec["sha256"]

    # Provenance labels only. Anything else (epochs, lr, channels, loss weights, patch/ROI size, batch size ...) must match.
    _LABEL_ONLY = ("compute_protocol", "phase_seconds_limit", "gpu_training_protocol", "precision_policy")
    stage2_handoff_notes = []
    for _key,_expected in {"stage1_sha256":STAGE1_MODEL_SHA256,"split":SPLIT_CASE_IDS,
            "dataset_fingerprint":DATASET_FINGERPRINT,"stage2_config":RUN_CONFIG["voxel"]}.items():
        if _canon(MAIN_STAGE2_STATE.get(_key))==_canon(_expected):
            continue
        if _key == "stage2_config" and _local_copy_is_not_the_hub_copy():
            import shutil as _shutil
            _path = MAIN_STORE.path(STAGE2_LATEST_NAME)
            _backup = _path.with_name(_path.name + ".stale_local_backup")
            _shutil.move(str(_path), str(_backup))
            MAIN_STORE.pending.discard(STAGE2_LATEST_NAME)
            MAIN_STORE._index()
            print("NOTE: the local copy of the Part 2 state was not the Hub copy; it was moved to", _backup, "and the Hub copy is used.")
            stage2_handoff_notes.append("local Part 2 state differed from the Hub copy; the Hub copy was used")
            MAIN_STAGE2_STATE = require_main_complete(MAIN_STORE,STAGE2_LATEST_NAME)
            if _canon(MAIN_STAGE2_STATE.get(_key))==_canon(_expected):
                continue
        _stored = MAIN_STAGE2_STATE.get(_key)
        _detail = "saved " + type(_stored).__name__ + " vs current " + type(_expected).__name__
        if isinstance(_stored, dict) and isinstance(_expected, dict):
            _names, _pairs = _entries(_stored, _expected)
            _detail += "; differing entries -> " + "; ".join(
                k + ": saved=" + repr(_pairs[k][0])[:120] + " current=" + repr(_pairs[k][1])[:120] for k in _names[:12])
            if _key == "stage2_config" and _names and all(k in _LABEL_ONLY for k in _names):
                print("WARNING Part 2 handoff: only provenance labels differ ->", _detail)
                print("         (every training hyper-parameter matches; continuing, and recording this in the saved results)")
                stage2_handoff_notes.append("stage2_config differed only in provenance labels: " + _detail)
                continue
        raise RuntimeError("Part 2 handoff mismatch: " + _key + " (" + _detail + ")")
    print("Part 2 record: epochs=%s best_epoch=%s stopping_reason=%s phase_seconds_limit=%s" % (
        MAIN_STAGE2_STATE.get("epoch"), MAIN_STAGE2_STATE.get("best_vox_epoch"),
        MAIN_STAGE2_STATE.get("stopping_reason"), MAIN_STAGE2_STATE.get("phase_seconds_limit")))
    stage2_training_complete = True
    return MAIN_STAGE2_STATE, stage2_handoff_notes, stage2_training_complete


@app.cell
def _(
    APPEARANCE_DIM,
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
    HeteroMaskedReconstructor,
    INFERENCE_MAX_BATCH,
    INFERENCE_OVERLAP,
    K_MAX,
    LAMBDA_REC,
    LR,
    MAIN_STAGE2_STATE,
    MAIN_STORE,
    MODALITIES,
    NODE_FEAT_DIM,
    NODE_TYPES,
    NUM_CLASSES,
    N_SEGMENTS,
    PART2_START_TIME,
    PERSISTENT_BASE,
    QoSHRGN,
    RAW_SOURCE_AVAILABLE,
    REC_DECODER_HIDDEN,
    REC_EDGE_MASK_RATE,
    REC_FEAT_LOSS,
    REC_FLAGS,
    REC_MASK_RATE,
    REC_NEG_PER_POS,
    REC_SCE_GAMMA,
    REC_SEPARATE_CLEAN_PASS,
    REC_WARMUP_EPOCHS,
    RUN_CONFIG,
    SEED,
    SLIC_ITERS,
    STAGE1_MODEL_SHA256,
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
    ZIP_CONTEXT,
    build_and_save,
    class_weights,
    device,
    focal_loss_soft,
    get_case_id,
    hf_try_download,
    hf_upload_file_verified,
    hf_upload_receipt,
    load_meta,
    mask_hetero_graph,
    mo,
    node_region_dice,
    postprocess_prediction_auto,
    project_node_probs,
    rec_lambda,
    reconstruction_loss,
    region_dice_loss,
    research_store_factory,
    soft_cross_entropy,
    stage1_checkpoint,
    structural_teacher_prob,
    test_items,
    train_items,
    val_items,
    valid_cases,
    voxel_loss,
):
    RUN_ABLATIONS = True
    RESEARCH_SEEDS = (SEED,)
    RUN_EXTENDED_HOP_CONTROLS = False  # Optional mechanism detail; never change after starting Part 3.
    from brats_experiments import experiment_plan as _plan_for_selection
    from brats_practical import MAIN_ROWS as _main_rows
    RESEARCH_PLAN_NAMES = _main_rows
    RESEARCH_ACTIVE_SEEDS = ()  # Schedule a subset if desired; empty runs all remaining seeds.
    RESEARCH_MAX_CASES_PER_SPLIT = 0  # Full saved cohort. The old 300-case debug cap cannot produce final results.
    RESEARCH_BUDGET_HOURS = TRAIN_BUDGET_HOURS
    RESEARCH_MODE = "test"
    RESEARCH_JOBS = ()
    RESEARCH_PIPELINE_SHA256 = "b5ba00ce35682e2bc574e22ceec967be0d99c5bed21bb1993b4ef11c9a6c20d2"
    # ---- Preflight: the helper .py files beside this notebook must be the BRATS_v9 set ----
    import hashlib as _hashlib, importlib as _importlib
    _V9_SHA256 = {
        "brats_bundles": "70f3d920e930f03519d33664ec1fd66c19e866c64e6b4ab9bed6ce4f05931b81",
        "brats_experiments": "41c50d3ab5706f2a4e63d6f4f8ee56cff8eed017a67e6f623798872c5b946af0",
        "brats_gpu": "9a19f0738eef303925720a2f5e2b15b9d0d08be9c4853406b20d9710fbeb9908",
        "brats_graph": "4bc634078370b47c5aa77a5b36a35e8829b04e2b70adda04e2d5462b3e8c612a",
        "brats_practical": "f3ba3c08a9f5d215d75305736009fd6a6446d2e0293ccf930ff2f915bbbca906",
        "brats_protocol": "6b4b0b52eed2c788c7d54aee31bef31ade5630dc5ce11741ab87442e1e1214fb",
        "brats_reporting": "1a9a36147b89847fa28b6eb6d3038234f5895e12c3e33beca7ff6bd5595133f3",
        "brats_transfer": "26835038a1cabcba5e7882c8ead4efcad7e2544dbf233a79121b8060e50f0c05",
        "brats_workflow": "fc1e4cf9a7b3c7a3445fab5e98ae121869fb12b3b3109c409934af6653cf7d00",
    }
    _NEEDED = {
        "brats_workflow": ("discover_main_plan_key", "describe_pilot_receipt", "find_pilot_receipt", "open_existing_store", "make_store",
                           "load_part1", "load_exact_model", "require_main_complete", "restore_main", "sync_main"),
        "brats_protocol": ("select_main_plan_key", "adopt_recorded_plan", "segmentation_metrics", "sliding_window_predict"),
        "brats_experiments": ("ResearchRunner", "BudgetPause", "experiment_plan", "run_research", "HomogeneousSAGE"),
        "brats_practical": ("MAIN_ROWS", "SegResNetBaseline"),
    }
    _problems, _hash_notes = [], []
    for _module_name, _expected_sha in _V9_SHA256.items():
        try:
            _module = _importlib.import_module(_module_name)
        except Exception as _exc:
            _problems.append(_module_name + ".py cannot be imported: " + type(_exc).__name__ + ": " + str(_exc)[:120])
            continue
        _missing = [_n for _n in _NEEDED.get(_module_name, ()) if not hasattr(_module, _n)]
        if _missing:
            _problems.append(_module_name + ".py at " + str(_module.__file__) + " is an OLD version; it lacks: " + ", ".join(_missing))
        with open(_module.__file__, "rb") as _handle:
            if _hashlib.sha256(_handle.read()).hexdigest() != _expected_sha:
                _hash_notes.append(_module_name + ".py")
    if _problems:
        raise RuntimeError("Wrong helper files on this Molab session. Upload ALL nine helper files from BRATS_v9 (not BRATS_Part5_Molab_v11), "
                           "replace the old ones, and restart the kernel. Problems: " + " | ".join(_problems))
    if _hash_notes:
        print("WARNING: these helper files differ from the BRATS_v9 release:", ", ".join(_hash_notes),
              "- the plan check will refuse changes to brats_gpu.py / brats_graph.py (only practical, experiments and protocol may differ).")
    else:
        print("Helper files verified: identical to the BRATS_v9 release.")
    from brats_workflow import discover_main_plan_key, describe_pilot_receipt, find_pilot_receipt
    MAIN_PLAN_KEY_OVERRIDE = None  # Normally leave None: Part 5 finds the store that holds the six trained rows.
    MAIN_PLAN_KEY, _plan_report = discover_main_plan_key(MAIN_STORE, research_store_factory, RESEARCH_PLAN_NAMES, RESEARCH_SEEDS, MAIN_PLAN_KEY_OVERRIDE)
    # After Part 1 is retrained the main store moves; the earlier validation-only SLIC pilot is carried over unchanged.
    PILOT_SOURCE_PREFIX_OVERRIDE = None  # e.g. "continuation_v8/670ec244f0e27740def0" to choose an earlier Part 4 pilot explicitly
    PILOT_RECEIPT, PILOT_STORE_FACTORY, PILOT_SOURCE_PREFIX = find_pilot_receipt(ZIP_CONTEXT, MAIN_STORE, PILOT_SOURCE_PREFIX_OVERRIDE)
    describe_pilot_receipt(MAIN_STORE, PILOT_RECEIPT)
    from brats_experiments import run_research, experiment_plan, ResearchRunner, BudgetPause
    RESEARCH_AVAILABLE_ROWS = experiment_plan(N_SEGMENTS, HOPS, K_MAX, USE_MASKED_RECONSTRUCTION)
    print("Available research jobs:", [row["name"] for row in RESEARCH_AVAILABLE_ROWS])
    RESEARCH_CONTEXT = dict(MAIN_STORE=MAIN_STORE,
        RESEARCH_STORE_FACTORY=research_store_factory,STAGE1_MODEL_SHA256=STAGE1_MODEL_SHA256,
        RAW_SOURCE_AVAILABLE=RAW_SOURCE_AVAILABLE,
        stage1_checkpoint=stage1_checkpoint,MAIN_STAGE2_STATE=MAIN_STAGE2_STATE,
        HeteroMaskedReconstructor=HeteroMaskedReconstructor,APPEARANCE_DIM=APPEARANCE_DIM,
        REC_DECODER_HIDDEN=REC_DECODER_HIDDEN,rec_lambda=rec_lambda,mask_hetero_graph=mask_hetero_graph,
        reconstruction_loss=reconstruction_loss,REC_SEPARATE_CLEAN_PASS=REC_SEPARATE_CLEAN_PASS,
        LAMBDA_REC=LAMBDA_REC,REC_WARMUP_EPOCHS=REC_WARMUP_EPOCHS,REC_MASK_RATE=REC_MASK_RATE,
        REC_EDGE_MASK_RATE=REC_EDGE_MASK_RATE,REC_FLAGS=REC_FLAGS,REC_NEG_PER_POS=REC_NEG_PER_POS,
        REC_FEAT_LOSS=REC_FEAT_LOSS,REC_SCE_GAMMA=REC_SCE_GAMMA,
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
        node_region_dice=node_region_dice,
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
        budget_hours=RESEARCH_BUDGET_HOURS,session_start=PART2_START_TIME,
        adopt_plan_key=MAIN_PLAN_KEY)  # discovered above; the runner verifies it against the stored protocol.json

    def _flat(_x, _prefix=""):
        if isinstance(_x, dict):
            _out = {}
            for _k, _v in _x.items():
                _out.update(_flat(_v, _prefix + str(_k) + "."))
            return _out
        if isinstance(_x, (list, tuple)):
            return {_prefix.rstrip("."): [_flat(_v, "")[""] if not isinstance(_v, (dict, list, tuple)) else repr(_v)[:60] for _v in _x]}
        return {_prefix.rstrip("."): _x}

    def _explain_plan_mismatch(_exc):
        # ResearchRunner only says WHICH manifest field differs. Show the entries of the stored protocol that differ.
        try:
            _stored = RESEARCH_CONTEXT["RESEARCH_STORE_FACTORY"](MAIN_PLAN_KEY).read("protocol.json") or {}
            _a, _b = _flat(_stored.get("run_config", {})), _flat(RUN_CONFIG)
            _diff = sorted(k for k in set(_a) | set(_b) if _a.get(k, "<absent>") != _b.get(k, "<absent>"))
            print("PLAN MISMATCH DETAIL (", str(_exc)[:200], ")")
            print("  run_config entries that differ (stored in protocol.json vs this notebook):")
            for _k in _diff[:25]:
                print("   ", _k, ": stored =", repr(_a.get(_k, "<absent>"))[:100], "| now =", repr(_b.get(_k, "<absent>"))[:100])
            if not _diff:
                print("  run_config is identical; the difference is in another manifest field (rows, split, class weights, epochs, patience, postprocess, dataset).")
                for _name, _now in (("epochs", EPOCHS), ("voxel_epochs", VOXEL_EPOCHS), ("patience", EARLY_STOP_PATIENCE), ("dataset", DATASET_FINGERPRINT)):
                    print("   ", _name, ": stored =", repr(_stored.get(_name))[:70], "| now =", repr(_now)[:70])
        except Exception as _inner:
            print("could not read the stored protocol for a detailed diff:", type(_inner).__name__, str(_inner)[:200])
    try:
        adv_runner = ResearchRunner(RESEARCH_CONTEXT, RESEARCH_SETTINGS)  # test mode: adopts the recorded plan, trains nothing
    except (RuntimeError, ValueError) as _exc:
        if "differs from plan" in str(_exc) or "does not hash" in str(_exc) or "Unexpected source change" in str(_exc):
            _explain_plan_mismatch(_exc)
        raise
    print("Adopted research plan:", adv_runner.key, "| rows:", [_row["name"] for _row in adv_runner.plan])
    RESEARCH_REPORT_READY = True
    return (BudgetPause, RESEARCH_CONTEXT, RESEARCH_REPORT_READY, adv_runner)


@app.cell
def _(
    BudgetPause,
    CLASS_NAMES,
    MODALITIES,
    NODE_TYPES,
    NUM_CLASSES,
    QUANTILES,
    RESEARCH_REPORT_READY,
    RUN_REGION_XAI,
    SEED,
    SESSION_DEADLINE,
    USE_STRUCTURAL_REFINEMENT,
    device,
    math,
    mo,
    model,
    np,
    plt,
    stage2_training_complete,
    time,
    torch,
    train_items,
):
    mo.stop(not RESEARCH_REPORT_READY,mo.md("Finish research evaluation first."))
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
            if time.time()>=SESSION_DEADLINE: raise BudgetPause("XAI budget reached")
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

    xai_background = build_xai_background(train_items,sample_nodes=500)
    return exact_modality_shapley, xai_background, xai_predict_probabilities


@app.cell
def _(SEED, adv_runner, mo, stage2_training_complete):
    # Load the three trained models that the advisor tests need. Nothing is trained here:
    # fit_graph / fit_voxel return the saved best-validation checkpoint when the phase is complete
    # (this is exactly what ResearchRunner.evaluate_run does).
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 first."))
    adv_specs = {_row["name"]: _row for _row in adv_runner.plan}
    _missing = [_n for _n in ("full", "cnn_only") if _n not in adv_specs]
    if _missing:
        raise RuntimeError("The adopted plan lacks rows: " + ", ".join(_missing))
    adv_groups = adv_runner.graphs(adv_specs["full"])
    adv_graph, _graph_state, _graph_name = adv_runner.fit_graph(adv_specs["full"], SEED, adv_groups)
    adv_head, _head_state, _head_name = adv_runner.fit_voxel(adv_specs["full"], SEED, adv_groups, adv_graph)
    adv_head_cnn, _cnn_state, _cnn_name = adv_runner.fit_voxel(adv_specs["cnn_only"], SEED, adv_groups, None)
    for _module in (adv_graph, adv_head, adv_head_cnn):
        _module.eval()
        for _parameter in _module.parameters():
            _parameter.requires_grad_(False)
    print("Loaded checkpoints:", _graph_name, "|", _head_name, "|", _cnn_name)
    print("Best validation Dice  graph:", round(_graph_state["best_score"], 4),
          " full voxel:", round(_head_state["best_score"], 4), " cnn_only voxel:", round(_cnn_state["best_score"], 4))
    return adv_graph, adv_groups, adv_head, adv_head_cnn, adv_specs


@app.cell
def _(SEED, adv_groups, np):
    # Fixed, declared settings. Choose them BEFORE looking at any result; do not tune on test outcomes.
    from brats_experiments import case_id as _case_id
    ADV_N_PATIENTS = 60            # patients for voxel-level / CNN-level tests (all 189 are used for node-level and Test 2)
    ADV_EDGE_BUDGETS = (0.05, 0.10, 0.20)   # fraction of edges removed per relation
    ADV_RANDOM_DRAWS = 3           # random-removal repeats (node-level); voxel-level uses 1
    ADV_NOISE_SIGMA = 0.5          # Gaussian noise, in units of the modality's in-brain SD
    ADV_BIAS_AMPLITUDE = 0.3       # smooth additive bias field, in units of the modality's in-brain SD
    ADV_BIAS_SIGMA_VOXELS = 20.0   # smoothness of the bias field
    ADV_BOOT_DRAWS = 5000
    ADV_OUT = "advisor_followup"   # prefix inside the research store on the Hub
    adv_case_id = _case_id
    adv_test_all = sorted(adv_groups["test"], key=_case_id)
    _rng = np.random.default_rng(SEED + 606)
    _pick = sorted(_rng.choice(len(adv_test_all), size=min(ADV_N_PATIENTS, len(adv_test_all)), replace=False).tolist())
    adv_test_subset = [adv_test_all[_i] for _i in _pick]
    print(f"Test patients: {len(adv_test_all)} (all) / {len(adv_test_subset)} (fixed subset, seed {SEED + 606})")
    return (ADV_BIAS_AMPLITUDE, ADV_BIAS_SIGMA_VOXELS, ADV_BOOT_DRAWS, ADV_EDGE_BUDGETS, ADV_NOISE_SIGMA,
            ADV_OUT, ADV_RANDOM_DRAWS, adv_case_id, adv_test_all, adv_test_subset)


@app.cell
def _(ADV_BIAS_AMPLITUDE, ADV_BIAS_SIGMA_VOXELS, ADV_NOISE_SIGMA, ADV_OUT, INFERENCE_OVERLAP, MODALITIES,
      NODE_TYPES, QUANTILES, VOXEL_MAX_SIZE, adv_runner, device, np, postprocess_prediction_auto,
      project_node_probs, sliding_window_predict, torch):
    # Helper functions (graph interventions, perturbations, statistics). All globals are prefixed adv_/ADV_.
    from contextlib import contextmanager as _contextmanager
    from brats_protocol import region_masks as _region_masks
    import json as _json

    ADV_RELATIONS = (("t1ce", "spatial", "t1ce"), ("flair", "spatial", "flair"),
                     ("t1ce", "corresponds", "flair"), ("flair", "corresponds", "t1ce"))
    ADV_REGIONS = ("WT", "TC", "ET")
    _WIDTH = 2 + len(QUANTILES)

    def adv_slice(modality):
        _i = MODALITIES.index(modality)
        return slice(_i * _WIDTH, (_i + 1) * _WIDTH)

    @torch.no_grad()
    def adv_graph_probs(graph, data):
        graph.eval()
        _logits, _ = graph(data.clone().to(device))
        return {nt: _logits[nt].float().softmax(-1).cpu().numpy() for nt in NODE_TYPES}

    def adv_read_state(graph):
        """After a forward pass: per-node effective hop, hop weights, and per-edge gate (mean over the propagation hops)."""
        _eff = {nt: graph.last_hop_info[nt]["effective_hop"].float().cpu().numpy() for nt in NODE_TYPES}
        _weights = {nt: graph.last_hop_info[nt]["weights"].float().cpu().numpy() for nt in NODE_TYPES}
        _gates = {et: torch.stack([hop[et].float() for hop in graph.last_edge_gates]).mean(0).cpu().numpy()
                  for et in ADV_RELATIONS}
        return _eff, _weights, _gates

    @_contextmanager
    def adv_hop_control(graph, force=None, drop=None):
        """force=j: every node uses only hop j. drop=j: hop j is removed from the mixture, other learned scores kept."""
        class _Selector(torch.nn.Module):
            def __init__(self, original, calls_per_node):
                super().__init__()
                self.original = original
                self.n = calls_per_node
                self.calls = 0
            def forward(self, z):
                _k = self.calls % self.n
                self.calls += 1
                if force is not None:
                    return torch.full((z.size(0), 1), 50.0 if _k == force else -50.0, device=z.device, dtype=z.dtype)
                _score = self.original(z)
                return torch.full_like(_score, -50.0) if _k == drop else _score
        _saved = {nt: graph.hop_selector[nt] for nt in NODE_TYPES}
        try:
            for nt in NODE_TYPES:
                graph.hop_selector[nt] = _Selector(_saved[nt], graph.k_max + 1)
            yield
        finally:
            for nt in NODE_TYPES:
                graph.hop_selector[nt] = _saved[nt]

    @_contextmanager
    def adv_prune_propagation(graph, keep):
        """Remove edges only inside the gated propagation layer (the HGT layers still see every edge)."""
        _original = graph.adaptive_prop.forward
        _masks = {et: torch.as_tensor(np.asarray(m)) for et, m in keep.items()}
        def _patched(x_dict, edge_index_dict, edge_attr_dict, logits_dict, return_gates=False):
            _m = {et: _masks[et].to(edge_index_dict[et].device) for et in edge_index_dict}
            _ei = {et: edge_index_dict[et][:, _m[et]] for et in edge_index_dict}
            _ea = {et: edge_attr_dict[et][_m[et]] for et in edge_attr_dict}
            return _original(x_dict, _ei, _ea, logits_dict, return_gates=return_gates)
        graph.adaptive_prop.forward = _patched
        try:
            yield
        finally:
            del graph.adaptive_prop.forward

    def adv_prune_data(data, keep):
        """Remove edges from the whole graph (HGT layers and propagation both lose them)."""
        _d = data.clone()
        for et, m in keep.items():
            _m = torch.as_tensor(np.asarray(m))
            _d[et].edge_index = _d[et].edge_index[:, _m]
            _d[et].edge_attr = _d[et].edge_attr[_m]
        return _d

    def adv_edge_masks(gates, budget, mode, rng):
        """Keep-masks that remove round(budget * E) edges in EACH relation: 'high' = largest gates,
        'low' = smallest gates, 'random' = uniform. Same count per relation, so budgets are matched."""
        _keep = {}
        for et, g in gates.items():
            _n = len(g)
            _k = int(round(budget * _n))
            _m = np.ones(_n, dtype=bool)
            if _k > 0:
                if mode == "high":
                    _idx = np.argpartition(-g, _k - 1)[:_k]
                elif mode == "low":
                    _idx = np.argpartition(g, _k - 1)[:_k]
                else:
                    _idx = rng.choice(_n, _k, replace=False)
                _m[_idx] = False
            _keep[et] = _m
        return _keep

    def adv_region_dice(pred, seg):
        _t = _region_masks(seg)
        _out = {}
        for _name, _p in _region_masks(pred).items():
            _d = int(_p.sum()) + int(_t[_name].sum())
            _out[_name] = 2 * np.count_nonzero(_p & _t[_name]) / _d if _d else 1.0
        return _out

    def adv_graph_only_pred(probs, meta):
        return postprocess_prediction_auto(project_node_probs(probs, meta).copy())

    def adv_final_pred(head, probs, meta):
        _raw = sliding_window_predict(head, meta, probs, roi=VOXEL_MAX_SIZE, overlap=INFERENCE_OVERLAP)
        return postprocess_prediction_auto(_raw.copy())

    def adv_node_blocks(meta, vis, modalities):
        """Rebuild the per-node intensity block (mean, SD, 5 quantiles) of each requested modality from the
        (possibly perturbed) cropped volumes, exactly as build_hetero_case/node_table does."""
        _brain = np.asarray(meta["node_map"][NODE_TYPES[0]]) >= 0
        _out = {}
        for nt in NODE_TYPES:
            _lab = np.asarray(meta["node_map"][nt])[_brain].astype(np.int64)
            _k = int(_lab.max()) + 1
            _cnt = np.bincount(_lab, minlength=_k).astype(np.float64)
            _out[nt] = {}
            for _m in modalities:
                _v = np.asarray(vis[_m])[_brain].astype(np.float64)
                _s1 = np.bincount(_lab, weights=_v, minlength=_k)
                _s2 = np.bincount(_lab, weights=_v * _v, minlength=_k)
                _mean = _s1 / np.maximum(_cnt, 1)
                _std = np.sqrt(np.maximum(_s2 / np.maximum(_cnt, 1) - _mean * _mean, 0.0))
                _order = np.lexsort((_v, _lab))
                _ls, _vs = _lab[_order], _v[_order]
                _c = np.bincount(_ls, minlength=_k)
                _starts = np.concatenate(([0], np.cumsum(_c)[:-1]))
                _q = np.empty((_k, len(QUANTILES)), dtype=np.float64)
                for _qi, _qq in enumerate(QUANTILES):
                    _pos = _starts + np.floor(_qq * np.maximum(_c - 1, 0)).astype(np.int64)
                    _q[:, _qi] = _vs[np.minimum(_pos, len(_vs) - 1)]
                _out[nt][_m] = np.concatenate([_mean[:, None], _std[:, None], _q], axis=1).astype(np.float32)
        return _out

    def adv_with_blocks(data, blocks):
        _d = data.clone()
        for nt, per_modality in blocks.items():
            _x = _d[nt].x.clone()
            for _m, _block in per_modality.items():
                _x[:, adv_slice(_m)] = torch.from_numpy(_block).to(_x.dtype)
            _d[nt].x = _x
        return _d

    def adv_perturb(volume, brain, kind, rng):
        """Perturb one modality inside the brain mask (volumes are z-scored per brain, so 0 = brain mean)."""
        from scipy.ndimage import gaussian_filter as _gaussian_filter
        _v = np.asarray(volume).astype(np.float32).copy()
        _s = float(_v[brain].std()) or 1.0
        if kind == "dropout":
            _v[brain] = 0.0
        elif kind == "noise":
            _v[brain] += rng.normal(0.0, ADV_NOISE_SIGMA * _s, size=int(brain.sum())).astype(np.float32)
        elif kind == "bias":
            _f = _gaussian_filter(rng.normal(size=_v.shape).astype(np.float32), sigma=ADV_BIAS_SIGMA_VOXELS)
            _f = _f / (float(np.abs(_f[brain]).max()) + 1e-8)
            _v[brain] += ADV_BIAS_AMPLITUDE * _s * _f[brain]
        else:
            raise ValueError("Unknown perturbation: " + str(kind))
        return _v

    def adv_seed(*parts):
        import hashlib as _hashlib
        return int(_hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)

    def adv_boot_ci(values, draws=5000, seed=1942):
        _v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
        if len(_v) == 0:
            return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
        if len(_v) < 2:
            return dict(mean=float(_v.mean()), lo=float("nan"), hi=float("nan"), n=1)
        _means = np.random.default_rng(seed).choice(_v, size=(draws, len(_v)), replace=True).mean(axis=1)
        return dict(mean=float(_v.mean()), lo=float(np.quantile(_means, 0.025)), hi=float(np.quantile(_means, 0.975)), n=int(len(_v)))

    def adv_signflip_p(values, draws=5000, seed=1943):
        _v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
        if len(_v) < 2:
            return float("nan")
        _signs = np.random.default_rng(seed).choice((-1.0, 1.0), size=(draws, len(_v)))
        return float(((np.abs((_signs * _v).mean(axis=1)) >= abs(_v.mean()) - 1e-15).sum() + 1) / (draws + 1))

    def adv_holm(pvalues):
        _order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
        _adjusted, _previous = [0.0] * len(pvalues), 0.0
        for _rank, _i in enumerate(_order):
            _previous = max(_previous, min(1.0, (len(pvalues) - _rank) * pvalues[_i]))
            _adjusted[_i] = _previous
        return _adjusted

    def adv_jsonable(value):
        if isinstance(value, dict):
            return {("|".join(k) if isinstance(k, tuple) else str(k)): adv_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [adv_jsonable(v) for v in value]
        if isinstance(value, np.ndarray):
            return adv_jsonable(value.tolist())
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    def adv_save(name, value):
        """Save a JSON result inside the research store and push it to the Hub with a verified receipt."""
        adv_runner.store.save(ADV_OUT + "/" + name, adv_jsonable(value), push=True)
        adv_runner.store.flush()   # push=True only flushes every ZIP_SYNC_MINUTES; commit to the Hub now so a Molab restart cannot lose it
        print("saved + committed to the Hub:", ADV_OUT + "/" + name)

    return (ADV_RELATIONS, ADV_REGIONS, adv_boot_ci, adv_edge_masks, adv_final_pred, adv_graph_only_pred,
            adv_graph_probs, adv_hop_control, adv_holm, adv_jsonable, adv_node_blocks, adv_perturb,
            adv_prune_data, adv_prune_propagation, adv_read_state, adv_region_dice, adv_save, adv_seed,
            adv_signflip_p, adv_slice, adv_with_blocks)


@app.cell
def _(ADV_BOOT_DRAWS, ADV_REGIONS, adv_boot_ci, adv_case_id, adv_final_pred, adv_graph, adv_graph_only_pred,
      adv_graph_probs, adv_head, adv_hop_control, adv_read_state, adv_region_dice, adv_save, adv_test_all,
      adv_test_subset, load_meta, mo, np, NODE_TYPES, stage2_training_complete):
    # ---------------------------------------------------------------------------------------------
    # TEST 3c  Hop sensitivity.  Question: are the learned effective-hop values faithful, i.e. do nodes the
    # model says need neighbourhood information actually change prediction when that information is removed?
    #   force_j : every node may use ONLY information from graph distance j (j = 0, 1, 2)
    #   drop_j  : information from graph distance j is removed, the other learned weights are kept
    # Node-level metrics use all test patients; voxel-level Dice uses the fixed subset.
    # ---------------------------------------------------------------------------------------------
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 first."))
    from scipy.stats import spearmanr as _spearmanr
    _CONDITIONS = [("force_0", dict(force=0)), ("force_1", dict(force=1)), ("force_2", dict(force=2)),
                   ("drop_0", dict(drop=0)), ("drop_1", dict(drop=1)), ("drop_2", dict(drop=2))]
    _subset_ids = {adv_case_id(_i) for _i in adv_test_subset}
    _records = []
    _node_pool = {name: [] for name, _ in _CONDITIONS}   # (effective_hop of baseline-tumour nodes, flip indicator) per patient
    _checked = False
    for _item in adv_test_all:
        _cid = adv_case_id(_item)
        _data = _item[0]
        _vox = _cid in _subset_ids
        _base = adv_graph_probs(adv_graph, _data)
        _eff, _weights, _gates = adv_read_state(adv_graph)
        _base_cls = {nt: _base[nt].argmax(1) for nt in NODE_TYPES}
        _tumour = {nt: _base_cls[nt] != 0 for nt in NODE_TYPES}
        if _vox:
            _meta = load_meta(_item[1])
            _seg = _meta["seg"]
            _present = {r: bool(np.any({"WT": _seg > 0, "TC": (_seg == 1) | (_seg == 3), "ET": _seg == 3}[r])) for r in ADV_REGIONS}
            _g0 = adv_region_dice(adv_graph_only_pred(_base, _meta), _seg)
            _f0 = adv_region_dice(adv_final_pred(adv_head, _base, _meta), _seg)
        for _name, _kw in _CONDITIONS:
            with adv_hop_control(adv_graph, **_kw):
                _p = adv_graph_probs(adv_graph, _data)
                if not _checked and "force" in _kw:
                    _, _w, _ = adv_read_state(adv_graph)
                    for nt in NODE_TYPES:
                        assert np.all(_w[nt].argmax(1) == _kw["force"]), "hop forcing did not take effect"
            _flips = [(_p[nt].argmax(1) != _base_cls[nt])[_tumour[nt]] for nt in NODE_TYPES]
            _n_tumour = int(sum(len(f) for f in _flips))
            _row = dict(case=_cid, cond=_name, n_tumour_nodes=_n_tumour,
                        flip_rate=float(sum(f.sum() for f in _flips) / _n_tumour) if _n_tumour else float("nan"))
            _node_pool[_name].append((np.concatenate([_eff[nt][_tumour[nt]] for nt in NODE_TYPES]),
                                      np.concatenate(_flips).astype(float)))
            if _vox:
                _g = adv_region_dice(adv_graph_only_pred(_p, _meta), _seg)
                _f = adv_region_dice(adv_final_pred(adv_head, _p, _meta), _seg)
                for _reg in ADV_REGIONS:
                    _row["graph_delta_" + _reg] = _g[_reg] - _g0[_reg]
                    _row["final_delta_" + _reg] = _f[_reg] - _f0[_reg]
                    _row["present_" + _reg] = _present[_reg]
            _records.append(_row)
        _checked = True
    # ---- summaries --------------------------------------------------------------------------------
    hop_results = dict(conditions={}, faithfulness={})
    print("TEST 3c  hop sensitivity  (baseline = learned hop mixture; delta = condition - baseline)")
    print(f"{'condition':<9} {'flip rate (tumour nodes)':>26} " + " ".join(f"{'dGraphDice ' + r:>16}" for r in ADV_REGIONS) +
          " " + " ".join(f"{'dFinalDice ' + r:>16}" for r in ADV_REGIONS))
    for _name, _ in _CONDITIONS:
        _rows = [r for r in _records if r["cond"] == _name]
        _entry = dict(flip=adv_boot_ci([r["flip_rate"] for r in _rows], ADV_BOOT_DRAWS))
        _line = f"{_name:<9} {_entry['flip']['mean']:>12.4f} [{_entry['flip']['lo']:.4f},{_entry['flip']['hi']:.4f}]"
        for _level in ("graph", "final"):
            for _reg in ADV_REGIONS:
                _vals = [x[f"{_level}_delta_{_reg}"] for x in _rows if "present_" + _reg in x and x["present_" + _reg]]
                _entry[f"{_level}_delta_{_reg}"] = adv_boot_ci(_vals, ADV_BOOT_DRAWS)
                _line += f" {_entry[f'{_level}_delta_{_reg}']['mean']:>+9.4f}(n={_entry[f'{_level}_delta_{_reg}']['n']})"
        hop_results["conditions"][_name] = _entry
        print(_line)
    # faithfulness: does a higher learned effective hop go with a bigger prediction change when neighbour info is removed?
    for _name in ("force_0", "drop_1", "drop_2", "force_2"):
        _rhos = []
        for _eff_vec, _flip_vec in _node_pool[_name]:
            if len(_flip_vec) > 20 and _flip_vec.std() > 0 and _eff_vec.std() > 0:
                _rhos.append(float(_spearmanr(_eff_vec, _flip_vec).statistic))
        hop_results["faithfulness"][_name] = adv_boot_ci(_rhos, ADV_BOOT_DRAWS)
    print("\nPer-patient Spearman rho between learned effective hop and 'prediction flipped' (tumour nodes).")
    print("force_0 / drop_1 / drop_2 removing neighbour information: faithful hops give rho > 0 (CI above 0).")
    print("force_2 forcing maximum depth on nodes that chose few hops: faithful hops give rho < 0.")
    for _name, _v in hop_results["faithfulness"].items():
        print(f"  {_name:<8} mean rho {_v['mean']:+.4f}  95% CI [{_v['lo']:+.4f}, {_v['hi']:+.4f}]  (patients with variation: {_v['n']})")
    # terciles of the learned effective hop, pooled over patients
    _all_eff = np.concatenate([e for e, _ in _node_pool["force_0"]])
    _cuts = np.quantile(_all_eff, [1 / 3, 2 / 3])
    hop_results["tercile_cut_points"] = _cuts
    hop_results["tercile_flip_rate"] = {}
    print(f"\nFlip rate by tercile of learned effective hop (cut points {_cuts[0]:.3f}, {_cuts[1]:.3f}); low = node chose little neighbour information")
    for _name, _ in _CONDITIONS:
        _per_patient = [[], [], []]
        for _eff_vec, _flip_vec in _node_pool[_name]:
            _bins = np.digitize(_eff_vec, _cuts)
            for _b in range(3):
                if np.any(_bins == _b):
                    _per_patient[_b].append(float(_flip_vec[_bins == _b].mean()))
        hop_results["tercile_flip_rate"][_name] = [adv_boot_ci(_v, ADV_BOOT_DRAWS) for _v in _per_patient]
        print(f"  {_name:<8} low {hop_results['tercile_flip_rate'][_name][0]['mean']:.4f} | mid {hop_results['tercile_flip_rate'][_name][1]['mean']:.4f} | high {hop_results['tercile_flip_rate'][_name][2]['mean']:.4f}")
    adv_save("hop_sensitivity.json", dict(settings=dict(conditions=[c for c, _ in _CONDITIONS], voxel_patients=len(adv_test_subset), all_patients=len(adv_test_all)),
                                          results=hop_results, per_patient=_records))
    hop_done = True
    return (hop_done, hop_results)


@app.cell
def _(ADV_BOOT_DRAWS, ADV_EDGE_BUDGETS, ADV_RANDOM_DRAWS, ADV_REGIONS, ADV_RELATIONS, adv_boot_ci, adv_case_id,
      adv_edge_masks, adv_final_pred, adv_graph, adv_graph_only_pred, adv_graph_probs, adv_head, adv_holm,
      adv_prune_data, adv_prune_propagation, adv_read_state, adv_region_dice, adv_save, adv_seed, adv_signflip_p,
      adv_test_all, adv_test_subset, load_meta, mo, np, NODE_TYPES, stage2_training_complete):
    # ---------------------------------------------------------------------------------------------
    # TEST 3a  Edge perturbation.  Question: do high-gate edges matter more than low-gate edges?
    # For every patient the gate of each edge (mean alpha over the propagation hops) is read from the
    # unperturbed forward pass.  Then, with a MATCHED budget (the same number of edges removed in each
    # relation), edges are removed by: high gate / low gate / random.  Two scopes:
    #   propagation : edges disappear only inside the gated propagation layer (what the gates act on)
    #   whole_graph : edges disappear from the HGT layers as well (a stronger, less specific intervention)
    # Faithful gates => removing high-gate edges changes the tumour prediction MORE than low-gate / random.
    # Node-level metrics: all patients.  Voxel-level Dice: fixed subset (final model only for scope=propagation).
    # ---------------------------------------------------------------------------------------------
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 first."))
    _MODES = ("high", "low", "random")
    _SCOPES = ("propagation", "whole_graph")
    _subset_ids = {adv_case_id(_i) for _i in adv_test_subset}
    _records = []
    for _item in adv_test_all:
        _cid = adv_case_id(_item)
        _data = _item[0]
        _vox = _cid in _subset_ids
        _base = adv_graph_probs(adv_graph, _data)
        _eff, _w, _gates = adv_read_state(adv_graph)
        _base_cls = {nt: _base[nt].argmax(1) for nt in NODE_TYPES}
        _tumour = {nt: _base_cls[nt] != 0 for nt in NODE_TYPES}
        _tumour_prob = {nt: 1.0 - _base[nt][:, 0] for nt in NODE_TYPES}
        _edges = {et: _data[et].edge_index.cpu().numpy() for et in ADV_RELATIONS}
        if _vox:
            _meta = load_meta(_item[1])
            _seg = _meta["seg"]
            _present = {r: bool(np.any({"WT": _seg > 0, "TC": (_seg == 1) | (_seg == 3), "ET": _seg == 3}[r])) for r in ADV_REGIONS}
            _g0 = adv_region_dice(adv_graph_only_pred(_base, _meta), _seg)
            _f0 = adv_region_dice(adv_final_pred(adv_head, _base, _meta), _seg)
        for _q in ADV_EDGE_BUDGETS:
            for _mode in _MODES:
                for _scope in _SCOPES:
                    _draws = ADV_RANDOM_DRAWS if _mode == "random" else 1
                    _flip, _shift, _incident, _gd, _fd = [], [], [], [], []
                    for _r in range(_draws):
                        _rng = np.random.default_rng(adv_seed(_cid, _q, _mode, _scope, _r))
                        _keep = adv_edge_masks(_gates, _q, _mode, _rng)
                        if _scope == "propagation":
                            with adv_prune_propagation(adv_graph, _keep):
                                _p = adv_graph_probs(adv_graph, _data)
                        else:
                            _p = adv_graph_probs(adv_graph, adv_prune_data(_data, _keep))
                        _n_t = sum(int(_tumour[nt].sum()) for nt in NODE_TYPES)
                        _flip.append(sum(int(((_p[nt].argmax(1) != _base_cls[nt]) & _tumour[nt]).sum()) for nt in NODE_TYPES) / max(_n_t, 1))
                        _shift.append(float(np.mean(np.concatenate([np.abs((1.0 - _p[nt][:, 0]) - _tumour_prob[nt])[_tumour[nt]] for nt in NODE_TYPES]))) if _n_t else float("nan"))
                        _removed = 0
                        _inc = 0
                        for et in ADV_RELATIONS:
                            _gone = ~_keep[et]
                            _removed += int(_gone.sum())
                            _touch = _tumour[et[0]][_edges[et][0]] | _tumour[et[2]][_edges[et][1]]
                            _inc += int((_touch & _gone).sum())
                        _incident.append(_inc / max(_removed, 1))
                        if _vox and (_scope == "propagation" or _r == 0):
                            _g = adv_region_dice(adv_graph_only_pred(_p, _meta), _seg)
                            _gd.append({r: _g[r] - _g0[r] for r in ADV_REGIONS})
                            if _scope == "propagation" and _r == 0:
                                _f = adv_region_dice(adv_final_pred(adv_head, _p, _meta), _seg)
                                _fd.append({r: _f[r] - _f0[r] for r in ADV_REGIONS})
                    _row = dict(case=_cid, budget=_q, mode=_mode, scope=_scope, flip_rate=float(np.mean(_flip)),
                                tumour_prob_shift=float(np.nanmean(_shift)) if not np.all(np.isnan(_shift)) else float("nan"),
                                removed_tumour_incident_fraction=float(np.mean(_incident)))
                    if _gd:
                        for _reg in ADV_REGIONS:
                            _row["graph_delta_" + _reg] = float(np.mean([d[_reg] for d in _gd]))
                            _row["present_" + _reg] = _present[_reg]
                    if _fd:
                        for _reg in ADV_REGIONS:
                            _row["final_delta_" + _reg] = _fd[0][_reg]
                    _records.append(_row)
    # ---- summaries --------------------------------------------------------------------------------
    edge_results = dict(by_condition={}, contrasts=[])
    _index = {(r["case"], r["budget"], r["mode"], r["scope"]): r for r in _records}
    _cases = sorted({r["case"] for r in _records})
    print("TEST 3a  edge perturbation  (matched budget per relation; effects are changes relative to the unperturbed model)")
    for _scope in _SCOPES:
        print(f"\nscope = {_scope}")
        print(f"{'budget':>6} {'mode':<7} {'flip rate':>10} {'|d tumour prob|':>16} {'tumour-incident':>16} " +
              " ".join(f"{'dGraphDice ' + r:>16}" for r in ADV_REGIONS))
        for _q in ADV_EDGE_BUDGETS:
            for _mode in _MODES:
                _rows = [_index[(c, _q, _mode, _scope)] for c in _cases]
                _entry = dict(flip=adv_boot_ci([r["flip_rate"] for r in _rows], ADV_BOOT_DRAWS),
                              shift=adv_boot_ci([r["tumour_prob_shift"] for r in _rows], ADV_BOOT_DRAWS),
                              incident=adv_boot_ci([r["removed_tumour_incident_fraction"] for r in _rows], ADV_BOOT_DRAWS))
                _line = f"{_q:>6.2f} {_mode:<7} {_entry['flip']['mean']:>10.4f} {_entry['shift']['mean']:>16.4f} {_entry['incident']['mean']:>16.3f}"
                for _reg in ADV_REGIONS:
                    _vals = [x["graph_delta_" + _reg] for x in _rows if x.get("present_" + _reg)]
                    _entry["graph_delta_" + _reg] = adv_boot_ci(_vals, ADV_BOOT_DRAWS)
                    _line += f" {_entry['graph_delta_' + _reg]['mean']:>+9.4f}(n={_entry['graph_delta_' + _reg]['n']})"
                edge_results["by_condition"][f"{_scope}|{_q}|{_mode}"] = _entry
                print(_line)
    # primary contrasts: high minus low (and minus random) in the damage they cause.
    # damage = flip rate (node level, all patients); more flips => more influence.  For Dice, damage = -delta.
    print("\nContrasts (paired over patients).  Faithful gates: high-gate removal does MORE damage than low-gate / random.")
    _family = []
    for _scope in _SCOPES:
        for _q in ADV_EDGE_BUDGETS:
            for _other in ("low", "random"):
                _d = [_index[(c, _q, "high", _scope)]["flip_rate"] - _index[(c, _q, _other, _scope)]["flip_rate"] for c in _cases]
                _family.append(dict(scope=_scope, budget=_q, metric="flip_rate", contrast=f"high-{_other}", ci=adv_boot_ci(_d, ADV_BOOT_DRAWS), p=adv_signflip_p(_d)))
            for _reg in ADV_REGIONS:
                for _other in ("low", "random"):
                    _d = [-( _index[(c, _q, "high", _scope)]["graph_delta_" + _reg] - _index[(c, _q, _other, _scope)]["graph_delta_" + _reg])
                          for c in _cases if _index[(c, _q, "high", _scope)].get("present_" + _reg)]
                    _family.append(dict(scope=_scope, budget=_q, metric="graph_dice_damage_" + _reg, contrast=f"high-{_other}", ci=adv_boot_ci(_d, ADV_BOOT_DRAWS), p=adv_signflip_p(_d)))
    for _entry, _adj in zip(_family, adv_holm([e["p"] for e in _family])):
        _entry["holm_p"] = _adj
    for _e in _family:
        print(f"  {_e['scope']:<12} q={_e['budget']:.2f} {_e['metric']:<22} {_e['contrast']:<12} mean {_e['ci']['mean']:+.4f} CI [{_e['ci']['lo']:+.4f},{_e['ci']['hi']:+.4f}] n={_e['ci']['n']} Holm p={_e['holm_p']:.4f}")
    edge_results["contrasts"] = _family
    # final-model (graph + CNN) effect, propagation scope, subset
    print("\nFinal model (graph + CNN), scope = propagation, subset patients: mean change in Dice")
    edge_results["final"] = {}
    for _q in ADV_EDGE_BUDGETS:
        for _mode in _MODES:
            _rows = [_index[(c, _q, _mode, "propagation")] for c in _cases if "final_delta_WT" in _index[(c, _q, _mode, "propagation")]]
            _entry = {r: adv_boot_ci([x["final_delta_" + r] for x in _rows if x.get("present_" + r)], ADV_BOOT_DRAWS) for r in ADV_REGIONS}
            edge_results["final"][f"{_q}|{_mode}"] = _entry
            print(f"  q={_q:.2f} {_mode:<7} " + " ".join(f"{r} {_entry[r]['mean']:+.4f}" for r in ADV_REGIONS))
    adv_save("edge_perturbation.json", dict(settings=dict(budgets=ADV_EDGE_BUDGETS, modes=_MODES, scopes=_SCOPES, random_draws=ADV_RANDOM_DRAWS,
                                                          voxel_patients=len(adv_test_subset), all_patients=len(adv_test_all)),
                                            results=edge_results, per_patient=_records))
    edge_done = True
    return (edge_done, edge_results)


@app.cell
def _(ADV_BOOT_DRAWS, ADV_REGIONS, MODALITIES, NODE_TYPES, adv_boot_ci, adv_case_id, adv_final_pred, adv_graph,
      adv_graph_only_pred, adv_graph_probs, adv_head, adv_node_blocks, adv_perturb, adv_region_dice, adv_save,
      adv_seed, adv_test_subset, adv_with_blocks, exact_modality_shapley, load_meta, mo, np, stage2_training_complete,
      xai_background, QUANTILES):
    # ---------------------------------------------------------------------------------------------
    # TEST 3b  Modality intervention.  Question: do the Shapley modality rankings agree with what actually
    # happens to the prediction when a modality is degraded in a realistic way?
    # Perturbations act on the IMAGE volume (not on the feature vector that the Shapley values replace):
    #   dropout : modality missing (set to the brain mean)     noise : Gaussian noise
    #   bias    : smooth additive intensity-bias field
    # The node intensity features are rebuilt from the perturbed image exactly as in graph construction, and the
    # CNN sees the perturbed image too.  Reference = the same rebuild WITHOUT perturbation, so that float16
    # storage and rebuild effects cancel.
    # ---------------------------------------------------------------------------------------------
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 first."))
    from scipy.stats import kendalltau as _kendalltau
    _KINDS = ("dropout", "noise", "bias")
    _records = []
    _checks = []
    for _item in adv_test_subset:
        _cid = adv_case_id(_item)
        _data = _item[0]
        _meta = load_meta(_item[1])
        _seg = _meta["seg"]
        _brain = np.asarray(_meta["node_map"][NODE_TYPES[0]]) >= 0
        _vis = {m: np.asarray(_meta["vis"][m]) for m in MODALITIES}
        _present = {"WT": bool(np.any(_seg > 0)), "TC": bool(np.any((_seg == 1) | (_seg == 3))), "ET": bool(np.any(_seg == 3))}
        _ref_blocks = adv_node_blocks(_meta, _vis, MODALITIES)
        _cols = len(MODALITIES) * (2 + len(QUANTILES))
        _stored = float(np.mean([np.abs(_data[nt].x[:, :_cols].numpy() - np.concatenate([_ref_blocks[nt][m] for m in MODALITIES], axis=1)).max()
                                 for nt in NODE_TYPES]))
        _checks.append(_stored)
        _data_ref = adv_with_blocks(_data, _ref_blocks)
        _ref_probs = adv_graph_probs(adv_graph, _data_ref)
        _g_ref = adv_region_dice(adv_graph_only_pred(_ref_probs, _meta), _seg)
        _f_ref = adv_region_dice(adv_final_pred(adv_head, _ref_probs, _meta), _seg)
        # exact Shapley on the stored (unperturbed) graph; region attribution per modality (WT, TC, ET)
        _shap = exact_modality_shapley(_item, xai_background)
        _num = sum(_shap[nt]["region_attribution_sum"] for nt in NODE_TYPES)
        _den = sum(_shap[nt]["region_voxel_weight"] for nt in NODE_TYPES)
        _phi = {r: (_num[i] / _den[i] if _den[i] > 0 else float("nan")) for i, r in enumerate(ADV_REGIONS)}
        for _kind in _KINDS:
            for _m in MODALITIES:
                _rng = np.random.default_rng(adv_seed(_cid, _kind, _m))
                _pv = adv_perturb(_vis[_m], _brain, _kind, _rng)
                _blocks = adv_node_blocks(_meta, dict(_vis, **{_m: _pv}), (_m,))
                _p = adv_graph_probs(adv_graph, adv_with_blocks(_data_ref, _blocks))
                _meta_p = dict(_meta, vis=dict(_meta["vis"], **{_m: _pv.astype(np.float16)}))
                _g = adv_region_dice(adv_graph_only_pred(_p, _meta_p), _seg)
                _f = adv_region_dice(adv_final_pred(adv_head, _p, _meta_p), _seg)
                _row = dict(case=_cid, kind=_kind, modality=_m)
                for _reg in ADV_REGIONS:
                    _row["graph_drop_" + _reg] = _g_ref[_reg] - _g[_reg]
                    _row["final_drop_" + _reg] = _f_ref[_reg] - _f[_reg]
                    _row["present_" + _reg] = _present[_reg]
                    _row["phi_" + _reg] = float(_phi[_reg][MODALITIES.index(_m)]) if np.ndim(_phi[_reg]) else float("nan")
                _records.append(_row)
    print(f"Feature-rebuild check: mean over patients of max |stored - rebuilt| node feature = {np.mean(_checks):.5f}"
          " (float16 storage only; a large value means the rebuild is wrong and this test is invalid)")
    assert np.mean(_checks) < 0.05, "Rebuilt node features do not match the stored features; do not trust Test 3b"
    # ---- agreement between Shapley ranking and observed damage -------------------------------------------
    mod_results = dict(feature_rebuild_max_abs_diff=float(np.mean(_checks)), by_kind={}, cohort={})
    print("\nTEST 3b  modality intervention.  drop = reference Dice - perturbed Dice (bigger = the modality mattered more)")
    for _level in ("graph", "final"):
        for _kind in _KINDS:
            print(f"\n[{_level}-level damage, perturbation = {_kind}]")
            mod_results["by_kind"][f"{_level}|{_kind}"] = {}
            for _reg in ADV_REGIONS:
                _taus, _top1 = [], []
                for _cid in sorted({x["case"] for x in _records}):
                    _rows = [x for x in _records if x["case"] == _cid and x["kind"] == _kind]
                    if not _rows[0]["present_" + _reg]:
                        continue
                    _rows = sorted(_rows, key=lambda x: MODALITIES.index(x["modality"]))
                    _phi_vec = np.asarray([x["phi_" + _reg] for x in _rows])
                    _drop_vec = np.asarray([x[f"{_level}_drop_{_reg}"] for x in _rows])
                    if np.all(np.isfinite(_phi_vec)) and _phi_vec.std() > 0 and _drop_vec.std() > 0:
                        _taus.append(float(_kendalltau(_phi_vec, _drop_vec).statistic))
                        _top1.append(float(np.argmax(_phi_vec) == np.argmax(_drop_vec)))
                _mean_phi = np.asarray([np.nanmean([x["phi_" + _reg] for x in _records if x["kind"] == _kind and x["modality"] == m and x["present_" + _reg]]) for m in MODALITIES])
                _mean_drop = np.asarray([np.mean([x[f"{_level}_drop_{_reg}"] for x in _records if x["kind"] == _kind and x["modality"] == m and x["present_" + _reg]]) for m in MODALITIES])
                _entry = dict(kendall_tau=adv_boot_ci(_taus, ADV_BOOT_DRAWS), top1_agreement=adv_boot_ci(_top1, ADV_BOOT_DRAWS),
                              cohort_mean_shapley=dict(zip(MODALITIES, _mean_phi.tolist())),
                              cohort_mean_drop=dict(zip(MODALITIES, _mean_drop.tolist())),
                              cohort_rank_by_shapley=[MODALITIES[i] for i in np.argsort(-_mean_phi)],
                              cohort_rank_by_drop=[MODALITIES[i] for i in np.argsort(-_mean_drop)])
                mod_results["by_kind"][f"{_level}|{_kind}"][_reg] = _entry
                print(f"  {_reg}: patient-level Kendall tau {_entry['kendall_tau']['mean']:+.3f} [{_entry['kendall_tau']['lo']:+.3f},{_entry['kendall_tau']['hi']:+.3f}] (n={_entry['kendall_tau']['n']}); "
                      f"top-1 agreement {_entry['top1_agreement']['mean']:.2f} (chance 0.25); "
                      f"cohort ranking Shapley {_entry['cohort_rank_by_shapley']} vs damage {_entry['cohort_rank_by_drop']}")
    print("\nReading: tau clearly above 0 with a CI excluding 0, and top-1 agreement well above 0.25, means the Shapley ranking tracks real prediction changes."
          " tau near 0 means the Shapley ranking is only plausible-looking.")
    adv_save("modality_intervention.json", dict(settings=dict(kinds=_KINDS, patients=len(adv_test_subset)), results=mod_results, per_patient=_records))
    mod_done = True
    return (mod_done, mod_results)


@app.cell
def _(ADV_BOOT_DRAWS, ADV_OUT, ADV_RELATIONS, adv_boot_ci, adv_case_id, adv_final_pred, adv_graph, adv_graph_only_pred,
      adv_graph_probs, adv_head, adv_head_cnn, adv_prune_data, adv_region_dice, adv_runner, adv_save, adv_test_all,
      load_meta, mo, np, plt, stage2_training_complete):
    # ---------------------------------------------------------------------------------------------
    # TEST 2  Which wrong CNN predictions are corrected because of graph relationships (enhancing tumour first)?
    #   cnn  = cnn_only (same CNN family, MRI only)        full = graph prior + CNN (the proposed model)
    #   noedge = the SAME trained CNN fed a graph prior computed with ALL edges removed (nodes only).
    #            A voxel that full gets right, cnn gets wrong, and noedge also gets wrong needed the graph
    #            relationships.  (Caveat: the CNN was trained on priors that used the edges, so noedge is
    #            a dependence check, not a clean retrained ablation.)
    # All numbers are voxel counts pooled over the 189 test patients, plus island-level counts for ET.
    # ---------------------------------------------------------------------------------------------
    mo.stop(not stage2_training_complete, mo.md("Resume Stage 2 first."))
    from scipy.ndimage import label as _label, find_objects as _find_objects
    _MIN_ISLAND = 20
    _keep_none = {et: np.zeros(0, dtype=bool) for et in ADV_RELATIONS}
    _tot = dict(corrected=0, introduced=0, both_wrong=0, both_right=0)
    _by_class = {k: dict(corrected=0, introduced=0, gt_voxels=0) for k in (1, 2, 3)}
    _et = dict(fn_fixed=0, fp_removed=0, fn_introduced=0, fp_introduced=0, fn_fixed_prior_right=0,
               fn_fixed_needs_edges=0, fp_removed_needs_edges=0)
    _islands = dict(cnn_fp_islands=0, cnn_fp_removed_by_full=0, cnn_fp_removed_needs_edges=0,
                    cnn_fn_islands=0, cnn_fn_fixed_by_full=0, cnn_fn_fixed_needs_edges=0)
    _patients = []
    for _item in adv_test_all:
        _cid = adv_case_id(_item)
        _data = _item[0]
        _meta = load_meta(_item[1])
        _gt = _meta["seg"]
        _probs = adv_graph_probs(adv_graph, _data)
        _pf = adv_final_pred(adv_head, _probs, _meta)
        _pc = adv_final_pred(adv_head_cnn, None, _meta)
        _pn = adv_final_pred(adv_head, adv_graph_probs(adv_graph, adv_prune_data(_data, {et: np.zeros(_data[et].edge_index.shape[1], dtype=bool) for et in ADV_RELATIONS})), _meta)
        _pg = adv_graph_only_pred(_probs, _meta)
        _wrong_c, _right_f = _pc != _gt, _pf == _gt
        _corr, _intro = _wrong_c & _right_f, (_pc == _gt) & (_pf != _gt)
        _tot["corrected"] += int(_corr.sum()); _tot["introduced"] += int(_intro.sum())
        _tot["both_wrong"] += int((_wrong_c & ~_right_f).sum()); _tot["both_right"] += int((~_wrong_c & _right_f).sum())
        for _k in (1, 2, 3):
            _m = _gt == _k
            _by_class[_k]["gt_voxels"] += int(_m.sum())
            _by_class[_k]["corrected"] += int((_corr & _m).sum())
            _by_class[_k]["introduced"] += int((_intro & _m).sum())
        _g, _c, _f, _n, _p = _gt == 3, _pc == 3, _pf == 3, _pn == 3, _pg == 3
        _fn_fixed, _fp_removed = _g & ~_c & _f, ~_g & _c & ~_f
        _et["fn_fixed"] += int(_fn_fixed.sum()); _et["fp_removed"] += int(_fp_removed.sum())
        _et["fn_introduced"] += int((_g & _c & ~_f).sum()); _et["fp_introduced"] += int((~_g & ~_c & _f).sum())
        _et["fn_fixed_prior_right"] += int((_fn_fixed & _p).sum())
        _et["fn_fixed_needs_edges"] += int((_fn_fixed & ~_n).sum()); _et["fp_removed_needs_edges"] += int((_fp_removed & _n).sum())
        for _name, _mask in (("fp", _c & ~_g), ("fn", _g & ~_c)):
            _lab, _count = _label(_mask)
            for _i, _slc in enumerate(_find_objects(_lab), start=1):
                _isl = _lab[_slc] == _i
                if _isl.sum() < _MIN_ISLAND:
                    continue
                if _name == "fp":      # CNN false-positive ET island: cleared if >= half its voxels are no longer ET in full
                    _by_full = bool((~_f[_slc][_isl]).mean() >= 0.5)
                    _by_full_and_edges = _by_full and bool(_n[_slc][_isl].mean() >= 0.5)
                    _islands["cnn_fp_islands"] += 1
                    _islands["cnn_fp_removed_by_full"] += int(_by_full)
                    _islands["cnn_fp_removed_needs_edges"] += int(_by_full_and_edges)
                else:                  # true-ET island the CNN missed: recovered if >= half its voxels are ET in full
                    _by_full = bool(_f[_slc][_isl].mean() >= 0.5)
                    _by_full_and_edges = _by_full and bool((~_n[_slc][_isl]).mean() >= 0.5)
                    _islands["cnn_fn_islands"] += 1
                    _islands["cnn_fn_fixed_by_full"] += int(_by_full)
                    _islands["cnn_fn_fixed_needs_edges"] += int(_by_full_and_edges)
        _dc, _df, _dn = adv_region_dice(_pc, _gt), adv_region_dice(_pf, _gt), adv_region_dice(_pn, _gt)
        _patients.append(dict(case=_cid, et_voxels=int(_g.sum()), corrected=int(_corr.sum()), introduced=int(_intro.sum()),
                              **{f"dice_{r}_cnn": _dc[r] for r in _dc}, **{f"dice_{r}_full": _df[r] for r in _df}, **{f"dice_{r}_noedge": _dn[r] for r in _dn}))
    # ---- summaries --------------------------------------------------------------------------------
    corr_results = dict(totals=_tot, by_gt_class=_by_class, enhancing_tumour=_et, islands=_islands, min_island_voxels=_MIN_ISLAND)
    print("TEST 2  voxel-level corrections of cnn_only errors by the full (graph + CNN) model, 189 test patients")
    print(f"  CNN wrong -> full right (corrected): {_tot['corrected']:,}   CNN right -> full wrong (introduced): {_tot['introduced']:,}   net {_tot['corrected'] - _tot['introduced']:+,}")
    for _k, _name in ((1, "NCR/NET"), (2, "ED"), (3, "ET")):
        _d = _by_class[_k]
        print(f"  ground truth {_name:<8}: corrected {_d['corrected']:>9,}  introduced {_d['introduced']:>9,}  net {_d['corrected'] - _d['introduced']:>+9,}  (of {_d['gt_voxels']:,} voxels)")
    print("\nEnhancing tumour (binary ET mask)")
    print(f"  missed ET recovered by full (CNN false negative -> found):  {_et['fn_fixed']:,}   of which the graph-only prior already said ET: {_et['fn_fixed_prior_right']:,}")
    print(f"  false ET removed by full (CNN false positive -> cleared):  {_et['fp_removed']:,}")
    print(f"  new ET errors introduced by full:  false negatives {_et['fn_introduced']:,}, false positives {_et['fp_introduced']:,}")
    print(f"  relationship dependence (same CNN, graph prior with all edges removed): recovered ET that still needs the edges {_et['fn_fixed_needs_edges']:,};"
          f" cleared false ET that needs the edges {_et['fp_removed_needs_edges']:,}")
    print(f"\nET islands of at least {_MIN_ISLAND} voxels:  CNN false-positive islands {_islands['cnn_fp_islands']}, cleared by full {_islands['cnn_fp_removed_by_full']}"
          f" (needing the edges: {_islands['cnn_fp_removed_needs_edges']});  CNN missed true-ET islands {_islands['cnn_fn_islands']},"
          f" recovered by full {_islands['cnn_fn_fixed_by_full']} (needing the edges: {_islands['cnn_fn_fixed_needs_edges']})")
    _gain = [p["dice_ET_full"] - p["dice_ET_cnn"] for p in _patients if p["et_voxels"] > 0]
    corr_results["et_dice_gain_full_minus_cnn"] = adv_boot_ci(_gain, ADV_BOOT_DRAWS)
    _gain_edges = [p["dice_ET_full"] - p["dice_ET_noedge"] for p in _patients if p["et_voxels"] > 0]
    corr_results["et_dice_gain_full_minus_noedge"] = adv_boot_ci(_gain_edges, ADV_BOOT_DRAWS)
    print(f"\nPer-patient ET Dice (patients with ET): full - cnn_only {corr_results['et_dice_gain_full_minus_cnn']['mean']:+.4f}"
          f" [{corr_results['et_dice_gain_full_minus_cnn']['lo']:+.4f},{corr_results['et_dice_gain_full_minus_cnn']['hi']:+.4f}] (n={corr_results['et_dice_gain_full_minus_cnn']['n']});"
          f" full - noedge-prior {corr_results['et_dice_gain_full_minus_noedge']['mean']:+.4f}"
          f" [{corr_results['et_dice_gain_full_minus_noedge']['lo']:+.4f},{corr_results['et_dice_gain_full_minus_noedge']['hi']:+.4f}]")
    # qualitative figure: patients with the largest ET gain / loss
    _ranked = sorted([p for p in _patients if p["et_voxels"] > 0], key=lambda p: p["dice_ET_full"] - p["dice_ET_cnn"])
    _show = _ranked[-4:][::-1] + _ranked[:2]
    _by_id = {adv_case_id(i): i for i in adv_test_all}
    _fig, _axes = plt.subplots(len(_show), 5, figsize=(17, 3.3 * len(_show)), squeeze=False)
    for _row_i, _p in enumerate(_show):
        _item = _by_id[_p["case"]]
        _meta = load_meta(_item[1])
        _gt = _meta["seg"]
        _probs = adv_graph_probs(adv_graph, _item[0])
        _pf = adv_final_pred(adv_head, _probs, _meta)
        _pc = adv_final_pred(adv_head_cnn, None, _meta)
        _lo, _hi = _meta["lo"], _meta["hi"]
        _img = np.zeros(tuple(_meta["original_shape"]), dtype=np.float32)
        _img[_lo[0]:_hi[0], _lo[1]:_hi[1], _lo[2]:_hi[2]] = np.asarray(_meta["vis"]["t1ce"], dtype=np.float32)
        _z = int(np.argmax(((_gt == 3) | ((_pf == 3) != (_pc == 3))).sum(axis=(0, 1))))
        _corr_map = np.zeros(_gt.shape[:2] + (4,), dtype=np.float32)   # RGBA, fully transparent = no change
        _c_slice, _f_slice, _g_slice = _pc[:, :, _z], _pf[:, :, _z], _gt[:, :, _z]
        _corr_map[(_c_slice != _g_slice) & (_f_slice == _g_slice)] = (0.1, 0.8, 0.2, 0.9)
        _corr_map[(_c_slice == _g_slice) & (_f_slice != _g_slice)] = (0.9, 0.1, 0.1, 0.9)
        for _col, (_title, _arr) in enumerate((("T1ce", None), ("ground truth", _g_slice), ("cnn_only", _c_slice), ("full", _f_slice), ("green = fixed by full, red = broken by full", None))):
            _ax = _axes[_row_i][_col]
            _ax.imshow(np.rot90(_img[:, :, _z]), cmap="gray")
            if _arr is not None:
                _ax.imshow(np.rot90(np.ma.masked_where(_arr == 0, _arr)), cmap="jet", vmin=0, vmax=3, alpha=0.55)
            if _col == 4:
                _ax.imshow(np.rot90(_corr_map))
            _ax.set_title(f"{_p['case'][-5:]}  {_title}" if _col else f"{_p['case'][-5:]} ET Dice {_p['dice_ET_cnn']:.2f}->{_p['dice_ET_full']:.2f}", fontsize=9)
            _ax.axis("off")
    _fig.tight_layout()
    _fig_name = ADV_OUT + "/et_corrections.png"
    _fig.savefig(str(adv_runner.store.path(_fig_name)), dpi=140)
    adv_runner.store.push(_fig_name)
    adv_runner.store.flush()
    plt.show()
    adv_save("cnn_error_correction.json", dict(results=corr_results, per_patient=_patients))
    corr_done = True
    return (corr_done, corr_results)


@app.cell
def _(ADV_OUT, adv_runner, corr_done, corr_results, edge_done, edge_results, hop_done, hop_results, mod_done, mod_results, mo, stage2_handoff_notes):
    # Index file so the audit notebook can find the advisor follow-up outputs.
    mo.stop(not (corr_done and edge_done and hop_done and mod_done), mo.md("Finish all four tests first."))
    adv_runner.store.save(ADV_OUT + "/index.json", dict(
        plan_key=adv_runner.key,
        files=["hop_sensitivity.json", "edge_perturbation.json", "modality_intervention.json", "cnn_error_correction.json", "et_corrections.png"],
        note="Single seed (42); patient-level bootstrap CIs; see each file's settings. Inference only; no model was trained or changed.",
        part2_handoff_notes=list(stage2_handoff_notes)), push=True)
    adv_runner.store.flush()
    print("ADVISOR FOLLOW-UP TESTS COMPLETE: results are in", ADV_OUT, "of research store", adv_runner.key)
    return

if __name__ == "__main__":
    app.run()
