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
    # Practical BraTS v8 — five-session compute plan

    Fresh Part 1 is required. Keep all matching helpers beside the notebook and fill the blank HF token locally.
    15k SLIC, HGT, learned 0–2 extra hops; structural refinement and reconstruction are off.
    One seed; declared 135-minute model-training allowances, with best completed validation checkpoints.
    A compute-budget stop is not convergence. See RUN_NEXT.md and ADVISOR_SCOPE.md before running.
    Session work stops at 9.5 hours to reserve time for ZIP transfer. Individual operations/transfers are not preemptible.
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
    GPU_MEMORY_FRACTION = 0.70
    VOXEL_EFFECTIVE_BATCH_SIZE = 16
    VOXEL_MAX_MICROBATCH = 4
    INFERENCE_MAX_BATCH = 8
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
    GRAPH_MEMORY_FRACTION = 0.75

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
            "phase_seconds_limit": 8100, "compute_protocol": "practical-v8",
            "graph_training_protocol": "practical-v8-patient-mean",
            "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "dice_weight": DICE_WEIGHT, "et_focal_weight": ET_FOCAL_WEIGHT, "focal_gamma": FOCAL_GAMMA,
            "amp": USE_AMP, "hop_reg_weight": HOP_REG_WEIGHT, "structural_aux_weight": STRUCTURAL_AUX_WEIGHT,
            "lambda_rec": LAMBDA_REC,
        },
        "voxel": {
            "phase_seconds_limit": 8100, "compute_protocol": "practical-v8",
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

    return focal_loss_soft, region_dice_loss, soft_cross_entropy


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
    for _key,_expected in {"stage1_sha256":STAGE1_MODEL_SHA256,"split":SPLIT_CASE_IDS,
            "dataset_fingerprint":DATASET_FINGERPRINT,"stage2_config":RUN_CONFIG["voxel"]}.items():
        if MAIN_STAGE2_STATE.get(_key)!=_expected:
            raise RuntimeError("Part 2 handoff mismatch: "+_key)
    stage2_training_complete = True
    return MAIN_STAGE2_STATE, stage2_training_complete


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
    research_status = run_research(RESEARCH_CONTEXT,RESEARCH_SETTINGS)
    print(research_status)
    mo.stop(research_status.get("paused") or not research_status.get("report_ready"),
            mo.md("Evaluation paused. Rerun Part 5 to restore saved patient results and continue."))
    RESEARCH_REPORT_READY = True
    return (
        BudgetPause,
        RESEARCH_CONTEXT,
        RESEARCH_REPORT_READY,
        research_status,
    )


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
def _(
    CLASS_NAMES,
    HOPS,
    K_MAX,
    NODE_TYPES,
    NUM_CLASSES,
    PERSISTENT_BASE,
    base_model,
    mo,
    np,
    os,
    plt,
    stage2_training_complete,
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
        plt.savefig(os.path.join(PERSISTENT_BASE,"figures", "local_graph.png" if "target_type" in locals() else "gate_hop_summary.png"),dpi=180)
        plt.close(fig)
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
        plt.savefig(os.path.join(PERSISTENT_BASE,"figures", "local_graph.png" if "target_type" in locals() else "gate_hop_summary.png"),dpi=180)
        plt.close(fig)
        return {'relation_names': relation_names, 'gate_by_class': gate_matrix, 'hop_rows': hop_rows, 'effective_hop_by_class': hop_matrix}

    return explain_local_hetero_graph, global_graph_explanation_summary


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
    model,
    np,
    plt,
    rec_module,
    sample_hard_negatives,
    torch,
):
    def _auc(scores, labels):
        """Rank-based ROC-AUC (no sklearn dependency)."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        n_pos, n_neg = int((labels == 1).sum()), int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            return float("nan")
        from scipy.stats import rankdata
        ranks = rankdata(scores,method="average")
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

    return (rec_recover_case,)


@app.cell
def _(
    MAIN_STAGE2_STATE,
    MAIN_STORE,
    PERSISTENT_BASE,
    RESEARCH_CONTEXT,
    SEED,
    SESSION_DEADLINE,
    STAGE1_MODEL_SHA256,
    ZIP_CONTEXT,
    base_model,
    exact_modality_shapley,
    explain_local_hetero_graph,
    forward_model,
    global_graph_explanation_summary,
    make_store,
    mask_hetero_graph,
    model,
    rec_module,
    rec_recover_case,
    reconstruction_loss,
    research_status,
    stage1_checkpoint,
    xai_background,
):
    from brats_reporting import run_advisor_reports
    REPORT_CONTEXT = dict(RESEARCH_CONTEXT, model=model,base_model=base_model,rec_module=rec_module,
        exact_modality_shapley=exact_modality_shapley,xai_background=xai_background,
        explain_local_hetero_graph=explain_local_hetero_graph,global_graph_explanation_summary=global_graph_explanation_summary,
        PERSISTENT_BASE=PERSISTENT_BASE,SESSION_DEADLINE=SESSION_DEADLINE,
        stage1_checkpoint=stage1_checkpoint,MAIN_STAGE2_STATE=MAIN_STAGE2_STATE,
        forward_model=forward_model, mask_hetero_graph=mask_hetero_graph,reconstruction_loss=reconstruction_loss,
        REC_VAL_SEED=SEED+1234,STAGE1_MODEL_SHA256=STAGE1_MODEL_SHA256,
        rec_recover_case=rec_recover_case,RESEARCH_STUDY_KEY=research_status["plan"],MAIN_STORE=MAIN_STORE)
    advisor_report_status = run_advisor_reports(REPORT_CONTEXT,make_store(ZIP_CONTEXT,"advisor_reports"))
    print(advisor_report_status)
    return


if __name__ == "__main__":
    app.run()
