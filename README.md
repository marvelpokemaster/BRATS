# QoS-HRGN: Heterogeneous Adaptive Graph Learning for BraTS 3D Segmentation

A heterogeneous graph neural network pipeline for multimodal 3D brain tumor segmentation on BraTS 2020 MRI volumes. The pipeline combines modality-specific supervoxel graphs, heterogeneous graph transformers, node-wise learned propagation depth, voxel-level refinement, and multi-channel explainability.

---

## Architecture Overview

```
BraTS 3D MRI (T1, T1ce, T2, FLAIR)
     │
     ├── T1ce  ──► Modality-Specific 3D SLIC ──► T1ce Supervoxels  ──┐
     │                                                              │
     └── FLAIR ──► Modality-Specific 3D SLIC ──► FLAIR Supervoxels ──┤
                                                                   │
                                          Heterogeneous Graph Construction
                                          (spatial + cross-modal correspondence edges)
                                                                   │
                                          Residual HGT (2 layers, 4 heads)
                                                                   │
                                          Node-Wise Adaptive Hop Propagation
                                          (shared recurrent, k=0..K_MAX=4,
                                           uncertainty-conditioned soft mixture)
                                                                   │
                                          Per-Node-Type Segmentation Head
                                                                   │
                                          Voxel Refinement Head (3D U-Net)
                                          (breaks supervoxel oracle ceiling)
                                                                   │
                                          WT / TC / ET Dice Evaluation
                                                                   │
                                          Multi-Channel Explainability
                                          (Shapley + edge gates + effective hops)
```

The pipeline explicitly decouples regional representation from tumor classification:

1. **Modality-Specific 3D SLIC Supervoxels:** Regionalizes T1ce and FLAIR volumes independently to preserve distinct tissue boundaries. Each supervoxel summarizes all four MRI modalities (mean, std, 5 quantiles per modality = 28 appearance features) plus 4 geometry features (normalized centroid xyz, log-relative size) = 32 features per node.

2. **Heterogeneous Multimodal Graph (`HeteroData`):**
   - **Node Types:** `t1ce`, `flair` (each with its own SLIC partition)
   - **Intra-Modal Relations:** `("t1ce", "spatial", "t1ce")`, `("flair", "spatial", "flair")` — face-adjacent supervoxels with edge attributes [normalized centroid distance, log-size ratio]
   - **Cross-Modal Relations:** `("t1ce", "corresponds", "flair")`, `("flair", "corresponds", "t1ce")` — supervoxels that overlap in voxel space, with the same edge attributes

3. **HGT Backbone:** PyTorch Geometric `HGTConv` (2 layers, 4 heads) processes heterogeneous node features and typed relations with multi-head attention. Residual connections with LayerNorm.

4. **Node-Wise Adaptive Hop Propagation:** Instead of fixing the number of propagation stages, the model computes candidate representations at depths k=0,...,K_MAX (default 4) using a single shared recurrent propagation layer. Each node learns a soft distribution β_{v,k} over these depths, conditioned on:
   - Node embedding at that hop
   - Prediction uncertainty (normalized Shannon entropy)
   - Representation change from the preceding hop
   - Node type (separate selectors for T1ce and FLAIR)
   
   The final embedding is a weighted mixture: h_v* = Σ_k β_{v,k} · h_v^{(k)}. An expected-hop regularization term discourages collapse to maximum depth. This lets ET nodes prefer local context while edema nodes may select broader propagation — without assuming every node needs the same receptive field.

5. **Adaptive Edge Gating:** Within each propagation step, per-edge messages are gated by α = σ(MLP([h_src, h_dst, edge_attr, cosine_similarity, cross_modal_flag, uncertainty])). This is the model's native edge-importance signal and is exposed for explainability.

6. **Pathology-Aware Loss:** Supervoxel-level loss combining weighted soft cross-entropy with volume-weighted binary Dice losses on clinical target regions (WT, TC, ET), plus optional soft focal loss for ET.

7. **Voxel Refinement Head:** A lightweight 3D U-Net (~1M params) that takes 4 MRI modalities + 4 projected node probability maps and refines predictions at full voxel resolution. This breaks the supervoxel oracle ceiling (ET oracle ≈ 0.72 at 15k segments) by discriminating within supervoxels — e.g. splitting ET from NCR/NET using T1ce intensity. Two-stage training: graph model frozen.

8. **Connected-Component Post-Processing:** Removes individual small ET islands rather than using a single global volume threshold, preserving large valid ET regions while suppressing spurious detections.

---

## Explainability

Three complementary explanation channels, each answering a different question:

### 1. Exact Modality Shapley Attribution
Adapted from Saueressig et al. (DataMod 2020). The four MRI modalities (T1, T1CE, T2, FLAIR) are treated as cooperative-game players. All 2^4 = 16 modality coalitions are evaluated exactly (no neural approximation). Missing modalities are replaced by a 500-node training-background mean; geometry and graph topology remain unchanged. All nodes in a patient graph are perturbed simultaneously because GNN predictions depend on neighbors.

**Output:** Four-panel violin plots grouped by predicted class × modality × bright/dark intensity, matching the paper's presentation. Positive values support the predicted class; negative values oppose it.

### 2. Labelled Local Heterogeneous Graph
A bounded k-hop neighborhood around a clinically relevant target node (default: highest-confidence predicted ET). Renders:
- **Node fill color:** predicted tissue class
- **Node shape:** circle = T1ce, square = FLAIR
- **Node size:** prediction confidence
- **Node border color:** learned effective hop depth
- **Node labels:** type, ID, predicted class, confidence, effective hop
- **Edge color:** relation type (spatial vs correspondence)
- **Edge width/opacity/numeric label:** learned adaptive gate α

### 3. Global Relation and Receptive-Field Heatmaps
- Mean learned gate α by relation type and destination class
- Mean effective hop by node type, class, and prediction correctness

---

## Ablation Structure

| Run | `USE_ADAPTIVE_HOPS` | `HOPS`/`K_MAX` | `USE_VOXEL_REFINEMENT` | `ET_FOCAL_WEIGHT` | What it isolates |
|---|---|---|---|---|---|
| HGT only | `False` | `0` | `False` | `0` | No adaptive propagation |
| Fixed propagation | `False` | `2` | `False` | `0` | Original QoS-HRGN baseline |
| + focal loss | `False` | `2` | `False` | `0.5` | Focal loss helps close gap to oracle |
| Adaptive depth | `True` | `4` | `False` | `0.5` | Node-wise learned receptive field |
| + voxel head | `True` | `4` | `True` | `0.5` | Full pipeline |
| No hop cost | `True` | `4` | `False` | `0.5` | Tests expected-hop penalty importance |

Additional ablations:
- **Option B:** Structure-aware graph refinement (structural descriptors from the model's own predicted tumour graph)
- **Option C:** Heterogeneous masked graph reconstruction (auxiliary self-supervised task)

---

## BraTS Label Formulation

- **Raw Labels:** `0`: Background, `1`: NCR/NET, `2`: ED, `4`: ET
- **Internal Mapping:**
  - `0`: Background (BG)
  - `1`: Necrotic / Non-Enhancing Tumor (NCR/NET)
  - `2`: Peritumoral Edema (ED)
  - `3`: Enhancing Tumor (ET)
- **Evaluation Subregions:**
  - **WT (Whole Tumor):** Classes 1, 2, 3
  - **TC (Tumor Core):** Classes 1, 3
  - **ET (Enhancing Tumor):** Class 3

---

## Key Configuration

```python
# Graph
N_SEGMENTS = 15000          # SLIC supervoxel count
COMPACTNESS = 0.3
NODE_TYPES = ("t1ce", "flair")
NODE_FEAT_DIM = 32           # 28 appearance + 4 geometry

# Model
HIDDEN_DIM = 128
HEADS = 4
HGT_LAYERS = 2
USE_ADAPTIVE_HOPS = True    # node-wise learned propagation depth
K_MAX = 4                   # candidate depths k=0..4
HOP_REG_WEIGHT = 0.002      # expected-hop regularization

# Training
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-3

# Voxel refinement
USE_VOXEL_REFINEMENT = True
VOXEL_EPOCHS = 40
VOXEL_BASE_CHANNELS = 32

# Loss
ET_FOCAL_WEIGHT = 0.5
FOCAL_GAMMA = 2.0
```

---

## Repository Structure

- [`QoS_HRGN_BraTS.ipynb`](QoS_HRGN_BraTS.ipynb): Complete end-to-end runnable notebook containing dataset discovery, graph caching, training, validation, testing, voxel refinement, explainability analysis, and visualizations.

---

## Dependencies

- Python 3.10+
- PyTorch & CUDA
- `torch-geometric`
- `nibabel`
- `scikit-image`
- `matplotlib`
- `kagglehub`
- `scipy`
- `joblib`

---

## References

- Saueressig, C., Berkley, A., Kang, E., Munbodh, R., & Singh, R. (2020). *Exploring graph-based neural networks for automatic brain tumor segmentation.* DataMod 2020. — SLIC supervoxel construction, k≈15000, quantile features, and SHAP-based modality explainability.
- Hu, Z., Dong, Y., Wang, K., & Sun, Y. (2020). *Heterogeneous Graph Transformer.* WWW 2020. — HGTConv backbone.
- BraTS 2020 challenge dataset via Kaggle (`awsaf49/brats20-dataset-training-validation`).
