# QoS-HRGN: Heterogeneous Adaptive Graph Learning for BraTS 3D Segmentation

A heterogeneous graph neural network pipeline for multimodal 3D brain tumor segmentation on BraTS MRI volumes.

---

## Architecture Overview

```
BraTS 3D MRI
     │
     ├── T1ce ──────► Modality-Specific 3D SLIC ──► T1ce Supervoxels
     │                                                      │
     └── FLAIR ─────► Modality-Specific 3D SLIC ──► FLAIR Supervoxels
                                                            │
                                             Heterogeneous Graph Construction
                                                            │
                                             Heterogeneous Graph Transformer (HGT)
                                                            │
                                             Uncertainty-Aware Adaptive Propagation (×2)
                                                            │
                                             Pathology-Aware Prediction (BG / NCR-NET / ED / ET)
                                                            │
                                             3D Voxel-Level Reconstruction
                                                            │
                                             WT / TC / ET Dice Evaluation
```

The pipeline explicitly decouples regional representation from tumor classification:
1. **Modality-Specific 3D SLIC Supervoxels:** Regionalizes T1ce and FLAIR volumes independently to preserve distinct tissue boundaries.
2. **Heterogeneous Multimodal Graph (`HeteroData`):**
   - **Node Types:** `t1ce`, `flair`
   - **Intra-Modal Relations:** `("t1ce", "spatial", "t1ce")`, `("flair", "spatial", "flair")`
   - **Cross-Modal Relations:** `("t1ce", "corresponds", "flair")`, `("flair", "corresponds", "t1ce")`
3. **HGT Backbone:** PyTorch Geometric `HGTConv` processes heterogeneous node features and typed relations with multi-head attention.
4. **Adaptive Multi-Hop Propagation:** Two propagation stages where edge message weights are dynamically modulated by:
   - Source and destination node embeddings
   - Normalized spatial Euclidean distance
   - Embedding cosine similarity
   - Cross-modal relation indicators
   - Prediction uncertainty (normalized Shannon entropy)
5. **Pathology-Aware Loss:** Supervoxel-level loss combining weighted cross-entropy with binary Dice losses on clinical target regions (Whole Tumor, Tumor Core, Enhancing Tumor).
6. **Voxel Reconstruction & Verification:** Reconstructs continuous 3D segmentation masks from dual-modality supervoxel probability maps with spatial alignment verification.

---

## Repository Structure

- [`QoS_HRGN_BraTS.ipynb`](QoS_HRGN_BraTS.ipynb): Complete, end-to-end runnable notebook containing dataset discovery, graph caching, training, validation, testing, voxel reconstruction, and slice visualizations.

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

## Dependencies

- Python 3.10+
- PyTorch & CUDA
- `torch-geometric`
- `nibabel`
- `scikit-image`
- `matplotlib`
- `kagglehub`
