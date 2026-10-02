import json

with open("notebook33_part2.ipynb", "r") as f:
    data = json.load(f)

new_cell_30_source = """# ============================================================
# VoxelRefinementHead: 3D U-Net that refines node predictions at voxel resolution
# ============================================================
class VoxelRefinementHead(nn.Module):
    \"\"\"Lightweight 3D U-Net for voxel-level refinement.

    Input: (B, 8, X, Y, Z) — 4 MRI modalities + 4 projected node probability maps
    Output: ((B, 4, X, Y, Z), (B, 1, X, Y, Z)) — 4-class voxel-level logits and
    a boundary-map logit predicted from the same full-resolution decoder
    features, so the shared decoder is explicitly supervised on where label
    transitions are (the hardest voxels for a supervoxel-projected prior).

    Can exceed the supervoxel oracle ceiling because it operates at voxel
    resolution and can split ET from NCR/NET within the same supervoxel using
    T1ce intensity patterns.
    \"\"\"

    def __init__(self, in_channels=8, num_classes=4, base_channels=32):
        super().__init__()
        c = base_channels
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
    \"\"\"Ground-truth boundary map from a label volume via morphological gradient.

    One-hot the labels, then dilate and erode each class channel with a 3x3x3
    structuring element (`F.max_pool3d`, erosion = -max_pool3d(-x)). A voxel
    where dilation and erosion disagree for any class sits on a label
    transition. Returns (B, 1, X, Y, Z) floats in {0, 1}.
    \"\"\"
    oh = F.one_hot(seg, num_classes).permute(0, 4, 1, 2, 3).float()
    dil = F.max_pool3d(oh, kernel_size=3, stride=1, padding=1)
    ero = -F.max_pool3d(-oh, kernel_size=3, stride=1, padding=1)
    return (dil - ero).amax(dim=1, keepdim=True).clamp(0, 1)


def voxel_loss(logits, seg, class_weights=None, dice_weight=1.0,
               ce_weight=1.0, focal_weight=0.0, focal_gamma=2.0,
               boundary_logits=None, boundary_weight=0.0,
               per_patient_dice=True):
    \"\"\"Dice + ce_weight*CE (+ focal_weight*Focal) (+ boundary_weight*BoundaryBCE) over the 4-class voxel problem.

    logits: (B, 4, X, Y, Z), seg: (B, X, Y, Z) with values 0-3,
    boundary_logits: (B, 1, X, Y, Z) raw logits of the boundary head.

    When per_patient_dice=True (default), Dice is computed per volume and averaged
    across the batch. This ensures consistent per-patient gradient weighting under
    gradient accumulation and unequal microbatch sizes. For B=1, per-patient Dice
    and global batch Dice are identical.
    \"\"\"
    ce = F.cross_entropy(logits, seg, weight=class_weights)
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
    \"\"\"Graph model + voxel refinement head → full-volume prediction.\"\"\"
    node_probs = predict_node_probs(model, data)
    probs_c = project_node_probs_cropped(node_probs, meta)   # (X, Y, Z, 4)
    mri_c = np.stack([meta["vis"][m].astype(np.float32) for m in MODALITIES])  # (4, X, Y, Z)
    seg_c = crop(meta["seg"], meta["lo"], meta["hi"])
    tlo, thi = tumour_crop_bounds(seg_c)
    pred_c = probs_c.argmax(axis=-1).astype(np.uint8)        # graph-only in cropped region
    if tlo is not None:
        x = np.concatenate([
            mri_c[:, tlo[0]:thi[0], tlo[1]:thi[1], tlo[2]:thi[2]],
            probs_c.transpose(3, 0, 1, 2)[:, tlo[0]:thi[0], tlo[1]:thi[1], tlo[2]:thi[2]]
        ], axis=0)                                            # (8, Xt, Yt, Zt)
        x = torch.from_numpy(x).float().unsqueeze(0).to(device)
        logits, _boundary_logits = voxel_head(x)   # boundary head is auxiliary: training only
        pred_t = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        pred_c[tlo[0]:thi[0], tlo[1]:thi[1], tlo[2]:thi[2]] = pred_t
    pred = np.zeros(tuple(meta["original_shape"]), dtype=np.uint8)
    pred[meta["lo"][0]:meta["hi"][0], meta["lo"][1]:meta["hi"][1], meta["lo"][2]:meta["hi"][2]] = pred_c
    return postprocess_prediction_auto(pred) if postprocess else pred"""

cell_source = [line + '\n' for line in new_cell_30_source.split('\n')]
if cell_source[-1] == '\n':
    cell_source = cell_source[:-1]
else:
    cell_source[-1] = cell_source[-1][:-1]

data["cells"][30]["source"] = cell_source

with open("notebook33_part2.ipynb", "w") as f:
    json.dump(data, f, indent=1)

print("Fixed Cell 30")
