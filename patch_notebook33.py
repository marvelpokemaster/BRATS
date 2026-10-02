import json

with open("notebook33_part2.ipynb", "r") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell["cell_type"] == "code":
        source_str = "".join(cell["source"])
        
        # Patch voxel_loss
        if "def voxel_loss(" in source_str and "boundary_weight=0.0):" in source_str:
            new_source = []
            for line in cell["source"]:
                if "boundary_logits=None, boundary_weight=0.0):" in line:
                    line = line.replace("boundary_logits=None, boundary_weight=0.0):",
                        "boundary_logits=None, boundary_weight=0.0,\n" +
                        "               per_patient_dice=True):\n")
                if '    """Dice + ce_weight*CE' in line:
                    line = line.replace('    """Dice + ce_weight*CE',
                        '    """Dice + ce_weight*CE (+ focal_weight*Focal) (+ boundary_weight*BoundaryBCE) over the 4-class voxel problem.\n\n' +
                        '    logits: (B, 4, X, Y, Z), seg: (B, X, Y, Z) with values 0-3,\n' +
                        '    boundary_logits: (B, 1, X, Y, Z) raw logits of the boundary head.\n\n' +
                        '    When per_patient_dice=True (default), Dice is computed per volume and averaged\n' +
                        '    across the batch. This ensures consistent per-patient gradient weighting under\n' +
                        '    gradient accumulation and unequal microbatch sizes. For B=1, per-patient Dice\n' +
                        '    and global batch Dice are identical.\n' +
                        '    """\n')
                new_source.append(line)
            
            # Now replace the dice calculation block in the new source
            full_new_source = "".join(new_source)
            old_dice_block = """    dice = ((1 - (2 * (p_wt * g_wt).sum() + eps) / ((p_wt + g_wt).sum() + eps)) +
             1 - (2 * (p_tc * g_tc).sum() + eps) / ((p_tc + g_tc).sum() + eps)) / 3.0
    # ET Dice is the hardest and most important — weight it explicitly
    dice = dice + (1 - (2 * (p_et * g_et).sum() + eps) / ((p_et + g_et).sum() + eps)) / 3.0"""
            
            new_dice_block = """    if per_patient_dice and logits.dim() == 5:
        dims = (-3, -2, -1)
        d_wt = 1.0 - (2.0 * (p_wt * g_wt).sum(dim=dims) + eps) / (p_wt.sum(dim=dims) + g_wt.sum(dim=dims) + eps)
        d_tc = 1.0 - (2.0 * (p_tc * g_tc).sum(dim=dims) + eps) / (p_tc.sum(dim=dims) + g_tc.sum(dim=dims) + eps)
        d_et = 1.0 - (2.0 * (p_et * g_et).sum(dim=dims) + eps) / (p_et.sum(dim=dims) + g_et.sum(dim=dims) + eps)
        dice = ((d_wt + d_tc + d_et) / 3.0).mean()
    else:
        dice = ((1 - (2 * (p_wt * g_wt).sum() + eps) / ((p_wt + g_wt).sum() + eps)) +
                 1 - (2 * (p_tc * g_tc).sum() + eps) / ((p_tc + g_tc).sum() + eps)) / 3.0
        # ET Dice is the hardest and most important — weight it explicitly
        dice = dice + (1 - (2 * (p_et * g_et).sum() + eps) / ((p_et + g_et).sum() + eps)) / 3.0"""
            
            full_new_source = full_new_source.replace(old_dice_block, new_dice_block)
            
            # Reconstruct cell["source"] as list of lines with \n
            cell["source"] = [line + '\n' for line in full_new_source.split('\n')]
            # Remove the last empty newline added by split
            if cell["source"][-1] == '\n':
                cell["source"] = cell["source"][:-1]
            else:
                cell["source"][-1] = cell["source"][-1][:-1]

        # Patch baseline_case_input
        if "def baseline_case_input(meta):" in source_str:
            old_baseline = """def baseline_case_input(meta):
    \"\"\"Return CNN input and target using MRI-only crop coordinates.\"\"\"
    seg_c = crop(meta["seg"], meta["lo"], meta["hi"])
    lo, hi = baseline_mri_crop_bounds(meta)
    mri = np.stack([meta["vis"][m].astype(np.float32) for m in MODALITIES])
    x = torch.from_numpy(mri[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).float()
    y = torch.from_numpy(seg_c[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.int64))
    return x.unsqueeze(0), y.unsqueeze(0), lo, hi, seg_c.shape"""
            
            new_baseline = """def baseline_case_input(meta):
    \"\"\"Return CNN input and target using MRI-only crop coordinates.
    
    Independent of ground-truth segmentation during pure inference:
    if `meta['seg']` is absent or None, `y` returns None while `x`
    and spatial coordinates are computed strictly from MRI foreground.
    \"\"\"
    lo, hi = baseline_mri_crop_bounds(meta)
    mri = np.stack([meta["vis"][m].astype(np.float32) for m in MODALITIES])
    x = torch.from_numpy(mri[:, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]).float()
    crop_shape = (
        tuple(int(hi[i] - lo[i]) for i in range(3))
        if ("seg" not in meta or meta["seg"] is None)
        else crop(meta["seg"], meta["lo"], meta["hi"]).shape
    )
    if "seg" in meta and meta["seg"] is not None:
        seg_c = crop(meta["seg"], meta["lo"], meta["hi"])
        y = torch.from_numpy(seg_c[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.int64)).unsqueeze(0)
    else:
        y = None
    return x.unsqueeze(0), y, lo, hi, crop_shape"""
            
            full_source = "".join(cell["source"]).replace(old_baseline, new_baseline)
            
            old_assert = """    assert lo_a == lo_b and hi_a == hi_b and torch.equal(x_a, x_b)"""
            new_assert = """    assert lo_a == lo_b and hi_a == hi_b and torch.equal(x_a, x_b)
    # Also verify with seg completely omitted:
    no_seg = {k: v for k, v in meta.items() if k != "seg"}
    x_c, y_c, lo_c, hi_c, _ = baseline_case_input(no_seg)
    assert lo_a == lo_c and hi_a == hi_c and torch.equal(x_a, x_c) and y_c is None"""
            
            full_source = full_source.replace(old_assert, new_assert)
            
            cell["source"] = [line + '\n' for line in full_source.split('\n')]
            if cell["source"][-1] == '\n':
                cell["source"] = cell["source"][:-1]
            else:
                cell["source"][-1] = cell["source"][-1][:-1]

with open("notebook33_part2.ipynb", "w") as f:
    json.dump(data, f, indent=1)

print("Patched successfully")
