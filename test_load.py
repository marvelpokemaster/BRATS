import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def load_baseline_pipeline():
    with open("notebook33_part2.ipynb") as handle:
        notebook = json.load(handle)
    
    namespace = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "np": np,
        "MODALITIES": ("t1", "t1ce", "t2", "flair"),
        "NUM_CLASSES": 4,
        "ET_MIN_VOXELS": 500,
        "USE_CC_POSTPROCESS": False
    }
    
    utils_cell = next(cell for cell in notebook["cells"] if "def segmentation_metrics" in "".join(cell["source"]))
    exec("".join(utils_cell["source"]), namespace)
    
    baselines_cell = next(cell for cell in notebook["cells"] if cell.get("id") == "Baselines")
    exec("".join(baselines_cell["source"]), namespace)
    
    return namespace

ns = load_baseline_pipeline()
print(ns.keys())
