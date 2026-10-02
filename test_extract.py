import ast, json, torch
import torch.nn as nn, torch.nn.functional as F, numpy as np
def extract_notebook_definitions():
    with open("notebook33_part2.ipynb") as handle:
        notebook = json.load(handle)
    namespace = {
        "torch": torch, "nn": nn, "F": F, "np": np,
        "MODALITIES": ("t1", "t1ce", "t2", "flair"), "NUM_CLASSES": 4,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code": continue
        source = "".join(cell["source"])
        try:
            tree = ast.parse(source)
            clean_body = [
                node for node in tree.body 
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))
            ]
            if clean_body:
                tree.body = clean_body
                exec(compile(tree, filename="<ast>", mode="exec"), namespace)
        except Exception as e:
            print(f"Error parsing cell: {e}")
    return namespace

ns = extract_notebook_definitions()
print("ThreeDUNetBaseline:", "ThreeDUNetBaseline" in ns)
print("baseline_case_input:", "baseline_case_input" in ns)
print("baseline_cnn_predict:", "baseline_cnn_predict" in ns)
print("segmentation_metrics:", "segmentation_metrics" in ns)
