import json
import os

filepath = "/home/marvelpokemaster/Documents/brats_op/BRATS/part1_corrected.ipynb"
with open(filepath, "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "_ordered = sorted(graph_items, key=lambda item: os.path.basename(item[1]).replace(" in line:
                # Replace with the two lines
                source[i] = "_unique_items = {os.path.basename(m).replace(\".meta.pt\", \"\"): (d, m) for d, m in graph_items}\n"
                source.insert(i+1, "_ordered = sorted(_unique_items.values(), key=lambda item: os.path.basename(item[1]).replace(\".meta.pt\", \"\"))\n")
                break

with open(filepath, "w") as f:
    json.dump(nb, f, indent=1)
