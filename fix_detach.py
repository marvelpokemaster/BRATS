import json

for file in ["notebook34_part1.ipynb", "notebook33_part2.ipynb"]:
    try:
        with open(file, "r") as f:
            data = json.load(f)
        
        changed = False
        for cell in data.get("cells", []):
            if cell["cell_type"] == "code":
                new_source = []
                for line in cell["source"]:
                    if "_logits.detach()" in line and "node_region_dice" in line:
                        line = line.replace("_logits.detach()", "{k: v.detach() for k, v in _logits.items()}")
                        changed = True
                    new_source.append(line)
                cell["source"] = new_source
                
        if changed:
            with open(file, "w") as f:
                json.dump(data, f, indent=1)
            print(f"Fixed {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")
