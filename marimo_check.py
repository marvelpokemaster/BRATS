import json
import ast

def check_marimo(notebook_path):
    with open(notebook_path, 'r') as f:
        data = json.load(f)

    defined_globals = {}
    errors = []

    for idx, cell in enumerate(data.get("cells", [])):
        if cell["cell_type"] != "code":
            continue
            
        # correctly handle newlines
        source_lines = cell["source"]
        if isinstance(source_lines, list):
            source = "".join(line if line.endswith('\n') else line + '\n' for line in source_lines)
        else:
            source = source_lines
            
        if not source.strip():
            continue
            
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            errors.append(f"Cell {idx} SyntaxError: {e}")
            continue

        cell_globals = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        cell_globals.add(target.id)
                    elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                        for elt in target.elts:
                            if isinstance(elt, ast.Name):
                                cell_globals.add(elt.id)
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    cell_globals.add(node.target.id)
            elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef) or isinstance(node, ast.ClassDef):
                cell_globals.add(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name.split('.')[0]
                    cell_globals.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    cell_globals.add(name)

        cell_globals = {g for g in cell_globals if not g.startswith('_')}
        
        for g in cell_globals:
            if g in defined_globals:
                errors.append(f"Multiple Definition Error: '{g}' defined in Cell {defined_globals[g]} and redefined in Cell {idx}")
            else:
                defined_globals[g] = idx

    if errors:
        print(f"--- Issues in {notebook_path} ---")
        for err in errors:
            print(err)
    else:
        print(f"--- {notebook_path} is PERFECT for Marimo ---")

check_marimo("notebook34_part1.ipynb")
check_marimo("notebook33_part2.ipynb")
check_marimo("part1_corrected.ipynb")
check_marimo("part2_corrected.ipynb")
check_marimo("research_experiments.ipynb")
