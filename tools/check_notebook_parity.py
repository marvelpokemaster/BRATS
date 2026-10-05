import argparse
import ast
import difflib
import io
import json
from pathlib import Path
import textwrap
import tokenize


DEFAULT_PARTS = (
    "part1_corrected",
    "part2_refinement",
    "part3_experiments",
    "part4_slic_study",
    "part5_reports",
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def strip_marimo_package_comments(source):
    return "\n".join(
        line for line in source.splitlines()
        if not line.startswith("# packages added via marimo")
    ).strip()


def is_marimo_cell(node):
    return isinstance(node, ast.FunctionDef) and any(
        isinstance(decorator, ast.Attribute)
        and isinstance(decorator.value, ast.Name)
        and decorator.value.id == "app"
        and decorator.attr == "cell"
        for decorator in node.decorator_list
    )


def is_hidden_markdown_cell(node):
    if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
        return False
    value = node.body[0].value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "mo"
        and value.func.attr == "md"
    )


def signature_end_line(source, function):
    depth = 0
    found_def = False
    found_colon = False
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if not found_def:
            if token.type == tokenize.NAME and token.string == "def" and token.start[0] == function.lineno:
                found_def = True
            continue
        if token.type == tokenize.OP:
            if token.string in "([{":
                depth += 1
            elif token.string in ")]}":
                depth -= 1
            elif token.string == ":" and depth == 0:
                found_colon = True
        elif found_colon and token.type == tokenize.NEWLINE:
            return token.end[0]
    raise ValueError(f"Could not find signature end for cell at line {function.lineno}")


def cell_source(source, function):
    lines = source.splitlines()
    start = signature_end_line(source, function)
    trailing_return = function.body[-1] if function.body else None
    if isinstance(trailing_return, ast.Return):
        stop = trailing_return.lineno - 1
    else:
        stop = function.end_lineno
    body = "\n".join(lines[start:stop]).rstrip()
    return strip_marimo_package_comments(textwrap.dedent(body))


def python_cells(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    cells = [node for node in tree.body if is_marimo_cell(node)]
    if cells and is_hidden_markdown_cell(cells[0]):
        cells = cells[1:]
    return [cell_source(source, function) for function in cells]


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        cells.append(strip_marimo_package_comments(source))
    return cells


def print_cell_diff(part, index, python_source, notebook_source):
    print(f"\n{part} cell {index} differs:")
    diff = difflib.unified_diff(
        python_source.splitlines(keepends=True),
        notebook_source.splitlines(keepends=True),
        fromfile=f"{part}.py cell {index}",
        tofile=f"{part}.ipynb cell {index}",
    )
    print("".join(diff), end="")


def check_part(part):
    py_path = REPO_ROOT / f"{part}.py"
    notebook_path = REPO_ROOT / f"{part}.ipynb"
    py_cells = python_cells(py_path)
    nb_cells = notebook_cells(notebook_path)
    mismatches = 0
    shared_count = min(len(py_cells), len(nb_cells))
    for index in range(shared_count):
        if py_cells[index] != nb_cells[index]:
            mismatches += 1
            print_cell_diff(part, index + 1, py_cells[index], nb_cells[index])
    for index in range(shared_count, max(len(py_cells), len(nb_cells))):
        mismatches += 1
        py_source = py_cells[index] if index < len(py_cells) else ""
        nb_source = nb_cells[index] if index < len(nb_cells) else ""
        print_cell_diff(part, index + 1, py_source, nb_source)
    print(
        f"{part}: {len(py_cells)} Python cells, {len(nb_cells)} notebook code cells; "
        f"{mismatches} mismatches"
    )
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parts", nargs="*", choices=DEFAULT_PARTS, default=DEFAULT_PARTS)
    args = parser.parse_args()
    mismatches = sum(check_part(part) for part in args.parts)
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
