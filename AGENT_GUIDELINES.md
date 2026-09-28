# Agent Guidelines & Marimo Notebook Rules

This repository uses [Marimo](https://marimo.io/), a reactive Python notebook framework. Any AI coding agents (Antigravity, Devin, Cursor, etc.) modifying `.py` or `.ipynb` notebook files in this repository MUST adhere to the following ground rules to prevent breaking the reactive DAG (Directed Acyclic Graph):

## 1. No Variable Redeclaration
In Marimo, **global variables must be uniquely defined by a single cell**. 
- You **cannot** redeclare the same variable name in multiple cells.
- If you need to redefine a variable, you must do it within the exact same cell where it was originally defined, or mutate it locally.

## 2. Cell-Scoped Variables (Underscore Prefix)
If you need to create a temporary variable or a loop counter that shouldn't pollute the global state (and thus avoid "Multiple Definition" errors):
- **Prefix the variable name with an underscore** (e.g., `_x`, `_tmp`, `_i`).
- Variables beginning with an underscore are strictly **local to the cell** they are defined in and cannot be accessed by other cells.
- You *can* reuse the same underscore-prefixed variable name across multiple different cells without triggering a redeclaration error.

## 3. Avoid Cross-Cell Mutations
Marimo tracks state across cells. Mutating a global object (like appending to a list or modifying a dictionary) across different cells can lead to unpredictable reactive execution. 
- Try to define and mutate variables within the same cell, or create new variables (e.g., `cleaned_data = process(raw_data)`) rather than mutating `raw_data` in a downstream cell.

*Note: These rules were explicitly requested by the user and verified against the official Marimo documentation.*
