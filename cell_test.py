def _(
    hashlib,
    json,
    os,
    random,
    torch,
    valid_cases,
):
    if len(graph_items) != len(valid_cases):
        raise RuntimeError("Graph build is incomplete; resume cache construction before splitting")
    _unique_items = {os.path.basename(m).replace(".meta.pt", ""): (d, m) for d, m in graph_items}
    _ordered = sorted(_unique_items.values(), key=lambda item: os.path.basename(item[1]).replace(".meta.pt", ""))
    random.Random(SEED).shuffle(_ordered)
    n_items = len(_ordered)
    n_train = max(1, int(0.7 * n_items))
    n_val = max(1, int(0.15 * n_items)) if n_items >= 7 else 1
    train_items = _ordered[:n_train]
    val_items = _ordered[n_train:n_train + n_val]
    test_items = _ordered[n_train + n_val:]
    if not all((train_items, val_items, test_items)):
        raise RuntimeError("Need at least three disjoint, nonempty splits")
    SPLIT_CASE_IDS = {
        "train": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in train_items],
        "val": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in val_items],
        "test": [os.path.basename(_m).replace(".meta.pt", "") for _, _m in test_items],
    }
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["val"]))
    assert not (set(SPLIT_CASE_IDS["train"]) & set(SPLIT_CASE_IDS["test"]))
    assert not (set(SPLIT_CASE_IDS["val"]) & set(SPLIT_CASE_IDS["test"]))
