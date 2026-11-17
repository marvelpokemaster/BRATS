"""Plan adoption: Part 5 may reuse Part 3's trained rows after a source-only bug fix."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from brats_protocol import adopt_recorded_plan

ALLOWED = {"brats_practical.py", "brats_experiments.py", "brats_protocol.py"}


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def manifest(**overrides):
    base = dict(version="research-v8-practical", rows=[dict(name="full"), dict(name="segresnet")], seeds=[42],
                split=dict(train=["a"], val=["b"], test=["c"]), debug=False, class_weights=[1.0, 2.0],
                source_hashes={"brats_practical.py": "old", "brats_experiments.py": "e", "brats_gpu.py": "g",
                               "brats_protocol.py": "p", "brats_graph.py": "h"})
    base.update(overrides)
    return base


def test_source_only_change_is_adopted_and_disclosed():
    stored = manifest()
    key = identity(stored)[:20]
    running = manifest(source_hashes=dict(stored["source_hashes"], **{"brats_practical.py": "new"}))
    assert identity(running)[:20] != key  # the bug being fixed
    adopted, amendment = adopt_recorded_plan(stored, running, key, identity, ALLOWED)
    assert adopted is stored
    assert amendment["adopted_key"] == key and amendment["recomputed_key"] == identity(running)[:20]
    assert amendment["changed_sources"] == {"brats_practical.py": {"recorded": "old", "running": "new"}}


def test_wrong_key_or_corrupt_store_is_refused():
    stored = manifest()
    with pytest.raises(RuntimeError, match="does not hash"):
        adopt_recorded_plan(stored, stored, "0" * 20, identity, ALLOWED)
    with pytest.raises(RuntimeError, match="missing"):
        adopt_recorded_plan(None, stored, "0" * 20, identity, ALLOWED)


@pytest.mark.parametrize("change", [dict(seeds=[1]), dict(split=dict(train=["a"], val=["b"], test=["x"])),
                                    dict(rows=[dict(name="full")]), dict(debug=True), dict(class_weights=[1.0, 3.0])])
def test_any_non_source_difference_is_refused(change):
    stored = manifest()
    key = identity(stored)[:20]
    with pytest.raises(RuntimeError, match="differs"):
        adopt_recorded_plan(stored, manifest(**change), key, identity, ALLOWED)


def test_unlisted_source_change_is_refused():
    stored = manifest()
    key = identity(stored)[:20]
    running = manifest(source_hashes=dict(stored["source_hashes"], **{"brats_graph.py": "changed"}))
    with pytest.raises(RuntimeError, match="brats_graph.py"):
        adopt_recorded_plan(stored, running, key, identity, ALLOWED)


def test_json_roundtrip_of_stored_protocol_keeps_key():
    stored = manifest()
    key = identity(stored)[:20]
    again = json.loads(json.dumps(stored))
    assert adopt_recorded_plan(again, stored, key, identity, ALLOWED)[1]["changed_sources"] == {}


from brats_protocol import select_main_plan_key

ROWS = ("full", "no_voxel", "segresnet")


def trained(*rows):
    return {"runs/%s/seed_42/trained.json" % r for r in rows}


def test_selects_the_store_that_holds_all_trained_rows_and_ignores_orphans():
    candidates = {"newkey_orphan": {"protocol.json"},                      # created by the failed run
                  "pilotkey": {"protocol.json", "evaluation/val/full/seed_42.json"},
                  "d90f": trained(*ROWS) | {"protocol.json", "evaluation/test/full/seed_42.json"}}
    key, report = select_main_plan_key(candidates, ROWS, (42,))
    assert key == "d90f" and report["d90f"]["trained"] == 3 and report["newkey_orphan"]["trained"] == 0


def test_partial_training_is_refused_and_names_the_gap():
    with pytest.raises(RuntimeError, match="Re-run Part 3"):
        select_main_plan_key({"a": trained("full", "no_voxel")}, ROWS, (42,))
    with pytest.raises(RuntimeError, match="Re-run Part 3"):
        select_main_plan_key({}, ROWS, (42,))


def test_two_complete_stores_prefer_more_test_evaluations_and_refuse_a_tie():
    a = trained(*ROWS) | {"evaluation/test/full/seed_42.json"}
    b = trained(*ROWS)
    assert select_main_plan_key({"a": a, "b": b}, ROWS, (42,))[0] == "a"
    with pytest.raises(RuntimeError, match="MAIN_PLAN_KEY_OVERRIDE"):
        select_main_plan_key({"a": b, "b": set(b)}, ROWS, (42,))
