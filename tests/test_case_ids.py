from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from brats_protocol import case_id_from_path, select_canonical_nifti


MODALITIES = ("t1", "t1ce", "t2", "flair", "seg")
PATCHED_CASES = ("BraTS2021_00495", "BraTS2021_00621")


def synthetic_paths():
    canonical_paths = []
    unfiltered_paths = []
    loose_paths = []
    for index in range(1, 1252):
        case_id = f"BraTS2021_{index:05d}"
        for modality in MODALITIES:
            canonical = f"/r/_extracted/{case_id}/{case_id}_{modality}.nii.gz"
            canonical_paths.append(canonical)
            unfiltered_paths.append(canonical)
            if case_id in PATCHED_CASES:
                loose = f"/r/_extracted/{case_id}_{modality}.nii.gz"
                loose_paths.append(loose)
                unfiltered_paths.append(loose)
    return canonical_paths, unfiltered_paths, loose_paths


def test_canonical_selection_keeps_all_1251_five_modality_cases():
    canonical_paths, unfiltered_paths, loose_paths = synthetic_paths()

    selected = select_canonical_nifti(unfiltered_paths)

    assert len(selected) == 6255
    assert set(selected) == set(canonical_paths)
    assert not set(selected) & set(loose_paths)
    case_counts = Counter(case_id_from_path(path) for path in selected)
    assert len(case_counts) == 1251
    assert set(case_counts.values()) == {5}


def test_filename_ids_preserve_loose_case_identity_and_expose_folder_bug():
    _, unfiltered_paths, loose_paths = synthetic_paths()

    assert {case_id_from_path(path) for path in loose_paths} == set(PATCHED_CASES)
    assert {Path(path).parent.name for path in loose_paths} == {"_extracted"}

    old_filter_paths = [
        path for path in unfiltered_paths
        if not (
            Path(path).parent.name == case_id_from_path(path)
            and case_id_from_path(path) in PATCHED_CASES
        )
    ]
    old_folder_ids = {Path(path).parent.name for path in old_filter_paths}
    assert "_extracted" in old_folder_ids
    assert len(old_folder_ids) == 1250


def test_case_id_rejects_unrecognized_filename():
    with pytest.raises(ValueError, match="Unrecognized case ID: foo.nii.gz"):
        case_id_from_path("/r/_extracted/foo.nii.gz")


def test_case_id_normalizes_case_insensitive_filename():
    assert case_id_from_path("brats2021_00001_t1.nii.gz") == "BraTS2021_00001"
