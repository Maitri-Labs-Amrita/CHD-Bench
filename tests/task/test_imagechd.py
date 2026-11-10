"""chd_client.task.imagechd on a tiny synthetic dataset: spacing/flag sheets, case list, header checks, HU, geometry."""

import json

import numpy as np
import pytest

from chd_client.task import imagechd
from tests.task.imagechd_fake import SHAPE, make_dataset, raw_ct, write_nifti, write_xlsx

pytest.importorskip("SimpleITK")


@pytest.fixture
def root(tmp_path):
    return make_dataset(tmp_path / "ImageCHD")


def test_spacing_table_reads_only_the_spacing_columns(root, monkeypatch):
    seen = []
    orig = imagechd._rows

    def spy(path, cols=None):
        for r in orig(path, cols):
            seen.append(r)
            yield r
    monkeypatch.setattr(imagechd, "_rows", spy)
    assert imagechd.spacing_table(root) == {1003: (0.3, 0.32, 0.75), 1008: (0.3, 0.32, 0.75)}
    assert seen and all(set(r) <= {"B", "J", "K", "M"} for r in seen)
    assert not any("1999" in str(v) for r in seen for v in r.values())


def test_diagnoses_and_list_cases(root):
    assert imagechd.diagnoses(root) == {1003: [], 1008: ["DORV"]}
    cases = imagechd.list_cases(root)
    assert [c["id"] for c in cases] == [1003, 1008]
    assert cases[1] == {"id": 1008, "flags": ["DORV"], "spacing_mm": [0.3, 0.32, 0.75]}


def test_the_2021_sheet_layout(root, tmp_path):
    sheet = tmp_path / "imageCHD_dataset_diagnosis_info(updated20210510).xlsx"
    write_xlsx(sheet, [{"A": "index", "B": "AVSD", "C": "PuA", "D": "Normal"}, {"A": "", "B": "18"},
                       {"A": 1003, "D": 1}, {"A": 1008, "B": 1, "C": 1}, {"B": "18", "C": "16", "D": "6"}])
    assert imagechd.diagnoses(root, sheet) == {1003: [], 1008: ["AVSD", "PA"]}
    assert imagechd.list_cases(root, sheet)[1]["flags"] == ["AVSD", "PA"]
    assert imagechd.diagnosis_sheet(root, {"imagechd_diagnosis_sheet": str(sheet)}) == sheet
    assert imagechd.diagnosis_sheet(root, {}) == root / imagechd.DIAGNOSIS_INFO


def test_list_cases_needs_image_label_and_spacing(tmp_path):
    root = make_dataset(tmp_path / "d", cases=(1001, 1002, 1003), labels_for=(1001, 1003),
                        spacing_rows={1001: (0.3, 0.3, 0.75), 1002: (0.3, 0.3, 0.75)})
    (root / "ct_1004_label.nii.gz").write_bytes(b"")
    assert [c["id"] for c in imagechd.list_cases(root)] == [1001]
    with pytest.raises(imagechd.ImageCHDError, match="not found"):
        imagechd.list_cases(tmp_path / "missing")


def test_load_ct_subtracts_the_offset_keeps_the_order_and_uses_the_sheet_spacing(root):
    hu, sp = imagechd.load_ct(root, 1008)
    assert hu.dtype == np.int16 and hu.shape == SHAPE and sp == (0.3, 0.32, 0.75)
    assert np.array_equal(hu, raw_ct(1008).astype(np.int32) - 1024)
    assert hu[2, 3, 4] == 100 * 2 + 10 * 3 + 4 + 1008 % 7
    lab = imagechd.load_labels(root, 1008)
    assert lab.dtype == np.uint8 and lab.shape == SHAPE
    assert imagechd.blood_mask(lab).sum() == 2 * 3 * 4 and not imagechd.blood_mask(lab)[0, 0, 0]


@pytest.mark.parametrize("kw, msg", [
    ({"spacing": (0.5, 1.0, 1.0)}, "spacing"),
    ({"direction": (1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0)}, "direction"),
])
def test_unexpected_headers_are_refused(root, kw, msg):
    write_nifti(root / "ct_1003_image.nii.gz", raw_ct(1003), **kw)
    with pytest.raises(imagechd.ImageCHDError, match=msg):
        imagechd.load_ct(root, 1003)
    write_nifti(root / "ct_1003_label.nii.gz", np.zeros(SHAPE, np.int8), **kw)
    with pytest.raises(imagechd.ImageCHDError, match=msg):
        imagechd.load_labels(root, 1003)


def test_negative_raw_values_are_refused(root):
    write_nifti(root / "ct_1003_image.nii.gz", (raw_ct(1003) - 2000).astype(np.int16))
    with pytest.raises(imagechd.ImageCHDError, match="raw minimum"):
        imagechd.load_ct(root, 1003)


def test_missing_spacing_row_or_file(tmp_path):
    root = make_dataset(tmp_path / "d", cases=(1003,), spacing_rows={})
    with pytest.raises(imagechd.ImageCHDError, match="no spacing row"):
        imagechd.load_ct(root, 1003)
    root2 = make_dataset(tmp_path / "e", cases=(1003,))
    with pytest.raises(imagechd.ImageCHDError, match="missing"):
        imagechd.load_labels(root2, 1009)


def test_ijk_to_ras_flips_i_and_j():
    m = np.array(imagechd.ijk_to_ras((0.3, 0.32, 0.75)))
    assert np.allclose(m, np.diag([-0.3, -0.32, 0.75, 1.0]))
    assert (m @ [1, 0, 0, 1])[0] < 0 and (m @ [0, 1, 0, 1])[1] < 0 and (m @ [0, 0, 1, 1])[2] > 0


def test_case_num_from_path():
    assert imagechd.case_num_from_path("/x/ct_1085_image.nii.gz") == 1085
    for bad in ("ct_1085_label.nii.gz", "ct_1085_image.nii", "image.nii.gz", "ct_10a5_image.nii.gz"):
        with pytest.raises(imagechd.ImageCHDError):
            imagechd.case_num_from_path(bad)


def test_dataset_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("CHD_IMAGECHD_DIR", raising=False)
    assert imagechd.dataset_dir({"imagechd_dir": str(tmp_path)}) == tmp_path
    assert imagechd.dataset_dir({}) == imagechd.Path(imagechd.DEFAULT_DIR).expanduser()
    monkeypatch.setenv("CHD_IMAGECHD_DIR", str(tmp_path / "x"))
    assert imagechd.dataset_dir({"imagechd_dir": str(tmp_path)}) == tmp_path / "x"


def test_inline_and_numeric_cells(tmp_path):
    p = tmp_path / "s.xlsx"
    write_xlsx(p, [{"A": "a", "B": 2.5}])
    assert list(imagechd._rows(p)) == [{"A": "a", "B": "2.5"}]


def test_cli_list_json(root, capsys):
    imagechd.main(["list", "--json", "--dir", str(root)])
    out = json.loads(capsys.readouterr().out)
    assert out["dir"] == str(root) and [c["id"] for c in out["cases"]] == [1003, 1008]
    imagechd.main(["list", "--dir", str(root)])
    assert "2 cases" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        imagechd.main(["list", "--dir", str(root / "nope")])
