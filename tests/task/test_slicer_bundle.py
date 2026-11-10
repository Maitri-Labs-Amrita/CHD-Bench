"""chd_client.task.slicer_bundle on synthetic run folders (no Slicer): files, CT geometry, ground-truth remap, the scene."""

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from chd_client.task import imagechd, slicer_bundle as sb  # noqa: E402
from tests.task.imagechd_fake import make_dataset, raw_ct  # noqa: E402

N = 1008


@pytest.fixture
def root(tmp_path):
    return make_dataset(tmp_path / "ImageCHD", cases=(N,), spacing_rows={N: (0.3, 0.32, 0.75)})


def scene_files(d):
    tree = ET.parse(d / sb.SCENE)
    return tree, [e.get("fileName") for e in tree.getroot() if e.get("fileName")]


def test_imagechd_bundle(tmp_path, root):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"case_source": {"imagechd_id": N}}))
    out = sb.build(run, imagechd_id=N, dataset_dir=root)
    d = run / "slicer"
    assert out["dir"] == str(d) and out["scene"] == "open_me.mrml"
    assert set(out["files"]) == {"ct.nii.gz", "ground_truth.nii.gz", "chd_labels.ctbl", "open_me.mrml", "README.txt",
                                 "truth/case.json", "truth/labels.npz"}
    assert sorted(p.name for p in d.iterdir()) == sorted({f.split("/")[0] for f in out["files"]})
    assert not list(run.glob(".slicer_tmp_*"))
    again = sb.build(run, imagechd_id=N, dataset_dir=root)
    assert set(again["files"]) == set(out["files"])

    ct = sitk.ReadImage(str(d / "ct.nii.gz"))
    assert ct.GetPixelID() == sitk.sitkInt16
    np.testing.assert_array_equal(sitk.GetArrayFromImage(ct), raw_ct(N).astype(np.int32) - 1024)
    np.testing.assert_allclose(ct.GetSpacing(), (0.3, 0.32, 0.75), atol=1e-6)
    np.testing.assert_allclose(ct.GetOrigin(), (0, 0, 0), atol=1e-6)
    np.testing.assert_allclose(ct.GetDirection(), np.eye(3).flatten(), atol=1e-6)
    m = np.array(imagechd.ijk_to_ras((0.3, 0.32, 0.75)))
    ijk = np.array([2, 3, 1, 1.0])
    lps = np.array(ct.TransformContinuousIndexToPhysicalPoint(ijk[:3].tolist()))
    np.testing.assert_allclose(lps * [-1, -1, 1], (m @ ijk)[:3], atol=1e-6)

    gt = sitk.GetArrayFromImage(sitk.ReadImage(str(d / "ground_truth.nii.gz")))
    lab = imagechd.load_labels(root, N)
    assert sb.same_geometry(sitk.ReadImage(str(d / "ground_truth.nii.gz")), ct)
    assert np.all(gt[lab == 1] == 1) and np.all(gt[lab == 4] == 4) and np.all(gt[lab == 5] == sb.MYO)
    assert np.all(gt[lab == 0] == 0)

    tree, files = scene_files(d)
    assert all((d / f).exists() for f in files) and "ct.nii.gz" in files
    assert sorted(files) == ["chd_labels.ctbl", "ct.nii.gz", "ground_truth.nii.gz"]
    r = tree.getroot()
    lm = {e.get("name"): e for e in r.iter("LabelMapVolume")}
    assert set(lm) == {"ground_truth"}
    disp = {e.get("id"): e.get("visibility") for e in r.iter("LabelMapVolumeDisplay")}
    assert disp[lm["ground_truth"].get("references").split("display:")[1].split(";")[0]] == "false"
    for sc in r.iter("SliceComposite"):
        assert "labelVolume" not in sc.get("references")
    assert not list(r.iter("Model"))
    ctbl = (d / "chd_labels.ctbl").read_text().splitlines()
    lv = " ".join(str(int(round(x * 255))) for x in sb.COLORS["LV"])
    assert f"1 LV {lv} 255" in ctbl and any(line.startswith("10 Myo ") for line in ctbl)


def test_remap():
    lab = np.array([0, 1, 2, 3, 4, 5, 6, 7, 15], np.uint8)
    assert sb.remap_gt(lab).tolist() == [0, 1, 2, 3, 4, 10, 5, 6, 0]


def test_mismatch_refused(tmp_path, root):
    run = tmp_path / "run"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"case_source": {"imagechd_id": 1003}}))
    with pytest.raises(sb.BundleError, match="1003"):
        sb.build(run, imagechd_id=N, dataset_dir=root)
    assert not (run / "slicer").exists() and not list(run.glob(".slicer_tmp_*"))


def test_cli_json(tmp_path, root, capsys):
    run = tmp_path / "run"
    run.mkdir()
    sb.main(["--run", str(run), "--imagechd", str(N), "--dataset-dir", str(root), "--json"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["scene"] == "open_me.mrml" and "ground_truth.nii.gz" in out["files"]
    with pytest.raises(SystemExit) as e:
        sb.main(["--run", str(tmp_path / "missing"), "--imagechd", str(N), "--dataset-dir", str(root)])
    assert e.value.code == 1
