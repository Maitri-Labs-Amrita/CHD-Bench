"""A run folder's `slicer/` bundle: the input CT and (ImageCHD) the ground truth, opened in 3D Slicer in one step.
No Slicer needed to build it.

    uv run python -m chd_client.task.slicer_bundle --run <run folder> --imagechd <id> [--dataset-dir <dir>] [--json]

Writes `<run>/slicer/` (replaced as a whole if it exists):

    ct.nii.gz              the input CT in HU (int16), the same physical geometry as the case the run used
    ground_truth.nii.gz    the dataset's labels on the CT grid: 1 LV 2 RV 3 LA 4 RA 5 Ao 6 PA, myocardium 10
    chd_labels.ctbl        Slicer colour table for codes 0-10 (COLORS below; Myo tan)
    open_me.mrml           the scene (relative paths): CT on the slices, ground truth loaded but hidden
    README.txt
    truth/case.json        what scoring needs, so the run re-scores offline (chd_client.task.run_truth): the labels in
    truth/labels.npz       the dataset's own codes, the exact spacing and IJK -> RAS, array hashes, and the case's
                           diagnosis flags

Geometry: the CT is rebuilt with chd_client.task.imagechd.load_ct and the IJK -> RAS of tools/prepare_nifti.py
(diag(-sx, -sy, sz), origin 0), i.e. exactly the case NRRD. The run's summary.json case_source.imagechd_id must be
the given id when present; otherwise it is an error. The ground truth is written after the run and never reaches the model.
"""

import argparse
import json
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from chd_client.task import datasets, imagechd, run_truth

NAMES = ("LV", "RV", "LA", "RA", "Ao", "PA", "SVC", "IVC", "PV")
CODE = {n: i + 1 for i, n in enumerate(NAMES)}
MYO = 10
MYO_COLOR = (0.82, 0.71, 0.55)
COLORS = {"LV": (0.80, 0.12, 0.15), "RV": (0.15, 0.30, 0.90), "LA": (0.95, 0.55, 0.70), "RA": (0.45, 0.75, 1.00),
          "Ao": (0.95, 0.45, 0.08), "PA": (0.15, 0.72, 0.30), "SVC": (0.55, 0.30, 0.85), "IVC": (0.90, 0.25, 0.75),
          "PV": (0.55, 0.55, 0.10)}
GT_REMAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: MYO, 6: 5, 7: 6}
CT_WINDOW, CT_LEVEL = 800.0, 200.0
LABEL_OPACITY = 0.5
OUT_DIR = "slicer"
SCENE = "open_me.mrml"
GEOM_TOL = 1e-3


class BundleError(RuntimeError):
    pass


def _sitk():
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise BundleError("SimpleITK is needed (uv run, dev dependencies)") from e
    return sitk


def default_colors() -> dict:
    """{name: rgb}: the colour table's structure colours."""
    return {n: list(COLORS[n]) for n in NAMES}


def imagechd_ct(root, n):
    """The ImageCHD CT as a SimpleITK image with the case NRRD's geometry (LPS: spacing, identity direction, 0)."""
    hu, sp = imagechd.load_ct(root, n)
    return _with_ras(hu, imagechd.ijk_to_ras(sp))


def _with_ras(arr, ijk_to_ras):
    """A (k, j, i) array as a SimpleITK image whose geometry is the given IJK -> RAS (converted to LPS)."""
    sitk = _sitk()
    img = sitk.GetImageFromArray(np.ascontiguousarray(arr, np.int16))
    m = np.array(ijk_to_ras, float)
    lps = np.diag([-1.0, -1.0, 1.0]) @ m[:3, :3]
    spacing = np.linalg.norm(lps, axis=0)
    img.SetSpacing([float(s) for s in spacing])
    img.SetDirection([float(x) for x in (lps / spacing).flatten()])
    img.SetOrigin([float(x) for x in np.diag([-1.0, -1.0, 1.0]) @ m[:3, 3]])
    return img


def geometry(img) -> dict:
    return {"size": list(img.GetSize()), "spacing": list(img.GetSpacing()), "origin": list(img.GetOrigin()),
            "direction": list(img.GetDirection())}


def same_geometry(a, b, tol=GEOM_TOL) -> bool:
    ga, gb = geometry(a), geometry(b)
    return ga["size"] == gb["size"] and all(
        np.allclose(ga[k], gb[k], atol=tol, rtol=0) for k in ("spacing", "origin", "direction"))


def remap_gt(labels, remap=None) -> np.ndarray:
    lut = np.zeros(256, np.uint8)
    for src, dst in (remap or GT_REMAP).items():
        lut[src] = dst
    return lut[np.asarray(labels, np.uint8)]


def _rgb255(c):
    return [int(round(max(0.0, min(1.0, float(x))) * 255)) for x in c]


def write_ctbl(path, colors):
    lines = ["# CHD labels: structure colours (chd_client.task.slicer_bundle)", "# 11 values",
             "0 Background 0 0 0 0"]
    for n in NAMES:
        r, g, b = _rgb255(colors[n])
        lines.append(f"{CODE[n]} {n} {r} {g} {b} 255")
    r, g, b = _rgb255(MYO_COLOR)
    lines.append(f"{MYO} Myo {r} {g} {b} 255")
    Path(path).write_text("\n".join(lines) + "\n")


def _fmt(v):
    return " ".join(f"{float(x):g}" for x in v)


def write_mrml(path, ct, labelmaps, models, window_level=(CT_WINDOW, CT_LEVEL)):
    """labelmaps: [(name, file, shown)], the first shown one is the slices' label layer; models: [(name, file, rgb,
    opacity, visible)]. Storage paths are relative to the scene. window_level None: Slicer's automatic one (MRI)."""
    root = ET.Element("MRML", version="Slicer 5.10.0", userTags="")
    nid = {"n": 0}

    def new(prefix):
        nid["n"] += 1
        return f"{prefix}{nid['n']}"

    ctbl = None
    if labelmaps:
        ctbl = new("vtkMRMLColorTableNodeCHD")
        ctbl_st = new("vtkMRMLColorTableStorageNode")
        ET.SubElement(root, "ColorTable", id=ctbl, name="chd_labels", hideFromEditors="false", type="13",
                      numcolors="8", references=f"storage:{ctbl_st};")
        ET.SubElement(root, "ColorTableStorage", id=ctbl_st, name="ColorTableStorage", hideFromEditors="true",
                      fileName="chd_labels.ctbl")

    ct_id, ct_disp, ct_st = new("vtkMRMLScalarVolumeNode"), new("vtkMRMLScalarVolumeDisplayNode"), \
        new("vtkMRMLVolumeArchetypeStorageNode")
    ET.SubElement(root, "Volume", id=ct_id, name="CT", hideFromEditors="false",
                  references=f"display:{ct_disp};storage:{ct_st};")
    ET.SubElement(root, "VolumeDisplay", id=ct_disp, name="VolumeDisplay", hideFromEditors="true",
                  colorNodeID="vtkMRMLColorTableNodeGrey",
                  **({"window": f"{window_level[0]:g}", "level": f"{window_level[1]:g}", "autoWindowLevel": "0"}
                     if window_level else {"autoWindowLevel": "1"}), interpolate="1", visibility="true")
    ET.SubElement(root, "VolumeArchetypeStorage", id=ct_st, name="VolumeArchetypeStorage", hideFromEditors="true",
                  fileName=ct, fileListMember0=ct, useCompression="1", UseOrientationFromFile="1",
                  centerImage="0")

    label_layer = None
    for name, fname, shown in labelmaps:
        lid, ldisp, lst = new("vtkMRMLLabelMapVolumeNode"), new("vtkMRMLLabelMapVolumeDisplayNode"), \
            new("vtkMRMLVolumeArchetypeStorageNode")
        ET.SubElement(root, "LabelMapVolume", id=lid, name=name, hideFromEditors="false",
                      references=f"display:{ldisp};storage:{lst};")
        ET.SubElement(root, "LabelMapVolumeDisplay", id=ldisp, name="LabelMapVolumeDisplay", hideFromEditors="true",
                      colorNodeID=ctbl, visibility="true" if shown else "false")
        ET.SubElement(root, "VolumeArchetypeStorage", id=lst, name="VolumeArchetypeStorage", hideFromEditors="true",
                      fileName=fname, fileListMember0=fname, useCompression="1", UseOrientationFromFile="1",
                      centerImage="0")
        if shown and label_layer is None:
            label_layer = lid

    for name, fname, rgb, opacity, visible in models:
        mid, mdisp, mst = new("vtkMRMLModelNode"), new("vtkMRMLModelDisplayNode"), new("vtkMRMLModelStorageNode")
        ET.SubElement(root, "Model", id=mid, name=name, hideFromEditors="false",
                      references=f"display:{mdisp};storage:{mst};")
        ET.SubElement(root, "ModelDisplay", id=mdisp, name="ModelDisplay", hideFromEditors="true", color=_fmt(rgb),
                      opacity=f"{opacity:g}", visibility="true" if visible else "false",
                      visibility2D="true" if visible else "false", visibility3D="true",
                      sliceDisplayMode="intersection", scalarVisibility="false", backfaceCulling="false")
        ET.SubElement(root, "ModelStorage", id=mst, name="ModelStorage", hideFromEditors="true", fileName=fname,
                      coordinateSystem="LPS")

    for view in ("Red", "Yellow", "Green"):
        refs = f"backgroundVolume:{ct_id};" + (f"labelVolume:{label_layer};" if label_layer else "")
        ET.SubElement(root, "SliceComposite", id=f"vtkMRMLSliceCompositeNode{view}", name="SliceComposite",
                      hideFromEditors="true", singletonTag=view, layoutName=view, references=refs,
                      labelOpacity=f"{LABEL_OPACITY:g}", foregroundOpacity="0", compositing="0")
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


README = """Open this folder's scene in 3D Slicer: drag open_me.mrml onto the Slicer window (or File > Load Scene).
Everything is aligned in the scanner's coordinates; the paths are relative, so the folder can be moved as a whole.

ct.nii.gz            the input CT in HU (window 800 / level 200 on the slices)
ground_truth.nii.gz  the dataset's labels (loaded but hidden; written after the run): 1 LV, 2 RV, 3 LA, 4 RA,
                     5 Ao, 6 PA, 10 myocardium
chd_labels.ctbl      its colour table
truth/               the scoring copy of the labels and the case's diagnosis flags (chd_client.task.run_truth)
Rebuild: uv run python -m chd_client.task.slicer_bundle --run <run folder> --imagechd <id>
"""


def _check_summary(run, key, n, name):
    src = run / "summary.json"
    if src.exists():
        cs = (json.loads(src.read_text()).get("case_source") or {})
        if cs.get(key) is not None and int(cs[key]) != n:
            raise BundleError(f"summary.json says {name} {cs[key]}, not {n}")


def build(run, imagechd_id, dataset_dir=None) -> dict:
    sitk = _sitk()
    run = Path(run).expanduser().resolve()
    if not run.is_dir():
        raise BundleError(f"run folder not found: {run}")
    n = int(imagechd_id)
    _check_summary(run, "imagechd_id", n, "ImageCHD")
    root = Path(dataset_dir).expanduser() if dataset_dir else imagechd.dataset_dir()
    try:
        ct = imagechd_ct(root, n)
        gt = remap_gt(imagechd.load_labels(root, n)) if imagechd.paths(root, n)[1].exists() else None
    except imagechd.ImageCHDError as e:
        raise BundleError(str(e)) from e
    colors = default_colors()

    out = run / OUT_DIR
    tmp = Path(tempfile.mkdtemp(prefix=".slicer_tmp_", dir=run))
    try:
        files = ["ct.nii.gz"]
        sitk.WriteImage(ct, str(tmp / "ct.nii.gz"), True)
        labelmaps = []
        if gt is not None:
            g = sitk.GetImageFromArray(gt)
            g.CopyInformation(ct)
            sitk.WriteImage(g, str(tmp / "ground_truth.nii.gz"), True)
            files.append("ground_truth.nii.gz")
            labelmaps.append(("ground_truth", "ground_truth.nii.gz", False))
            try:
                run_truth.write(tmp / "truth", datasets.IMAGECHD, n, root)
            except (*run_truth.ERRORS, OSError) as e:
                raise BundleError(f"cannot write the scoring copy: {e}") from e
            files += ["truth/case.json", "truth/labels.npz"]
        write_ctbl(tmp / "chd_labels.ctbl", colors)
        write_mrml(tmp / SCENE, "ct.nii.gz", labelmaps, [])
        (tmp / "README.txt").write_text(README)
        files += ["chd_labels.ctbl", SCENE, "README.txt"]
        if out.exists():
            shutil.rmtree(out)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"dir": str(out), "files": files, "scene": SCENE}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the run folder (reports/<name>/)")
    ap.add_argument("--imagechd", type=int, required=True, help="ImageCHD case id the run used")
    ap.add_argument("--dataset-dir", help="ImageCHD folder (default: runtime.json imagechd_dir)")
    ap.add_argument("--json", action="store_true", help="print one JSON line at the end")
    a = ap.parse_args(argv)
    try:
        out = build(a.run, a.imagechd, a.dataset_dir)
    except (BundleError, OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if a.json:
        print(json.dumps(out))
    else:
        print(f"{out['dir']}: {', '.join(out['files'])}\nOpen {out['scene']} in Slicer.")


if __name__ == "__main__":
    main()
