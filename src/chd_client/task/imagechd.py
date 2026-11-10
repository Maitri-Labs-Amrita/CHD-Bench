"""ImageCHD cases (public dataset, Xu et al. MICCAI 2020): loader, spacing, labels.

    uv run python -m chd_client.task.imagechd list [--json] [--dir <dataset dir>]

The dataset folder (runtime.json `imagechd_dir`, or CHD_IMAGECHD_DIR) is only read, never copied:
`ct_<id>_image.nii.gz`, `ct_<id>_label.nii.gz`, the 2021 diagnosis sheet
`imageCHD_dataset_diagnosis_info(updated20210510).xlsx` (flags; col A = case id, "1" = present; runtime.json
`imagechd_diagnosis_sheet` gives its path when it is not in the dataset folder; the older released
`imageCHD_dataset_info.xlsx` is never read) and `imagechd_dataset_image_info.xlsx` (per-scan info; it also holds birth and acquisition dates, so
ONLY the spacing columns are read: B `idx` = id - 1000, J/K PixelSpacing x/y, M `calculate_z_thick` z).

Data quirks:
- Values are HU + 1024 (HU_OFFSET).
- The NIfTI headers are wrong: spacing 1 x 1 x 1 mm and an RAS identity affine (ITK direction diag(-1, -1, 1)).
  The arrays are in DICOM order (i -> patient left, j -> posterior, k -> superior), so with the real spacing the
  IJK -> RAS matrix is diag(-sx, -sy, sz) with origin 0 (`ijk_to_ras`). `check_header` refuses any other header:
  a different header means the data changed (stop and ask).
- Labels 1 LV, 2 RV, 3 LA, 4 RA, 5 myocardium, 6 Ao, 7 PA; others (e.g. 15) are ignored. No PV/SVC/IVC labels.

Stdlib + numpy; SimpleITK is imported only to read the NIfTI files (bundled in Slicer; a dev dependency here).
The diagnosis flags are for scoring only; they never go into a trace or the model's input.
"""

import argparse
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from chd_client.task.config import REPO

DEFAULT_DIR = str(REPO / "data" / "ImageCHD_dataset")
IMAGE_INFO = "imagechd_dataset_image_info.xlsx"
DIAGNOSIS_INFO = "imageCHD_dataset_diagnosis_info(updated20210510).xlsx"
DIAGNOSIS_ALIASES = {"PuA": "PA"}
HU_OFFSET = 1024
LABELS = {1: "LV", 2: "RV", 3: "LA", 4: "RA", 5: "Myo", 6: "Ao", 7: "PA"}
BLOOD_LABELS = (1, 2, 3, 4, 6, 7)
DIAGNOSES = ("ASD", "VSD", "AVSD", "ToF", "TGA", "DORV", "CAT", "CA", "AAH", "DAA", "IAA", "PA", "APVC", "DSVC",
             "PDA", "PAS")
SPACING_COLS = {"idx": "B", "x": "J", "y": "K", "z": "M"}
IMAGE_RE = re.compile(r"^ct_(\d{4})_image\.nii\.gz$")
HEADER_DIRECTION = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class ImageCHDError(RuntimeError):
    pass


def dataset_dir(runtime=None) -> Path:
    """CHD_IMAGECHD_DIR, else runtime.json `imagechd_dir` (a relative path is relative to CHD-Bench), else
    DEFAULT_DIR. `runtime` is a loaded runtime dict, a runtime.json path or None (the default runtime)."""
    if os.environ.get("CHD_IMAGECHD_DIR"):
        return Path(os.environ["CHD_IMAGECHD_DIR"]).expanduser()
    if not isinstance(runtime, dict):
        from chd_client.task.config import load_runtime
        runtime = load_runtime(runtime)
    p = Path(runtime.get("imagechd_dir") or DEFAULT_DIR).expanduser()
    return p if p.is_absolute() else REPO / p


def _rows(path, cols=None):
    """Rows of the first sheet of an .xlsx as {column letter: text}; only the columns in `cols` (all if None) are
    read. Stdlib only (zipfile + xml.etree); `t="s"` cells index the shared strings."""
    try:
        z = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as e:
        raise ImageCHDError(f"cannot read {path}: {e}") from e
    with z:
        names = set(z.namelist())
        shared = []
        if "xl/sharedStrings.xml" in names:
            shared = ["".join(t.text or "" for t in si.iter(_NS + "t"))
                      for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(_NS + "si")]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    for row in sheet.iter(_NS + "row"):
        out = {}
        for c in row.findall(_NS + "c"):
            col = "".join(ch for ch in c.get("r", "") if ch.isalpha())
            if cols is not None and col not in cols:
                continue
            v = c.find(_NS + "v")
            x = "" if v is None or v.text is None else v.text
            if c.get("t") == "s" and x:
                x = shared[int(x)]
            elif c.get("t") == "inlineStr":
                x = "".join(t.text or "" for t in c.iter(_NS + "t"))
            out[col] = x
        yield out


def spacing_table(root) -> dict:
    """{case id: (x, y, z) mm} from IMAGE_INFO (only the spacing columns are read; header rows are skipped)."""
    c = SPACING_COLS
    t = {}
    for r in _rows(Path(root) / IMAGE_INFO, cols=set(c.values())):
        try:
            sp = (float(r[c["x"]]), float(r[c["y"]]), float(r[c["z"]]))
            t[1000 + int(r[c["idx"]])] = sp
        except (KeyError, ValueError):
            continue
    return t


def diagnosis_sheet(root, runtime=None) -> Path:
    """runtime.json `imagechd_diagnosis_sheet` if set, else DIAGNOSIS_INFO (the 2021 sheet) in the dataset
    folder. Never the older released sheet."""
    if not isinstance(runtime, dict):
        from chd_client.task.config import load_runtime
        runtime = load_runtime(runtime)
    p = runtime.get("imagechd_diagnosis_sheet")
    return Path(p).expanduser() if p else Path(root) / DIAGNOSIS_INFO


def diagnoses(root, sheet=None) -> dict:
    """{case id: [flags present]} from `sheet` (default diagnosis_sheet(root); first row = the flag names, column A =
    the case id; rows without a numeric id are skipped). [] = no flag: a normal heart."""
    rows = list(_rows(Path(sheet) if sheet else diagnosis_sheet(root)))
    if not rows:
        return {}
    names = {k: DIAGNOSIS_ALIASES.get(v, v) for k, v in rows[0].items() if k != "A"}
    names = {k: v for k, v in names.items() if v in DIAGNOSES}
    out = {}
    for r in rows[1:]:
        try:
            n = int(float(r["A"]))
        except (KeyError, ValueError):
            continue
        out[n] = [names[k] for k in sorted(names, key=lambda k: (len(k), k)) if r.get(k, "").strip() == "1"]
    return out


def paths(root, n) -> tuple:
    """(image, label) paths of case n (they may not exist)."""
    root = Path(root)
    return root / f"ct_{int(n)}_image.nii.gz", root / f"ct_{int(n)}_label.nii.gz"


def case_num_from_path(p) -> int:
    """The case id of a `ct_<id>_image.nii.gz` path; ImageCHDError for any other name."""
    m = IMAGE_RE.match(Path(p).name)
    if not m:
        raise ImageCHDError(f"not an ImageCHD image (ct_<id>_image.nii.gz): {Path(p).name}")
    return int(m.group(1))


def list_cases(root, sheet=None) -> list:
    """Cases with an image, a label map and a spacing row, sorted by id: [{id, flags, spacing_mm}]. sheet: the
    diagnosis sheet (default diagnosis_sheet(root))."""
    root = Path(root)
    if not root.is_dir():
        raise ImageCHDError(f"ImageCHD folder not found: {root}")
    sp = spacing_table(root)
    sheet = Path(sheet) if sheet else diagnosis_sheet(root)
    dx = diagnoses(root, sheet) if sheet.exists() else {}
    out = []
    for p in sorted(root.glob("ct_*_image.nii.gz")):
        try:
            n = case_num_from_path(p)
        except ImageCHDError:
            continue
        if n in sp and paths(root, n)[1].exists():
            out.append({"id": n, "flags": dx.get(n, []), "spacing_mm": list(sp[n])})
    return out


def ijk_to_ras(spacing_xyz) -> list:
    """The 4 x 4 IJK -> RAS matrix of a prepared ImageCHD case (array order unchanged, origin 0)."""
    sx, sy, sz = (float(s) for s in spacing_xyz)
    return [[-sx, 0.0, 0.0, 0.0], [0.0, -sy, 0.0, 0.0], [0.0, 0.0, sz, 0.0], [0.0, 0.0, 0.0, 1.0]]


def check_header(img, raw_min=None, what="image"):
    """Refuse a NIfTI whose header is not the known (wrong) one: spacing 1 mm, direction diag(-1, -1, 1); and, when
    given, a raw minimum below 0 (the values are HU + 1024). Anything else means the data changed."""
    sp = tuple(img.GetSpacing())
    d = tuple(float(x) + 0.0 for x in img.GetDirection())
    problems = []
    if any(abs(s - 1.0) > 1e-6 for s in sp):
        problems.append(f"spacing {sp} (expected 1 x 1 x 1)")
    if any(abs(a - b) > 1e-6 for a, b in zip(d, HEADER_DIRECTION)) or len(d) != 9:
        problems.append(f"direction {d} (expected diag(-1, -1, 1))")
    if raw_min is not None and raw_min < 0:
        problems.append(f"raw minimum {raw_min} < 0 (expected HU + 1024)")
    if problems:
        raise ImageCHDError(f"unexpected ImageCHD {what} header: " + "; ".join(problems)
                            + ". The dataset may have changed: stop and check the loader (chd_client.task.imagechd)")


def _read(path):
    import SimpleITK as sitk
    if not Path(path).exists():
        raise ImageCHDError(f"missing: {path}")
    return sitk, sitk.ReadImage(str(path))


def spacing_of(root, n) -> tuple:
    sp = spacing_table(root).get(int(n))
    if sp is None:
        raise ImageCHDError(f"case {n}: no spacing row in {IMAGE_INFO}")
    return sp


def load_ct(root, n) -> tuple:
    """(HU array (k, j, i) int16, spacing (x, y, z) mm). A copy (GetArrayFromImage), never a view."""
    spacing = spacing_of(root, n)
    sitk, img = _read(paths(root, n)[0])
    raw = sitk.GetArrayFromImage(img)
    check_header(img, int(raw.min()), "image")
    hu = raw.astype(np.int32) - HU_OFFSET
    if hu.min() < -32768 or hu.max() > 32767:
        raise ImageCHDError(f"case {n}: HU range {hu.min()}..{hu.max()} does not fit int16")
    return hu.astype(np.int16), spacing


def check_raw(root, n, labels=False):
    """Refuse case n unless its files are the raw ImageCHD ones the preparation assumes (check_header: the known header,
    values HU + 1024): run before every case preparation, which applies the -1024 offset and the LPS identity without
    looking. labels: also the case's own label map."""
    sitk, img = _read(paths(root, n)[0])
    check_header(img, int(sitk.GetArrayViewFromImage(img).min()), f"case {n} image")
    if labels:
        lab = paths(root, n)[1]
        if not lab.exists():
            raise ImageCHDError(f"missing: {lab}")
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(lab))
        reader.ReadImageInformation()
        check_header(reader, what=f"case {n} label")


def load_labels(root, n) -> np.ndarray:
    """The raw label map (k, j, i) uint8 (LABELS; other values ignored by the users)."""
    sitk, img = _read(paths(root, n)[1])
    check_header(img, what="label")
    lab = sitk.GetArrayFromImage(img)
    if lab.min() < 0 or lab.max() > 255:
        raise ImageCHDError(f"case {n}: label values {lab.min()}..{lab.max()} outside 0..255")
    return lab.astype(np.uint8)


def blood_mask(labels) -> np.ndarray:
    return np.isin(labels, BLOOD_LABELS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ls = sub.add_parser("list", help="cases with image, labels and spacing (+ diagnosis flags)")
    a_ls.add_argument("--json", action="store_true")
    a_ls.add_argument("--dir", help="the dataset folder (default: runtime.json imagechd_dir)")
    a_ls.add_argument("--runtime")
    a = ap.parse_args(argv)
    try:
        root = Path(a.dir).expanduser() if a.dir else dataset_dir(a.runtime)
        cases = list_cases(root, diagnosis_sheet(root, a.runtime))
    except ImageCHDError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if a.json:
        print(json.dumps({"dir": str(root), "cases": cases}))
    else:
        for c in cases:
            print(f"{c['id']}  {' '.join(c['flags']) or '-':24s} spacing {c['spacing_mm']}")
        print(f"{len(cases)} cases in {root}")


if __name__ == "__main__":
    main()
