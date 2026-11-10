"""A run folder's own copy of what scoring needs, so a run re-scores offline without the dataset.

slicer_bundle writes it next to the bundle's CT (<run>/slicer/ct.nii.gz, the CT in HU as the case had it):

    <run>/slicer/truth/case.json    {"format", "dataset", "id_key", "id", "shape" (k, j, i), "spacing" (x, y, z; the
                                     exact floats the scorers use), "affine" (IJK -> RAS, 4x4), "ct_sha256",
                                     "labels_sha256", "diagnosis_flags" (ImageCHD, from the diagnosis sheet; null when
                                     the sheet was not found), "diagnosis_sheet"}
    <run>/slicer/truth/labels.npz   labels: the dataset's label map in its ORIGINAL codes (uint8, (k, j, i));
                                     ignore: the ignore zones (bool), HVSMR-2.0 only

The hashes are sha256 of the arrays' bytes ((k, j, i), C order, their dtype). No birth dates or other columns of the
image-info sheet are copied: only the spacing, already in the geometry.

Scorers pick their source with resolve(): "dataset" (the dataset folder: --dir, else runtime.json), "run" (this copy;
an error if it is missing or does not match the trace's case), or "auto" (the dataset when --dir is given, else the
run's copy when present, else the dataset).
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from chd_client.task import datasets

TRUTH_DIR = Path("slicer") / "truth"
CASE = "case.json"
LABELS = "labels.npz"
CT = Path("slicer") / "ct.nii.gz"
FORMAT = "chd-truth-v1"
SOURCES = ("auto", "dataset", "run")


class TruthError(RuntimeError):
    pass


def array_sha256(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _flags(ds, n, root):
    if ds.key != "imagechd":
        return None, None
    from chd_client.task import imagechd
    try:
        sheet = imagechd.diagnosis_sheet(root)
        if not sheet.exists():
            return None, None
        return imagechd.diagnoses(root, sheet).get(int(n)), sheet.name
    except Exception:  # noqa: BLE001
        return None, None


def write(out_dir, ds, n, root) -> dict:
    """Write case.json and labels.npz for dataset `ds` case `n` (read from `root`) into out_dir; returns case.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ct, sp = ds.load_ct(root, n)
    lab = ds.load_labels(root, n)
    ign = ds.ignore_mask(root, n) if ds.ignore_mask else None
    arrays = {"labels": np.ascontiguousarray(lab, np.uint8)}
    if ign is not None:
        arrays["ignore"] = np.ascontiguousarray(ign, bool)
    np.savez_compressed(out / LABELS, **arrays)
    flags, sheet = _flags(ds, n, root)
    case = {"format": FORMAT, "dataset": ds.key, "id_key": ds.id_key, "id": int(n), "shape": list(ct.shape),
            "spacing": [float(x) for x in sp], "affine": [[float(x) for x in r] for r in ds.affine(root, n, sp)],
            "ct_sha256": array_sha256(ct), "labels_sha256": array_sha256(arrays["labels"]),
            "diagnosis_flags": flags, "diagnosis_sheet": sheet}
    (out / CASE).write_text(json.dumps(case, indent=2))
    return case


def read_case(run_dir) -> dict | None:
    p = Path(run_dir) / TRUTH_DIR / CASE
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def from_run(ds, n, run_dir):
    """`ds` with loaders that read the run folder's copy instead of the dataset (the root argument is ignored)."""
    d = Path(run_dir)
    case = read_case(d)
    if case is None:
        raise TruthError(f"no scoring copy in {d / TRUTH_DIR}")
    if case.get("format") != FORMAT or case.get("dataset") != ds.key or int(case.get("id", -1)) != int(n):
        raise TruthError(f"{d / TRUTH_DIR / CASE} is {case.get('dataset')} {case.get('id')}, not the trace's "
                         f"{ds.key} {n}")
    cache = {}

    def npz():
        if "npz" not in cache:
            with np.load(d / TRUTH_DIR / LABELS) as z:
                cache["npz"] = {k: z[k] for k in z.files}
        return cache["npz"]

    def load_ct(_root, _n):
        if "ct" not in cache:
            import SimpleITK as sitk
            a = sitk.GetArrayFromImage(sitk.ReadImage(str(d / CT)))
            if list(a.shape) != case["shape"] or array_sha256(a.astype(np.int16)) != case["ct_sha256"]:
                raise TruthError(f"{d / CT} is not the CT case.json describes (shape or sha256)")
            cache["ct"] = a.astype(np.int16)
        return cache["ct"], tuple(case["spacing"])

    def load_labels(_root, _n):
        lab = npz()["labels"]
        if array_sha256(lab) != case["labels_sha256"]:
            raise TruthError(f"{d / TRUTH_DIR / LABELS}: labels sha256 differs from case.json")
        return lab

    def ignore(_root, _n):
        return npz().get("ignore")

    return dataclasses.replace(ds, load_ct=load_ct, load_labels=load_labels,
                               affine=lambda _root, _n, _sp: case["affine"],
                               ignore_mask=ignore if ds.ignore_mask else None)


def resolve(ds, n, run_dir, root=None, source="auto"):
    """(dataset view, root, source used) for scoring: see the module docstring."""
    if source not in SOURCES:
        raise TruthError(f"unknown source {source!r}; expected one of {SOURCES}")
    if source == "run" or (source == "auto" and root is None and read_case(run_dir) is not None):
        return from_run(ds, n, run_dir), None, "run"
    return ds, (Path(root).expanduser() if root else ds.dataset_dir()), "dataset"


def flags_from_run(run_dir, n):
    """(flags, sheet name) from the run's copy, or (None, None)."""
    case = read_case(run_dir)
    if not case or case.get("dataset") != "imagechd" or int(case.get("id", -1)) != int(n):
        return None, None
    return case.get("diagnosis_flags"), case.get("diagnosis_sheet")


ERRORS = (TruthError, *datasets.ERRORS)
