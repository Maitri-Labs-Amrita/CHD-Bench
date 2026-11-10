"""Datasets other than ImageCHD: HVSMR-2.0 (cardiac MRI) and any user-supplied NIfTI scan.

Each dataset has a task profile (its report plugin holds the class list), a default system prompt, and a source of
diagnosis truth:

    hvsmr   the HVSMR-2.0 release: orig/pat<N>_orig.nii.gz, its ground-truth segmentation
            orig/pat<N>_orig_seg.nii.gz and hvsmr_clinical.csv (the truth). The MRI is rescaled so its 99.5th
            percentile is 1000 (the chd_mr_v1 window and render presets use this scale).
    custom  one NIfTI volume given with --volume. Its truth, when known, is given with --truth; without it the run
            is not scored.

HVSMR classes: the 19 yes/no CHD columns of hvsmr_clinical.csv plus TortuousVessels. Surgical repairs, Marfan,
imaging artefacts and graded dilation are not scored. A case is in the cohort when it has a scored class or is
Normal.
"""

import csv
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from chd_client import paths
from chd_client.task.config import REPO

HVSMR_IMAGES, HVSMR_CLINICAL = "hvsmr_images", "hvsmr_clinical"
NORMAL = "normal"

MR_PERCENTILE = 99.5
MR_TARGET = 1000.0
MR_MAX = 2000

HVSMR_COLUMNS = {
    "ventricular_septal_defect": "VSD",
    "atrial_septal_defect": "ASD",
    "double_outlet_right_ventricle": "DORV",
    "d_loop_transposition_of_great_arteries": "DLoopTGA",
    "l_loop_transposition_of_great_arteries": "LLoopTGA",
    "bilateral_svc": "BilateralSVC",
    "dextrocardia": "Dextrocardia",
    "mesocardia": "Mesocardia",
    "inverted_ventricles": "InvertedVentricles",
    "inverted_atria": "InvertedAtria",
    "left_or_central_ivc": "LeftCentralIVC",
    "left_or_central_svc": "LeftCentralSVC",
    "single_ventricle": "SingleVentricle",
    "double_inlet_left_ventricle": "DILV",
    "double_inlet_double_outlet_right_ventricle": "DIDORV",
    "common_atrium": "CommonAtrium",
    "heterotaxy": "Heterotaxy",
    "superoinferior_ventricles": "SuperoinferiorVentricles",
    "pulmonary_atresia_or_mpa_stump": "PAAtresiaOrMPAStump",
    "tortuous_vessels": "TortuousVessels",
}

HVSMR_LABEL_MAP = "1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA"
IMAGECHD_LABEL_MAP = "1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA"

PROFILES = {"ct": REPO / "profiles" / "chd_ct_v1", "mr": REPO / "profiles" / "chd_mr_v1"}
PROMPTS = {"ct": "chd_ct_v1", "mr": "chd_mr_v1"}


class CohortError(ValueError):
    pass


def profile_classes(profile) -> tuple:
    """The defect classes of a task profile's report plugin (normal not included), in report order."""
    f = Path(profile) / "report.py"
    spec = importlib.util.spec_from_file_location(f"_report_{Path(profile).name}", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(mod.CHD_CLASSES)


def _hvsmr_rows(path=None) -> dict:
    with open(path or paths.get(HVSMR_CLINICAL)) as f:
        return {int(r["Pat"]): r for r in csv.DictReader(f)}


def hvsmr_truth(n: int, path=None) -> list:
    """The present classes of HVSMR case n. CohortError for a case with no row or no scored finding."""
    rows = _hvsmr_rows(path)
    if n not in rows:
        raise CohortError(f"HVSMR case {n} has no row in {Path(path or paths.get(HVSMR_CLINICAL)).name}")
    r = rows[n]
    present = [c for c, col in HVSMR_COLUMNS.items() if r[col].strip() == "X"]
    if not present and r["Normal"].strip() != "X":
        raise CohortError(f"HVSMR case {n} has neither a scored class nor Normal: not in the cohort")
    return present


def hvsmr_cases(path=None) -> list:
    out = []
    for n in sorted(_hvsmr_rows(path)):
        try:
            hvsmr_truth(n, path)
        except CohortError:
            continue
        out.append(n)
    return out


def normalize_mr(src, dest) -> dict:
    """Write src rescaled so its 99.5th percentile is 1000, clipped to [0, 2000], as int16 NIfTI with src's
    geometry. Returns the scaling record."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(src))
    a = sitk.GetArrayFromImage(img).astype(np.float64)
    p = float(np.percentile(a, MR_PERCENTILE))
    if p <= 0:
        raise CohortError(f"{src}: the {MR_PERCENTILE}th percentile is {p}: cannot rescale")
    b = np.clip(a * (MR_TARGET / p), 0, MR_MAX).round().astype(np.int16)
    out = sitk.GetImageFromArray(b)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(dest), True)
    return {"percentile": MR_PERCENTILE, "value": p, "to": MR_TARGET, "clip": [0, MR_MAX]}


@dataclass(frozen=True)
class Cohort:
    key: str
    name: str
    id_key: str
    case_name: str
    modality: str
    label_map: str
    truth: object
    cases: object = None
    has_truth: bool = True

    @property
    def profile(self) -> Path:
        return PROFILES[self.modality]

    @property
    def prompt(self) -> str:
        return PROMPTS[self.modality]

    @property
    def classes(self) -> tuple:
        return profile_classes(self.profile)

    @property
    def kind(self) -> str:
        return f"{self.key}_temporary"

    def stem(self, n) -> str:
        return self.case_name.format(n=n)

    def raw(self, n) -> Path:
        if self.key != "hvsmr":
            raise CohortError(f"{self.name} has no dataset folder")
        return Path(paths.get(HVSMR_IMAGES)) / f"pat{int(n)}_orig.nii.gz"

    def gt_labels(self, n) -> Path:
        if self.key != "hvsmr":
            raise CohortError(f"{self.name} has no ground-truth segmentation: pass a label map file")
        return Path(paths.get(HVSMR_IMAGES)) / f"pat{int(n)}_orig_seg.nii.gz"


HVSMR = Cohort(key="hvsmr", name="HVSMR-2.0", id_key="hvsmr_id", case_name="hvsmr_{n}", modality="mr",
               label_map=HVSMR_LABEL_MAP, truth=hvsmr_truth, cases=hvsmr_cases)
COHORTS = {HVSMR.key: HVSMR}


def custom(modality: str, truth=None) -> Cohort:
    """The cohort of a --volume case. truth: the present classes, [] for normal, or None (not scored)."""
    if modality not in PROFILES:
        raise CohortError(f"modality {modality!r}: one of {', '.join(PROFILES)}")
    fixed = None if truth is None else list(truth)

    def _truth(_n):
        if fixed is None:
            raise CohortError("no truth was given for this scan (--truth)")
        return fixed

    return Cohort(key="custom", name="Custom scan", id_key="name", case_name="{n}", modality=modality,
                  label_map=IMAGECHD_LABEL_MAP, truth=_truth, has_truth=fixed is not None)


def parse_truth(text: str, classes) -> list:
    """--truth: comma-separated class names, or 'normal'. Unknown names are an error."""
    names = [t.strip() for t in text.split(",") if t.strip()]
    if names == [NORMAL]:
        return []
    bad = [t for t in names if t not in classes]
    if bad or not names:
        raise CohortError(f"--truth {text!r}: unknown classes {bad}; use 'normal' or names from: {', '.join(classes)}")
    return [c for c in classes if c in names]


def for_case_source(src: dict | None):
    """(cohort, case id) of a run's case_source, or (None, None)."""
    src = src or {}
    if src.get("dataset") == "custom":
        return custom(src.get("modality", "ct"), src.get("truth")), src.get("name")
    c = COHORTS.get(src.get("dataset"))
    if c is None or src.get(c.id_key) is None:
        return None, None
    return c, int(src[c.id_key])
