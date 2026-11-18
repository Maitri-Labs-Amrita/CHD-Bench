"""The CHD class manifest, config/chd_classes.csv: every class CHD-Bench scores, with its short code (what people
type after --truth and read in the terminal and the PDF), its full name, its class name in the model's report, and
the scans whose report has it (ct: the CT report, ImageCHD's 16 classes; mri: the MRI report, HVSMR-2.0's 20).

Codes and the other accepted spellings are unique without regard to case, and none is a segmentation structure name
(SEGMENT_NAMES: LV, RV, LA, RA, Ao, PA, Myo), so "PA" always means the pulmonary artery segment, never a diagnosis.
tests/test_chd_classes.py checks this and that the manifest matches the report profiles.
"""

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "config" / "chd_classes.csv"
NORMAL = "normal"
SEGMENT_NAMES = {"LV": "left ventricle", "RV": "right ventricle", "LA": "left atrium", "RA": "right atrium",
                 "Ao": "aorta", "PA": "pulmonary artery", "Myo": "myocardium"}


class ClassNameError(ValueError):
    pass


@dataclass(frozen=True)
class CHDClass:
    code: str
    full_name: str
    name: str
    ct: bool
    mri: bool
    also: tuple

    @property
    def scans(self) -> str:
        return " and ".join(s for s, on in (("CT", self.ct), ("MRI", self.mri)) if on)


@lru_cache(maxsize=None)
def load(path=None) -> tuple:
    """The manifest rows, in file order."""
    with open(path or MANIFEST, newline="") as f:
        rows = list(csv.DictReader(f))
    return tuple(CHDClass(code=r["code"].strip(), full_name=r["full_name"].strip(), name=r["class"].strip(),
                          ct=r["ct"].strip() == "yes", mri=r["mri"].strip() == "yes",
                          also=tuple((r.get("also_accepted") or "").split())) for r in rows)


def by_name() -> dict:
    return {c.name: c for c in load()}


def code(name: str) -> str:
    c = by_name().get(name)
    return c.code if c else name


def full_name(name: str) -> str:
    c = by_name().get(name)
    return c.full_name if c else name.replace("_", " ").capitalize()


@lru_cache(maxsize=None)
def spellings() -> dict:
    """{lowercase spelling: class name}: the code, the other accepted spellings and the class name itself."""
    out = {}
    for c in load():
        for s in (c.code, c.name, *c.also):
            out[s.lower()] = c.name
    return out


def resolve(text, classes, scan="this scan") -> list:
    """The class names of a --truth value (codes, other accepted spellings or class names; separated by commas or
    spaces; any case), in `classes` order. [] for 'normal'. ClassNameError with the fix for anything else."""
    parts = text if isinstance(text, (list, tuple)) else [text]
    tokens = [t.strip() for p in parts for t in str(p).replace(",", " ").split() if t.strip()]
    if not tokens:
        raise ClassNameError("--truth is empty: give class codes such as VSD, or 'normal'")
    if any(t.lower() == NORMAL for t in tokens):
        if len(tokens) > 1:
            raise ClassNameError("--truth: 'normal' cannot be combined with defect classes")
        return []
    seg = {k.lower(): (k, v) for k, v in SEGMENT_NAMES.items()}
    names, problems = set(), []
    for t in tokens:
        name = spellings().get(t.lower())
        if name is None and t.lower() in seg:
            k, what = seg[t.lower()]
            hint = " For pulmonary atresia use PuA; for a pulmonary artery sling use PAS." if k == "PA" else ""
            problems.append(f"{t} is a segmentation structure ({what}), not a diagnosis.{hint}")
        elif name is None:
            problems.append(f"{t} is not a class CHD-Bench scores (see ./run.sh --list classes). The model only "
                            "reports the listed classes, so a scan whose defect is not listed cannot be scored "
                            "correctly: leave --truth out to get the model's diagnosis without a score.")
        elif name not in classes:
            problems.append(f"{t} ({full_name(name)}) is not in the class list for {scan} "
                            f"({by_name()[name].scans} only). Classes for {scan}: "
                            + ", ".join(code(c) for c in classes))
        else:
            names.add(name)
    if problems:
        raise ClassNameError("--truth: " + " ".join(problems))
    return [c for c in classes if c in names]
