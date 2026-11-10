"""The 3d arm's final answer: a multi-label CHD classification, schema diagnosis-v1.

Sixteen classes, each {present, confidence}; none present = none of the 16 found (not proof of a normal heart).
No free text. The class names are descriptive, so the pulmonary-atresia class can never be confused with the PA
seed, the pulmonary artery; IMAGECHD_FLAG maps each to the ImageCHD diagnosis flag it is scored
against (chd_client.task.scoring.multilabel). Confidence is the probability that the call is correct, so it lies in
[0.5, 1] (below 0.5 the call should flip).

Labelling convention (following ImageCHD): the VSD that is part of tetralogy is not a separate
ventricular_septal_defect; a VSD with DORV, pulmonary atresia, TGA or common trunk is. That is the prompt's job;
the validator only checks shape (every class, nothing else, confidences in range, evidence refs).

Pass context={"view_refs": <bookmarked refs>} to also check that evidence cites existing bookmarks.
"""

import re

from pydantic import (BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, create_model, field_validator,
                      model_validator)

from chd_client.task.imagechd import DIAGNOSES

DIAGNOSIS_SCHEMA_VERSION = "diagnosis-v1"
VIEW_REF_PATTERN = r"^v\d{3,}$"

IMAGECHD_FLAG = {
    "atrial_septal_defect": "ASD",
    "ventricular_septal_defect": "VSD",
    "atrioventricular_septal_defect": "AVSD",
    "tetralogy_of_fallot": "ToF",
    "transposition_of_great_arteries": "TGA",
    "double_outlet_right_ventricle": "DORV",
    "common_arterial_trunk": "CAT",
    "coarctation": "CA",
    "aortic_arch_hypoplasia": "AAH",
    "double_aortic_arch": "DAA",
    "interrupted_aortic_arch": "IAA",
    "pulmonary_atresia": "PA",
    "anomalous_pulmonary_venous_connection": "APVC",
    "double_svc": "DSVC",
    "patent_ductus_arteriosus": "PDA",
    "pulmonary_artery_sling": "PAS",
}
CHD_CLASSES = tuple(IMAGECHD_FLAG)
CLASS_FOR_FLAG = {f: c for c, f in IMAGECHD_FLAG.items()}
assert tuple(IMAGECHD_FLAG.values()) == DIAGNOSES


def check_evidence(v: dict, keys, info: ValidationInfo, per="field") -> dict:
    """An evidence map's problems in one ValueError: keys among `keys`, at most 6 well-formed view_refs each, and
    (with context={"view_refs": ...}) only bookmarked ones."""
    problems = []
    bad = set(v) - set(keys)
    if bad:
        problems.append(f"keys must be among {list(keys)}, got {sorted(bad)}")
    if any(len(rs) > 6 for rs in v.values()):
        problems.append(f"at most 6 view_refs per {per}")
    refs = [r for rs in v.values() for r in rs]
    malformed = sorted({r for r in refs if not re.match(VIEW_REF_PATTERN, r)})
    if malformed:
        problems.append(f"bad view_refs {malformed}")
    known = (info.context or {}).get("view_refs")
    if known is not None:
        unknown = sorted({r for r in refs if r not in known and r not in malformed})
        if unknown:
            problems.append(f"cites unknown view_refs {unknown}")
    if problems:
        raise ValueError("; ".join(problems))
    return v


def format_errors(e: ValidationError) -> str:
    """All of a report's problems as one model-facing line: 'field: message; field: message'. No pydantic
    boilerplate (URLs, input echoes, error counts)."""
    parts = []
    for err in e.errors(include_url=False, include_input=False):
        loc = ".".join(str(p) for p in err["loc"]) or "report"
        msg = err["msg"].removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            msg = "unknown field"
        elif err["type"] == "missing":
            msg = "required field missing"
        parts.append(f"{loc}: {msg}")
    return "; ".join(parts)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassCall(_Strict):
    present: bool
    confidence: float = Field(ge=0.5, le=1.0, description="probability that this call is correct")


Classes = create_model("Classes", __base__=_Strict, **{c: (ClassCall, ...) for c in CHD_CLASSES})


class DiagnosisReport(_Strict):
    classes: Classes = Field(description="every one of the 16 classes, present or absent")
    evidence: dict[str, list[str]] = Field(default_factory=dict,
                                           description="class name -> view_refs from bookmark_view")

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v, info: ValidationInfo):
        return check_evidence(v, CHD_CLASSES, info, per="class")

    def present(self) -> list[str]:
        """The classes called present, in CHD_CLASSES order."""
        return [c for c in CHD_CLASSES if getattr(self.classes, c).present]

    def view_refs(self) -> set[str]:
        return {r for refs in self.evidence.values() for r in refs}


DIAGNOSIS_NORMAL_SCHEMA_VERSION = "diagnosis-normal-v1"
NORMAL = "normal"
ClassesNormal = create_model("ClassesNormal", __base__=_Strict,
                             **{c: (ClassCall, ...) for c in (*CHD_CLASSES, NORMAL)})


class DiagnosisNormalReport(_Strict):
    classes: ClassesNormal = Field(description="every one of the 16 classes and normal, present or absent")
    evidence: dict[str, list[str]] = Field(default_factory=dict,
                                           description="class name (or normal) -> view_refs from bookmark_view")

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v, info: ValidationInfo):
        return check_evidence(v, (*CHD_CLASSES, NORMAL), info, per="class")

    @model_validator(mode="after")
    def _normal_is_no_defect(self):
        defects = self.present()
        if self.classes.normal.present and defects:
            raise ValueError(f"normal is present but so are {defects}: normal means no defect. Mark normal absent, "
                             "or the defects absent")
        if not self.classes.normal.present and not defects:
            raise ValueError("no class is present and normal is absent: mark normal present for a heart with no "
                             "defect, or mark the defects you found")
        return self

    def present(self) -> list[str]:
        """The defect classes called present (normal is not one), in CHD_CLASSES order."""
        return [c for c in CHD_CLASSES if getattr(self.classes, c).present]

    def normal_call(self) -> dict:
        return self.classes.normal.model_dump(mode="json")

    def scored(self) -> DiagnosisReport:
        """The 16 classes as a diagnosis-v1 report (report.json, what the scorer reads)."""
        classes = {c: getattr(self.classes, c).model_dump(mode="json") for c in CHD_CLASSES}
        evidence = {k: v for k, v in self.evidence.items() if k != NORMAL}
        return DiagnosisReport.model_validate({"classes": classes, "evidence": evidence})

    def view_refs(self) -> set[str]:
        return {r for refs in self.evidence.values() for r in refs}


def is_diagnosis_report(report) -> bool:
    """True for a diagnosis-v1 report (a dict or model): it has `classes` and no segmental fields."""
    if isinstance(report, DiagnosisReport):
        return True
    return isinstance(report, dict) and "classes" in report and "visceroatrial_situs" not in report
