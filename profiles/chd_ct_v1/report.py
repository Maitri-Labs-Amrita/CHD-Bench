"""Report plugin of the CT profiles (slicer_mcp/reports.py): a multi-label congenital heart disease classification,
16 defect classes and normal (no defect), schema diagnosis-normal-v1.

Each class is {present, confidence}; no free text.
Confidence is the probability that the call is correct, so it lies in [0.5, 1]. normal is present exactly when none of
the 16 is. evidence (optional) maps class names to view_refs from bookmark_view.

An accepted report writes report.json (the 16 classes: diagnosis-v1, what a scorer reads) and
normal.json (the normal call) into the trace.
"""

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, create_model, field_validator, model_validator

SCHEMA_VERSION = "diagnosis-normal-v1"
DIAGNOSIS_SCHEMA_VERSION = "diagnosis-v1"
VIEW_REF_PATTERN = r"^v\d{3,}$"

CHD_CLASSES = (
    "atrial_septal_defect",
    "ventricular_septal_defect",
    "atrioventricular_septal_defect",
    "tetralogy_of_fallot",
    "transposition_of_great_arteries",
    "double_outlet_right_ventricle",
    "common_arterial_trunk",
    "coarctation",
    "aortic_arch_hypoplasia",
    "double_aortic_arch",
    "interrupted_aortic_arch",
    "pulmonary_atresia",
    "anomalous_pulmonary_venous_connection",
    "double_svc",
    "patent_ductus_arteriosus",
    "pulmonary_artery_sling",
)
NORMAL = "normal"

DESCRIPTION = (
    "Submit the final classification. An accepted report ends the episode: no further tool calls are "
    "accepted. An invalid report is rejected with every reason at once and the episode continues, so you can "
    "correct it and submit again. classes gives every one of the 16 classes (none may be left out, no others "
    "added) with present (true or false) and confidence, the probability in [0.5, 1] that the call is "
    "correct. Besides the 16, classes has one more entry, normal (no defect), with present and "
    "confidence: normal is present exactly when none of the 16 is, so a report with normal and a defect "
    "both present, or with nothing present, is rejected. There are no text fields. "
    "evidence (optional) maps class names to view_refs from bookmark_view.")
ARGUMENT_DESCRIPTION = "the classification: every one of the 16 classes and normal (see the schema)"


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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ClassCall(_Strict):
    present: bool
    confidence: float = Field(ge=0.5, le=1.0, description="probability that this call is correct")


Classes = create_model("Classes", __base__=_Strict, **{c: (ClassCall, ...) for c in CHD_CLASSES})
ClassesNormal = create_model("ClassesNormal", __base__=_Strict,
                             **{c: (ClassCall, ...) for c in (*CHD_CLASSES, NORMAL)})


class DiagnosisReport(_Strict):
    """report.json: the 16 classes (diagnosis-v1)."""
    classes: Classes = Field(description="every one of the 16 classes, present or absent")
    evidence: dict[str, list[str]] = Field(default_factory=dict,
                                           description="class name -> view_refs from bookmark_view")

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v, info: ValidationInfo):
        return check_evidence(v, CHD_CLASSES, info, per="class")


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

    def scored(self) -> DiagnosisReport:
        """The 16 classes as a diagnosis-v1 report (report.json)."""
        classes = {c: getattr(self.classes, c).model_dump(mode="json") for c in CHD_CLASSES}
        evidence = {k: v for k, v in self.evidence.items() if k != NORMAL}
        return DiagnosisReport.model_validate({"classes": classes, "evidence": evidence})


Model = DiagnosisNormalReport


def accept(rep: DiagnosisNormalReport):
    """The model's result, and the trace files: report.json (the 16 classes) and normal.json (the normal call)."""
    result = {"accepted": True, "present": rep.present(), "normal": rep.classes.normal.present}
    files = {"report.json": rep.scored().model_dump(mode="json"),
             "normal.json": {"normal": rep.classes.normal.model_dump(mode="json"),
                             "evidence": rep.evidence.get(NORMAL, []),
                             "report_schema_version": SCHEMA_VERSION}}
    return result, files
