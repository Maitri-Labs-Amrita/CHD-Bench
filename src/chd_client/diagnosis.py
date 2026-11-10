"""The classification report (schemas diagnosis-v1 and diagnosis-normal-v1) for
display: the class names in the scorer's order with short labels, and helpers that work on the report dict. A copy:
tests/test_diagnosis.py checks it against the scorer's schema (chd_client.task.diagnosis_schema).
"""

CLASSES = (
    ("atrial_septal_defect", "ASD", "Atrial septal defect"),
    ("ventricular_septal_defect", "VSD", "Ventricular septal defect"),
    ("atrioventricular_septal_defect", "AVSD", "Atrioventricular septal defect"),
    ("tetralogy_of_fallot", "ToF", "Tetralogy of Fallot"),
    ("transposition_of_great_arteries", "TGA", "Transposition of the great arteries"),
    ("double_outlet_right_ventricle", "DORV", "Double outlet right ventricle"),
    ("common_arterial_trunk", "CAT", "Common arterial trunk"),
    ("coarctation", "CA", "Coarctation"),
    ("aortic_arch_hypoplasia", "AAH", "Aortic arch hypoplasia"),
    ("double_aortic_arch", "DAA", "Double aortic arch"),
    ("interrupted_aortic_arch", "IAA", "Interrupted aortic arch"),
    ("pulmonary_atresia", "PA", "Pulmonary atresia"),
    ("anomalous_pulmonary_venous_connection", "APVC", "Anomalous pulmonary venous connection"),
    ("double_svc", "DSVC", "Double SVC"),
    ("patent_ductus_arteriosus", "PDA", "Patent ductus arteriosus"),
    ("pulmonary_artery_sling", "PAS", "Pulmonary artery sling"),
)
CLASS_NAMES = tuple(c for c, _, _ in CLASSES)
SHORT = {c: s for c, s, _ in CLASSES}
LONG = {c: name for c, _, name in CLASSES}


def is_diagnosis_report(report) -> bool:
    """A diagnosis-v1 report: it has `classes` and no segmental fields."""
    return isinstance(report, dict) and isinstance(report.get("classes"), dict) and "visceroatrial_situs" not in report


def class_names(report) -> tuple:
    """The report's defect classes in order: ImageCHD's CLASS_NAMES, or (an HVSMR or custom-scan report, whose classes
    are its profile's: chd_client.task.cohorts) the report's own keys in their order, normal left out."""
    keys = [c for c in (report.get("classes") or {}) if c != "normal"]
    return CLASS_NAMES if set(keys) <= set(CLASS_NAMES) else tuple(keys)


def class_rows(report) -> tuple:
    """(name, short, long) per class of the report (CLASSES for an ImageCHD report)."""
    names = class_names(report)
    if names is CLASS_NAMES:
        return CLASSES
    return tuple((c, SHORT.get(c, c), LONG.get(c, c.replace("_", " ").capitalize())) for c in names)


def present(report) -> list[str]:
    """The classes called present, in class order ([] = none of the classes found)."""
    classes = report.get("classes") or {}
    return [c for c in class_names(report) if (classes.get(c) or {}).get("present") is True]


def present_short(report) -> str:
    """'DORV + VSD' style (in class order), 'none of 16' when none is present (not proof of normal)."""
    p = present(report)
    return " + ".join(SHORT.get(c, c) for c in p) if p else f"none of {len(class_names(report))}"


def dry_report(classes=CLASS_NAMES) -> dict:
    """The scripted dry run's classification (plumbing only): every class absent at 0.5."""
    return {"classes": {c: {"present": False, "confidence": 0.5} for c in classes}}


def dry_normal_report(classes=CLASS_NAMES) -> dict:
    """The scripted dry run's report for --report classes_normal: every class absent, normal present."""
    return {"classes": {**dry_report(classes)["classes"], "normal": {"present": True, "confidence": 0.5}}}
