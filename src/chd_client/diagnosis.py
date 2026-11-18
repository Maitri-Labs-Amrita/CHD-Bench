"""The classification report (schemas diagnosis-v1 and diagnosis-normal-v1) for
display: the class names in the scorer's order with short labels, and helpers that work on the report dict. A copy:
tests/test_diagnosis.py checks it against the scorer's schema (chd_client.task.diagnosis_schema).
"""

from chd_client.chd_classes import load as _manifest

CLASS_NAMES = ("atrial_septal_defect", "ventricular_septal_defect", "atrioventricular_septal_defect",
               "tetralogy_of_fallot", "transposition_of_great_arteries", "double_outlet_right_ventricle",
               "common_arterial_trunk", "coarctation", "aortic_arch_hypoplasia", "double_aortic_arch",
               "interrupted_aortic_arch", "pulmonary_atresia", "anomalous_pulmonary_venous_connection", "double_svc",
               "patent_ductus_arteriosus", "pulmonary_artery_sling")
# Codes and full names for display come from the class manifest (config/chd_classes.csv, chd_client.chd_classes),
# for the CT and the MRI classes; never the dataset's own column names (ImageCHD's sheet says PA for pulmonary
# atresia, which is also the pulmonary artery segment).
SHORT = {c.name: c.code for c in _manifest()}
LONG = {c.name: c.full_name for c in _manifest()}
CLASSES = tuple((c, SHORT[c], LONG[c]) for c in CLASS_NAMES)


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
