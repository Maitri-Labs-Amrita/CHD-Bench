"""Reports for tests. NORMAL is a report-v1 segmental report (a normal heart), a format that eval_diagnosis must skip.
NORMAL_DX is the diagnosis-v1 classification (the 16 classes); NORMAL_NC adds normal, the report the 3d_seg runs submit.
"""

SCOREABLE = ("visceroatrial_situs", "ventricular_loop", "av_connection", "av_valve_morphology", "va_connection",
             "great_artery_relationship", "primary_diagnosis")

NORMAL = dict(
    study=dict(acquisition="non_gated_cta", opacification="balanced", upper_abdomen_in_view="partial",
               image_quality="adequate"),
    visceroatrial_situs="solitus", ventricular_loop="D_loop", av_connection="concordant",
    av_valve_morphology="two_valves", va_connection="concordant", great_artery_relationship="normally_related_solitus",
    primary_diagnosis="normal", associated_lesions=[], postoperative=dict(prior_surgery_evident="no"),
    confidence={f: 0.8 for f in SCOREABLE}, impression="Structurally normal heart.")

DX_CLASSES = ("atrial_septal_defect", "ventricular_septal_defect", "atrioventricular_septal_defect",
              "tetralogy_of_fallot", "transposition_of_great_arteries", "double_outlet_right_ventricle",
              "common_arterial_trunk", "coarctation", "aortic_arch_hypoplasia", "double_aortic_arch",
              "interrupted_aortic_arch", "pulmonary_atresia", "anomalous_pulmonary_venous_connection", "double_svc",
              "patent_ductus_arteriosus", "pulmonary_artery_sling")
NORMAL_DX = {"classes": {c: {"present": False, "confidence": 0.8} for c in DX_CLASSES}}


def dx(*present, confidence=0.8) -> dict:
    """A diagnosis-v1 report with these classes present."""
    return {"classes": {c: {"present": c in present, "confidence": confidence} for c in DX_CLASSES}}


NORMAL_NC = {"classes": {**NORMAL_DX["classes"], "normal": {"present": True, "confidence": 0.8}}}


def report_for(arm="3d_seg") -> dict:
    """A valid normal report for submit_report (classes_normal)."""
    return NORMAL_NC
