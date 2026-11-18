"""The class manifest (config/chd_classes.csv): codes people type after --truth and read in the output. No code may
be mistaken for a segmentation structure (PA is the pulmonary artery), and the manifest must match the reports."""

import pytest

from chd_client import chd_classes, diagnosis
from chd_client.task import cohorts, imagechd


def spellings_of(c):
    return [c.code, *c.also]


def test_codes_and_spellings_are_unique_without_regard_to_case():
    seen = {}
    for c in chd_classes.load():
        for s in spellings_of(c) + [c.name]:
            assert seen.setdefault(s.lower(), c.name) == c.name, f"{s} is used by {seen[s.lower()]} and {c.name}"


def test_no_code_is_a_segmentation_structure_or_normal():
    taken = {s.lower() for s in chd_classes.SEGMENT_NAMES} | {chd_classes.NORMAL}
    for c in chd_classes.load():
        for s in spellings_of(c):
            assert s.lower() not in taken, f"{s} ({c.full_name}) reads as a segmentation structure or normal"


def test_segment_names_cover_every_label_map():
    names = set(imagechd.LABELS.values())
    for m in (cohorts.IMAGECHD_LABEL_MAP, cohorts.HVSMR_LABEL_MAP):
        names |= {part.split("=")[1] for part in m.split(",")}
    assert names <= set(chd_classes.SEGMENT_NAMES)


@pytest.mark.parametrize("modality,field", [("ct", "ct"), ("mr", "mri")])
def test_the_manifest_matches_the_report_profiles(modality, field):
    report = set(cohorts.profile_classes(cohorts.PROFILES[modality]))
    no3d = set(cohorts.profile_classes(cohorts.NO3D_PROFILES[modality]))
    listed = {c.name for c in chd_classes.load() if getattr(c, field)}
    assert listed == report == no3d


def test_every_class_has_a_display_code_and_name():
    for modality in ("ct", "mr"):
        for c in cohorts.profile_classes(cohorts.PROFILES[modality]):
            assert diagnosis.SHORT[c] and diagnosis.LONG[c] != c


CT = cohorts.custom("ct").classes
MR = cohorts.custom("mr").classes


@pytest.mark.parametrize("given,want", [
    (["VSD", "ToF"], ["ventricular_septal_defect", "tetralogy_of_fallot"]),
    (["vsd,tof"], ["ventricular_septal_defect", "tetralogy_of_fallot"]),
    (["PuA"], ["pulmonary_atresia"]),
    (["CA"], ["coarctation"]),
    (["normal"], []),
])
def test_truth_takes_codes_in_any_case_separated_by_commas_or_spaces(given, want):
    assert cohorts.parse_truth(given, CT, "a CT scan") == want


@pytest.mark.parametrize("given,says", [
    (["PA"], "PA is a segmentation structure (pulmonary artery), not a diagnosis. For pulmonary atresia use PuA"),
    (["ao"], "segmentation structure (aorta)"),
    (["DTGA"], "not in the class list for a CT scan (MRI only)"),
    (["HLHS"], "leave --truth out to get the model's diagnosis without a score"),
    (["normal", "VSD"], "'normal' cannot be combined"),
])
def test_truth_refuses_with_the_fix(given, says):
    with pytest.raises(cohorts.CohortError) as e:
        cohorts.parse_truth(given, CT, "a CT scan")
    assert says in str(e.value)


def test_mri_truth_uses_the_mri_classes():
    assert cohorts.parse_truth(["D-TGA", "heterotaxy"], MR, "an MRI scan") == [
        "d_loop_transposition_of_great_arteries", "heterotaxy"]
    with pytest.raises(cohorts.CohortError, match="CT only"):
        cohorts.parse_truth(["TGA"], MR, "an MRI scan")
