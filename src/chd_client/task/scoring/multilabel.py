"""Multi-label scorer: a diagnosis-v1 report vs a case's present classes.

Pure and deterministic, importable without Slicer. Truth is a set of class names (CHD_CLASSES) or ImageCHD flags
(ASD, VSD, ...); an empty set = none of the 16. Classes whose truth is unknown for a dataset
can be left out of every number with `exclude`.

Scorer version multilabel-2.
"""

from chd_client.task.diagnosis_schema import CHD_CLASSES, CLASS_FOR_FLAG, DiagnosisReport

SCORER_VERSION = "multilabel-2"


def truth_classes(truth) -> set[str]:
    """Class names from class names or ImageCHD flags; an unknown name is an error."""
    out = set()
    for t in truth:
        c = CLASS_FOR_FLAG.get(t, t)
        if c not in CHD_CLASSES:
            raise ValueError(f"unknown class or flag {t!r}")
        out.add(c)
    return out


def score(report: DiagnosisReport | dict, truth, exclude=()) -> dict:
    """Per-class outcome (tp/fp/fn/tn) and, over the scored classes (all 16 less `exclude`): exact match of the present sets, Hamming accuracy,
    Jaccard, precision / recall / F1 of the present sets (an empty set on both sides counts as 1.0; an empty
    prediction against a non-empty truth, or the reverse, as 0.0), and the mean Brier score of P(present)
    (confidence when called present, 1 - confidence when called absent)."""
    r = report if isinstance(report, DiagnosisReport) else DiagnosisReport.model_validate(report)
    skip = truth_classes(exclude)
    gold = truth_classes(truth) - skip
    return {"scorer_version": SCORER_VERSION,
            **_numbers(lambda c: getattr(r.classes, c), set(r.present()) - skip, gold, skip)}


def score_classes(report: dict, truth, classes) -> dict:
    """score() for a report with another class list (HVSMR, custom scans: chd_client.task.cohorts): report.json's
    `classes` must have exactly `classes` (each {present, confidence in [0.5, 1]}), truth a subset of them. Same
    numbers; nothing excluded."""
    calls = (report or {}).get("classes") or {}
    if set(calls) != set(classes):
        raise ValueError(f"report classes {sorted(set(calls) ^ set(classes))} differ from the dataset's")
    for c, call in calls.items():
        if not isinstance(call.get("present"), bool) or not 0.5 <= float(call.get("confidence", -1)) <= 1.0:
            raise ValueError(f"class {c}: bad call {call}")
    gold = set(truth)
    if gold - set(classes):
        raise ValueError(f"unknown truth classes {sorted(gold - set(classes))}")
    pred = {c for c in classes if calls[c]["present"]}
    return {"scorer_version": SCORER_VERSION,
            **_numbers(lambda c: _Call(**calls[c]), pred, gold, set(), tuple(classes))}


class _Call:
    def __init__(self, present, confidence):
        self.present, self.confidence = present, float(confidence)


def _numbers(call_of, pred, gold, skip, classes=CHD_CLASSES) -> dict:
    scored = [c for c in classes if c not in skip]
    per_class, brier = {}, 0.0
    for c in scored:
        call = call_of(c)
        t, p = c in gold, c in pred
        per_class[c] = ("tp" if p else "fn") if t else ("fp" if p else "tn")
        p_present = call.confidence if call.present else 1.0 - call.confidence
        brier += (p_present - float(t)) ** 2
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    precision = tp / (tp + fp) if pred else float(not gold)
    recall = tp / (tp + fn) if gold else float(not pred)
    f1 = 1.0 if not pred and not gold else (0.0 if tp == 0 else 2 * precision * recall / (precision + recall))
    union = pred | gold
    tn = sum(v == "tn" for v in per_class.values())
    neg = sum(v in ("tn", "fp") for v in per_class.values())
    return {
        "predicted": [c for c in classes if c in pred],
        "truth": [c for c in classes if c in gold],
        "exact_match": float(pred == gold),
        "hamming_accuracy": round(sum(v in ("tp", "tn") for v in per_class.values()) / len(scored), 6),
        "jaccard": round(len(pred & gold) / len(union), 6) if union else 1.0,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "specificity": round(tn / neg, 6) if neg else 1.0,
        "f1": round(f1, 6),
        "brier": round(brier / len(scored), 6),
        "per_class": per_class,
        "excluded": [c for c in classes if c in skip],
    }
