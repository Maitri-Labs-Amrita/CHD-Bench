"""Classification metrics of a batch, per model:

    uv run --group analysis python slurm/batch_metrics.py <batch folder> [--runtime config/runtime.local.json]
        [--out <batch folder>/metrics.json]

Reads every job's plan.json and results.jsonl (run_batch.py) and each run folder's trace/report.json and
trace/normal.json. Truth: the ImageCHD diagnosis sheet (less the classes config/scoring_v1.json excludes), or
the HVSMR-2.0 clinical table.

Counting: each planned episode counts once. When a case was attempted more than once, its last attempt counts. A
planned case with no accepted report (cost cap, step budget, error, timeout, never started) counts as every class
absent, Normal included, at P(present) 0.5, and is listed under `missing`.

Normal has two scores: `normal` derives it from the defect calls (called present when no defect is; score
1 - the largest defect P(present)); `normal_explicit` uses the model's own Normal call and confidence from
normal.json when the run has one.

Per model, computed with scikit-learn on the pooled episodes (one row per episode, one column per class):
- per class: n_pos, tp/fp/fn/tn, sensitivity, specificity, precision, accuracy, balanced accuracy, AUC (undefined when
  the class is all present or all absent in the truth);
- macro (the headline): the mean of each per-class number over the classes with at least one positive case;
- micro: the same numbers over every (episode, class) cell pooled;
- exact match: the share of episodes whose present set equals the truth.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, precision_score, recall_score,
                             roc_auc_score)

from chd_client.task.diagnosis_schema import CHD_CLASSES, CLASS_FOR_FLAG

NORMAL = "normal"
NORMAL_EXPLICIT = "normal_explicit"
METRICS = ("sensitivity", "specificity", "precision", "accuracy", "balanced_accuracy", "auc")


def columns(classes) -> list:
    return list(classes) + [NORMAL, NORMAL_EXPLICIT]


def truth_row(present, classes=CHD_CLASSES) -> np.ndarray:
    present = set(present)
    return np.array([c in present for c in classes] + [not present, not present], dtype=int)


def imagechd_truth(flags) -> list:
    return [CLASS_FOR_FLAG[f] for f in flags]


def report_rows(report, normal=None, classes=CHD_CLASSES) -> tuple:
    """(called present 0/1, P(present)) over columns(classes) from a report and its normal.json call. None = no
    report. Without a normal.json call, normal_explicit falls back to the derived Normal."""
    n = len(classes)
    if report is None:
        return np.zeros(n + 2, int), np.full(n + 2, 0.5)
    cls = report["classes"]
    call = np.array([bool(cls[c]["present"]) for c in classes], dtype=int)
    p = np.array([cls[c]["confidence"] if cls[c]["present"] else 1 - cls[c]["confidence"] for c in classes])
    derived_call, derived_p = int(not call.any()), 1 - p.max()
    if normal is not None:
        ex_call = int(bool(normal["present"]))
        ex_p = normal["confidence"] if normal["present"] else 1 - normal["confidence"]
    else:
        ex_call, ex_p = derived_call, derived_p
    return np.append(call, [derived_call, ex_call]), np.append(p, [derived_p, ex_p])


def _nan(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def binary(y, pred, score) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = 0 < y.sum() < len(y)
    return {"n_pos": int(y.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "sensitivity": _nan(recall_score(y, pred, zero_division=np.nan)) if y.any() else None,
            "specificity": _nan(recall_score(y, pred, pos_label=0, zero_division=np.nan)) if not y.all() else None,
            "precision": _nan(precision_score(y, pred, zero_division=np.nan)),
            "accuracy": _nan(accuracy_score(y, pred)),
            "balanced_accuracy": _nan(balanced_accuracy_score(y, pred)) if both else None,
            "auc": _nan(roc_auc_score(y, score)) if both else None}


def model_metrics(Y, P, S, classes=CHD_CLASSES) -> dict:
    """Metrics over the defect classes and the derived Normal; normal_explicit is reported per class only."""
    cols = columns(classes)
    per_class = {c: binary(Y[:, i], P[:, i], S[:, i]) for i, c in enumerate(cols)}
    k = len(classes) + 1
    scored = [c for c in cols[:k] if per_class[c]["n_pos"] > 0]
    macro = {m: _nan(np.mean([per_class[c][m] for c in scored if per_class[c][m] is not None]))
             if any(per_class[c][m] is not None for c in scored) else None for m in METRICS}
    return {"n_episodes": int(len(Y)), "exact_match": _nan(np.mean((Y[:, :k] == P[:, :k]).all(axis=1))),
            "macro": {**macro, "classes": scored}, "micro": binary(Y[:, :k].ravel(), P[:, :k].ravel(),
                                                                   S[:, :k].ravel()),
            "per_class": per_class}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _jsonl(path) -> list:
    try:
        return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    except OSError:
        return []


def load(batch_dir, truth, classes=CHD_CLASSES) -> dict:
    """{model: {"rows": [(case, run_folder, Y, P, S)], "missing": [...], "attempts": n}}. truth: case -> present
    classes. Each job's planned cases (plan.json; without it, the cases in results.jsonl) count once, at their last
    attempt."""
    out = {}
    for jdir in sorted(p for p in Path(batch_dir).iterdir() if p.is_dir()):
        rows = _jsonl(jdir / "results.jsonl")
        plan = _read(jdir / "plan.json")
        if plan is None and not rows:
            continue
        last = {}
        for r in rows:
            last[r["case"]] = r
        cases = [c["case"] for c in plan["cases"]] if plan else list(dict.fromkeys(r["case"] for r in rows))
        model = plan["model"] if plan else rows[0]["model"]
        name = plan["name"] if plan else rows[0].get("name")
        m = out.setdefault(model, {"rows": [], "missing": [], "attempts": 0})
        m["attempts"] += len(rows)
        for case in cases:
            r = last.get(case)
            rf = r.get("run_folder") if r else None
            rep = _read(Path(rf) / "trace" / "report.json") if rf else None
            normal = _read(Path(rf) / "trace" / "normal.json") if rf and rep is not None else None
            if rep is None:
                m["missing"].append({"case": case, "job": name, "exit": r.get("exit") if r else "not run"})
            p, s = report_rows(rep, (normal or {}).get("normal"), classes)
            m["rows"].append((case, rf, truth_row(truth(case), classes), p, s))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("batch_dir")
    ap.add_argument("--runtime")
    ap.add_argument("--dataset", choices=("imagechd", "hvsmr"),
                    help="default: the dataset in the jobs' plan.json, else imagechd")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    plans = [_read(f) for f in sorted(Path(a.batch_dir).glob("*/plan.json"))]
    dataset = a.dataset or next((p.get("dataset") for p in plans if p), None) or "imagechd"
    if dataset == "hvsmr":
        from chd_client.task import cohorts
        classes, truth, source = cohorts.HVSMR.classes, cohorts.hvsmr_truth, "hvsmr_clinical.csv"
    else:
        from chd_client.task import imagechd
        root = imagechd.dataset_dir(a.runtime)
        sheet = imagechd.diagnosis_sheet(root, a.runtime)
        flags = imagechd.diagnoses(root, sheet)
        from chd_client.task.datasets import SCORING_CONFIG
        excluded = json.loads(SCORING_CONFIG.read_text())["imagechd"]["diagnosis"].get("excluded_classes", [])
        classes = tuple(c for c in CHD_CLASSES if c not in excluded)
        truth, source = (lambda c: imagechd_truth(flags[int(c)])), sheet.name
    res = {"batch": str(a.batch_dir), "dataset": dataset, "excluded_classes": [c for c in CHD_CLASSES if dataset == "imagechd" and c not in classes], "truth": source, "classes": columns(classes), "models": {}}
    for model, d in load(a.batch_dir, truth, classes).items():
        Y, P, S = (np.array([row[i] for row in d["rows"]]) for i in (2, 3, 4))
        res["models"][model] = {**model_metrics(Y, P, S, classes), "attempts": d["attempts"],
                                "missing": d["missing"],
                                "episodes": [{"case": c, "run_folder": rf} for c, rf, *_ in d["rows"]]}
    Path(a.out or Path(a.batch_dir) / "metrics.json").write_text(json.dumps(res, indent=1) + "\n")
    for model, m in res["models"].items():
        mac = m["macro"]
        print(f"{model}: {m['n_episodes']} planned episodes, {m['attempts']} attempts, {len(m['missing'])} without a "
              f"report; macro over {len(mac['classes'])} classes: " + ", ".join(f"{k} {mac[k]}" for k in METRICS)
              + f"; exact match {m['exact_match']}")


if __name__ == "__main__":
    main()
