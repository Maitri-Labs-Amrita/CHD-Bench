"""Diagnosis score of a 3d_seg run: its multi-label classification vs the case's diagnosis flags. No
Slicer.

    uv run python -m chd_client.task.eval_diagnosis --run <run folder> [--json] [--dir <dataset folder>]
                                            [--truth <class or flag> ...]

Reads <run>/trace: meta.json (client_meta.case_source: the dataset and case id, chd_client.task.datasets.for_run) and
report.json (the accepted diagnosis-v1 report). Truth = the ImageCHD case's diagnosis flags (imagechd.diagnoses),
or the classes / flags given with --truth (for a case without dataset flags). Score: chd_client.task.scoring.multilabel.

ImageCHD truth (config/scoring_v1.json imagechd.diagnosis): the 2021 sheet (imagechd.diagnosis_sheet). Every
class is scored (AVSD too) and a case without flags is a normal heart (truth_verified true). The config's
excluded_classes leaves classes out of every number, and unflagged_truth "unverified" marks a case without flags as
unverified. Given truth (--truth) is
taken as verified, with nothing left out.

Returns (prints) the result; writes nothing (the client keeps it in <run>/metrics.json). The flags are read only
here, after the episode: they never go into a trace or the model's input. A run without a submitted report, with
a report that is not diagnosis-v1, or without truth gives
{"skipped": reason}.
"""

import argparse
import json
import sys
from pathlib import Path

from chd_client.task import datasets, imagechd
from chd_client.task.diagnosis_schema import is_diagnosis_report
from chd_client.task import run_truth
from chd_client.task.scoring.multilabel import score
from slicer_mcp.tracing.logger import read_trace

TRACE = "trace"
REPORT = "report.json"


def evaluate_run(run_dir, root=None, truth=None, source="auto") -> dict:
    """The score of a run folder. truth (a list of class names or flags) overrides the dataset's flags."""
    d = Path(run_dir)
    meta, _ = read_trace(d / TRACE)
    base = {"run": d.name, "trace": meta.get("episode_id"), "case_id": meta.get("case_id"),
            "arm": meta.get("arm") or (meta.get("client_meta") or {}).get("arm"),
            "status": meta.get("status")}
    rp = d / TRACE / REPORT
    if not rp.is_file():
        return {**base, "skipped": "no report submitted"}
    report = json.loads(rp.read_text())
    if not is_diagnosis_report(report):
        return {**base, "skipped": "a segmental report, not a diagnosis classification"}
    if truth is not None:
        return _with_normal(d, {**base, "truth_source": "given", "truth_verified": True,
                                **score(report, truth)})
    ds, n = datasets.for_run(meta)
    if ds is None:
        return {**base, "skipped": n}
    if ds.key != "imagechd":
        return {**base, "dataset": ds.name, ds.id_key: n, "skipped": f"no diagnosis flags in {ds.name}"}
    if source == "run" or (source == "auto" and root is None and run_truth.read_case(d) is not None):
        flags_n, sheet_name = run_truth.flags_from_run(d, n)
        if flags_n is None:
            if source == "run":
                return {**base, "dataset": ds.name, ds.id_key: n, "skipped": "no diagnosis flags in the run's copy"}
        else:
            return _with_normal(d, _scored(base, ds, n, report, flags_n, sheet_name, "run"))
    root = root or imagechd.dataset_dir()
    sheet = imagechd.diagnosis_sheet(root)
    if not sheet.exists():
        return {**base, "dataset": ds.name, ds.id_key: n, "skipped": f"diagnosis sheet not found: {sheet} (runtime.json "
                                                                     f"imagechd_diagnosis_sheet)"}
    flags = imagechd.diagnoses(root, sheet)
    if n not in flags:
        return {**base, "dataset": ds.name, ds.id_key: n, "skipped": f"case {n} has no row in {sheet.name}"}
    return _with_normal(d, _scored(base, ds, n, report, flags[n], sheet.name, "dataset"))


def _scored(base, ds, n, report, flags, sheet_name, data_source) -> dict:
    dx = (datasets.scoring_config().get("imagechd") or {}).get("diagnosis") or {}
    verified = bool(flags) or dx.get("unflagged_truth") != "unverified"
    return {
        **base, "dataset": ds.name, ds.id_key: n, "truth_source": sheet_name, "truth_verified": verified,
        "data_source": data_source,
        **({} if verified else {"truth_note": "no flags: a normal heart or lost labels"}),
        **score(report, flags, exclude=dx.get("excluded_classes", ()))}


def _with_normal(d, out: dict) -> dict:
    """The model's normal call (normal.json, --report classes_normal) against truth: normal = no scored class."""
    p = d / TRACE / "normal.json"
    if not p.is_file():
        return out
    call = json.loads(p.read_text())["normal"]
    truth = not out["truth"]
    out["normal_call"] = {"present": call["present"], "confidence": call["confidence"], "truth": truth,
                          "correct": call["present"] == truth}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a run folder (reports/runs/<name>: trace/)")
    ap.add_argument("--json", action="store_true", help="print the result as one JSON object")
    ap.add_argument("--dir", help="the ImageCHD folder (default: runtime.json imagechd_dir)")
    ap.add_argument("--source", default="auto", choices=run_truth.SOURCES,
                    help="the diagnosis flags from the dataset or the run folder's own copy (slicer/truth); auto = the "
                         "dataset with --dir, else the run's copy when present")
    ap.add_argument("--truth", nargs="*",
                    help="the present classes or flags (none = none of the 16), instead of the dataset's")
    a = ap.parse_args(argv)
    try:
        out = evaluate_run(a.run, a.dir, a.truth, a.source)
    except (OSError, *run_truth.ERRORS, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if a.json:
        print(json.dumps(out))
    elif "skipped" in out:
        print(f"{out['run']}: skipped ({out['skipped']})")
    else:
        print(f"{out['run']}: predicted {out['predicted'] or 'none'}, truth {out['truth'] or 'none'}"
              f"{'' if out.get('truth_verified', True) else ' (unverified)'}: exact "
              f"{out['exact_match']:.0f}, F1 {out['f1']}, Hamming {out['hamming_accuracy']}, Brier {out['brier']}")


if __name__ == "__main__":
    main()
