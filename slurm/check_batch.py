"""Audit of a batch's run folders (run-isolation checks).

    python3 slurm/check_batch.py audit <batch folder>                        # stdlib only
    uv run python slurm/check_batch.py audit <batch folder> --rescore        # needs the project environment

For every row of every job's results.jsonl:
  - the run folder exists, with exactly one of COMPLETE.json / INCOMPLETE.json, and COMPLETE only when the episode
    finished (stop_reason submitted / cost_cap / max_steps)
  - summary.json names the row's ImageCHD case and its own episode id
  - trace/meta.json and every trace step carry this episode's identity and the reset's loaded-volume sha256
  - provenance/provenance.json records VTK, OpenGL and Mesa (no not_reported)
  - the slicer/ bundle with its truth copy is there
Across the batch: no episode id twice, and one case always has one volume sha256 (whichever job ran it).
--rescore (check 8): re-scores each complete run from its own folder with every dataset path pointing nowhere
(a machine without the dataset) and compares the diagnosis with metrics.json. Exit 1 on any problem.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FINISHED = ("submitted", "cost_cap", "step_budget")
DATASET_KEYS = ("imagechd_dir", "imagechd_masks_dir", "imagechd_diagnosis_sheet")


def read(p):
    return json.loads(Path(p).read_text())


def steps(trace):
    with open(trace / "steps.jsonl") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def audit_run(row) -> tuple[list, dict]:
    probs, facts = [], {}
    d = Path(row["run_folder"]) if row.get("run_folder") else None
    if d is None or not d.is_dir():
        return [f"no run folder ({row.get('run_folder')})"], facts
    s = read(d / "summary.json")
    eid, stop = s["episode_id"], s.get("stop_reason")
    facts.update(episode=eid, stop=stop, dir=d)
    done = [n for n in ("COMPLETE.json", "INCOMPLETE.json") if (d / n).exists()]
    if len(done) != 1:
        probs.append(f"completion markers {done}")
    elif done == ["COMPLETE.json"] and stop not in FINISHED:
        probs.append(f"COMPLETE.json with stop_reason {stop}")
    elif done == ["INCOMPLETE.json"]:
        probs.append(f"INCOMPLETE.json: {read(d / 'INCOMPLETE.json').get('problems')}")
    src = s.get("case_source") or {}
    if src.get("imagechd_id") != row["case"]:
        probs.append(f"case_source {src.get('imagechd_id')} != batch case {row['case']}")
    meta = read(d / "trace" / "meta.json")
    ident = meta.get("identity") or {}
    vol = ident.get("volume_sha256")
    facts["volume"] = vol
    if ident.get("episode_id") != eid:
        probs.append(f"meta identity episode {ident.get('episode_id')} != {eid}")
    if not vol:
        probs.append("meta identity has no volume_sha256")
    n = 0
    for st in steps(d / "trace"):
        i = st.get("identity")
        if i is None:
            continue
        n += 1
        if i.get("episode_id") != eid or i.get("volume_sha256") != vol:
            probs.append(f"step {st.get('step')} {st.get('tool')}: identity {i.get('episode_id')} / "
                         f"{str(i.get('volume_sha256'))[:12]}")
    facts["steps_checked"] = n
    env = read(d / "provenance" / "provenance.json").get("environment") or {}
    if env.get("not_reported"):
        probs.append(f"provenance not_reported {env['not_reported']}")
    facts["render"] = {k: env.get(k) for k in ("vtk", "mesa")} | {"gl": (env.get("opengl") or {}).get("renderer")}
    for f in ("slicer/ct.nii.gz", "slicer/open_me.mrml", "slicer/truth/case.json", "slicer/truth/labels.npz"):
        if not (d / f).exists():
            probs.append(f"missing {f}")
    return probs, facts


def no_dataset_runtime(tmp) -> Path:
    from chd_client.task.config import load_runtime
    rt = load_runtime()
    for k in DATASET_KEYS:
        if k in rt:
            rt[k] = f"{tmp}/no_dataset/{k}"
    p = Path(tmp) / "runtime.json"
    p.write_text(json.dumps(rt))
    return p


SOURCE_KEYS = ("run", "trace", "truth_source", "data_source")


def score_parts(m) -> dict:
    """The comparable part of metrics.json / rescore output: the whole diagnosis score (truth, prediction, per class,
    Brier, the normal call, ...) without where it came from."""
    return {k: v for k, v in (m.get("diagnosis") or {}).items() if k not in SOURCE_KEYS}


def rescore(d, rt, stop=None) -> tuple[list, dict]:
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "CHD_IMAGECHD_DIR")}
    env["SLICER_MCP_RUNTIME"] = str(rt)
    r = subprocess.run(["uv", "run", "python", "-m", "chd_client.task.rescore", "--run", str(d), "--source", "run", "--json"],
                       cwd=REPO, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        return [f"rescore exit {r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}"], {}
    new = json.loads(r.stdout.strip().splitlines()[-1])
    errs = [f"rescore {k}: {v['error']}" for k, v in new.items() if isinstance(v, dict) and v.get("error")]
    old_m = read(d / "metrics.json")
    for name, m in (("metrics.json", old_m), ("offline rescore", new)):
        dx = m.get("diagnosis")
        if name == "offline rescore" and isinstance(dx, dict) and dx.get("error"):
            continue
        scored = isinstance(dx, dict) and not dx.get("error") and dx.get("truth") is not None
        if not scored and (stop == "submitted" or not (isinstance(dx, dict) and dx.get("skipped"))):
            why = (dx or {}).get("error") or (dx or {}).get("skipped") or "missing"
            errs.append(f"{name}: no diagnosis scores ({why})")
    old, got = score_parts(old_m), score_parts(new)
    if old != got and not errs:
        errs.append(f"rescore differs: metrics.json {old} vs offline {got}")
    return errs, got


def audit(batch, do_rescore=False) -> int:
    rows = []
    for res in sorted(Path(batch).glob("job*/results.jsonl")):
        rows += [json.loads(x) for x in res.read_text().splitlines() if x.strip()]
    if not rows:
        print(f"no results.jsonl rows under {batch}")
        return 1
    bad, episodes, volumes = 0, {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        rt = no_dataset_runtime(tmp) if do_rescore else None
        for row in rows:
            probs, f = audit_run(row)
            if f.get("episode"):
                if f["episode"] in episodes:
                    probs.append(f"episode id also in job {episodes[f['episode']]}")
                episodes[f["episode"]] = row["job"]
                volumes.setdefault(row["case"], set()).add(f.get("volume"))
            extra = ""
            if do_rescore and f.get("dir") and f.get("stop") in FINISHED:
                errs, got = rescore(f["dir"], rt, f.get("stop"))
                probs += errs
                extra = f" | offline rescore exact_match {got.get('exact_match')} f1 {got.get('f1')}"
            bad += bool(probs)
            print(f"{'OK  ' if not probs else 'FAIL'} job{row['job']} case {row['case']} {f.get('stop')} "
                  f"{f.get('steps_checked', 0)} steps {f.get('render', '')}{extra}")
            for p in probs:
                print(f"     - {p}")
    for case, vols in volumes.items():
        if len(vols) != 1:
            bad += 1
            print(f"FAIL case {case}: {len(vols)} different volume hashes across jobs")
    print(f"{len(rows) - bad if bad <= len(rows) else 0} of {len(rows)} runs OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit")
    a.add_argument("batch", help="slurm/batches/<batch> folder")
    a.add_argument("--rescore", action="store_true", help="also re-score each run without the dataset (check 8)")
    args = ap.parse_args(argv)
    return audit(args.batch, args.rescore)


if __name__ == "__main__":
    sys.exit(main())
