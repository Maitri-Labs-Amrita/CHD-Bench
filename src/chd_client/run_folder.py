"""One folder per run in <project root>/reports/runs/. While a run is going, the runner writes runs/<E>/ and the
env writes traces/<E>/ (the runner and the PDF use those paths); when it is over `collect_run` moves everything into

    reports/runs/<case>_<YYYY-MM-DD>_<HH-MM>_<kind>_<arm>_<prompt>_<model>_<id suffix>/   (folder_name; local time)
        report.pdf  summary.json  transcript.jsonl
        metrics.json        ImageCHD runs: {"diagnosis": chd_client.task.eval_diagnosis, "tool_use": ...}
        slicer/             the case for 3D Slicer (chd_client.task.slicer_bundle): ct.nii.gz, open_me.mrml,
                            README.txt, the ground-truth labels and their scoring copy (truth/)
        trace/              the env trace: meta.json, steps.jsonl, report.json, normal.json, frames/

summary.json's trace_dir / report_pdf then point into the folder, and run_dir names it.
"""

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from chd_client.dicom_run import TASK_SCRIPTS, DicomRunError, task_python
from chd_client.run_record import read_jsonl, sha256_bytes, utc_now, write_json_atomic

METRICS = (("diagnosis", "chd_client.task.eval_diagnosis"),)
METRICS_FILE = "metrics.json"
RUNS_SUBDIR = "runs"
SLICER_DIR = "slicer"


def reports_dir_default() -> Path:
    """Where the run folders go: run_episode.REPORTS_DIR / runs (looked up at call time: tests point it into a tmp
    dir)."""
    from chd_client import run_episode
    return Path(run_episode.REPORTS_DIR) / RUNS_SUBDIR


def scorer_args(module: str, run_dir, dataset_dir=None) -> list:
    """One scorer's arguments for a run folder (scored against the dataset; rescore uses chd_client.task.rescore)."""
    return ["-m", module, "--run", str(run_dir), "--json", *(["--dir", str(dataset_dir)] if dataset_dir else [])]


def imagechd_metrics(run_dir, dataset_dir=None, run=None) -> dict:
    """The evaluations of an ImageCHD run folder (its trace/; the scorer finds the dataset in the trace's case_source) in
    one dict (chd_client.task modules in a subprocess; they write nothing). A failure is that part's {"error"}; a run they do not apply to gives their
    {"skipped"}."""
    out = {}
    for key, module in METRICS:
        args = scorer_args(module, run_dir, dataset_dir)
        try:
            text = (run or task_python)(args, timeout_s=600)
            out[key] = json.loads(text.strip().splitlines()[-1])
        except (DicomRunError, OSError, ValueError, IndexError) as e:
            out[key] = {"error": str(e)}
    return out


COHORT_TRUTH_FILES = ("metrics.json", "slicer/truth/case.json")


def _cohort_case(run_dir):
    from chd_client.task.cohorts import for_case_source
    s = _read_summary(Path(run_dir)) or {}
    return for_case_source(s.get("case_source")), s.get("case_source") or {}


def add_cohort_bundle(run_dir, cohort, echo=None) -> dict | None:
    """slicer/ for an HVSMR or custom-scan run: ct.nii.gz (the volume the agent saw: for HVSMR the scaled MRI), the mask
    it was given as shown_mask.nii.gz (bundle codes, hidden), the scene, and truth/case.json (the dataset's present
    classes and the input hashes). Recorded in summary.json like add_slicer_bundle; never raises."""
    import tempfile
    d = Path(run_dir)
    s = _read_summary(d) or {}
    out, err = None, None
    try:
        import numpy as np
        import SimpleITK as sitk
        from chd_client.task import slicer_bundle as sb
        (c, n), src = _cohort_case(d)
        if c is None or c.key != cohort.key:
            raise ValueError(f"summary case_source is not a {cohort.name} case: {src}")
        vol = Path(src["volume"])
        mask = Path(src["labels_file"]) if src.get("labels_file") else None
        tmp = Path(tempfile.mkdtemp(prefix=".slicer_tmp_", dir=d))
        try:
            shutil.copyfile(vol, tmp / "ct.nii.gz")
            files = ["ct.nii.gz"]
            layers = []
            if mask is not None:
                m = sitk.ReadImage(str(mask))
                code = np.zeros(256, np.uint8)
                for k, v in bundle_remap(src.get("label_map") or c.label_map).items():
                    code[k] = v
                mm = sitk.GetImageFromArray(code[sitk.GetArrayFromImage(m)])
                mm.CopyInformation(m)
                sitk.WriteImage(mm, str(tmp / "shown_mask.nii.gz"), True)
                files.append("shown_mask.nii.gz")
                layers.append(("shown_mask", "shown_mask.nii.gz", False))
            truth = {"format": "chd-cohort-truth-v1", "dataset": c.key, "id_key": c.id_key, "id": n,
                     "classes": list(c.classes), "present": c.truth(n) if c.has_truth else None,
                     "volume_sha256": sha256_bytes(vol.read_bytes())}
            if mask is not None:
                truth["mask_sha256"] = sha256_bytes(mask.read_bytes())
            (tmp / "truth").mkdir()
            write_json_atomic(tmp / "truth" / "case.json", truth)
            sb.write_ctbl(tmp / "chd_labels.ctbl", sb.default_colors())
            mr = c.modality == "mr"
            sb.write_mrml(tmp / sb.SCENE, "ct.nii.gz", layers, [], None if mr else (sb.CT_WINDOW, sb.CT_LEVEL))
            (tmp / "README.txt").write_text(
                f"{c.name} case {c.stem(n)}: ct.nii.gz is the volume the agent saw"
                f"{' (MRI, rescaled so its 99.5th percentile is 1000)' if mr else ' (CT, HU)'}"
                f"{'; shown_mask.nii.gz is the segmentation it was given (hidden)' if mask is not None else ''}. "
                "truth/case.json: the case's present classes. Open open_me.mrml in 3D Slicer.\n")
            files += ["truth/case.json", "chd_labels.ctbl", sb.SCENE, "README.txt"]
            dst = d / SLICER_DIR
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        out = {"scene": sb.SCENE, "files": files}
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    s.pop("slicer_bundle", None)
    s.pop("slicer_bundle_error", None)
    s.update({"slicer_bundle": out} if out is not None else {"slicer_bundle_error": err})
    try:
        write_json_atomic(d / "summary.json", s)
    except OSError:
        pass
    if echo:
        echo(f"slicer bundle: {d / SLICER_DIR}" if out is not None else f"slicer bundle failed: {err}")
    return out


def bundle_remap(label_map: str) -> dict:
    """{label value: bundle code} for a 'value=Name,...' label map (bundle codes: 1 LV 2 RV 3 LA 4 RA 5 Ao 6 PA)."""
    from chd_client.task import slicer_bundle as sb
    codes = {"LV": 1, "RV": 2, "LA": 3, "RA": 4, "Ao": 5, "PA": 6, "Myo": sb.MYO}
    out = {}
    for part in label_map.split(","):
        k, _, name = part.partition("=")
        if name.strip() in codes:
            out[int(k)] = codes[name.strip()]
    return out


def cohort_metrics(run_dir, cohort) -> dict:
    """{"diagnosis": ...} for an HVSMR or custom-scan run: trace/report.json vs the dataset's present classes
    (multilabel.score_classes), plus the model's normal call (normal.json) as eval_diagnosis adds it."""
    from chd_client.task.scoring.multilabel import score_classes
    d = Path(run_dir)
    (c, n), src = _cohort_case(d)
    base = {"run": d.name, "dataset": cohort.name, cohort.id_key: n}
    try:
        if c is None or c.key != cohort.key:
            raise ValueError(f"summary case_source is not a {cohort.name} case")
        if not c.has_truth:
            return {"diagnosis": {**base, "skipped": "no truth given (--truth)"}}
        rp = d / "trace" / "report.json"
        if not rp.is_file():
            return {"diagnosis": {**base, "skipped": "no report submitted"}}
        source = "--truth" if c.key == "custom" else f"{c.name} clinical table"
        out = {**base, "truth_source": source, "truth_verified": True,
               **score_classes(json.loads(rp.read_text()), c.truth(n), c.classes)}
        np_ = d / "trace" / "normal.json"
        if np_.is_file():
            call = json.loads(np_.read_text())["normal"]
            truth = not out["truth"]
            out["normal_call"] = {"present": call["present"], "confidence": call["confidence"], "truth": truth,
                                  "correct": call["present"] == truth}
        return {"diagnosis": out}
    except Exception as e:  # noqa: BLE001
        return {"diagnosis": {**base, "error": f"{type(e).__name__}: {e}"}}


def tool_use(run_dir) -> dict:
    """How the episode used the viewer: calls per tool, the frames it was shown per kind (3d, the slice views),
    and the errors."""
    from collections import Counter
    p = Path(run_dir) / "trace" / "steps.jsonl"
    if not p.is_file():
        return {"error": "no steps.jsonl"}
    steps = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    tools = Counter(s.get("tool") for s in steps if s.get("step"))
    frames = Counter((f.rsplit("/", 1)[-1].split("_", 1)[-1].rsplit(".", 1)[0])
                     for s in steps if s.get("step") for f in (s.get("frames") or []))
    errs = [s for s in steps if s.get("error")]
    return {"n_calls": sum(tools.values()), "calls": dict(tools.most_common()), "frames": dict(frames),
            "errors": len(errs)}


def write_metrics(run_dir, metrics: dict, name: str = METRICS_FILE) -> Path:
    return write_json_atomic(Path(run_dir) / name, metrics)


def refresh_pdf(run_dir, echo=None) -> bool:
    """Render a run folder's report.pdf again after metrics.json is written, so the cover shows the scores. A failure
    is recorded in summary.json (report_pdf_refresh_error, which keeps the run from COMPLETE.json) and echoed, never
    raised. True when re-rendered."""
    d = Path(run_dir)
    if not (d / "report.pdf").is_file() or not (d / "trace" / "meta.json").is_file():
        return False
    s = _read_summary(d)
    try:
        from chd_client.report_pdf import render_episode_pdf, system_tz_name
        e = json.loads((d / "trace" / "meta.json").read_text()).get("episode_id") or d.name
        render_episode_pdf(e, trace_dir=d / "trace", run_dir=d, out=d / "report.pdf", tz=system_tz_name())
    except Exception as ex:  # noqa: BLE001
        err = f"{type(ex).__name__}: {ex}"
        if s is not None:
            write_json_atomic(d / "summary.json", {**s, "report_pdf_refresh_error": err})
        if echo:
            echo(f"could not re-render {d / 'report.pdf'} ({err})")
        return False
    if s is not None and s.pop("report_pdf_refresh_error", None) is not None:
        write_json_atomic(d / "summary.json", s)
    return True


def read_metrics(run_dir) -> dict | None:
    """metrics.json, or None without it."""
    try:
        return json.loads((Path(run_dir) / METRICS_FILE).read_text())
    except (OSError, ValueError):
        return None


def case_label(meta: dict | None, summary: dict | None) -> str | None:
    """The case as a person names it: ImageCHD-<id>, else the case id. The source comes from summary.json's
    case_source, else the trace's client_meta."""
    meta, summary = meta or {}, summary or {}
    src = summary.get("case_source") or (meta.get("client_meta") or {}).get("case_source") or {}
    if src.get("imagechd_id") is not None:
        return f"ImageCHD-{src['imagechd_id']}"
    from chd_client.task.cohorts import for_case_source
    c, n = for_case_source(src)
    if c is not None:
        return f"HVSMR-{n}" if c.key == "hvsmr" else str(n)
    return meta.get("case_id") or summary.get("case")


def model_short(model_id) -> str | None:
    """claude-opus-5-5 -> opus-5-5; others (scripted, openrouter:...) as they are."""
    m = str(model_id) if model_id else None
    return m[len("claude-"):] if m and m.startswith("claude-") else m


def start_time(episode_id: str, meta: dict | None) -> datetime | None:
    """The trace's start_time, else the episode id's UTC prefix (20261013T081835Z_...)."""
    from chd_client.report_pdf import parse_time
    dt = parse_time((meta or {}).get("start_time"))
    if dt is None:
        try:
            dt = datetime.strptime(str(episode_id)[:16], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return dt


def run_kind(meta: dict | None, summary: dict | None) -> str | None:
    """'eval' (an evaluation that submitted its report), 'eval-stopped' (an evaluation that ended without one:
    stopped, step or cost limit, error); None when unknown."""
    meta, summary = meta or {}, summary or {}
    status = summary.get("stop_reason") or meta.get("status")
    if status is None:
        return None
    return "eval" if status == "submitted" else "eval-stopped"


def prompt_tag(meta: dict, summary: dict, arm) -> str | None:
    """The run's prompt version in its folder name: the recorded prompt_version without a leading 'system_' and
    '<arm>_' (e.g. 'chd_ct_v1'); None when the run recorded none."""
    v = (meta.get("client_meta") or summary.get("client_meta") or {}).get("prompt_version")
    return v.removeprefix("system_").removeprefix(f"{arm}_") if v else None


def folder_name(episode_id: str, meta: dict | None, summary: dict | None = None, tz: str = "UTC") -> str:
    """A run folder's name: '<case>_<YYYY-MM-DD>_<HH-MM>_<kind>_<arm>_<prompt>_<model>_<id
    suffix>', local 24-hour start time in `tz`, e.g. 'ImageCHD-1025_2026-01-15_13-19_eval_3d_seg_chd_ct_v1_opus-5-5_78b876'
    (kind: run_kind; kind and prompt left out when unknown). meta: the trace's meta.json; summary: the run's summary.json (either may be empty). Characters
    outside [A-Za-z0-9._-] become '-'."""
    from chd_client.report_pdf import get_zone
    meta, summary = meta or {}, summary or {}
    dt = start_time(episode_id, meta)
    when = dt.astimezone(get_zone(tz)).strftime("%Y-%m-%d_%H-%M") if dt else "unknown-time"
    model = (meta.get("model") or {}).get("id") or summary.get("model")
    arm = meta.get("arm") or (meta.get("client_meta") or {}).get("arm") or summary.get("arm")
    parts = [case_label(meta, summary), when, run_kind(meta, summary), arm, prompt_tag(meta, summary, arm),
             model_short(model), str(episode_id).rsplit("_", 1)[-1]]
    name = "_".join(str(p) for p in parts if p)
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name)


def slicer_bundle_args(run_dir, summary: dict, *, dataset_dir=None) -> list | None:
    """The `chd_client.task.slicer_bundle` arguments for a run folder: --imagechd <id> [--dataset-dir]; None for a run
    that is not an ImageCHD case."""
    src = (summary or {}).get("case_source") or {}
    if src.get("imagechd_id") is None:
        return None
    source = ["--imagechd", int(src["imagechd_id"]), *(["--dataset-dir", str(dataset_dir)] if dataset_dir else [])]
    return ["-m", "chd_client.task.slicer_bundle", "--run", str(run_dir), *source, "--json"]


def add_slicer_bundle(run_dir, *, dataset_dir=None, run=None, echo=None) -> dict | None:
    """Write <run folder>/slicer/ with chd_client.task.slicer_bundle and record the outcome in summary.json
    (`slicer_bundle` {scene, files}, or `slicer_bundle_error`). Never raises: a failure must not lose the run."""
    d = Path(run_dir)
    sp = d / "summary.json"
    s = _read_summary(d)
    s = s if isinstance(s, dict) else {}
    out, err = None, None
    args = slicer_bundle_args(d, s, dataset_dir=dataset_dir)
    if args is None:
        err = "no CT for the Slicer bundle (not an ImageCHD case)"
    else:
        try:
            text = (run or task_python)(args, timeout_s=900)
            res = json.loads(text.strip().splitlines()[-1])
            out = {"scene": res.get("scene"), "files": res.get("files")}
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
    s.pop("slicer_bundle", None)
    s.pop("slicer_bundle_error", None)
    s.update({"slicer_bundle": out} if out is not None else {"slicer_bundle_error": err})
    try:
        if sp.is_file() or s:
            write_json_atomic(sp, s)
    except OSError:
        pass
    if echo:
        echo(f"slicer bundle: {d / SLICER_DIR}" if out is not None else f"slicer bundle failed: {err}")
    return out


def collect_run(episode_id: str, *, runs_dir, traces_dir, reports_dir=None, tz=None) -> Path:
    """Move runs/<E>/ and traces/<E>/ into reports/runs/<name>/ (above; reports_dir = the folder that holds the run
    folders) and return the folder. Call it once nothing writes to either any more (the CLI after the episode)."""
    from chd_client.report_pdf import read_json, system_tz_name
    run = Path(runs_dir) / episode_id
    trace = Path(traces_dir) / episode_id if traces_dir else None
    meta = (read_json(trace / "meta.json") if trace else None) or {}
    if not meta and not run.is_dir():
        raise FileNotFoundError(f"no run or trace for {episode_id}")
    summary = (read_json(run / "summary.json") if run.is_dir() else None) or {}
    dest = Path(reports_dir or reports_dir_default()) / folder_name(episode_id, meta, summary, tz or system_tz_name())
    dest.mkdir(parents=True, exist_ok=True)
    if run.is_dir():
        for item in sorted(run.iterdir()):
            shutil.move(str(item), str(dest / item.name))
        run.rmdir()
    if trace is not None and trace.is_dir():
        shutil.move(str(trace), str(dest / "trace"))
    s = read_json(dest / "summary.json")
    if isinstance(s, dict):
        s["run_dir"] = str(dest)
        if (dest / "trace").is_dir():
            s["trace_dir"] = str(dest / "trace")
        if s.get("report_pdf"):
            s["report_pdf"] = str(dest / "report.pdf")
        s.pop("report_pdf_copy", None)
        write_json_atomic(dest / "summary.json", s)
    return dest


def _read_summary(d: Path):
    try:
        return json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return None


def dataset_dir_for(runtime, case_source: dict | None):
    """The dataset folder of an ImageCHD run from the run's runtime.json (imagechd_dir; relative = relative to
    CHD-Bench), or None (not an ImageCHD case, no runtime, or the key unset: scoring then uses its default)."""
    from chd_client.task.config import load_runtime
    src = case_source or {}
    if src.get("imagechd_id") is None or not runtime:
        return None
    try:
        v = load_runtime(runtime).get("imagechd_dir")
    except (OSError, ValueError, KeyError):
        return None
    return Path(v) if v else None


def input_manifest(summary: dict, *, dataset_dir=None, run=None) -> dict:
    """The input files of a run with their sha256 (provenance `inputs`): a public-dataset case's dataset
    files, read by task_scripts/input_manifest.py (no spreadsheet is copied: the ImageCHD image-info
    sheet only by hash and its spacing values), and the prepared case file's hash (no path). Never raises:
    {"error"} instead. The scoring copy itself is the bundle's slicer/truth/."""
    src = (summary or {}).get("case_source") or {}
    case_file = {k: src.get(k) for k in ("kind", "sha256", "n_files", "dims", "spacing_mm", "labels",
                                         "labels_sha256") if k in src}
    if src.get("imagechd_id") is None:
        return {"dataset": None, "case_file": case_file}
    which = ["--imagechd", int(src["imagechd_id"])]
    args = [TASK_SCRIPTS / "input_manifest.py", *which, *(["--dataset-dir", dataset_dir] if dataset_dir else [])]
    try:
        text = (run or task_python)(args, timeout_s=600)
        out = json.loads(text.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "case_file": case_file}
    return {**out, "case_file": case_file}


COMPLETE_FILE = "COMPLETE.json"
INCOMPLETE_FILE = "INCOMPLETE.json"
REQUIRED_FILES = ("summary.json", "transcript.jsonl", "requests.jsonl", "responses.jsonl",
                  "provenance/system_prompt.txt", "provenance/tools.json", "provenance/provenance.json",
                  "trace/meta.json", "trace/steps.jsonl", "report.pdf", "slicer/ct.nii.gz", "slicer/open_me.mrml")
TRUTH_FILES = ("metrics.json", "slicer/truth/case.json", "slicer/truth/labels.npz")
UNFINISHED_STOPS = (None, "terminated", "runner_exception", "rate_limited", "environment_error",
                    "case_identity_mismatch")


def _parses(p: Path) -> str | None:
    """None when the file is what its name says, else the problem."""
    try:
        if p.suffix == ".json":
            json.loads(p.read_text())
        elif p.suffix == ".jsonl":
            read_jsonl(p)
        elif p.name.endswith(".gz"):
            if p.read_bytes()[:2] != b"\x1f\x8b":
                return "not gzip"
        elif p.suffix == ".npz":
            if p.read_bytes()[:2] != b"PK":
                return "not an npz"
        elif p.suffix == ".pdf":
            if p.read_bytes()[:5] != b"%PDF-":
                return "not a PDF"
        elif p.stat().st_size == 0:
            return "empty"
    except (OSError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def validate_run(run_dir, *, has_truth: bool, truth_files=TRUTH_FILES) -> tuple[list, dict]:
    """(problems, checked): every expected file exists and parses; summary, metrics and bundle carry no error; the
    episode ended on its own; every frame in steps.jsonl and every image sent (requests.jsonl, by sha256) is in the
    trace."""
    d = Path(run_dir)
    problems, checked = [], {}
    for rel in REQUIRED_FILES + (tuple(truth_files) if has_truth else ()):
        p = d / rel
        if not p.is_file():
            problems.append(f"missing {rel}")
            continue
        err = _parses(p)
        if err:
            problems.append(f"{rel}: {err}")
        else:
            checked[rel] = sha256_bytes(p.read_bytes()) if p.stat().st_size < (64 << 20) else "large"
    s = _read_summary(d) or {}
    if s.get("stop_reason") in UNFINISHED_STOPS:
        problems.append(f"episode did not finish: stop_reason {s.get('stop_reason')!r}")
    for k in ("report_pdf_error", "report_pdf_refresh_error", "slicer_bundle_error", "exception", "inputs_error"):
        if s.get(k):
            problems.append(f"summary {k}: {s[k]}")
    if has_truth and (d / METRICS_FILE).is_file():
        m = read_metrics(d) or {}
        problems += [f"metrics {k}: {v['error']}" for k, v in m.items() if isinstance(v, dict) and v.get("error")]
        if s.get("stop_reason") == "submitted":
            dx = m.get("diagnosis")
            if not isinstance(dx, dict) or "exact_match" not in dx or dx.get("skipped"):
                why = (dx or {}).get("skipped") if isinstance(dx, dict) else None
                problems.append(f"the submitted report was not scored{f': {why}' if why else ''}")
    if (d / "requests.jsonl").is_file() and s.get("n_requests") is not None:
        n = len(read_jsonl(d / "requests.jsonl"))
        if n != s["n_requests"]:
            problems.append(f"requests.jsonl has {n} lines, summary says {s['n_requests']}")
    steps = read_jsonl(d / "trace" / "steps.jsonl") if (d / "trace" / "steps.jsonl").is_file() else []
    for st in steps:
        for f in st.get("frames") or []:
            if not (d / "trace" / f).is_file():
                problems.append(f"step {st.get('step')}: frame {f} missing")
    if (d / "requests.jsonl").is_file():
        want = {}
        for line in read_jsonl(d / "requests.jsonl"):
            for im in line.get("images") or []:
                if im.get("frame"):
                    want[im["frame"]] = im["sha256"]
        for frame, sha in want.items():
            p = d / frame
            if not p.is_file():
                problems.append(f"sent image {frame} missing")
            elif sha256_bytes(p.read_bytes()) != sha:
                problems.append(f"sent image {frame}: sha256 differs from what was sent")
        checked["sent_images"] = len(want)
    return problems, checked


def write_completion(run_dir, *, has_truth: bool, truth_files=TRUTH_FILES) -> tuple[bool, list]:
    """COMPLETE.json when validate_run finds nothing wrong, else INCOMPLETE.json with the problems (and no
    COMPLETE.json). The batch audit counts only folders with COMPLETE.json."""
    d = Path(run_dir)
    problems, checked = validate_run(d, has_truth=has_truth, truth_files=truth_files)
    s = _read_summary(d) or {}
    body = {"episode_id": s.get("episode_id"), "stop_reason": s.get("stop_reason"), "validated_at": utc_now(),
            "has_truth": has_truth, "required": list(REQUIRED_FILES + (tuple(truth_files) if has_truth else ()))}
    for name in (COMPLETE_FILE, INCOMPLETE_FILE):
        (d / name).unlink(missing_ok=True)
    if problems:
        write_json_atomic(d / INCOMPLETE_FILE, {**body, "problems": problems})
        return False, problems
    write_json_atomic(d / COMPLETE_FILE, {**body, "files_sha256": checked})
    return True, []


def scorer_commit(repo=None) -> str:
    """The scorer's (CHD-Bench's) short commit, '-dirty' when it has local changes."""
    from chd_client.provenance import git_info
    from chd_client.task.config import REPO
    g = git_info(repo or REPO)
    c = (g.get("commit") or "unknown")[:12]
    return c + ("-dirty" if g.get("dirty") else "")


RESCORE_MODULE = "chd_client.task.rescore"


def rescore(run_dir, *, dataset_dir=None, runtime=None, source="run", run=None, repo=None) -> Path:
    """Score a run folder again with the current scorer (`python -m chd_client.task.rescore`) and write
    metrics_<scorer commit>.json beside metrics.json (never overwritten). source "run" (default): from the run
    folder's own copies (slicer/ct.nii.gz, slicer/truth/), no dataset needed; "dataset" / "auto" as chd_client.task.run_truth describes."""
    import functools
    d = Path(run_dir).resolve()
    if not (d / "trace" / "meta.json").is_file():
        raise FileNotFoundError(f"{d}: not a run folder (no trace/meta.json)")
    runner = run or functools.partial(task_python, runtime=runtime)
    args = ["-m", RESCORE_MODULE, "--run", str(d), "--source", source, "--json",
            *(["--dir", str(dataset_dir)] if dataset_dir else [])]
    try:
        m = json.loads(runner(args, timeout_s=1800).strip().splitlines()[-1])
    except (DicomRunError, OSError, ValueError, IndexError) as e:
        m = {"error": str(e)}
    m["tool_use"] = tool_use(d)
    commit = scorer_commit(repo)
    m["rescored"] = {"scorer_commit": commit, "source": source,
                     "dataset_dir": str(dataset_dir) if dataset_dir else None, "at": utc_now()}
    return write_metrics(d, m, f"metrics_{commit}.json")


def rescore_main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="rescore", description="score a run folder again; writes "
                                 "metrics_<scorer commit>.json beside the run's metrics.json")
    ap.add_argument("run_dir", help="a run folder (reports/runs/<name>/)")
    ap.add_argument("--source", default="run", choices=("run", "dataset", "auto"),
                    help="ground truth: the run folder's own copy (default), the dataset, or auto (env chd_client.task.rescore)")
    ap.add_argument("--dataset-dir", help="the dataset folder (with --source dataset / auto)")
    ap.add_argument("--runtime", help="the runtime.json to score with (imagechd_dir; SLICER_MCP_RUNTIME)")
    a = ap.parse_args(argv)
    p = rescore(a.run_dir, dataset_dir=a.dataset_dir, runtime=a.runtime, source=a.source)
    m = json.loads(p.read_text())
    errors = {k: v["error"] for k, v in m.items() if isinstance(v, dict) and v.get("error")}
    if m.get("error"):
        errors["rescore"] = m["error"]
    print(p)
    for k, e in errors.items():
        print(f"{k}: {e}")
    return 1 if errors else 0
