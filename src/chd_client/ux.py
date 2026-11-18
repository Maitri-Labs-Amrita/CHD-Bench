"""What a person sees around a run: the plan before it starts (run_plan, shown by ./run.sh with a confirmation for a
paid model) and the result box after it ends (result_box: the model's diagnosis, the truth and the score when there
is one, the cost, and the path of report.pdf). Plain words; no internal codes without an explanation."""

import json
import os
import sys
from pathlib import Path

RULE = "=" * 64

STOP_TEXT = {
    "submitted": "finished: the model submitted its diagnosis",
    "step_budget": "stopped: the model used all its steps without a diagnosis (raise --max-steps)",
    "cost_cap": "stopped: the cost limit was reached before a diagnosis (raise --cost-cap)",
    "cost_unknown": "stopped: the provider did not report a cost, so the run stopped to protect the budget",
    "no_tool_call": "stopped: the model answered without using a tool or submitting a diagnosis",
    "environment_error": "stopped: 3D Slicer failed during the run (see the log above)",
    "rate_limited": "stopped: the model provider refused more requests (rate limit). Wait and run again",
    "model_refusal": "stopped: the model refused the task",
    "model_max_tokens": "stopped: the model's answer was cut off at its output limit (raise --max-tokens)",
    "runner_exception": "stopped: an error in CHD-Bench (see the error above)",
    "case_identity_mismatch": "stopped: Slicer showed a different case than the one prepared",
    "resume_mismatch": "stopped: the resumed run did not match its first part",
    "terminated": "stopped: you (or the system) stopped the run",
}


STEP_WORDS = {
    "reset_view": "reset the view", "bookmark_view": "bookmark a view as evidence",
    "submit_report": "submit the diagnosis", "set_slice": "go to a slice", "scroll_slice": "scroll through slices",
    "set_window_preset": "change the contrast", "jump_to_point": "jump to a point", "point_at": "point at a spot",
    "orbit": "rotate the 3D view", "set_view_direction": "turn the 3D view", "zoom": "zoom", "pan_to": "move the view",
    "set_render_preset": "change the 3D rendering", "set_clip_box": "cut away part of the 3D view",
    "set_clip_plane": "cut the 3D view with a plane", "clear_clip": "remove the cuts",
    "set_segment_visibility": "show or hide heart structures", "set_segment_opacity": "change structure transparency",
    "set_3d_content": "change what the 3D view shows", "set_slice_outlines": "show or hide outlines on slices",
    "center_on_segments": "center on heart structures", "measure": "measure a distance",
}


def step_line(n, max_steps, tool, cost, seconds, refused=False) -> str:
    """One progress line per step: 'step 12/75 · rotate the 3D view · $0.84 so far · 6 min'."""
    what = STEP_WORDS.get(tool, tool.replace("_", " "))
    if refused:
        what += " (the viewer refused it; the model is told why)"
    return f"step {n}/{max_steps} · {what} · ${cost:.2f} so far · {_minutes(seconds)}"


def stop_text(stop) -> str:
    if stop in STOP_TEXT:
        return STOP_TEXT[stop]
    if str(stop).startswith("model_"):
        return f"stopped: the model ended its answer in an unexpected way ({str(stop)[6:]})"
    return f"stopped: {stop}"


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _class_name(c: str) -> str:
    from chd_client.diagnosis import LONG
    return LONG.get(c, c.replace("_", " ").capitalize())


def _pct(x) -> str:
    return f"{float(x) * 100:.0f}%" if isinstance(x, (int, float)) else "?"


def _minutes(s) -> str:
    if not isinstance(s, (int, float)):
        return "?"
    return f"{s / 60:.0f} min" if s >= 90 else f"{s:.0f} s"


def model_call(run_dir) -> dict | None:
    """{"present": [(class, confidence)], "n_classes": int, "normal": (present, confidence) | None} from the run's
    trace/report.json (and trace/normal.json); None when no report was submitted."""
    from chd_client.diagnosis import class_names, present
    t = Path(run_dir) / "trace"
    rep = _load(t / "report.json")
    if not isinstance(rep, dict) or not isinstance(rep.get("classes"), dict):
        return None
    cls = rep["classes"]
    out = {"present": [(c, (cls.get(c) or {}).get("confidence")) for c in present(rep)],
           "n_classes": len(class_names(rep)), "normal": None}
    nj = _load(t / "normal.json")
    n = (nj or {}).get("normal") if isinstance(nj, dict) else None
    if n is None and isinstance(cls.get("normal"), dict):
        n = cls["normal"]
    if isinstance(n, dict) and "present" in n:
        out["normal"] = (bool(n["present"]), n.get("confidence"))
    return out


def result_box(run_dir=None, error=None, before_model=False) -> list[str]:
    """The lines of the result box for a finished run folder (reports/runs/<run>), or for a run that ended without
    one (error: the reason; before_model: it ended before the first model call, so nothing was charged)."""
    lines = ["", RULE, " RESULT".center(64), RULE]
    if run_dir is None:
        reason = error or "unknown (see the messages above)"
        if before_model:
            lines += [" The run stopped before the model started, so nothing was charged.", f" Reason: {reason}"]
        else:
            lines += [" The run ended without a result.", f" Reason: {reason}",
                      " Model calls made before it stopped may have been charged: check your provider's usage page.",
                      " A partial run folder may be in reports/runs/ (the newest one)."]
        lines += [RULE, ""]
        return lines
    d = Path(run_dir)
    s = _load(d / "summary.json") or {}
    m = _load(d / "metrics.json") or {}
    stop = s.get("stop_reason")
    lines.append(f" Status          {stop_text(stop)}")
    call = model_call(d)
    if call is None:
        lines.append(" Diagnosis       none: the model did not submit one")
    else:
        if call["present"]:
            for i, (c, conf) in enumerate(call["present"]):
                lines.append(f" {'Model diagnosis' if i == 0 else '':<15} {_class_name(c)} (confidence {_pct(conf)})")
        else:
            lines.append(f" Model diagnosis no defect found (none of the {call['n_classes']} classes)")
        if call["normal"] is not None:
            yes, conf = call["normal"]
            lines.append(f" {'':<15} heart called normal: {'yes' if yes else 'no'} (confidence {_pct(conf)})")
    dx = m.get("diagnosis") or {}
    if "exact_match" in dx:
        truth = ", ".join(_class_name(c) for c in dx.get("truth") or []) or "normal (no defect)"
        verdict = "right" if dx["exact_match"] == 1 else ("partly right" if dx.get("f1", 0) > 0 else "wrong")
        lines.append(f" True diagnosis  {truth}")
        lines.append(f" Score           {verdict} (exact match {dx['exact_match']:.0f}, F1 {dx.get('f1', 0):.2f})")
    elif dx.get("error"):
        lines.append(f" Score           not scored: {dx['error']}")
    elif call is not None:
        lines.append(" Score           not scored: no true diagnosis given (--truth)")
    wall = (s.get("wall_s") or {}).get("total")
    cost = s.get("cost_usd")
    lines.append(f" Cost / time     ${cost:.2f} · {_minutes(wall)} · {s.get('n_model_steps', '?')} steps"
                 if isinstance(cost, (int, float)) else f" Cost / time     ? · {_minutes(wall)}")
    lines.append(f" Model           {s.get('model', '?')}")
    pdf = d / "report.pdf"
    lines.append(f" Report (PDF)    {pdf.resolve()}" if pdf.is_file() else
                 f" Report (PDF)    not written ({s.get('report_pdf_error') or 'see the log above'})")
    lines.append(f" Run folder      {d.resolve()}")
    if call is not None:
        lines.append(" Model output for research only, not a clinical diagnosis.")
    lines += [RULE, ""]
    return lines


# ---- the plan before a run --------------------------------------------------------------------------------------

def is_paid(a) -> bool:
    """A run that sends requests to a paid provider: not the scripted dry run and not a local: model."""
    if getattr(a, "dry_run_scripted", False):
        return False
    return not str(a.model).startswith("local:")


def missing_data(a) -> str | None:
    """Why the scan of the parsed options cannot be found (a plain sentence with the fix), or None."""
    if a.imagechd is not None:
        from chd_client.task import imagechd
        try:
            where = imagechd.dataset_dir(a.runtime)
        except Exception:  # noqa: BLE001
            where = None
        if not where or not Path(where).is_dir():
            return (f"the ImageCHD folder was not found ({where or 'not set'}). Tell CHD-Bench where it is: "
                    "./setup.sh --imagechd-dir /path/to/ImageCHD_dataset")
        if not any(Path(where).glob(f"ct_{a.imagechd}_image.nii*")):
            return (f"ImageCHD case {a.imagechd} is not in {where} (no ct_{a.imagechd}_image.nii.gz). List the cases: "
                    "./run.sh --list imagechd")
    elif a.hvsmr is not None:
        from chd_client import paths
        try:
            img = Path(paths.get("hvsmr_images"))
        except Exception:  # noqa: BLE001
            img = None
        if img is None or not img.is_dir():
            return (f"the HVSMR-2.0 folder was not found ({img or 'not set'}). Tell CHD-Bench where it is: "
                    "./setup.sh --hvsmr-dir /path/to/HVSMR-2.0")
        if not (img / f"pat{a.hvsmr}_orig.nii.gz").is_file():
            return f"HVSMR case {a.hvsmr} is not in {img} (no pat{a.hvsmr}_orig.nii.gz). List the cases: ./run.sh --list hvsmr"
    elif a.volume is not None and not Path(a.volume).expanduser().exists():
        return f"the scan {a.volume} does not exist. Check the path (--volume takes a file or a folder)"
    return None


def _scan_line(a, coh) -> tuple[str, str]:
    if a.imagechd is not None:
        from chd_client.task import imagechd
        try:
            where = imagechd.dataset_dir(a.runtime)
        except Exception:  # noqa: BLE001
            where = None
        return "Scan", f"ImageCHD case {a.imagechd} (CT)" + (f", from {where}" if where else "")
    if a.hvsmr is not None:
        from chd_client import paths
        try:
            where = paths.get("hvsmr_images")
        except Exception:  # noqa: BLE001
            where = None
        return "Scan", f"HVSMR-2.0 case {a.hvsmr} (MRI)" + (f", from {where}" if where else "")
    from chd_client.dicom_run import is_nifti
    p = Path(a.volume).expanduser()
    kind = "NIfTI, used as it is" if is_nifti(p) else (
        "a DICOM folder, converted to NIfTI first" if p.is_dir() else "converted to NIfTI first")
    return "Scan", f"your scan {p} ({kind})"



# ---- checks before a run: the scan, the segmentation, privacy ---------------------------------------------------

def provider_of(model: str) -> str:
    """The provider a --model string sends requests to, in words."""
    m = str(model)
    if ":" not in m:
        return "Anthropic"
    return {"openrouter": "OpenRouter", "openrouter_bedrock": "OpenRouter", "openai": "OpenAI",
            "anthropic": "Anthropic", "local": "your own model server"}.get(m.split(":", 1)[0], m.split(":", 1)[0])


def dicom_info(path) -> dict | None:
    """For a --volume that is a DICOM folder or file: {"folder", "n_files", "n_series", "modality"} of the series
    the run will use (dicom_run.to_nifti picks it the same way); None for any other scan."""
    import SimpleITK as sitk
    from chd_client.dicom_run import _dicom_series_files, is_nifti
    p = Path(path).expanduser()
    if not p.exists() or (p.is_file() and is_nifti(p)):
        return None
    if p.is_dir():
        uid, files, n = _dicom_series_files(p, recursive=True)
    else:
        try:
            kind = sitk.ImageFileReader.GetImageIOFromFileName(str(p))
        except RuntimeError:
            kind = ""
        if kind != "GDCMImageIO":
            return None
        uid, files, n = _dicom_series_files(p.parent, p)
    if not files:
        return None
    modality = ""
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(str(files[0]))
        r.LoadPrivateTagsOff()
        r.ReadImageInformation()
        if r.HasMetaDataKey("0008|0060"):
            modality = r.GetMetaData("0008|0060").strip().upper()
    except RuntimeError:
        pass
    return {"folder": str(Path(files[0]).parent), "n_files": len(files), "n_series": n, "modality": modality}


def _header(path):
    """(size, spacing, origin, direction) of an image file from its header only, or None."""
    import SimpleITK as sitk
    try:
        r = sitk.ImageFileReader()
        r.SetFileName(str(path))
        r.ReadImageInformation()
        return r.GetSize(), r.GetSpacing(), r.GetOrigin(), r.GetDirection()
    except RuntimeError:
        return None


def _scan_file(a) -> Path | None:
    """The single image file of the run's scan (None for a DICOM scan)."""
    if a.imagechd is not None:
        from chd_client.task import imagechd
        try:
            return next(imagechd.dataset_dir(a.runtime).glob(f"ct_{a.imagechd}_image.nii*"), None)
        except Exception:  # noqa: BLE001
            return None
    if a.hvsmr is not None:
        from chd_client import paths
        try:
            return Path(paths.get("hvsmr_images")) / f"pat{a.hvsmr}_orig.nii.gz"
        except Exception:  # noqa: BLE001
            return None
    p = Path(a.volume).expanduser()
    return p if p.is_file() and _header(p) is not None and dicom_info(p) is None else None


def label_check(a, dicom=None) -> tuple[list, list, str | None]:
    """(plan rows, warnings, error) for a --segmentation file: its values against the label map, its grid against the
    scan's. Nothing for --segmentation gt or no --segmentation."""
    if a.labels in (None, "gt"):
        return [], [], None
    import numpy as np
    import SimpleITK as sitk
    from chd_client.task.cohorts import IMAGECHD_LABEL_MAP
    lab = Path(a.labels).expanduser()
    if not lab.is_file():
        return [], [], f"the segmentation file {a.labels} does not exist. Check the path after --segmentation"
    try:
        img = sitk.ReadImage(str(lab))
    except RuntimeError as e:
        return [], [], f"the segmentation file {a.labels} cannot be read ({str(e).strip().splitlines()[-1]})"
    values, counts = np.unique(sitk.GetArrayViewFromImage(img), return_counts=True)
    found = {int(v): int(c) for v, c in zip(values, counts) if v != 0}
    default = a.label_map is None
    text = a.label_map or IMAGECHD_LABEL_MAP
    try:
        mapping = {int(k): v.strip() for k, v in (part.split("=") for part in text.split(","))}
    except ValueError:
        return [], [], (f"--segmentation-map {text!r} is not in the form value=structure,..., e.g. "
                        "1=LV,2=RV,3=LA,4=RA,6=Ao,7=PA")
    rows, warns = [], []
    shown = ", ".join(f"{v}={n}" for v, n in sorted(mapping.items()))
    rows.append(("Structures", f"{shown} ({'ImageCHD values, the default' if default else 'from --segmentation-map'})"))
    if not found:
        return rows, warns, f"the segmentation file {a.labels} has no labelled voxels (all values are 0)"
    # ImageCHD's own files also hold 5 (myocardium) and, in some cases, 14 (airway): known, not shown
    extra = sorted(set(found) - set(mapping) - ({5, 14} if default else set()))
    absent = sorted(set(mapping) - set(found))
    if extra or absent:
        parts = []
        if extra:
            parts.append(f"the file has values {', '.join(map(str, extra))}, which the label map does not name "
                         "(not shown)")
        if absent:
            parts.append("the label map expects " + ", ".join(f"{v} ({mapping[v]})" for v in absent) +
                         ", which the file does not have")
        warns.append("The segmentation does not match the label map: " + "; ".join(parts) + ". If the file uses "
                     "other values, pass --segmentation-map (HVSMR's: --segmentation-map 1=LV,2=RV,3=LA,4=RA,5=Ao,6=PA).")
    if dicom is not None:
        warns.append("The segmentation must be on the voxel grid of the scan as CHD-Bench converts it from DICOM. "
                     "If it was made on a NIfTI copy of this scan, give that NIfTI file as --volume instead.")
        return rows, warns, None
    scan = _scan_file(a)
    h = _header(scan) if scan is not None else None
    if h is not None:
        if tuple(img.GetSize()) != tuple(h[0]):
            return rows, warns, (f"the segmentation's grid ({'x'.join(map(str, img.GetSize()))} voxels) is not the "
                                 f"scan's ({'x'.join(map(str, h[0]))}). It must be made on this scan")
        if a.imagechd is None and (not np.allclose(img.GetSpacing(), h[1], atol=1e-3)
                                   or not np.allclose(img.GetOrigin(), h[2], atol=1e-2)
                                   or not np.allclose(img.GetDirection(), h[3], atol=1e-3)):
            warns.append("The segmentation has the scan's size, but not the same spacing, origin or orientation "
                         "in its header. Check that it was made on this scan.")
    return rows, warns, None


def intensity_warning(a, dicom=None) -> str | None:
    """A warning when a --volume NIfTI's values do not look like the --modality given (CT in HU goes below -500;
    MRI does not)."""
    if a.volume is None or dicom is not None:
        return None
    import numpy as np
    import SimpleITK as sitk
    p = _scan_file(a)
    if p is None:
        return None
    try:
        img = sitk.ReadImage(str(p))  # kept in a name: a view of a freed image reads freed memory
        lo = float(np.min(sitk.GetArrayViewFromImage(img)))
    except RuntimeError:
        return None
    if a.modality == "ct" and lo > -100:
        return (f"The scan's lowest value is {lo:.0f}: a CT in HU goes down to about -1000 (air). If this is an MRI, "
                "add --modality mr. If it is a CT without HU scaling, convert it to HU first.")
    if a.modality == "mr" and lo < -500:
        return f"The scan's lowest value is {lo:.0f}, which looks like a CT in HU, not an MRI. Check --modality."
    return None


def run_plan(a, dicom=None, modality_note=None) -> tuple[list[str], list[str]]:
    """(lines, warnings) describing what the parsed options will run. dicom: dicom_info of the --volume (run.sh
    reads it once); modality_note: where the modality came from."""
    from chd_client.run_episode import NOMASK_ARM, DEFAULT_MAX_STEPS, cohort_of
    coh = cohort_of(a)
    rows, warn = [], []
    rows.append(_scan_line(a, coh))
    if a.volume is not None:
        mod = "CT" if a.modality == "ct" else "MRI"
        if modality_note and modality_note.startswith("WARNING: "):
            warn.append(modality_note[len("WARNING: "):] + ".")
            how = "from --modality"
        elif modality_note:
            how = modality_note
        elif a.modality == "ct":
            how = "the default; add --modality mr for an MRI"
        else:
            how = "from --modality"
        rows.append(("Modality", f"{mod} ({how})"))
    if dicom is not None:
        more = (f"; {dicom['n_series']} series found, using the one with the most files" if dicom["n_series"] > 1
                else "")
        rows.append(("DICOM series", f"{dicom['folder']} ({dicom['n_files']} files{more})"))
        if dicom["modality"] and dicom["modality"] not in ("CT", "MR"):
            warn.append(f"The DICOM series is not a CT or an MRI (Modality {dicom['modality']}). Check that the "
                        "folder holds the scan.")
    w = intensity_warning(a, dicom)
    if w:
        warn.append(w)
    if a.arm == NOMASK_ARM:
        rows.append(("Segmentation", "none"))
        warn.append("No segmentation (--segmentation): the model gets the scan alone. Add --segmentation gt (dataset scans) "
                    "or --segmentation <file>.")
    else:
        seg = "the dataset's own" if a.labels == "gt" else f"{a.labels}"
        rows.append(("Segmentation", seg))
        lrows, lwarn, _ = label_check(a, dicom)
        rows += lrows
        warn += lwarn
    if a.arm == "2d_seg":
        rows.append(("Viewer", "slices only: no 3D view (--no-3d)"))
    else:
        rows.append(("Viewer", "3D view and slices"))
    if a.imagechd is not None:
        rows.append(("Truth", "the ImageCHD diagnosis table"))
    elif a.hvsmr is not None:
        rows.append(("Truth", "the HVSMR-2.0 clinical table"))
    elif coh is not None and coh.has_truth:
        t = ", ".join(_class_name(c) for c in coh.truth(None)) or "normal (no defect)"
        rows.append(("Truth", f"{t} (--truth)"))
    else:
        rows.append(("Truth", "not given"))
        warn.append("No true diagnosis (--truth): the run is not scored.")
    if a.dry_run_scripted:
        rows.append(("Model", "none: scripted test (--dry-run-scripted)"))
    elif str(a.model).startswith("local:"):
        rows.append(("Model", f"{a.model} (your own server)"))
    else:
        rows.append(("Model", f"{a.model} (paid API)"))
    steps = a.max_steps if a.max_steps is not None else DEFAULT_MAX_STEPS
    cap = "no cost limit" if a.cost_cap is None else f"stops at ${a.cost_cap:.2f}"
    rows.append(("Limits", f"{cap} · at most {steps} steps"))
    if is_paid(a) and a.cost_cap is None:
        warn.append("No cost limit (--cost-cap none): only the step limit (--max-steps) stops the run.")
    if is_paid(a) and a.volume is not None:
        warn.append(f"Images of your scan are sent to {provider_of(a.model)}. Use de-identified scans only: "
                    "CHD-Bench drops the DICOM patient fields, but it cannot remove text burned into the images.")
    lines = ["", "CHD-Bench run plan"] + [f"  {k:<13} {v}" for k, v in rows]
    return lines, warn


def confirm(prompt: str, stdin=None, stdout=None) -> bool:
    """Ask a yes/no question on the terminal; no answer (EOF) is no."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stderr
    stdout.write(prompt)
    stdout.flush()
    try:
        ans = stdin.readline()
    except (OSError, KeyboardInterrupt):
        return False
    return ans.strip().lower() in ("y", "yes")


def print_lines(lines, file=None):
    for ln in lines:
        print(ln, file=file or sys.stdout)


def write_result_pointer(run_dir=None, error=None, env=os.environ, before_model=False) -> None:
    """Under ./run.sh: write where the result is ($CHD_RUN_SH_RESULT) so run.sh prints the box after Slicer stops."""
    p = env.get("CHD_RUN_SH_RESULT")
    if not p:
        return
    try:
        Path(p).write_text(json.dumps({"run_dir": str(run_dir) if run_dir else None, "error": error,
                                       "before_model": before_model}))
    except OSError:
        pass
