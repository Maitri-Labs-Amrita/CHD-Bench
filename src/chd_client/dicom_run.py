"""Run an episode on a temporary case: nothing about the case is kept afterwards.

`temporary_imagechd_case` converts a case of the public ImageCHD dataset with `slicer-mcp case prepare --volume`
(a headless Slicer, ~20 s; case id case_nii_<hex> from the key imagechd:<id>). The ImageCHD rules are its arguments
(task.imagechd): the spacing from the image-info sheet, values stored as HU + 1024, the LPS-identity
header, and the 3d_seg segmentation shells from --labels with the ImageCHD label codes (the 3d arm: no --labels, no
shells, under the profile chd_ct_nomask_v1). It writes a temporary copy of the
runtime.json (paths made absolute) for the server. The episode then runs on it. On
exit (also after an error) the case's files and the temporary folder are deleted. Traces and runs/<E>/ are kept as
for any episode. Its trace gets the dataset id but never the dataset's diagnosis flags.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import sys

from chd_client import slicer
from chd_client.task import imagechd
from chd_client.task.config import REPO, resolved_runtime

TASK_SCRIPTS = Path(__file__).resolve().parent / "task_scripts"
IMAGECHD_LABEL_MAP = ",".join(f"{code}={name}" for code, name in imagechd.LABELS.items() if code in imagechd.BLOOD_LABELS)

IMAGECHD_TRACE_KEYS = ("imagechd_id", "sha256", "dims", "spacing_mm")
LABELS_KEYS = ("labels", "labels_sha256", "labels_file", "label_map")


class DicomRunError(RuntimeError):
    pass


class CaseIdentityError(RuntimeError):
    pass


def _file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class TemporaryCase:
    case_id: str
    runtime: Path
    info: dict
    run: object = subprocess.run
    identity_check: dict | None = None

    @property
    def imagechd_id(self) -> int | None:
        return self.info.get("imagechd_id")

    def expected_volume_sha256(self) -> str | None:
        """The sha256 of the prepared case's voxels, hashed as the bridge hashes what Slicer loaded: from
        the converter's output when it gives one, else computed once from the case file with slicer-mcp's voxel hash."""
        if self.info.get("volume_sha256"):
            return self.info["volume_sha256"]
        nrrd = self.info.get("nrrd")
        if not nrrd:
            return None
        out = task_python([TASK_SCRIPTS / "voxel_sha256.py", Path(nrrd).resolve()], self.run, timeout_s=600)
        self.info["volume_sha256"] = json.loads(out.strip().splitlines()[-1])["volume_sha256"]
        return self.info["volume_sha256"]

    def check_loaded(self, case_info: dict, volume_sha256: str | None = None) -> dict:
        """Refuse an episode whose loaded volume is not this case. volume_sha256: the hash of the voxels
        Slicer loaded, from the reset's own response (mcp_session.loaded_volume_sha256); it must equal the prepared
        case's voxel hash. The geometry (dims, spacing) must match too. Without a loaded-voxel hash in the reset's
        response the case file is re-hashed instead, and the check says so. Called after reset, before the
        model's first turn, so a mismatch costs no model call. Raises CaseIdentityError; returns the outcome (the
        runner keeps it in summary.json `case_identity`), also kept in `identity_check`."""
        problems = []
        dims, sp = self.info.get("dims"), self.info.get("spacing_mm")
        if dims is not None and list(case_info.get("dims") or []) != list(dims):
            problems.append(f"dims {case_info.get('dims')} != prepared {dims}")
        got_sp = case_info.get("spacing_mm") or []
        if sp is not None and (len(got_sp) != len(sp) or any(abs(a - b) > 1e-3 for a, b in zip(got_sp, sp))):
            problems.append(f"spacing {got_sp} != prepared {sp}")
        check = {"method": None, "loaded_volume_sha256": volume_sha256}
        if volume_sha256:
            check["method"] = "loaded_voxels"
            try:
                want = self.expected_volume_sha256()
            except (DicomRunError, OSError, ValueError, KeyError, IndexError) as e:
                want = None
                problems.append(f"could not hash the prepared voxels: {e}")
            check["prepared_volume_sha256"] = want
            if want and want != volume_sha256:
                problems.append(f"loaded voxels sha256 {volume_sha256[:12]} != prepared {want[:12]}")
            elif not want and not problems:
                problems.append("no prepared voxel hash to compare the loaded volume with")
        else:
            check["method"] = "case_file (the env sent no loaded-voxel hash)"
            nrrd, sha = self.info.get("nrrd"), self.info.get("sha256")
            if nrrd and sha:
                try:
                    got = _file_sha256(nrrd)
                except OSError as e:
                    problems.append(f"case file unreadable: {e}")
                else:
                    if got != sha:
                        problems.append(f"case file sha256 {got[:12]} != prepared {sha[:12]}")
        check["ok"] = not problems
        if problems:
            check["problems"] = problems
        self.identity_check = check
        if problems:
            raise CaseIdentityError(f"{self.case_id}: the loaded volume is not the prepared case: "
                                    + "; ".join(problems))
        return check

    @property
    def has_truth(self) -> bool:
        """A public-dataset case with ground truth to score the run against (ImageCHD)."""
        return self.imagechd_id is not None

    def trace_meta(self) -> dict:
        return {"case_source": {"kind": "imagechd_temporary", **{k: self.info.get(k) for k in IMAGECHD_TRACE_KEYS},
                                **{k: self.info[k] for k in LABELS_KEYS if k in self.info}}}

    def summary_meta(self) -> dict:
        return {"kind": "imagechd_temporary", "source_dir": self.info.get("source_dir"),
                **{k: self.info.get(k) for k in IMAGECHD_TRACE_KEYS},
                **{k: self.info[k] for k in LABELS_KEYS if k in self.info}}


def task_python(args, run=subprocess.run, timeout_s=1200, runtime=None) -> str:
    """`python <args>` in this venv, from CHD-Bench (task scripts, scoring and slicer-mcp's CLI); returns stdout,
    raises DicomRunError on failure. runtime: the run's runtime.json, passed as SLICER_MCP_RUNTIME (so the CLI's
    --runtime reaches scoring)."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    if runtime:
        env["SLICER_MCP_RUNTIME"] = str(Path(runtime).resolve())
    cmd = [sys.executable, *map(str, args)]
    p = run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=timeout_s)
    if p.returncode != 0:
        tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-15:])
        raise DicomRunError(f"{' '.join(map(str, args[:4]))} failed (exit {p.returncode}):\n{tail}")
    return p.stdout


def imagechd_prepare_args(imagechd_id, dataset_dir=None, labels=None, runtime=None, profile=None,
                          label_map=None) -> tuple[list, dict]:
    """`slicer-mcp case prepare` arguments for an ImageCHD case, and the ImageCHD keys for its info. labels: "gt" (the
    case's own label map), a NIfTI label map on the case's grid with the ImageCHD codes (e.g. an nnU-Net prediction),
    or None (no shells). profile: the task profile the case is prepared under (None: chd-slicer's chd_ct_v1); a
    profile without a segmentation (the 3d arm's chd_ct_nomask_v1) takes no labels. Raises DicomRunError for a missing
    case or spacing row, files that are not raw ImageCHD (imagechd.check_raw), or labels with a profile that has no
    segmentation, before Slicer starts."""
    n = int(imagechd_id)
    if labels and profile is not None and not _has_segmentation(profile):
        raise DicomRunError(f"profile {Path(profile).name} has no segmentation, so the case takes no labels "
                            f"(--labels {labels}): leave --labels out, or use the 3d_seg arm's profile")
    root = Path(dataset_dir).expanduser().resolve() if dataset_dir else imagechd.dataset_dir(runtime).resolve()
    if not root.is_dir():
        raise DicomRunError(f"ImageCHD folder not found: {root} (runtime.json imagechd_dir)")
    image, label = imagechd.paths(root, n)
    if not image.is_file():
        raise DicomRunError(f"ImageCHD case {n}: {image.name} not found in {root}")
    try:
        spacing = imagechd.spacing_of(root, n)
    except (ValueError, imagechd.ImageCHDError) as e:
        raise DicomRunError(f"ImageCHD case {n}: {e}") from None
    try:
        imagechd.check_raw(root, n, labels=labels == "gt")
    except imagechd.ImageCHDError as e:
        raise DicomRunError(str(e)) from None
    args = ["--volume", image, "--spacing", *spacing, "--intensity-offset", -imagechd.HU_OFFSET, "--lps-identity",
            "--case-key", case_key(f"imagechd:{n}", labels, label if labels == "gt" else None)]
    extra = {"imagechd_id": n, "source_dir": str(root), "has_labels": label.is_file()}
    if labels:
        lab = label if labels == "gt" else Path(labels).expanduser().resolve()
        if labels != "gt" and not (lab.name.endswith(".nii") or lab.name.endswith(".nii.gz")):
            raise DicomRunError(f"--labels must be 'gt' or a .nii/.nii.gz file: {labels}")
        if not lab.is_file():
            raise DicomRunError(f"label map not found: {lab}")
        args += ["--labels", lab, "--label-map", label_map or IMAGECHD_LABEL_MAP]
        extra["shell_source"] = "gt" if labels == "gt" else "label_map"
    if profile is not None:
        args += ["--profile", profile]
    return args, extra


def _has_segmentation(profile) -> bool:
    """Whether a task profile (a folder or its protocol.json) has a segmentation section (slicer-mcp's rule)."""
    p = Path(profile)
    return json.loads((p / "protocol.json" if p.is_dir() else p).read_text()).get("segmentation") is not None


@contextmanager
def temporary_imagechd_case(imagechd_id, runtime=None, run=subprocess.run, echo=print, dataset_dir=None, labels=None,
                            profile=None, label_map=None):
    """An ImageCHD case (`slicer-mcp case prepare --volume`, ~20 s) for one episode, removed on exit. profile: the
    episode's task profile (None: chd_ct_v1; the 3d arm: chd_ct_nomask_v1, with labels None)."""
    if echo:
        echo(f"preparing ImageCHD case {int(imagechd_id)} ...")
    prep, extra = imagechd_prepare_args(imagechd_id, dataset_dir, labels, runtime, profile, label_map)
    with _temporary(prep, extra, runtime, run, echo) as tc:
        if labels:
            tc.info["labels"] = "gt" if labels == "gt" else str(Path(labels).expanduser().resolve())
            if labels != "gt":
                tc.info["labels_sha256"] = hashlib.sha256(Path(labels).expanduser().read_bytes()).hexdigest()
        yield tc


class CohortCase(TemporaryCase):
    """An HVSMR or custom-scan case (chd_client.task.cohorts): case_source has kind <dataset>_temporary, dataset and the
    cohort's id key; its truth is the dataset's clinical table, scored by run_folder.cohort_metrics."""

    @property
    def cohort(self):
        from chd_client.task.cohorts import for_case_source
        return for_case_source(self.info)[0]

    @property
    def has_truth(self) -> bool:
        return self.cohort.has_truth

    def _keys(self) -> dict:
        c = self.cohort
        ident = ({"name": self.info["name"], "modality": self.info["modality"], "truth": self.info["truth"]}
                 if c.key == "custom" else {c.id_key: self.info[c.id_key]})
        return {"kind": c.kind, "dataset": c.key, **ident, "volume": self.info["volume"],
                **{k: self.info.get(k) for k in ("sha256", "dims", "spacing_mm")},
                **{k: self.info[k] for k in LABELS_KEYS if k in self.info}}

    def trace_meta(self) -> dict:
        return {"case_source": self._keys()}

    def summary_meta(self) -> dict:
        return {**self._keys(), "source_dir": self.info.get("source_dir")}


def case_key(base, labels, gt_file=None) -> str:
    """The case key: the source plus the identity of the shells, so two preparations of one scan with different
    labels never share a case folder."""
    if not labels:
        return f"{base}:nolabels"
    f = gt_file if labels == "gt" else labels
    return f"{base}:{_file_sha256(Path(f).expanduser())[:12]}"


def cohort_prepare_args(cohort, n, labels, workdir, label_map=None, profile=None) -> tuple[list, dict]:
    """`slicer-mcp case prepare --volume` arguments for case n of a cohort (an HVSMR id, or the --volume file of a
    custom scan). The volume keeps its own header geometry; an MR volume is rescaled into workdir first."""
    custom = cohort.key == "custom"
    raw = Path(n).expanduser().resolve() if custom else cohort.raw(n)
    if not raw.is_file():
        raise DicomRunError(f"{cohort.name} case {n}: volume not found: {raw}")
    if custom and not (raw.name.endswith(".nii") or raw.name.endswith(".nii.gz")):
        raise DicomRunError(f"--volume must be a .nii or .nii.gz file: {raw}")
    extra = {"dataset": cohort.key, "source_volume": str(raw), "source_dir": str(raw.parent), "has_labels": False}
    if custom:
        extra.update(name=raw.name.removesuffix(".gz").removesuffix(".nii"), modality=cohort.modality,
                     truth=None if not cohort.has_truth else cohort.truth(None))
    else:
        extra[cohort.id_key] = int(n)
    vol = raw
    if cohort.modality == "mr":
        from chd_client.task.cohorts import normalize_mr
        vol = Path(workdir) / "volume_mr_scaled.nii.gz"
        extra["scaling"] = normalize_mr(raw, vol)
    extra["volume"] = str(vol)
    base = f"{cohort.key}:{extra.get('name') if custom else int(n)}"
    args = ["--volume", vol, "--profile", profile or cohort.profile]
    if labels:
        lab = cohort.gt_labels(n) if labels == "gt" else Path(labels).expanduser().resolve()
        if not lab.is_file():
            raise DicomRunError(f"label map not found: {lab}")
        args += ["--labels", lab, "--label-map", label_map or cohort.label_map]
        extra.update(shell_source="gt" if labels == "gt" else "label_map", labels_file=str(lab),
                     label_map=label_map or cohort.label_map)
    args += ["--case-key", case_key(base, labels, cohort.gt_labels(n) if labels == "gt" else None)]
    return args, extra


@contextmanager
def temporary_cohort_case(cohort, n, labels, runtime=None, run=subprocess.run, echo=print, label_map=None,
                          profile=None):
    """An HVSMR or custom-scan case for one episode, removed on exit."""
    if echo:
        echo(f"preparing {cohort.name} case {n} ...")
    work = Path(tempfile.mkdtemp(prefix=".chd_volume_", dir=Path(runtime).resolve().parent if runtime else None))
    try:
        prep, extra = cohort_prepare_args(cohort, n, labels, work, label_map, profile)
        with _temporary(prep, extra, runtime, run, echo, factory=CohortCase) as tc:
            if labels:
                lab = Path(extra["labels_file"])
                tc.info["labels"] = "gt" if labels == "gt" else str(lab)
                tc.info["labels_sha256"] = _file_sha256(lab)
            yield tc
    finally:
        shutil.rmtree(work, ignore_errors=True)


@contextmanager
def _temporary(prepare_args, extra, runtime, run, echo, factory=TemporaryCase):
    pin = slicer.check_pin()
    launched = slicer.check_launched_pin(resolved_runtime(runtime))
    tmp = Path(tempfile.mkdtemp(prefix=".chd_dicom_run_", dir=Path(runtime).resolve().parent if runtime else None))
    try:
        rt_path = resolved_runtime(runtime, out=tmp / "runtime.json")
        rt_args = ["--runtime", str(rt_path)]
        out = task_python([*slicer.CLI, *slicer.with_defaults(["case", "prepare", *prepare_args, *rt_args])], run)
        info = {**json.loads(out.strip().splitlines()[-1]), **extra, "slicer_mcp_checkout": pin,
                "slicer_mcp_launched": launched}
        case_id = info["case_id"]
        try:
            if echo:
                n = f"{info.get('n_files')} files, " if info.get("n_files") else ""
                echo(f"{case_id}: {n}{info.get('dims')} voxels")
            yield factory(case_id, rt_path, info, run)
        finally:
            lease = ["--lease", info["lease"]] if info.get("lease") else []
            task_python([*slicer.CLI, "case", "remove", case_id, *rt_args, *lease], run)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
