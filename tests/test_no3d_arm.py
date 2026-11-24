"""The no-3D arm 2d_seg: the 3d_seg arm without the 3D view. Its task profiles (profiles/*_no3d_v1), its
prompts (chd_ct_2dseg_v1 and chd_mr_2dseg_v1), run-episode (the tool check and the
runtime guard on 3D frames), slurm/run_batch.py, ./run.sh and the 3d_seg arm left as it was. No Slicer, no API call."""

import contextlib
import hashlib
import json
import re
import types
from pathlib import Path

import anyio
import pytest

from chd_client import dicom_run, prompt_registry, run_sh
from chd_client import run_episode as re_mod
from chd_client.mcp_session import EnvironmentAbort, EnvMCP, Observation
from chd_client.models.scripted import ScriptedModel
from chd_client.run_episode import PROMPT_DIR, dry_run_script, finish_run, main, render_system, run_episode
from chd_client.task import cohorts
from chd_client.task.config import NO3D_PROFILE, PROFILE
from slicer_mcp.config import load_profile
from slicer_mcp.tools.registry import THREED_TOOLS, tool_specs
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, make_env
from tests.test_nomask_arm import config, fake_cli, rb
from tests.test_run_folder import IMAGECHD_SRC, fake_env_python

ROOT = Path(__file__).resolve().parents[1]
NO3D = "chd_ct_2dseg_v1"
LITE_V3 = "chd_ct_v1"
NO3D_MR = "chd_mr_2dseg_v1"
LITE_V3_MR = "chd_mr_v1"
SLICES = ["axial", "sagittal", "coronal"]
TOOLS_12 = ["reset_view", "bookmark_view", "submit_report", "set_slice", "scroll_slice", "set_window_preset",
            "jump_to_point", "point_at", "zoom", "set_segment_visibility", "set_slice_outlines", "measure"]
# (3d_seg profile, 2d_seg profile, 3d_seg prompt, 2d_seg prompt, dataset)
ARMS = [(PROFILE, NO3D_PROFILE, LITE_V3, NO3D, None),
        (cohorts.HVSMR.profile, cohorts.HVSMR.no3d_profile, LITE_V3_MR, NO3D_MR, "hvsmr")]
IDS = ["imagechd", "hvsmr"]


# ---------------------------------------------------------------------- the profiles

@pytest.mark.parametrize("parent, no3d, v3, v_no3d, ds", ARMS, ids=IDS)
def test_each_no3d_profile_is_its_3d_seg_profile_without_the_3d_view(parent, no3d, v3, v_no3d, ds):
    seg, n3 = (json.loads((p / "protocol.json").read_text()) for p in (parent, no3d))
    assert n3["mcp"]["tools"] == [t for t in seg["mcp"]["tools"] if t not in THREED_TOOLS] == TOOLS_12
    assert n3["observation"] == {"views": SLICES, **seg["observation"]}
    for k in set(seg) - {"mcp", "observation", "config_version", "_comment"}:  # segmentation, presets, camera, ...
        assert n3[k] == seg[k], k
    assert {k: v for k, v in n3["mcp"].items() if k != "tools"} == {k: v for k, v in seg["mcp"].items() if k != "tools"}
    assert set(n3) == set(seg) and n3["config_version"] == seg["config_version"].replace("-v1", "-no3d-v1")
    assert n3["_comment"].startswith(f"Profile {no3d.name}: {parent.name} without the 3D view")
    for f in ("presets.xml", "report.py"):
        assert (no3d / f).read_bytes() == (parent / f).read_bytes(), f
    assert sorted(p.name for p in no3d.iterdir() if p.is_file()) == ["presets.xml", "protocol.json", "report.py"]


@pytest.mark.parametrize("parent, no3d, v3, v_no3d, ds", ARMS, ids=IDS)
def test_the_env_serves_no_3d_view_and_no_3d_text(parent, no3d, v3, v_no3d, ds):
    from slicer_mcp.tools_md import render
    p = load_profile(no3d)
    assert p.views == tuple(SLICES) and not p.has_3d and list(p.tools) == TOOLS_12 and p.segmentation
    assert p.report.SCHEMA_VERSION == load_profile(parent).report.SCHEMA_VERSION
    for s in tool_specs(p):
        if s.runner_only:
            continue
        text = s.description + json.dumps(s.input_schema)
        assert not re.search(r"3d|3D", text), s.name
        assert not [t for t in THREED_TOOLS if t in text], s.name
    zoom = next(s for s in tool_specs(p) if s.name == "zoom")
    assert zoom.input_schema["properties"]["view"] == {
        "type": "string", "enum": ["slices"], "default": "slices",
        "description": "'slices' (the only choice) zooms the three slice views about the focus"}
    md = render(str(no3d))
    assert "| 12 | `reset_episode`, `end_episode`, `observe` | axial, sagittal, coronal |" in md


# ---------------------------------------------------------------------- the prompts

def _from_parent(text: str) -> str:
    """The 2d_seg prompt as the registry entry says it is made from its parent."""
    text = text.replace(" in a 3D viewer that ", " in a viewer that ")
    text = re.sub(r"- \*\*3D view:\*\* .*\n", "", text)
    text = text.replace("`set_segment_visibility`, `set_segment_opacity`, `set_3d_content`, `set_slice_outlines`, "
                        "`center_on_segments`.", "`set_segment_visibility`, `set_slice_outlines`.")
    return text.replace("`point_at` (reads the HU at a point), `measure`",
                        "`point_at` (reads the HU at a point), `zoom`, `measure`")


@pytest.mark.parametrize("parent, no3d, v3, v_no3d, ds", ARMS, ids=IDS)
def test_each_no3d_prompt_is_its_parent_with_the_3d_tools_deleted(parent, no3d, v3, v_no3d, ds):
    v3_text, text = ((PROMPT_DIR / f"{v}.md").read_text() for v in (v3, v_no3d))
    assert text == _from_parent(v3_text) != v3_text
    # line by line: three lines changed, one deleted, nothing added
    old, new = v3_text.splitlines(), text.splitlines()
    assert len(old) - len(new) == 1 and set(new) - set(old) == {
        next(ln for ln in new if "in a viewer that" in ln),
        next(ln for ln in new if ln.startswith("- **Slices:**")),
        "- **Segmentation display:** `set_segment_visibility`, `set_slice_outlines`."}
    named = set(re.findall(r"`([a-z_0-9]+)`", text)) - {"calls_left", "classes", "present", "confidence", "normal",
                                                         "evidence"}
    assert named == set(load_profile(no3d).tools)  # exactly the profile's tools
    assert not [t for t in THREED_TOOLS if t in text] and "3D view" not in text and "3D viewer" not in text


@pytest.mark.parametrize("parent, no3d, v3, v_no3d, ds", ARMS, ids=IDS)
def test_each_no3d_prompt_is_registered_for_2d_seg(parent, no3d, v3, v_no3d, ds):
    reg = prompt_registry.load(PROMPT_DIR)
    e = reg[v_no3d]
    assert e["sha256"] == hashlib.sha256((PROMPT_DIR / e["file"]).read_bytes()).hexdigest()
    assert e["arms"] == ["2d_seg"] and e["toolset"] == "no3d" and e["modality"] == reg[v3]["modality"]
    assert e["report_kind"] == reg[v3]["report_kind"] and e["summary"]
    assert ROOT / e["profile"] == no3d
    prompt_registry.check_runnable(e, "2d_seg")
    with pytest.raises(prompt_registry.PromptRegistryError, match=r"for the 2d_seg arm, not 3d_seg"):
        prompt_registry.check_runnable(e)
    with pytest.raises(prompt_registry.PromptRegistryError, match=r"for the 3d_seg arm, not 2d_seg"):
        prompt_registry.check_runnable(reg[v3], "2d_seg")
    assert "at most 75 tool calls" in render_system(PROMPT_DIR, 75, version=v_no3d)


def test_the_2d_seg_default_and_the_others_unchanged():
    assert prompt_registry.default_version(PROMPT_DIR, "2d_seg") == NO3D == re_mod.prompt_version(arm="2d_seg")
    assert prompt_registry.default_version(PROMPT_DIR) == LITE_V3
    assert prompt_registry.default_version(PROMPT_DIR, "3d") == "chd_ct_nomask_v1"
    assert prompt_registry.ARMS == ("3d_seg", "3d", "2d_seg") == re_mod.ARMS


@pytest.mark.parametrize("entry, msg", [
    ({"version": "x", "arms": ["2d_seg"]}, "has no toolset"),
    ({"version": "x", "arms": ["2d_seg"], "toolset": "base"}, r"toolset 'base' is not the 2d_seg arm's \(no3d\)"),
    ({"version": "x", "arms": ["2d_seg"], "toolset": "ct"}, r"toolset 'ct' is not the 2d_seg arm's \(no3d\)"),
    ({"version": "x", "arms": ["2d_seg"], "toolset": "no3d"}, "is not served"),
])
def test_bad_2d_seg_prompt_entries_are_refused(entry, msg):
    with pytest.raises(prompt_registry.PromptRegistryError, match=msg):
        prompt_registry.check_runnable(entry, "2d_seg")
    with pytest.raises(prompt_registry.PromptRegistryError, match="is not the 3d_seg arm's"):
        prompt_registry.check_runnable({"version": "x", "arms": ["3d_seg"], "toolset": "no3d"})


def test_cohorts_give_each_arm_its_profile_and_prompt():
    for c in cohorts.COHORTS.values():
        assert c.for_arm("3d_seg") == (c.profile, c.prompt)
        assert c.for_arm("2d_seg") == (c.no3d_profile, c.no3d_prompt)
        assert c.for_arm("3d") == (cohorts.NOMASK_PROFILES[c.modality], cohorts.NOMASK_PROMPTS[c.modality])
        assert re_mod.arm_profile("3d", c) == cohorts.NOMASK_PROFILES[c.modality]
        assert re_mod.arm_profile("2d_seg", c) == c.no3d_profile and re_mod.arm_profile("3d_seg", c) == c.profile
    assert re_mod.arm_profile("2d_seg") == NO3D_PROFILE and re_mod.arm_profile("3d_seg") is None


# ---------------------------------------------------------------------- the episode (in-process, fake bridge)

def run_no3d(tmp_path, profile=NO3D_PROFILE, prompt=None, classes=None, **kw):
    srv, fake, traces = make_env(tmp_path, profile)
    model = ScriptedModel(dry_run_script("2d_seg", classes=classes))
    summary = anyio.run(lambda: run_episode(CASE, "2d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, prompt=prompt, **kw))
    return summary, srv, fake, traces, model


@pytest.mark.parametrize("parent, no3d, v3, v_no3d, ds", ARMS, ids=IDS)
def test_an_arm_2d_seg_episode_never_sees_the_3d_view(tmp_path, parent, no3d, v3, v_no3d, ds):
    c = cohorts.COHORTS.get(ds)
    src = {"dataset": c.key, c.id_key: 1} if c else None
    summary, srv, fake, _, model = run_no3d(tmp_path, no3d, v_no3d, c.classes if c else None, case_source=src)
    assert summary["stop_reason"] == "submitted" and summary["arm"] == "2d_seg"
    assert [t.name for t in model.tools] == TOOLS_12
    assert model.system_blocks[0] == render_system(PROMPT_DIR, 75, version=v_no3d)
    assert all(o.frames and set(o.frames) <= set(SLICES) for o in model.observations if not o.is_error)
    assert all("3d" not in capture for _, _, capture in fake.calls)
    meta, steps = read_trace(srv.session.trace_dir)
    assert meta["profile"] == no3d.name and meta["image"]["views"] == SLICES
    assert meta["client_meta"]["arm"] == "2d_seg" and meta["client_meta"]["prompt_version"] == v_no3d
    assert meta["segments_manifest_sha256"]  # the same preloaded segmentation as 3d_seg
    assert [s["tool"] for s in steps] == ["reset_episode", "point_at", "set_segment_visibility", "set_slice_outlines",
                                          "scroll_slice", "zoom", "bookmark_view", "reset_view", "submit_report"]
    assert all(not s.get("error") for s in steps)
    zoom = next(s for s in steps if s["tool"] == "zoom")
    assert zoom["args_after_validation"]["view"] == "slices" and zoom["frames"] and all(
        "_3d" not in f for f in zoom["frames"])
    assert not list(Path(srv.session.trace_dir).glob("frames/*_3d*.png"))
    prov = json.loads((tmp_path / "runs" / summary["episode_id"] / "provenance" / "provenance.json").read_text())
    assert prov["profile"] == no3d.name  # the cohort's own no-3D profile, not chd_ct_v1's name
    assert prov["profile_protocol_sha256"] == hashlib.sha256((no3d / "protocol.json").read_bytes()).hexdigest()
    tools = json.loads((tmp_path / "runs" / summary["episode_id"] / "provenance" / "tools.json").read_text())
    assert [t["name"] for t in tools] == TOOLS_12 and not re.search(r"3d|3D", json.dumps(tools))


def test_the_dry_run_scripts():
    assert [c[0][0] for c in dry_run_script("2d_seg")] == [
        "point_at", "set_segment_visibility", "set_slice_outlines", "scroll_slice", "zoom", "bookmark_view",
        "reset_view", "submit_report"]
    assert not {c[0][0] for c in dry_run_script("2d_seg")} & set(THREED_TOOLS)
    assert [c[0][0] for c in dry_run_script()] == ["point_at", "set_segment_visibility", "set_slice_outlines",
                                                   "orbit", "submit_report"]  # 3d_seg's, as before


def _leaves(e):  # the MCP client's task group may wrap it in an exception group
    return [x for s in getattr(e, "exceptions", ()) for x in _leaves(s)] or [e]


def test_arm_2d_seg_refuses_a_server_with_3d_tools(tmp_path):
    srv, fake, traces = make_env(tmp_path)  # chd_ct_v1: the 3d_seg profile
    with pytest.raises(BaseException) as ei:
        anyio.run(lambda: run_episode(CASE, "2d_seg", ScriptedModel(dry_run_script("2d_seg")), srv.server,
                                      runs_dir=tmp_path / "runs", traces_dir=traces, echo=None))
    err = [e for e in _leaves(ei.value) if isinstance(e, ValueError)]
    assert err and re.search(r"arm 2d_seg: the env server offers the 3D tools \['orbit', .*no3d_v1", str(err[0]))
    assert not any(c[0] == "reset_episode" for c in fake.calls)  # refused before the reset


def test_arm_2d_seg_aborts_on_a_3d_frame_before_the_model_sees_it(tmp_path):
    """A profile with the 2d_seg tools but the 3D view (observation.views left out): the tool check passes, the reset
    returns a 3D frame, and the episode aborts before the first model turn."""
    d = tmp_path / "with_3d_view"
    d.mkdir()
    p = json.loads((NO3D_PROFILE / "protocol.json").read_text())
    p["observation"].pop("views")
    (d / "protocol.json").write_text(json.dumps(p))
    for f in ("presets.xml", "report.py"):
        (d / f).write_bytes((NO3D_PROFILE / f).read_bytes())
    srv, fake, traces = make_env(tmp_path, d)
    model = ScriptedModel(dry_run_script("2d_seg"))
    with pytest.raises(BaseException) as ei:
        anyio.run(lambda: run_episode(CASE, "2d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                      traces_dir=traces, echo=None))
    err = [e for e in _leaves(ei.value) if isinstance(e, EnvironmentAbort)]
    assert err and "reset_episode: the env returned frames ['3d']" in str(err[0])
    assert model.observations == []  # the model never got the reset observation


def test_the_runtime_guard_only_for_its_views():
    env = EnvMCP(None)
    obs = Observation("orbit", "{}", {}, frames=["3d", "axial"])
    env._check_views(obs)  # no forbidden views (3d_seg, 3d): nothing happens
    env.forbidden_views = ("3d",)
    env._check_views(Observation("point_at", "{}", {}, frames=SLICES))
    with pytest.raises(EnvironmentAbort, match=r"orbit: the env returned frames \['3d'\]"):
        env._check_views(obs)


def test_an_arm_2d_seg_run_folder_is_named_and_complete(tmp_path, _reports_dir_in_tmp):
    summary, _, _, traces, _ = run_no3d(tmp_path, case_source=IMAGECHD_SRC)
    dest = finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=None,
                      run=fake_env_python([]))
    e = summary["episode_id"]
    assert dest.name.startswith("ImageCHD-1008_")
    assert dest.name.endswith(f"_eval_2d_seg_chd_ct_2dseg_v1_scripted_{e.rsplit('_', 1)[-1]}")
    assert (dest / "COMPLETE.json").is_file() and (dest / "report.pdf").is_file()
    prov = json.loads((dest / "provenance" / "provenance.json").read_text())
    assert prov["profile"] == "chd_ct_no3d_v1" == prov["environment"]["profile"] and prov["prompt_version"] == NO3D


# ---------------------------------------------------------------------- run-episode's command line

def test_cli_arm_2d_seg_imagechd(monkeypatch, tmp_path):
    seen = fake_cli(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as ex:
        main(["--imagechd", "1008", "--arm", "2d_seg", "--labels", "gt", "--dry-run-scripted"])
    assert ex.value.code == 0 and seen["arm"] == "2d_seg" and seen["prompt"] == NO3D
    assert seen["params"] == {"runtime": None, "slicer_url": None, "profile": NO3D_PROFILE}
    assert seen["case"][0][0] == 1008 and seen["case"][1] == {"labels": "gt", "label_map": None,
                                                                 "profile": NO3D_PROFILE}
    assert not {c[0][0] for c in seen["model"].script} & set(THREED_TOOLS)


@pytest.mark.parametrize("arm", ["2d_seg", "3d_seg"])
@pytest.mark.parametrize("c", list(cohorts.COHORTS.values()), ids=list(cohorts.COHORTS))
def test_cli_cohorts_pick_the_arms_profile_and_prompt(monkeypatch, tmp_path, c, arm):
    seen = fake_cli(monkeypatch, tmp_path)

    @contextlib.contextmanager
    def fake_cohort_case(*a, **k):
        seen["case"] = (a, k)
        yield types.SimpleNamespace(runtime=None, case_id="case_test", has_truth=True, info={},
                                    trace_meta=lambda: {}, summary_meta=lambda: {}, check_loaded=None)
    monkeypatch.setattr(dicom_run, "temporary_cohort_case", fake_cohort_case)
    with pytest.raises(SystemExit) as ex:
        main([f"--{c.key}", "1", "--arm", arm, "--labels", "/m/x_pred.nii.gz", "--dry-run-scripted"])
    profile, prompt = c.for_arm(arm)
    assert ex.value.code == 0 and seen["arm"] == arm and seen["prompt"] == prompt
    assert seen["params"]["profile"] == profile and seen["case"][1] == {"label_map": None, "profile": profile}


@pytest.mark.parametrize("argv, msg", [
    (["--imagechd", "1008", "--arm", "2d_seg"], "slices only \\(--no-3d\\) also shows the model a segmentation"),
    (["--imagechd", "1008", "--no-3d"], "slices only \\(--no-3d\\) also shows the model a segmentation: add --segmentation"),
    (["--imagechd", "1008", "--arm", "2d_seg", "--labels", "gt", "--prompt", LITE_V3], r"for the 3d_seg arm, not 2d_seg"),
    (["--imagechd", "1008", "--labels", "gt", "--prompt", NO3D], r"for the 2d_seg arm, not 3d_seg"),
    (["--hvsmr", "1", "--arm", "2d_seg", "--labels", "/x.nii.gz", "--prompt", NO3D],
     "written for profiles/chd_ct_no3d_v1, but this case runs chd_mr_no3d_v1"),
])
def test_cli_refuses_a_wrong_2d_seg_combination(argv, msg, capsys):
    with pytest.raises(SystemExit) as ex:
        main(argv + ["--dry-run-scripted"])
    assert ex.value.code == 2 and re.search(msg, capsys.readouterr().err)


def test_a_cohort_case_is_prepared_under_the_arms_profile(tmp_path):
    vol = tmp_path / "scan.nii.gz"
    vol.write_bytes(b"x")
    lab = tmp_path / "lab.nii.gz"
    lab.write_bytes(b"y")
    c = cohorts.custom("ct")
    args, _ = dicom_run.cohort_prepare_args(c, vol, str(lab), tmp_path)
    assert args[args.index("--profile") + 1] == c.profile  # 3d_seg: as before
    args, _ = dicom_run.cohort_prepare_args(c, vol, str(lab), tmp_path, profile=c.no3d_profile)
    assert args[args.index("--profile") + 1] == c.no3d_profile == NO3D_PROFILE


# ---------------------------------------------------------------------- slurm/run_batch.py and ./run.sh

@pytest.mark.parametrize("ds, labels, profile", [
    ("imagechd", "/m/imagechd/ct_{case}_pred.nii.gz", NO3D_PROFILE),
    ("hvsmr", "/m/hvsmr/{stem}_pred.nii.gz", cohorts.HVSMR.no3d_profile)])
def test_run_batch_arm_2d_seg_for_every_dataset(tmp_path, ds, labels, profile):
    c = rb.load_config(config(tmp_path, arm="2d_seg", dataset=ds, labels=labels))
    cmd = rb.episode_command(c, 0, 1, "/x/runtime.json")
    assert cmd[3:7] == [f"--{ds}", "1", "--arm", "2d_seg"] and "--labels" in cmd
    assert rb.slicer_profile(c) == str(profile)
    assert rb.slicer_command("/rt.json", "check", rb.slicer_profile(c))[-3:] == ["--profile", str(profile), "--check"]
    c3 = rb.load_config(config(tmp_path, arm="3d_seg", dataset=ds, labels=labels))  # 3d_seg: as before
    assert rb.slicer_profile(c3) == (None if ds == "imagechd" else str(cohorts.COHORTS[ds].profile))
    with pytest.raises(rb.BatchError, match="the config needs `labels`"):
        rb.load_config(config(tmp_path, arm="2d_seg", dataset=ds, labels=None))


def test_run_sh_launches_the_eval_slicer_with_the_2d_seg_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env, _ = run_sh.prepare(["--imagechd", "1", "--arm", "2d_seg", "--labels", "gt", "--dry-run-scripted"],
                            {"PATH": "/bin", run_sh.PROFILE_ENV: "stale"}, env_path=tmp_path / "none")
    assert env[run_sh.PROFILE_ENV] == str(NO3D_PROFILE)
