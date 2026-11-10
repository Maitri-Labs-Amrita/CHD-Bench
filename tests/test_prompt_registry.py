"""The prompt registry (prompts/registry.json), --prompt (a registered name or a prompt file) and the prompt in run
folder names."""

import contextlib
import hashlib
import json
import shutil
import types

import anyio
import pytest

from chd_client import prompt_registry
from chd_client import run_episode as re_mod
from chd_client.models.scripted import ScriptedModel
from chd_client.run_episode import PROMPT_DIR, dry_run_script, main, render_system, run_episode
from chd_client.run_folder import folder_name
from slicer_mcp.config import load_profile
from slicer_mcp.tracing.logger import read_trace

from chd_client.task.config import PROFILE
from tests.fakes import CASE, make_env

REG = json.loads((PROMPT_DIR / "registry.json").read_text())
CT, MR, NOMASK = "chd_ct_v1", "chd_mr_v1", "chd_ct_nomask_v1"
ENTRY_KEYS = {"version", "file", "sha256", "arms", "modality", "profile", "report_kind", "toolset", "summary"}


def test_every_prompt_file_is_registered_and_unchanged():
    files = {p.name for p in PROMPT_DIR.glob("*.md")}
    assert files == {e["file"] for e in REG["prompts"]} == {f"{CT}.md", f"{MR}.md", f"{NOMASK}.md"}
    for e in REG["prompts"]:
        assert hashlib.sha256((PROMPT_DIR / e["file"]).read_bytes()).hexdigest() == e["sha256"], e["file"]
        assert set(e) == ENTRY_KEYS and e["summary"], e["version"]
        assert e["file"] == f"{e['version']}.md" and (PROMPT_DIR.parent / e["profile"]).is_dir()
    assert {p.name for p in PROMPT_DIR.iterdir() if p.is_file()} == {f"{CT}.md", f"{MR}.md", f"{NOMASK}.md",
                                                                      "runner_messages.json", "registry.json"}


def test_defaults_resolve():
    assert REG["defaults"] == {"3d_seg": CT, "3d": NOMASK}
    e = prompt_registry.entry(PROMPT_DIR, prompt_registry.default_version(PROMPT_DIR))
    prompt_registry.check_runnable(e)
    assert e["version"] == CT and e["arms"] == ["3d_seg"] and e["modality"] == "ct"
    assert prompt_registry.default_version(PROMPT_DIR, "3d") == NOMASK


def test_an_edited_prompt_is_refused(tmp_path):
    d = tmp_path / "prompts"
    shutil.copytree(PROMPT_DIR, d)
    (d / f"{CT}.md").write_text((d / f"{CT}.md").read_text() + "\nedited")
    with pytest.raises(prompt_registry.PromptRegistryError, match="never edited"):
        render_system(d, 70)


def test_the_default_render_is_unchanged_by_the_registry():
    """The render is the registered file with only {max_steps} and {image_note} filled."""
    text = (PROMPT_DIR / f"{CT}.md").read_text()
    assert render_system(PROMPT_DIR, 70) == text.replace("{max_steps}", "70").replace("{image_note}", "")


def test_episode_records_the_chosen_prompt(tmp_path):
    srv, _, traces = make_env(tmp_path)
    model = ScriptedModel(dry_run_script())
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, prompt=CT))
    meta, _ = read_trace(srv.session.trace_dir)
    assert summary["stop_reason"] == "submitted" and meta["client_meta"]["prompt_version"] == CT
    assert meta["system_prompt_sha256"] == hashlib.sha256(model.system_blocks[0].encode()).hexdigest()


def my_prompt(tmp_path, name="my_prompt.md"):
    f = tmp_path / name
    f.write_text((PROMPT_DIR / f"{CT}.md").read_text().replace("You are an expert", "You are a careful expert"))
    return f


def test_a_prompt_file_gets_a_file_entry(tmp_path):
    f = my_prompt(tmp_path)
    e = prompt_registry.entry(PROMPT_DIR, str(f))
    assert e["version"] == "file:my_prompt.md" and e["source"] == "file" and e["file"] == str(f.resolve())
    assert e["sha256"] == hashlib.sha256(f.read_bytes()).hexdigest() and e["toolset"] == "base"
    prompt_registry.check_runnable(e)
    e3 = prompt_registry.entry(PROMPT_DIR, str(f), "3d")
    assert e3["arms"] == ["3d"] and e3["toolset"] == "ct"
    prompt_registry.check_runnable(e3, "3d")
    text = render_system(PROMPT_DIR, 33, version=str(f))
    assert "You are a careful expert" in text and "at most 33 tool calls" in text and "{max_steps}" not in text


def test_an_episode_with_a_prompt_file_records_its_file_version(tmp_path):
    f = my_prompt(tmp_path)
    srv, _, traces = make_env(tmp_path)
    model = ScriptedModel(dry_run_script())
    summary = anyio.run(lambda: run_episode(CASE, "3d_seg", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, prompt=str(f)))
    meta, _ = read_trace(srv.session.trace_dir)
    assert summary["stop_reason"] == "submitted" and "careful expert" in model.system_blocks[0]
    assert meta["client_meta"]["prompt_version"] == "file:my_prompt.md"


def test_cli_accepts_a_prompt_file(tmp_path):
    f = my_prompt(tmp_path)
    a = re_mod.parser().parse_args(["--imagechd", "1008", "--labels", "gt", "--dry-run-scripted", "--prompt", str(f)])
    version, _ = re_mod.prelaunch(a)
    assert prompt_registry.entry(PROMPT_DIR, version)["version"] == "file:my_prompt.md"


@pytest.mark.parametrize("argv,msg", [
    (["--prompt", "system_nope"], "unknown prompt 'system_nope'"),
    (["--arm", "2d"], "invalid choice: '2d'"),
    (["--arm", "3d"], "leave out --labels gt"),
    (["--prompt", NOMASK], f"prompt {NOMASK} is for the 3d arm, not 3d_seg"),
    (["--prompt", MR], f"prompt {MR} is written for profiles/chd_mr_v1, but this case runs chd_ct_v1"),
])
def test_cli_refuses_a_prompt_it_cannot_run(argv, msg, capsys):
    with pytest.raises(SystemExit):
        main(["--imagechd", "1008", "--labels", "gt", "--dry-run-scripted"] + argv)
    assert msg in capsys.readouterr().err


def test_a_registered_prompt_for_another_profile_is_refused_for_every_case():
    ap = re_mod.parser()
    for argv, msg in [
        (["--hvsmr", "24", "--labels", "/x.nii.gz", "--prompt", CT], "but this case runs chd_mr_v1"),
        (["--volume", "/x.nii.gz", "--modality", "mr", "--labels", "/m.nii.gz", "--prompt", CT],
         "but this case runs chd_mr_v1"),
        (["--volume", "/x.nii.gz", "--labels", "/m.nii.gz", "--prompt", MR], "but this case runs chd_ct_v1"),
    ]:
        with pytest.raises(re_mod.PrelaunchError, match=msg):
            re_mod.prelaunch(ap.parse_args(argv + ["--dry-run-scripted"]))
    assert re_mod.prelaunch(ap.parse_args(["--hvsmr", "24", "--labels", "/x.nii.gz", "--prompt", MR,
                                           "--dry-run-scripted"]))[0] == MR


def test_a_prompt_for_another_arm_is_refused():
    with pytest.raises(prompt_registry.PromptRegistryError, match="is for the 3d arm, not 3d_seg"):
        prompt_registry.check_runnable({"version": "x", "arms": ["3d"]})
    with pytest.raises(prompt_registry.PromptRegistryError, match="not one of"):
        prompt_registry.check_runnable({"version": "x"}, "2d")


def test_run_folder_names_the_prompt():
    e = "20261013T074900Z_case_nii_ab_3d_seg_78b876"
    meta = {"arm": "3d_seg", "model": {"id": "claude-opus-5-5"}, "client_meta": {"prompt_version": CT}}
    assert folder_name(e, meta, None, "UTC").endswith("_3d_seg_chd_ct_v1_opus-5-5_78b876")
    assert folder_name(e, {"arm": "3d_seg", "model": {"id": "claude-opus-5-5"}}, None, "UTC").endswith(
        "_3d_seg_opus-5-5_78b876")


def test_the_prompt_asks_for_the_17_classes_and_the_server_takes_them():
    """The chd_ct_v1 report is the 16 classes and normal, so the client starts the env server with --report
    classes_normal, and the scripted dry run submits a normal report."""
    from chd_client.diagnosis import dry_normal_report
    e = prompt_registry.entry(PROMPT_DIR, CT)
    assert e["arms"] == ["3d_seg"] and e["report_kind"] == "diagnosis_normal"
    assert prompt_registry.server_report(e) == "classes_normal"
    assert prompt_registry.server_report({"version": "x", "report_kind": "segmental"}) is None
    text = render_system(PROMPT_DIR, 75, version=CT)
    assert "at most 75 tool calls" in text and "{" not in text
    assert "- normal - no defects found." in text and "Decide, for each of these 17 classes" in text
    assert load_profile(PROFILE).report.SCHEMA_VERSION == "diagnosis-normal-v1"
    assert dry_run_script("3d_seg")[-1] == [("submit_report", {"report": dry_normal_report()})]
    assert dry_normal_report()["classes"]["normal"]["present"] is True


@pytest.mark.parametrize("prompt", [None, CT])
def test_cli_starts_the_profile_server_and_the_normal_dry_run(prompt, monkeypatch, tmp_path):
    """run-episode starts slicer-mcp with the task profile and gives the scripted dry run the normal report."""
    from chd_client import dicom_run
    from chd_client.diagnosis import dry_normal_report
    seen = {}

    @contextlib.contextmanager
    def fake_case(*a, **k):
        seen["case"] = (a, k)
        yield types.SimpleNamespace(runtime=None, case_id="case_test", has_truth=False, info={},
                                    trace_meta=lambda: {}, summary_meta=lambda: {}, check_loaded=None)

    def fake_params(**k):
        seen["params"] = k
        return "target"

    def fake_episode(case_id, arm, model, target, **k):
        seen["model"] = model
        seen["prompt"] = k["prompt"]
        return {"stop_reason": "submitted"}

    monkeypatch.setattr(dicom_run, "temporary_imagechd_case", fake_case)
    monkeypatch.setattr(re_mod, "server_params", fake_params)
    monkeypatch.setattr(re_mod, "traces_dir_for", lambda runtime=None: tmp_path)
    monkeypatch.setattr(re_mod, "finish_run", lambda *a, **k: None)
    monkeypatch.setattr(re_mod, "run_episode", fake_episode)
    monkeypatch.setattr(re_mod.anyio, "run", lambda fn: fn())
    with pytest.raises(SystemExit) as ex:
        main(["--imagechd", "1008", "--labels", "gt", "--dry-run-scripted"]
             + (["--prompt", prompt] if prompt else []))
    assert ex.value.code == 0 and seen["params"] == {"runtime": None, "slicer_url": None, "profile": None}
    assert seen["case"][0][0] == 1008 and seen["case"][1]["labels"] == "gt" and seen["prompt"] == CT
    assert seen["model"].script[-1] == [("submit_report", {"report": dry_normal_report()})]


def test_every_prompt_names_its_arms_toolset():
    """The prompt decides the tools: base for 3d_seg, ct for 3d. A prompt for another toolset is refused."""
    reg = prompt_registry.load(PROMPT_DIR)
    assert {v: prompt_registry.server_toolset(e) for v, e in reg.items()} == {CT: "base", MR: "base", NOMASK: "ct"}
    assert prompt_registry.ARM_TOOLSETS == {"3d_seg": "base", "3d": "ct"}
    with pytest.raises(prompt_registry.PromptRegistryError, match="has no toolset"):
        prompt_registry.server_toolset({"version": "x", "arms": ["3d_seg"]})
    with pytest.raises(prompt_registry.PromptRegistryError, match="unknown toolset"):
        prompt_registry.server_toolset({"version": "x", "arms": ["3d_seg"], "toolset": "v5"})
    assert prompt_registry.state_format(reg[CT]) == "full"
    for e in reg.values():
        prompt_registry.check_runnable(e, e["arms"][0])
    with pytest.raises(prompt_registry.PromptRegistryError, match="is not the 3d_seg arm's"):
        prompt_registry.check_runnable({"version": "x", "arms": ["3d_seg"], "toolset": "ct"})


def test_a_prompt_for_a_report_the_profile_does_not_serve_is_refused():
    with pytest.raises(prompt_registry.PromptRegistryError, match="is not served"):
        prompt_registry.check_runnable({"version": "x", "arms": ["3d_seg"], "toolset": "base",
                                        "report_kind": "something_else"})


def test_no_base_prompt_names_a_tool_it_is_not_offered():
    """A base prompt that named a tool of another toolset would ask for a tool its episode does not offer."""
    reg = prompt_registry.load(PROMPT_DIR)
    for v, e in reg.items():
        if e.get("toolset") == "base":
            text = (PROMPT_DIR / e["file"]).read_text()
            assert not any(t in text for t in ("trace_path", "slice_montage", "hu_profile", "measure_area",
                                               "set_oblique_plane", "record_signs")), v


def test_diagnosis_line_shows_the_normal_call():
    from chd_client.run_episode import diagnosis_line
    dx = {"exact_match": 1.0, "f1": 1.0, "predicted": [], "truth": [],
          "normal_call": {"present": True, "confidence": 0.8, "truth": True, "correct": True}}
    assert diagnosis_line(dx).endswith("| normal called yes (0.80), truth normal: right")
