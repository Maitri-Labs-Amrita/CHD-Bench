"""The no-mask arm 3d: the model sees the CT only, with no segmentation of any kind. Its task profile
(profiles/chd_ct_nomask_v1), its prompt (chd_ct_nomask_v1), run-episode, slurm/run_batch.py, ./run.sh, W&B and the
OpenRouter privacy routing, and the 3d_seg arm left as it is. No Slicer, no API call."""

import contextlib
import hashlib
import importlib.util
import json
import re
import types
from pathlib import Path

import anyio
import pytest

from chd_client import dicom_run, prompt_registry, run_sh, wandb_log
from chd_client import run_episode as re_mod
from chd_client.diagnosis import dry_normal_report
from chd_client.models import registry
from chd_client.models.scripted import ScriptedModel
from chd_client.run_episode import PROMPT_DIR, dry_run_script, finish_run, main, render_system, run_episode
from chd_client.task.config import NOMASK_PROFILE, PROFILE
from slicer_mcp.config import load_profile
from slicer_mcp.tools.registry import SEG_TOOLS
from slicer_mcp.tracing.logger import read_trace
from tests.fakes import CASE, NORMAL, make_env
from tests.test_dicom_run import FakeRun, make_dataset
from tests.test_openrouter import FakeOpenAI, completion
from tests.test_run_folder import DX, IMAGECHD_SRC, fake_env_python

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_batch", ROOT / "slurm" / "run_batch.py")
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)

NOMASK = "chd_ct_nomask_v1"
CT = "chd_ct_v1"


def test_the_nomask_profile_is_chd_ct_v1_without_the_segmentation():
    seg, nomask = (json.loads((p / "protocol.json").read_text()) for p in (PROFILE, NOMASK_PROFILE))
    assert "segmentation" not in nomask and seg["segmentation"]
    assert nomask["mcp"]["tools"] == [t for t in seg["mcp"]["tools"] if t not in SEG_TOOLS]
    assert len(nomask["mcp"]["tools"]) == 17 and not set(SEG_TOOLS) & set(nomask["mcp"]["tools"])
    for k in set(seg) - {"segmentation", "mcp", "config_version", "_comment"}:
        assert nomask[k] == seg[k], k
    assert {k: v for k, v in nomask["mcp"].items() if k != "tools"} == \
        {k: v for k, v in seg["mcp"].items() if k != "tools"}
    assert set(nomask) == set(seg) - {"segmentation"} and nomask["config_version"] == "chd-ct-nomask-v1"
    for f in ("presets.xml", "report.py"):
        assert (NOMASK_PROFILE / f).read_bytes() == (PROFILE / f).read_bytes(), f
    assert sorted(p.name for p in NOMASK_PROFILE.iterdir() if p.is_file()) == ["presets.xml", "protocol.json",
                                                                                "report.py"]


def test_the_env_loads_the_nomask_profile_and_its_tool_text_names_no_segmentation():
    from slicer_mcp.tools_md import render
    p = load_profile(NOMASK_PROFILE)
    assert p.name == "chd_ct_nomask_v1" and not p.segmentation and len(p.tools) == 17
    assert p.report.SCHEMA_VERSION == "diagnosis-normal-v1" == load_profile(PROFILE).report.SCHEMA_VERSION
    md = render(str(NOMASK_PROFILE))
    assert "- Segmentation: none" in md
    model_text = md.split("## Overview", 1)[1]
    for word in ("inside_segment", "shell", "set_segment", "center_on_segments", "set_3d_content", "outline"):
        assert word not in model_text, word


def test_the_nomask_prompt_is_the_ct_prompt_minus_the_segmentation_only():
    v3 = (PROMPT_DIR / f"{CT}.md").read_text().splitlines(keepends=True)
    ct = (PROMPT_DIR / f"{NOMASK}.md").read_text().splitlines(keepends=True)
    removed = [ln for ln in v3 if ln not in ct]
    assert removed == ["A segmentation of the heart is already loaded, with six structures: LV, RV, LA, RA, Ao and "
                       "PA. It is a guide, not ground truth, and it can be wrong.\n",
                       "- **Segmentation display:** `set_segment_visibility`, `set_segment_opacity`, "
                       "`set_3d_content`, `set_slice_outlines`, `center_on_segments`.\n"]
    i = v3.index(removed[0])
    assert v3[:i] + v3[i + 2:v3.index(removed[1])] + v3[v3.index(removed[1]) + 1:] == ct
    text = "".join(ct)
    assert not re.search(r"segment|shell|mask", text, re.I)
    named = set(re.findall(r"`([a-z_0-9]+)`", text)) - {"calls_left", "classes", "present", "confidence", "normal",
                                                         "evidence"}
    assert named == set(load_profile(NOMASK_PROFILE).tools)


def test_the_nomask_prompt_is_registered_as_the_3d_default():
    reg = prompt_registry.load(PROMPT_DIR)
    e = reg[NOMASK]
    assert e["sha256"] == hashlib.sha256((PROMPT_DIR / e["file"]).read_bytes()).hexdigest() == \
        "da2e9f68878783dae6a8921765dd9e6ed73bc971da5e28d59c34da03ca5110f2"
    assert e["arms"] == ["3d"] and e["toolset"] == "ct" and e["modality"] == "ct"
    assert e["report_kind"] == reg[CT]["report_kind"] and e["summary"]
    assert ROOT / e["profile"] == NOMASK_PROFILE
    assert prompt_registry.default_version(PROMPT_DIR, "3d") == NOMASK == re_mod.prompt_version(arm="3d")
    assert prompt_registry.default_version(PROMPT_DIR) == CT
    prompt_registry.check_runnable(e, "3d")
    assert prompt_registry.server_toolset(e) == "ct" and prompt_registry.server_report(e) == "classes_normal"
    with pytest.raises(prompt_registry.PromptRegistryError, match="is for the 3d arm, not 3d_seg"):
        prompt_registry.check_runnable(e)
    with pytest.raises(prompt_registry.PromptRegistryError, match="is for the 3d_seg arm, not 3d"):
        prompt_registry.check_runnable(reg[CT], "3d")
    text = render_system(PROMPT_DIR, 70, version=NOMASK)
    assert "at most 70 tool calls" in text and "{" not in text


@pytest.mark.parametrize("entry, arm, msg", [
    ({"version": "x", "arms": ["3d"]}, "3d", "x has no toolset"),
    ({"version": "x", "arms": ["3d"], "toolset": "base"}, "3d", r"toolset 'base' is not the 3d arm's \(ct\)"),
    ({"version": "x", "arms": ["3d_seg"], "toolset": "ct"}, "3d_seg", r"toolset 'ct' is not the 3d_seg arm's"),
    ({"version": "x", "toolset": "ct"}, "3d_seg", "toolset 'ct' is not the 3d_seg arm's"),
    ({"version": "x", "toolset": "base"}, "3d", "toolset 'base' is not the 3d arm's"),
    ({"version": "x", "arms": ["3d"], "toolset": "v5"}, "3d", "unknown toolset 'v5'"),
])
def test_bad_3d_prompt_entries_are_refused(entry, arm, msg):
    with pytest.raises(prompt_registry.PromptRegistryError, match=msg):
        prompt_registry.check_runnable(entry, arm)


def test_unknown_arm_and_missing_default_are_named(tmp_path):
    with pytest.raises(prompt_registry.PromptRegistryError, match="arm '2d': not one of"):
        prompt_registry.check_runnable({"version": "x"}, "2d")
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "registry.json").write_text(json.dumps({"defaults": {"3d_seg": CT}, "prompts": []}))
    with pytest.raises(prompt_registry.PromptRegistryError, match="no default prompt for arm '3d'.*pass --prompt"):
        prompt_registry.default_version(d, "3d")


def run_nomask(tmp_path, model=None, **kw):
    srv, fake, traces = make_env(tmp_path, NOMASK_PROFILE)
    model = model or ScriptedModel(dry_run_script("3d"))
    summary = anyio.run(lambda: run_episode(CASE, "3d", model, srv.server, runs_dir=tmp_path / "runs",
                                            traces_dir=traces, echo=None, **kw))
    return summary, srv, fake, traces, model


def test_an_arm_3d_episode_has_no_segmentation_anywhere(tmp_path):
    summary, srv, fake, _, model = run_nomask(tmp_path)
    assert summary["stop_reason"] == "submitted" and summary["arm"] == "3d"
    names = [t.name for t in model.tools]
    assert names == list(load_profile(NOMASK_PROFILE).tools) and not set(SEG_TOOLS) & set(names)
    assert "segment" not in model.system_blocks[0].lower()
    assert model.system_blocks[0] == render_system(PROMPT_DIR, 70, version=NOMASK)
    meta, steps = read_trace(srv.session.trace_dir)
    assert meta["profile"] == "chd_ct_nomask_v1" and "segments_manifest_sha256" not in meta
    assert meta["client_meta"]["arm"] == "3d" and meta["client_meta"]["prompt_version"] == NOMASK
    reset = next(c for c in fake.calls if c[0] == "reset_episode")
    assert "with_segments" not in reset[1]
    assert all("segments" not in json.dumps(s.get("state_after") or {}) for s in steps)
    assert [s["tool"] for s in steps] == ["reset_episode", "point_at", "orbit", "submit_report"]


def test_the_3d_dry_run_uses_no_segmentation_tool_and_3d_seg_keeps_its_script():
    script = dry_run_script("3d")
    assert [c[0][0] for c in script] == ["point_at", "orbit", "submit_report"]
    assert script[-1] == [("submit_report", {"report": dry_normal_report()})]
    assert [c[0][0] for c in dry_run_script()] == ["point_at", "set_segment_visibility", "set_slice_outlines",
                                                   "orbit", "submit_report"]


def test_arm_3d_refuses_a_server_with_segmentation_tools(tmp_path):
    srv, fake, traces = make_env(tmp_path)
    with pytest.raises(BaseException) as ei:
        anyio.run(lambda: run_episode(CASE, "3d", ScriptedModel(dry_run_script("3d")), srv.server,
                                      runs_dir=tmp_path / "runs", traces_dir=traces, echo=None))

    def leaves(e):
        return [x for s in getattr(e, "exceptions", ()) for x in leaves(s)] or [e]
    err = [e for e in leaves(ei.value) if isinstance(e, ValueError)]
    assert err and re.search(r"arm 3d: the env server offers the segmentation tools .*serve "
                             r"profiles/chd_ct_nomask_v1", str(err[0]))
    assert not any(c[0] == "reset_episode" for c in fake.calls)


def test_an_arm_3d_run_folder_is_named_scored_and_complete(tmp_path, _reports_dir_in_tmp):
    summary, _, _, traces, _ = run_nomask(tmp_path, case_source=IMAGECHD_SRC)
    calls = []
    dest = finish_run(summary, traces, imagechd=True, runs_dir=tmp_path / "runs", echo=None,
                      run=fake_env_python(calls))
    e = summary["episode_id"]
    assert dest.name.startswith("ImageCHD-1008_")
    assert dest.name.endswith("_eval_3d_chd_ct_nomask_v1_scripted_" + e.rsplit("_", 1)[-1])
    m = json.loads((dest / "metrics.json").read_text())
    assert m.pop("tool_use")["n_calls"] >= 1 and m == {"diagnosis": DX}
    assert (dest / "COMPLETE.json").is_file()
    s = json.loads((dest / "summary.json").read_text())
    assert s["arm"] == "3d" and s["client_meta"]["prompt_version"] == NOMASK
    prov = json.loads((dest / "provenance" / "provenance.json").read_text())
    assert prov["profile"] == "chd_ct_nomask_v1" == prov["environment"]["profile"]
    assert prov["profile_protocol_sha256"] == hashlib.sha256((NOMASK_PROFILE / "protocol.json").read_bytes()).hexdigest()
    assert prov["prompt_version"] == NOMASK


def test_openrouter_privacy_routing_reaches_every_arm_3d_request(tmp_path):
    """config/providers.json routing (data_collection deny, zdr) goes into each request of an arm-3d episode too."""
    reg = registry.load_registry(registry.DEFAULT_FILE, environ={})
    m = registry.make_adapter("openrouter:moonshotai/kimi-k3.1-pro", registry=reg,
                              environ={"OPENROUTER_API_KEY": "test-not-a-key"})
    m.client = fake = FakeOpenAI([
        completion([("c1", "point_at", {"view": "axial", "x": 256, "y": 256})], content="look"),
        completion([("c2", "submit_report", {"report": NORMAL})], content="done")])
    m.sleep = lambda s: None
    summary, *_ = run_nomask(tmp_path, model=m)
    assert summary["stop_reason"] == "submitted" and len(fake.requests) == 2
    want = {"data_collection": "deny", "zdr": True, "ignore": ["google-vertex"]}
    assert [r["extra_body"]["provider"] for r in fake.requests] == [want, want]
    for r in fake.requests:
        sent = {t["function"]["name"] for t in r["tools"]}
        assert sent == set(load_profile(NOMASK_PROFILE).tools)
    recorded = [json.loads(ln) for ln in (Path(summary["trace_dir"]).parent.parent / "runs" / summary["episode_id"]
                                           / "requests.jsonl").read_text().splitlines()]
    assert len(recorded) == 2 and all(r["provider"] == "openrouter" for r in recorded)


def fake_cli(monkeypatch, tmp_path):
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
        seen["arm"], seen["model"], seen["prompt"] = arm, model, k.get("prompt")
        return {"stop_reason": "submitted"}

    monkeypatch.setattr(dicom_run, "temporary_imagechd_case", fake_case)
    monkeypatch.setattr(re_mod, "server_params", fake_params)
    monkeypatch.setattr(re_mod, "traces_dir_for", lambda runtime=None: tmp_path)
    monkeypatch.setattr(re_mod, "finish_run", lambda *a, **k: None)
    monkeypatch.setattr(re_mod, "run_episode", fake_episode)
    monkeypatch.setattr(re_mod.anyio, "run", lambda fn: fn())
    return seen


def test_cli_arm_3d_serves_the_nomask_profile_with_no_labels(monkeypatch, tmp_path):
    seen = fake_cli(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as ex:
        main(["--imagechd", "1008", "--arm", "3d", "--dry-run-scripted"])
    assert ex.value.code == 0 and seen["arm"] == "3d" and seen["prompt"] == NOMASK
    assert seen["params"] == {"runtime": None, "slicer_url": None, "profile": NOMASK_PROFILE}
    assert seen["case"][0][0] == 1008
    assert seen["case"][1] == {"labels": None, "label_map": None, "profile": NOMASK_PROFILE}
    assert not {c[0][0] for c in seen["model"].script} & set(SEG_TOOLS)


def test_cli_3d_seg_serves_the_default_profile_with_labels(monkeypatch, tmp_path):
    seen = fake_cli(monkeypatch, tmp_path)
    with pytest.raises(SystemExit) as ex:
        main(["--imagechd", "1008", "--labels", "gt", "--dry-run-scripted"])
    assert ex.value.code == 0 and seen["arm"] == "3d_seg" and seen["prompt"] == CT
    assert seen["params"] == {"runtime": None, "slicer_url": None, "profile": None}
    assert seen["case"][1] == {"labels": "gt", "label_map": None}


@pytest.mark.parametrize("argv, msg", [
    (["--imagechd", "1008", "--arm", "3d", "--labels", "gt"], "--arm 3d shows the model the scan only.*leave out "
                                                              "--labels gt"),
    (["--hvsmr", "24", "--arm", "3d"], "--arm 3d \\(no mask\\) is for CT only: use --imagechd or --volume with "
                                      "--modality ct"),
    (["--volume", "/x.nii.gz", "--modality", "mr", "--arm", "3d"], "--arm 3d \\(no mask\\) is for CT only"),
    (["--imagechd", "1008", "--arm", "3d", "--prompt", CT], "is for the 3d_seg arm, not 3d"),
    (["--imagechd", "1008", "--arm", "3d_seg", "--labels", "gt", "--prompt", NOMASK], "is for the 3d arm, not 3d_seg"),
    (["--imagechd", "1008", "--arm", "3d_seg"], "the following arguments are required: --labels"),
    (["--imagechd", "1008"], "the following arguments are required: --labels"),
])
def test_cli_refuses_a_wrong_arm_combination(argv, msg, capsys):
    with pytest.raises(SystemExit) as ex:
        main(argv + ["--dry-run-scripted"])
    assert ex.value.code == 2 and re.search(msg, capsys.readouterr().err)


def test_an_imagechd_case_for_arm_3d_is_prepared_without_shells(tmp_path):
    root = make_dataset(tmp_path / "ImageCHD")
    args, extra = dicom_run.imagechd_prepare_args(1008, root, None, profile=NOMASK_PROFILE)
    assert "--labels" not in args and "--label-map" not in args
    assert args[args.index("--profile") + 1] == NOMASK_PROFILE and "shell_source" not in extra
    with pytest.raises(dicom_run.DicomRunError, match="chd_ct_nomask_v1 has no segmentation.*leave --labels out"):
        dicom_run.imagechd_prepare_args(1008, root, "gt", profile=NOMASK_PROFILE)
    seg_args, _ = dicom_run.imagechd_prepare_args(1008, root, "gt")
    assert "--profile" not in seg_args and "--labels" in seg_args
    fake = FakeRun()
    with dicom_run.temporary_imagechd_case(1008, run=fake, echo=None, dataset_dir=root, profile=NOMASK_PROFILE) as tc:
        meta = tc.trace_meta()
    prep = fake.calls[0][1]
    assert prep[prep.index("--profile") + 1] == str(NOMASK_PROFILE) and "--labels" not in prep
    assert "labels" not in meta["case_source"] and meta["case_source"]["imagechd_id"] == 1008


def config(tmp_path, **over):
    """The example batch config with `over`; arm 3d without `labels` in `over` gets none (the example has a mask)."""
    c = json.loads((ROOT / "slurm" / "batch_config.example.json").read_text())
    if over.get("arm") == "3d" and "labels" not in over:
        c.pop("labels", None)
    c.update(over)
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c))
    return p


def test_run_batch_arm_3d_commands_and_slicer_profile(tmp_path):
    c = rb.load_config(config(tmp_path, arm="3d"))
    cmd = rb.episode_command(c, 0, 1001, "/x/runtime.json")
    assert cmd == ["uv", "run", "run-episode", "--imagechd", "1001", "--arm", "3d", "--runtime", "/x/runtime.json",
                   "--cost-cap", "8", "--model", "claude-opus-5-5", "--effort", "max"]
    assert "--labels" not in rb.episode_command(c, 0, 1001, "/x/runtime.json", dry_run=True)
    assert rb.slicer_profile(c) == str(NOMASK_PROFILE)
    assert rb.slicer_command("/rt.json", "check", rb.slicer_profile(c))[-3:] == ["--profile", str(NOMASK_PROFILE),
                                                                                  "--check"]
    assert rb.load_config(config(tmp_path, arm="3d", labels=None))["labels"] is None


def test_run_batch_3d_seg_command_is_pinned(tmp_path):
    """The 3d_seg command is pinned: its command_sha256 decides which cases a requeued job skips, so a change to it
    reruns every finished case."""
    c = rb.load_config(config(tmp_path, labels="gt"))
    cmd = rb.episode_command(c, 0, 1001, "/x/runtime.json")
    assert cmd == ["uv", "run", "run-episode", "--imagechd", "1001", "--arm", "3d_seg", "--runtime",
                   "/x/runtime.json", "--cost-cap", "8", "--labels", "gt", "--model", "claude-opus-5-5", "--effort",
                   "max"]
    assert rb.command_sha256(cmd) == "2b13adc17d5b58fd515042e8a131363cd3267e9a64113e45f30f532eab91664a"
    assert rb.slicer_profile(c) is None


@pytest.mark.parametrize("over, msg", [
    ({"arm": "3d", "labels": "gt"}, "arm 3d shows no segmentation: remove `labels`"),
    ({"arm": "3d", "labels": "/m/ct_{case}_pred.nii.gz"}, "remove `labels`"),
    ({"arm": "3d", "dataset": "hvsmr"}, "arm 3d \\(no mask\\) runs dataset imagechd only: hvsmr"),
    ({"arm": "3d_seg", "dataset": "other"}, r"dataset 'other': one of \('imagechd', 'hvsmr'\)"),
    ({"arm": "2d", "labels": "gt"}, r"arm '2d': one of \('3d_seg', '3d'\)"),
    ({"arm": "3d_seg", "labels": None}, "the config needs `labels`"),
])
def test_run_batch_refuses_a_wrong_arm_config(tmp_path, over, msg):
    with pytest.raises(rb.BatchError, match=msg):
        rb.load_config(config(tmp_path, **over))


def test_run_batch_plan_for_arm_3d(tmp_path, capsys):
    base = tmp_path / "rt.json"
    base.write_text(json.dumps({"bridge_host": "127.0.0.1", "bridge_port": 2037, "cases_dir": "c", "traces_dir": "t",
                                "client_timeout_s": 300, "case_prep": {"nrrd_compression": False}}))
    p = config(tmp_path, arm="3d", runtime_base=str(base), out_dir=str(tmp_path / "out"))
    assert rb.main(["--config", str(p), "--job", "0", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert all(e[e.index("--arm") + 1] == "3d" and "--labels" not in e for e in plan["episodes"])
    assert not (tmp_path / "out").exists()


class _Run:
    def __init__(self, **kw):
        self.kw, self.summary, self.url = kw, {}, "https://wandb/x"

    def log(self, row, step):
        pass

    def finish(self):
        pass


class _Wandb:
    def __init__(self):
        self.runs = []

    def init(self, **kw):
        self.runs.append(_Run(**kw))
        return self.runs[-1]


def test_wandb_names_the_3d_arm_only(tmp_path):
    w = _Wandb()
    wandb_log.log_episode(tmp_path, model="m", case=1001, wandb=w, arm="3d")
    wandb_log.log_episode(tmp_path, model="m", case=1001, wandb=w)
    wandb_log.log_episode(tmp_path, model="m", case=1001, wandb=w, arm="3d_seg")
    assert [r.kw["name"] for r in w.runs] == ["m/ImageCHD-1001/3d", "m/ImageCHD-1001", "m/ImageCHD-1001"]
    assert w.runs[0].kw["config"]["arm"] == "3d" and "arm" not in w.runs[1].kw["config"]
    assert w.runs[1].kw == w.runs[2].kw and w.runs[0].kw["group"] == w.runs[1].kw["group"] == "m"


def test_run_batch_passes_the_arm_to_wandb_for_arm_3d_only(tmp_path):
    got = []

    def log(folder, **kw):
        got.append(kw)
        return "u"
    base = {"wandb": {"project": "p"}, "batch": "b"}
    for arm in ("3d", "3d_seg", None):
        c = {**base, **({"arm": arm} if arm else {})}
        assert rb.log_wandb(c, {"name": "t0", "model": "m"}, 1001, str(tmp_path),
                            {"exit": 0, "command_sha256": "x"}, False, tmp_path, log=log) == "u"
    assert got[0]["arm"] == "3d" and got[0]["config"]["arm"] == "3d"
    assert got[1] == got[2] and "arm" not in got[1] and "arm" not in got[1]["config"]


def test_run_sh_launches_the_eval_slicer_with_the_arm_3d_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    env, _ = run_sh.prepare(["--imagechd", "1", "--arm", "3d", "--dry-run-scripted"],
                            {"PATH": "/bin", run_sh.PROFILE_ENV: "stale"}, env_path=tmp_path / "none")
    assert env[run_sh.PROFILE_ENV] == str(NOMASK_PROFILE)
    env, _ = run_sh.prepare(["--imagechd", "1", "--labels", "gt", "--dry-run-scripted"],
                            {"PATH": "/bin", run_sh.PROFILE_ENV: "stale"}, env_path=tmp_path / "none")
    assert run_sh.PROFILE_ENV not in env
    with pytest.raises(run_sh.RunShError, match="leave out --labels"):
        run_sh.prepare(["--imagechd", "1", "--arm", "3d", "--labels", "gt", "--dry-run-scripted"], {"PATH": "/bin"},
                       env_path=tmp_path / "none")


def test_run_sh_episode_shell_passes_the_profile_to_launch_and_stop(tmp_path):
    import os
    import stat
    import subprocess
    log = tmp_path / "calls.log"
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text(f'#!/bin/bash\necho "$*" >> {log}\nexit 0\n')
    uv.chmod(uv.stat().st_mode | stat.S_IEXEC)
    base = {k: v for k, v in os.environ.items() if k not in (run_sh.PROFILE_ENV, run_sh.NONCE_ENV)}
    base["PATH"] = f"{uv.parent}:{os.environ['PATH']}"
    for profile in (str(NOMASK_PROFILE), None):
        log.write_text("")
        env = {**base, run_sh.NONCE_ENV: "n0nce", **({run_sh.PROFILE_ENV: profile} if profile else {})}
        p = subprocess.run(run_sh.command(["--imagechd", "1001", "--arm", "3d"], linux=False), env=env,
                           capture_output=True, text=True)
        calls = log.read_text().splitlines()
        assert p.returncode == 0
        extra = f" --profile {profile}" if profile else ""
        assert calls[0] == "run --no-sync chd-slicer launch --no-main-window" + extra
        assert calls[-1] == "run --no-sync chd-slicer launch --stop" + extra


def test_the_shipped_nomask_example_config_loads(tmp_path):
    c = rb.load_config(ROOT / "slurm" / "batch_config_nomask.example.json")
    assert c["arm"] == "3d" and c["labels"] is None and c["dataset"] == "imagechd"
    cmd = rb.episode_command(c, 0, 1007, "/x/runtime.json")
    assert cmd[cmd.index("--arm") + 1] == "3d" and "--labels" not in cmd and cmd[-2:] == ["--max-steps", "75"]
